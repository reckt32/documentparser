"""Questionnaire endpoints."""

import re

from flask import Blueprint, g, jsonify, request

from backend.auth import optional_auth
from backend.db import (
    create_questionnaire,
    get_latest_questionnaire_for_user,
    get_questionnaire,
    save_estate,
    save_family_info,
    save_goals,
    save_insurance,
    save_lifestyle,
    save_personal_info,
    save_risk_profile,
    save_tax_info,
    update_questionnaire_status,
)
from backend.extraction.insights import build_prefill_from_insights

bp = Blueprint("questionnaire", __name__)


def _questionnaire_section_saver(section: str):
    mapping = {
        "personal_info": save_personal_info,
        "family_info": save_family_info,
        "goals": save_goals,
        "risk_profile": save_risk_profile,
        "insurance": save_insurance,
        "estate": save_estate,
        "lifestyle": save_lifestyle,
        "tax_info": save_tax_info,
    }
    return mapping.get(section)


def _normalize_risk_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}
    out = {}
    # Enumerated fields normalization (lowercase)
    def _norm_enum(val, allowed):
        if not isinstance(val, str):
            return None
        v = val.strip().lower()
        return v if v in allowed else None

    out["primary_horizon"] = _norm_enum(payload.get("primary_horizon"), ["short","medium","med","long"])
    # numeric horizons
    def _num(v, lo=None, hi=None):
        try:
            if v in (None, "", "N/A"):
                return None
            n = float(v)
            if lo is not None: n = max(lo, n)
            if hi is not None: n = min(hi, n)
            return n
        except Exception:
            return None

    out["primary_horizon_years"] = _num(payload.get("primary_horizon_years"), 0, 100)
    out["goal_tenure_years"] = _num(payload.get("goal_tenure_years"), 0, 100)

    out["goal_flexibility"] = _norm_enum(payload.get("goal_flexibility"), ["critical","fixed","flexible"])
    out["goal_importance"] = _norm_enum(payload.get("goal_importance"), ["essential","important","lifestyle"])
    out["behavior"] = _norm_enum(payload.get("behavior"), ["sell","reduce","hold","buy","aggressive buy","aggressive_buy"])
    out["income_stability"] = _norm_enum(payload.get("income_stability"), ["very unstable","unstable","average","stable","very stable"])

    out["loss_tolerance_percent"] = _num(payload.get("loss_tolerance_percent"), 0, 100)
    out["emergency_fund_months"] = _num(payload.get("emergency_fund_months"), 0, 240)

    # Pass through tolerance if present
    out["tolerance"] = _norm_enum(payload.get("tolerance"), ["low","medium","med","high"])

    # Preserve original raw values for anything not normalized (optional)
    for k, v in payload.items():
        if k not in out:
            out[k] = v
    return out


def _validate_risk_payload(payload: dict):
    errs = []
    if not isinstance(payload, dict):
        return ["invalid_payload_type"]
    # Percentage fields
    for f in ["loss_tolerance_percent", "equity_allocation_percent"]:
        v = payload.get(f)
        if v not in (None, ""):
            try:
                fv = float(v)
                if fv < 0 or fv > 100:
                    errs.append(f"{f}:out_of_range")
            except Exception:
                errs.append(f"{f}:not_numeric")
    # Horizon years
    hy = payload.get("primary_horizon_years")
    if hy not in (None, ""):
        try:
            hv = float(hy)
            if hv < 0 or hv > 100:
                errs.append("primary_horizon_years:out_of_range")
        except Exception:
            errs.append("primary_horizon_years:not_numeric")
    # Emergency fund months
    efm = payload.get("emergency_fund_months")
    if efm not in (None, ""):
        try:
            emv = float(efm)
            if emv < 0 or emv > 240:
                errs.append("emergency_fund_months:out_of_range")
        except Exception:
            errs.append("emergency_fund_months:not_numeric")
    # Enum validations
    enums = {
        "goal_importance": {"essential","important","lifestyle"},
        "goal_flexibility": {"critical","fixed","flexible"},
        "behavior": {"sell","reduce","hold","buy","aggressive buy","aggressive_buy"},
        "income_stability": {"very unstable","unstable","average","stable","very stable"},
        "tolerance": {"low","medium","med","high"},
        "primary_horizon": {"short","medium","med","long"},
    }
    for k, allowed in enums.items():
        v = payload.get(k)
        if v not in (None, "") and str(v).lower() not in allowed:
            errs.append(f"{k}:invalid_value")
    return errs


def _validate_personal_info_payload(payload: dict):
    errs = []
    if not isinstance(payload, dict):
        return ["invalid_payload_type"]
    
    pan = payload.get("pan")
    if not pan:
        errs.append("pan:mandatory")
    else:
        pan = str(pan).strip().upper()
        if not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', pan):
            errs.append("pan:invalid_format")
            
    return errs


@bp.route("/questionnaire/start", methods=["POST"])
def questionnaire_start():
    data = request.get_json(force=True) or {}
    user_id = data.get("user_id") or "user"
    qid = create_questionnaire(user_id=user_id)
    return jsonify({"questionnaire_id": qid}), 201


@bp.route("/questionnaire/<int:qid>/<section>", methods=["PUT"])
def questionnaire_save_section(qid: int, section: str):
    saver = _questionnaire_section_saver(section)
    if not saver:
        return jsonify({"error": "Unknown section"}), 400
    payload = request.get_json(force=True) or {}

    # Server-side normalization + validation for risk_profile section
    if section == "risk_profile":
        payload = _normalize_risk_payload(payload)
        errors = _validate_risk_payload(payload)
        if errors:
            return jsonify({"error": "validation_failed", "fields": errors}), 400
    elif section == "personal_info":
        errors = _validate_personal_info_payload(payload)
        if errors:
            return jsonify({"error": "validation_failed", "fields": errors}), 400

    saver(qid, payload)
    update_questionnaire_status(qid, "in_progress")
    return jsonify({"status": "ok"}), 200


@bp.route("/questionnaire/<int:qid>", methods=["GET"])
def questionnaire_get(qid: int):
    q = get_questionnaire(qid)
    if not q:
        return jsonify({"error": "not found"}), 404
    return jsonify(q), 200


@bp.route("/questionnaire/<int:qid>/prefill", methods=["GET"])
def questionnaire_prefill(qid: int):
    """
    Provide prefill suggestions derived from uploaded documents for this questionnaire.
    Frontend can use these to pre-populate fields and allow user edits.
    """
    try:
        prefill = build_prefill_from_insights(qid)
        return jsonify(prefill), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/questionnaire/latest", methods=["GET"])
@optional_auth
def get_latest_questionnaire():
    """Fetch the latest questionnaire for the logged-in user to pre-fill the form."""
    user_id = getattr(g, 'user_id', None)
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401
        
    qid = get_latest_questionnaire_for_user(user_id)
    if not qid:
        return jsonify({"message": "No questionnaire found", "data": None}), 200
        
    data = get_questionnaire(qid)
    return jsonify({"questionnaire_id": qid, "data": data}), 200
