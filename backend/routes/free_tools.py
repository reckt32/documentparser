"""Free-tier calculator endpoints."""

import logging
from typing import Any, Dict

from flask import Blueprint, g, jsonify, request

from backend.analysis.financial_calculators import (
    compute_golden_number,
    compute_retirement_gap,
    compute_spend_right,
    format_indian_compact,
)
from backend.auth import optional_auth
from backend.db import (
    create_questionnaire,
    get_latest_questionnaire_for_user,
    get_questionnaire,
    save_goals,
    save_lifestyle,
)

logger = logging.getLogger(__name__)
bp = Blueprint("free_tools", __name__)


def _autosave_free_tool_to_questionnaire(user_id, data: Dict[str, Any], tool_type: str):
    """
    Helper to sync free tool inputs to the user's latest questionnaire lifestyle/goals sections.
    tool_type: 'spend-right' or 'retirement'
    """
    try:
        qid = get_latest_questionnaire_for_user(user_id)
        if not qid:
            qid = create_questionnaire(user_id)
        
        # Fetch current questionnaire state to merge
        q_data = get_questionnaire(qid)
        lifestyle = q_data.get("lifestyle") or {}
        goals = q_data.get("goals") or {}
        
        if tool_type == 'spend-right':
            lifestyle.update({
                "annual_income": str(float(data.get("income", 0)) * 12),
                "monthly_expenses": str(data.get("rent", 0) + data.get("basic_spends", 0) + data.get("comfort_spends", 0)),
            })
        elif tool_type == 'golden-number':
            # New 4-question Golden Number format
            from backend.analysis.financial_calculators import compute_golden_number
            result = compute_golden_number(
                annual_clothing=float(data.get("annual_clothing", 0)),
                annual_travel=float(data.get("annual_travel", 0)),
                monthly_lifestyle=float(data.get("monthly_lifestyle", 0)),
                total_gadget_value=float(data.get("total_gadget_value", 0)),
            )
            lifestyle.update({
                "recommended_monthly_income": str(result["golden_number"]),
                "monthly_expenses": str(result["total_monthly_wants"]),
            })
            if data.get("actual_income"):
                lifestyle["annual_income"] = str(float(data["actual_income"]) * 12)
        elif tool_type == 'retirement':
            lifestyle.update({
                "monthly_expenses": str(data.get("monthly_expense", 0)),
                "expected_pension": str(data.get("expected_pension", 0)),
                "manual_corpus": str(data.get("existing_corpus", 0)),
                "manual_sip": str(data.get("ongoing_sip", 0)),
            })
            # Also set wants_retirement_planning in goals
            goals.update({
                "wants_retirement_planning": True,
                "expected_pension": str(data.get("expected_pension", 0))
            })
            save_goals(qid, goals)
            
        save_lifestyle(qid, lifestyle)
        logger.info(f"Autosaved {tool_type} data to questionnaire {qid} for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to autosave {tool_type} data: {e}")


@bp.route("/api/free/spend-right", methods=["POST"])
@optional_auth
def free_spend_right():
    """
    Free-tier Golden Number calculator (4-question version).
    Accepts: {annual_clothing, annual_travel, monthly_lifestyle, total_gadget_value, actual_income?}
    Returns: Golden Number (ideal monthly income), monthly wants breakdown, and optional status badge.

    Backward compatible: if old fields (income, rent, basic_spends, comfort_spends) are sent,
    routes to the legacy compute_spend_right function.
    """
    data = request.get_json(force=True) or {}

    # ── Backward compatibility: detect old-format request ──
    old_fields = ["income", "rent", "basic_spends", "comfort_spends"]
    if all(f in data for f in old_fields):
        try:
            income = float(data["income"])
            rent = float(data["rent"])
            basic_spends = float(data["basic_spends"])
            comfort_spends = float(data["comfort_spends"])
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"All values must be numeric: {e}"}), 400

        result = compute_spend_right(income, rent, basic_spends, comfort_spends)
        result["golden_number_formatted"] = format_indian_compact(result["golden_number"])
        result["surplus_formatted"] = format_indian_compact(result["surplus"])

        user_id = getattr(g, 'user_id', None)
        if user_id:
            _autosave_free_tool_to_questionnaire(user_id, data, 'spend-right')
        return jsonify(result), 200

    # ── New 4-question Golden Number format ──
    new_required = ["annual_clothing", "annual_travel", "monthly_lifestyle", "total_gadget_value"]
    missing = [f for f in new_required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        annual_clothing = float(data["annual_clothing"])
        annual_travel = float(data["annual_travel"])
        monthly_lifestyle = float(data["monthly_lifestyle"])
        total_gadget_value = float(data["total_gadget_value"])
        actual_income = float(data.get("actual_income", 0))
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"All values must be numeric: {e}"}), 400

    result = compute_golden_number(
        annual_clothing=annual_clothing,
        annual_travel=annual_travel,
        monthly_lifestyle=monthly_lifestyle,
        total_gadget_value=total_gadget_value,
        actual_income=actual_income,
    )

    # Add formatted values for display convenience
    result["golden_number_formatted"] = format_indian_compact(result["golden_number"])
    result["total_wants_formatted"] = format_indian_compact(result["total_monthly_wants"])
    if "surplus_deficit" in result:
        result["surplus_deficit_formatted"] = format_indian_compact(result["surplus_deficit"])

    # Autosave if user is logged in
    user_id = getattr(g, 'user_id', None)
    if user_id:
        _autosave_free_tool_to_questionnaire(user_id, data, 'golden-number')

    return jsonify(result), 200


@bp.route("/api/free/retirement-calc", methods=["POST"])
@optional_auth
def free_retirement_calc():
    """
    Free-tier Retirement Gap calculator.
    Accepts: {monthly_expense, years_to_retire, existing_corpus?, ongoing_sip?, expected_pension?}
    Returns: Target corpus, FV of existing, FV of SIP, gap, required step-up SIP.
    No authentication required.
    """
    data = request.get_json(force=True) or {}

    # Validate required fields
    required = ["monthly_expense", "years_to_retire"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        monthly_expense = float(data["monthly_expense"])
        years_to_retire = float(data["years_to_retire"])
        existing_corpus = float(data.get("existing_corpus", 0))
        ongoing_sip = float(data.get("ongoing_sip", 0))
        expected_pension = float(data.get("expected_pension", 0))
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"All values must be numeric: {e}"}), 400

    if monthly_expense <= 0:
        return jsonify({"error": "Monthly expense must be greater than zero."}), 400
    if years_to_retire <= 0 or years_to_retire > 60:
        return jsonify({"error": "Years to retire must be between 1 and 60."}), 400

    result = compute_retirement_gap(
        monthly_expense=monthly_expense,
        years_to_retire=years_to_retire,
        existing_corpus=existing_corpus,
        ongoing_sip=ongoing_sip,
        expected_pension=expected_pension,
    )

    # Add formatted values for display convenience
    result["target_corpus_formatted"] = format_indian_compact(result["target_corpus"])
    result["fv_existing_formatted"] = format_indian_compact(result["fv_existing_corpus"])
    result["fv_sip_formatted"] = format_indian_compact(result["fv_ongoing_sip"])
    result["gap_formatted"] = format_indian_compact(result["gap"])
    result["required_sip_formatted"] = format_indian_compact(result["required_step_up_sip"])

    # Autosave if user is logged in
    user_id = getattr(g, 'user_id', None)
    if user_id:
        _autosave_free_tool_to_questionnaire(user_id, data, 'retirement')

    return jsonify(result), 200
