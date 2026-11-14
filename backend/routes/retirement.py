"""Retirement module endpoints."""

import json
import logging
import os
import uuid
from datetime import datetime, timedelta

from flask import Blueprint, g, jsonify, request, url_for

from backend.auth import require_auth, require_payment
from backend.config import LOGO_PATH, OUTPUT_DIR
from backend.db import (
    consume_user_credit,
    create_retirement_questionnaire,
    get_active_dashboard_report_by_pan,
    get_latest_retirement_questionnaire,
    get_retirement_questionnaire,
    get_user_credits,
    insert_aggregate_action,
    insert_dashboard_report,
    mark_aggregate_actions_status_by_report,
    mark_dashboard_report_status,
    save_retirement_questionnaire,
)
from backend.retirement.engine import (
    analyze_retirement,
    dashboard_action_items,
    retirement_plan_is_accepted,
    validate_retirement_input,
)
from backend.retirement.report import generate_retirement_pdf

logger = logging.getLogger(__name__)
bp = Blueprint("retirement", __name__)


# --- Retirement corpus-deployment MVP -------------------------------------

def _retirement_user_id() -> str:
    return str(getattr(g, "user_id", "") or "")


@bp.route("/retirement/questionnaire/start", methods=["POST"])
@require_auth
def retirement_questionnaire_start():
    user_id = _retirement_user_id()
    initial = request.get_json(silent=True) or {}
    qid = create_retirement_questionnaire(user_id, initial.get("data") or {})
    return jsonify({"questionnaire_id": qid, "status": "in_progress"}), 201


@bp.route("/retirement/questionnaire/latest", methods=["GET"])
@require_auth
def retirement_questionnaire_latest():
    draft = get_latest_retirement_questionnaire(_retirement_user_id())
    return jsonify(draft or {"data": None}), 200


@bp.route("/retirement/questionnaire/<int:qid>", methods=["GET", "PUT"])
@require_auth
def retirement_questionnaire_detail(qid: int):
    user_id = _retirement_user_id()
    if request.method == "GET":
        draft = get_retirement_questionnaire(qid, user_id)
        if not draft:
            return jsonify({"error": "Retirement questionnaire not found"}), 404
        return jsonify(draft), 200

    payload = request.get_json(force=True) or {}
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    status = str(payload.get("status") or "in_progress")
    if status not in {"in_progress", "completed", "archived"}:
        return jsonify({"error": "Invalid questionnaire status"}), 400
    if not get_retirement_questionnaire(qid, user_id):
        return jsonify({"error": "Retirement questionnaire not found"}), 404
    save_retirement_questionnaire(qid, user_id, data, status=status)
    return jsonify({"questionnaire_id": qid, "status": status}), 200


def _retirement_payload_from_request() -> tuple[dict, int | None, tuple | None]:
    body = request.get_json(force=True) or {}
    qid = body.get("questionnaire_id")
    data = body.get("data") if isinstance(body.get("data"), dict) else None
    if qid is not None:
        try:
            qid = int(qid)
        except (TypeError, ValueError):
            return {}, None, (jsonify({"error": "Invalid questionnaire_id"}), 400)
        draft = get_retirement_questionnaire(qid, _retirement_user_id())
        if not draft:
            return {}, None, (jsonify({"error": "Retirement questionnaire not found"}), 404)
        data = data or draft.get("data") or {}
    if not isinstance(data, dict):
        data = body if isinstance(body, dict) else {}
    return data, qid, None


@bp.route("/retirement/analyze", methods=["POST"])
@require_auth
def retirement_analyze():
    payload, qid, error = _retirement_payload_from_request()
    if error:
        return error
    validation_errors = validate_retirement_input(payload)
    if validation_errors:
        return jsonify({"error": "validation_failed", "fields": validation_errors}), 400
    analysis = analyze_retirement(payload)
    if qid:
        save_retirement_questionnaire(qid, _retirement_user_id(), payload, status="in_progress")
    return jsonify({"questionnaire_id": qid, "analysis": analysis}), 200


def _retirement_dashboard_snapshot(analysis: dict) -> dict:
    solver = analysis.get("solver") or {}
    cashflow = analysis.get("cashflow") or {}
    net_worth = analysis.get("net_worth") or {}
    flags = analysis.get("flags") or []
    feasible = solver.get("selected_feasible") is True
    return {
        "module": "retirement",
        "overall_health": {
            "score": None,
            "label": "Plan Feasible" if feasible else "Requires Adjustment",
        },
        "retirement": {
            "status": solver.get("status"),
            "selected_feasible": feasible,
            "selected_rate": solver.get("selected_rate"),
            "minimum_feasible_rate": solver.get("minimum_feasible_rate"),
            "top_flag": flags[0].get("message") if flags else None,
            "available_corpus": net_worth.get("available_corpus"),
            "income_bucket": solver.get("income_bucket"),
            "legacy_residual": solver.get("legacy_residual"),
            "shortfall": solver.get("shortfall_at_selected_rate"),
            "effective_monthly_expense": cashflow.get("effective_monthly_expense"),
            "design_gap": cashflow.get("design_gap"),
        },
        "goal_summary": analysis.get("goals") or [],
        "flags": flags,
    }


@bp.route("/retirement/report/generate", methods=["POST"])
@require_auth
@require_payment
def retirement_report_generate():
    payload, qid, error = _retirement_payload_from_request()
    if error:
        return error
    validation_errors = validate_retirement_input(payload)
    if validation_errors:
        return jsonify({"error": "validation_failed", "fields": validation_errors}), 400

    if not retirement_plan_is_accepted(payload):
        return jsonify(
            {
                "error": "plan_not_accepted",
                "message": "The adviser must review and accept the settled plan before generating the report.",
            }
        ), 409

    user_id = _retirement_user_id()
    credits = get_user_credits(user_id)
    if credits <= 0:
        return jsonify({"error": "Insufficient credits", "remaining_credits": 0}), 403

    analysis = analyze_retirement(payload)
    if not (analysis.get("solver") or {}).get("selected_feasible"):
        return jsonify(
            {
                "error": "plan_not_settled",
                "message": "Adjust the retirement plan until it is feasible before generating the report.",
                "analysis": analysis,
            }
        ), 409
    profile = analysis["profile"]
    client_pan = profile["pan"]
    existing = get_active_dashboard_report_by_pan(user_id, client_pan, "retirement")
    if existing:
        mark_dashboard_report_status(existing["id"], "SUPERSEDED")
        mark_aggregate_actions_status_by_report(existing["id"], "PENDING", "SUPERSEDED")

    pdf_filename = f"RetirementPlan_{os.urandom(8).hex()}.pdf"
    pdf_path = os.path.join(OUTPUT_DIR, pdf_filename)
    actions = dashboard_action_items(analysis)
    report_id = None
    pdf_created = False
    try:
        generate_retirement_pdf(analysis, pdf_path, logo_path=LOGO_PATH)
        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) <= 0:
            raise RuntimeError("Generated retirement PDF was not saved")
        pdf_created = True
        generated_at = datetime.utcnow()
        expires_at = (generated_at + timedelta(days=90)).isoformat()
        snapshot = _retirement_dashboard_snapshot(analysis)
        report_id = insert_dashboard_report(
            mfd_firebase_uid=user_id,
            client_pan=client_pan,
            client_name=profile["client_name"],
            expires_at=expires_at,
            pdf_filename=pdf_filename,
            status="ACTIVE",
            snapshot_json=json.dumps(snapshot, default=str),
            action_items_json=json.dumps(actions, default=str),
            report_type="retirement",
        )
        for item in actions:
            insert_aggregate_action(
                id=str(uuid.uuid4()),
                mfd_firebase_uid=user_id,
                client_pan=client_pan,
                report_id=report_id,
                item_id=item["item_id"],
                dimension=item["dimension"],
                urgency=item["urgency"],
                value_type=item["value_type"],
                value_num=item["value_num"],
                final_status="PENDING",
                report_generated_at=generated_at.isoformat(),
            )
        if not consume_user_credit(user_id):
            raise RuntimeError("Failed to consume report credit")
        if qid:
            save_retirement_questionnaire(qid, user_id, payload, status="completed")
    except Exception as exc:
        logger.exception("Retirement report generation failed")
        if report_id:
            mark_dashboard_report_status(report_id, "FAILED")
            mark_aggregate_actions_status_by_report(report_id, "PENDING", "FAILED")
        if pdf_created and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except OSError:
                pass
        return jsonify({"error": "Failed to generate retirement report", "message": str(exc)}), 500

    url = url_for("documents.download_file", filename=pdf_filename, _external=True)
    return jsonify(
        {
            "retirement_plan_pdf_url": url,
            "financial_plan_pdf_url": url,
            "report_type": "retirement",
            "report_id": report_id,
            "questionnaire_id": qid,
            "analysis": analysis,
            "remaining_credits": get_user_credits(user_id),
        }
    ), 200
