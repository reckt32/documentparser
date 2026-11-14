"""Financial plan report generation endpoint."""

import json
import logging
import os
import re
import uuid
from datetime import datetime, timedelta

from flask import Blueprint, g, jsonify, request, url_for

from backend.analysis.client_facts import _assemble_financial_inputs, _build_client_facts
from backend.analysis.dashboard import _build_dashboard_action_items, _build_dashboard_snapshot
from backend.analysis.health import analyze_financial_health
from backend.analysis.llm_sections import run_report_sections
from backend.auth import require_auth, require_payment
from backend.config import OUTPUT_DIR
from backend.db import (
    consume_user_credit,
    get_active_dashboard_report_by_pan,
    get_questionnaire,
    get_user_credits,
    insert_aggregate_action,
    insert_dashboard_report,
    mark_aggregate_actions_status_by_report,
    mark_dashboard_report_status,
)
from backend.extraction.insights import aggregate_doc_insights_for_questionnaire
from backend.reports.financial_plan import generate_financial_plan_pdf
from backend.reports.helpers import _allocation_output

logger = logging.getLogger(__name__)
bp = Blueprint("reports", __name__)


@bp.route("/report/generate", methods=["POST"])
@require_auth
@require_payment
def report_generate():
    data = request.get_json(force=True) or {}
    questionnaire_id = data.get("questionnaire_id")
    if not questionnaire_id:
        return jsonify({"error": "questionnaire_id required"}), 400
    q = get_questionnaire(questionnaire_id)
    if not q:
        return jsonify({"error": "questionnaire not found"}), 404

    # --- 1. Resolve authenticated MFD user and verify available credits ---
    current_user = getattr(g, "current_user", None) or {}
    firebase_uid = current_user.get("firebase_uid") or getattr(g, "user_id", None)
    if not firebase_uid:
        return jsonify({"error": "Authentication required"}), 401

    live_credits = get_user_credits(firebase_uid)
    if live_credits <= 0:
        return jsonify({
            "error": "Insufficient credits",
            "message": "You have no report credits remaining. Please purchase more credits to continue.",
            "remaining_credits": 0,
        }), 403

    # --- 2. Extract & validate PAN from personal_info for the duplicate check ---
    personal_info = q.get("personal_info") or {}
    raw_pan = personal_info.get("pan")
    if not raw_pan:
        return jsonify({"error": "PAN missing in questionnaire personal_info"}), 400
    client_pan = str(raw_pan).strip().upper()
    if not re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", client_pan):
        return jsonify({"error": "Invalid PAN format in questionnaire personal_info"}), 400
    client_name = (personal_info.get("name") or current_user.get("display_name") or "Client")

    # --- 3. Duplicate Check: supersede any existing ACTIVE report for same MFD + PAN ---
    try:
        existing_active = get_active_dashboard_report_by_pan(firebase_uid, client_pan, "wealth")
    except Exception as e:
        logger.error(f"Failed to query active dashboard report: {e}")
        existing_active = None

    if existing_active:
        try:
            mark_dashboard_report_status(existing_active["id"], "SUPERSEDED")
        except Exception as e:
            logger.error(f"Failed to mark existing dashboard report {existing_active.get('id')} as SUPERSEDED: {e}")
        try:
            mark_aggregate_actions_status_by_report(existing_active["id"], "PENDING", "SUPERSEDED")
        except Exception as e:
            logger.error(f"Failed to mark pending aggregate_actions as SUPERSEDED for report {existing_active.get('id')}: {e}")
        old_pdf_filename = existing_active.get("pdf_filename")
        if old_pdf_filename:
            old_pdf_path = os.path.join(OUTPUT_DIR, old_pdf_filename)
            try:
                if os.path.exists(old_pdf_path):
                    os.remove(old_pdf_path)
            except Exception as e:
                logger.error(f"Failed to delete old PDF '{old_pdf_path}': {e}")

    # --- 4. Build analysis context (used for both PDF generation and snapshot) ---
    doc_insights = {}
    try:
        doc_insights = aggregate_doc_insights_for_questionnaire(questionnaire_id)
    except Exception as e:
        doc_insights = {"error": str(e)}

    inputs = _assemble_financial_inputs(q, doc_insights)
    analysis = analyze_financial_health(inputs)

    # Build client_facts early — needed for PDF, dashboard snapshot, and action items
    client_facts = _build_client_facts(q, analysis, doc_insights)

    sections = None
    try:
        sections = run_report_sections(questionnaire_id, client_facts)
    except Exception as e:
        analysis["llm_error"] = str(e)

    # Compute allocation_output — same engine the PDF pages use
    allocation_output = _allocation_output(client_facts)

    # --- 5. Generate the PDF, then persist dashboard records, then deduct credit ---
    pdf_filename = f"FinancialPlan_{os.urandom(8).hex()}.pdf"
    pdf_path = os.path.join(OUTPUT_DIR, pdf_filename)
    pdf_created = False
    try:
        generate_financial_plan_pdf(q, analysis, pdf_path, doc_insights=doc_insights, narratives=sections)
        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) <= 0:
            raise RuntimeError("Generated PDF was not saved to disk")
        pdf_created = True

        # 5a. Build snapshot_json + action_items_json from full analysis + client_facts + allocation
        snapshot_payload = _build_dashboard_snapshot(analysis, q, doc_insights, client_facts, allocation_output)
        action_items = _build_dashboard_action_items(analysis, client_facts, allocation_output)
        snapshot_json = json.dumps(snapshot_payload, default=str)
        action_items_json = json.dumps(action_items, default=str)

        # 5b. Compute expiry and persist the new dashboard report
        generated_at = datetime.utcnow()
        expires_at = (generated_at + timedelta(days=90)).isoformat()
        report_id = insert_dashboard_report(
            mfd_firebase_uid=firebase_uid,
            client_pan=client_pan,
            client_name=client_name,
            expires_at=expires_at,
            pdf_filename=pdf_filename,
            status="ACTIVE",
            snapshot_json=snapshot_json,
            action_items_json=action_items_json,
        )

        # 5c. Persist every action item with status PENDING
        report_generated_at_iso = generated_at.isoformat()
        for item in action_items:
            insert_aggregate_action(
                id=str(uuid.uuid4()),
                mfd_firebase_uid=firebase_uid,
                client_pan=client_pan,
                report_id=report_id,
                item_id=item.get("item_id") or str(uuid.uuid4()),
                dimension=item.get("dimension"),
                urgency=item.get("urgency"),
                value_type=item.get("value_type"),
                value_num=item.get("value_num"),
                final_status="PENDING",
                report_generated_at=report_generated_at_iso,
            )

        # 5d. Only now deduct the token (atomically, after everything is saved)
        if not consume_user_credit(firebase_uid):
            raise RuntimeError("Failed to consume credit after report generation")

    except Exception as e:
        logger.error(f"/report/generate failed for user {firebase_uid}, questionnaire {questionnaire_id}: {e}")
        if pdf_created:
            try:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
            except Exception as cleanup_err:
                logger.error(f"Failed to clean up new PDF '{pdf_path}' after error: {cleanup_err}")
        return jsonify({
            "error": "Failed to generate report",
            "message": str(e),
        }), 500

    url = url_for("documents.download_file", filename=pdf_filename, _external=True)
    remaining_credits = get_user_credits(firebase_uid)

    return jsonify({
        "financial_plan_pdf_url": url,
        "summary_pdf_url": url,
        "report_type": "financial_plan",
        "questionnaire_id": questionnaire_id,
        "report_id": report_id,
        "client_pan": client_pan,
        "analysis": analysis,
        "docInsights": doc_insights,
        "sections": sections,
        "remaining_credits": remaining_credits,
    }), 200
