"""
Meerkat Dashboard API

Provides the read/write endpoints the Flutter dashboard uses to render
opportunity metrics, browse a client's most recent active report, mark
individual action items as CONVERTED / PENDING, and query a custom
period of activity.

All routes are mounted under ``/api/dashboard`` and require a valid
Firebase ID token (the MFD). Every database query is strictly scoped
to the authenticated MFD's ``mfd_firebase_uid`` so that no MFD can
read or mutate another MFD's data.

Endpoints
---------
GET  /api/dashboard/overview
    Aggregate metrics across all the MFD's clients (totals + per-category
    breakdown) plus the list of active reports ordered by nearest expiry.

GET  /api/dashboard/client/<pan>
    Snapshot + action items for the active report belonging to ``<pan>``.

PUT  /api/dashboard/action/<item_id>
    Mark a single action item as CONVERTED or PENDING.

GET  /api/dashboard/annual?period=<period>
    Identified / converted totals + conversion % over a rolling period.
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from flask import Blueprint, g, jsonify, request

from auth import require_auth
from db import (
    get_active_dashboard_report_by_pan,
    get_aggregate_metrics_for_period,
    get_aggregate_metrics_overview,
    list_active_dashboard_reports,
    update_aggregate_action_status_for_mfd,
    get_aggregate_action_statuses,
)

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/api/dashboard")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_ACTION_STATUSES = {"CONVERTED", "PENDING"}
PAN_PATTERN = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
MISSED_QUARTER_DAYS = 90  # a calendar quarter
DEFAULT_ANNUAL_PERIOD_DAYS = 365


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_mfd_uid() -> Optional[str]:
    """
    Pull the authenticated MFD's firebase_uid out of Flask's ``g``.

    ``require_auth`` always sets ``g.current_user`` for valid tokens, so a
    None return means the route was somehow hit without auth.
    """
    user = getattr(g, "current_user", None) or {}
    return user.get("firebase_uid") or getattr(g, "user_id", None)


def _bad_request(message: str, code: int = 400):
    return jsonify({"error": message}), code


def _parse_json_field(raw: Optional[str]) -> Any:
    """
    The dashboard_reports table persists snapshot / action_items as JSON
    strings. The API contract is to return them parsed, so we attempt a
    decode and fall back to the raw string if the column is corrupt.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def _parse_period(period: Optional[str]) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Translate a free-form ``?period=`` string into a (start, end) UTC
    datetime tuple. Supported forms:

      - 30d / 90d / 180d / 365d         -> last N days
      - 6m / 12m                        -> last N months
      - 1y / 2y                         -> last N years
      - 2025                            -> calendar year 2025
      - 2025-Q1 / 2025q4                -> calendar quarter
      - 2025-03                         -> calendar month
      - ISO date (2025-03-15)           -> single day window
    """
    if not period:
        return (
            datetime.now(timezone.utc) - timedelta(days=DEFAULT_ANNUAL_PERIOD_DAYS),
            datetime.now(timezone.utc),
        )

    raw = period.strip().lower()
    now = datetime.now(timezone.utc)

    relative = re.match(r"^(\d+)\s*([dmy])$", raw)
    if relative:
        n = int(relative.group(1))
        unit = relative.group(2)
        if unit == "d":
            return (now - timedelta(days=n), now)
        if unit == "m":
            return (now - timedelta(days=30 * n), now)
        if unit == "y":
            return (now - timedelta(days=365 * n), now)

    quarter = re.match(r"^(\d{4})[\-\s]?q([1-4])$", raw)
    if quarter:
        year = int(quarter.group(1))
        q = int(quarter.group(2))
        start_month = 3 * (q - 1) + 1
        end_month = start_month + 3
        end_year = year
        if end_month > 12:
            end_month = 3
            end_year = year + 1
        start = datetime(year, start_month, 1, tzinfo=timezone.utc)
        end = datetime(end_year, end_month, 1, tzinfo=timezone.utc)
        return (start, end)

    month_match = re.match(r"^(\d{4})[\-\.](\d{1,2})$", raw)
    if month_match:
        year = int(month_match.group(1))
        month = int(month_match.group(2))
        if month == 12:
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
        return (datetime(year, month, 1, tzinfo=timezone.utc), end)

    year_match = re.match(r"^(\d{4})$", raw)
    if year_match:
        year = int(year_match.group(1))
        return (
            datetime(year, 1, 1, tzinfo=timezone.utc),
            datetime(year + 1, 1, 1, tzinfo=timezone.utc),
        )

    try:
        day = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return (day, day + timedelta(days=1))
    except ValueError:
        pass

    raise ValueError(
        f"Unrecognised period '{period}'. Use forms like 30d, 6m, 1y, 2025, 2025-Q1, 2025-03, or 2025-03-15."
    )


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# GET /api/dashboard/overview
# ---------------------------------------------------------------------------

@dashboard_bp.route("/overview", methods=["GET"])
@require_auth
def overview():
    """
    Return aggregate metrics and the MFD's active report list.

    Response shape::

        {
          "mfd_firebase_uid": "...",
          "metrics": {
            "total_opportunity": float,   # PENDING + CONVERTED value_num sum
            "converted":         float,
            "pending":           float,
            "missed_quarter":    float,   # PENDING items older than 90d
            "action_count":      int,
            "missed_quarter_days": int,
            "categories": [
              {"dimension": "sip", "total_opportunity": ..., ...},
              ...
            ]
          },
          "active_reports": [
            {"id": 12, "client_pan": "...", "client_name": "...",
             "generated_at": "...", "expires_at": "...", "pdf_filename": "..."},
            ...
          ]
        }
    """
    mfd_uid = _get_mfd_uid()
    if not mfd_uid:
        return _bad_request("Authentication required", 401)

    metrics = get_aggregate_metrics_overview(mfd_uid, missed_quarter_days=MISSED_QUARTER_DAYS)
    metrics["missed_quarter_days"] = MISSED_QUARTER_DAYS

    reports = list_active_dashboard_reports(mfd_uid)
    active_reports = []
    for r in reports:
        entry = {
            "id": r.get("id"),
            "client_pan": r.get("client_pan"),
            "client_name": r.get("client_name"),
            "generated_at": r.get("generated_at"),
            "expires_at": r.get("expires_at"),
            "pdf_filename": r.get("pdf_filename"),
            "status": r.get("status"),
        }
        # Extract per-client summary from snapshot_json (already stored)
        snapshot_raw = r.get("snapshot_json")
        if snapshot_raw:
            try:
                snap = json.loads(snapshot_raw) if isinstance(snapshot_raw, str) else snapshot_raw
                entry["health_score"] = (snap.get("overall_health") or {}).get("score")
                entry["health_label"] = (snap.get("overall_health") or {}).get("label")
                entry["risk_profile"] = snap.get("risk_profile")
                # Protection summary
                prot = snap.get("protection") or {}
                entry["life_cover_gap"] = prot.get("life_cover_gap")
                entry["health_cover_gap"] = prot.get("health_cover_gap")
                # Allocation summary
                alloc_sum = snap.get("allocation_summary") or {}
                entry["total_ideal_sip"] = alloc_sum.get("total_ideal_sip")
                entry["goal_achievement_pct"] = alloc_sum.get("goal_achievement_pct")
                # Goal count
                entry["goal_count"] = len(snap.get("goal_summary") or [])
            except Exception:
                pass
        active_reports.append(entry)

    return jsonify({
        "mfd_firebase_uid": mfd_uid,
        "metrics": metrics,
        "active_reports": active_reports,
        "active_report_count": len(active_reports),
    }), 200


# ---------------------------------------------------------------------------
# GET /api/dashboard/client/<pan>
# ---------------------------------------------------------------------------

@dashboard_bp.route("/client/<string:pan>", methods=["GET"])
@require_auth
def client_detail(pan: str):
    """
    Return the active dashboard report (snapshot + action items) for the
    given PAN, but only if the report belongs to the authenticated MFD.
    """
    mfd_uid = _get_mfd_uid()
    if not mfd_uid:
        return _bad_request("Authentication required", 401)

    normalized_pan = (pan or "").strip().upper()
    if not PAN_PATTERN.match(normalized_pan):
        return _bad_request("Invalid PAN format. Expected ABCDE1234F.")

    report = get_active_dashboard_report_by_pan(mfd_uid, normalized_pan)
    if not report:
        return jsonify({
            "client_pan": normalized_pan,
            "snapshot": None,
            "action_items": None,
            "message": "No active report found for this PAN.",
        }), 404

    action_items = _parse_json_field(report.get("action_items_json")) or []
    
    status_map = get_aggregate_action_statuses(report.get("id"))
    for item in action_items:
        iid = item.get("item_id")
        if iid in status_map:
            item["final_status"] = status_map[iid]
            item["is_converted"] = (status_map[iid] == "CONVERTED")

    return jsonify({
        "report_id": report.get("id"),
        "client_pan": report.get("client_pan"),
        "client_name": report.get("client_name"),
        "generated_at": report.get("generated_at"),
        "expires_at": report.get("expires_at"),
        "pdf_filename": report.get("pdf_filename"),
        "status": report.get("status"),
        "snapshot": _parse_json_field(report.get("snapshot_json")),
        "action_items": action_items,
    }), 200


# ---------------------------------------------------------------------------
# PUT /api/dashboard/action/<item_id>
# ---------------------------------------------------------------------------

@dashboard_bp.route("/action/<string:item_id>", methods=["PUT"])
@require_auth
def update_action(item_id: str):
    """
    Update a single aggregate_actions row's ``final_status``.

    Body: ``{"status": "CONVERTED"}`` or ``{"status": "PENDING"}``
    """
    mfd_uid = _get_mfd_uid()
    if not mfd_uid:
        return _bad_request("Authentication required", 401)

    if not item_id or not item_id.strip():
        return _bad_request("item_id is required", 400)

    payload = request.get_json(silent=True) or {}
    new_status = (payload.get("status") or "").strip().upper()
    if new_status not in ALLOWED_ACTION_STATUSES:
        return _bad_request(
            "Invalid status. Allowed values: CONVERTED, PENDING",
            400,
        )

    updated = update_aggregate_action_status_for_mfd(
        mfd_firebase_uid=mfd_uid,
        item_id=item_id.strip(),
        final_status=new_status,
    )
    if not updated:
        return jsonify({
            "error": "Action item not found",
            "message": "No action item with that ID exists for the authenticated MFD.",
        }), 404

    logger.info(
        "MFD %s set action %s -> %s",
        mfd_uid, item_id, new_status,
    )

    return jsonify({
        "success": True,
        "action": updated,
    }), 200


# ---------------------------------------------------------------------------
# GET /api/dashboard/annual?period=...
# ---------------------------------------------------------------------------

@dashboard_bp.route("/annual", methods=["GET"])
@require_auth
def annual_metrics():
    """
    Return identified / converted totals + conversion % over a period.

    Query string: ``?period=30d|6m|1y|2025|2025-Q1|2025-03|2025-03-15``

    Response shape::

        {
          "period": "30d",
          "period_start": "2025-05-04T00:00:00+00:00",
          "period_end":   "2025-06-03T00:00:00+00:00",
          "total_identified_value": float,
          "total_identified_count": int,
          "converted_value":        float,
          "converted_count":        int,
          "pending_value":          float,
          "conversion_pct":         float   # 0.0 - 100.0
        }
    """
    mfd_uid = _get_mfd_uid()
    if not mfd_uid:
        return _bad_request("Authentication required", 401)

    period_arg = request.args.get("period")
    try:
        start_dt, end_dt = _parse_period(period_arg)
    except ValueError as e:
        return _bad_request(str(e), 400)

    summary = get_aggregate_metrics_for_period(
        mfd_firebase_uid=mfd_uid,
        start_iso=_iso(start_dt),
        end_iso=_iso(end_dt),
    )

    return jsonify({
        "period": period_arg or f"{DEFAULT_ANNUAL_PERIOD_DAYS}d",
        "period_start": _iso(start_dt),
        "period_end": _iso(end_dt),
        "total_identified_value": float(summary.get("total_identified_value") or 0),
        "total_identified_count": int(summary.get("total_identified_count") or 0),
        "converted_value": float(summary.get("converted_value") or 0),
        "converted_count": int(summary.get("converted_count") or 0),
        "pending_value": float(summary.get("pending_value") or 0),
        "conversion_pct": float(summary.get("conversion_pct") or 0.0),
    }), 200
