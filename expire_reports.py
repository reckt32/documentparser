"""
expire_reports.py — daily cron job to prune expired dashboard reports.

Reports are persisted with an ``expires_at`` timestamp set 90 days after
generation. Azure's free web tier only gives us ephemeral local storage,
so the PDFs and the related aggregate-action rows must be cleaned up
periodically to avoid filling the disk.

Run this script once a day from Linux ``cron``::

    0 3 * * * cd /home/site/wwwroot/backend && /home/site/wwwroot/backend/venv/bin/python expire_reports.py >> /var/log/meerkat-expire.log 2>&1

Behaviour for each expired ACTIVE report:

1. Delete ``backend/output/<pdf_filename>`` from disk (errors are logged
   but do not abort the run — a missing file is not fatal).
2. Flip every still-``PENDING`` row in ``aggregate_actions`` for that
   report to ``MISSED``.
3. Mark the report itself ``EXPIRED`` so it is no longer served.

The script is idempotent: re-running it on the same day is a no-op
because the WHERE clause already filters by ``status = 'ACTIVE'``.
"""

import logging
import os
import sys
import traceback
from typing import Dict, List

# db.py is the only dependency we need. It opens the same SQLite file
# the Flask app uses, initialises the schema if missing, and exposes the
# helpers we need for the prune logic.
from db import (
    list_expired_active_dashboard_reports,
    mark_aggregate_actions_status_by_report,
    mark_dashboard_report_status,
)

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

# Resolve OUTPUT_DIR the same way app.py does, but anchored to THIS file
# so the script works no matter what directory cron invokes it from.
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_HERE, os.getenv("OUTPUT_DIR_NAME", "output"))

# Optional log file — falls back to stdout (which cron captures / emails).
LOG_FILE = os.getenv("EXPIRE_REPORTS_LOG", "").strip() or None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("expire_reports")
logger.setLevel(logging.INFO)
_log_format = logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Always log to stdout so cron can capture it.
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setFormatter(_log_format)
logger.addHandler(_stdout_handler)

if LOG_FILE:
    try:
        _file_handler = logging.FileHandler(LOG_FILE)
        _file_handler.setFormatter(_log_format)
        logger.addHandler(_file_handler)
    except OSError as exc:
        # Don't kill the job just because we can't open the log file.
        logger.warning("Could not open log file %s: %s", LOG_FILE, exc)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _delete_pdf(pdf_filename: str) -> bool:
    """
    Best-effort delete of ``OUTPUT_DIR/<pdf_filename>``.

    Returns True if the file was deleted, False if it was already missing
    or could not be removed. Never raises — a missing/unlocked file
    must not abort the expiry run.
    """
    if not pdf_filename:
        return False

    pdf_path = os.path.join(OUTPUT_DIR, pdf_filename)
    if not os.path.exists(pdf_path):
        logger.info("  PDF already absent on disk: %s", pdf_path)
        return False

    try:
        os.remove(pdf_path)
        logger.info("  Deleted PDF: %s", pdf_path)
        return True
    except OSError as exc:
        logger.warning("  Could not delete PDF %s: %s", pdf_path, exc)
        return False


def _expire_report(report: Dict) -> Dict:
    """
    Run the 3-step expiry for a single dashboard report and return a
    small summary dict that the caller can use for the run totals.
    """
    report_id = report["id"]
    pan = report.get("client_pan")
    pdf_filename = report.get("pdf_filename")
    expires_at = report.get("expires_at")

    logger.info(
        "Expiring report id=%s pan=%s expires_at=%s pdf=%s",
        report_id, pan, expires_at, pdf_filename,
    )

    pdf_deleted = _delete_pdf(pdf_filename)

    try:
        actions_missed = mark_aggregate_actions_status_by_report(
            report_id=report_id,
            from_status="PENDING",
            to_status="MISSED",
        )
    except Exception as exc:
        # Surface but don't crash — the report is the source of truth.
        logger.error("  Failed to mark aggregate_actions for report %s: %s", report_id, exc)
        actions_missed = 0

    try:
        report_marked = mark_dashboard_report_status(report_id, "EXPIRED")
    except Exception as exc:
        logger.error("  Failed to mark report %s as EXPIRED: %s", report_id, exc)
        report_marked = False

    return {
        "report_id": report_id,
        "client_pan": pan,
        "pdf_filename": pdf_filename,
        "pdf_deleted": pdf_deleted,
        "actions_marked_missed": actions_missed,
        "report_marked_expired": bool(report_marked),
    }


def run() -> int:
    """
    Main entry point. Returns a Unix exit code (0 on success, even when
    no reports were due; non-zero on a hard failure).
    """
    try:
        expired: List[Dict] = list_expired_active_dashboard_reports()
    except Exception as exc:
        logger.error("Failed to query expired reports: %s", exc)
        logger.error(traceback.format_exc())
        return 1

    if not expired:
        logger.info("No expired ACTIVE reports to process. Exiting cleanly.")
        return 0

    logger.info("Found %d expired ACTIVE report(s).", len(expired))

    pdfs_deleted = 0
    actions_missed = 0
    reports_expired = 0
    failures = 0

    for report in expired:
        try:
            summary = _expire_report(report)
            if summary["pdf_deleted"]:
                pdfs_deleted += 1
            actions_missed += summary["actions_marked_missed"]
            if summary["report_marked_expired"]:
                reports_expired += 1
            else:
                failures += 1
        except Exception as exc:
            failures += 1
            logger.error(
                "Unexpected error while expiring report %s: %s",
                report.get("id"), exc,
            )
            logger.error(traceback.format_exc())

    logger.info(
        "Expire run complete: %d report(s) processed, %d PDF(s) deleted, "
        "%d action(s) marked MISSED, %d failure(s).",
        len(expired), pdfs_deleted, actions_missed, failures,
    )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
