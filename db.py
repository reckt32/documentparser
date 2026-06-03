import os
import json
import sqlite3
import logging
import threading
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

# SQLite file path:
# - Prefer DB_PATH env var (for persistent cloud storage mounts)
# - Fall back to local repo output/index.db for development
_BASE_DIR = os.path.dirname(__file__)
_DEFAULT_DB_DIR = os.path.join(_BASE_DIR, "output")
os.makedirs(_DEFAULT_DB_DIR, exist_ok=True)
DB_PATH = os.getenv("DB_PATH", os.path.join(_DEFAULT_DB_DIR, "index.db"))

# Ensure parent directory exists for custom DB paths (e.g., Azure /home/site/data)
_db_parent = os.path.dirname(DB_PATH)
if _db_parent:
    os.makedirs(_db_parent, exist_ok=True)

_CONN_LOCK = threading.Lock()
_CONN: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        with _CONN_LOCK:
            if _CONN is None:
                _CONN = sqlite3.connect(DB_PATH, check_same_thread=False)
                _CONN.row_factory = sqlite3.Row
    return _CONN


def _exec(sql: str, params: Iterable[Any] = ()):
    conn = _get_conn()
    with _CONN_LOCK:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur


def _query(sql: str, params: Iterable[Any] = ()):
    conn = _get_conn()
    with _CONN_LOCK:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
    return rows


def init_db():
    _exec(
        """
        CREATE TABLE IF NOT EXISTS documents (
          id INTEGER PRIMARY KEY,
          sha256 TEXT UNIQUE NOT NULL,
          filename TEXT,
          page_count INTEGER,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    _exec(
        """
        CREATE TABLE IF NOT EXISTS sections (
          id INTEGER PRIMARY KEY,
          document_id INTEGER NOT NULL,
          page_number INTEGER NOT NULL,
          heading TEXT,
          section_type TEXT,            -- snapshot | statement_summary | sip_summary | other
          y_top REAL,
          y_bottom REAL,
          text TEXT NOT NULL,
          FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
        )
        """
    )
    _exec(
        """
        CREATE TABLE IF NOT EXISTS tables (
          id INTEGER PRIMARY KEY,
          document_id INTEGER NOT NULL,
          section_id INTEGER,
          page_number INTEGER NOT NULL,
          header_json TEXT,
          rows_json TEXT,
          FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
          FOREIGN KEY(section_id) REFERENCES sections(id) ON DELETE SET NULL
        )
        """
    )
    _exec(
        """
        CREATE TABLE IF NOT EXISTS metrics (
          id INTEGER PRIMARY KEY,
          document_id INTEGER NOT NULL,
          key TEXT NOT NULL,
          value_num REAL,
          source_section_id INTEGER,
          FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
          FOREIGN KEY(source_section_id) REFERENCES sections(id) ON DELETE SET NULL
        )
        """
    )
    _exec("CREATE INDEX IF NOT EXISTS idx_sections_doc_page ON sections(document_id, page_number, y_top)")
    _exec("CREATE INDEX IF NOT EXISTS idx_metrics_doc_key ON metrics(document_id, key)")

    # Questionnaire: core container
    _exec(
        """
        CREATE TABLE IF NOT EXISTS questionnaires (
          id INTEGER PRIMARY KEY,
          user_id TEXT,
          status TEXT, -- in_progress | completed | archived
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    _exec("CREATE INDEX IF NOT EXISTS idx_questionnaires_user ON questionnaires(user_id, created_at)")

    # Questionnaire sections (normalized per section, JSON blob per row)
    _exec(
        """
        CREATE TABLE IF NOT EXISTS personal_info (
          questionnaire_id INTEGER PRIMARY KEY,
          data_json TEXT NOT NULL,
          FOREIGN KEY(questionnaire_id) REFERENCES questionnaires(id) ON DELETE CASCADE
        )
        """
    )
    _exec(
        """
        CREATE TABLE IF NOT EXISTS family_info (
          questionnaire_id INTEGER PRIMARY KEY,
          data_json TEXT NOT NULL,
          FOREIGN KEY(questionnaire_id) REFERENCES questionnaires(id) ON DELETE CASCADE
        )
        """
    )
    _exec(
        """
        CREATE TABLE IF NOT EXISTS goals (
          questionnaire_id INTEGER PRIMARY KEY,
          data_json TEXT NOT NULL,
          FOREIGN KEY(questionnaire_id) REFERENCES questionnaires(id) ON DELETE CASCADE
        )
        """
    )
    _exec(
        """
        CREATE TABLE IF NOT EXISTS risk_profile (
          questionnaire_id INTEGER PRIMARY KEY,
          data_json TEXT NOT NULL,
          FOREIGN KEY(questionnaire_id) REFERENCES questionnaires(id) ON DELETE CASCADE
        )
        """
    )
    _exec(
        """
        CREATE TABLE IF NOT EXISTS insurance (
          questionnaire_id INTEGER PRIMARY KEY,
          data_json TEXT NOT NULL,
          FOREIGN KEY(questionnaire_id) REFERENCES questionnaires(id) ON DELETE CASCADE
        )
        """
    )
    _exec(
        """
        CREATE TABLE IF NOT EXISTS estate (
          questionnaire_id INTEGER PRIMARY KEY,
          data_json TEXT NOT NULL,
          FOREIGN KEY(questionnaire_id) REFERENCES questionnaires(id) ON DELETE CASCADE
        )
        """
    )
    _exec(
        """
        CREATE TABLE IF NOT EXISTS lifestyle (
          questionnaire_id INTEGER PRIMARY KEY,
          data_json TEXT NOT NULL,
          FOREIGN KEY(questionnaire_id) REFERENCES questionnaires(id) ON DELETE CASCADE
        )
        """
    )
    _exec(
        """
        CREATE TABLE IF NOT EXISTS tax_info (
          questionnaire_id INTEGER PRIMARY KEY,
          data_json TEXT NOT NULL,
          FOREIGN KEY(questionnaire_id) REFERENCES questionnaires(id) ON DELETE CASCADE
        )
        """
    )

    # Link uploaded docs to questionnaire
    _exec(
        """
        CREATE TABLE IF NOT EXISTS questionnaire_uploads (
          id INTEGER PRIMARY KEY,
          questionnaire_id INTEGER NOT NULL,
          document_id INTEGER,
          sha256 TEXT,
          doc_type TEXT,    -- Bank statement | ITR | Insurance document | Mutual fund CAS...
          filename TEXT,
          metadata_json TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(questionnaire_id) REFERENCES questionnaires(id) ON DELETE CASCADE,
          FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE SET NULL
        )
        """
    )
    _exec("CREATE INDEX IF NOT EXISTS idx_q_uploads_qid ON questionnaire_uploads(questionnaire_id)")

    # Users table for Firebase authentication and payment tracking
    _exec(
        """
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY,
          firebase_uid TEXT UNIQUE NOT NULL,
          email TEXT,
          display_name TEXT,
          has_paid INTEGER DEFAULT 0,
          report_credits INTEGER DEFAULT 0,
          payment_id TEXT,
          payment_order_id TEXT,
          payment_amount_paise INTEGER,
          payment_date TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    _exec("CREATE INDEX IF NOT EXISTS idx_users_firebase_uid ON users(firebase_uid)")
    
    # Add report_credits column if it doesn't exist (migration for existing databases)
    try:
        _exec("ALTER TABLE users ADD COLUMN report_credits INTEGER DEFAULT 0")
    except Exception:
        pass  # Column already exists
    
    # Payments table for tracking payment history and credits granted
    _exec(
        """
        CREATE TABLE IF NOT EXISTS payments (
          id INTEGER PRIMARY KEY,
          user_id INTEGER NOT NULL,
          firebase_uid TEXT NOT NULL,
          payment_id TEXT NOT NULL,
          order_id TEXT,
          amount_paise INTEGER,
          credits_granted INTEGER DEFAULT 3,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )
    _exec("CREATE INDEX IF NOT EXISTS idx_payments_firebase_uid ON payments(firebase_uid)")
    _exec("CREATE INDEX IF NOT EXISTS idx_payments_user_id ON payments(user_id)")
    _exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_payments_payment_id ON payments(payment_id)")

    # Dashboard Reports Table
    _exec(
        """
        CREATE TABLE IF NOT EXISTS dashboard_reports (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          mfd_firebase_uid TEXT NOT NULL,
          client_pan TEXT NOT NULL,
          client_name TEXT,
          generated_at TEXT DEFAULT CURRENT_TIMESTAMP,
          expires_at TEXT,
          pdf_filename TEXT,
          status TEXT,
          snapshot_json TEXT,
          action_items_json TEXT,
          FOREIGN KEY(mfd_firebase_uid) REFERENCES users(firebase_uid) ON DELETE CASCADE
        )
        """
    )
    _exec("CREATE INDEX IF NOT EXISTS idx_dashboard_reports_pan ON dashboard_reports(client_pan)")
    _exec("CREATE INDEX IF NOT EXISTS idx_dashboard_reports_mfd ON dashboard_reports(mfd_firebase_uid)")

    # Aggregate Actions Table
    _exec(
        """
        CREATE TABLE IF NOT EXISTS aggregate_actions (
          id TEXT PRIMARY KEY,
          mfd_firebase_uid TEXT NOT NULL,
          client_pan TEXT NOT NULL,
          report_id INTEGER NOT NULL,
          item_id TEXT NOT NULL,
          dimension TEXT,
          urgency TEXT,
          value_type TEXT,
          value_num REAL,
          final_status TEXT,
          report_generated_at TEXT,
          actioned_at TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(mfd_firebase_uid) REFERENCES users(firebase_uid) ON DELETE CASCADE,
          FOREIGN KEY(report_id) REFERENCES dashboard_reports(id) ON DELETE CASCADE
        )
        """
    )
    _exec("CREATE INDEX IF NOT EXISTS idx_aggregate_actions_mfd ON aggregate_actions(mfd_firebase_uid)")
    _exec("CREATE INDEX IF NOT EXISTS idx_aggregate_actions_report ON aggregate_actions(report_id)")


def upsert_document(sha256: str, filename: str, page_count: int) -> int:
    rows = _query("SELECT id FROM documents WHERE sha256=?", (sha256,))
    if rows:
        return rows[0]["id"]
    cur = _exec(
        "INSERT INTO documents (sha256, filename, page_count) VALUES (?, ?, ?)",
        (sha256, filename, page_count),
    )
    return cur.lastrowid


def get_document_by_sha(sha256: str) -> Optional[sqlite3.Row]:
    rows = _query("SELECT * FROM documents WHERE sha256=?", (sha256,))
    return rows[0] if rows else None


def insert_section(
    document_id: int,
    page_number: int,
    heading: Optional[str],
    section_type: str,
    y_top: Optional[float],
    y_bottom: Optional[float],
    text: str,
) -> int:
    cur = _exec(
        """
        INSERT INTO sections (document_id, page_number, heading, section_type, y_top, y_bottom, text)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (document_id, page_number, heading, section_type, y_top, y_bottom, text),
    )
    return cur.lastrowid


def insert_table(
    document_id: int,
    section_id: Optional[int],
    page_number: int,
    header: Iterable[str],
    rows: Iterable[Iterable[str]],
) -> int:
    header_json = json.dumps(list(header) if header else [])
    rows_json = json.dumps([list(r) for r in rows] if rows else [])
    cur = _exec(
        """
        INSERT INTO tables (document_id, section_id, page_number, header_json, rows_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (document_id, section_id, page_number, header_json, rows_json),
    )
    return cur.lastrowid


def insert_metric(document_id: int, key: str, value_num: Optional[float], source_section_id: Optional[int]) -> int:
    cur = _exec(
        """
        INSERT INTO metrics (document_id, key, value_num, source_section_id)
        VALUES (?, ?, ?, ?)
        """,
        (document_id, key, value_num, source_section_id),
    )
    return cur.lastrowid


def delete_metrics_for_doc_keys(document_id: int, keys: Iterable[str]):
    keys = list(keys)
    if not keys:
        return
    placeholders = ",".join("?" for _ in keys)
    _exec(
        f"DELETE FROM metrics WHERE document_id=? AND key IN ({placeholders})",
        (document_id, *keys),
    )


def list_sections(document_id: int):
    return _query(
        """
        SELECT id, page_number, heading, section_type, y_top, y_bottom, substr(text,1,200) AS preview
        FROM sections WHERE document_id=?
        ORDER BY page_number ASC, y_top ASC
        """,
        (document_id,),
    )


def list_metrics(document_id: int):
    return _query(
        """
        SELECT m.id, m.key, m.value_num, m.source_section_id, s.page_number, s.section_type
        FROM metrics m
        LEFT JOIN sections s ON s.id = m.source_section_id
        WHERE m.document_id=?
        ORDER BY m.key ASC
        """,
        (document_id,),
    )


# Questionnaire helpers

def create_questionnaire(user_id: str, status: str = "in_progress") -> int:
    cur = _exec(
        "INSERT INTO questionnaires (user_id, status) VALUES (?, ?)",
        (user_id, status),
    )
    return cur.lastrowid

def update_questionnaire_status(qid: int, status: str):
    _exec(
        "UPDATE questionnaires SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (status, qid),
    )

def _upsert_section(table: str, questionnaire_id: int, data: Dict[str, Any]):
    payload = json.dumps(data or {})
    # Try insert, if conflict then update (SQLite UPSERT)
    _exec(
        f"""
        INSERT INTO {table} (questionnaire_id, data_json)
        VALUES (?, ?)
        ON CONFLICT(questionnaire_id) DO UPDATE SET
          data_json=excluded.data_json
        """,
        (questionnaire_id, payload),
    )

def save_personal_info(qid: int, data: Dict[str, Any]): _upsert_section("personal_info", qid, data)
def save_family_info(qid: int, data: Dict[str, Any]): _upsert_section("family_info", qid, data)
def save_goals(qid: int, data: Dict[str, Any]): _upsert_section("goals", qid, data)
def save_risk_profile(qid: int, data: Dict[str, Any]): _upsert_section("risk_profile", qid, data)
def save_insurance(qid: int, data: Dict[str, Any]): _upsert_section("insurance", qid, data)
def save_estate(qid: int, data: Dict[str, Any]): _upsert_section("estate", qid, data)
def save_lifestyle(qid: int, data: Dict[str, Any]): _upsert_section("lifestyle", qid, data)
def save_tax_info(qid: int, data: Dict[str, Any]): _upsert_section("tax_info", qid, data)

def get_questionnaire(qid: int) -> Dict[str, Any]:
    # base row
    rows = _query("SELECT id, user_id, status, created_at, updated_at FROM questionnaires WHERE id=?", (qid,))
    if not rows:
        return {}
    base = dict(rows[0])

    def _fetch_section(table: str) -> Optional[Dict[str, Any]]:
        srows = _query(f"SELECT data_json FROM {table} WHERE questionnaire_id=?", (qid,))
        if not srows:
            return None
        try:
            return json.loads(srows[0]["data_json"] or "{}")
        except Exception:
            return None

    return {
        "id": base["id"],
        "user_id": base["user_id"],
        "status": base["status"],
        "created_at": base["created_at"],
        "updated_at": base["updated_at"],
        "personal_info": _fetch_section("personal_info"),
        "family_info": _fetch_section("family_info"),
        "goals": _fetch_section("goals"),
        "risk_profile": _fetch_section("risk_profile"),
        "insurance": _fetch_section("insurance"),
        "estate": _fetch_section("estate"),
        "lifestyle": _fetch_section("lifestyle"),
        "tax_info": _fetch_section("tax_info"),
    }

def get_latest_questionnaire_for_user(user_id: str) -> Optional[int]:
    rows = _query(
        "SELECT id FROM questionnaires WHERE user_id=? ORDER BY created_at DESC LIMIT 1",
        (user_id,),
    )
    return rows[0]["id"] if rows else None

def link_questionnaire_upload(
    questionnaire_id: int,
    document_id: Optional[int],
    sha256: Optional[str],
    doc_type: Optional[str],
    filename: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    cur = _exec(
        """
        INSERT INTO questionnaire_uploads (questionnaire_id, document_id, sha256, doc_type, filename, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (questionnaire_id, document_id, sha256, doc_type, filename, json.dumps(metadata or {})),
    )
    return cur.lastrowid

def list_questionnaire_uploads(questionnaire_id: int):
    return _query(
        """
        SELECT id, questionnaire_id, document_id, sha256, doc_type, filename, metadata_json, created_at
        FROM questionnaire_uploads
        WHERE questionnaire_id=?
        ORDER BY created_at DESC
        """,
        (questionnaire_id,),
    )

def update_questionnaire_upload_metadata(upload_id: int, metadata: Dict[str, Any]):
    """Update metadata_json for an existing upload record."""
    _exec(
        "UPDATE questionnaire_uploads SET metadata_json = ? WHERE id = ?",
        (json.dumps(metadata), upload_id)
    )


# User helpers for Firebase authentication
def get_user_by_firebase_uid(firebase_uid: str) -> Optional[Dict[str, Any]]:
    """Get user by Firebase UID."""
    rows = _query("SELECT * FROM users WHERE firebase_uid = ?", (firebase_uid,))
    if not rows:
        return None
    row = rows[0]
    # Handle report_credits column which might not exist in older databases
    try:
        report_credits = row["report_credits"] or 0
    except (KeyError, IndexError):
        report_credits = 0
    return {
        "id": row["id"],
        "firebase_uid": row["firebase_uid"],
        "email": row["email"],
        "display_name": row["display_name"],
        "has_paid": bool(row["has_paid"]),
        "report_credits": report_credits,
        "payment_id": row["payment_id"],
        "payment_order_id": row["payment_order_id"],
        "payment_amount_paise": row["payment_amount_paise"],
        "payment_date": row["payment_date"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def create_or_update_user(firebase_uid: str, email: str = None, display_name: str = None) -> int:
    """Create a new user or update existing user info. Returns user id."""
    existing = get_user_by_firebase_uid(firebase_uid)
    if existing:
        # Update existing user
        _exec(
            """
            UPDATE users SET email = COALESCE(?, email), display_name = COALESCE(?, display_name),
            updated_at = CURRENT_TIMESTAMP WHERE firebase_uid = ?
            """,
            (email, display_name, firebase_uid)
        )
        return existing["id"]
    else:
        # Create new user
        cur = _exec(
            "INSERT INTO users (firebase_uid, email, display_name) VALUES (?, ?, ?)",
            (firebase_uid, email, display_name)
        )
        return cur.lastrowid



def mark_user_as_paid(firebase_uid: str, payment_id: str, order_id: str = None, amount_paise: int = None, credits_to_add: int = 3) -> bool:
    """
    Mark a user as paid after successful payment.
    Adds credits and records payment in payments table.
    
    IDEMPOTENT: If payment_id was already processed, skips credit addition
    and returns True. This prevents duplicate credits from verify + webhook + reconcile.
    
    Returns True if updated (or already processed).
    """
    # Idempotency: check if this payment_id was already processed
    existing = _query("SELECT id FROM payments WHERE payment_id = ?", (payment_id,))
    if existing:
        logger.info(f"Payment {payment_id} already processed for user {firebase_uid} (idempotent skip)")
        return True
    
    user = get_user_by_firebase_uid(firebase_uid)
    if not user:
        return False
    
    # Update user record with payment info and add credits
    result = _exec(
        """
        UPDATE users SET has_paid = 1, payment_id = ?, payment_order_id = ?,
        payment_amount_paise = ?, payment_date = CURRENT_TIMESTAMP,
        report_credits = report_credits + ?,
        updated_at = CURRENT_TIMESTAMP WHERE firebase_uid = ?
        """,
        (payment_id, order_id, amount_paise, credits_to_add, firebase_uid)
    )
    
    if result.rowcount > 0:
        # Record payment in payments table for audit trail
        _exec(
            """
            INSERT INTO payments (user_id, firebase_uid, payment_id, order_id, amount_paise, credits_granted)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user["id"], firebase_uid, payment_id, order_id, amount_paise, credits_to_add)
        )
        return True
    return False


def check_user_payment_status(firebase_uid: str) -> bool:
    """Check if user has paid. Returns False if user doesn't exist."""
    rows = _query("SELECT has_paid FROM users WHERE firebase_uid = ?", (firebase_uid,))
    if not rows:
        return False
    return bool(rows[0]["has_paid"])


def get_user_credits(firebase_uid: str) -> int:
    """Get remaining report credits for a user. Returns 0 if user doesn't exist."""
    rows = _query("SELECT report_credits FROM users WHERE firebase_uid = ?", (firebase_uid,))
    if not rows:
        return 0
    return rows[0]["report_credits"] or 0


def add_user_credits(firebase_uid: str, credits: int = 3) -> bool:
    """Add credits to a user's account. Returns True if updated."""
    result = _exec(
        """
        UPDATE users SET report_credits = report_credits + ?,
        updated_at = CURRENT_TIMESTAMP WHERE firebase_uid = ?
        """,
        (credits, firebase_uid)
    )
    return result.rowcount > 0


def consume_user_credit(firebase_uid: str) -> bool:
    """
    Atomically consume one report credit from user's account.
    Returns True if successful, False if no credits available or user doesn't exist.
    
    Uses atomic SQL UPDATE with WHERE clause to prevent race conditions.
    """
    result = _exec(
        """
        UPDATE users SET report_credits = report_credits - 1,
        updated_at = CURRENT_TIMESTAMP
        WHERE firebase_uid = ? AND report_credits > 0
        """,
        (firebase_uid,)
    )
    return result.rowcount > 0


def get_payment_history(firebase_uid: str) -> list:
    """Get payment history for a user."""
    rows = _query(
        """
        SELECT id, payment_id, order_id, amount_paise, credits_granted, created_at
        FROM payments WHERE firebase_uid = ?
        ORDER BY created_at DESC
        """,
        (firebase_uid,)
    )
    return [dict(row) for row in rows]


def migrate_existing_paid_users(credits: int = 3) -> int:
    """
    One-time migration: Grant credits to existing paid users who have 0 credits.
    Returns number of users updated.
    """
    result = _exec(
        """
        UPDATE users SET report_credits = ?
        WHERE has_paid = 1 AND (report_credits IS NULL OR report_credits = 0)
        """,
        (credits,)
    )
    return result.rowcount


# --- Admin helpers ---

def list_all_users(page: int = 1, per_page: int = 25, search: Optional[str] = None):
    """List all users with pagination and optional email search. Returns (users, total)."""
    offset = (page - 1) * per_page

    if search:
        like = f"%{search}%"
        total_rows = _query(
            "SELECT COUNT(*) AS cnt FROM users WHERE email LIKE ? OR display_name LIKE ?",
            (like, like),
        )
        total = total_rows[0]["cnt"] if total_rows else 0
        rows = _query(
            """
            SELECT id, firebase_uid, email, display_name, has_paid,
                   report_credits, payment_date, created_at, updated_at
            FROM users
            WHERE email LIKE ? OR display_name LIKE ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (like, like, per_page, offset),
        )
    else:
        total_rows = _query("SELECT COUNT(*) AS cnt FROM users")
        total = total_rows[0]["cnt"] if total_rows else 0
        rows = _query(
            """
            SELECT id, firebase_uid, email, display_name, has_paid,
                   report_credits, payment_date, created_at, updated_at
            FROM users
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (per_page, offset),
        )

    users = [dict(r) for r in rows]
    return users, total


def delete_user(firebase_uid: str) -> bool:
    """Delete a user and their payment history. Returns True if user existed."""
    user = get_user_by_firebase_uid(firebase_uid)
    if not user:
        return False
    _exec("DELETE FROM payments WHERE firebase_uid = ?", (firebase_uid,))
    _exec("DELETE FROM users WHERE firebase_uid = ?", (firebase_uid,))
    return True


def set_user_credits(firebase_uid: str, credits: int) -> bool:
    """Set credits to an exact value. Returns True if updated."""
    result = _exec(
        """
        UPDATE users SET report_credits = ?,
        updated_at = CURRENT_TIMESTAMP WHERE firebase_uid = ?
        """,
        (credits, firebase_uid),
    )
    return result.rowcount > 0


def get_user_count() -> Dict[str, int]:
    """Get total and paid user counts for dashboard stats."""
    total_rows = _query("SELECT COUNT(*) AS cnt FROM users")
    paid_rows = _query("SELECT COUNT(*) AS cnt FROM users WHERE has_paid = 1")
    credits_rows = _query("SELECT COUNT(*) AS cnt FROM users WHERE report_credits > 0")
    return {
        "total": total_rows[0]["cnt"] if total_rows else 0,
        "paid": paid_rows[0]["cnt"] if paid_rows else 0,
        "with_credits": credits_rows[0]["cnt"] if credits_rows else 0,
    }


# --- Dashboard & Aggregate Helpers ---

def insert_dashboard_report(mfd_firebase_uid: str, client_pan: str, client_name: str, expires_at: str, pdf_filename: str, status: str, snapshot_json: str, action_items_json: str) -> int:
    cur = _exec(
        """
        INSERT INTO dashboard_reports (mfd_firebase_uid, client_pan, client_name, expires_at, pdf_filename, status, snapshot_json, action_items_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (mfd_firebase_uid, client_pan, client_name, expires_at, pdf_filename, status, snapshot_json, action_items_json)
    )
    return cur.lastrowid

def insert_aggregate_action(id: str, mfd_firebase_uid: str, client_pan: str, report_id: int, item_id: str, dimension: str, urgency: str, value_type: str, value_num: float, final_status: str, report_generated_at: str):
    _exec(
        """
        INSERT INTO aggregate_actions (id, mfd_firebase_uid, client_pan, report_id, item_id, dimension, urgency, value_type, value_num, final_status, report_generated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (id, mfd_firebase_uid, client_pan, report_id, item_id, dimension, urgency, value_type, value_num, final_status, report_generated_at)
    )

def update_action_status(item_id: str, final_status: str) -> bool:
    result = _exec(
        "UPDATE aggregate_actions SET final_status = ?, actioned_at = CURRENT_TIMESTAMP WHERE item_id = ?",
        (final_status, item_id)
    )
    return result.rowcount > 0


def get_active_dashboard_report_by_pan(mfd_firebase_uid: str, client_pan: str) -> Optional[Dict[str, Any]]:
    """
    Return the most recent ACTIVE dashboard report for the given MFD + client PAN,
    or None if no active report exists.
    """
    rows = _query(
        """
        SELECT id, mfd_firebase_uid, client_pan, client_name, generated_at,
               expires_at, pdf_filename, status, snapshot_json, action_items_json
        FROM dashboard_reports
        WHERE mfd_firebase_uid = ? AND client_pan = ? AND status = 'ACTIVE'
        ORDER BY generated_at DESC
        LIMIT 1
        """,
        (mfd_firebase_uid, client_pan),
    )
    if not rows:
        return None
    return dict(rows[0])


def get_aggregate_action_statuses(report_id: int) -> dict:
    rows = _query("SELECT item_id, final_status FROM aggregate_actions WHERE report_id = ?", (report_id,))
    return {r["item_id"]: r["final_status"] for r in rows}


def mark_dashboard_report_status(report_id: int, status: str) -> bool:
    """Update a dashboard report's status (e.g., ACTIVE -> SUPERSEDED)."""
    result = _exec(
        "UPDATE dashboard_reports SET status = ? WHERE id = ?",
        (status, report_id),
    )
    return result.rowcount > 0


def mark_aggregate_actions_status_by_report(report_id: int, from_status: str, to_status: str) -> int:
    """
    Bulk update aggregate_actions belonging to a report whose final_status equals
    `from_status`, setting them to `to_status`. Returns number of rows updated.
    """
    result = _exec(
        "UPDATE aggregate_actions SET final_status = ? WHERE report_id = ? AND final_status = ?",
        (to_status, report_id, from_status),
    )
    return result.rowcount


def list_active_dashboard_reports(mfd_firebase_uid: str):
    """
    Return all ACTIVE dashboard reports for the given MFD, sorted by nearest
    expiry first (NULL expiry last).
    """
    rows = _query(
        """
        SELECT id, mfd_firebase_uid, client_pan, client_name, generated_at,
               expires_at, pdf_filename, status, snapshot_json, action_items_json
        FROM dashboard_reports
        WHERE mfd_firebase_uid = ? AND status = 'ACTIVE'
        ORDER BY
          CASE WHEN expires_at IS NULL OR expires_at = '' THEN 1 ELSE 0 END ASC,
          expires_at ASC,
          generated_at DESC
        """,
        (mfd_firebase_uid,),
    )
    return [dict(r) for r in rows]


def list_expired_active_dashboard_reports():
    """
    Return ACTIVE dashboard reports whose ``expires_at`` is in the past.
    Used by the expire_reports.py cron job to prune local PDF storage.
    """
    rows = _query(
        """
        SELECT id, mfd_firebase_uid, client_pan, client_name, generated_at,
               expires_at, pdf_filename, status
        FROM dashboard_reports
        WHERE status = 'ACTIVE'
          AND expires_at IS NOT NULL
          AND expires_at != ''
          AND expires_at < CURRENT_TIMESTAMP
        ORDER BY expires_at ASC
        """
    )
    return [dict(r) for r in rows]


def get_aggregate_metrics_overview(mfd_firebase_uid: str, missed_quarter_days: int = 90) -> Dict[str, Any]:
    """
    Compute dashboard overview metrics for an MFD:
      - total_opportunity: SUM(value_num) of PENDING + CONVERTED items
      - converted:         SUM(value_num) of CONVERTED items
      - pending:           SUM(value_num) of PENDING items
      - missed_quarter:    SUM(value_num) of PENDING items whose report was
                           generated more than `missed_quarter_days` days ago
      - action_count:      count of non-superseded items
    Also returns a per-dimension breakdown keyed by `dimension` from
    aggregate_actions.
    """
    overview_rows = _query(
        """
        SELECT
          COALESCE(SUM(CASE WHEN final_status IN ('PENDING','CONVERTED') THEN value_num ELSE 0 END), 0) AS total_opportunity,
          COALESCE(SUM(CASE WHEN final_status = 'CONVERTED' THEN value_num ELSE 0 END), 0) AS converted,
          COALESCE(SUM(CASE WHEN final_status = 'PENDING' THEN value_num ELSE 0 END), 0) AS pending,
          COALESCE(SUM(CASE WHEN final_status = 'PENDING'
                              AND report_generated_at IS NOT NULL
                              AND julianday('now') - julianday(report_generated_at) >= ?
                         THEN value_num ELSE 0 END), 0) AS missed_quarter,
          COUNT(*) AS action_count
        FROM aggregate_actions
        WHERE mfd_firebase_uid = ?
          AND final_status IN ('PENDING','CONVERTED')
        """,
        (missed_quarter_days, mfd_firebase_uid),
    )
    overview = dict(overview_rows[0]) if overview_rows else {
        "total_opportunity": 0,
        "converted": 0,
        "pending": 0,
        "missed_quarter": 0,
        "action_count": 0,
    }

    category_rows = _query(
        """
        SELECT
          COALESCE(NULLIF(TRIM(dimension), ''), 'uncategorized') AS dimension,
          COALESCE(SUM(CASE WHEN final_status IN ('PENDING','CONVERTED') THEN value_num ELSE 0 END), 0) AS total_opportunity,
          COALESCE(SUM(CASE WHEN final_status = 'CONVERTED' THEN value_num ELSE 0 END), 0) AS converted,
          COALESCE(SUM(CASE WHEN final_status = 'PENDING' THEN value_num ELSE 0 END), 0) AS pending,
          COALESCE(SUM(CASE WHEN final_status = 'PENDING'
                              AND report_generated_at IS NOT NULL
                              AND julianday('now') - julianday(report_generated_at) >= ?
                         THEN value_num ELSE 0 END), 0) AS missed_quarter,
          COUNT(*) AS action_count
        FROM aggregate_actions
        WHERE mfd_firebase_uid = ?
          AND final_status IN ('PENDING','CONVERTED')
        GROUP BY dimension
        ORDER BY total_opportunity DESC
        """,
        (missed_quarter_days, mfd_firebase_uid),
    )
    overview["categories"] = [dict(r) for r in category_rows]
    return overview


def update_aggregate_action_status_for_mfd(
    mfd_firebase_uid: str, item_id: str, final_status: str
) -> Optional[Dict[str, Any]]:
    """
    Update final_status + actioned_at on a single aggregate_action row, but
    ONLY if the row belongs to the given MFD. Returns the updated row, or None
    if no matching row was found.
    """
    result = _exec(
        """
        UPDATE aggregate_actions
        SET final_status = ?, actioned_at = CURRENT_TIMESTAMP
        WHERE item_id = ? AND mfd_firebase_uid = ?
        """,
        (final_status, item_id, mfd_firebase_uid),
    )
    if result.rowcount == 0:
        return None
    rows = _query(
        """
        SELECT id, mfd_firebase_uid, client_pan, report_id, item_id, dimension,
               urgency, value_type, value_num, final_status, report_generated_at,
               actioned_at, created_at
        FROM aggregate_actions
        WHERE item_id = ? AND mfd_firebase_uid = ?
        """,
        (item_id, mfd_firebase_uid),
    )
    return dict(rows[0]) if rows else None


def get_aggregate_metrics_for_period(
    mfd_firebase_uid: str, start_iso: str, end_iso: str
) -> Dict[str, Any]:
    """
    Aggregate-action metrics for a date range (inclusive of start, exclusive
    of end), scoped strictly to the MFD. The range is applied to
    report_generated_at for "identified" counts and to actioned_at for
    "converted" counts.
    """
    rows = _query(
        """
        SELECT
          COUNT(*) AS total_identified_count,
          COALESCE(SUM(value_num), 0) AS total_identified_value,
          COALESCE(SUM(CASE WHEN final_status = 'CONVERTED' THEN value_num ELSE 0 END), 0) AS converted_value,
          COALESCE(SUM(CASE WHEN final_status = 'CONVERTED' THEN 1 ELSE 0 END), 0) AS converted_count,
          COALESCE(SUM(CASE WHEN final_status = 'PENDING' THEN value_num ELSE 0 END), 0) AS pending_value
        FROM aggregate_actions
        WHERE mfd_firebase_uid = ?
          AND report_generated_at >= ?
          AND report_generated_at <  ?
        """,
        (mfd_firebase_uid, start_iso, end_iso),
    )
    summary = dict(rows[0]) if rows else {
        "total_identified_count": 0,
        "total_identified_value": 0,
        "converted_value": 0,
        "converted_count": 0,
        "pending_value": 0,
    }
    total_value = float(summary.get("total_identified_value") or 0)
    converted_value = float(summary.get("converted_value") or 0)
    summary["conversion_pct"] = (
        round((converted_value / total_value) * 100.0, 2) if total_value > 0 else 0.0
    )
    return summary


# Initialize schema on import
init_db()

# Auto-migrate existing paid users who have 0 credits (from old one-time payment model)
_migrated = migrate_existing_paid_users(credits=3)
if _migrated > 0:
    print(f"[db] Migrated {_migrated} existing paid user(s) → granted 3 report credits each")
