import os
import json
import sqlite3
import threading
from typing import Any, Dict, Iterable, Optional

# SQLite file lives under backend/output/index.db
_BASE_DIR = os.path.dirname(__file__)
_DB_DIR = os.path.join(_BASE_DIR, "output")
os.makedirs(_DB_DIR, exist_ok=True)
DB_PATH = os.path.join(_DB_DIR, "index.db")

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
    return {
        "id": row["id"],
        "firebase_uid": row["firebase_uid"],
        "email": row["email"],
        "display_name": row["display_name"],
        "has_paid": bool(row["has_paid"]),
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


def mark_user_as_paid(firebase_uid: str, payment_id: str, order_id: str = None, amount_paise: int = None) -> bool:
    """Mark a user as paid after successful payment. Returns True if updated."""
    result = _exec(
        """
        UPDATE users SET has_paid = 1, payment_id = ?, payment_order_id = ?,
        payment_amount_paise = ?, payment_date = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP WHERE firebase_uid = ?
        """,
        (payment_id, order_id, amount_paise, firebase_uid)
    )
    return result.rowcount > 0


def check_user_payment_status(firebase_uid: str) -> bool:
    """Check if user has paid. Returns False if user doesn't exist."""
    rows = _query("SELECT has_paid FROM users WHERE firebase_uid = ?", (firebase_uid,))
    if not rows:
        return False
    return bool(rows[0]["has_paid"])


# Initialize schema on import
init_db()

