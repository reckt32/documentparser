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


# Initialize schema on import
init_db()
