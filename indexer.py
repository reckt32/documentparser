import io
import re
import json
import hashlib
from typing import List, Tuple, Optional

import pdfplumber

from db import (
    upsert_document,
    insert_section,
    insert_table,
)

# ---- Utilities ----

def _sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def _classify_section(text: str) -> str:
    t = (text or "").lower()
    # High-confidence headings/keywords
    if "investment snapshot" in t or re.search(r"\binvestment\s*\(a\)", t):
        return "snapshot"
    if "sip summary" in t:
        return "sip_summary"
    if "statement summary" in t:
        return "statement_summary"
    # Totals related final summary cues
    if "grand total" in t:
        return "statement_summary"
    if any(k in t for k in ["opening balance", "closing balance", "total inflows", "total outflows", "total credits", "total debits"]):
        return "statement_summary"
    return "other"

def _group_words_into_blocks(words: List[dict]) -> List[dict]:
    """
    Coarse grouping:
    - First group by line (close in 'top')
    - Then aggregate consecutive lines into blocks if vertical gap small
    Returns list of blocks with: text, top, bottom
    """
    if not words:
        return []
    # sort by top then x0
    words_sorted = sorted(words, key=lambda w: (w.get("top", 0.0), w.get("x0", 0.0)))
    # group into lines
    lines = []
    cur_line: List[dict] = []
    last_top: Optional[float] = None
    line_tol = 3.5  # tolerance for same line
    for w in words_sorted:
        top = float(w.get("top", 0.0))
        if last_top is None or abs(top - last_top) <= line_tol:
            cur_line.append(w)
            last_top = top if last_top is None else (last_top + top) / 2.0
        else:
            if cur_line:
                lines.append(cur_line)
            cur_line = [w]
            last_top = top
    if cur_line:
        lines.append(cur_line)

    # convert lines to text with bbox
    line_objs = []
    for ln in lines:
        ln_sorted = sorted(ln, key=lambda w: w.get("x0", 0.0))
        text = " ".join(w.get("text", "") for w in ln_sorted).strip()
        if not text:
            continue
        top = min(float(w.get("top", 0.0)) for w in ln_sorted)
        bottom = max(float(w.get("bottom", top)) for w in ln_sorted)
        line_objs.append({"text": text, "top": top, "bottom": bottom})

    # aggregate lines into blocks by vertical proximity
    blocks = []
    cur_block_lines: List[dict] = []
    last_bottom: Optional[float] = None
    gap_tol = 12.0  # gap threshold to split blocks
    for lo in line_objs:
        if last_bottom is None or (lo["top"] - last_bottom) <= gap_tol:
            cur_block_lines.append(lo)
            last_bottom = lo["bottom"]
        else:
            # flush
            if cur_block_lines:
                text = "\n".join(l["text"] for l in cur_block_lines).strip()
                top = min(l["top"] for l in cur_block_lines)
                bottom = max(l["bottom"] for l in cur_block_lines)
                blocks.append({"text": text, "top": top, "bottom": bottom})
            cur_block_lines = [lo]
            last_bottom = lo["bottom"]
    if cur_block_lines:
        text = "\n".join(l["text"] for l in cur_block_lines).strip()
        top = min(l["top"] for l in cur_block_lines)
        bottom = max(l["bottom"] for l in cur_block_lines)
        blocks.append({"text": text, "top": top, "bottom": bottom})

    return blocks

# ---- Public API ----

def index_document(pdf_bytes: bytes, filename: str = "upload.pdf") -> Tuple[str, int]:
    """
    Index the PDF into SQLite:
      - documents row (sha, filename, page_count)
      - sections (page_number, section_type, y_top, y_bottom, text)
      - tables (optional)
    Returns (sha256, document_id)
    """
    sha = _sha256(pdf_bytes)
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page_count = len(pdf.pages)
        document_id = upsert_document(sha, filename, page_count)

        # Idempotency: skip if sections already exist for this document
        try:
            from db import _query as _db_query
            rows = _db_query("SELECT 1 FROM sections WHERE document_id=? LIMIT 1", (document_id,))
            if rows:
                return sha, document_id
        except Exception:
            pass

        for i, page in enumerate(pdf.pages, start=1):
            # words with positions
            words = page.extract_words(keep_blank_chars=False, use_text_flow=True) or []
            blocks = _group_words_into_blocks(words)
            for blk in blocks:
                text = blk.get("text", "")
                y_top = float(blk.get("top") or 0.0)
                y_bottom = float(blk.get("bottom") or y_top)
                stype = _classify_section(text)
                # Basic heading detection: use first line as a heading if it looks like a title
                heading = None
                first_line = text.split("\n", 1)[0]
                if re.search(r"(investment snapshot|statement summary|grand total|sip summary)", first_line, flags=re.I):
                    heading = first_line.strip()
                insert_section(
                    document_id=document_id,
                    page_number=i,
                    heading=heading,
                    section_type=stype,
                    y_top=y_top,
                    y_bottom=y_bottom,
                    text=text,
                )

            # store tables (optional)
            tables = page.extract_tables() or []
            for tbl in tables:
                if not tbl or len(tbl) == 0:
                    continue
                header = [str(h).replace("\n", " ") if h is not None else "" for h in tbl[0]]
                rows = []
                for row in tbl[1:]:
                    rows.append([str(c).replace("\n", " ") if c is not None else "" for c in row])
                insert_table(document_id=document_id, section_id=None, page_number=i, header=header, rows=rows)

    return sha, document_id
