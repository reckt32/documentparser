import re
import json
from typing import Dict, Optional, Tuple

# Internal DB/index access
import db
from indexer import index_document

# --- Local numeric cleaner (kept consistent with app.py) ---
def _clean_to_float(value_str: Optional[str]):
    if not value_str or value_str == "N/A":
        return None
    if isinstance(value_str, (int, float)):
        return float(value_str)
    if not isinstance(value_str, str):
        return None
    try:
        import re as _re
        cleaned_str = _re.sub(r"[₹$,\s]", "", value_str).strip()
        cleaned_str = _re.sub(r"[()]", "", cleaned_str)
        if cleaned_str.endswith("-"):
            cleaned_str = "-" + cleaned_str[:-1]
        return float(cleaned_str) if cleaned_str else None
    except Exception:
        return None

# --- Regex parsers (narrow scope, deterministic) ---

def _parse_investment_snapshot(text: str) -> Optional[Dict]:
    if not text:
        return None

    def last_num(pattern: str):
        last = None
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            val = _clean_to_float(m.group(1))
            if val is not None:
                last = val
        return last

    fields = {}
    fields["investment"] = last_num(r"investment\s*\(A\)\s*([0-9,]+)")
    fields["switch_in"] = last_num(r"switch\s*in\s*\(B\)\s*([0-9,]+)")
    fields["switch_out"] = last_num(r"switch\s*out\s*\(C\)\s*([0-9,]+)")
    fields["redemption"] = last_num(r"redemption\s*\(D\)\s*([0-9,]+)")
    fields["div_payout_fd_interest"] = last_num(r"div\.\s*payout/FD\s*interest\s*\(E\)\s*([0-9,]+)") or last_num(r"div(?:idend)?\s*payout.*?\(E\)\s*([0-9,]+)")
    fields["net_investment"] = last_num(r"net\s*investment\s*\(F[^\)]*\)\s*([0-9,]+)")
    fields["current_value"] = last_num(r"current\s*value\s*\(G\)\s*([0-9,]+)")
    fields["net_gain"] = last_num(r"net\s*gain\s*\(H[^\)]*\)\s*([0-9,]+)")

    xirr_last = None
    for m in re.finditer(r"\bXIRR\b\s*([0-9]+(?:\.[0-9]+)?)\s*%", text, flags=re.IGNORECASE):
        try:
            xirr_last = float(m.group(1))
        except Exception:
            continue
    if xirr_last is not None:
        fields["xirr_percent"] = xirr_last

    filtered = {k: v for k, v in fields.items() if v is not None}
    return filtered or None

def _parse_statement_summary(text: str) -> Optional[Dict]:
    if not text:
        return None

    def last_num_from_any(patterns):
        last = None
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                val = _clean_to_float(m.group(1))
                if val is not None:
                    last = val
        return last

    opening = last_num_from_any([
        r"opening\s+balance[^\d\-]*\(?([\d,]+\-?)\)?",
        r"opening\s+bal\.?[^\d\-]*\(?([\d,]+\-?)\)?",
    ])
    closing = last_num_from_any([
        r"closing\s+balance[^\d\-]*\(?([\d,]+\-?)\)?",
        r"closing\s+bal\.?[^\d\-]*\(?([\d,]+\-?)\)?",
    ])
    inflow = last_num_from_any([
        r"(?:total\s+)?credits?\s*[:\-]?\s*\(?([\d,]+\-?)\)?",
        r"total\s+deposit[s]?\s*[:\-]?\s*\(?([\d,]+\-?)\)?",
        r"total\s+inflow[s]?\s*[:\-]?\s*\(?([\d,]+\-?)\)?",
    ])
    outflow = last_num_from_any([
        r"(?:total\s+)?debits?\s*[:\-]?\s*\(?([\d,]+\-?)\)?",
        r"total\s+withdrawal[s]?\s*[:\-]?\s*\(?([\d,]+\-?)\)?",
        r"total\s+outflow[s]?\s*[:\-]?\s*\(?([\d,]+\-?)\)?",
    ])

    summary = {}
    if opening is not None:
        summary["opening_balance"] = opening
    if closing is not None:
        summary["closing_balance"] = closing
    if inflow is not None:
        summary["total_inflows"] = inflow
    if outflow is not None:
        summary["total_outflows"] = outflow
    return summary or None

# --- Section picking helpers ---

def _pick_final_section(document_id: int, section_type: str) -> Optional[Dict]:
    rows = db._query(
        """
        SELECT id, page_number, y_bottom, text
        FROM sections
        WHERE document_id=? AND section_type=?
        ORDER BY page_number DESC, y_bottom DESC
        LIMIT 1
        """,
        (document_id, section_type),
    )
    if not rows:
        return None
    r = rows[0]
    return {"id": r["id"], "page_number": r["page_number"], "y_bottom": r["y_bottom"], "text": r["text"]}

def _store_metrics(document_id: int, metrics: Dict[str, float], source_section_id: Optional[int]):
    if not metrics:
        return
    keys = list(metrics.keys())
    db.delete_metrics_for_doc_keys(document_id, keys)
    for k, v in metrics.items():
        db.insert_metric(document_id, k, v if isinstance(v, (int, float)) else _clean_to_float(v), source_section_id)

def _canon_header(h: Optional[str]) -> str:
    if not h:
        return ""
    s = re.sub(r"\s+", " ", str(h)).strip().lower()
    aliases = {
        "purchase": "purchase",
        "switch in": "switch_in",
        "switchin": "switch_in",
        "div reinvest": "div_reinvest",
        "dividend reinvest": "div_reinvest",
        "redemption": "redemption",
        "switch out": "switch_out",
        "switchout": "switch_out",
        "current value": "current_value",
        "dividend payout": "div_payout",
        "unrealized gain": "unrealized_gain",
        "realized gain": "realized_gain",
        "abs. return": "abs_return",
        "abs return": "abs_return",
        "xirr": "xirr",
    }
    for k, v in aliases.items():
        if s.startswith(k):
            return v
    return s.replace(" ", "_")

def _parse_portfolio_grand_total(document_id: int) -> Optional[Dict]:
    """
    Scan indexed tables (last page first) for a 'Grand Total' row and extract portfolio totals.
    Returns keys like:
      - total_purchase, total_redemption, total_current_value
      - unrealized_gain, realized_gain, xirr_percent
      - net_investment, net_gain
      - table_page
    """
    rows = db._query(
        "SELECT page_number, header_json, rows_json FROM tables WHERE document_id=? ORDER BY page_number DESC, id DESC",
        (document_id,),
    )
    for r in rows:
        page_no = r["page_number"]
        try:
            header = json.loads(r["header_json"] or "[]")
            body = json.loads(r["rows_json"] or "[]")
        except Exception:
            continue
        if not header or not body:
            continue
        keys = [_canon_header(h) for h in header]
        for row in body:
            cells = [str(c) if c is not None else "" for c in row]
            joined = " ".join(cells).lower()
            if ("grand total" in joined) or re.search(r"\btotal\s*:\b", joined):
                def num_at(key: str) -> Optional[float]:
                    if key in keys:
                        idx = keys.index(key)
                        return _clean_to_float(cells[idx])
                    return None
                purchase = num_at("purchase")
                switch_in = num_at("switch_in") or 0.0
                redemption = num_at("redemption")
                switch_out = num_at("switch_out") or 0.0
                div_reinvest = num_at("div_reinvest") or 0.0
                div_payout = num_at("div_payout") or 0.0
                current_value = num_at("current_value")
                unrealized_gain = num_at("unrealized_gain")
                realized_gain = num_at("realized_gain")
                xirr = num_at("xirr")

                result: Dict[str, float] = {}
                if purchase is not None:
                    result["total_purchase"] = purchase
                if redemption is not None:
                    result["total_redemption"] = redemption
                if current_value is not None:
                    result["total_current_value"] = current_value
                if unrealized_gain is not None:
                    result["unrealized_gain"] = unrealized_gain
                if realized_gain is not None:
                    result["realized_gain"] = realized_gain
                if xirr is not None:
                    result["xirr_percent"] = xirr

                # Compute net investment and net gain if possible
                if purchase is not None and redemption is not None:
                    net_inv = (purchase + switch_in + div_reinvest) - (redemption + switch_out + div_payout)
                    result["net_investment"] = net_inv
                    if current_value is not None:
                        result["net_gain"] = current_value - net_inv

                result["table_page"] = page_no
                return result
    return None

def _parse_statement_summary_from_tables(document_id: int) -> Optional[Dict]:
    """
    Parse bank-style statement summary from indexed tables on the last pages.
    Looks for a row containing 'Statement Summary' (with/without space/colon),
    a header containing 'OpeningBalance ... Debits Credits ClosingBal',
    and then a numeric row with values.
    Returns: {opening_balance, closing_balance, total_inflows, total_outflows} or None
    """
    rows = db._query(
        "SELECT page_number, header_json, rows_json FROM tables WHERE document_id=? ORDER BY page_number DESC, id DESC",
        (document_id,),
    )
    for r in rows:
        page_no = r["page_number"]
        try:
            body = json.loads(r["rows_json"] or "[]")
        except Exception:
            continue
        if not body:
            continue

        # Flatten rows to strings per cell
        flat_rows = [[(str(c) if c is not None else "").strip() for c in row] for row in body]

        # Find an index where a cell contains 'statement summary' marker
        marker_idx = None
        for i, row in enumerate(flat_rows):
            joined = " ".join(row).lower()
            if "statementsummary" in joined.replace(" ", "") or "statement summary" in joined:
                marker_idx = i
                break
        if marker_idx is None:
            # Some PDFs put only the header row without the marker; look for a header-style row
            for i, row in enumerate(flat_rows):
                joined = " ".join(row).lower()
                if ("openingbalance" in joined.replace(" ", "")) and ("closingbal" in joined.replace(" ", "")):
                    marker_idx = i - 1 if i > 0 else i
                    break
        if marker_idx is None:
            continue

        # Search within next few rows for a numeric summary line
        for j in range(marker_idx + 1, min(marker_idx + 6, len(flat_rows))):
            textline = " ".join(flat_rows[j])
            tokens = textline.split()
            nums = []
            for token in tokens:
                # numeric-like tokens including parentheses/negatives/commas
                if re.search(r"^[\(\)\-]?[0-9][0-9,]*([.][0-9]+)?\-?$", token):
                    nums.append(token)
            # Expect at least 6 numbers: Opening, DrCount, CrCount, Debits, Credits, Closing
            if len(nums) >= 6:
                def n(idx):
                    return _clean_to_float(nums[idx]) if idx < len(nums) else None
                opening = n(0)
                debits = n(3)
                credits = n(4)
                closing = n(5)
                out: Dict[str, float] = {}
                if opening is not None:
                    out["opening_balance"] = opening
                if closing is not None:
                    out["closing_balance"] = closing
                if credits is not None:
                    out["total_inflows"] = credits
                if debits is not None:
                    out["total_outflows"] = debits
                if out:
                    out["table_page"] = page_no
                    return out
    return None

# --- Public API ---

def extract_and_store_from_indexed(document_id: int) -> Dict:
    """
    Select last snapshot/summary sections and extract metrics with provenance.
    Also scan tables for a final 'Grand Total' portfolio summary.
    Returns:
      {
        "investment_snapshot": {...} | None,
        "account_summary": {...} | None,
        "portfolio_summary": {...} | None,
        "provenance": {
            "snapshot_section_id": id or None,
            "statement_section_id": id or None
        }
      }
    """
    result = {"investment_snapshot": None, "account_summary": None, "portfolio_summary": None, "provenance": {}}

    # Snapshot
    snap = _pick_final_section(document_id, "snapshot")
    if snap:
        snap_data = _parse_investment_snapshot(snap["text"])
        if snap_data:
            _store_metrics(document_id, snap_data, snap["id"])
            result["investment_snapshot"] = snap_data
            result["provenance"]["snapshot_section_id"] = snap["id"]

    # Statement summary
    stmt = _pick_final_section(document_id, "statement_summary")
    if stmt:
        stmt_data = _parse_statement_summary(stmt["text"])
        if stmt_data:
            _store_metrics(document_id, stmt_data, stmt["id"])
            result["account_summary"] = stmt_data
            result["provenance"]["statement_section_id"] = stmt["id"]

    # If still missing or incomplete, try table-driven bank statement summary
    if (not result["account_summary"]) or any(
        k not in result["account_summary"] for k in ("opening_balance", "closing_balance", "total_inflows", "total_outflows")
    ):
        try:
            tbl_sum = _parse_statement_summary_from_tables(document_id)
            if tbl_sum:
                _store_metrics(document_id, tbl_sum, None)
                if result["account_summary"]:
                    result["account_summary"].update({k: v for k, v in tbl_sum.items() if v is not None})
                else:
                    result["account_summary"] = tbl_sum
        except Exception:
            pass

    # Portfolio 'Grand Total' from tables (last page first)
    try:
        ptotal = _parse_portfolio_grand_total(document_id)
        if ptotal:
            # store numeric metrics under 'portfolio_*' keys
            numeric_metrics = {f"portfolio_{k}": v for k, v in ptotal.items() if isinstance(v, (int, float, float))}
            _store_metrics(document_id, numeric_metrics, None)
            result["portfolio_summary"] = ptotal
    except Exception:
        pass

    return result

def index_and_extract(pdf_bytes: bytes, filename: str = "upload.pdf") -> Tuple[str, int, Dict]:
    """
    Convenience: index PDF (idempotent by sha in documents) and extract summaries.
    Returns (sha, document_id, summaries_dict)
    """
    sha, doc_id = index_document(pdf_bytes, filename=filename)
    summaries = extract_and_store_from_indexed(doc_id)
    return sha, doc_id, summaries
