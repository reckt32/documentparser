"""Bank statement extraction: structured transactions + LLM hybrid."""

import json
import logging
import re
from datetime import datetime

import pdfplumber

from backend.config import DETERMINISTIC_RECURRING, client
from backend.extraction.metadata import (
    _add_extraction_metadata,
    _create_extraction_meta,
    _merge_llm_data,
)
from backend.extraction.mutual_fund import _extract_investment_snapshot
from backend.extraction.pdf_text import _select_text_for_llm
from backend.extraction.transaction_analytics import (
    compute_average_monthly_balance,
    detect_recurring_transactions,
    merge_recurring,
    months_from_period,
    statement_months,
)
from backend.utils import clean_and_convert_to_float

logger = logging.getLogger(__name__)


# --- Helpers for Dates and Header Normalization ---
def _normalize_header(h):
    if not h:
        return ""
    h = re.sub(r"\s+", " ", str(h)).strip().lower()
    # direct canonicalizations
    direct = {
        "txn date": "date",
        "transaction date": "date",
        "value date": "value_date",
        "post date": "date",
        "description": "description",
        "narration": "description",
        "particulars": "description",
        "ref no": "reference",
        "reference no": "reference",
        "cheque no": "reference",
        "chq/ ref no": "reference",
        "instrument no": "reference",
        "debit": "debit",
        "withdrawal": "debit",
        "dr": "debit",
        "credit": "credit",
        "deposit": "credit",
        "cr": "credit",
        "amount": "amount",
        "balance": "balance",
        "closing balance": "balance",
    }
    if h in direct:
        return direct[h]
    # fuzzy rules
    if "date" in h and "value" in h:
        return "value_date"
    if "date" in h:
        return "date"
    if any(k in h for k in ["narration", "description", "particular"]):
        return "description"
    if any(k in h for k in ["ref", "cheque", "chq", "instrument", "utr"]):
        return "reference"
    if any(k in h for k in ["debit", "withdraw"] ) or "(dr" in h or h.endswith(" dr") or h == "dr":
        return "debit"
    if any(k in h for k in ["credit", "deposit"]) or "(cr" in h or h.endswith(" cr") or h == "cr":
        return "credit"
    if "amount" in h:
        return "amount"
    if "balance" in h:
        return "balance"
    return h


def _parse_date(s):
    if not s:
        return None
    s = str(s).strip()
    # try multiple common formats
    fmts = [
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",
        "%d %b %Y", "%d-%b-%Y", "%d/%b/%Y",
        "%d %b %y", "%d-%b-%y", "%d/%b/%y",
        "%Y-%m-%d",
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f).date().isoformat()
        except Exception:
            continue
    # sometimes date is like 01JAN2024 or 01-Jan-24
    try:
        return datetime.strptime(s.replace(" ", "-"), "%d-%b-%Y").date().isoformat()
    except Exception:
        pass
    return None


def _to_float_or_none(x):
    v = clean_and_convert_to_float(x)
    return None if v == "N/A" else v


def _extract_bank_statement_end_summary(text):
    """
    Extracts the final 'Statement Summary' style numbers from the raw text,
    preferring the last occurrence in the document.
    Returns dict with any of: opening_balance, closing_balance, total_inflows, total_outflows.
    """
    if not text:
        return {}
    summary = {}

    def last_value(patterns):
        last = None
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                candidate = m.group(1)
                val = clean_and_convert_to_float(candidate)
                if val != "N/A":
                    last = val
        return last

    summary["opening_balance"] = last_value([
        r"opening\s+balance[^\d\-]*\(?([\d,]+\-?)\)?",
        r"opening\s+bal\.?[^\d\-]*\(?([\d,]+\-?)\)?",
    ])
    summary["closing_balance"] = last_value([
        r"closing\s+balance[^\d\-]*\(?([\d,]+\-?)\)?",
        r"closing\s+bal\.?[^\d\-]*\(?([\d,]+\-?)\)?",
    ])
    inflow = last_value([
        r"(?:total\s+)?credits?\s*[:\-]?\s*\(?([\d,]+\-?)\)?",
        r"total\s+deposit[s]?\s*[:\-]?\s*\(?([\d,]+\-?)\)?",
        r"total\s+inflow[s]?\s*[:\-]?\s*\(?([\d,]+\-?)\)?",
    ])
    outflow = last_value([
        r"(?:total\s+)?debits?\s*[:\-]?\s*\(?([\d,]+\-?)\)?",
        r"total\s+withdrawal[s]?\s*[:\-]?\s*\(?([\d,]+\-?)\)?",
        r"total\s+outflow[s]?\s*[:\-]?\s*\(?([\d,]+\-?)\)?",
    ])
    if inflow is not None:
        summary["total_inflows"] = inflow
    if outflow is not None:
        summary["total_outflows"] = outflow

    return {k: v for k, v in summary.items() if v is not None}


# --- Bank Statement Structured Extraction ---
def extract_bank_statement_transactions(file_like):
    """Extract transactions as structured rows from a bank statement PDF using pdfplumber.
    Returns dict with keys: transactions: [..], totals: {...}
    Heuristics: detect tables with date/description and debit/credit/amount columns,
    handle wrapped rows (blank date implies continuation), and numeric cleaning.
    """
    transactions = []
    try:
        with pdfplumber.open(file_like) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables() or []
                if not tables:
                    # Borderless statements defeat the default (line-based)
                    # strategy; retry using text alignment. Junk pseudo-tables
                    # are filtered by the header checks below.
                    try:
                        tables = page.extract_tables({
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                        }) or []
                    except Exception:
                        tables = []
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    header = [_normalize_header(h) for h in table[0]]
                    cols = {name: idx for idx, name in enumerate(header) if name}
                    # must have a date+description and at least one of debit/credit/amount
                    if not any(h in cols for h in ("date", "value_date")) or "description" not in cols:
                        continue
                    has_debit = "debit" in cols
                    has_credit = "credit" in cols
                    has_amount = "amount" in cols
                    # Likely a transaction table only if amount-like present
                    if not (has_debit or has_credit or has_amount):
                        continue

                    last_txn = None
                    for row in table[1:]:
                        if row is None:
                            continue
                        # guard against ragged rows
                        row = [ (str(c) if c is not None else "").replace("\n"," ").strip() for c in row ]
                        # pad to header length
                        if len(row) < len(header):
                            row += [""] * (len(header) - len(row))

                        date_str = row[cols.get("date", cols.get("value_date"))] if any(k in cols for k in ("date","value_date")) else ""
                        description = row[cols["description"]] if "description" in cols else ""
                        ref = row[cols["reference"]] if "reference" in cols else ""
                        debit = _to_float_or_none(row[cols["debit"]]) if has_debit else None
                        credit = _to_float_or_none(row[cols["credit"]]) if has_credit else None
                        amount = _to_float_or_none(row[cols["amount"]]) if has_amount else None
                        balance = _to_float_or_none(row[cols["balance"]]) if "balance" in cols else None

                        # Handle wrapped/continuation lines: no date but more description
                        if (not date_str) and last_txn and description:
                            last_txn["description"] = (last_txn["description"] + " " + description).strip()
                            # merge reference if provided
                            if ref:
                                last_txn["reference"] = (last_txn.get("reference") or "") + ("; " if last_txn.get("reference") else "") + ref
                            # fill missing numeric cells if present only here
                            if debit is not None and last_txn.get("debit") is None:
                                last_txn["debit"] = debit
                            if credit is not None and last_txn.get("credit") is None:
                                last_txn["credit"] = credit
                            if amount is not None and last_txn.get("amount") is None:
                                last_txn["amount"] = amount
                            if balance is not None and last_txn.get("balance") is None:
                                last_txn["balance"] = balance
                            continue

                        iso_date = _parse_date(date_str)
                        # If the row clearly doesn't look like a txn, skip
                        if not iso_date and not description:
                            continue

                        # derive amount/type if only debit/credit available
                        txn_type = None
                        if amount is None:
                            if debit is not None and (credit is None or debit > 0):
                                amount = debit
                                txn_type = "debit"
                            elif credit is not None and (debit is None or credit > 0):
                                amount = credit
                                txn_type = "credit"
                        else:
                            # If both debit/credit present, prefer sign via non-null
                            if debit not in (None, 0) and (credit in (None, 0)):
                                txn_type = "debit"
                            elif credit not in (None, 0) and (debit in (None, 0)):
                                txn_type = "credit"

                        txn = {
                            "date": iso_date or date_str or None,
                            "description": description or None,
                            "reference": ref or None,
                            "debit": debit,
                            "credit": credit,
                            "amount": amount,
                            "type": txn_type,
                            "balance": balance,
                        }
                        transactions.append(txn)
                        last_txn = txn
    except Exception as e:
        logger.error(f"Error extracting bank transactions: {e}")

    # post-process: clean None-only rows
    clean_txns = []
    for t in transactions:
        # valid if has date+desc or amount
        if (t.get("date") or t.get("description")) and (t.get("amount") is not None or t.get("debit") is not None or t.get("credit") is not None):
            clean_txns.append(t)

    # compute totals
    total_debit = sum([v for v in [(tx.get("debit") or (tx.get("amount") if tx.get("type") == "debit" else None)) for tx in clean_txns] if isinstance(v,(int,float))])
    total_credit = sum([v for v in [(tx.get("credit") or (tx.get("amount") if tx.get("type") == "credit" else None)) for tx in clean_txns] if isinstance(v,(int,float))])
    opening_balance = None
    closing_balance = None
    # try derive from balances if present
    balances = [tx.get("balance") for tx in clean_txns if tx.get("balance") is not None]
    if balances:
        opening_balance = balances[0]
        closing_balance = balances[-1]

    return {
        "transactions": clean_txns,
        "totals": {
            "total_inflows": total_credit,
            "total_outflows": total_debit,
            "opening_balance": opening_balance,
            "closing_balance": closing_balance,
            "count": len(clean_txns),
        }
    }


def extract_bank_statement_hybrid(text, transactions_payload=None, save_json_path=None):
    data = {}
    
    patterns = {
        "account_number": [
            r"(?i)Account\s*(?:No\.?|Number)[\s:\-]*([Xx\*\d]{8,20})",
            r"(?i)A/C\s*(?:No\.?|Number)[\s:\-]*([Xx\*\d]{8,20})",
            r"(?i)Account[\s:\-]*([Xx\*\d]{8,20})"
        ],
        "ifsc": [
            r"\b([A-Z]{4}0[A-Z0-9]{6})\b"
        ],
        "statement_period": [
            r"(?i)Statement\s+(?:Period|From)[\s:\-]*(\d{1,2}[-/]\w{3}[-/]\d{2,4})\s*(?:to|-)\s*(\d{1,2}[-/]\w{3}[-/]\d{2,4})",
            r"(?i)Period[\s:\-]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s*(?:to|-)\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"(?i)From[\s:\-]*(\d{1,2}[-/]\w{3}[-/]\d{2,4})\s*To[\s:\-]*(\d{1,2}[-/]\w{3}[-/]\d{2,4})"
        ]
    }

    for key, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                if key == "statement_period" and match.lastindex >= 2:
                    data[key] = f"{match.group(1)} to {match.group(2)}"
                else:
                    data[key] = match.group(1).strip() if match.group(1) else "N/A"
                break
        if key not in data:
            data[key] = "N/A"

    # Prepare a concise prompt relying on structured JSON if provided
    tx_json_str = None
    if transactions_payload and isinstance(transactions_payload, dict):
        try:
            # limit size to keep within token budget
            tx_copy = {
                "totals": transactions_payload.get("totals", {}),
                # include only essential fields from first 2000 txns to reduce size
                "transactions": [
                    {
                        "date": t.get("date"),
                        "description": t.get("description"),
                        "amount": t.get("amount"),
                        "type": t.get("type"),
                        "debit": t.get("debit"),
                        "credit": t.get("credit"),
                    }
                    for t in transactions_payload.get("transactions", [])[:2000]
                ],
            }
            tx_json_str = json.dumps(tx_copy)
        except Exception:
            tx_json_str = None

    llm_prompt = f"""
    You are given a bank statement. Prefer the provided structured JSON of transactions if available, otherwise infer from raw text.
    Extract the following information in JSON format:

    1. account_summary:
        - account_holder_name: Full name of the account holder (look for "Account Holder", "Customer Name", "Name", etc.)
        - opening_balance
        - closing_balance
        - total_inflows
        - total_outflows
        - average_monthly_balance

    2. recurring_credits: Array of objects with description, amount, frequency, dates
    
    3. recurring_debits: Array of objects with:
        - description: Transaction description
        - amount: Numeric amount (no currency symbols)
        - frequency: Monthly/Quarterly/Yearly/Ad-hoc
        - dates: Array of dates when this recurring debit occurred
        - is_emi: Boolean - true if this is an EMI/loan payment, false otherwise
        
        **EMI IDENTIFICATION RULES (IMPORTANT):**
        Mark is_emi=true for transactions that match ANY of these patterns:
        - Contains "EMI", "LOAN", "LN", "MORTGAGE", "INSTALMENT", "INSTALLMENT", "REPAYMENT"
        - Contains bank-specific loan codes: "ELM" (ICICI EMI), "HDFC LN", "SBI LN", "AXIS LN"
        - Contains "NACH" (National Automated Clearing House - often used for auto-debit EMIs/loans)
        - Contains "SI/" or "STANDING INSTRUCTION" for loan payments
        - Contains "ECS" (Electronic Clearing Service) for recurring loan debits
        - Contains "AUTO DEBIT" or "AUTODEBIT" for loan/EMI payments
        - Contains "HOME LOAN", "CAR LOAN", "PERSONAL LOAN", "VEHICLE LOAN", "HOUSING LOAN", "EDUCATION LOAN"
        - Same amount debited on or around the same date each month (±3 days) - likely an EMI
        
        Common EMI transaction patterns in Indian banks:
        - ICICI: "BIL/INFT/ELM...", "NACH/..."
        - HDFC: "HDFC LN...", "NACH/HDFC..."
        - SBI: "EMI DED/...", "NACH/SBI..."
        - Axis: "AXIS LN...", "NACH/AXIS..."
        
    4. high_value_transactions: Array with date, description, type, amount (threshold: 100000)
    5. bounce_penalty_charges: Array with date, description, amount

    Notes:
    - Use the structured JSON transactions as the source of truth when present.
    - Use exact numeric values; do not include currency symbols.
    - Frequency can be Monthly/Quarterly/Yearly/Ad-hoc.
    - For recurring_debits, ALWAYS include the is_emi boolean field.

    Structured Transactions JSON (optional):
    {tx_json_str if tx_json_str else "<none>"}

    Raw Bank Statement Text (head and tail of document):
    {_select_text_for_llm(text, 8000)}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": "You are a financial analyst expert at extracting structured data from Indian bank statements. All monetary values are in Indian Rupees (INR). Return valid JSON only."},
                {"role": "user", "content": llm_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.05
        )
        llm_data = json.loads(response.choices[0].message.content)
        # Add confidence metadata for LLM-extracted fields
        for llm_field in llm_data.keys():
            _add_extraction_metadata(data, {
                llm_field: _create_extraction_meta(llm_field, llm_data[llm_field], "llm")
            })
        # Regex-extracted account identifiers are authoritative; LLM fills gaps only.
        _merge_llm_data(data, llm_data, protected=(
            "account_number", "ifsc", "statement_period",
        ))
    except Exception as e:
        logger.error(f"LLM extraction error for bank statement: {str(e)}")
        data['extraction_error'] = str(e)
        _add_extraction_metadata(data, {
            "_llm_extraction": {"confidence": 0.0, "source": "llm", "needs_review": True, "reason": str(e)}
        })

    # Deterministic recurring detection from structured transactions.
    # Deterministic groups win on overlap; LLM entries are kept for descriptions
    # the structured path didn't catch (failed table extraction, single-occurrence
    # EMIs). If detection finds nothing, the LLM output is left untouched.
    if DETERMINISTIC_RECURRING and transactions_payload and isinstance(transactions_payload, dict):
        try:
            txns = transactions_payload.get("transactions") or []
            det = detect_recurring_transactions(txns)
            for side in ("recurring_debits", "recurring_credits"):
                det_entries = det.get(side) or []
                if not det_entries:
                    continue
                llm_entries = data.get(side) or []
                merged = merge_recurring(det_entries, llm_entries)
                logger.info(
                    f"Recurring {side}: deterministic={len(det_entries)} llm={len(llm_entries)} merged={len(merged)}"
                )
                data[side] = merged
                _add_extraction_metadata(data, {
                    side: _create_extraction_meta(side, merged, "regex")
                })
            amb = compute_average_monthly_balance(txns)
            if amb is not None:
                acct = data.get("account_summary") or {}
                acct["average_monthly_balance"] = amb
                data["account_summary"] = acct
        except Exception as e:
            logger.warning(f"Deterministic recurring detection failed; keeping LLM output: {e}")

    # Statement coverage in months: the declared period wins, transaction-date
    # span is the fallback. Downstream prefill uses this to normalize monthly
    # figures instead of assuming every statement covers 12 months.
    try:
        months = months_from_period(data.get("statement_period"))
        if months is None and transactions_payload and isinstance(transactions_payload, dict):
            months = statement_months(transactions_payload.get("transactions") or [])
        if months:
            acct = data.get("account_summary") or {}
            acct["statement_months"] = months
            data["account_summary"] = acct
    except Exception as e:
        logger.warning(f"Statement period detection failed: {e}")

    # Merge in any precomputed totals from structured parsing when LLM omitted
    if transactions_payload and isinstance(transactions_payload, dict):
        totals = transactions_payload.get("totals", {})
        account_summary = data.get("account_summary", {}) or {}
        for k in ["opening_balance","closing_balance","total_inflows","total_outflows"]:
            if account_summary.get(k) in (None, "N/A") and totals.get(k) is not None:
                account_summary[k] = totals.get(k)
        data["account_summary"] = account_summary

    # Prefer the explicit end-of-document statement summary if detected
    try:
        bank_summary = _extract_bank_statement_end_summary(text)
        if bank_summary:
            account_summary = data.get("account_summary", {}) or {}
            account_summary.update({k: v for k, v in bank_summary.items() if v is not None})
            data["account_summary"] = account_summary
    except Exception as _:
        pass

    # Add Investment Snapshot if present (common in portfolio summaries)
    try:
        inv_snapshot = _extract_investment_snapshot(text)
        if inv_snapshot:
            data["investment_snapshot"] = inv_snapshot
    except Exception as _:
        pass

    # attach a reference to saved JSON artifact if available
    if save_json_path:
        data["transactions_json_path"] = save_json_path
        
    return data
