from flask import Flask, request, jsonify, url_for, send_from_directory, send_file
import fitz  # PyMuPDF
import pdfplumber
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Frame, PageTemplate, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
from reportlab.pdfgen import canvas
from extractors import index_and_extract, extract_and_store_from_indexed
from llm_sections import run_report_sections
from db import (
    list_sections,
    list_metrics,
    get_document_by_sha,
    create_questionnaire,
    update_questionnaire_status,
    save_personal_info,
    save_family_info,
    save_goals,
    save_risk_profile,
    save_insurance,
    save_estate,
    save_lifestyle,
    get_questionnaire,
    get_latest_questionnaire_for_user,
    link_questionnaire_upload,
    list_questionnaire_uploads,
    insert_metric,
    delete_metrics_for_doc_keys,
    update_questionnaire_upload_metadata,
)
# --- Initialization ---
load_dotenv()

app = Flask(__name__)
CORS(app)
# Respect reverse proxy headers on Render (scheme/host) for correct external URLs
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
app.config['PREFERRED_URL_SCHEME'] = 'https'
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=openai_api_key)

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Storage behavior:
# STORE_REPORTS = "disk" (default) -> write to OUTPUT_DIR and serve via /download/<file>
# STORE_REPORTS = "memory" -> do not persist; generate to disk, read once, delete, and offer one-time memory download via /download-temp/<token>
STORE_REPORTS = os.getenv("STORE_REPORTS", "disk").lower()
# SAVE_TX_JSON = "true"/"false" -> whether to persist extracted bank tx JSON artifacts
SAVE_TX_JSON = os.getenv("SAVE_TX_JSON", "false").lower() == "true"
# Simple in-memory store for ephemeral downloads
TEMP_REPORTS = {}

# Logo path - uses app directory for production compatibility
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")

def _format_indian_amount(amount: float) -> str:
    """Format amount in Indian currency style (lakhs/crores) for compact display in PDFs."""
    if amount is None or amount == 0:
        return "0"
    amount = abs(amount)
    if amount >= 1_00_00_000:  # 1 crore
        return f"{amount / 1_00_00_000:.1f} Cr"
    elif amount >= 1_00_000:  # 1 lakh
        return f"{amount / 1_00_000:.1f} L"
    elif amount >= 1000:
        return f"{amount / 1000:.1f} K"
    else:
        return str(int(round(amount)))

def _register_temp_download(data: bytes, filename: str, mimetype: str = "application/pdf") -> str:
    token = os.urandom(16).hex()
    TEMP_REPORTS[token] = (filename, data, mimetype)
    return token

def _persist_metrics_for_doc(document_id: int, data: dict):
    """
    Persist key numeric insights from extracted data into metrics for aggregation/prefill.
    Keys stored:
      - opening_balance, closing_balance, total_inflows, total_outflows (bank)
      - gross_total_income, taxable_income, total_tax_paid (ITR)
      - sum_assured_or_insured (insurance)
      - portfolio_equity/debt/gold/realEstate/insuranceLinked/cash (CAS allocation)
    """
    if not isinstance(data, dict):
        return

    def n(v):
        try:
            if isinstance(v, (int, float)):
                return float(v)
            if v in (None, "", "N/A"):
                return None
            cv = clean_and_convert_to_float(str(v))
            return cv if cv != "N/A" else None
        except Exception:
            return None

    to_store = {}

    # Bank account summary
    acct = data.get("account_summary") or {}
    for k in ["opening_balance", "closing_balance", "total_inflows", "total_outflows"]:
        v = acct.get(k)
        nv = n(v)
        if nv is not None:
            to_store[k] = nv

    # ITR numbers (top-level or under tax_computation)
    itr_top = {k: data.get(k) for k in ["gross_total_income", "taxable_income", "total_tax_paid"]}
    tax_comp = data.get("tax_computation") or {}
    for k in ["gross_total_income", "taxable_income", "total_tax_paid"]:
        v = itr_top.get(k, None)
        if v in (None, "", "N/A"):
            v = tax_comp.get(k)
        nv = n(v)
        if nv is not None:
            to_store[k] = nv

    # Insurance
    ins_sum = data.get("sum_assured_or_insured")
    ins_nv = n(ins_sum)
    if ins_nv is not None:
        to_store["sum_assured_or_insured"] = ins_nv

    # CAS allocation
    alloc = data.get("asset_allocation") or {}
    mapping = {
        "equity_percentage": "portfolio_equity",
        "debt_percentage": "portfolio_debt",
        "gold_percentage": "portfolio_gold",
        "real_estate_percentage": "portfolio_realEstate",
        "insurance_linked_percentage": "portfolio_insuranceLinked",
        "cash_percentage": "portfolio_cash",
        # Support alternate keys if present
        "equity": "portfolio_equity",
        "debt": "portfolio_debt",
        "gold": "portfolio_gold",
        "realEstate": "portfolio_realEstate",
        "insuranceLinked": "portfolio_insuranceLinked",
        "cash": "portfolio_cash",
    }
    for src, dst in mapping.items():
        if src in alloc:
            nv = n(alloc.get(src))
            if nv is not None:
                to_store[dst] = nv

    if to_store:
        try:
            delete_metrics_for_doc_keys(document_id, list(to_store.keys()))
        except Exception:
            pass
        for k, v in to_store.items():
            try:
                insert_metric(document_id, k, v, None)
            except Exception:
                continue

# --- Utility Function for Cleaning Numbers ---
def clean_and_convert_to_float(value_str):
    """Cleans currency symbols, commas, and parentheses, then converts to float."""
    if not value_str or value_str == "N/A":
        return "N/A"
    if isinstance(value_str, (int, float)):
        return float(value_str)
    if not isinstance(value_str, str):
        return "N/A"
    try:
        # Remove currency symbols, commas, and whitespace
        cleaned_str = re.sub(r'[₹$,\s]', '', value_str).strip()
        # Handle negative numbers represented with parentheses
        cleaned_str = re.sub(r'[\(\)]', '', cleaned_str)
        if cleaned_str.endswith('-'):
            cleaned_str = '-' + cleaned_str[:-1]
        return float(cleaned_str) if cleaned_str else "N/A"
    except (ValueError, TypeError):
        return "N/A"

# PDF text sanitization helper to avoid unsupported glyphs
def sanitize_pdf_text(s):
    try:
        if s is None:
            return ""
        t = str(s)
        # Normalize common unicode symbols to ASCII
        t = (t.replace("•", "-")
               .replace("–", "-")
               .replace("—", "-")
               .replace("×", "x")
               .replace("₹", "Rs.")
               .replace("“", '"')
               .replace("”", '"')
               .replace("’", "'")
               .replace("‘", "'")
               .replace("≈", "~"))
        # Remove non-printable/non-ASCII chars (preserve tab/newline/carriage return + printable ASCII 0x20-0x7E)
        t = re.sub(r"[^\t\n\r -~]", "", t)
        # Collapse repeated dashes from earlier bullet normalization
        t = re.sub(r"-{2,}", "-", t)
        # Soft-break very long unbroken tokens to allow wrapping in PDF tables/paras
        t = re.sub(r"([A-Za-z0-9:/\.\-_]{30})(?=[A-Za-z0-9:/\.\-_])", r"\1 ", t)
        # Collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t
    except Exception:
        return str(s) if s is not None else ""
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

def _extract_investment_snapshot(text):
    """
    Extracts 'Investment Snapshot' style summaries often present in portfolio PDFs.
    Returns dict with keys like investment, switch_in, switch_out, redemption, div_payout_fd_interest,
    net_investment, current_value, net_gain, xirr_percent. Only includes fields found.
    """
    if not text:
        return None

    def find_last(label_pattern):
        last = None
        for m in re.finditer(label_pattern, text, flags=re.IGNORECASE):
            val = m.group(1)
            num = clean_and_convert_to_float(val)
            if num != "N/A":
                last = num
        return last

    fields = {}
    fields["investment"] = find_last(r"investment\s*\(A\)\s*([0-9,]+)")
    fields["switch_in"] = find_last(r"switch\s*in\s*\(B\)\s*([0-9,]+)")
    fields["switch_out"] = find_last(r"switch\s*out\s*\(C\)\s*([0-9,]+)")
    fields["redemption"] = find_last(r"redemption\s*\(D\)\s*([0-9,]+)")
    fields["div_payout_fd_interest"] = find_last(r"div\.\s*payout/FD\s*interest\s*\(E\)\s*([0-9,]+)")
    if fields.get("div_payout_fd_interest") is None:
        fields["div_payout_fd_interest"] = find_last(r"div(?:idend)?\s*payout.*?\(E\)\s*([0-9,]+)")
    fields["net_investment"] = find_last(r"net\s*investment\s*\(F[^\)]*\)\s*([0-9,]+)")
    fields["current_value"] = find_last(r"current\s*value\s*\(G\)\s*([0-9,]+)")
    fields["net_gain"] = find_last(r"net\s*gain\s*\(H[^\)]*\)\s*([0-9,]+)")

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
        print(f"Error extracting bank transactions: {e}")

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

# --- Hybrid Extraction Functions (Refactored) ---

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
    3. recurring_debits: Array of objects with description, amount, frequency, dates
    4. high_value_transactions: Array with date, description, type, amount (threshold: 100000)
    5. bounce_penalty_charges: Array with date, description, amount

    Notes:
    - Use the structured JSON transactions as the source of truth when present.
    - Use exact numeric values; do not include currency symbols.
    - Frequency can be Monthly/Quarterly/Yearly/Ad-hoc.

    Structured Transactions JSON (optional):
    {tx_json_str if tx_json_str else "<none>"}

    Raw Bank Statement Text (truncated):
    {text[:8000]}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial analyst expert at extracting structured data from Indian bank statements. All monetary values are in Indian Rupees (INR). Return valid JSON only."},
                {"role": "user", "content": llm_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        llm_data = json.loads(response.choices[0].message.content)
        data.update(llm_data)
    except Exception as e:
        print(f"LLM Error: {str(e)}")
        data['extraction_error'] = str(e)
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

def extract_itr_hybrid(text):
    data = {}

    patterns = {
        "assessee_name": [
            r"(?i)(?:Name|Assessee)(?:\s+of\s+(?:the\s+)?(?:Assessee|Tax\s*Payer))?[\s:\-]*([A-Z][A-Za-z\s\.]+?)(?:\n|PAN)",
            r"(?i)Name\s+as\s+per\s+PAN[\s:\-]*([A-Z][A-Za-z\s\.]+)",
            r"(?i)Full\s+Name[\s:\-]*([A-Z][A-Za-z\s\.]+)"
        ],
        "date_of_birth": [
            r"(?i)(?:Date\s+of\s+Birth|DOB|D\.O\.B)[\s:\-]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"(?i)Birth\s+Date[\s:\-]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
        ],
        "assessment_year": [
            r"(?i)Assessment\s*Year[\s:\-]*(\d{4}[\s\-]+\d{2,4})",
            r"(?i)A\.?Y\.?[\s:\-]*(\d{4}[\s\-]+\d{2,4})"
        ],
        "filing_date": [
            r"(?i)Date\s*of\s*Filing[\s:\-]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"(?i)Filed\s*on[\s:\-]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"(?i)e-Filing\s*Date[\s:\-]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
        ],
        "pan": [
            r"(?i)PAN[\s:\-]*([A-Z]{5}\d{4}[A-Z])",
            r"\b([A-Z]{5}\d{4}[A-Z])\b"
        ],
        "gross_total_income": [
            r"(?i)Gross\s+Total\s+Income[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)",
            r"(?i)Total\s+Gross\s+Income[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)"
        ],
        "taxable_income": [
            r"(?i)(?:Total\s+)?Taxable\s+Income[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)",
            r"(?i)Net\s+Taxable\s+Income[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)"
        ],
        "total_tax_paid": [
            r"(?i)(?:Total\s+)?Tax\s+(?:Paid|Payable)[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)",
            r"(?i)Tax\s+Amount[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)"
        ],
        "refund_status": [
            r"(?i)Refund[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)",
            r"(?i)Tax\s+Payable[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)"
        ]
    }
    
    for key, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                if key in ["gross_total_income", "taxable_income", "total_tax_paid", "refund_status"]:
                    data[key] = clean_and_convert_to_float(match.group(1))
                else:
                    data[key] = match.group(1).strip() if match.group(1) else "N/A"
                break
        if key not in data:
            data[key] = "N/A"

    deduction_patterns = [
        (r"(?i)(?:Section\s*)?80C[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)", "80C"),
        (r"(?i)(?:Section\s*)?80D[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)", "80D"),
        (r"(?i)(?:Section\s*)?80E[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)", "80E"),
        (r"(?i)(?:Section\s*)?80G[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)", "80G"),
        (r"(?i)NPS.*?(?:80CCD|Deduction)[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)", "80CCD (NPS)")
    ]
    
    deductions = []
    for pattern, section in deduction_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount = clean_and_convert_to_float(match.group(1))
            if amount != "N/A" and amount > 0:
                deductions.append({"section": section, "amount": amount})
    
    if deductions:
        data["deductions_claimed"] = deductions

    llm_prompt = f"""
    Analyze this ITR document and extract the following information in JSON format:
    
    1. income_sources: Object with keys and their amounts:
        - salary: Income from salary
        - business_income: Income from business/profession
        - capital_gains: Capital gains (short term + long term)
        - house_property: Income from house property
        - other_sources: Income from other sources
    2. deductions_claimed: Array of objects with:
        - section: Section name (e.g., "80C", "80D", "80CCD")
        - amount: Deduction amount
    3. carry_forward_losses: Object with types of losses carried forward:
        - business_loss: Amount
        - capital_loss_short: Short term capital loss
        - capital_loss_long: Long term capital loss
        - house_property_loss: House property loss
    4. assets_and_liabilities: Object with:
        - immovable_assets: Total value of immovable assets
        - movable_assets: Total value of movable assets (bank deposits, shares, etc.)
        - total_liabilities: Total liabilities
    5. tax_computation: Object with:
        - gross_total_income: Total income before deductions
        - total_deductions: Sum of all deductions
        - taxable_income: Income after deductions
        - tax_before_rebate: Tax calculated
        - rebate_87a: Rebate under 87A if applicable
        - total_tax_paid: Final tax paid
        - tds_collected: TDS amount
        - advance_tax: Advance tax paid
        - self_assessment_tax: Self assessment tax paid
    
    All amounts should be numbers without currency symbols. Use 0 for not found/not applicable.
    
    ITR Document Text:
    {text[:20000]}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a tax expert analyzing Indian ITR documents. All monetary values are in Indian Rupees (INR). Extract all financial data accurately. Return valid JSON only."},
                {"role": "user", "content": llm_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        llm_data = json.loads(response.choices[0].message.content)
        
        # Merge LLM data intelligently
        for key in ["income_sources", "carry_forward_losses", "assets_and_liabilities", "tax_computation"]:
            if key in llm_data and llm_data[key]:
                data[key] = llm_data[key]
        
        if "deductions_claimed" in llm_data and llm_data["deductions_claimed"]:
            existing_deductions = data.get("deductions_claimed", [])
            existing_sections = {d["section"] for d in existing_deductions}
            for d in llm_data["deductions_claimed"]:
                if d.get("section") not in existing_sections:
                    existing_deductions.append(d)
            data["deductions_claimed"] = existing_deductions
            
    except Exception as e:
        print(f"LLM Error: {str(e)}")
        data['extraction_error'] = str(e)
        
    return data

def extract_insurance_hybrid(text):
    data = {}
    
    is_life_insurance = bool(re.search(r"(?i)(life\s+insurance|term\s+plan|endowment|ULIP|whole\s+life)", text))
    is_health_insurance = bool(re.search(r"(?i)(health\s+insurance|mediclaim|medical\s+insurance|hospitali[sz]ation)", text))
    is_general_insurance = bool(re.search(r"(?i)(motor\s+insurance|vehicle\s+insurance|property\s+insurance|home\s+insurance|fire\s+insurance)", text))
    
    if is_life_insurance:
        data["insurance_type"] = "Life Insurance"
    elif is_health_insurance:
        data["insurance_type"] = "Health Insurance"
    elif is_general_insurance:
        data["insurance_type"] = "General Insurance"
    else:
        data["insurance_type"] = "Unknown"

    patterns = {
        "policy_number": [
            r"(?i)Policy\s*(?:No\.?|Number)[\s:\-]*([A-Z0-9\-/]{6,25})",
            r"(?i)Policy[\s:\-]*([A-Z0-9\-/]{6,25})"
        ],
        "insurer_name": [
            r"(?i)(?:Insurer|Company|Insurance\s+Company)[\s:\-]*([A-Za-z\s&]+?)(?:\n|Ltd|Limited|Insurance)",
            r"([A-Za-z\s&]+?)\s+(?:Life|General|Health)\s+Insurance"
        ],
        "policy_holder": [
            r"(?i)(?:Policy\s*Holder|Insured|Proposer)(?:\s*Name)?[\s:\-]*([A-Za-z\s\.]+)",
            r"(?i)Name\s+of\s+(?:the\s+)?Insured[\s:\-]*([A-Za-z\s\.]+)"
        ],
        "policy_term": [
            r"(?i)Policy\s*(?:Term|Period|Duration)[\s:\-]*(\d+)\s*(?:Years?|Yrs?)",
            r"(?i)Term[\s:\-]*(\d+)\s*(?:Years?|Yrs?)"
        ],
        "policy_start_date": [
            r"(?i)(?:Policy\s*)?(?:Start|Commencement|Inception)\s*Date[\s:\-]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"(?i)Date\s*of\s*Commencement[\s:\-]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"(?i)From[\s:\-]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
        ],
        "policy_end_date": [
            r"(?i)(?:Policy\s*)?(?:End|Maturity|Expiry)\s*Date[\s:\-]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"(?i)Valid\s*(?:Till|Until|Upto)[\s:\-]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"(?i)To[\s:\-]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
        ],
        "nominee": [
            r"(?i)Nominee(?:\s*Name)?[\s:\-]*([A-Za-z\s\.]+?)(?:\n|Relationship)",
            r"(?i)Name\s+of\s+(?:the\s+)?Nominee[\s:\-]*([A-Za-z\s\.]+)"
        ],
        "date_of_birth": [
            r"(?i)(?:Date\s+of\s+Birth|DOB|D\.O\.B)[\s:\-]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"(?i)Birth\s+Date[\s:\-]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"(?i)DOB\s+of\s+(?:Life\s+)?Insured[\s:\-]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
        ]
    }

    for key, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                data[key] = match.group(1).strip() if match.group(1) else "N/A"
                break
        if key not in data:
            data[key] = "N/A"

    sum_patterns = (
        [
            r"(?i)Sum\s*(?:Assured|Insured)[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)",
            r"(?i)Life\s*Cover[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)",
            r"(?i)Death\s*Benefit[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)"
        ] if is_life_insurance else [
            r"(?i)Sum\s*(?:Insured|Assured)[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)",
            r"(?i)Cover(?:age)?\s*Amount[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)",
            r"(?i)Policy\s*Coverage[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)"
        ]
    )
    
    for pattern in sum_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data["sum_assured_or_insured"] = clean_and_convert_to_float(match.group(1))
            break
    if "sum_assured_or_insured" not in data:
        data["sum_assured_or_insured"] = "N/A"

    premium_patterns = [
        r"(?i)(?:Annual\s*)?Premium(?:\s*Amount)?[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)",
        r"(?i)Premium\s*Payable[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)",
        r"(?i)Total\s*Premium[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+)"
    ]
    
    for pattern in premium_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data["premium_amount"] = clean_and_convert_to_float(match.group(1))
            break
    if "premium_amount" not in data:
        data["premium_amount"] = "N/A"

    freq_match = re.search(r"(?i)Premium\s*(?:Payment\s*)?Frequency[\s:\-]*(Annual|Semi-Annual|Quarterly|Monthly)", text)
    if freq_match:
        data["premium_frequency"] = freq_match.group(1)
    else:
        if re.search(r"(?i)Annual\s*Premium", text):
            data["premium_frequency"] = "Annual"
        elif re.search(r"(?i)Monthly\s*Premium", text):
            data["premium_frequency"] = "Monthly"
        else:
            data["premium_frequency"] = "N/A"

    llm_prompt = f"""
    Analyze this insurance document and extract detailed information based on the insurance type.
    
    Insurance Type Detected: {data.get('insurance_type', 'Unknown')}
    
    For LIFE INSURANCE, extract:
    1. policy_details: Object with:
        - policy_type: Term/Endowment/ULIP/Whole Life/Money Back
        - death_benefit: Death benefit amount
        - maturity_benefit: Maturity benefit amount (if applicable)
        - surrender_value: Current surrender value (if mentioned)
    2. riders: Array of rider names (Critical Illness, Accidental Death, Waiver of Premium, etc.)
    3. premium_details: Object with:
        - base_premium: Base premium amount
        - rider_premium: Additional premium for riders
        - gst: GST amount
        - total_premium: Total premium including all charges
        - payment_mode: Single/Regular/Limited Pay
    
    For HEALTH INSURANCE, extract:
    1. coverage_details: Object with:
        - coverage_type: Individual/Family Floater
        - base_sum_insured: Base coverage amount
        - restoration_benefit: If sum insured restoration is available
        - room_rent_limit: Daily room rent limit or percentage
        - co_payment: Co-payment percentage if any
    2. covered_members: Array of objects with:
        - name: Member name
        - relationship: Self/Spouse/Child/Parent
    3. exclusions: Array of major exclusions
    4. waiting_periods: Object with:
        - initial_waiting: Initial waiting period
        - specific_diseases: Waiting for specific diseases
        - pre_existing: Pre-existing disease waiting period
    
    For GENERAL INSURANCE, extract:
    1. asset_details: Object with:
        - asset_type: Vehicle/Property/Other
        - asset_description: Make/model for vehicle, property address for home
        - asset_value: Insured declared value (IDV) or property value
    2. coverage_details: Object with:
        - own_damage_cover: For motor insurance
        - third_party_cover: For motor insurance
        - deductibles: Deductible amounts
    3. add_ons: Array of add-on covers purchased
    
    Return empty objects/arrays if not found. All amounts should be numbers.
    
    Insurance Document Text:
    {text[:20000]}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an insurance expert analyzing Indian insurance policies. All monetary values are in Indian Rupees (INR). Extract all details accurately based on the insurance type. Return valid JSON only."},
                {"role": "user", "content": llm_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        llm_data = json.loads(response.choices[0].message.content)
        data.update(llm_data)
    except Exception as e:
        print(f"LLM Error: {str(e)}")
        data['extraction_error'] = str(e)
        
    return data

def extract_mutual_fund_cas_hybrid(text):
    data = {}
    
    patterns = {
        "investor_name": [
            r"(?i)(?:Investor|Unit\s*Holder|First\s*Holder)\s*(?:Name)?[\s:\-]*([A-Z][A-Z\s\.]+?)(?:\n|PAN)",
            r"(?i)Name\s*of\s*(?:the\s*)?(?:First\s*)?(?:Unit)?holder[\s:\-]*([A-Z][A-Z\s\.]+)"
        ],
        "pan": [
            r"(?i)PAN[\s:\-]*([A-Z]{5}\d{4}[A-Z])",
            r"\b([A-Z]{5}\d{4}[A-Z])\b"
        ],
        "email": [
            r"(?i)E-?mail[\s:\-]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
        ]
    }

    for key, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                data[key] = match.group(1).strip() if match.group(1) else "N/A"
                break
        if key not in data:
            data[key] = "N/A"

    folio_pattern = r"(?i)Folio\s*(?:No\.?|Number)?[\s:\-]*([A-Z0-9/]+)"
    data['folio_numbers'] = list(set(re.findall(folio_pattern, text, re.IGNORECASE)))
    if not data['folio_numbers']:
        data['folio_numbers'] = []

    llm_prompt = f"""
    Analyze this Mutual Fund CAS (Consolidated Account Statement) and extract:
    
    1. holdings: Array of fund holdings with each object containing:
        - amc_name: Asset Management Company name
        - scheme_name: Full scheme name
        - scheme_category: Equity/Debt/Hybrid/Liquid and sub-category
        - folio_number: Folio number for this holding
        - units_held: Total units currently held
        - current_nav: Latest NAV
        - current_value: Current market value
        - investment_cost: Total amount invested
        - unrealized_gain_loss: Profit/Loss amount
    
    2. transaction_summary: Object with:
        - total_purchase_amount: Total amount invested across all funds
        - total_redemption_amount: Total amount redeemed
        - total_current_value: Sum of current values of all holdings
        - total_unrealized_gain: Total unrealized profit/loss
    
    3. sip_details: Array of active SIPs with:
        - scheme_name: Scheme name
        - sip_amount: Monthly SIP amount
        - sip_date: SIP deduction date
        - frequency: Monthly/Quarterly/etc.
    
    4. nominee_details: Array of nominees with:
        - name: Nominee name
        - relationship: Relationship with investor
        - allocation_percentage: Percentage allocation
        - folio_numbers: Folios where this nominee is registered
        
    5. asset_allocation: Object with:
        - equity_percentage: Percentage in equity funds
        - debt_percentage: Percentage in debt funds
        - hybrid_percentage: Percentage in hybrid funds
        - total_aum: Total assets under management
    
    All amounts should be numbers without currency symbols. Return empty arrays/objects if not found.
    
    Mutual Fund CAS Text:
    {text[:20000]}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a mutual fund analysis expert for Indian markets. All monetary values are in Indian Rupees (INR). Extract structured data into a valid JSON format."},
                {"role": "user", "content": llm_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        llm_data = json.loads(response.choices[0].message.content)
        data.update(llm_data)
    except Exception as e:
        print(f"LLM Error: {str(e)}")
        data['extraction_error'] = str(e)
    
    # Also extract explicit 'Investment Snapshot' style summary (prefer last occurrence in doc)
    try:
        inv_snapshot = _extract_investment_snapshot(text)
        if inv_snapshot:
            data["investment_snapshot"] = inv_snapshot
    except Exception as _:
        pass

    return data

def extract_structured_text_with_tables(file_stream):
    """
    Extracts structured text and tables from a PDF file stream using pdfplumber.
    Tables are converted to a markdown-like format.
    """
    print("Starting structured text extraction...")
    text_content = []
    try:
        # Use the file stream directly with pdfplumber
        with pdfplumber.open(file_stream) as pdf:
            print(f"PDF has {len(pdf.pages)} pages.")
            for page_num, page in enumerate(pdf.pages):
                print(f"Processing page {page_num + 1}...")
                text_content.append(f"\n--- Page {page_num + 1} ---\n")
                
                # Extract text first, which is the default behavior of page.extract_text()
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)

                # Extract tables and format them as markdown
                tables = page.extract_tables()
                if tables:
                    text_content.append("\n\n--- Extracted Tables ---\n")
                    for i, table in enumerate(tables):
                        if not table or len(table) == 0: continue
                        
                        text_content.append(f"\n**Table {i+1} on Page {page_num + 1}:**\n")
                        
                        # Filter out None values from header and rows to prevent errors
                        header = [str(h).replace('\n', ' ') if h is not None else '' for h in table[0]]
                        table_md = f"| {' | '.join(header)} |\n"
                        table_md += f"| {' | '.join(['---'] * len(header))} |\n"
                        for row in table[1:]:
                            cleaned_row = [str(cell).replace('\n', ' ') if cell is not None else '' for cell in row]
                            table_md += f"| {' | '.join(cleaned_row)} |\n"
                        text_content.append(table_md)
                        text_content.append("\n")
    
    
        print("Finished structured text extraction.")
        return "".join(text_content)
    except Exception as e:
        print(f"Error in extract_structured_text_with_tables: {e}")
        raise

# --- Financial Health Analysis Helpers ---

def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        v = clean_and_convert_to_float(s)
        return v if v != "N/A" else default
    except Exception:
        return default

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _band_midpoint(band: str):
    if not band:
        return None
    b = str(band).strip().lower()
    if b.startswith("<") or b.startswith("<"):
        return 5.0
    if "10-20" in b:
        return 15.0
    if "20-30" in b:
        return 25.0
    if b.startswith(">") or b.startswith(">"):
        return 35.0
    return None

def resolve_savings_percent(savings_percent, savings_band):
    sp = None
    if savings_percent is not None:
        try:
            sp = float(savings_percent)
        except Exception:
            sp = None
    if sp is None:
        mp = _band_midpoint(savings_band)
        sp = mp if mp is not None else 0.0
    return _clamp(sp, 0.0, 100.0)

def compute_risk_profile(age, tolerance, horizon):
    a = _safe_float(age, 0.0)
    # age points
    if a < 35:
        age_pts = 2
    elif a <= 50:
        age_pts = 1
    else:
        age_pts = 0
    # tolerance points
    tol_map = {"low": 0, "medium": 1, "med": 1, "high": 2}
    tol_pts = tol_map.get(str(tolerance).strip().lower(), 0)
    # horizon points
    hor_map = {"short": 0, "medium": 1, "med": 1, "long": 2}
    hor_pts = hor_map.get(str(horizon).strip().lower(), 0)
    score = age_pts + tol_pts + hor_pts  # 0-6
    if score <= 2:
        return "Conservative", score
    if score <= 4:
        return "Balanced", score
    return "Aggressive", score

def compute_surplus_band(savings_percent):
    sp = _safe_float(savings_percent, 0.0)
    # Spec:
    # Savings < 20% -> Low
    # Savings 20-30% -> Adequate
    # Savings > 30% -> Strong
    if sp < 20.0:
        return "Low"
    if sp <= 30.0:
        return "Adequate"
    return "Strong"

def compute_insurance_gap(annual_income, life_cover):
    ai = _safe_float(annual_income, 0.0)
    lc = _safe_float(life_cover, 0.0)
    required = 10.0 * ai
    status = "Adequate" if lc >= required else "Underinsured"
    return status, required

def compute_debt_stress(monthly_emi, annual_income):
    emi = _safe_float(monthly_emi, 0.0)
    ai = _safe_float(annual_income, 0.0)
    monthly_income = ai / 12.0 if ai > 0 else 0.0
    ratio_pct = (emi / monthly_income * 100.0) if monthly_income > 0 else 0.0
    if ratio_pct > 40.0:
        return "Stressed", ratio_pct
    if ratio_pct >= 20.0:
        return "Moderate", ratio_pct
    return "Healthy", ratio_pct

def compute_liquidity(monthly_expenses, emergency_fund_amount):
    me = _safe_float(monthly_expenses, 0.0)
    ef = _safe_float(emergency_fund_amount, 0.0)
    months = (ef / me) if me > 0 else 0.0
    status = "Adequate" if months >= 6.0 else "Insufficient"
    return status, months

def compute_retirement_corpus(age, monthly_income, desired_monthly_pension=None, withdrawal_rate=0.07, retirement_age=60):
    """
    Calculate retirement corpus needed.
    
    Returns dict with:
      - years_to_retirement: retirement_age - age
      - standard_corpus: (retirement_age - age) * monthly_income * 12 (human life value method)
      - pension_annual: desired_monthly_pension * 12 (if pension provided)
      - pension_corpus: (pension * 12) / withdrawal_rate - ideal corpus for perpetual pension
    """
    age_val = _safe_float(age, 30)
    mi = _safe_float(monthly_income, 0.0)
    years_to_retirement = max(0, retirement_age - age_val)
    
    # Standard retirement corpus formula: (retirement_age - age) * monthly_income * 12
    standard_corpus = years_to_retirement * mi * 12
    
    result = {
        "years_to_retirement": years_to_retirement,
        "retirement_age": retirement_age,
        "standard_corpus": round(standard_corpus, 0),
    }
    
    # If desired monthly pension is provided, calculate pension-based corpus
    # Using perpetuity formula: Corpus = (Annual Pension) / Withdrawal Rate
    # This gives the corpus needed to generate the annual pension indefinitely at the given rate
    if desired_monthly_pension is not None:
        pension = _safe_float(desired_monthly_pension, 0.0)
        if pension > 0:
            annual_pension = pension * 12
            # Ideal retirement corpus = annual pension / 7% (perpetuity formula)
            pension_corpus = annual_pension / withdrawal_rate
            
            result["desired_monthly_pension"] = round(pension, 0)
            result["pension_annual"] = round(annual_pension, 0)
            result["pension_corpus"] = round(pension_corpus, 0)
            result["withdrawal_rate_percent"] = withdrawal_rate * 100
    
    return result


def compute_term_insurance_need(age, monthly_income, retirement_age=60):
    """
    Calculate term insurance requirement using Human Life Value method.
    Formula: (retirement_age - age) * monthly_income * 12
    
    Returns the required term cover amount.
    """
    age_val = _safe_float(age, 30)
    mi = _safe_float(monthly_income, 0.0)
    years_to_retirement = max(0, retirement_age - age_val)
    
    required_cover = years_to_retirement * mi * 12
    return round(required_cover, 0)

def compute_ihs(savings_percent, current_products, allocation):
    # Savings (0-40)
    sp = _clamp(_safe_float(savings_percent, 0.0), 0.0, 40.0)
    savings_score = sp  # linear to 40

    # Products (0-30)
    current = [str(x).strip().lower() for x in (current_products or [])]
    def has_any(keys):
        return any(k in current for k in keys)

    products_score = 0.0
    if has_any(["mf", "mutual funds", "mutual fund", "stocks", "ulip"]):
        products_score += 12.0
    if has_any(["fd", "fixed deposit", "ppf"]):
        products_score += 8.0
    if has_any(["gold"]):
        products_score += 4.0
    if has_any(["re", "real estate", "realestate"]):
        products_score += 4.0
    if has_any(["other", "crypto"]):
        products_score += 2.0
    products_score = _clamp(products_score, 0.0, 30.0)

    # Diversification (0-30) using HHI on six buckets
    alloc = allocation or {}
    keys = ["equity", "debt", "gold", "realEstate", "insuranceLinked", "cash"]
    total = sum([_safe_float(alloc.get(k, 0.0), 0.0) for k in keys])
    if total <= 0:
        diversification_score = 0.0
    else:
        # normalize to fractions
        fracs = [(_safe_float(alloc.get(k, 0.0), 0.0) / total) for k in keys]
        hhi = sum([f * f for f in fracs])
        diversification_score = _clamp((1.0 - hhi) * 30.0, 0.0, 30.0)

    score = _clamp(savings_score + products_score + diversification_score, 0.0, 100.0)
    if score < 40.0:
        band = "Poor"
    elif score < 60.0:
        band = "Average"
    elif score < 80.0:
        band = "Good"
    else:
        band = "Excellent"

    return {
        "score": round(score, 1),
        "band": band,
        "breakdown": {
            "savings": round(savings_score, 1),
            "products": round(products_score, 1),
            "diversification": round(diversification_score, 1),
        }
    }

# --- Advanced Risk Logic Engine (Phases 1–3) ---

_RISK_CATEGORIES = [
    "Ultra Conservative",
    "Conservative",
    "Moderate",
    "Growth",
    "Aggressive",
    "Very Aggressive",
]

_RISK_EQUITY_BANDS = {
    "Ultra Conservative": (0, 25),
    "Conservative": (25, 40),
    "Moderate": (40, 55),
    "Growth": (55, 70),
    "Aggressive": (70, 85),
    "Very Aggressive": (85, 95),
}

def _tenure_limit_category(years):
    if years is None:
        return "Moderate"
    try:
        y = float(years)
    except Exception:
        return "Moderate"
    if y < 3:
        return "Conservative"
    if 3 <= y < 5:
        return "Moderate"
    if 5 <= y < 7:
        return "Growth"
    if 7 <= y < 10:
        return "Aggressive"
    return "Very Aggressive"

def _score_to_category(score):
    if score < 1.8:
        return "Ultra Conservative"
    if score < 2.4:
        return "Conservative"
    if score < 3.2:
        return "Moderate"
    if score < 4.0:
        return "Growth"
    if score < 4.6:
        return "Aggressive"
    return "Very Aggressive"

def _bucket_loss_tolerance(percent):
    if percent is None:
        return 2
    p = max(0.0, float(percent))
    if p <= 5: return 1
    if p <= 10: return 2
    if p <= 15: return 3
    if p <= 25: return 4
    if p <= 35: return 5
    return 6

def _bucket_savings(percent):
    if percent is None:
        return 2
    p = max(0.0, float(percent))
    if p < 10: return 1
    if p < 20: return 2
    if p < 30: return 3
    if p < 40: return 4
    return 5

def _bucket_income_stability(label):
    if not label:
        return 3
    mapping = {
        "very unstable": 1,
        "unstable": 2,
        "average": 3,
        "stable": 4,
        "very stable": 5,
    }
    return mapping.get(label.strip().lower(), 3)

def _bucket_emergency_fund(months):
    if months is None:
        return 2
    m = max(0.0, float(months))
    if m < 1: return 1
    if m < 3: return 2
    if m < 6: return 3
    if m < 12: return 4
    return 5

def _bucket_behavior(label):
    if not label:
        return 3
    mapping = {
        "sell": 1,
        "reduce": 2,
        "hold": 3,
        "buy": 4,
        "aggressive buy": 5,
        "aggressive_buy": 5,
    }
    return mapping.get(label.strip().lower(), 3)

def _category_index(cat):
    try:
        return _RISK_CATEGORIES.index(cat)
    except Exception:
        return _RISK_CATEGORIES.index("Moderate")

def _index_to_category(idx):
    idx = max(0, min(len(_RISK_CATEGORIES) - 1, idx))
    return _RISK_CATEGORIES[idx]

def compute_advanced_risk(payload, age):
    risk = (payload.get("risk") or {})
    goals = (payload.get("goals") or {})
    lifestyle = (payload.get("savings") or {})
    investments = (payload.get("investments") or {})
    allocation = investments.get("allocation") or {}

    tenure_years = risk.get("goal_tenure_years") or risk.get("primary_horizon_years")
    if tenure_years is None:
        horizon_text = risk.get("primary_horizon")
        if isinstance(horizon_text, str):
            ht = horizon_text.lower()
            if ht.startswith("short"):
                tenure_years = 3
            elif ht.startswith("med"):
                tenure_years = 5
            else:
                tenure_years = 10
    try:
        tenure_years = float(tenure_years) if tenure_years not in (None, "") else None
    except Exception:
        tenure_years = None

    flexibility = risk.get("goal_flexibility") or goals.get("flexibility")
    importance = risk.get("goal_importance") or goals.get("importance")
    behavior = risk.get("behavior")
    loss_pct = risk.get("loss_tolerance_percent")
    try:
        loss_pct = float(loss_pct) if loss_pct not in (None, "") else None
    except Exception:
        loss_pct = None
    savings_percent = lifestyle.get("savingsPercent") or lifestyle.get("savings_percent")
    try:
        savings_percent = float(savings_percent) if savings_percent not in (None, "") else None
    except Exception:
        savings_percent = None
    income_stability = risk.get("income_stability")
    emergency_months = risk.get("emergency_fund_months")
    if emergency_months in (None, ""):
        ef_amt = payload.get("emergencyFundAmount")
        mexp = (payload.get("income") or {}).get("monthlyExpenses")
        try:
            ef_amt_f = float(ef_amt) if ef_amt not in (None, "") else None
            mexp_f = float(mexp) if mexp not in (None, "") else None
            if ef_amt_f and mexp_f and mexp_f > 0:
                emergency_months = ef_amt_f / mexp_f
        except Exception:
            emergency_months = None
    try:
        emergency_months = float(emergency_months) if emergency_months not in (None, "") else None
    except Exception:
        emergency_months = None

    tenure_limit = _tenure_limit_category(tenure_years)

    b_behavior = _bucket_behavior(behavior)
    b_loss = _bucket_loss_tolerance(loss_pct)
    b_savings = _bucket_savings(savings_percent)
    b_income = _bucket_income_stability(income_stability)
    b_emergency = _bucket_emergency_fund(emergency_months)

    score = (b_behavior * 0.25) + (b_loss * 0.25) + (b_savings * 0.20) + (b_income * 0.15) + (b_emergency * 0.15)
    appetite_category = _score_to_category(score)

    baseline_category = _index_to_category(min(_category_index(appetite_category), _category_index(tenure_limit)))
    adjustments = []

    def _goal_adj(value, mapping):
        if not value:
            return 0
        return mapping.get(str(value).strip().lower(), 0)

    flex_adj = _goal_adj(flexibility, {"critical": -1, "fixed": 0, "flexible": 1})
    imp_adj = _goal_adj(importance, {"essential": -1, "important": 0, "lifestyle": 1})
    net_goal_adj = flex_adj + imp_adj
    if net_goal_adj != 0:
        new_idx = _category_index(baseline_category) + net_goal_adj
        new_cat = _index_to_category(new_idx)
        if _category_index(new_cat) > _category_index(tenure_limit):
            if flex_adj == 1 and imp_adj == 1 and net_goal_adj == 2:
                new_cat = _index_to_category(_category_index(tenure_limit) + 1)
            else:
                new_cat = tenure_limit
        adjustments.append(f"GoalAdjustment: {baseline_category} -> {new_cat} (net {net_goal_adj})")
        baseline_category = new_cat

    equity_pct = None
    try:
        eq_raw = allocation.get("equity")
        equity_pct = float(eq_raw) if eq_raw not in (None, "") else None
    except Exception:
        equity_pct = None

    if equity_pct is not None:
        band = _RISK_EQUITY_BANDS.get(baseline_category)
        if band:
            band_min, band_max = band
            nearest_edge_dist = 0
            if equity_pct < band_min:
                nearest_edge_dist = band_min - equity_pct
            elif equity_pct > band_max:
                nearest_edge_dist = equity_pct - band_max
            if nearest_edge_dist > 20:
                forced = "Moderate"
                if _category_index(baseline_category) > _category_index(forced):
                    adjustments.append(f"PortfolioForce: {baseline_category} -> {forced} (drift {nearest_edge_dist:.1f}pp)")
                    baseline_category = forced
            else:
                if equity_pct > (band_max + 10):
                    new_cat = _index_to_category(_category_index(baseline_category) - 1)
                    adjustments.append(f"PortfolioDowngrade: equity {equity_pct:.1f}% > {band_max + 10}%")
                    baseline_category = new_cat
                elif equity_pct < (band_min - 10):
                    if _category_index(baseline_category) < _category_index(tenure_limit):
                        new_cat = _index_to_category(_category_index(baseline_category) + 1)
                        if _category_index(new_cat) > _category_index(tenure_limit):
                            new_cat = tenure_limit
                        adjustments.append(f"PortfolioUpgrade: equity {equity_pct:.1f}% < {band_min - 10}%")
                        baseline_category = new_cat

    final_category = baseline_category
    if _category_index(final_category) > _category_index(tenure_limit):
        final_category = tenure_limit

    final_band = _RISK_EQUITY_BANDS.get(final_category)
    band_mid = None
    if final_band:
        band_mid = round((final_band[0] + final_band[1]) / 2, 1)

    reasoning_parts = [
        f"Score {score:.2f} -> Appetite {appetite_category}",
        f"Tenure limit {tenure_limit}",
        f"Baseline after adjustments {baseline_category}",
    ]
    if adjustments:
        reasoning_parts.append("Adjustments: " + "; ".join(adjustments))
    reasoning_parts.append(f"Final {final_category}")
    reasoning_text = " | ".join(reasoning_parts)

    return {
        "score": round(score, 2),
        "tenureLimitCategory": tenure_limit,
        "appetiteCategory": appetite_category,
        "baselineCategory": baseline_category,
        "finalCategory": final_category,
        "adjustmentsApplied": adjustments,
        "recommendedEquityBand": {
            "min": final_band[0] if final_band else None,
            "max": final_band[1] if final_band else None,
        } if final_band else None,
        "recommendedEquityMid": band_mid,
        "reasoningText": reasoning_text,
        "raw": {
            "tenureYears": tenure_years,
            "flexibility": flexibility,
            "importance": importance,
            "behavior": behavior,
            "lossTolerancePercent": loss_pct,
            "savingsPercent": savings_percent,
            "incomeStability": income_stability,
            "emergencyFundMonths": emergency_months,
            "equityAllocationPercent": equity_pct,
        },
    }

def _generate_unified_equity_recommendation(results, inputs):
    """
    Generate unified equity recommendation by comparing current allocation
    vs recommended band from advanced risk assessment.
    Prevents contradictory advice.
    """
    allocation = (inputs.get("investments") or {}).get("allocation") or {}
    current_equity = _safe_float(allocation.get("equity"), 0.0)

    advanced_risk = results.get("advancedRisk") or {}
    rec_band = advanced_risk.get("recommendedEquityBand") or {}
    rec_min = rec_band.get("min")
    rec_max = rec_band.get("max")
    rec_mid = advanced_risk.get("recommendedEquityMid")

    ihs_band = results.get("ihs", {}).get("band")

    # Compare current vs recommended
    if rec_min is not None and rec_max is not None:
        if current_equity < rec_min - 5:
            # Below recommended range
            gap = rec_min - current_equity
            if gap > 20:
                return f"Increase equity from {round(current_equity, 1)}% to at least {rec_min}% (target: {rec_mid}%). Consider gradual rebalancing via SIP."
            else:
                return f"Increase equity from {round(current_equity, 1)}% towards {rec_min}-{rec_max}% range (target: {rec_mid}%)."

        elif current_equity > rec_max + 5:
            # Above recommended range
            gap = current_equity - rec_max
            if gap > 20:
                return f"Reduce equity from {round(current_equity, 1)}% to {rec_max}% or below. Consider booking profits and rebalancing to debt."
            else:
                return f"Consider rebalancing: current equity {round(current_equity, 1)}% exceeds recommended {rec_min}-{rec_max}%."

        else:
            # Within recommended range
            if ihs_band == "Excellent":
                return f"Maintain allocation (equity: {round(current_equity, 1)}%). Rebalance annually and add international exposure."
            elif ihs_band == "Good":
                return f"Equity allocation ({round(current_equity, 1)}%) is well-aligned. Consider international exposure and tax optimization."
            else:
                return f"Equity allocation ({round(current_equity, 1)}%) is within recommended range ({rec_min}-{rec_max}%). Focus on fund quality and diversification."

    # Fallback if no advanced risk data
    rp = results.get("riskProfile")
    if rp == "Conservative":
        return "Focus on debt-heavy allocation (25-40% equity) with low-volatility funds."
    elif rp == "Balanced":
        return "Aim for balanced 40-55% equity allocation with quality debt funds."
    elif rp == "Aggressive":
        return "Target 70-85% equity allocation; add gold (5-10%) as hedge."

    return "Review asset allocation with advisor to align with risk profile and goals."

def generate_flags_and_recommendations(results, inputs):
    flags = []
    recs = []

    # Convenience
    annual_income = _safe_float(inputs.get("income", {}).get("annualIncome"), 0.0)
    monthly_expenses = _safe_float(inputs.get("income", {}).get("monthlyExpenses"), 0.0)
    monthly_emi = _safe_float(inputs.get("income", {}).get("monthlyEmi"), 0.0)
    life_cover = _safe_float(inputs.get("insurance", {}).get("lifeCover"), 0.0)
    savings_percent = resolve_savings_percent(
        inputs.get("savings", {}).get("savingsPercent"),
        inputs.get("savings", {}).get("savingsBand"),
    )
    allocation = (inputs.get("investments") or {}).get("allocation") or {}
    equity = _safe_float(allocation.get("equity"), 0.0)
    debt = _safe_float(allocation.get("debt"), 0.0)
    gold = _safe_float(allocation.get("gold"), 0.0)
    real_estate = _safe_float(allocation.get("realEstate"), 0.0)

    # Flags
    if results.get("surplusBand") == "Low":
        flags.append(f"Low Surplus: Saving {round(savings_percent,1)}% vs Benchmark 20%+")
    if results.get("insuranceGap") == "Underinsured":
        required = 10.0 * annual_income
        flags.append(f"Underinsured: Cover Rs. {_format_indian_amount(life_cover)} vs Required Rs. {_format_indian_amount(required)}")
    if results.get("debtStress") == "Stressed":
        # Recompute ratio for message
        monthly_income = annual_income / 12.0 if annual_income > 0 else 0.0
        ratio_pct = (monthly_emi / monthly_income * 100.0) if monthly_income > 0 else 0.0
        flags.append(f"Debt Stress: EMI {round(ratio_pct,1)}% of Income vs Benchmark <=40%")
    if results.get("liquidity") == "Insufficient":
        # Recompute months for message
        ef = _safe_float(inputs.get("emergencyFundAmount"), 0.0)
        months = (ef / monthly_expenses) if monthly_expenses > 0 else 0.0
        flags.append(f"Insufficient Liquidity: Emergency fund {round(months,1)} months vs Benchmark 6+")
    ihs = results.get("ihs") or {}
    if ihs.get("band") in ["Poor", "Average", "Good"]:
        skew = ""
        if (debt + gold) > 60.0 and equity < 20.0:
            skew = "skewed allocation to FD/Gold"
        elif equity > 80.0:
            skew = "high equity concentration"
        elif real_estate > 50.0:
            skew = "concentration in Real Estate"
        extra = f", {skew}" if skew else ""
        flags.append(f"Score = {ihs.get('score')} -> {ihs.get('band')}: Saving {round(savings_percent,1)}% income{extra}")

    # Recommendations
    # 1. Unified Equity Recommendation (replaces old risk profile + IHS equity recommendations)
    equity_rec = _generate_unified_equity_recommendation(results, inputs)
    recs.append(equity_rec)

    # 2. Surplus recommendations
    sb = results.get("surplusBand")
    if sb == "Low":
        recs.append("Cut discretionary spending; automate an SIP.")
    elif sb == "Adequate":
        recs.append("Maintain discipline and allocate savings across your goals.")
    elif sb == "Strong":
        recs.append("Accelerate retirement savings and explore growth instruments.")

    # 3. Insurance recommendations
    ig = results.get("insuranceGap")
    if ig == "Underinsured":
        recs.append("Buy term life insurance for 10-12x your income and add/increase health cover.")
    else:
        recs.append("Review your cover every 2-3 years.")

    # 4. Debt recommendations
    ds = results.get("debtStress")
    if ds == "Stressed":
        recs.append("Refinance high-cost debt; repay high-cost debt first; avoid new borrowing.")
    elif ds == "Moderate":
        recs.append("Prioritize repaying unsecured debt.")
    else:
        recs.append("You can leverage strategically if needed.")

    # 5. Liquidity recommendations
    liq = results.get("liquidity")
    if liq == "Insufficient":
        recs.append("Build a liquid buffer (cash/liquid funds) until 6 months of expenses are covered.")
    else:
        recs.append("Invest your incremental surplus into growth assets.")

    # 6. IHS portfolio quality recommendations (non-equity advice only)
    ihs_band = ihs.get("band")
    if ihs_band == "Poor":
        recs.append("Increase savings rate, exit unsuitable products, and improve portfolio diversification.")
    elif ihs_band == "Average":
        # Check if equity is already high before suggesting to increase it
        if equity > 60.0:
            recs.append("Your portfolio is equity-heavy; consider rebalancing towards debt/hybrid to reduce risk while aligning with your goals.")
        else:
            recs.append("Increase your equity allocation and align investments with your goals.")
    elif ihs_band == "Good":
        recs.append("Conduct an annual review, add international exposure, and optimize for tax.")
    elif ihs_band == "Excellent":
        recs.append("Maintain your strategy, rebalance annually, and explore advanced strategies.")

    return flags, recs

def analyze_financial_health(payload: dict):
    payload = payload or {}
    personal = payload.get("personal") or {}
    income = payload.get("income") or {}
    goals = payload.get("goals") or {}
    risk = payload.get("risk") or {}
    insurance = payload.get("insurance") or {}
    savings = payload.get("savings") or {}
    investments = payload.get("investments") or {}
    allocation = investments.get("allocation") or {}
    current_products = investments.get("current") or []
    emergency_fund_amount = payload.get("emergencyFundAmount", 0)

    # Resolve inputs
    age = personal.get("age")
    tolerance = risk.get("tolerance")
    horizon = goals.get("goalHorizon")
    savings_percent = resolve_savings_percent(savings.get("savingsPercent"), savings.get("savingsBand"))

    basic_risk_profile, basic_risk_score = compute_risk_profile(age, tolerance, horizon)
    surplus_band = compute_surplus_band(savings_percent)
    try:
        advanced_risk = compute_advanced_risk(payload, age)
    except Exception:
        advanced_risk = None
    insurance_gap, required_cover = compute_insurance_gap(income.get("annualIncome"), insurance.get("lifeCover"))
    debt_stress, emi_ratio_pct = compute_debt_stress(income.get("monthlyEmi"), income.get("annualIncome"))
    liquidity, liquidity_months = compute_liquidity(income.get("monthlyExpenses"), emergency_fund_amount)
    ihs = compute_ihs(savings_percent, current_products, allocation)

    results = {
        "riskProfile": (advanced_risk["finalCategory"] if advanced_risk else basic_risk_profile),
        "advancedRisk": advanced_risk,
        "surplusBand": surplus_band,
        "insuranceGap": insurance_gap,
        "debtStress": debt_stress,
        "liquidity": liquidity,
        "ihs": ihs,
        # extra diagnostics
        "_diagnostics": {
            "riskScore": basic_risk_score,
            "emiPct": round(emi_ratio_pct, 2),
            "liquidityMonths": round(liquidity_months, 2),
            "requiredLifeCover": required_cover
        }
    }

    flags, recs = generate_flags_and_recommendations(results, payload)
    results["flags"] = flags
    results["recommendations"] = recs
    return results

# --- Document insights aggregation (from linked uploads) ---
def aggregate_doc_insights_for_questionnaire(qid: int) -> dict:
    """
    Aggregate insights using deterministic extractors from all documents linked to the questionnaire.
    Combines:
      - DB metrics (numeric aggregations for bank/CAS/ITR/insurance)
      - Indexed section/table-driven summaries via extract_and_store_from_indexed(document_id)
    Returns:
      {
        "bank": {...},                   # totals + opening/closing + net_cashflow
        "portfolio": {...},              # CAS allocation percentages if present
        "insurance": {"sum_assured_or_insured": number}?,
        "itr": {"gross_total_income": n, "taxable_income": n, "total_tax_paid": n}?,
        "raw_extracts": [                # per-document extracted summaries from index
          {
            "document_id": int,
            "summary": {
              "investment_snapshot": {...} | None,
              "account_summary": {...} | None,
              "portfolio_summary": {...} | None,
              "provenance": {...}
            }
          },
          ...
        ]
      }
    """
    uploads = list_questionnaire_uploads(qid) or []
    doc_ids = [r["document_id"] for r in uploads if r["document_id"] is not None]

    # Aggregates
    bank = {"total_inflows": 0.0, "total_outflows": 0.0, "opening_balance": None, "closing_balance": None}
    portfolio_alloc = {}
    insurance = {}
    itr = {}

    per_doc_extracts = []

    for did in doc_ids:
        # 1) Deterministic summaries from indexed sections/tables
        try:
            idx_summary = extract_and_store_from_indexed(did) or {}
            per_doc_extracts.append({"document_id": did, "summary": idx_summary})
            # Merge account summary for opening/closing only to avoid double counting with metrics
            acct = idx_summary.get("account_summary") or {}
            try:
                ob = acct.get("opening_balance")
                cb = acct.get("closing_balance")
                if bank["opening_balance"] is None and isinstance(ob, (int, float)):
                    bank["opening_balance"] = float(ob)
                if bank["closing_balance"] is None and isinstance(cb, (int, float)):
                    bank["closing_balance"] = float(cb)
            except Exception:
                pass
            # No direct allocation percentages in idx_summary; keep for narratives via facts
        except Exception:
            # If extraction fails for any doc, continue with metrics-only for that doc
            pass

        # 2) Numeric metrics for CAS allocation/ITR/insurance/bank (authoritative numeric store)
        try:
            mets = list_metrics(did) or []
        except Exception:
            mets = []
        for m in mets:
            md = dict(m)
            k = (md.get("key") or "").strip().lower()
            vnum = md.get("value_num")
            try:
                if k in ("total_inflows", "total_outflows"):
                    if vnum is not None:
                        if k == "total_inflows":
                            bank["total_inflows"] += float(vnum)
                        else:
                            bank["total_outflows"] += float(vnum)
                elif k in ("opening_balance", "closing_balance"):
                    if vnum is not None:
                        bank[k] = float(vnum)
                elif k.startswith("portfolio_"):
                    if vnum is not None:
                        portfolio_alloc[k.replace("portfolio_", "")] = float(vnum)
                elif k in ("insurance_sum_assured_or_insured", "sum_assured_or_insured"):
                    if vnum is not None:
                        insurance["sum_assured_or_insured"] = float(vnum)
                elif k in ("gross_total_income", "taxable_income", "total_tax_paid"):
                    if vnum is not None:
                        itr[k] = float(vnum)
            except Exception:
                continue

    # Compute net cashflow
    try:
        inflow = float(bank.get("total_inflows") or 0.0)
        outflow = float(bank.get("total_outflows") or 0.0)
        bank["net_cashflow"] = inflow - outflow
    except Exception:
        bank["net_cashflow"] = None

    out: Dict[str, dict] = {}
    if any(v not in (None, 0.0) for v in bank.values()):
        out["bank"] = bank
    if portfolio_alloc:
        out["portfolio"] = portfolio_alloc
    if insurance:
        out["insurance"] = insurance
    if itr:
        out["itr"] = itr
    if per_doc_extracts:
        out["raw_extracts"] = per_doc_extracts
    return out

def _get_cas_data_for_questionnaire(qid: int) -> dict:
    """
    Retrieve CAS data from uploaded Mutual Fund CAS documents.
    Returns dict with sip_details, transaction_summary, asset_allocation or empty dict.
    """
    uploads = list_questionnaire_uploads(qid) or []

    for upload in uploads:
        if upload["doc_type"] == "Mutual fund CAS (Consolidated Account Statement)":
            metadata_json = upload["metadata_json"]
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                    cas_data = metadata.get("cas_data", {})
                    if cas_data:
                        return cas_data
                except Exception as e:
                    print(f"Error parsing CAS metadata: {e}")
                    continue

    return {}

def build_prefill_from_insights(qid: int) -> dict:
    """
    Derive questionnaire prefill values from linked uploads (Bank/CAS/ITR/Insurance).
    Prefill rules:
      - Lifestyle:
          annual_income := ITR.gross_total_income if present else Bank.total_inflows
          monthly_expenses := Bank.total_outflows / 12
          savings_percent := max(0, inflows - outflows)/inflows * 100
      - Allocation:
          from CAS portfolio_* metrics (equity, debt, gold, realEstate, insuranceLinked, cash)
      - Insurance:
          life_cover := sum_assured_or_insured if insurance_type suggests life
          health_cover := sum_assured_or_insured if insurance_type suggests health
    """
    di = aggregate_doc_insights_for_questionnaire(qid) or {}
    bank = di.get("bank") or {}
    portfolio = di.get("portfolio") or {}
    itr = di.get("itr") or {}
    ins = di.get("insurance") or {}

    lifestyle = {}
    try:
        # Annual income prefers ITR; monthly expenses from bank; savings% strictly from bank flows
        itr_income = itr.get("gross_total_income")
        bank_inflow = bank.get("total_inflows")
        bank_outflow = bank.get("total_outflows")

        inflow = itr_income if isinstance(itr_income, (int, float)) and itr_income > 0 else bank_inflow
        outflow = bank_outflow

        if isinstance(inflow, (int, float)) and inflow > 0:
            lifestyle["annual_income"] = round(float(inflow), 2)
        if isinstance(outflow, (int, float)) and outflow > 0:
            lifestyle["monthly_expenses"] = round(float(outflow) / 12.0, 2)
        if isinstance(bank_inflow, (int, float)) and bank_inflow > 0 and isinstance(bank_outflow, (int, float)):
            sp = max(0.0, (float(bank_inflow) - float(bank_outflow))) / float(bank_inflow) * 100.0
            sp = max(0.0, min(100.0, sp))
            lifestyle["savings_percent"] = round(sp, 2)
    except Exception:
        pass

    # Extract monthly EMI from bank statement recurring debits
    try:
        uploads = list_questionnaire_uploads(qid) or []
        total_monthly_emi = 0.0
        emi_keywords = ["emi", "loan", "mortgage", "instalment", "installment", "repayment", "home loan", "car loan", "personal loan", "vehicle loan", "housing loan"]
        for upload in uploads:
            if (upload["doc_type"] or "").lower() == "bank statement":
                metadata_json = upload["metadata_json"]
                if metadata_json:
                    try:
                        metadata = json.loads(metadata_json)
                        bank_data = metadata.get("bank_data") or {}
                        recurring_debits = bank_data.get("recurring_debits") or []
                        for debit in recurring_debits:
                            desc = (debit.get("description") or "").lower()
                            amount = debit.get("amount")
                            freq = (debit.get("frequency") or "").lower()
                            # Check if this is an EMI payment
                            if any(kw in desc for kw in emi_keywords):
                                if isinstance(amount, (int, float)) and amount > 0:
                                    # Convert to monthly if not already monthly
                                    if freq in ("monthly", "month"):
                                        total_monthly_emi += float(amount)
                                    elif freq in ("quarterly", "quarter"):
                                        total_monthly_emi += float(amount) / 3.0
                                    elif freq in ("yearly", "annual", "year"):
                                        total_monthly_emi += float(amount) / 12.0
                                    else:
                                        # Assume monthly if frequency unclear
                                        total_monthly_emi += float(amount)
                    except Exception:
                        continue
        if total_monthly_emi > 0:
            lifestyle["monthly_emi"] = round(total_monthly_emi, 2)
            print(f"[Prefill] Extracted monthly_emi: {lifestyle['monthly_emi']}")
    except Exception as e:
        print(f"[Prefill] Error extracting monthly_emi: {e}")

    allocation = {}
    try:
        for k in ["equity", "debt", "gold", "realEstate", "insuranceLinked", "cash"]:
            v = portfolio.get(k)
            if isinstance(v, (int, float)) and v >= 0:
                allocation[k] = float(v)
    except Exception:
        pass

    # Scan uploads for insurance prefill - need to read insurance_type from metadata
    insurance_prefill = {}
    try:
        uploads = list_questionnaire_uploads(qid) or []
        for upload in uploads:
            doc_type = (upload["doc_type"] or "").lower()
            if "insurance" in doc_type:
                metadata_json = upload["metadata_json"]
                if metadata_json:
                    try:
                        metadata = json.loads(metadata_json)
                        ins_type = str(metadata.get("insurance_type") or "").lower()
                        sum_val = metadata.get("sum_assured_or_insured")
                        
                        # Also check aggregated insights if not in metadata
                        if sum_val is None or sum_val == "N/A":
                            sum_val = ins.get("sum_assured_or_insured")
                        
                        if isinstance(sum_val, (int, float)) and sum_val > 0:
                            if "health" in ins_type or "mediclaim" in ins_type:
                                # Add to health cover (may have multiple health policies)
                                existing_health = insurance_prefill.get("health_cover", 0.0)
                                insurance_prefill["health_cover"] = existing_health + float(sum_val)
                            elif "life" in ins_type or "term" in ins_type or "ulip" in ins_type:
                                # Add to life cover
                                existing_life = insurance_prefill.get("life_cover", 0.0)
                                insurance_prefill["life_cover"] = existing_life + float(sum_val)
                            else:
                                # Unknown type: default to life_cover
                                existing_life = insurance_prefill.get("life_cover", 0.0)
                                insurance_prefill["life_cover"] = existing_life + float(sum_val)
                    except Exception:
                        continue
        
        # Fallback: if no metadata found, use aggregated insights
        if not insurance_prefill:
            sum_val = ins.get("sum_assured_or_insured")
            if isinstance(sum_val, (int, float)) and sum_val > 0:
                # Default to life_cover when type unknown
                insurance_prefill["life_cover"] = float(sum_val)
    except Exception:
        pass

    # Personal info extraction - priority order:
    # 1. Bank statement account_holder_name (most reliable, directly from user's bank)
    # 2. CAS investor_name
    # 3. Insurance policy_holder
    # 4. ITR assessee_name (least reliable - may pick up father's name field)
    personal_info = {}
    try:
        uploads = list_questionnaire_uploads(qid) or []
        
        # First pass: look for bank statement (highest priority for name)
        for upload in uploads:
            if personal_info.get("name"):
                break
            doc_type = (upload["doc_type"] or "").lower()
            if "bank" in doc_type:
                metadata_json = upload["metadata_json"]
                if metadata_json:
                    try:
                        metadata = json.loads(metadata_json)
                        account_holder = metadata.get("account_holder_name")
                        if account_holder and account_holder != "N/A" and len(str(account_holder)) > 2:
                            personal_info["name"] = str(account_holder).strip().title()
                    except Exception:
                        continue
        
        # Second pass: look for CAS investor_name
        if not personal_info.get("name"):
            for upload in uploads:
                if personal_info.get("name"):
                    break
                doc_type = (upload["doc_type"] or "").lower()
                if "cas" in doc_type or "mutual fund" in doc_type:
                    metadata_json = upload["metadata_json"]
                    if metadata_json:
                        try:
                            metadata = json.loads(metadata_json)
                            cas_data = metadata.get("cas_data") or {}
                            investor_name = metadata.get("investor_name") or cas_data.get("investor_name")
                            if investor_name and investor_name != "N/A" and len(investor_name) > 2:
                                personal_info["name"] = investor_name.strip().title()
                        except Exception:
                            continue
        
        # Third pass: look for Insurance policy_holder
        if not personal_info.get("name"):
            for upload in uploads:
                if personal_info.get("name"):
                    break
                doc_type = (upload["doc_type"] or "").lower()
                if "insurance" in doc_type:
                    metadata_json = upload["metadata_json"]
                    if metadata_json:
                        try:
                            metadata = json.loads(metadata_json)
                            policy_holder = metadata.get("policy_holder")
                            if policy_holder and policy_holder != "N/A" and len(policy_holder) > 2:
                                personal_info["name"] = policy_holder.strip().title()
                        except Exception:
                            continue
        
        # Fourth pass: look for ITR assessee_name (lowest priority)
        if not personal_info.get("name"):
            for upload in uploads:
                if personal_info.get("name"):
                    break
                doc_type = (upload["doc_type"] or "").lower()
                if "itr" in doc_type:
                    metadata_json = upload["metadata_json"]
                    if metadata_json:
                        try:
                            metadata = json.loads(metadata_json)
                            assessee_name = metadata.get("assessee_name")
                            if assessee_name and assessee_name != "N/A" and len(assessee_name) > 2:
                                personal_info["name"] = assessee_name.strip().title()
                        except Exception:
                            continue
        
        # Extract age from date_of_birth (from any document)
        for upload in uploads:
            if personal_info.get("age"):
                break
            metadata_json = upload["metadata_json"]
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                    dob = metadata.get("date_of_birth")
                    if dob and dob != "N/A":
                        from datetime import datetime
                        for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"]:
                            try:
                                birth_date = datetime.strptime(dob, fmt)
                                if birth_date.year < 100:  # Handle 2-digit years
                                    birth_date = birth_date.replace(year=birth_date.year + 1900)
                                today = datetime.now()
                                age = today.year - birth_date.year
                                if (today.month, today.day) < (birth_date.month, birth_date.day):
                                    age -= 1
                                if 18 <= age <= 100:
                                    personal_info["age"] = age
                                break
                            except ValueError:
                                continue
                except Exception:
                    continue
    except Exception:
        pass

    return {
        "questionnaire_id": qid,
        "docInsights": di,
        "personal_info": personal_info,
        "lifestyle": lifestyle,
        "allocation": allocation,
        "insurance": insurance_prefill,
    }

# --- Flask Routes ---

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({"status": "ok"}), 200

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if data.get('username') == 'user' and data.get('password') == 'password':
        return jsonify({"message": "Login successful", "token": "fake-jwt-token"}), 200
    return jsonify({"message": "Invalid credentials"}), 401

@app.route('/upload', methods=['POST'])
def upload_document():
    print("Upload request received.")
    files, types = [], []
    for key in request.files:
        if key.startswith('file'):
            files.append(request.files[key])
            types.append(request.form.get(f"type{key[4:]}", "Unknown"))
    questionnaire_id_raw = request.form.get("questionnaireId") or request.form.get("questionnaire_id")
    questionnaire_id = None
    try:
        if questionnaire_id_raw:
            questionnaire_id = int(questionnaire_id_raw)
    except Exception:
        questionnaire_id = None

    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    extracted_data = {}
    debug_shas = []
    upload_link_ids = {}  # Track upload link IDs for metadata updates
    for idx, (file, doc_type) in enumerate(zip(files, types)):
        try:
            print(f"Processing file {idx + 1}: {file.filename}")
            # Read file bytes once so we can reuse for multiple parsers
            file_bytes = file.read()
            file_stream_for_text = io.BytesIO(file_bytes)
            text = extract_structured_text_with_tables(file_stream_for_text)
            print(f"Extracted text for file {idx + 1} successfully.")

            # Index and extract deterministic summaries (snapshot/statement) with provenance
            sha, doc_id, summaries = index_and_extract(file_bytes, filename=file.filename or "upload.pdf")
            debug_shas.append(sha)
            # Link to questionnaire if provided
            if questionnaire_id:
                try:
                    link_id = link_questionnaire_upload(
                        questionnaire_id=questionnaire_id,
                        document_id=doc_id,
                        sha256=sha,
                        doc_type=doc_type,
                        filename=file.filename,
                        metadata={"size_bytes": len(file_bytes)}
                    )
                    # Store link_id for updating CAS metadata later
                    upload_link_ids[idx] = link_id
                except Exception as e:
                    print(f"Failed linking upload to questionnaire {questionnaire_id}: {e}")
            
            if not text.strip():
                extracted_data[f"Document {idx+1} ({doc_type})"] = {"error": "No text could be extracted from this PDF."}
                continue
            
            # Map document types to functions
            doc_map = {
                "Bank statement": ("Bank Statement", extract_bank_statement_hybrid),
                "ITR": ("ITR", extract_itr_hybrid),
                "Insurance document": ("Insurance Document", extract_insurance_hybrid),
                "Mutual fund CAS (Consolidated Account Statement)": ("Mutual Fund CAS", extract_mutual_fund_cas_hybrid)
            }
            
            if doc_type in doc_map:
                name, func = doc_map[doc_type]
                if doc_type == "Bank statement":
                    # Build structured transactions and optionally persist as JSON
                    tx_payload = extract_bank_statement_transactions(io.BytesIO(file_bytes))
                    json_url = None
                    if SAVE_TX_JSON:
                        json_name = f"bank_transactions_{os.urandom(6).hex()}.json"
                        json_path = os.path.join(OUTPUT_DIR, json_name)
                        try:
                            with open(json_path, 'w', encoding='utf-8') as jf:
                                json.dump(tx_payload, jf, ensure_ascii=False, indent=2)
                            json_url = url_for('download_file', filename=json_name, _external=True)
                        except Exception as je:
                            print(f"Failed to save transactions JSON: {je}")
                            json_url = None

                    bank_data = func(text, transactions_payload=tx_payload, save_json_path=json_url)
                    # Merge DB-backed summaries if present
                    if isinstance(summaries, dict):
                        if summaries.get("account_summary"):
                            bank_data["account_summary"] = summaries["account_summary"]
                        if summaries.get("investment_snapshot"):
                            bank_data["investment_snapshot"] = summaries["investment_snapshot"]
                        if summaries.get("portfolio_summary"):
                            bank_data["portfolio_summary"] = summaries["portfolio_summary"]
                        if summaries.get("provenance"):
                            bank_data["provenance"] = summaries["provenance"]
                    # Do not include raw transactions in PDF; attach summary only
                    extracted_data[f"{name} {idx+1}"] = bank_data
                    try:
                        _persist_metrics_for_doc(doc_id, bank_data)
                    except Exception as e:
                        print(f"Persist metrics (bank) failed: {e}")
                    
                    # Save bank statement metadata including recurring debits (for EMI prefill)
                    if idx in upload_link_ids:
                        try:
                            bank_metadata = {
                                "size_bytes": len(file_bytes),
                                "account_holder_name": (bank_data.get("account_summary") or {}).get("account_holder_name"),
                                "bank_data": {
                                    "account_summary": bank_data.get("account_summary", {}),
                                    "recurring_debits": bank_data.get("recurring_debits", []),
                                    "recurring_credits": bank_data.get("recurring_credits", []),
                                }
                            }
                            update_questionnaire_upload_metadata(upload_link_ids[idx], bank_metadata)
                            recurring_debits_count = len(bank_data.get("recurring_debits", []))
                            print(f"[Upload] Bank statement metadata saved: {recurring_debits_count} recurring debits")
                        except Exception as e:
                            print(f"Error updating Bank statement metadata: {e}")
                else:
                    other_data = func(text)
                    # Merge DB-backed summaries if present (useful for CAS/Portfolio PDFs)
                    if isinstance(summaries, dict):
                        if summaries.get("account_summary"):
                            other_data["account_summary"] = summaries["account_summary"]
                        if summaries.get("investment_snapshot"):
                            other_data["investment_snapshot"] = summaries["investment_snapshot"]
                        if summaries.get("portfolio_summary"):
                            other_data["portfolio_summary"] = summaries["portfolio_summary"]
                        if summaries.get("provenance"):
                            other_data["provenance"] = summaries["provenance"]

                    # Update document metadata if linked to questionnaire
                    if idx in upload_link_ids:
                        try:
                            metadata_update = {"size_bytes": len(file_bytes)}
                            
                            # CAS metadata
                            if doc_type == "Mutual fund CAS (Consolidated Account Statement)":
                                metadata_update["investor_name"] = other_data.get("investor_name")
                                metadata_update["cas_data"] = {
                                    "sip_details": other_data.get("sip_details", []),
                                    "transaction_summary": other_data.get("transaction_summary", {}),
                                    "asset_allocation": other_data.get("asset_allocation", {}),
                                    "investment_snapshot": other_data.get("investment_snapshot", {}),
                                    "holdings": other_data.get("holdings", []),
                                }
                            
                            # ITR metadata - store personal info
                            elif doc_type == "ITR":
                                metadata_update["assessee_name"] = other_data.get("assessee_name")
                                metadata_update["date_of_birth"] = other_data.get("date_of_birth")
                                metadata_update["pan"] = other_data.get("pan")
                            
                            # Insurance metadata - store policy holder and insurance_type
                            elif doc_type == "Insurance document":
                                metadata_update["policy_holder"] = other_data.get("policy_holder")
                                metadata_update["insurance_type"] = other_data.get("insurance_type")
                                metadata_update["sum_assured_or_insured"] = other_data.get("sum_assured_or_insured")
                                metadata_update["date_of_birth"] = other_data.get("date_of_birth")
                            
                            update_questionnaire_upload_metadata(upload_link_ids[idx], metadata_update)
                        except Exception as e:
                            print(f"Error updating {doc_type} metadata: {e}")

                    extracted_data[f"{name} {idx+1}"] = other_data
                    try:
                        _persist_metrics_for_doc(doc_id, other_data)
                    except Exception as e:
                        print(f"Persist metrics ({doc_type}) failed: {e}")
            else:
                # Generic fallback: still return deterministic DB-backed summaries even if doc type is unknown
                generic = {}
                if isinstance(summaries, dict):
                    if summaries.get("account_summary"):
                        generic["account_summary"] = summaries["account_summary"]
                    if summaries.get("investment_snapshot"):
                        generic["investment_snapshot"] = summaries["investment_snapshot"]
                    if summaries.get("portfolio_summary"):
                        generic["portfolio_summary"] = summaries["portfolio_summary"]
                    if summaries.get("provenance"):
                        generic["provenance"] = summaries["provenance"]
                if not generic:
                    generic = {"error": "Unknown document type"}
                extracted_data[f"Document {idx+1} ({doc_type})"] = generic
                
        except Exception as e:
            extracted_data[f"Document {idx+1}"] = {"error": f"Failed to process file: {str(e)}"}

    # If questionnaire_id is present, DO NOT generate the plan here.
    # Only link uploads and return insights/prefill; plan should be generated after questionnaire submission.
    if questionnaire_id:
        try:
            doc_insights = aggregate_doc_insights_for_questionnaire(questionnaire_id)
        except Exception as e:
            doc_insights = {"error": str(e)}
        # Build lightweight prefill suggestions to allow frontend to pre-populate fields
        try:
            prefill = build_prefill_from_insights(questionnaire_id)
        except Exception as e:
            prefill = {"error": str(e)}
        return jsonify({
            "summary_pdf_url": None,  # no summary when using questionnaire flow
            "debug_shas": debug_shas,
            "questionnaire_id": questionnaire_id,
            "docInsights": doc_insights,
            "prefill": prefill
        }), 200

    # Fallback to legacy summary report if no questionnaire_id provided
    pdf_filename = f"PortfolioSummary_{os.urandom(8).hex()}.pdf"
    pdf_path = os.path.join(OUTPUT_DIR, pdf_filename)
    generate_pdf_summary(extracted_data, pdf_path)

    if STORE_REPORTS == "memory":
        with open(pdf_path, "rb") as f:
            data = f.read()
        token = _register_temp_download(data, pdf_filename)
        try:
            os.remove(pdf_path)
        except Exception:
            pass
        download_url = url_for('download_temp', token=token, _external=True)
    else:
        download_url = url_for('download_file', filename=pdf_filename, _external=True)
    return jsonify({
        "summary_pdf_url": download_url,
        "debug_shas": debug_shas
    }), 200

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

@app.route('/download-temp/<token>')
def download_temp(token):
    item = TEMP_REPORTS.pop(token, None)
    if not item:
        return jsonify({"error": "not found"}), 404
    filename, data, mimetype = item
    return send_file(io.BytesIO(data), mimetype=mimetype, as_attachment=True, download_name=filename)

# Debug endpoints to inspect indexing and extracted metrics
@app.route('/debug/docs/<sha>', methods=['GET'])
def debug_list_sections(sha):
    row = get_document_by_sha(sha)
    if not row:
        return jsonify({"error": "not found"}), 404
    doc_id = row["id"]
    secs = [dict(r) for r in list_sections(doc_id)]
    return jsonify({
        "sha": sha,
        "filename": row["filename"],
        "page_count": row["page_count"],
        "sections": secs
    }), 200

@app.route('/debug/docs/<sha>/metrics', methods=['GET'])
def debug_list_metrics(sha):
    row = get_document_by_sha(sha)
    if not row:
        return jsonify({"error": "not found"}), 404
    doc_id = row["id"]
    mets = [dict(r) for r in list_metrics(doc_id)]
    return jsonify({
        "sha": sha,
        "metrics": mets
    }), 200

# --- Questionnaire & Financial Plan Endpoints ---

def _questionnaire_section_saver(section: str):
    mapping = {
        "personal_info": save_personal_info,
        "family_info": save_family_info,
        "goals": save_goals,
        "risk_profile": save_risk_profile,
        "insurance": save_insurance,
        "estate": save_estate,
        "lifestyle": save_lifestyle,
    }
    return mapping.get(section)

def _normalize_risk_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}
    out = {}
    # Enumerated fields normalization (lowercase)
    def _norm_enum(val, allowed):
        if not isinstance(val, str):
            return None
        v = val.strip().lower()
        return v if v in allowed else None

    out["primary_horizon"] = _norm_enum(payload.get("primary_horizon"), ["short","medium","med","long"])
    # numeric horizons
    def _num(v, lo=None, hi=None):
        try:
            if v in (None, "", "N/A"):
                return None
            n = float(v)
            if lo is not None: n = max(lo, n)
            if hi is not None: n = min(hi, n)
            return n
        except Exception:
            return None

    out["primary_horizon_years"] = _num(payload.get("primary_horizon_years"), 0, 100)
    out["goal_tenure_years"] = _num(payload.get("goal_tenure_years"), 0, 100)

    out["goal_flexibility"] = _norm_enum(payload.get("goal_flexibility"), ["critical","fixed","flexible"])
    out["goal_importance"] = _norm_enum(payload.get("goal_importance"), ["essential","important","lifestyle"])
    out["behavior"] = _norm_enum(payload.get("behavior"), ["sell","reduce","hold","buy","aggressive buy","aggressive_buy"])
    out["income_stability"] = _norm_enum(payload.get("income_stability"), ["very unstable","unstable","average","stable","very stable"])

    out["loss_tolerance_percent"] = _num(payload.get("loss_tolerance_percent"), 0, 100)
    out["emergency_fund_months"] = _num(payload.get("emergency_fund_months"), 0, 240)

    # Pass through tolerance if present
    out["tolerance"] = _norm_enum(payload.get("tolerance"), ["low","medium","med","high"])

    # Preserve original raw values for anything not normalized (optional)
    for k, v in payload.items():
        if k not in out:
            out[k] = v
    return out

def _validate_risk_payload(payload: dict):
    errs = []
    if not isinstance(payload, dict):
        return ["invalid_payload_type"]
    # Percentage fields
    for f in ["loss_tolerance_percent", "equity_allocation_percent"]:
        v = payload.get(f)
        if v not in (None, ""):
            try:
                fv = float(v)
                if fv < 0 or fv > 100:
                    errs.append(f"{f}:out_of_range")
            except Exception:
                errs.append(f"{f}:not_numeric")
    # Horizon years
    hy = payload.get("primary_horizon_years")
    if hy not in (None, ""):
        try:
            hv = float(hy)
            if hv < 0 or hv > 100:
                errs.append("primary_horizon_years:out_of_range")
        except Exception:
            errs.append("primary_horizon_years:not_numeric")
    # Emergency fund months
    efm = payload.get("emergency_fund_months")
    if efm not in (None, ""):
        try:
            emv = float(efm)
            if emv < 0 or emv > 240:
                errs.append("emergency_fund_months:out_of_range")
        except Exception:
            errs.append("emergency_fund_months:not_numeric")
    # Enum validations
    enums = {
        "goal_importance": {"essential","important","lifestyle"},
        "goal_flexibility": {"critical","fixed","flexible"},
        "behavior": {"sell","reduce","hold","buy","aggressive buy","aggressive_buy"},
        "income_stability": {"very unstable","unstable","average","stable","very stable"},
        "tolerance": {"low","medium","med","high"},
        "primary_horizon": {"short","medium","med","long"},
    }
    for k, allowed in enums.items():
        v = payload.get(k)
        if v not in (None, "") and str(v).lower() not in allowed:
            errs.append(f"{k}:invalid_value")
    return errs

@app.route("/questionnaire/start", methods=["POST"])
def questionnaire_start():
    data = request.get_json(force=True) or {}
    user_id = data.get("user_id") or "user"
    qid = create_questionnaire(user_id=user_id)
    return jsonify({"questionnaire_id": qid}), 201

@app.route("/questionnaire/<int:qid>/<section>", methods=["PUT"])
def questionnaire_save_section(qid: int, section: str):
    saver = _questionnaire_section_saver(section)
    if not saver:
        return jsonify({"error": "Unknown section"}), 400
    payload = request.get_json(force=True) or {}

    # Server-side normalization + validation for risk_profile section
    if section == "risk_profile":
        payload = _normalize_risk_payload(payload)
        errors = _validate_risk_payload(payload)
        if errors:
            return jsonify({"error": "validation_failed", "fields": errors}), 400

    saver(qid, payload)
    update_questionnaire_status(qid, "in_progress")
    return jsonify({"status": "ok"}), 200

@app.route("/questionnaire/<int:qid>", methods=["GET"])
def questionnaire_get(qid: int):
    q = get_questionnaire(qid)
    if not q:
        return jsonify({"error": "not found"}), 404
    return jsonify(q), 200

@app.route("/questionnaire/<int:qid>/prefill", methods=["GET"])
def questionnaire_prefill(qid: int):
    """
    Provide prefill suggestions derived from uploaded documents for this questionnaire.
    Frontend can use these to pre-populate fields and allow user edits.
    """
    try:
        prefill = build_prefill_from_insights(qid)
        return jsonify(prefill), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _assemble_financial_inputs(q: dict, doc_insights=None) -> dict:
    personal = q.get("personal_info") or {}
    family = q.get("family_info") or {}
    goals = q.get("goals") or {}
    risk = q.get("risk_profile") or {}
    insurance = q.get("insurance") or {}
    lifestyle = q.get("lifestyle") or {}

    payload = {
        "personal": {
            "age": personal.get("age"),
            "name": personal.get("name"),
        },
        "income": {
            "annualIncome": lifestyle.get("annual_income"),
            "monthlyExpenses": lifestyle.get("monthly_expenses"),
            "monthlyEmi": lifestyle.get("monthly_emi"),
        },
        "goals": {
            "goalHorizon": risk.get("primary_horizon"),
            "items": goals.get("items") or [],
        },
        "risk": {
            "tolerance": risk.get("tolerance"),
            # Advanced risk raw inputs (persisted in risk_profile JSON)
            "primary_horizon": risk.get("primary_horizon"),
            "primary_horizon_years": risk.get("primary_horizon_years"),
            "goal_tenure_years": risk.get("goal_tenure_years"),
            "goal_flexibility": risk.get("goal_flexibility"),
            "goal_importance": risk.get("goal_importance"),
            "behavior": risk.get("behavior"),
            "loss_tolerance_percent": risk.get("loss_tolerance_percent"),
            "income_stability": risk.get("income_stability"),
            "emergency_fund_months": risk.get("emergency_fund_months"),
        },
        "insurance": {
            "lifeCover": insurance.get("life_cover"),
            "healthCover": insurance.get("health_cover"),
        },
        "savings": {
            "savingsPercent": lifestyle.get("savings_percent"),
            "savingsBand": lifestyle.get("savings_band"),
        },
        "investments": {
            "current": lifestyle.get("products") or [],
            "allocation": lifestyle.get("allocation") or {},
        },
        "emergencyFundAmount": lifestyle.get("emergency_fund"),
    }

    # Merge document-derived signals if questionnaire fields are missing
    di = doc_insights or {}
    bank = di.get("bank") or {}
    try:
        # Annual income fallback from bank inflows
        if not payload["income"].get("annualIncome"):
            inflow = bank.get("total_inflows")
            if isinstance(inflow, (int, float)) and inflow > 0:
                payload["income"]["annualIncome"] = inflow
        # Monthly expenses fallback from bank outflows
        if not payload["income"].get("monthlyExpenses"):
            outflow = bank.get("total_outflows")
            if isinstance(outflow, (int, float)) and outflow > 0:
                payload["income"]["monthlyExpenses"] = round(outflow / 12.0, 2)
        # Savings percent fallback from net cashflow
        if not payload["savings"].get("savingsPercent"):
            inflow = bank.get("total_inflows")
            outflow = bank.get("total_outflows")
            if isinstance(inflow, (int, float)) and inflow > 0 and isinstance(outflow, (int, float)):
                sp = max(0.0, (inflow - outflow)) / inflow * 100.0
                sp = max(0.0, min(100.0, sp))
                payload["savings"]["savingsPercent"] = round(sp, 2)
    except Exception:
        pass

    # Merge insurance covers from document insights when questionnaire values are absent
    try:
        ins = di.get("insurance") or {}
        sum_val = ins.get("sum_assured_or_insured")
        ins_type = str(ins.get("insurance_type") or "").lower()
        if isinstance(sum_val, (int, float)) and sum_val > 0:
            if not payload["insurance"].get("lifeCover") and ("life" in ins_type or "term" in ins_type or "ulip" in ins_type):
                payload["insurance"]["lifeCover"] = float(sum_val)
            if not payload["insurance"].get("healthCover") and ("health" in ins_type or "mediclaim" in ins_type):
                payload["insurance"]["healthCover"] = float(sum_val)
            # Unknown type: default to life cover if both missing
            if not payload["insurance"].get("lifeCover") and not payload["insurance"].get("healthCover"):
                payload["insurance"]["lifeCover"] = float(sum_val)
    except Exception:
        pass

    # Merge portfolio allocation from document insights (CAS) when missing
    try:
        port = di.get("portfolio") or {}
        alloc = payload.get("investments", {}).get("allocation") or {}
        def _set_if_missing(key, src_key):
            if alloc.get(key) in (None, "", 0):
                v = port.get(src_key)
                if isinstance(v, (int, float)) and v >= 0:
                    alloc[key] = float(v)
        _set_if_missing("equity", "equity")
        _set_if_missing("debt", "debt")
        _set_if_missing("gold", "gold")
        _set_if_missing("realEstate", "realEstate")
        _set_if_missing("insuranceLinked", "insuranceLinked")
        _set_if_missing("cash", "cash")
        payload["investments"]["allocation"] = alloc
    except Exception:
        pass

    return payload

def _build_client_facts(q: dict, analysis: dict, doc_insights=None) -> dict:
    personal = q.get("personal_info") or {}
    family = q.get("family_info") or {}
    lifestyle = q.get("lifestyle") or {}
    insurance = q.get("insurance") or {}
    goals_data = q.get("goals") or {}
    goals = goals_data.get("items") or []

    # Calculate dependents - children automatically count as financial dependents
    children = family.get("children") or []
    other_dependents = family.get("dependents") or []
    has_spouse = bool(family.get("spouse"))
    dependents_count = (1 if has_spouse else 0) + len(children) + len(other_dependents)
    
    # Financial dependents: children or explicitly marked dependents
    # If user has children, they automatically have financial dependents
    has_financial_dependents = len(children) > 0 or len(other_dependents) > 0 or family.get("has_financial_dependents", False)
    
    di = doc_insights or {}
    bank = di.get("bank") or {}
    portfolio = di.get("portfolio") or {}
    
    # Get CAS data for SIP commitments
    qid = q.get("id")
    cas_data = _get_cas_data_for_questionnaire(qid) if qid else {}
    
    # Calculate total monthly SIP from CAS SIP details
    total_monthly_sip = 0.0
    sip_details = cas_data.get("sip_details") or []
    for sip in sip_details:
        freq = (sip.get("frequency") or "Monthly").lower()
        amount = sip.get("sip_amount", 0)
        if freq == "monthly" and isinstance(amount, (int, float)):
            total_monthly_sip += float(amount)
    
    # Enrich portfolio with SIP data
    enriched_portfolio = dict(portfolio)
    enriched_portfolio["total_monthly_sip"] = total_monthly_sip
    enriched_portfolio["sip_count"] = len(sip_details)
    
    # Calculate retirement planning if enabled
    age = personal.get("age")
    annual_income = lifestyle.get("annual_income")
    monthly_income = _safe_float(annual_income, 0.0) / 12.0 if annual_income else 0.0
    monthly_expenses = _safe_float(lifestyle.get("monthly_expenses"), 0.0)
    
    retirement_planning = None
    if goals_data.get("wants_retirement_planning"):
        desired_pension = goals_data.get("desired_monthly_pension")
        retirement_planning = compute_retirement_corpus(
            age=age,
            monthly_income=monthly_income,
            desired_monthly_pension=desired_pension
        )
        retirement_planning["enabled"] = True
        retirement_planning["monthly_expenses"] = monthly_expenses
    
    # Calculate term insurance requirement if has financial dependents
    term_insurance = None
    if has_financial_dependents and monthly_income > 0:
        required_cover = compute_term_insurance_need(age, monthly_income)
        current_life_cover = _safe_float(insurance.get("life_cover"), 0.0)
        term_insurance = {
            "has_financial_dependents": True,
            "required_cover": required_cover,
            "current_cover": current_life_cover,
            "gap": max(0, required_cover - current_life_cover),
            "is_adequate": current_life_cover >= required_cover,
        }

    facts = {
        "questionnaire_id": q.get("id"),
        "personal": {
            "name": personal.get("name"),
            "age": personal.get("age"),
            "dependents_count": dependents_count,
            "has_financial_dependents": has_financial_dependents,
        },
        "income": {
            "annualIncome": lifestyle.get("annual_income"),
            "monthlyExpenses": lifestyle.get("monthly_expenses"),
            "monthlyEmi": lifestyle.get("monthly_emi"),
        },
        "insurance": {
            "lifeCover": insurance.get("life_cover"),
            "healthCover": insurance.get("health_cover"),
        },
        "savings": {
            "savingsPercent": lifestyle.get("savings_percent"),
        },
        "goals": [
            {
                "name": (g.get("name") or g.get("goal")),
                "target_amount": g.get("target_amount"),
                "horizon_years": g.get("horizon_years") or g.get("horizon"),
                "risk_tolerance": g.get("risk_tolerance", "medium"),
                "goal_importance": g.get("goal_importance", "important"),
                "goal_flexibility": g.get("goal_flexibility", "fixed"),
                "behavior": g.get("behavior", "hold"),
            }
            for g in goals
        ],
        "bank": {
            "total_inflows": bank.get("total_inflows"),
            "total_outflows": bank.get("total_outflows"),
            "net_cashflow": bank.get("net_cashflow"),
            "opening_balance": bank.get("opening_balance"),
            "closing_balance": bank.get("closing_balance"),
        },
        "portfolio": enriched_portfolio,
        "extracts": di.get("raw_extracts"),
        "analysis": analysis,
        "retirement_planning": retirement_planning,
        "term_insurance": term_insurance,
    }
    return facts


def _render_narrative_section(story, styles, section_key, section_obj):
    try:
        title = section_obj.get("title") or section_key.replace("_", " ").title()
        story.append(Paragraph(sanitize_pdf_text(title), styles["h2"]))
        for b in (section_obj.get("bullets") or [])[:6]:
            story.append(Paragraph(sanitize_pdf_text(f"• {b}"), styles["BodyText"]))
        for p in (section_obj.get("paragraphs") or [])[:3]:
            story.append(Paragraph(sanitize_pdf_text(p), styles["BodyText"]))
        actions = section_obj.get("actions") or []
        if actions:
            story.append(Paragraph("Actions:", styles["BodyText"]))
            for a in actions[:6]:
                story.append(Paragraph(sanitize_pdf_text(f"- {a}"), styles["BodyText"]))
        story.append(Spacer(1, 8))
    except Exception:
        pass

def generate_financial_plan_pdf(q: dict, analysis: dict, output_path: str, doc_insights=None, narratives=None):
    styles = get_custom_styles()
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            rightMargin=inch*0.75, leftMargin=inch*0.75,
                            topMargin=inch, bottomMargin=inch)
    story = []

    name = (q.get("personal_info") or {}).get("name") or "Client"
    age = (q.get("personal_info") or {}).get("age") or "N/A"
    family = q.get("family_info") or {}
    spouse = family.get("spouse")
    children = family.get("children") or []
    dependents = family.get("dependents") or []
    goals = (q.get("goals") or {}).get("items") or []
    lifestyle = (q.get("lifestyle") or {}) or {}
    di = doc_insights or {}
    bank = di.get("bank") or {}
    portfolio = (di.get("portfolio") if di else None) or {}
    ihs = analysis.get("ihs") or {}
    advanced_risk = analysis.get("advancedRisk")
    categorized = _categorize_recommendations(analysis.get("recommendations") or [])

    # ==========================================================================
    # PAGE 1: INPUTS & DATA CAPTURED
    # ==========================================================================
    story.append(Paragraph("Financial Plan", styles["Title"]))
    story.append(Paragraph("Page 1: Inputs & Data Captured", styles["h1"]))

    # Client Profile Section
    story.append(Paragraph("Client Profile", styles["h2"]))
    profile_rows = [
        ["Name", name],
        ["Age", str(age)],
        ["Marital Status", "Married" if spouse else "Single"],
        ["Children", str(len(children))],
        ["Other Dependents", str(len(dependents))],
    ]
    profile_table = Table(
        [["Field", "Value"]] + profile_rows,
        hAlign="LEFT",
        colWidths=[180, 320],
    )
    profile_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#457B9D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('BOTTOMPADDING',(0,0),(-1,0),6),
    ]))
    story.append(profile_table)
    story.append(Spacer(1, 12))

    # Data Sources Status
    story.append(Paragraph("Data Sources Status", styles["h2"]))
    income_declared = bool(lifestyle.get("annual_income"))
    expenses_declared = bool(lifestyle.get("monthly_expenses"))
    netcf = bank.get("net_cashflow")

    data_rows = [
        ["Income Declaration", "Provided" if income_declared else "Not declared (derived bands used)"],
        ["Expense Data", "Captured" if expenses_declared else "Not declared (emergency fund based)"],
        ["Cashflow Pattern", f"{'Surplus' if netcf >= 0 else 'Deficit'} pattern" if netcf is not None else "Cannot be inferred"],
    ]

    # Document Upload Status
    uploads = list_questionnaire_uploads(q.get("id"))
    present_types = {row["doc_type"] for row in uploads}
    expected_types = {
        "Bank statement",
        "ITR",
        "Insurance document",
        "Mutual fund CAS (Consolidated Account Statement)",
    }
    missing = expected_types - present_types

    for doc_type in sorted(expected_types):
        status = "Uploaded" if doc_type in present_types else "Missing"
        data_rows.append([doc_type, status])

    data_table = Table(
        [["Data Source", "Status"]] + data_rows,
        hAlign="LEFT",
        colWidths=[280, 220],
    )
    data_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#457B9D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('BOTTOMPADDING',(0,0),(-1,0),6),
    ]))
    story.append(data_table)
    story.append(Spacer(1, 12))

    # Portfolio Snapshot (from CAS)
    cas_data = _get_cas_data_for_questionnaire(q.get("id"))

    if cas_data and (cas_data.get("transaction_summary") or cas_data.get("sip_details") or cas_data.get("investment_snapshot") or cas_data.get("asset_allocation")):
        story.append(Paragraph("Portfolio Snapshot (Mutual Fund CAS)", styles["h2"]))

        # Get data from transaction_summary or fallback to investment_snapshot
        trans_sum = cas_data.get("transaction_summary") or {}
        inv_snapshot = cas_data.get("investment_snapshot") or {}
        
        # Use transaction_summary first, fallback to investment_snapshot
        total_inv = trans_sum.get("total_purchase_amount") or inv_snapshot.get("investment") or inv_snapshot.get("net_investment") or 0
        current_val = trans_sum.get("total_current_value") or inv_snapshot.get("current_value") or 0
        unrealized_gain = trans_sum.get("total_unrealized_gain") or inv_snapshot.get("net_gain") or 0
        xirr_pct = inv_snapshot.get("xirr_percent")

        portfolio_rows = []
        if total_inv:
            portfolio_rows.append(["Total Investment", f"Rs. {float(total_inv):,.0f}"])
        if current_val:
            portfolio_rows.append(["Current Value", f"Rs. {float(current_val):,.0f}"])
        if unrealized_gain is not None and unrealized_gain != 0:
            gain_val = float(unrealized_gain)
            gain_text = f"Rs. {abs(gain_val):,.0f} ({'+' if gain_val >= 0 else '-'})"
            portfolio_rows.append(["Net Gain/Loss", gain_text])

        # Returns percentage
        if total_inv and float(total_inv) > 0 and current_val:
            returns_pct = ((float(current_val) - float(total_inv)) / float(total_inv)) * 100
            portfolio_rows.append(["Absolute Returns", f"{returns_pct:+.1f}%"])
        
        # XIRR if available
        if xirr_pct is not None:
            portfolio_rows.append(["XIRR", f"{float(xirr_pct):.1f}%"])

        if portfolio_rows:
            portfolio_table = Table(
                [["Metric", "Value"]] + portfolio_rows,
                hAlign="LEFT",
                colWidths=[200, 300],
            )
            portfolio_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#457B9D')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                ('BOTTOMPADDING',(0,0),(-1,0),6),
            ]))
            story.append(portfolio_table)
            story.append(Spacer(1, 8))

        # Asset Allocation & Debt-Equity Ratio
        asset_alloc = cas_data.get("asset_allocation") or {}
        # Fallback to portfolio allocation from doc_insights if CAS data is empty
        if not asset_alloc and portfolio:
            asset_alloc = {
                "equity_percentage": portfolio.get("equity", 0),
                "debt_percentage": portfolio.get("debt", 0),
            }
        equity_pct = asset_alloc.get("equity_percentage", 0)
        debt_pct = asset_alloc.get("debt_percentage", 0)
        hybrid_pct = asset_alloc.get("hybrid_percentage", 0)

        if equity_pct or debt_pct or hybrid_pct:
            alloc_rows = [
                ["Equity", f"{equity_pct:.1f}%"],
                ["Debt", f"{debt_pct:.1f}%"],
                ["Hybrid", f"{hybrid_pct:.1f}%"],
            ]

            # Total AUM if available
            total_aum = asset_alloc.get("total_aum")
            if total_aum:
                alloc_rows.insert(0, ["Total AUM", f"Rs. {float(total_aum):,.0f}"])

            # Debt-Equity Ratio
            if equity_pct > 0:
                de_ratio = debt_pct / equity_pct
                alloc_rows.append(["Debt-Equity Ratio", f"{de_ratio:.2f}:1"])

            alloc_table = Table(
                [["Asset Class", "Allocation"]] + alloc_rows,
                hAlign="LEFT",
                colWidths=[200, 300],
            )
            alloc_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#457B9D')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                ('BOTTOMPADDING',(0,0),(-1,0),6),
            ]))
            story.append(alloc_table)
            story.append(Spacer(1, 8))

        # SIP Summary
        sip_details = cas_data.get("sip_details") or []
        if sip_details:
            story.append(Paragraph("Active SIPs", styles["BodyText"]))
            story.append(Spacer(1, 4))

            sip_rows = []
            total_monthly_sip = 0

            for sip in sip_details[:10]:  # Limit to 10 SIPs
                scheme = sip.get("scheme_name", "N/A")
                amount = sip.get("sip_amount", 0)
                date = sip.get("sip_date", "N/A")
                freq = sip.get("frequency", "Monthly")

                # Truncate long scheme names
                if len(scheme) > 40:
                    scheme = scheme[:37] + "..."

                sip_rows.append([
                    scheme,
                    f"Rs. {amount:,.0f}" if isinstance(amount, (int, float)) else str(amount),
                    str(date),
                    freq
                ])

                # Sum monthly SIPs
                if freq.lower() == "monthly" and isinstance(amount, (int, float)):
                    total_monthly_sip += amount

            if sip_rows:
                sip_table = Table(
                    [["Scheme", "Amount", "Date", "Frequency"]] + sip_rows,
                    hAlign="LEFT",
                    colWidths=[240, 90, 60, 110],
                )
                sip_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#457B9D')),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                    ('FONTSIZE',(0,0),(-1,-1), 8),
                    ('BOTTOMPADDING',(0,0),(-1,0),6),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ]))
                story.append(sip_table)

                if total_monthly_sip > 0:
                    story.append(Spacer(1, 4))
                    story.append(Paragraph(
                        f"<b>Total Monthly SIP Commitment:</b> Rs. {total_monthly_sip:,.0f}",
                        styles["BodyText"]
                    ))

        story.append(Spacer(1, 12))

    # Goals Captured
    story.append(Paragraph("Goals Captured", styles["h2"]))
    if goals:
        goal_input_rows = []
        for g in goals[:10]:
            desc = g.get("name") or g.get("goal") or "Goal"
            amt = g.get("target_amount") or "-"
            horizon = g.get("horizon_years") or g.get("horizon") or "-"
            goal_input_rows.append([desc, f"Rs. {amt}" if amt != "-" else "-", f"{horizon} yrs" if horizon != "-" else "-"])
        goals_table = Table(
            [["Goal Name", "Target Amount", "Horizon"]] + goal_input_rows,
            hAlign="LEFT",
            colWidths=[240, 130, 130],
        )
        goals_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#457B9D')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('BOTTOMPADDING',(0,0),(-1,0),6),
        ]))
        story.append(goals_table)
    else:
        story.append(Paragraph("No goals entered.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    # Retirement Planning Section (if enabled)
    goals_data = q.get("goals") or {}
    if goals_data.get("wants_retirement_planning"):
        story.append(Paragraph("Retirement Planning", styles["h2"]))
        
        age = _safe_float((q.get("personal_info") or {}).get("age"), 30)
        annual_income = lifestyle.get("annual_income")
        monthly_income = _safe_float(annual_income, 0.0) / 12.0 if annual_income else 0.0
        monthly_expenses = _safe_float(lifestyle.get("monthly_expenses"), 0.0)
        desired_pension = goals_data.get("desired_monthly_pension")
        
        retirement_data = compute_retirement_corpus(
            age=age,
            monthly_income=monthly_income,
            desired_monthly_pension=desired_pension
        )
        
        retirement_rows = [
            ["Current Age", f"{int(age)} years"],
            ["Years to Retirement", f"{int(retirement_data.get('years_to_retirement', 0))} years"],
            ["Retirement Age", f"{retirement_data.get('retirement_age', 60)} years"],
            ["Standard Retirement Corpus", f"Rs. {retirement_data.get('standard_corpus', 0):,.0f}"],
        ]
        
        # Add pension-based calculations if provided
        if desired_pension:
            pension_val = _safe_float(desired_pension, 0)
            if pension_val > 0:
                pension_warning = ""
                if monthly_expenses > 0 and pension_val < monthly_expenses:
                    pension_warning = " ⚠️ Below current expenses"
                retirement_rows.append(["Desired Monthly Pension", f"Rs. {pension_val:,.0f}{pension_warning}"])
                retirement_rows.append(["Annual Pension Requirement", f"Rs. {retirement_data.get('pension_annual', 0):,.0f}"])
                retirement_rows.append(["Ideal Retirement Corpus", f"Rs. {retirement_data.get('pension_corpus', 0):,.0f}"])
        
        retirement_table = Table(
            [["Parameter", "Value"]] + retirement_rows,
            hAlign="LEFT",
            colWidths=[250, 250],
        )
        retirement_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2D6A4F')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('BOTTOMPADDING',(0,0),(-1,0),6),
        ]))
        story.append(retirement_table)
        story.append(Spacer(1, 12))

    # Term Insurance Section (if has financial dependents)
    children = family.get("children") or []
    other_dependents = family.get("dependents") or []
    has_financial_dependents = len(children) > 0 or len(other_dependents) > 0
    
    if has_financial_dependents:
        story.append(Paragraph("Term Insurance Requirement", styles["h2"]))
        
        age = _safe_float((q.get("personal_info") or {}).get("age"), 30)
        annual_income = lifestyle.get("annual_income")
        monthly_income = _safe_float(annual_income, 0.0) / 12.0 if annual_income else 0.0
        current_life_cover = _safe_float((q.get("insurance") or {}).get("life_cover"), 0.0)
        
        required_cover = compute_term_insurance_need(age, monthly_income)
        gap = max(0, required_cover - current_life_cover)
        is_adequate = current_life_cover >= required_cover
        
        dependents_text = f"{len(children)} children"
        if other_dependents:
            dependents_text += f", {len(other_dependents)} other dependents"
        
        term_rows = [
            ["Financial Dependents", dependents_text],
            ["Required Term Cover", f"Rs. {required_cover:,.0f}"],
            ["Current Life Cover", f"Rs. {current_life_cover:,.0f}"],
            ["Coverage Status", "✓ Adequate" if is_adequate else f"⚠️ Gap of Rs. {gap:,.0f}"],
        ]
        
        term_table = Table(
            [["Parameter", "Value"]] + term_rows,
            hAlign="LEFT",
            colWidths=[250, 250],
        )
        term_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#9D4B45') if not is_adequate else colors.HexColor('#457B9D')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('BOTTOMPADDING',(0,0),(-1,0),6),
        ]))
        story.append(term_table)
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "<i>Formula: (60-age) × monthly income × 12 (Human Life Value method). "
            "Term insurance is essential when you have financial dependents.</i>",
            styles["BodyText"]
        ))
        story.append(Spacer(1, 12))

    story.append(PageBreak())

    # ==========================================================================
    # PAGE 2: TECHNICAL ASSESSMENT (ALL VALUES)
    # ==========================================================================
    story.append(Paragraph("Page 2: Technical Assessment", styles["h1"]))

    # Core Financial Metrics Dashboard
    story.append(Paragraph("Core Financial Metrics", styles["h2"]))
    metrics_rows = [
        ["Surplus Level", analysis.get("surplusBand") or "-", Paragraph(_interpret_surplus(analysis.get("surplusBand")))],
        ["Insurance Coverage", analysis.get("insuranceGap") or "-", Paragraph(_interpret_insurance(analysis.get("insuranceGap")))],
        ["Debt Status", analysis.get("debtStress") or "-", Paragraph(_interpret_debt(analysis.get("debtStress")))],
        ["Liquidity", analysis.get("liquidity") or "-", Paragraph(_interpret_liquidity(analysis.get("liquidity")))],
        ["Investment Health Score", ihs.get("band") or "-", Paragraph(_interpret_ihs(ihs.get("band")))],
    ]
    table = Table(
        [["Metric", "Result", "Interpretation"]] + metrics_rows,
        hAlign="LEFT",
        colWidths=[140, 90, 270],
    )
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#457B9D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('BOTTOMPADDING',(0,0),(-1,0),8),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # Advanced Risk Assessment (if available)
    if advanced_risk:
        story.append(Paragraph("Advanced Risk Assessment", styles["h2"]))

        # Core risk values in a compact table
        band = advanced_risk.get("recommendedEquityBand") or {}
        equity_band_str = f"{band.get('min')}% - {band.get('max')}% (mid {advanced_risk.get('recommendedEquityMid')}%)" if band.get("min") is not None else "-"

        ar_rows = [
            ["Calculated Score", str(advanced_risk.get("score") or "-")],
            ["Risk Appetite", advanced_risk.get("appetiteCategory") or "-"],
            ["Tenure Limit", advanced_risk.get("tenureLimitCategory") or "-"],
            ["Baseline Category", advanced_risk.get("baselineCategory") or "-"],
            ["Final Category", advanced_risk.get("finalCategory") or "-"],
            ["Recommended Equity Band", equity_band_str],
        ]

        ar_table = Table([["Parameter", "Value"]] + ar_rows, hAlign="LEFT", colWidths=[200, 300])
        ar_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1D3557')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('BOTTOMPADDING',(0,0),(-1,0),6),
        ]))
        story.append(ar_table)
        story.append(Spacer(1, 8))

        # Adjustments applied
        adjustments_list = advanced_risk.get("adjustmentsApplied") or []
        if adjustments_list:
            story.append(Paragraph("Adjustments Applied:", styles["BodyText"]))
            for adj in adjustments_list:
                story.append(Paragraph(sanitize_pdf_text(f"  - {adj}"), styles["BodyText"]))

        # Reasoning
        reasoning = advanced_risk.get("reasoningText")
        if reasoning:
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"Reasoning: {sanitize_pdf_text(reasoning)}", styles["BodyText"]))

    story.append(Spacer(1, 12))

    # Summary Metrics Box
    story.append(Paragraph("Assessment Summary", styles["h2"]))
    summary_rows = [
        ["Risk Profile", analysis.get('riskProfile') or "-"],
        ["Surplus Level", analysis.get('surplusBand') or "-"],
        ["Insurance Status", analysis.get('insuranceGap') or "-"],
        ["Debt Position", analysis.get('debtStress') or "-"],
        ["Liquidity", analysis.get('liquidity') or "-"],
        ["IHS Band", ihs.get('band') or "-"],
    ]
    summary_table = Table(
        [["Assessment", "Result"]] + summary_rows,
        hAlign="LEFT",
        colWidths=[200, 300],
    )
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#457B9D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('BOTTOMPADDING',(0,0),(-1,0),6),
    ]))
    story.append(summary_table)
    story.append(PageBreak())

    # ==========================================================================
    # PAGE 3: RECOMMENDATIONS & ACTIONS
    # ==========================================================================
    story.append(Paragraph("Page 3: Recommendations & Actions", styles["h1"]))

    # Flags / Attention Areas
    flags = analysis.get("flags") or []
    story.append(Paragraph("Attention Areas", styles["h2"]))
    if flags:
        def _format_rupee_match(match):
            """Convert matched rupee amount to compact Indian format (L/Cr)."""
            full_match = match.group(0)
            # Extract just the number part, removing Rs., ₹, spaces, and commas
            num_str = re.sub(r"[Rs\.₹,\s]", "", full_match)
            try:
                amount = float(num_str)
                return f"Rs. {_format_indian_amount(amount)}"
            except ValueError:
                return full_match  # Return original if parsing fails
        
        for f in flags:
            # Format actual rupee amounts (with Rs. or ₹ prefix) to compact Indian style (L/Cr)
            sanitized = re.sub(r"(Rs\.?|₹)\s*[0-9][0-9,]*\.?\d*", _format_rupee_match, f)
            story.append(Paragraph(sanitize_pdf_text(f"! {sanitized}"), styles["BodyText"]))
    else:
        story.append(Paragraph("No critical flags identified.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    # Categorized Recommendations
    story.append(Paragraph("Recommendations", styles["h2"]))
    for cat, items in categorized.items():
        story.append(Paragraph(f"<b>{cat}</b>", styles["BodyText"]))
        for it in items:
            story.append(Paragraph(sanitize_pdf_text(f"  - {it}"), styles["BodyText"]))
        story.append(Spacer(1, 4))
    story.append(Spacer(1, 12))

    # Personalized Narrative (LLM-generated)
    if narratives and isinstance(narratives, dict):
        story.append(Paragraph("Personalized Narrative", styles["h2"]))
        section_order = [
            "executive_summary",
            "flags_explainer",
            "protection_plan",
            "cashflow",
            "debt_strategy",
            "liquidity_plan",
            "risk_rationale",
            "goals_strategy",
            "portfolio_rebalance",
        ]
        for key in section_order:
            sec = narratives.get(key)
            if isinstance(sec, dict):
                _render_narrative_section(story, styles, key, sec)
        story.append(Spacer(1, 12))

    # Goal Mapping Table (compact)
    story.append(Paragraph("Goal Strategy Mapping", styles["h2"]))
    goal_rows = []
    for g in goals[:10]:
        nm = g.get("name") or g.get("goal") or "Goal"
        horizon = g.get("horizon_years") or g.get("horizon") or "-"
        strategy = g.get("suggested_strategy") or _default_strategy_for_goal(g)
        goal_rows.append([nm, f"{horizon}", Paragraph(strategy, styles["BodyText"])])
    if goal_rows:
        g_table = Table(
            [["Goal", "Horizon", "Suggested Strategy"]] + goal_rows,
            hAlign="LEFT",
            colWidths=[150, 60, 290],
        )
        g_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1D3557')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('BOTTOMPADDING',(0,0),(-1,0),6),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        story.append(g_table)
    else:
        story.append(Paragraph("No goals recorded.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    # Disclaimers
    story.append(Paragraph("Disclaimers", styles["h2"]))
    disclaimers = [
        "Indicative plan; not a legally binding advisory document.",
        "Market and regulatory changes can impact recommendations.",
        "Revisit annually or after major life events.",
    ]
    for d in disclaimers:
        story.append(Paragraph(f"- {d}", styles["BodyText"]))

    class FPCanvas(canvas.Canvas):
        page_fn = footer
    doc.build(story, onFirstPage=header, onLaterPages=header, canvasmaker=FPCanvas)

# --- Narrative helpers for qualitative interpretations ---

def _interpret_surplus(band: str):
    m = (band or "").lower()
    if m == "low":
        return "Limited savings capacity; prioritize increasing systematic savings."
    if m == "adequate":
        return "Reasonable surplus; allocate efficiently across priority goals."
    if m == "strong":
        return "Healthy surplus enabling acceleration of long-term goals."
    return "Surplus position unclear; gather more cashflow data."

def _interpret_insurance(status: str):
    s = (status or "").lower()
    if s == "underinsured":
        return "Protection gap exists; increase term coverage to benchmark (~10x income)."
    if s == "adequate":
        return "Coverage at or above benchmark; schedule periodic review."
    return "Coverage status indeterminate; verify policy details."

def _interpret_debt(status: str):
    s = (status or "").lower()
    if s == "stressed":
        return "Debt ratio elevated; restructure or accelerate repayments."
    if s == "moderate":
        return "Manageable leverage; monitor to avoid escalation."
    if s == "healthy":
        return "Debt load within prudent limits."
    return "Debt position unclear; capture EMI details."

def _interpret_liquidity(status: str):
    s = (status or "").lower()
    if s == "insufficient":
        return "Emergency reserves below 6 months; build liquid buffer."
    if s == "adequate":
        return "Emergency reserve at guideline level."
    return "Liquidity status unconfirmed; record emergency fund amount."

def _interpret_ihs(band: str):
    b = (band or "").lower()
    if b == "poor":
        return "Structure weak; raise savings rate and diversify holdings."
    if b == "average":
        return "Improve allocation balance and goal alignment."
    if b == "good":
        return "Solid foundation; refine tax and risk efficiency."
    if b == "excellent":
        return "Optimized; maintain discipline and periodic rebalancing."
    return "Investment health unclear; collect product and allocation details."

def _categorize_recommendations(recs):
    cats = {
        "Risk Profile Based Advice": [],
        "Insurance Improvement": [],
        "Debt Optimization": [],
        "Liquidity Management": [],
        "Investment Health": [],
        "General Planning": [],
    }
    for r in recs:
        rl = r.lower()
        if any(k in rl for k in ["equity", "allocation", "volatility", "mix"]):
            cats["Risk Profile Based Advice"].append(r)
        elif any(k in rl for k in ["cover", "insurance", "term", "health"]):
            cats["Insurance Improvement"].append(r)
        elif any(k in rl for k in ["debt", "emi", "borrow", "refinance"]):
            cats["Debt Optimization"].append(r)
        elif any(k in rl for k in ["liquid", "emergency"]):
            cats["Liquidity Management"].append(r)
        elif any(k in rl for k in ["savings", "diversify", "portfolio", "rebalance", "products"]):
            cats["Investment Health"].append(r)
        else:
            cats["General Planning"].append(r)
    return {k: v for k, v in cats.items() if v}

def _default_strategy_for_goal(g):
    horizon = g.get("horizon_years") or g.get("horizon")
    try:
        h = int(horizon)
    except Exception:
        h = None
    if h is None:
        return "Clarify horizon; then assign blend of equity/debt."
    if h <= 3:
        return "High-quality debt & liquid funds."
    if h <= 7:
        return "Balanced allocation (~50-60% equity)."
    if h > 7:
        return "Growth-oriented (~70-80% equity) with periodic rebalancing."
    return "Diversified approach."

@app.route("/report/generate", methods=["POST"])
def report_generate():
    data = request.get_json(force=True) or {}
    questionnaire_id = data.get("questionnaire_id")
    if not questionnaire_id:
        return jsonify({"error": "questionnaire_id required"}), 400
    q = get_questionnaire(questionnaire_id)
    if not q:
        return jsonify({"error": "questionnaire not found"}), 404

    # Merge questionnaire + linked document insights
    doc_insights = {}
    try:
        doc_insights = aggregate_doc_insights_for_questionnaire(questionnaire_id)
    except Exception as e:
        doc_insights = {"error": str(e)}

    inputs = _assemble_financial_inputs(q, doc_insights)
    analysis = analyze_financial_health(inputs)

    sections = None
    try:
        facts = _build_client_facts(q, analysis, doc_insights)
        sections = run_report_sections(questionnaire_id, facts)
    except Exception as e:
        analysis["llm_error"] = str(e)

    pdf_filename = f"FinancialPlan_{os.urandom(8).hex()}.pdf"
    pdf_path = os.path.join(OUTPUT_DIR, pdf_filename)
    generate_financial_plan_pdf(q, analysis, pdf_path, doc_insights=doc_insights, narratives=sections)
    url = url_for("download_file", filename=pdf_filename, _external=True)
    return jsonify({
        "financial_plan_pdf_url": url,
        "summary_pdf_url": url,
        "report_type": "financial_plan",
        "questionnaire_id": questionnaire_id,
        "analysis": analysis,
        "docInsights": doc_insights,
        "sections": sections
    }), 200

# --- PDF Generation Logic (Enhanced) ---

def get_custom_styles():
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    title_style.fontSize = 22
    title_style.alignment = TA_CENTER
    title_style.spaceAfter = 20
    title_style.textColor = colors.HexColor('#1D3557')
    h1_style = styles['h1']
    h1_style.fontSize = 16
    h1_style.leading = 22
    h1_style.spaceAfter = 12
    h1_style.textColor = colors.HexColor('#457B9D')
    h2_style = styles['h2']
    h2_style.fontSize = 12
    h2_style.leading = 18
    h2_style.spaceBefore = 10
    h2_style.spaceAfter = 6
    h2_style.textColor = colors.HexColor('#E63946')
    body_style = styles['BodyText']
    body_style.fontSize = 10
    body_style.leading = 14
    body_style.spaceAfter = 6
    body_style.wordWrap = 'CJK'
    styles.add(ParagraphStyle(name='Header', fontSize=8, alignment=TA_RIGHT, textColor=colors.grey))
    styles.add(ParagraphStyle(name='Footer', fontSize=8, alignment=TA_CENTER, textColor=colors.grey))
    styles.add(ParagraphStyle(name='TableHead', fontSize=9, fontName='Helvetica-Bold', alignment=TA_LEFT, textColor=colors.white))
    styles.add(ParagraphStyle(name='TableCell', fontSize=9, alignment=TA_LEFT, wordWrap='CJK'))
    return styles

def header(canvas, doc):
    canvas.saveState()
    # Draw logo at top-left in the margin area (above content)
    if os.path.exists(LOGO_PATH):
        try:
            logo_width = 60
            logo_height = 30
            # Position logo in the top margin, well above the content area
            logo_y = doc.height + doc.topMargin + 15
            canvas.drawImage(LOGO_PATH, doc.leftMargin, logo_y, 
                           width=logo_width, height=logo_height, preserveAspectRatio=True, mask='auto')
        except Exception:
            pass  # Skip logo if there's an error loading it
    # Draw date at top-right in the margin
    styles = get_custom_styles()
    p = Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Header'])
    w, h = p.wrap(doc.width - 80, doc.topMargin)
    p.drawOn(canvas, doc.leftMargin + 80, doc.height + doc.topMargin + 20)
    canvas.restoreState()

def footer(canvas, doc):
    canvas.saveState()
    styles = get_custom_styles()
    
    # Educational disclaimer - displayed on every page
    disclaimer_text = (
        "This is an educational analysis tool, not financial advice. "
        "This report is for informational purposes only. "
        "Consult a SEBI-registered Investment Advisor before making decisions. "
        "We do not recommend specific securities or products."
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        fontSize=6,
        alignment=TA_CENTER,
        textColor=colors.grey,
        leading=8
    )
    disclaimer = Paragraph(disclaimer_text, disclaimer_style)
    dw, dh = disclaimer.wrap(doc.width, doc.bottomMargin)
    disclaimer.drawOn(canvas, doc.leftMargin, dh + 15)
    
    # Page number
    p = Paragraph(f"Page {doc.page}", styles['Footer'])
    w, h = p.wrap(doc.width, doc.bottomMargin)
    p.drawOn(canvas, doc.leftMargin, h)
    canvas.restoreState()

def build_pdf_story(story, data, doc, styles):
    empty_values = [None, "", "N/A", [], {}]
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        if key in ["transactions", "transactions_json_path"]:
            if key == "transactions_json_path" and isinstance(value, str):
                story.append(Paragraph(sanitize_pdf_text(f"Transactions JSON: {value}"), styles["BodyText"]))
                story.append(Spacer(1, 4))
            continue
        if value in empty_values or key in ['extraction_error', 'llm_error']:
            continue
        key_title = key.replace('_', ' ').title()
        if isinstance(value, list) and value and isinstance(value[0], dict):
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph(key_title, styles['h2']))
            add_dynamic_table(story, value, doc, styles)
        elif isinstance(value, dict):
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph(key_title, styles['h2']))
            build_pdf_story(story, value, doc, styles)
        else:
            formatted_value = f"Rs. {value:,.2f}" if isinstance(value, (int, float)) else value
            p_text = f"<b>{key_title}:</b> {formatted_value}"
            story.append(Paragraph(sanitize_pdf_text(p_text), styles["BodyText"]))
            story.append(Spacer(1, 4))

def add_dynamic_table(story, data_list, doc, styles):
    filtered_data = [row for row in data_list if any(v not in [None, "", "N/A"] for v in row.values())]
    if not filtered_data:
        return
    headers = list(filtered_data[0].keys())
    header_row = [Paragraph(sanitize_pdf_text(h.replace('_', ' ').title()), styles['TableHead']) for h in headers]
    table_data = [header_row]
    for row_dict in filtered_data:
        row = []
        for h in headers:
            cell_value = row_dict.get(h, "")
            if isinstance(cell_value, (int, float)):
                cell_value = f"Rs. {cell_value:,.2f}"
            row.append(Paragraph(sanitize_pdf_text(str(cell_value)), styles['TableCell']))
        table_data.append(row)
    col_count = len(headers) if headers else 1
    col_widths = [doc.width / col_count] * col_count
    table = Table(table_data, hAlign='LEFT', repeatRows=1, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#457B9D')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(table)

def generate_pdf_summary(summary_data, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            rightMargin=inch*0.75, leftMargin=inch*0.75,
                            topMargin=inch, bottomMargin=inch)
    styles = get_custom_styles()
    story = [Paragraph("Holistic Financial Report", styles['Title']), Spacer(1, 0.25 * inch)]
    for section_key, section_content in summary_data.items():
        if not section_content or section_content.get("error"):
            story.append(Paragraph(sanitize_pdf_text(f"Could not process: {section_key}"), styles['h1']))
            if isinstance(section_content, dict) and section_content.get("error"):
                story.append(Paragraph(f"Reason: {section_content['error']}", styles['BodyText']))
            story.append(Spacer(1, 0.2 * inch))
            continue
        story.append(Paragraph(sanitize_pdf_text(section_key.replace('_', ' ').title()), styles['h1']))
        story.append(Spacer(1, 0.15 * inch))
        try:
            from reportlab.graphics.shapes import Drawing
            from reportlab.graphics.charts.piecharts import Pie
            key_l = str(section_key).lower()
            def _add_pie_chart(title, data_map):
                items = []
                for k, v in (data_map or {}).items():
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if fv > 0:
                        items.append((str(k), fv))
                if not items:
                    return
                story.append(Paragraph(title, styles['h2']))
                d = Drawing(300, 220)
                pie = Pie()
                pie.x = 50
                pie.y = 10
                pie.width = 200
                pie.height = 200
                pie.data = [v for _, v in items]
                pie.labels = [k for k, _ in items]
                pie.simpleLabels = True
                pie.slices.strokeWidth = 0.5
                d.add(pie)
                story.append(d)
                story.append(Spacer(1, 0.1 * inch))
            if "bank statement" in key_l:
                acct = (section_content or {}).get("account_summary") or {}
                inflows = acct.get("total_inflows")
                outflows = acct.get("total_outflows")
                if inflows or outflows:
                    _add_pie_chart("Cash Flows (Inflows vs Outflows)", {
                        "Inflows": inflows or 0,
                        "Outflows": outflows or 0,
                    })
                    try:
                        if isinstance(inflows, (int, float)) and isinstance(outflows, (int, float)):
                            net = float(inflows) - float(outflows)
                            note = "Net Cash Surplus" if net >= 0 else "Net Cash Deficit"
                            story.append(Paragraph(f"{note}: Rs. {abs(net):,.0f}", styles['BodyText']))
                            story.append(Spacer(1, 0.1 * inch))
                    except Exception:
                        pass
            if "mutual fund cas" in key_l:
                alloc = (section_content or {}).get("asset_allocation") or {}
                if any(k in alloc for k in ["equity_percentage", "debt_percentage", "hybrid_percentage"]):
                    _add_pie_chart("Asset Allocation (CAS)", {
                        "Equity": alloc.get("equity_percentage", 0),
                        "Debt": alloc.get("debt_percentage", 0),
                        "Hybrid": alloc.get("hybrid_percentage", 0),
                    })
            if key_l.startswith("itr"):
                inc = (section_content or {}).get("income_sources") or {}
                if inc:
                    _add_pie_chart("Income Sources (ITR)", inc)
        except Exception:
            pass
        build_pdf_story(story, section_content, doc, styles)
        story.append(PageBreak())
    if story and isinstance(story[-1], PageBreak):
        story.pop()
    class CustomCanvas(canvas.Canvas):
        page_fn = footer
    doc.build(story, onFirstPage=header, onLaterPages=header, canvasmaker=CustomCanvas)

# --- Main Execution ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
