from flask import Flask, request, jsonify, g, url_for, send_from_directory, send_file
import pdfplumber
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Frame, PageTemplate, Image, Flowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
from reportlab.pdfgen import canvas
from extractors import index_and_extract, extract_and_store_from_indexed
from llm_sections import run_report_sections, compute_goal_sip, compute_regime_comparison, PriorityAllocationEngine
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
    get_user_by_firebase_uid,
    create_or_update_user,
    list_all_users,
    delete_user,
    set_user_credits,
    get_user_count,
    get_payment_history,
)

# Auth and Payment modules
from auth import require_auth, require_payment, consume_credit, verify_firebase_token, optional_auth, require_admin
from payment import (
    create_razorpay_order,
    verify_razorpay_signature,
    verify_webhook_signature,
    process_payment_captured,
    get_payment_status,
    get_report_price_paise,
)
# --- Logging Configuration ---
import logging
import uuid

# Configure basic logging (simplified for compatibility)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def _redact_pii(text: str) -> str:
    """Redact PII like PAN, account numbers, names from log output."""
    if not text or not isinstance(text, str):
        return str(text) if text else ""
    redacted = text
    # Redact PAN (format: ABCDE1234F)
    redacted = re.sub(r'\b[A-Z]{5}\d{4}[A-Z]\b', 'PAN-XXXX', redacted)
    # Redact account numbers (8-20 digits, possibly with X or *)
    redacted = re.sub(r'\b[Xx\*\d]{8,20}\b', 'ACCT-XXXX', redacted)
    # Redact Aadhaar (12 digits)
    redacted = re.sub(r'\b\d{4}\s?\d{4}\s?\d{4}\b', 'AADHAAR-XXXX', redacted)
    # Redact email addresses
    redacted = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 'EMAIL-REDACTED', redacted)
    # Redact phone numbers (10 digits, possibly with country code)
    redacted = re.sub(r'(\+91[-\s]?)?\b\d{10}\b', 'PHONE-XXXX', redacted)
    return redacted

def _log_extraction(doc_type: str, field: str, value, confidence: float = None, extra: dict = None):
    """Structured logging for extraction events with optional confidence."""
    log_data = {
        "doc_type": doc_type,
        "field": field,
        "value_preview": _redact_pii(str(value)[:100]) if value else None,
        "confidence": confidence
    }
    if extra:
        log_data.update(extra)
    logger.info(f"Extracted {field}: {log_data}")

# --- Extraction Confidence Scoring ---
def _create_extraction_meta(field: str, value, source: str, pattern_idx: int = 0, 
                            total_patterns: int = 1) -> dict:
    """
    Create confidence metadata for an extracted field.
    
    Args:
        field: Field name being extracted
        value: Extracted value
        source: Source of extraction ('regex', 'llm', 'fallback')
        pattern_idx: Index of matching pattern (0 = primary pattern, higher = fallback)
        total_patterns: Total number of patterns tried
    
    Returns:
        Dict with confidence score (0.0-1.0) and needs_review flag
    """
    if value is None or value == "N/A" or value == "":
        return {
            "confidence": 0.0,
            "source": source,
            "needs_review": True,
            "reason": "No value extracted"
        }
    
    # Base confidence based on source
    if source == "regex":
        # Primary patterns = high confidence, fallback patterns = decreasing confidence  
        base_confidence = 0.95 - (pattern_idx * 0.1)
    elif source == "llm":
        base_confidence = 0.80  # LLM extractions are less certain
    else:
        base_confidence = 0.60  # Fallback/derived values
    
    # Clamp to valid range
    confidence = max(0.1, min(0.99, base_confidence))
    
    return {
        "confidence": round(confidence, 2),
        "source": source,
        "needs_review": confidence < 0.70,
        "pattern_position": f"{pattern_idx + 1}/{total_patterns}" if source == "regex" else None
    }

def _add_extraction_metadata(data: dict, field_meta: dict) -> None:
    """
    Add extraction metadata to a data dict, creating _extraction_meta if needed.
    
    Args:
        data: The extraction result dict to modify
        field_meta: Dict mapping field names to their confidence metadata
    """
    if "_extraction_meta" not in data:
        data["_extraction_meta"] = {}
    data["_extraction_meta"].update(field_meta)

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

# --- Security: PDF Validation ---
MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024  # 10MB limit

def _validate_pdf_file(file_bytes: bytes, filename: str = "upload.pdf") -> tuple:
    """
    Validate a PDF file for security concerns.
    
    Args:
        file_bytes: Raw file content
        filename: Original filename for logging
    
    Returns:
        (is_valid, error_message) tuple
    """
    # Check file size
    if len(file_bytes) > MAX_PDF_SIZE_BYTES:
        return False, f"File too large ({len(file_bytes) / 1024 / 1024:.1f}MB). Maximum size is 10MB."
    
    if len(file_bytes) < 4:
        return False, "File is too small to be a valid PDF."
    
    # Check PDF magic bytes (%PDF)
    if not file_bytes[:4] == b'%PDF':
        return False, "Invalid PDF format. File does not start with PDF header."
    
    # Basic check for PDF end marker (%%EOF)
    # Note: Some PDFs have trailing content after %%EOF, so we check the last 1024 bytes
    if b'%%EOF' not in file_bytes[-1024:]:
        logger.warning(f"PDF {_redact_pii(filename)} may be incomplete (no %%EOF marker found)")
        # Don't fail validation, just log warning - some PDFs are valid without this
    
    return True, None

def _sanitize_text_for_llm(text: str, max_length: int = 20000) -> str:
    """
    Sanitize text before sending to LLM to prevent prompt injection.
    
    Args:
        text: Raw extracted text
        max_length: Maximum text length to send to LLM
    
    Returns:
        Sanitized text string
    """
    if not text:
        return ""
    
    # Truncate to max length
    sanitized = text[:max_length]
    
    # Remove potential prompt injection patterns (basic heuristic)
    injection_patterns = [
        r'(?i)ignore\s+(?:all\s+)?(?:previous|above)\s+instructions?',
        r'(?i)you\s+are\s+now\s+(?:a|in)\s+(?:new|different)',
        r'(?i)system\s*:\s*you\s+(?:are|will)',
        r'(?i)<\s*/?system\s*>',
    ]
    
    for pattern in injection_patterns:
        sanitized = re.sub(pattern, '[FILTERED]', sanitized)
    
    return sanitized

# Logo path - uses app directory for production compatibility
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")

def _format_indian_amount(amount: float) -> str:
    """Format amount in Indian currency style (lakhs/crores) for compact display in PDFs."""
    if amount is None or amount == 0:
        return "0"
    # Preserve negative sign for deficit/negative values
    is_negative = amount < 0
    amount = abs(amount)
    prefix = "-" if is_negative else ""
    if amount >= 1_00_00_000:  # 1 crore
        return f"{prefix}{amount / 1_00_00_000:.1f} Cr"
    elif amount >= 1_00_000:  # 1 lakh
        return f"{prefix}{amount / 1_00_000:.1f} L"
    elif amount >= 1000:
        return f"{prefix}{amount / 1000:.1f} K"
    else:
        return f"{prefix}{int(round(amount))}"

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

    Raw Bank Statement Text (truncated):
    {text[:8000]}
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
        data.update(llm_data)
    except Exception as e:
        logger.error(f"LLM extraction error for bank statement: {str(e)}")
        data['extraction_error'] = str(e)
        _add_extraction_metadata(data, {
            "_llm_extraction": {"confidence": 0.0, "source": "llm", "needs_review": True, "reason": str(e)}
        })
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
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": "You are a tax expert analyzing Indian ITR documents. All monetary values are in Indian Rupees (INR). Extract all financial data accurately. Return valid JSON only."},
                {"role": "user", "content": llm_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.05
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
        
        # Add confidence metadata for LLM-extracted fields  
        for key in ["income_sources", "carry_forward_losses", "assets_and_liabilities", "tax_computation", "deductions_claimed"]:
            if key in data:
                _add_extraction_metadata(data, {
                    key: _create_extraction_meta(key, data[key], "llm")
                })
            
    except Exception as e:
        logger.error(f"LLM extraction error for ITR: {str(e)}")
        data['extraction_error'] = str(e)
        _add_extraction_metadata(data, {
            "_llm_extraction": {"confidence": 0.0, "source": "llm", "needs_review": True, "reason": str(e)}
        })
        
    return data

def extract_insurance_hybrid(text):
    data = {}
    
    is_life_insurance = bool(re.search(r"(?i)(life\s+insurance|term\s+plan|endowment|ULIP|whole\s+life)", text))
    is_health_insurance = bool(re.search(r"(?i)(health\s+insurance|mediclaim|medical\s+insurance|hospitali[sz]ation|health\s*cover|in-?patient\s+treatment|annual\s+sum\s+insured|sum\s+insured|health\s+advantedge|health\s+plan|health\s+policy|medicare\s*premier|tata\s*aig\s*medicare)", text))
    is_general_insurance = bool(re.search(r"(?i)(motor\s+insurance|vehicle\s+insurance|property\s+insurance|home\s+insurance|fire\s+insurance)", text))
    
    # Debug logging
    logger.debug(f"Insurance type detection - life: {is_life_insurance}, health: {is_health_insurance}, general: {is_general_insurance}")
    
    if is_life_insurance:
        data["insurance_type"] = "Life Insurance"
    elif is_health_insurance:
        data["insurance_type"] = "Health Insurance"
    elif is_general_insurance:
        data["insurance_type"] = "General Insurance"
    else:
        data["insurance_type"] = "Unknown"
    
    logger.info(f"Detected insurance_type: {data['insurance_type']}")

    patterns = {
        "policy_number": [
            r"(?i)Policy\s*(?:No\.?|Number)[\s:\-]*([A-Z0-9\-/]{6,25})",
            r"(?i)Policy[\s:\-]*([A-Z0-9\-/]{6,25})",
            r"(?i)Member\s*(?:ID|No\.?)[\s:\-]*([A-Z0-9]{10,25})"  # TATA AIG uses Member ID
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
        matched_idx = None
        for idx, pattern in enumerate(pattern_list):
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                data[key] = match.group(1).strip() if match.group(1) else "N/A"
                matched_idx = idx
                break
        if key not in data:
            data[key] = "N/A"
        # Add confidence metadata for this field
        _add_extraction_metadata(data, {
            key: _create_extraction_meta(key, data[key], "regex", 
                                        pattern_idx=matched_idx if matched_idx is not None else len(pattern_list),
                                        total_patterns=len(pattern_list))
        })

    # Helper to convert Lakhs/Crores to actual numbers
    def _parse_indian_amount(num_str, suffix_str=""):
        """Convert amount with optional Lakhs/Crore suffix to actual number."""
        try:
            base_val = clean_and_convert_to_float(num_str)
            # clean_and_convert_to_float returns "N/A" string on failure, not None
            if base_val is None or base_val == "N/A" or not isinstance(base_val, (int, float)):
                return None
            suffix_lower = suffix_str.lower().strip() if suffix_str else ""
            if suffix_lower in ("lakh", "lakhs", "lac", "lacs"):
                return base_val * 100000
            elif suffix_lower in ("crore", "crores", "cr"):
                return base_val * 10000000
            return base_val
        except Exception as e:
            logger.warning(f"Error parsing Indian amount: {e}")
            return None

    # Patterns that capture amount AND optional Lakhs/Crore suffix
    # Priority order: Annual Sum Insured (most specific for health) > Sum Insured/Assured > Cover Amount
    sum_patterns = (
        [
            r"(?i)Sum\s*(?:Assured|Insured)[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)\s*(Lakhs?|Lacs?|Crores?|Cr)?",
            r"(?i)Life\s*Cover[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)\s*(Lakhs?|Lacs?|Crores?|Cr)?",
            r"(?i)Death\s*Benefit[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)\s*(Lakhs?|Lacs?|Crores?|Cr)?"
        ] if is_life_insurance else [
            # Highest priority: "Annual Sum Insured" - this is the actual policy value
            r"(?i)Annual\s+Sum\s+Insured[\s:\-\|]*(?:Rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)\s*(Lakhs?|Lacs?|Crores?|Cr)?",
            # TATA AIG format: "Sum Insured (₹)#" with value in same or next cell
            r"(?i)Sum\s+Insured\s*\(?₹?\)?[#*]*[\s:\-\|]*([\d,]+(?:\.\d+)?)\s*(Lakhs?|Lacs?|Crores?|Cr)?",
            # Next: general Sum Insured/Assured patterns
            r"(?i)Sum\s*(?:Insured|Assured)[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)\s*(Lakhs?|Lacs?|Crores?|Cr)?",
            r"(?i)Cover(?:age)?\s*Amount[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)\s*(Lakhs?|Lacs?|Crores?|Cr)?",
            r"(?i)Policy\s*Coverage[\s:\-]*(?:Rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)\s*(Lakhs?|Lacs?|Crores?|Cr)?"
        ]
    )
    
    # Try patterns in priority order
    # For "Annual Sum Insured" (first pattern) - take the FIRST match as that's the declared policy value
    # For fallback patterns - pick the largest to avoid catching small descriptive values
    sum_value = None
    for idx, pattern in enumerate(sum_patterns):
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            if idx == 0:
                # First pattern is "Annual Sum Insured" - take the first match (the declared policy value)
                m = matches[0]
                num_part = m.group(1) if m.lastindex >= 1 else None
                suffix_part = m.group(2) if m.lastindex >= 2 else ""
                parsed = _parse_indian_amount(num_part, suffix_part)
                if parsed is not None:
                    sum_value = parsed
                    logger.debug(f"Sum pattern matched (primary): value={sum_value}")
                    break
            else:
                # Fallback patterns - pick the largest value to avoid partial matches like "₹5 Lakhs" from descriptions
                best_val = None
                for m in matches:
                    num_part = m.group(1) if m.lastindex >= 1 else None
                    suffix_part = m.group(2) if m.lastindex >= 2 else ""
                    parsed = _parse_indian_amount(num_part, suffix_part)
                    if parsed is not None and (best_val is None or parsed > best_val):
                        best_val = parsed
                if best_val is not None:
                    sum_value = best_val
                    logger.debug(f"Sum pattern matched (fallback): value={best_val}")
                    sum_pattern_idx = idx  # Track which pattern matched
                    break
    
    if sum_value is not None:
        data["sum_assured_or_insured"] = sum_value
        # High confidence for primary "Annual Sum Insured" pattern (idx=0), lower for fallbacks
        _add_extraction_metadata(data, {
            "sum_assured_or_insured": _create_extraction_meta(
                "sum_assured_or_insured", sum_value, "regex",
                pattern_idx=sum_pattern_idx if 'sum_pattern_idx' in dir() else 0,
                total_patterns=len(sum_patterns))
        })
    else:
        data["sum_assured_or_insured"] = "N/A"
        _add_extraction_metadata(data, {
            "sum_assured_or_insured": {"confidence": 0.0, "source": "regex", "needs_review": True, "reason": "No value extracted"}
        })
    
    _log_extraction("insurance", "sum_assured_or_insured", data['sum_assured_or_insured'])

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
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": "You are an insurance expert analyzing Indian insurance policies. All monetary values are in Indian Rupees (INR). Extract all details accurately based on the insurance type. Return valid JSON only."},
                {"role": "user", "content": llm_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.05
        )
        llm_data = json.loads(response.choices[0].message.content)
        # Add confidence metadata for LLM-extracted fields
        for llm_field in llm_data.keys():
            if llm_field not in data.get("_extraction_meta", {}):  # Don't override regex confidence
                _add_extraction_metadata(data, {
                    llm_field: _create_extraction_meta(llm_field, llm_data[llm_field], "llm")
                })
        data.update(llm_data)
    except Exception as e:
        logger.error(f"LLM extraction error for insurance: {str(e)}")
        data['extraction_error'] = str(e)
        _add_extraction_metadata(data, {
            "_llm_extraction": {"confidence": 0.0, "source": "llm", "needs_review": True, "reason": str(e)}
        })
        
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
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": "You are a mutual fund analysis expert for Indian markets. All monetary values are in Indian Rupees (INR). Extract structured data into a valid JSON format."},
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
        data.update(llm_data)
    except Exception as e:
        logger.error(f"LLM extraction error for CAS: {str(e)}")
        data['extraction_error'] = str(e)
        _add_extraction_metadata(data, {
            "_llm_extraction": {"confidence": 0.0, "source": "llm", "needs_review": True, "reason": str(e)}
        })
    
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
    logger.debug("Starting structured text extraction...")
    text_content = []
    try:
        # Use the file stream directly with pdfplumber
        with pdfplumber.open(file_stream) as pdf:
            logger.debug(f"PDF has {len(pdf.pages)} pages.")
            for page_num, page in enumerate(pdf.pages):
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
    
    
        logger.debug("Finished structured text extraction.")
        return "".join(text_content)
    except Exception as e:
        logger.error(f"Error in extract_structured_text_with_tables: {e}")
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
    Calculate term insurance requirement.
    Formula: 15x annual income × 1.3 (inflation buffer) = 19.5x annual income
    
    For example: ₹30,000/month = ₹3.6 lakh/year → Required cover = ₹70.2 lakh
    
    Returns the required term cover amount.
    """
    mi = _safe_float(monthly_income, 0.0)
    annual_income = mi * 12
    
    # 15x annual income with 1.3x inflation buffer = 19.5x
    required_cover = annual_income * 15 * 1.3
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
    
    # Extract full ITR data from metadata_json (includes deductions_claimed, income_sources, etc.)
    for upload in uploads:
        doc_type = (upload.get("doc_type") or upload["doc_type"] if isinstance(upload, dict) else "").lower()
        if "itr" in doc_type:
            metadata_json = upload.get("metadata_json") or (upload["metadata_json"] if isinstance(upload, dict) else None)
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
                    # Merge full ITR extracted data into itr dict
                    # Include deductions_claimed, income_sources, tax_computation, etc.
                    if metadata.get("deductions_claimed"):
                        itr["deductions_claimed"] = metadata["deductions_claimed"]
                    if metadata.get("income_sources"):
                        itr["income_sources"] = metadata["income_sources"]
                    if metadata.get("tax_computation"):
                        itr["tax_computation"] = metadata["tax_computation"]
                    if metadata.get("carry_forward_losses"):
                        itr["carry_forward_losses"] = metadata["carry_forward_losses"]
                    if metadata.get("assets_and_liabilities"):
                        itr["assets_and_liabilities"] = metadata["assets_and_liabilities"]
                    # Also get numeric fields from metadata if not already from metrics
                    if metadata.get("gross_total_income") and not itr.get("gross_total_income"):
                        try:
                            itr["gross_total_income"] = float(metadata["gross_total_income"])
                        except (ValueError, TypeError):
                            pass
                    if metadata.get("taxable_income") and not itr.get("taxable_income"):
                        try:
                            itr["taxable_income"] = float(metadata["taxable_income"])
                        except (ValueError, TypeError):
                            pass
                    if metadata.get("total_tax_paid") and not itr.get("total_tax_paid"):
                        try:
                            itr["total_tax_paid"] = float(metadata["total_tax_paid"])
                        except (ValueError, TypeError):
                            pass
                except Exception as e:
                    logger.warning(f"Error parsing ITR metadata: {e}")
                    continue

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
                    logger.warning(f"Error parsing CAS metadata: {e}")
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
        # Expanded EMI keywords including bank-specific codes
        emi_keywords = [
            "emi", "loan", "ln", "mortgage", "instalment", "installment", "repayment",
            "home loan", "car loan", "personal loan", "vehicle loan", "housing loan", "education loan",
            # Bank-specific codes
            "elm",  # ICICI EMI/Loan
            "nach",  # National Automated Clearing House (auto-debit EMIs)
            "hdfc ln", "sbi ln", "icici ln", "axis ln", "kotak ln",  # Bank loan prefixes
            "si/",  # Standing Instruction
            "standing instruction",
            "ecs",  # Electronic Clearing Service
            "auto debit", "autodebit",
            "emi ded",  # SBI pattern
            "bil/inft/elm",  # ICICI pattern
        ]
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
                            is_emi = debit.get("is_emi", False)
                            
                            # Check if this is an EMI payment:
                            # 1. LLM marked it as EMI (is_emi=True)
                            # 2. OR description matches EMI keywords
                            is_emi_payment = is_emi or any(kw in desc for kw in emi_keywords)
                            
                            if is_emi_payment:
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
    except Exception as e:
        pass  # Silently handle errors to avoid breaking prefill

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
                print(f"[Prefill] Found insurance upload, metadata_json: {metadata_json[:500] if metadata_json else 'None'}")
                if metadata_json:
                    try:
                        metadata = json.loads(metadata_json)
                        ins_type = str(metadata.get("insurance_type") or "").lower()
                        sum_val = metadata.get("sum_assured_or_insured")
                        
                        print(f"[Prefill] Parsed insurance_type: '{ins_type}', sum_assured_or_insured: {sum_val}")
                        
                        # Also check aggregated insights if not in metadata
                        if sum_val is None or sum_val == "N/A":
                            sum_val = ins.get("sum_assured_or_insured")
                            print(f"[Prefill] Fallback to aggregated insights: {sum_val}")
                        
                        if isinstance(sum_val, (int, float)) and sum_val > 0:
                            if "health" in ins_type or "mediclaim" in ins_type:
                                # Add to health cover (may have multiple health policies)
                                existing_health = insurance_prefill.get("health_cover", 0.0)
                                insurance_prefill["health_cover"] = existing_health + float(sum_val)
                                print(f"[Prefill] Added to health_cover: {sum_val}")
                            elif "life" in ins_type or "term" in ins_type or "ulip" in ins_type:
                                # Add to life cover
                                existing_life = insurance_prefill.get("life_cover", 0.0)
                                insurance_prefill["life_cover"] = existing_life + float(sum_val)
                                print(f"[Prefill] Added to life_cover: {sum_val}")
                            else:
                                # Unknown type: default to life_cover
                                print(f"[Prefill] Unknown insurance type '{ins_type}', defaulting to life_cover")
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
            logger.info(f"Processing file {idx + 1}: {file.filename}")
            # Read file bytes once so we can reuse for multiple parsers
            file_bytes = file.read()
            file_stream_for_text = io.BytesIO(file_bytes)
            text = extract_structured_text_with_tables(file_stream_for_text)
            logger.debug(f"Extracted text for file {idx + 1} successfully.")

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
                    logger.error(f"Failed linking upload to questionnaire {questionnaire_id}: {e}")
            
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
                            logger.info(f"Bank statement metadata saved: {recurring_debits_count} recurring debits")
                        except Exception as e:
                            logger.error(f"Error updating Bank statement metadata: {e}")
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
                    logger.debug(f"Checking upload link: doc_type={doc_type}, idx={idx}")
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
                                logger.debug(f"Insurance metadata: type={metadata_update.get('insurance_type')}, sum={metadata_update.get('sum_assured_or_insured')}")
                            
                            update_questionnaire_upload_metadata(upload_link_ids[idx], metadata_update)
                            logger.info(f"Metadata updated: doc_type={doc_type}, upload_id={upload_link_ids[idx]}")
                        except Exception as e:
                            logger.error(f"Error updating {doc_type} metadata: {e}")

                    extracted_data[f"{name} {idx+1}"] = other_data
                    try:
                        _persist_metrics_for_doc(doc_id, other_data)
                    except Exception as e:
                        logger.error(f"Persist metrics ({doc_type}) failed: {e}")
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
        "itr": di.get("itr"),  # Full ITR data for tax optimization section
    }
    return facts


# ------------------------ Financial Health Score Calculator ------------------------ #
def compute_financial_health_score(analysis: dict, facts: dict) -> dict:
    """
    Compute overall financial health score (0-100) and component scores.
    
    Score = (Protection × 0.30) + (Liquidity × 0.20) + (PortfolioHealth × 0.20)
            + (DebtManagement × 0.15) + (GoalReadiness × 0.10) + (TaxEfficiency × 0.05)
    
    Returns:
        {
            "overall": 56,
            "overall_label": "Needs Attention",
            "components": {
                "protection": {"score": 0, "weight": 30, "weighted": 0, "label": "No Coverage"},
                ...
            },
            "priority_areas": [("protection", "URGENT"), ("liquidity", "HIGH"), ...]
        }
    """
    components = {}
    
    # 1. Protection Score (30% weight) - based on insurance gap
    insurance_gap = analysis.get("insuranceGap", "Unknown")
    if insurance_gap == "Adequate":
        protection_score = 100
        protection_label = "Adequate"
    elif insurance_gap == "Underinsured":
        protection_score = 50
        protection_label = "Underinsured"
    elif insurance_gap == "Critical":
        protection_score = 20
        protection_label = "Critical Gap"
    else:
        protection_score = 0
        protection_label = "No Coverage"
    
    components["protection"] = {
        "score": protection_score,
        "weight": 30,
        "weighted": protection_score * 0.30,
        "label": protection_label,
        "priority": "URGENT" if protection_score < 50 else ("MEDIUM" if protection_score < 80 else "GOOD")
    }
    
    # 2. Liquidity Score (20% weight) - based on liquidity assessment
    liquidity = analysis.get("liquidity", "Unknown")
    if liquidity == "Adequate":
        liquidity_score = 100
        liquidity_label = "Adequate"
    elif liquidity == "Low":
        liquidity_score = 50
        liquidity_label = "Low"
    elif liquidity == "Critical":
        liquidity_score = 20
        liquidity_label = "Critical"
    else:
        liquidity_score = 50  # Unknown defaults to middle
        liquidity_label = "Unknown"
    
    components["liquidity"] = {
        "score": liquidity_score,
        "weight": 20,
        "weighted": liquidity_score * 0.20,
        "label": liquidity_label,
        "priority": "HIGH" if liquidity_score < 50 else ("MEDIUM" if liquidity_score < 80 else "GOOD")
    }
    
    # 3. Portfolio Health Score (20% weight) - based on risk alignment
    advanced_risk = analysis.get("advancedRisk") or {}
    portfolio = facts.get("portfolio") or {}
    current_equity = _safe_float(portfolio.get("equity"), 50)
    rec_band = advanced_risk.get("recommendedEquityBand") or {}
    rec_min = rec_band.get("min", 40)
    rec_max = rec_band.get("max", 60)
    
    # Score based on how close to recommended band
    if rec_min <= current_equity <= rec_max:
        portfolio_score = 100
        portfolio_label = "Well Balanced"
    elif abs(current_equity - (rec_min + rec_max) / 2) <= 15:
        portfolio_score = 70
        portfolio_label = "Slightly Off"
    elif abs(current_equity - (rec_min + rec_max) / 2) <= 30:
        portfolio_score = 50
        portfolio_label = "Needs Rebalancing"
    else:
        portfolio_score = 30
        portfolio_label = "High Risk Deviation"
    
    components["portfolio_health"] = {
        "score": portfolio_score,
        "weight": 20,
        "weighted": portfolio_score * 0.20,
        "label": portfolio_label,
        "current_equity": current_equity,
        "recommended_range": f"{rec_min}%-{rec_max}%",
        "priority": "MEDIUM" if portfolio_score < 70 else "GOOD"
    }
    
    # 4. Debt Management Score (15% weight) - based on debt stress
    debt_stress = analysis.get("debtStress", "Low")
    if debt_stress == "Low":
        debt_score = 100
        debt_label = "Healthy"
    elif debt_stress == "Moderate":
        debt_score = 60
        debt_label = "Moderate"
    elif debt_stress == "High":
        debt_score = 30
        debt_label = "High Burden"
    else:
        debt_score = 80
        debt_label = "Unknown"
    
    components["debt_management"] = {
        "score": debt_score,
        "weight": 15,
        "weighted": debt_score * 0.15,
        "label": debt_label,
        "priority": "HIGH" if debt_score < 50 else ("MEDIUM" if debt_score < 70 else "GOOD")
    }
    
    # 5. Goal Readiness Score (10% weight) - based on goal funding status
    goals = facts.get("goals") or []
    total_goals = len(goals)
    # Use IHS or estimate from surplus
    ihs = analysis.get("ihs") or {}
    ihs_score = ihs.get("score", 50)
    
    # Approximate goal readiness from IHS and surplus band
    surplus_band = analysis.get("surplusBand", "Unknown")
    if surplus_band == "High":
        goal_score = min(80, ihs_score)
        goal_label = "Good Progress"
    elif surplus_band == "Adequate":
        goal_score = min(60, ihs_score)
        goal_label = "Moderate"
    elif surplus_band == "Low":
        goal_score = min(40, ihs_score)
        goal_label = "Needs Work"
    else:
        goal_score = 30
        goal_label = "At Risk"
    
    components["goal_readiness"] = {
        "score": goal_score,
        "weight": 10,
        "weighted": goal_score * 0.10,
        "label": goal_label,
        "total_goals": total_goals,
        "priority": "MEDIUM" if goal_score < 60 else "GOOD"
    }
    
    # 6. Tax Efficiency Score (5% weight) - based on ITR data if available
    itr = facts.get("itr") or {}
    if itr:
        # If significantly using deductions, good score
        deductions = itr.get("deductions_claimed") or []
        total_deductions = sum(d.get("amount", 0) for d in deductions)
        if total_deductions >= 150000:
            tax_score = 80
            tax_label = "Well Utilized"
        elif total_deductions >= 50000:
            tax_score = 50
            tax_label = "Partial Utilization"
        else:
            tax_score = 30
            tax_label = "Room for Improvement"
    else:
        tax_score = 50
        tax_label = "Not Assessed"
    
    components["tax_efficiency"] = {
        "score": tax_score,
        "weight": 5,
        "weighted": tax_score * 0.05,
        "label": tax_label,
        "priority": "MEDIUM" if tax_score < 60 else "GOOD"
    }
    
    # Calculate overall score
    overall = sum(c["weighted"] for c in components.values())
    overall = round(overall, 0)
    
    # Overall label
    if overall >= 80:
        overall_label = "Excellent"
    elif overall >= 60:
        overall_label = "Good"
    elif overall >= 40:
        overall_label = "Needs Attention"
    else:
        overall_label = "Critical"
    
    # Priority areas (sorted by score ascending)
    priority_areas = sorted(
        [(k, v["priority"], v["score"]) for k, v in components.items()],
        key=lambda x: x[2]
    )[:3]  # Top 3 priority areas
    
    return {
        "overall": int(overall),
        "overall_label": overall_label,
        "components": components,
        "priority_areas": [(p[0], p[1]) for p in priority_areas]
    }


# ------------------------ Action Timeline Generator ------------------------ #
def generate_action_timeline(analysis: dict, goals: list, facts: dict) -> dict:
    """
    Generate time-based action schedule from existing recommendations.
    
    Returns:
        {
            "week_1": [{"action": "...", "area": "Protection", "priority": "URGENT"}],
            "week_2": [...],
            "week_3": [...],
            "week_4": [...],
            "day_90": [...],
            "quarterly": [...]
        }
    """
    timeline = {
        "week_1": [],
        "week_2": [],
        "week_3": [],
        "week_4": [],
        "day_90": [],
        "quarterly": []
    }
    
    insurance_gap = analysis.get("insuranceGap", "Unknown")
    liquidity = analysis.get("liquidity", "Unknown")
    debt_stress = analysis.get("debtStress", "Low")
    advanced_risk = analysis.get("advancedRisk") or {}
    portfolio = facts.get("portfolio") or {}
    
    # WEEK 1: Protection Assessment (highest priority)
    if insurance_gap in ("Critical", "Unknown", "Underinsured"):
        timeline["week_1"].append({
            "action": "Check if you have term insurance. If yes, verify coverage is adequate. If no, prioritize obtaining term coverage.",
            "area": "Protection",
            "priority": "URGENT",
            "time_estimate": "2-3 hours"
        })
        timeline["week_1"].append({
            "action": "Check if you have health insurance. If yes, verify coverage is Rs. 10L+. If no, obtain health coverage.",
            "area": "Protection", 
            "priority": "URGENT",
            "time_estimate": "1-2 hours"
        })
    
    # WEEK 2: Emergency Fund Review
    if liquidity in ("Critical", "Low", "Unknown"):
        timeline["week_2"].append({
            "action": "Calculate actual monthly essential expenses",
            "area": "Emergency Fund",
            "priority": "HIGH",
            "time_estimate": "30 mins"
        })
        timeline["week_2"].append({
            "action": "Check savings account + FD balances to determine current liquid savings",
            "area": "Emergency Fund",
            "priority": "HIGH",
            "time_estimate": "30 mins"
        })
        timeline["week_2"].append({
            "action": "Set up monthly transfer to build emergency fund (target: 6 months expenses)",
            "area": "Emergency Fund",
            "priority": "HIGH",
            "time_estimate": "15 mins"
        })
    
    # WEEK 3: Portfolio Review
    current_equity = _safe_float(portfolio.get("equity"), 50)
    rec_band = advanced_risk.get("recommendedEquityBand") or {}
    rec_min = rec_band.get("min", 40)
    rec_max = rec_band.get("max", 60)
    
    if current_equity < rec_min - 10 or current_equity > rec_max + 10:
        timeline["week_3"].append({
            "action": f"Assess portfolio rebalancing need. Current equity: {current_equity}%, Recommended: {rec_min}%-{rec_max}%",
            "area": "Portfolio",
            "priority": "MEDIUM",
            "time_estimate": "1-2 hours"
        })
        timeline["week_3"].append({
            "action": "Review goal affordability and SIP allocation",
            "area": "Goals",
            "priority": "MEDIUM",
            "time_estimate": "1 hour"
        })
    
    # WEEK 4: Tax Planning
    itr = facts.get("itr") or {}
    if itr:
        deductions = itr.get("deductions_claimed") or []
        total_deductions = sum(d.get("amount", 0) for d in deductions)
        if total_deductions < 150000:
            timeline["week_4"].append({
                "action": "Calculate if old tax regime is beneficial for your situation",
                "area": "Tax Planning",
                "priority": "MEDIUM",
                "time_estimate": "1 hour"
            })
            timeline["week_4"].append({
                "action": "Plan 80C, 80D, 80CCD(1B) investments before financial year end",
                "area": "Tax Planning",
                "priority": "MEDIUM",
                "time_estimate": "30 mins"
            })
    
    # DAY 90: Full Progress Review
    timeline["day_90"].append({
        "action": "Review progress on all action items from Week 1-4",
        "area": "Review",
        "priority": "MEDIUM",
        "time_estimate": "1 hour"
    })
    if goals:
        timeline["day_90"].append({
            "action": f"Review goal SIP progress for {len(goals)} goal(s)",
            "area": "Goals",
            "priority": "MEDIUM",
            "time_estimate": "30 mins"
        })
    
    # QUARTERLY: Ongoing Review Schedule
    timeline["quarterly"].append({
        "action": "Recalculate Financial Health Score",
        "area": "Review",
        "priority": "MEDIUM",
        "time_estimate": "15 mins"
    })
    timeline["quarterly"].append({
        "action": "Check if income, expenses, or goals have changed - update plan accordingly",
        "area": "Review",
        "priority": "MEDIUM",
        "time_estimate": "30 mins"
    })
    timeline["quarterly"].append({
        "action": "Review portfolio allocation and rebalance if needed",
        "area": "Portfolio",
        "priority": "MEDIUM",
        "time_estimate": "30 mins"
    })
    
    return timeline


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

def _generate_financial_plan_pdf_legacy(q: dict, analysis: dict, output_path: str, doc_insights=None, narratives=None):
    """
    Generate 8-page Financial Plan PDF with new structure:
    Page 1: Executive Summary with Financial Health Score
    Page 2: Actuals vs Ideal Dashboard
    Page 3: Protection Gap Analysis
    Page 4: Liquidity & Emergency Fund
    Page 5: Portfolio Rebalancing
    Page 6: Goal Feasibility Analysis
    Page 7: Tax Optimization (if applicable)
    Page 8: Action Roadmap with Timeline
    """
    styles = get_custom_styles()
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            rightMargin=inch*0.75, leftMargin=inch*0.75,
                            topMargin=inch, bottomMargin=inch)
    story = []

    # Extract common data
    name = (q.get("personal_info") or {}).get("name") or "Client"
    age = (q.get("personal_info") or {}).get("age") or "N/A"
    family = q.get("family_info") or {}
    spouse = family.get("spouse")
    children = family.get("children") or []
    dependents = family.get("dependents") or []
    goals = (q.get("goals") or {}).get("items") or []
    lifestyle = (q.get("lifestyle") or {}) or {}
    insurance = q.get("insurance") or {}
    di = doc_insights or {}
    bank = di.get("bank") or {}
    portfolio = (di.get("portfolio") if di else None) or {}
    ihs = analysis.get("ihs") or {}
    advanced_risk = analysis.get("advancedRisk") or {}
    categorized = _categorize_recommendations(analysis.get("recommendations") or [])
    
    # Build facts dict for helper functions
    facts = {
        "goals": goals,
        "portfolio": portfolio,
        "itr": di.get("itr"),
    }
    
    # Compute Financial Health Score
    health_score = compute_financial_health_score(analysis, facts)
    
    # Generate Timeline
    timeline = generate_action_timeline(analysis, goals, facts)
    
    # Get CAS data
    cas_data = _get_cas_data_for_questionnaire(q.get("id"))
    trans_sum = (cas_data.get("transaction_summary") if cas_data else {}) or {}
    inv_snapshot = (cas_data.get("investment_snapshot") if cas_data else {}) or {}
    sip_details = (cas_data.get("sip_details") if cas_data else []) or []
    
    # Calculate key numbers
    annual_income = _safe_float(lifestyle.get("annual_income"), 0)
    monthly_income = annual_income / 12 if annual_income else 0
    monthly_expenses = _safe_float(lifestyle.get("monthly_expenses"), 0)
    monthly_emi = _safe_float(lifestyle.get("monthly_emi"), 0)
    monthly_surplus = monthly_income - monthly_expenses - monthly_emi
    savings_rate = (monthly_surplus / monthly_income * 100) if monthly_income > 0 else 0
    
    # Portfolio values
    current_portfolio = trans_sum.get("total_current_value") or inv_snapshot.get("current_value") or 0
    unrealized_gains = trans_sum.get("total_unrealized_gain") or inv_snapshot.get("net_gain") or 0
    current_equity = _safe_float(portfolio.get("equity"), 0)
    total_monthly_sip = sum(
        s.get("sip_amount", 0) 
        for s in sip_details 
        if (s.get("frequency") or "").lower() == "monthly"
    )
    
    # Protection values
    life_cover = _safe_float(insurance.get("life_cover"), 0)
    health_cover = _safe_float(insurance.get("health_cover"), 0)
    required_term_cover = compute_term_insurance_need(age if isinstance(age, (int, float)) else 30, monthly_income)
    required_health_cover = 1000000  # Rs. 10L recommended minimum
    
    # Emergency fund
    emergency_fund_target = monthly_expenses * 6 if monthly_expenses > 0 else 120000

    # ==========================================================================
    # PAGE 1: EXECUTIVE SUMMARY WITH FINANCIAL HEALTH SCORE
    # ==========================================================================
    story.append(Paragraph("Financial Plan", styles["Title"]))
    story.append(Paragraph("Page 1: Executive Summary", styles["h1"]))
    story.append(Spacer(1, 8))
    
    # Financial Health Score Box
    story.append(Paragraph("Financial Health Score", styles["h2"]))
    score = health_score["overall"]
    score_label = health_score["overall_label"]
    
    score_rows = [
        ["Overall Score", f"{score}/100"],
        ["Status", score_label],
    ]
    score_table = Table(score_rows, hAlign="LEFT", colWidths=[150, 350])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#1D3557')),
        ('TEXTCOLOR', (0,0), (0,-1), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 8))
    
    # Score Components Table
    story.append(Paragraph("Score Components", styles["BodyText"]))
    comp = health_score["components"]
    component_rows = []
    for key, val in comp.items():
        display_name = key.replace("_", " ").title()
        component_rows.append([
            display_name, 
            f"{val['score']}/100",
            f"{val['weight']}%",
            val['label']
        ])
    
    comp_table = Table(
        [["Area", "Score", "Weight", "Status"]] + component_rows,
        hAlign="LEFT",
        colWidths=[120, 70, 60, 250]
    )
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#457B9D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 12))
    
    # Your Snapshot Table
    story.append(Paragraph("Your Snapshot", styles["h2"]))
    rec_band = advanced_risk.get("recommendedEquityBand") or {}
    rec_equity_range = f"{rec_band.get('min', 40)}%-{rec_band.get('max', 60)}%"
    
    snapshot_rows = [
        ["Life Cover", f"Rs. {_format_indian_amount(life_cover)}" if life_cover > 0 else "Rs. 0", 
         f"Rs. {_format_indian_amount(required_term_cover)}", 
         "URGENT" if life_cover < required_term_cover * 0.5 else "MEDIUM" if life_cover < required_term_cover else "GOOD"],
        ["Health Cover", f"Rs. {_format_indian_amount(health_cover)}" if health_cover > 0 else "Rs. 0",
         f"Rs. {_format_indian_amount(required_health_cover)}",
         "URGENT" if health_cover < required_health_cover * 0.5 else "MEDIUM" if health_cover < required_health_cover else "GOOD"],
        ["Emergency Fund", "Unknown", f"Rs. {_format_indian_amount(emergency_fund_target)}", "HIGH"],
        ["Equity Allocation", f"{current_equity:.0f}%", rec_equity_range, 
         "MEDIUM" if abs(current_equity - (rec_band.get('min', 40) + rec_band.get('max', 60))/2) > 15 else "GOOD"],
        ["EMI Burden", f"{(monthly_emi/monthly_income*100) if monthly_income > 0 else 0:.0f}%", "<40%", 
         "HIGH" if monthly_emi/monthly_income > 0.4 else "GOOD" if monthly_income > 0 else "UNKNOWN"],
        ["Goals Funded", f"0/{len(goals)}", f"{len(goals)}/{len(goals)}", "MEDIUM"],
    ]
    
    snapshot_table = Table(
        [["Area", "Current", "Ideal", "Priority"]] + snapshot_rows,
        hAlign="LEFT",
        colWidths=[120, 100, 100, 80]
    )
    snapshot_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#457B9D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
    ]))
    story.append(snapshot_table)
    story.append(Spacer(1, 12))
    
    # Key Numbers Boxes
    story.append(Paragraph("Key Numbers", styles["h2"]))
    
    # Income & Cashflow
    key_rows_1 = [
        ["Annual Income", f"Rs. {_format_indian_amount(annual_income)}" if annual_income > 0 else "-"],
        ["Monthly Surplus", f"Rs. {_format_indian_amount(monthly_surplus)}" if monthly_surplus != 0 else "-"],
        ["Savings Rate", f"{savings_rate:.0f}%" if monthly_income > 0 else "-"],
    ]
    # Investments
    key_rows_2 = [
        ["Current Portfolio", f"Rs. {_format_indian_amount(current_portfolio)}" if current_portfolio > 0 else "-"],
        ["Equity Allocation", f"{current_equity:.0f}%"],
        ["Active SIP", f"Rs. {total_monthly_sip:,.0f}/month" if total_monthly_sip > 0 else "-"],
    ]
    
    key_table = Table(
        [["Income & Cashflow", ""]] + key_rows_1 + [["-", "-"]] + [["Investments", ""]] + key_rows_2,
        hAlign="LEFT",
        colWidths=[200, 300]
    )
    key_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2D6A4F')),
        ('BACKGROUND', (0,4), (-1,4), colors.HexColor('#457B9D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('TEXTCOLOR', (0,4), (-1,4), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,4), (-1,4), 'Helvetica-Bold'),
        ('SPAN', (0,0), (-1,0)),
        ('SPAN', (0,4), (-1,4)),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(key_table)
    story.append(Spacer(1, 12))
    
    # Top 3 Priorities
    story.append(Paragraph("Your Top 3 Priorities", styles["h2"]))
    priority_areas = health_score.get("priority_areas", [])
    for i, (area, priority) in enumerate(priority_areas[:3], 1):
        area_display = area.replace("_", " ").title()
        story.append(Paragraph(f"<b>Priority {i}:</b> {area_display} ({priority})", styles["BodyText"]))
    
    story.append(Spacer(1, 8))
    story.append(Paragraph("<i>Time to review this report: 15 minutes</i>", styles["BodyText"]))
    story.append(PageBreak())

    # ==========================================================================
    # PAGE 2: ACTUALS VS IDEAL DASHBOARD
    # ==========================================================================
    story.append(Paragraph("Page 2: Actuals vs Ideal", styles["h1"]))
    story.append(Paragraph("Financial Health Dashboard", styles["h2"]))
    
    # Comparison table for each area
    comparison_rows = [
        ["Protection - Life", f"Rs. {_format_indian_amount(life_cover)}", f"Rs. {_format_indian_amount(required_term_cover)}+", 
         f"{(life_cover/required_term_cover*100) if required_term_cover > 0 else 0:.0f}%"],
        ["Protection - Health", f"Rs. {_format_indian_amount(health_cover)}", f"Rs. {_format_indian_amount(required_health_cover)}+",
         f"{(health_cover/required_health_cover*100) if required_health_cover > 0 else 0:.0f}%"],
        ["Emergency Fund", "Unknown", f"Rs. {_format_indian_amount(emergency_fund_target)}", "Assess"],
        ["Equity Allocation", f"{current_equity:.0f}%", rec_equity_range, 
         "In Range" if rec_band.get('min', 40) <= current_equity <= rec_band.get('max', 60) else "Rebalance"],
        ["Debt Allocation", f"{100 - current_equity:.0f}%", f"{100-rec_band.get('max', 60)}%-{100-rec_band.get('min', 40)}%", "-"],
        ["EMI/Income Ratio", f"{(monthly_emi/monthly_income*100) if monthly_income > 0 else 0:.0f}%", "<40%", 
         "Good" if (monthly_emi/monthly_income if monthly_income > 0 else 0) < 0.4 else "High"],
    ]
    
    comparison_table = Table(
        [["Area", "You", "Recommended", "Status"]] + comparison_rows,
        hAlign="LEFT",
        colWidths=[130, 100, 130, 100]
    )
    comparison_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1D3557')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
    ]))
    story.append(comparison_table)
    story.append(Spacer(1, 12))
    
    # Priority Ranking Table
    story.append(Paragraph("Priority Ranking", styles["h2"]))
    priority_mapping = {
        "protection": comp["protection"]["score"],
        "liquidity": comp["liquidity"]["score"],
        "portfolio_health": comp["portfolio_health"]["score"],
        "debt_management": comp["debt_management"]["score"],
        "goal_readiness": comp["goal_readiness"]["score"],
        "tax_efficiency": comp["tax_efficiency"]["score"],
    }
    sorted_priorities = sorted(priority_mapping.items(), key=lambda x: x[1])
    
    priority_rows = []
    for area, score in sorted_priorities:
        area_display = area.replace("_", " ").title()
        if score < 40:
            action = "Immediate"
        elif score < 60:
            action = "Assess & Improve"
        elif score < 80:
            action = "Review"
        else:
            action = "Maintain"
        priority_rows.append([area_display, f"{score}/100", action])
    
    priority_rows.append(["OVERALL", f"{health_score['overall']}/100", health_score['overall_label'].upper()])
    
    priority_table = Table(
        [["Area", "Score", "Action Required"]] + priority_rows,
        hAlign="LEFT",
        colWidths=[200, 100, 200]
    )
    priority_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#457B9D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#1D3557')),
        ('TEXTCOLOR', (0,-1), (-1,-1), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
    ]))
    story.append(priority_table)
    story.append(Spacer(1, 12))
    
    # When to recalculate
    story.append(Paragraph("<b>When to recalculate this score:</b>", styles["BodyText"]))
    recalc_items = [
        "Quarterly review (every 3 months)",
        "After change in income or expenses",
        "After major asset/liability changes",
        "Before making significant financial decisions"
    ]
    for item in recalc_items:
        story.append(Paragraph(f"• {item}", styles["BodyText"]))
    story.append(Spacer(1, 12))
    
    # =========================================================================
    # ADVANCED RISK ASSESSMENT SECTION
    # =========================================================================
    story.append(Paragraph("Advanced Risk Assessment", styles["h2"]))
    
    # Extract risk assessment values
    risk_score = advanced_risk.get("score", 0)
    risk_appetite = advanced_risk.get("riskAppetite", "Moderate")
    tenure_limit = advanced_risk.get("tenureLimit", "Moderate")
    baseline_category = advanced_risk.get("baselineCategory", "Moderate")
    final_category = advanced_risk.get("finalCategory", "Moderate")
    rec_equity_min = rec_band.get("min", 40)
    rec_equity_max = rec_band.get("max", 60)
    rec_equity_mid = (rec_equity_min + rec_equity_max) / 2
    
    risk_assessment_rows = [
        ["Calculated Score", f"{risk_score:.1f}" if isinstance(risk_score, (int, float)) else str(risk_score)],
        ["Risk Appetite", risk_appetite],
        ["Tenure Limit", tenure_limit],
        ["Baseline Category", baseline_category],
        ["Final Category", final_category],
        ["Recommended Equity Band", f"{rec_equity_min}%-{rec_equity_max}% (mid {rec_equity_mid:.1f}%)"],
    ]
    
    risk_table = Table(
        [["Parameter", "Value"]] + risk_assessment_rows,
        hAlign="LEFT",
        colWidths=[180, 280]
    )
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#9D4B45')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 6))
    
    # Reasoning line
    reasoning = f"Reasoning: Score {risk_score:.2f} → Appetite {risk_appetite} | Tenure limit {tenure_limit} | Baseline after adjustments {baseline_category} | Final {final_category}"
    story.append(Paragraph(f"<i>{reasoning}</i>", styles["BodyText"]))
    story.append(Spacer(1, 12))
    
    # Assessment Summary
    story.append(Paragraph("Assessment Summary", styles["h2"]))
    
    # Determine status labels
    surplus_status = "Adequate" if monthly_surplus > 0 else ("Tight" if monthly_surplus >= -5000 else "Deficit")
    insurance_status = "Adequate" if (life_cover >= required_term_cover * 0.8 and health_cover >= required_health_cover * 0.8) else "Underinsured"
    debt_status = "Low" if monthly_emi <= monthly_income * 0.2 else ("Moderate" if monthly_emi <= monthly_income * 0.4 else "High")
    liquidity_status = "Sufficient" if comp["liquidity"]["score"] >= 60 else "Insufficient"
    ihs_band = health_score.get("overall_label", "Unknown")
    
    summary_assessment_rows = [
        ["Risk Profile", final_category],
        ["Surplus Level", surplus_status],
        ["Insurance Status", insurance_status],
        ["Debt Position", debt_status],
        ["Liquidity", liquidity_status],
        ["IHS Band", ihs_band],
    ]
    
    summary_table2 = Table(
        [["Assessment", "Result"]] + summary_assessment_rows,
        hAlign="LEFT",
        colWidths=[180, 280]
    )
    summary_table2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#457B9D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
    ]))
    story.append(summary_table2)
    
    story.append(PageBreak())

    # ==========================================================================
    # PAGE 3: PROTECTION GAP ANALYSIS
    # ==========================================================================
    story.append(Paragraph("Page 3: Protection Gap Analysis", styles["h1"]))
    
    # Life Insurance Section
    story.append(Paragraph("Life Insurance", styles["h2"]))
    life_status = "Adequate" if life_cover >= required_term_cover else ("Underinsured" if life_cover > 0 else "No coverage")
    life_rows = [
        ["Current Status", life_status],
        ["Current Coverage", f"Rs. {_format_indian_amount(life_cover)}" if life_cover > 0 else "Rs. 0"],
        ["Recommended Coverage", f"Rs. {_format_indian_amount(required_term_cover)} minimum"],
        ["Basis", "15x annual income (with 1.3x inflation buffer)"],
        ["Gap", f"Rs. {_format_indian_amount(max(0, required_term_cover - life_cover))}"],
    ]
    life_table = Table(
        [["Parameter", "Value"]] + life_rows,
        hAlign="LEFT",
        colWidths=[200, 300]
    )
    life_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#9D4B45') if life_cover < required_term_cover else colors.HexColor('#2D6A4F')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
    ]))
    story.append(life_table)
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("<b>What to look for:</b>", styles["BodyText"]))
    story.append(Paragraph("• Pure term insurance (no investment component)", styles["BodyText"]))
    story.append(Paragraph("• Coverage tenure until age 60-65", styles["BodyText"]))
    story.append(Paragraph(f"• Sum assured: Rs. {_format_indian_amount(required_term_cover)} minimum", styles["BodyText"]))
    story.append(Paragraph("• Consider critical illness rider for comprehensive protection", styles["BodyText"]))
    story.append(Spacer(1, 12))
    
    # Health Insurance Section
    story.append(Paragraph("Health Insurance", styles["h2"]))
    health_status = "Adequate" if health_cover >= required_health_cover else ("Underinsured" if health_cover > 0 else "No coverage")
    health_rows = [
        ["Current Status", health_status],
        ["Current Coverage", f"Rs. {_format_indian_amount(health_cover)}" if health_cover > 0 else "Rs. 0"],
        ["Recommended Coverage", f"Rs. {_format_indian_amount(required_health_cover)}-15L"],
        ["Structure", "Family floater adequate for married couple"],
        ["Gap", f"Rs. {_format_indian_amount(max(0, required_health_cover - health_cover))}"],
    ]
    health_table = Table(
        [["Parameter", "Value"]] + health_rows,
        hAlign="LEFT",
        colWidths=[200, 300]
    )
    health_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#9D4B45') if health_cover < required_health_cover else colors.HexColor('#2D6A4F')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
    ]))
    story.append(health_table)
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("<b>What to look for:</b>", styles["BodyText"]))
    story.append(Paragraph("• Family floater covering both spouses", styles["BodyText"]))
    story.append(Paragraph("• Rs. 10-15 lakh sum insured", styles["BodyText"]))
    story.append(Paragraph("• Cashless facility at major hospitals", styles["BodyText"]))
    story.append(Paragraph("• No room rent capping (or minimum 2% of sum insured)", styles["BodyText"]))
    story.append(Paragraph("<i>Tax benefit: Premiums eligible for 80D deduction (up to Rs. 25,000)</i>", styles["BodyText"]))
    story.append(PageBreak())

    # ==========================================================================
    # PAGE 4: LIQUIDITY & EMERGENCY FUND
    # ==========================================================================
    story.append(Paragraph("Page 4: Liquidity & Emergency Fund", styles["h1"]))
    
    story.append(Paragraph("Emergency Fund Assessment", styles["h2"]))
    ef_rows = [
        ["Target Amount", f"Rs. {_format_indian_amount(emergency_fund_target)}"],
        ["Basis", "6 months of essential expenses"],
        ["Monthly Expenses Used", f"Rs. {_format_indian_amount(monthly_expenses)}" if monthly_expenses > 0 else "Estimated Rs. 20,000"],
        ["Current Status", "Assess your savings account + FD balances"],
    ]
    ef_table = Table(
        [["Parameter", "Value"]] + ef_rows,
        hAlign="LEFT",
        colWidths=[200, 300]
    )
    ef_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#457B9D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
    ]))
    story.append(ef_table)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("<b>Building Strategy (if starting from zero):</b>", styles["BodyText"]))
    monthly_ef_12 = emergency_fund_target / 12
    monthly_ef_18 = emergency_fund_target / 18
    monthly_ef_24 = emergency_fund_target / 24
    story.append(Paragraph(f"• Option A (12 months): Set aside Rs. {monthly_ef_12:,.0f}/month", styles["BodyText"]))
    story.append(Paragraph(f"• Option B (18 months): Set aside Rs. {monthly_ef_18:,.0f}/month", styles["BodyText"]))
    story.append(Paragraph(f"• Option C (24 months): Set aside Rs. {monthly_ef_24:,.0f}/month", styles["BodyText"]))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("<b>Where to Park Emergency Fund:</b>", styles["BodyText"]))
    tier1 = emergency_fund_target * 0.4
    tier2 = emergency_fund_target * 0.4
    tier3 = emergency_fund_target * 0.2
    story.append(Paragraph(f"• Tier 1 - Instant Access (40%): Rs. {tier1:,.0f} in savings bank account", styles["BodyText"]))
    story.append(Paragraph(f"• Tier 2 - Quick Access (40%): Rs. {tier2:,.0f} in liquid funds or sweep-in FD", styles["BodyText"]))
    story.append(Paragraph(f"• Tier 3 - Short-term (20%): Rs. {tier3:,.0f} in short-term fixed deposits", styles["BodyText"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Priority:</b> Build emergency fund BEFORE investing in long-term goals", styles["BodyText"]))
    story.append(PageBreak())

    # ==========================================================================
    # PAGE 5: PORTFOLIO REBALANCING
    # ==========================================================================
    story.append(Paragraph("Page 5: Portfolio Rebalancing", styles["h1"]))
    
    story.append(Paragraph("Current Portfolio Analysis", styles["h2"]))
    portfolio_rows = [
        ["Current Value", f"Rs. {_format_indian_amount(current_portfolio)}" if current_portfolio > 0 else "-"],
        ["Equity Allocation", f"{current_equity:.0f}%"],
        ["Debt Allocation", f"{100 - current_equity:.0f}%"],
        ["Unrealized Gains", f"Rs. {_format_indian_amount(unrealized_gains)}" if unrealized_gains != 0 else "-"],
    ]
    portfolio_table = Table(
        [["Metric", "Value"]] + portfolio_rows,
        hAlign="LEFT",
        colWidths=[200, 300]
    )
    portfolio_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#457B9D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
    ]))
    story.append(portfolio_table)
    story.append(Spacer(1, 8))
    
    # Risk Assessment
    story.append(Paragraph("<b>Risk Assessment:</b>", styles["BodyText"]))
    if current_equity > rec_band.get('max', 60) + 10:
        story.append(Paragraph(f"• {current_equity:.0f}% equity allocation is aggressive for your goal mix", styles["BodyText"]))
        story.append(Paragraph("• No debt cushion for near-term goals", styles["BodyText"]))
    elif current_equity < rec_band.get('min', 40) - 10:
        story.append(Paragraph(f"• {current_equity:.0f}% equity allocation may be too conservative", styles["BodyText"]))
        story.append(Paragraph("• May not meet inflation-adjusted growth requirements", styles["BodyText"]))
    else:
        story.append(Paragraph("• Current allocation is within recommended range", styles["BodyText"]))
    story.append(Spacer(1, 12))
    
    # Recommended Rebalancing
    story.append(Paragraph("Recommended Allocation", styles["h2"]))
    rec_mid = (rec_band.get('min', 40) + rec_band.get('max', 60)) / 2
    rebal_rows = [
        ["Current Equity", f"{current_equity:.0f}%", "Target Equity", f"{rec_mid:.0f}%"],
        ["Current Debt", f"{100 - current_equity:.0f}%", "Target Debt", f"{100 - rec_mid:.0f}%"],
    ]
    rebal_table = Table(rebal_rows, hAlign="LEFT", colWidths=[120, 80, 120, 80])
    rebal_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (1,-1), colors.HexColor('#F4F4F4')),
        ('BACKGROUND', (2,0), (-1,-1), colors.HexColor('#E8F5E9')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(rebal_table)
    story.append(Spacer(1, 12))
    
    # What you control
    story.append(Paragraph("<b>What You Control:</b>", styles["BodyText"]))
    story.append(Paragraph("□ Which funds/instruments to use", styles["BodyText"]))
    story.append(Paragraph("□ When to execute rebalancing", styles["BodyText"]))
    story.append(Paragraph("□ How to split equity vs debt within each goal", styles["BodyText"]))
    story.append(Paragraph("□ Which goals to prioritize if all aren't affordable", styles["BodyText"]))
    story.append(PageBreak())

    # ==========================================================================
    # PAGE 6: GOAL FEASIBILITY ANALYSIS
    # ==========================================================================
    story.append(Paragraph("Page 6: Goal Feasibility Analysis", styles["h1"]))
    
    # Calculate SIPs and funding status for all goals
    total_required_sip = 0
    goal_analysis = []
    
    if goals:
        # Get risk profile for SIP calculations
        final_category = advanced_risk.get("finalCategory") or "Moderate"
        
        for i, g in enumerate(goals[:5], 1):  # Limit to 5 goals
            g_name = g.get("name") or g.get("goal") or f"Goal {i}"
            g_target = _safe_float(g.get("target_amount"), 0)
            g_horizon = g.get("horizon_years") or g.get("horizon") or None
            
            # Get per-goal risk tolerance if available, else use overall
            goal_risk = g.get("risk_tolerance") or final_category
            
            # Calculate required SIP using the function from llm_sections
            required_sip = None
            if g_target > 0 and g_horizon:
                try:
                    required_sip = compute_goal_sip(g_target, g_horizon, goal_risk)
                except:
                    required_sip = None
            
            goal_analysis.append({
                "name": g_name,
                "target": g_target,
                "horizon": g_horizon,
                "required_sip": required_sip or 0,
            })
            
            if required_sip:
                total_required_sip += required_sip
        
        # Determine funding status
        # Note: existing SIP is already invested FROM surplus - it represents current investing capacity
        # Total investing power = existing SIP (already committed) + surplus (available for new investments)
        # For goal allocation, we consider how much of total goal requirement can be met
        total_investing_capacity = total_monthly_sip + max(0, monthly_surplus)
        
        # For proportional allocation to individual goals, we use the surplus (new money available)
        # But existing SIP should be considered as already contributing to goals
        available_for_goals = max(0, monthly_surplus)  # New money available for goal allocation
        
        funding_pct = (total_investing_capacity / total_required_sip * 100) if total_required_sip > 0 else 100
        shortfall = max(0, total_required_sip - total_investing_capacity)
        
        # Render each goal with SIP analysis
        for i, ga in enumerate(goal_analysis, 1):
            g_name = ga["name"]
            g_target = ga["target"]
            g_horizon = ga["horizon"]
            required_sip = ga["required_sip"]
            
            # Calculate per-goal funding status
            if total_required_sip > 0 and required_sip > 0:
                # Proportionally allocate available funds to this goal
                allocated_sip = (required_sip / total_required_sip) * available_for_goals
                allocated_sip = max(0, allocated_sip)  # Never negative
                
                if allocated_sip >= required_sip * 0.95:
                    status = "Fully Funded"
                    status_color = colors.HexColor('#2D6A4F')  # Green
                elif allocated_sip >= required_sip * 0.5:
                    status = "Partially Funded"
                    status_color = colors.HexColor('#E9C46A')  # Yellow
                else:
                    status = "Gap Exists"
                    status_color = colors.HexColor('#9D4B45')  # Red
                
                # Gap is difference, capped at required_sip (can never be more than 100% shortfall)
                gap = min(required_sip, max(0, required_sip - allocated_sip))
            else:
                status = "Not Calculable"
                status_color = colors.HexColor('#457B9D')
                gap = 0
                allocated_sip = 0
            
            story.append(Paragraph(f"<b>Goal {i}: {sanitize_pdf_text(g_name)}</b>", styles["h2"]))
            
            goal_rows = [
                ["Target Amount", f"Rs. {_format_indian_amount(g_target)}" if g_target > 0 else "-"],
                ["Horizon", f"{g_horizon} years" if g_horizon else "-"],
                ["Required SIP", f"Rs. {required_sip:,.0f}/month" if required_sip else "Not calculable"],
                ["Status", status],
            ]
            if gap > 0:
                goal_rows.append(["Shortfall", f"Rs. {gap:,.0f}/month"])
            
            goal_table = Table(
                [["Parameter", "Value"]] + goal_rows,
                hAlign="LEFT",
                colWidths=[200, 300]
            )
            goal_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), status_color),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ]))
            story.append(goal_table)
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("No goals recorded. Please add your financial goals for feasibility analysis.", styles["BodyText"]))
        total_required_sip = 0
        shortfall = 0
        funding_pct = 100
        available_for_goals = monthly_surplus
    
    # Reality Check Summary Box - Enhanced with Existing SIP Context
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Reality Check: Your Actual Financial Situation</b>", styles["h2"]))
    
    # Calculate insurance provision for proper context
    term_gap = max(0, required_term_cover - life_cover)
    health_gap = max(0, required_health_cover - health_cover)
    client_age = _safe_float(age, 35)
    term_rate_per_lakh = 500 if client_age < 35 else (700 if client_age < 45 else 1200)
    health_rate_per_lakh = 2500 if client_age < 35 else (3500 if client_age < 45 else 5000)
    term_premium_yearly = (term_gap / 100000) * term_rate_per_lakh if term_gap > 0 else 0
    health_premium_yearly = (health_gap / 100000) * health_rate_per_lakh if health_gap > 0 else 0
    monthly_insurance_provision = (term_premium_yearly + health_premium_yearly) / 12
    
    # Calculate remaining after existing SIP and insurance (can be negative if in deficit)
    # Existing SIPs are already consuming part of the surplus, so subtract them first
    # Formula: available_for_new = surplus - existing_sip - insurance
    surplus_after_sip_and_insurance = monthly_surplus - total_monthly_sip - monthly_insurance_provision
    available_for_new_sips = max(0, surplus_after_sip_and_insurance)  # Can't invest negative
    
    # Total investing = existing SIP + new allocations (if any available)
    total_investing = total_monthly_sip + available_for_new_sips
    total_investing_pct = (total_investing / total_required_sip * 100) if total_required_sip > 0 else 100
    
    summary_rows = [
        ["Monthly Income", f"Rs. {_format_indian_amount(monthly_income)}"],
        ["Monthly Expenses", f"Rs. {_format_indian_amount(monthly_expenses)}"],
        ["Monthly Surplus", f"Rs. {_format_indian_amount(monthly_surplus)}"],
        ["", ""],  # Separator row
        ["Current SIP (Already Investing)", f"Rs. {total_monthly_sip:,.0f}/month" if total_monthly_sip > 0 else "None"],
    ]
    
    if monthly_insurance_provision > 0:
        summary_rows.append(["Insurance Provision Needed", f"Rs. {monthly_insurance_provision:,.0f}/month"])
        # Show clear deficit indicator if surplus is negative
        if surplus_after_sip_and_insurance < 0:
            summary_rows.append(["Available for NEW Additions", f"Rs. 0/month (Deficit: Rs. {abs(surplus_after_sip_and_insurance):,.0f})"])
        else:
            summary_rows.append(["Available for NEW Additions", f"Rs. {available_for_new_sips:,.0f}/month"])
    
    summary_rows.extend([
        ["", ""],  # Separator row
        ["TOTAL Investing", f"Rs. {total_investing:,.0f}/month"],
        ["Goal Requirement", f"Rs. {total_required_sip:,.0f}/month"],
        ["Coverage", f"{min(total_investing_pct, 100):.1f}% of requirement"],
    ])
    
    if total_required_sip > total_investing:
        summary_rows.append(["Shortfall", f"Rs. {total_required_sip - total_investing:,.0f}/month"])
    
    summary_table = Table(
        summary_rows,
        hAlign="LEFT",
        colWidths=[200, 300]
    )
    
    # Color based on funding status
    if total_investing_pct >= 95:
        header_color = colors.HexColor('#2D6A4F')  # Green
    elif total_investing_pct >= 50:
        header_color = colors.HexColor('#E9C46A')  # Yellow
    else:
        header_color = colors.HexColor('#9D4B45')  # Red
    
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), header_color),
        ('TEXTCOLOR', (0,0), (0,-1), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 12))
    
    # =========================================================================
    # GOAL-WISE SIP ALLOCATION TABLE
    # =========================================================================
    if len(goal_analysis) > 0:
        story.append(Paragraph("<b>Goal-wise SIP Allocation</b>", styles["h2"]))
        
        # Calculate per-goal existing portfolio allocation
        portfolio_per_goal = current_portfolio / len(goal_analysis) if len(goal_analysis) > 0 and current_portfolio > 0 else 0
        
        # Calculate per-goal allocations - include existing portfolio column
        goal_sip_rows = [["Goal", "Required SIP", "Allocated SIP", "Existing Corpus", "Gap", "Status"]]
        
        for ga in goal_analysis:
            g_name = ga["name"][:20]  # Truncate long names
            required_sip = ga["required_sip"]
            
            # Proportional allocation from available surplus
            if total_required_sip > 0 and required_sip > 0:
                allocated_sip = (required_sip / total_required_sip) * available_for_new_sips
            else:
                allocated_sip = available_for_new_sips / max(1, len(goal_analysis))
            
            # Add share of existing SIP (proportionally)
            if total_monthly_sip > 0 and total_required_sip > 0:
                existing_sip_share = (required_sip / total_required_sip) * total_monthly_sip
                allocated_sip += existing_sip_share
            
            gap = max(0, required_sip - allocated_sip)
            
            if gap <= 0 or (allocated_sip >= required_sip * 0.95):
                status = "✓ Funded"
            elif allocated_sip >= required_sip * 0.5:
                status = "Partial"
            else:
                status = "Gap"
            
            goal_sip_rows.append([
                sanitize_pdf_text(g_name),
                f"Rs. {required_sip:,.0f}",
                f"Rs. {allocated_sip:,.0f}",
                f"Rs. {_format_indian_amount(portfolio_per_goal)}" if portfolio_per_goal > 0 else "-",
                f"Rs. {gap:,.0f}" if gap > 0 else "-",
                status
            ])
        
        goal_sip_table = Table(goal_sip_rows, hAlign="LEFT", colWidths=[100, 70, 70, 70, 60, 50])
        goal_sip_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#457B9D')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ]))
        story.append(goal_sip_table)
        story.append(Spacer(1, 8))
        
        # Context message
        if total_monthly_sip > 0:
            story.append(Paragraph(
                f"<i>Note: You are already investing Rs. {total_monthly_sip:,.0f}/month through existing SIPs. "
                f"After insurance provision, Rs. {available_for_new_sips:,.0f}/month can be added. "
                f"Total: Rs. {total_investing:,.0f}/month ({total_investing_pct:.1f}% of goal needs).</i>",
                styles["BodyText"]
            ))
        else:
            if monthly_surplus < 0:
                story.append(Paragraph(
                    f"<i>Note: Your expenses exceed income by Rs. {abs(monthly_surplus):,.0f}/month. "
                    f"Focus on reducing expenses or increasing income before new SIPs.</i>",
                    styles["BodyText"]
                ))
            else:
                story.append(Paragraph(
                    f"<i>Note: After insurance provision of Rs. {monthly_insurance_provision:,.0f}/month, "
                    f"Rs. {available_for_new_sips:,.0f}/month is available for goal SIPs.</i>",
                    styles["BodyText"]
                ))
        
        # Existing portfolio note
        if current_portfolio > 0:
            story.append(Paragraph(
                f"<i>Your existing portfolio of Rs. {_format_indian_amount(current_portfolio)} has been allocated "
                f"Rs. {_format_indian_amount(portfolio_per_goal)} per goal to offset target amounts.</i>",
                styles["BodyText"]
            ))
    
    # =========================================================================
    # REALISTIC ACTION PLAN - What you CAN do with what you HAVE
    # =========================================================================
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Your Realistic Action Plan</b>", styles["h2"]))
    
    # Calculate insurance premium estimates
    term_gap = max(0, required_term_cover - life_cover)
    health_gap = max(0, required_health_cover - health_cover)
    
    # Use age-adjusted premium rates (matching llm_sections.py PriorityAllocationEngine)
    # Term insurance: Rs. 500/lakh (age < 35), Rs. 700/lakh (age < 45), Rs. 1200/lakh (age >= 45)
    # Health insurance: Rs. 2500/lakh (age < 35), Rs. 3500/lakh (age < 45), Rs. 5000/lakh (age >= 45)
    client_age = _safe_float(age, 35)  # Default to 35 if age not available
    term_rate_per_lakh = 500 if client_age < 35 else (700 if client_age < 45 else 1200)
    health_rate_per_lakh = 2500 if client_age < 35 else (3500 if client_age < 45 else 5000)
    
    term_premium_yearly = (term_gap / 100000) * term_rate_per_lakh if term_gap > 0 else 0
    health_premium_yearly = (health_gap / 100000) * health_rate_per_lakh if health_gap > 0 else 0
    total_insurance_yearly = term_premium_yearly + health_premium_yearly
    monthly_insurance_equivalent = total_insurance_yearly / 12
    
    # Available for goals after existing SIP and insurance provision
    # Existing SIPs are already consuming part of the surplus
    surplus_after_insurance = max(0, monthly_surplus - total_monthly_sip - monthly_insurance_equivalent)
    
    # Phase 1: Insurance (if needed)
    if term_gap > 0 or health_gap > 0:
        story.append(Paragraph("<b>Phase 1: Protection First (Months 1-6)</b>", styles["BodyText"]))
        story.append(Paragraph("Before investing in goals, secure your family's protection:", styles["BodyText"]))
        
        if term_gap > 0:
            story.append(Paragraph(f"• Get Term Insurance: Rs. {_format_indian_amount(term_gap)} cover → ~Rs. {term_premium_yearly:,.0f}/year premium", styles["BodyText"]))
        if health_gap > 0:
            story.append(Paragraph(f"• Get Health Insurance: Rs. {_format_indian_amount(health_gap)} cover → ~Rs. {health_premium_yearly:,.0f}/year premium", styles["BodyText"]))
        
        story.append(Paragraph(f"<i>Total insurance cost: ~Rs. {total_insurance_yearly:,.0f}/year (Rs. {monthly_insurance_equivalent:,.0f}/month equivalent)</i>", styles["BodyText"]))
        story.append(Spacer(1, 8))
    
    # Phase 2: Goal SIPs with realistic allocation
    story.append(Paragraph("<b>Phase 2: Goal-Based SIPs (After Insurance)</b>", styles["BodyText"]))
    
    if len(goal_analysis) > 0 and surplus_after_insurance > 0:
        story.append(Paragraph(f"Available for goals after insurance: <b>Rs. {surplus_after_insurance:,.0f}/month</b>", styles["BodyText"]))
        story.append(Paragraph(f"Split across {len(goal_analysis)} goals proportionally:", styles["BodyText"]))
        story.append(Spacer(1, 8))
        
        # Build allocation table with projected outcomes
        allocation_rows = [["Goal", "Allocated SIP", "Required SIP", "What You'll Achieve", "Gap"]]
        
        for ga in goal_analysis:
            g_name = ga["name"][:20]  # Truncate long names
            g_target = ga["target"]
            g_horizon = ga["horizon"]
            required_sip = ga["required_sip"]
            
            # Proportional allocation
            if total_required_sip > 0 and required_sip > 0:
                allocated = (required_sip / total_required_sip) * surplus_after_insurance
            else:
                allocated = surplus_after_insurance / max(1, len(goal_analysis))
            
            allocated = max(0, allocated)
            
            # Calculate what this SIP will actually achieve
            if g_horizon and allocated > 0:
                # Future Value formula: FV = P * [((1+r)^n - 1) / r]
                # Using ~9% annual return (0.75% monthly)
                r = 0.0075
                n = int(g_horizon) * 12
                if n > 0:
                    projected_value = allocated * (((1 + r) ** n - 1) / r)
                else:
                    projected_value = 0
            else:
                projected_value = 0
            
            # Calculate achievement percentage
            achievement_pct = (projected_value / g_target * 100) if g_target > 0 else 0
            gap_amount = max(0, g_target - projected_value)
            
            allocation_rows.append([
                sanitize_pdf_text(g_name),
                f"Rs. {allocated:,.0f}",
                f"Rs. {required_sip:,.0f}",
                f"Rs. {_format_indian_amount(projected_value)} ({achievement_pct:.0f}%)",
                f"Rs. {_format_indian_amount(gap_amount)}"
            ])
        
        allocation_table = Table(allocation_rows, hAlign="LEFT", colWidths=[100, 80, 80, 120, 80])
        allocation_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1D3557')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ]))
        story.append(allocation_table)
        story.append(Spacer(1, 8))
        
        story.append(Paragraph("<b>Key Insight:</b> Starting with available surplus is better than waiting. Increase SIPs annually as income grows.", styles["BodyText"]))
    
    elif surplus_after_insurance <= 0:
        story.append(Paragraph("<i>After insurance provision, no surplus remains for goal SIPs. Focus on increasing income or reducing expenses first.</i>", styles["BodyText"]))
    
    else:
        story.append(Paragraph("<i>No goals defined. Add financial goals to see allocation recommendations.</i>", styles["BodyText"]))
    
    story.append(PageBreak())

    # ==========================================================================
    # PAGE 7: TAX OPTIMIZATION (if applicable)
    # ==========================================================================
    story.append(Paragraph("Page 7: Tax Optimization", styles["h1"]))
    
    itr = di.get("itr") or {}
    
    # Tax deduction limits
    TAX_LIMITS = {
        "80C": 150000,
        "80CCD_1B": 50000,
        "80D_self": 25000,
        "80D_parents": 50000,
    }
    
    if itr:
        gross_income = _safe_float(itr.get("gross_total_income"), 0)
        taxable_income = _safe_float(itr.get("taxable_income"), 0)
        tax_paid = _safe_float(itr.get("total_tax_paid"), 0)
        deductions = itr.get("deductions_claimed") or []
        total_deductions = gross_income - taxable_income if gross_income > 0 and taxable_income > 0 else sum(d.get("amount", 0) for d in deductions)
        
        story.append(Paragraph("Tax Profile (from ITR)", styles["h2"]))
        tax_rows = [
            ["Gross Income", f"Rs. {_format_indian_amount(gross_income)}" if gross_income > 0 else "-"],
            ["Total Deductions", f"Rs. {_format_indian_amount(total_deductions)}"],
            ["Taxable Income", f"Rs. {_format_indian_amount(taxable_income)}" if taxable_income > 0 else "-"],
            ["Tax Paid", f"Rs. {_format_indian_amount(tax_paid)}" if tax_paid > 0 else "-"],
        ]
        tax_table = Table(
            [["Parameter", "Value"]] + tax_rows,
            hAlign="LEFT",
            colWidths=[200, 300]
        )
        tax_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#457B9D')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ]))
        story.append(tax_table)
        story.append(Spacer(1, 12))
        
        # Parse claimed deductions into a lookup FIRST (needed for regime comparison)
        claimed = {}
        for d in deductions:
            section = str(d.get("section", "")).upper().strip()
            amount = _safe_float(d.get("amount"), 0)
            if "80C" in section and "80CCD" not in section:
                claimed["80C"] = claimed.get("80C", 0) + amount
            elif "80CCD" in section or "NPS" in section.upper():
                claimed["80CCD_1B"] = claimed.get("80CCD_1B", 0) + amount
            elif "80D" in section:
                claimed["80D"] = claimed.get("80D", 0) + amount
        
        # =====================================================
        # OLD vs NEW REGIME COMPARISON
        # =====================================================
        story.append(Paragraph("Old vs New Tax Regime Comparison", styles["h2"]))
        
        # Calculate regime comparison using potential max deductions
        current_80c = claimed.get("80C", 0)
        current_nps = claimed.get("80CCD_1B", 0)
        current_80d = claimed.get("80D", 0)
        
        # Also consider potential deductions if user maximizes them
        max_80c = TAX_LIMITS["80C"]
        max_nps = TAX_LIMITS["80CCD_1B"]
        max_80d = TAX_LIMITS["80D_self"] + TAX_LIMITS["80D_parents"]
        
        # Current scenario comparison
        regime_current = compute_regime_comparison(
            gross_income=gross_income,
            deductions_80c=current_80c,
            deductions_80d=current_80d,
            deductions_nps=current_nps
        )
        
        # Optimized scenario (if user max out deductions)
        regime_optimal = compute_regime_comparison(
            gross_income=gross_income,
            deductions_80c=max_80c,
            deductions_80d=max_80d,
            deductions_nps=max_nps
        )
        
        regime_rows = [
            ["Scenario", "Old Regime Tax", "New Regime Tax", "Better Regime"],
            [
                "Current (Your ITR)",
                f"Rs. {_format_indian_amount(regime_current['old_regime']['tax_liability'])}",
                f"Rs. {_format_indian_amount(regime_current['new_regime']['tax_liability'])}",
                regime_current['better_regime'].title()
            ],
            [
                "If Max Deductions",
                f"Rs. {_format_indian_amount(regime_optimal['old_regime']['tax_liability'])}",
                f"Rs. {_format_indian_amount(regime_optimal['new_regime']['tax_liability'])}",
                regime_optimal['better_regime'].title()
            ],
        ]
        
        regime_table = Table(
            regime_rows,
            hAlign="LEFT",
            colWidths=[130, 110, 110, 110]
        )
        
        # Highlight better regime
        better_color = colors.HexColor('#2D6A4F')  # Green
        regime_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1D3557')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('ALIGN', (1,1), (-1,-1), 'CENTER'),
        ]))
        story.append(regime_table)
        story.append(Spacer(1, 8))
        
        # Recommendation text
        story.append(Paragraph(f"<b>Recommendation:</b> {regime_current['recommendation']}", styles["BodyText"]))
        if regime_current['savings'] > 0:
            story.append(Paragraph(f"<i>Potential savings by choosing {regime_current['better_regime'].title()} Regime: Rs. {_format_indian_amount(regime_current['savings'])}</i>", styles["BodyText"]))
        story.append(Spacer(1, 12))
        
        # Calculate marginal tax rate (simplified)
        if taxable_income > 1000000:
            marginal_rate = 0.30
        elif taxable_income > 500000:
            marginal_rate = 0.20
        elif taxable_income > 250000:
            marginal_rate = 0.05
        else:
            marginal_rate = 0.0
        
        # Calculate gaps and recommendations
        recommendations = []
        total_potential_saving = 0
        
        # Only show deduction recommendations if Old Regime is ACTUALLY recommended
        # If New Regime is recommended, these deductions are IRRELEVANT
        show_deduction_advice = regime_current['better_regime'] == 'old'
        
        if show_deduction_advice:
            # 80C Gap
            gap_80c = max(0, TAX_LIMITS["80C"] - current_80c)
            if gap_80c > 0:
                saving_80c = gap_80c * marginal_rate
                total_potential_saving += saving_80c
                recommendations.append({
                    "section": "Section 80C",
                    "current": current_80c,
                    "limit": TAX_LIMITS["80C"],
                    "gap": gap_80c,
                    "saving": saving_80c,
                    "action": f"Invest Rs. {_format_indian_amount(gap_80c)} in ELSS/PPF to save Rs. {_format_indian_amount(saving_80c)} in tax"
                })
            
            # 80CCD(1B) Gap - NPS
            gap_nps = max(0, TAX_LIMITS["80CCD_1B"] - current_nps)
            if gap_nps > 0:
                saving_nps = gap_nps * marginal_rate
                total_potential_saving += saving_nps
                recommendations.append({
                    "section": "Section 80CCD(1B)",
                    "current": current_nps,
                    "limit": TAX_LIMITS["80CCD_1B"],
                    "gap": gap_nps,
                    "saving": saving_nps,
                    "action": f"Invest Rs. {_format_indian_amount(gap_nps)} in NPS (additional to 80C) to save Rs. {_format_indian_amount(saving_nps)}"
                })
            
            # 80D Gap - Health Insurance (context-aware message)
            limit_80d = TAX_LIMITS["80D_self"] + TAX_LIMITS["80D_parents"]
            gap_80d = max(0, limit_80d - current_80d)
            if gap_80d > 0:
                saving_80d = gap_80d * marginal_rate
                total_potential_saving += saving_80d
                # Show different message based on whether they have health insurance
                if health_cover == 0 and current_80d == 0:
                    action_text = f"Get health insurance (Rs. 10L cover) - also saves Rs. {_format_indian_amount(min(saving_80d, 25000 * marginal_rate))} in tax"
                else:
                    action_text = f"Increase health insurance premium by Rs. {_format_indian_amount(gap_80d)} to save Rs. {_format_indian_amount(saving_80d)}"
                recommendations.append({
                    "section": "Section 80D",
                    "current": current_80d,
                    "limit": limit_80d,
                    "gap": gap_80d,
                    "saving": saving_80d,
                    "action": action_text
                })
        
        # Show personalized recommendations
        if recommendations:
            story.append(Paragraph("Your Tax Saving Opportunities (Old Regime)", styles["h2"]))
            story.append(Paragraph(f"<b>If you stay in Old Regime, you can save up to Rs. {_format_indian_amount(total_potential_saving)} by maximizing deductions.</b>", styles["BodyText"]))
            story.append(Spacer(1, 8))
            
            rec_rows = []
            for rec in recommendations:
                rec_rows.append([
                    rec["section"],
                    f"Rs. {_format_indian_amount(rec['current'])} / Rs. {_format_indian_amount(rec['limit'])}",
                    f"Rs. {_format_indian_amount(rec['gap'])}",
                    f"Rs. {_format_indian_amount(rec['saving'])}"
                ])
            
            rec_table = Table(
                [["Section", "Used / Limit", "Gap", "Tax Saving"]] + rec_rows,
                hAlign="LEFT",
                colWidths=[120, 150, 100, 100]
            )
            rec_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2D6A4F')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ]))
            story.append(rec_table)
            story.append(Spacer(1, 12))
            
            story.append(Paragraph("<b>Recommended Actions:</b>", styles["BodyText"]))
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(f"{i}. {rec['action']}", styles["BodyText"]))
            story.append(Spacer(1, 8))
            
            story.append(Paragraph(f"<b>Total Potential Tax Saving: Rs. {_format_indian_amount(total_potential_saving)}</b>", styles["BodyText"]))
        elif regime_current['better_regime'] == 'new':
            # New Regime recommended - explain that deductions don't apply
            story.append(Paragraph("Tax Optimization under New Regime", styles["h2"]))
            story.append(Paragraph("<b>Since New Regime is recommended for you, Section 80C, 80D, and 80CCD deductions do NOT apply.</b>", styles["BodyText"]))
            story.append(Spacer(1, 8))
            story.append(Paragraph("Under New Regime, focus on:", styles["BodyText"]))
            story.append(Paragraph("• <b>Employer NPS contribution</b> (Section 80CCD-2) - still allowed up to 14% of basic", styles["BodyText"]))
            story.append(Paragraph("• <b>Standard deduction</b> of Rs. 75,000 - automatically applied", styles["BodyText"]))
            story.append(Paragraph("• <b>Tax-efficient investments</b> - equity funds (held >1 year for LTCG exemption)", styles["BodyText"]))
            story.append(Paragraph("• <b>Health insurance</b> - still important for protection, even without tax benefit", styles["BodyText"]))
        else:
            story.append(Paragraph("Great! You have maximized all major tax deductions.", styles["BodyText"]))
    else:
        story.append(Paragraph("No ITR data available. Upload ITR for personalized tax optimization analysis.", styles["BodyText"]))
        story.append(Spacer(1, 12))
        
        # Show general guidance for users without ITR
        story.append(Paragraph("General Tax Saving Opportunities", styles["h2"]))
        story.append(Paragraph("• Section 80C: Invest up to Rs. 1,50,000 in ELSS, PPF, or Tax-saving FDs", styles["BodyText"]))
        story.append(Paragraph("• Section 80CCD(1B): Additional Rs. 50,000 in NPS (over 80C limit)", styles["BodyText"]))
        story.append(Paragraph("• Section 80D: Health insurance premium up to Rs. 75,000 (self + parents)", styles["BodyText"]))
    
    story.append(PageBreak())

    # ==========================================================================
    # PAGE 8: ACTION ROADMAP WITH TIMELINE (NEW)
    # ==========================================================================
    story.append(Paragraph("Page 8: Your Action Roadmap", styles["h1"]))
    story.append(Paragraph("Timeline for Financial Actions", styles["h2"]))
    
    # Week-by-week actions
    week_labels = {
        "week_1": "WEEK 1: Protection Assessment",
        "week_2": "WEEK 2: Emergency Fund Review", 
        "week_3": "WEEK 3: Portfolio Review",
        "week_4": "WEEK 4: Tax Planning",
        "day_90": "DAY 90: Full Progress Review",
        "quarterly": "QUARTERLY: Ongoing Review",
    }
    
    for week_key, week_label in week_labels.items():
        actions = timeline.get(week_key, [])
        if actions:
            story.append(Paragraph(f"<b>{week_label}</b>", styles["BodyText"]))
            for action in actions:
                time_est = action.get("time_estimate", "")
                time_str = f" ({time_est})" if time_est else ""
                story.append(Paragraph(f"□ {action['action']}{time_str}", styles["BodyText"]))
            story.append(Spacer(1, 8))
    
    story.append(Spacer(1, 12))
    
    # What this report contains
    story.append(Paragraph("What This Report Contains", styles["h2"]))
    contents = [
        "Financial health assessment across 6 areas",
        "Protection gap analysis",
        "Portfolio risk evaluation",
        "Goal feasibility assessment",
        "Tax optimization opportunities",
        "Prioritized action roadmap",
    ]
    for c in contents:
        story.append(Paragraph(f"✓ {c}", styles["BodyText"]))
    story.append(Spacer(1, 12))
    
    # Disclaimers
    story.append(Paragraph("Important Notes", styles["h2"]))
    disclaimers = [
        "This is an educational financial plan, not personalized investment advice.",
        "Calculations based on standard assumptions and data provided.",
        "Product selection and execution are your decisions.",
        "Market conditions and regulations may change.",
        "Review quarterly or after significant life events.",
    ]
    for d in disclaimers:
        story.append(Paragraph(f"• {d}", styles["BodyText"]))
    story.append(Spacer(1, 12))
    
    # Review Schedule
    story.append(Paragraph("<b>Review Schedule:</b>", styles["BodyText"]))
    story.append(Paragraph("• IMMEDIATE: Complete Priority 1 & 2 (Protection + Emergency)", styles["BodyText"]))
    story.append(Paragraph("• 30 DAYS: Portfolio rebalancing + Tax planning", styles["BodyText"]))
    story.append(Paragraph("• 90 DAYS: Full progress review", styles["BodyText"]))
    story.append(Paragraph("• QUARTERLY: Recalculate Financial Health Score", styles["BodyText"]))
    story.append(Paragraph("• ANNUALLY: Comprehensive plan update", styles["BodyText"]))

    # Build PDF with header and footer on all pages
    def header_and_footer(canvas, doc):
        header(canvas, doc)
        footer(canvas, doc)
    
    doc.build(story, onFirstPage=header_and_footer, onLaterPages=header_and_footer)

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
    if s == "high":
        return "Debt load stressing cash flows; consider refinancing or accelerated repayment."
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


# =============================================================================
# Meerkat Financial Health Report v5 - presentation layer
# =============================================================================

MEERKAT_BG = colors.HexColor("#F5F0E8")
MEERKAT_NAVY = colors.HexColor("#1A1A2E")
MEERKAT_RED = colors.HexColor("#C0392B")
MEERKAT_ORANGE = colors.HexColor("#E67E22")
MEERKAT_GREEN = colors.HexColor("#27AE60")
MEERKAT_CARD = colors.HexColor("#FAFAFA")
MEERKAT_GOLD = colors.HexColor("#B8860B")
MEERKAT_LINE = colors.HexColor("#DDDDDD")

IN_APP_REPORT_BLOCKLIST = ("in-app", "nudge", "push notification", "mobile alert")


def _strip_in_app_report_text(value):
    if isinstance(value, str):
        return "\n".join(
            line for line in value.splitlines()
            if not any(term in line.lower() for term in IN_APP_REPORT_BLOCKLIST)
        ).strip()
    if isinstance(value, list):
        return [_strip_in_app_report_text(v) for v in value if _strip_in_app_report_text(v)]
    if isinstance(value, dict):
        return {k: _strip_in_app_report_text(v) for k, v in value.items()}
    return value


def _report_styles():
    base = getSampleStyleSheet()
    return {
        "cover_name": ParagraphStyle("mk_cover_name", parent=base["Title"], fontName="Times-Roman", fontSize=34, leading=40, textColor=MEERKAT_NAVY, spaceAfter=8, alignment=TA_LEFT),
        "title": ParagraphStyle("mk_title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=MEERKAT_NAVY, alignment=TA_LEFT),
        "cover_title": ParagraphStyle("mk_cover_title", parent=base["Title"], fontName="Times-Roman", fontSize=32, leading=38, textColor=MEERKAT_NAVY, alignment=TA_LEFT),
        "cover_subtitle": ParagraphStyle("mk_cover_subtitle", parent=base["BodyText"], fontName="Times-Italic", fontSize=11, leading=16, textColor=colors.HexColor("#555555"), alignment=TA_LEFT),
        "section": ParagraphStyle("mk_section", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=colors.HexColor("#666666"), uppercase=True),
        "h2": ParagraphStyle("mk_h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=MEERKAT_NAVY, spaceBefore=4, spaceAfter=6),
        "cover_status": ParagraphStyle("mk_cover_status", parent=base["Heading2"], fontName="Times-Roman", fontSize=24, leading=28, textColor=MEERKAT_ORANGE, alignment=TA_LEFT, spaceBefore=4, spaceAfter=12),
        "body": ParagraphStyle("mk_body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5, leading=12, textColor=colors.HexColor("#222222")),
        "small": ParagraphStyle("mk_small", parent=base["BodyText"], fontName="Helvetica", fontSize=8, leading=10, textColor=colors.HexColor("#666666")),
        "label": ParagraphStyle("mk_label", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=colors.HexColor("#666666")),
        "table": ParagraphStyle("mk_table", parent=base["BodyText"], fontName="Helvetica", fontSize=8.2, leading=10),
        "table_head": ParagraphStyle("mk_table_head", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=8.2, leading=10, textColor=colors.white),
        "table_head_dark": ParagraphStyle("mk_table_head_dark", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=8.2, leading=10, textColor=colors.HexColor("#475569")),
    }


class CanvasBlock(Flowable):
    def __init__(self, width, height, draw_fn):
        super().__init__()
        self.width = width
        self.height = height
        self.draw_fn = draw_fn

    def wrap(self, avail_width, avail_height):
        return self.width, self.height

    def draw(self):
        self.draw_fn(self.canv, 0, 0, self.width, self.height)


class DrawingFlowable(Flowable):
    def __init__(self, draw_fn, width, height, *args, **kwargs):
        super().__init__()
        self.draw_fn = draw_fn
        self.width = width
        self.height = height
        self.args = args
        self.kwargs = kwargs

    def wrap(self, avail_width, avail_height):
        return self.width, self.height

    def draw(self):
        self.draw_fn(self.canv, 0, 0, *self.args, **self.kwargs)


class TagFlowable(Flowable):
    def __init__(self, label, height=15, bg_tint=False):
        super().__init__()
        self.label = str(label or "-")
        self.bg_tint = bg_tint
        self.width = len(self.label) * 5.5 + 16
        self.height = height

    def wrap(self, avail_width, avail_height):
        return self.width, self.height

    def draw(self):
        draw_tag(self.canv, 0, 1, self.label, bg_tint=self.bg_tint)


class KPIFlowable(Flowable):
    def __init__(self, tiles, width=500, height=70, card_height=54, gap=9, card_y=8):
        super().__init__()
        self.tiles = tiles
        self.width = width
        self.height = height
        self.card_height = card_height
        self.gap = gap
        self.card_y = card_y

    def wrap(self, avail_width, avail_height):
        return self.width, self.height

    def draw(self):
        count = max(1, len(self.tiles))
        tile_w = (self.width - self.gap * (count - 1)) / count
        for idx, tile in enumerate(self.tiles):
            x = idx * (tile_w + self.gap)
            draw_kpi_card(
                self.canv,
                x, self.card_y, tile_w, self.card_height,
                tile.get("label"),
                tile.get("value"),
                sub_value=tile.get("note"),
                sub_colour=tile.get("note_color")
            )


class ScoreBarFlowable(Flowable):
    def __init__(self, label, score, band=None, width=500, height=24):
        super().__init__()
        self.label = label
        self.score = max(0, min(100, _num(score, 0)))
        self.band = band or _urgency(self.score)
        self.width = width
        self.height = height

    def wrap(self, avail_width, avail_height):
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.setFillColor(MEERKAT_NAVY)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(0, 13, self.label)
        bar_x = 140
        bar_w = 245
        colour = MEERKAT_RED if self.score < 40 else MEERKAT_ORANGE if self.score < 75 else MEERKAT_GREEN
        c.setFillColor(colors.HexColor("#E8E8E8"))
        c.rect(bar_x, 9, bar_w, 8, stroke=0, fill=1)
        c.setFillColor(colour)
        c.rect(bar_x, 9, bar_w * self.score / 100, 8, stroke=0, fill=1)
        c.setFillColor(MEERKAT_NAVY)
        c.setFont("Helvetica-Bold", 8)
        c.drawRightString(self.width - 72, 13, f"{self.score:.0f}/100")
        draw_tag(c, self.width - 62, 7, self.band)


class GaugePairFlowable(Flowable):
    def __init__(self, current, target, ar=None, width=500, height=132):
        super().__init__()
        self.current = current
        self.target = target
        self.ar = ar
        self.width = width
        self.height = height

    def wrap(self, avail_width, avail_height):
        return self.width, self.height

    def draw(self):
        c = self.canv
        band = (self.ar or {}).get("recommendedEquityBand") or {}
        b_min = _num(band.get("min"), 40)
        b_max = _num(band.get("max"), 60)
        draw_equity_gauges(c, 130, 370, 60, 35, _num(self.current, 0), _num(self.target, 0), b_min, b_max)


class CoverageFlowable(Flowable):
    def __init__(self, pct, width=92, height=12):
        super().__init__()
        self.pct = pct
        self.width = width
        self.height = height

    def wrap(self, avail_width, avail_height):
        return self.width, self.height

    def draw(self):
        draw_coverage_bar(self.canv, 0, 0, self.width, self.pct, height=self.height)


def draw_saving_card(canvas, x, y, w, h, amount):
    canvas.saveState()
    # Background
    canvas.setFillColor(colors.HexColor("#F9F7F2"))
    canvas.roundRect(x, y, w, h, radius=8, fill=1, stroke=0)
    canvas.setStrokeColor(colors.HexColor("#E2E8F0"))
    canvas.setLineWidth(0.5)
    canvas.roundRect(x, y, w, h, radius=8, fill=0, stroke=1)
    
    # Label
    canvas.setFillColor(colors.HexColor("#A8813C"))
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawCentredString(x + w/2, y + h - 24, "POTENTIAL ANNUAL SAVING")
    
    # Value
    canvas.setFillColor(colors.HexColor("#27AE60"))
    canvas.setFont("Times-Bold", 32)
    canvas.drawCentredString(x + w/2, y + h/2 - 10, f"{_fmt_rs(amount, False)}")
    
    # Subtext
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(x + w/2, y + 20, "by switching to the New Tax Regime")
    
    canvas.restoreState()

class SavingCardFlowable(Flowable):
    def __init__(self, amount, width=220, height=140):
        super().__init__()
        self.amount = amount
        self.width = width
        self.height = height
    def wrap(self, aw, ah): return self.width, self.height
    def draw(self): draw_saving_card(self.canv, 0, 0, self.width, self.height, self.amount)


class InsightFlowable(Flowable):
    def __init__(self, title, text, width=500, height=80):
        super().__init__()
        self.title = title
        self.text = text
        self.width = width
        self.height = height

    def wrap(self, avail_width, avail_height):
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.saveState()
        
        # Background
        c.setFillColor(colors.HexColor("#F9F7F2"))
        c.roundRect(0, 0, self.width, self.height, radius=6, fill=1, stroke=0)
        
        # Gold bar on left
        c.setFillColor(colors.HexColor("#A8813C"))
        c.rect(0, 0, 4, self.height, fill=1, stroke=0)
        
        # Title
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(16, self.height - 24, self.title)
        
        # Text
        styles = _report_styles()
        p = Paragraph(self.text, ParagraphStyle("insight", parent=styles["small"], fontSize=9, leading=14))
        w, h = p.wrap(self.width - 40, self.height - 40)
        p.drawOn(c, 16, self.height - 32 - h)
        
        c.restoreState()


def _fmt_rs(amount, compact=True):
    try:
        val = float(amount or 0)
    except Exception:
        val = 0
    return f"Rs. {_format_indian_amount(val)}" if compact else f"Rs. {val:,.0f}"


def _num(value, default=0.0):
    return _safe_float(value, default)


def _portfolio_equity(portfolio):
    return _num(portfolio.get("equity_pct") or portfolio.get("equity") or portfolio.get("equity_percentage"), 0)


def _display_monthly_surplus(client_facts):
    income = client_facts.get("income") or {}
    bank = client_facts.get("bank") or {}
    monthly_income = _num(income.get("annualIncome"), 0) / 12
    computed = monthly_income - _num(income.get("monthlyExpenses"), 0) - _num(income.get("monthlyEmi"), 0)
    bank_net = _num(bank.get("net_cashflow"), 0)
    if monthly_income > 0 and abs(bank_net) <= monthly_income * 1.5:
        return bank_net
    return computed


def _recommended_band_text(advanced_risk):
    band = (advanced_risk or {}).get("recommendedEquityBand") or {}
    if isinstance(band, dict):
        return f"{_num(band.get('min'), 40):.0f}-{_num(band.get('max'), 60):.0f}%"
    return str(band or "40-60%")


def _recommended_band_mid(advanced_risk):
    if (advanced_risk or {}).get("recommendedEquityMid") is not None:
        return _num(advanced_risk.get("recommendedEquityMid"), 50)
    band = (advanced_risk or {}).get("recommendedEquityBand") or {}
    if isinstance(band, dict):
        return (_num(band.get("min"), 40) + _num(band.get("max"), 60)) / 2
    return 50


def _urgency(score):
    score = _num(score, 0)
    if score < 40:
        return "IMMEDIATE"
    if score < 75:
        return "HIGH"
    return "MAINTAIN"


def draw_tag(canvas, x, y, label, font_size=8, bg_tint=False):
    # y is baseline of text
    pad_x, pad_y = 6, 3
    text_width = canvas.stringWidth(label, 'Helvetica-Bold', font_size)
    rect_w = text_width + pad_x * 2
    rect_h = font_size + pad_y * 2
    colour_map = {
        'IMMEDIATE': '#C0392B', 'CRITICAL': '#C0392B', 'UNDERINSURED': '#C0392B', 'URGENT': '#C0392B',
        'HIGH': '#E67E22', 'NEEDS ATTENTION': '#E67E22', 'ATTENTION': '#E67E22', 'ASSESS & IMPROVE': '#E67E22', 'MODERATE': '#E67E22', 'GAP': '#E67E22', 'LOW': '#E67E22', 'UPGRADE RECOMMENDED': '#E67E22', 'REVIEW': '#E67E22',
        'GOOD': '#27AE60', 'MAINTAIN': '#27AE60', 'WELL OPTIMISED': '#27AE60', 'ADEQUATE': '#27AE60', 'HEALTHY': '#27AE60', 'FUNDED': '#27AE60', 'COMFORTABLE': '#27AE60',
    }
    hex_col = colour_map.get(label.upper(), '#888888')
    r, g, b = int(hex_col[1:3],16)/255, int(hex_col[3:5],16)/255, int(hex_col[5:7],16)/255
    canvas.setStrokeColorRGB(r, g, b)
    if bg_tint:
        canvas.setFillColorRGB(r * 0.15 + 0.85, g * 0.15 + 0.85, b * 0.15 + 0.85)
    else:
        canvas.setFillColorRGB(1, 1, 1)
    canvas.setLineWidth(0.8)
    canvas.roundRect(x, y - pad_y, rect_w, rect_h, radius=rect_h/2, fill=1, stroke=1)
    canvas.setFillColorRGB(r, g, b)
    canvas.setFont('Helvetica-Bold', font_size)
    canvas.drawString(x + pad_x, y + 1, label)


def draw_kpi_card(canvas, x, y, w, h, label, value, sub_value=None, sub_colour=None):
    is_active_sip = ("ACTIVE SIP" in label.upper())
    is_red = ("EXPENSES" in label.upper())
    is_tan = ("INSURANCE" in label.upper())
    
    if is_active_sip:
        bg_color = (0.99, 0.95, 0.94)  # Very light red/pink
        border_color = (0.95, 0.8, 0.77)
    else:
        bg_color = (1, 1, 1)
        border_color = (0.92, 0.92, 0.92)
        
    canvas.setFillColorRGB(*bg_color)
    canvas.setStrokeColorRGB(*border_color)
    canvas.setLineWidth(0.8)
    canvas.roundRect(x, y, w, h, radius=6, fill=1, stroke=1)
    
    # Label
    canvas.setFillColorRGB(0.5, 0.5, 0.5)
    canvas.setFont('Helvetica-Bold', 6)
    canvas.drawCentredString(x + w/2, y + h - 14, label.upper())
    
    # Value
    if is_active_sip or is_red:
        canvas.setFillColor(colors.HexColor("#C0392B")) # Meerkat Red
    elif is_tan:
        canvas.setFillColor(colors.HexColor("#A8813C")) # Meerkat Gold
    elif "SURPLUS" in label.upper() or "AVAILABLE" in label.upper() or "INCOME" in label.upper():
        canvas.setFillColor(colors.HexColor("#27AE60")) # Meerkat Green
    elif "PORTFOLIO" in label.upper():
        canvas.setFillColor(colors.HexColor("#2C3E50")) # Meerkat Navy
    else:
        canvas.setFillColor(colors.HexColor("#27AE60")) # Meerkat Green
        
    canvas.setFont('Helvetica-Bold', 12)
    canvas.drawCentredString(x + w/2, y + h - 34, value)
    
    # Sub value
    if sub_value:
        r, g, b = (0.9, 0.49, 0.13) if sub_colour == 'orange' else (0.4, 0.4, 0.4)
        if is_active_sip:
            r, g, b = (0.8, 0.4, 0.2)
        canvas.setFillColorRGB(r, g, b)
        canvas.setFont('Helvetica', 8)
        canvas.drawString(x + 12, y + h - 56, sub_value)


def draw_score_bar(canvas, x, y, bar_width, label, score, tag_label):
    label_w = 130
    score_w = 40
    actual_bar_w = bar_width - label_w - score_w - 20
    # Label
    canvas.setFillColorRGB(0.1, 0.1, 0.18)
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(x, y + 2, label)
    bx = x + label_w
    # Background bar
    canvas.setFillColorRGB(0.91, 0.91, 0.91)
    canvas.rect(bx, y, actual_bar_w, 10, fill=1, stroke=0)
    # Foreground bar
    fill_w = (score / 100) * actual_bar_w
    if score < 40:
        canvas.setFillColorRGB(0.75, 0.22, 0.17)
    elif score < 75:
        canvas.setFillColorRGB(0.9, 0.49, 0.13)
    else:
        canvas.setFillColorRGB(0.15, 0.68, 0.38)
    canvas.rect(bx, y, fill_w, 10, fill=1, stroke=0)
    # Score text
    canvas.setFillColorRGB(0.1, 0.1, 0.18)
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(bx + actual_bar_w + 6, y + 2, f'{score}/100')
    # Tag
    draw_tag(canvas, bx + actual_bar_w + 50, y, tag_label)


def _draw_dimension_score_bars(canvas, left_margin, starting_y, content_width, dimension_rows):
    bar_x = left_margin + 160
    bar_w = content_width - 160 - 80
    y = starting_y
    for label, score, tag in dimension_rows:
        canvas.setFillColorRGB(0.2, 0.2, 0.2)
        canvas.setFont("Helvetica", 10)
        canvas.drawString(left_margin, y + 2, str(label))
        canvas.setFillColorRGB(0.92, 0.92, 0.92)
        canvas.roundRect(bar_x, y, bar_w, 8, radius=4, fill=1, stroke=0)
        score_val = _num(score, 0)
        fill_w = (score_val / 100.0) * bar_w
        if score_val < 40:
            color = (0.75, 0.22, 0.17)
            tag_label = "Critical"
        elif score_val < 75:
            color = (0.9, 0.49, 0.13)
            tag_label = "Attention"
        else:
            color = (0.15, 0.68, 0.38)
            tag_label = "Good"
        canvas.setFillColorRGB(*color)
        canvas.roundRect(bar_x, y, fill_w, 8, radius=4, fill=1, stroke=0)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(bar_x + bar_w + 12, y + 1, str(int(score_val)))
        draw_tag(canvas, bar_x + bar_w + 40, y, tag_label, bg_tint=True)
        y -= 28


def draw_arc_gauge(canvas, cx, cy, radius, score, max_score=100,
                   label='Financial Health', score_font_size=28, colour=None):
    stroke_bg = 6      # background arc stroke width
    stroke_fg = 9      # foreground arc stroke width
    start_angle = 225
    total_sweep = 270

    # Background arc
    canvas.setStrokeColorRGB(0.87, 0.87, 0.87)
    canvas.setLineWidth(stroke_bg)
    canvas.setLineCap(1)  # round caps for smoother arc ends
    canvas.arc(cx - radius, cy - radius, cx + radius, cy + radius,
               startAng=start_angle, extent=-total_sweep)

    # Foreground arc
    sweep = (score / max_score) * total_sweep
    if colour:
        canvas.setStrokeColor(colour)
    else:
        if score < 40:
            canvas.setStrokeColorRGB(0.75, 0.22, 0.17)
        elif score < 75:
            canvas.setStrokeColorRGB(0.9, 0.49, 0.13)
        else:
            canvas.setStrokeColorRGB(0.15, 0.68, 0.38)
    canvas.setLineWidth(stroke_fg)
    canvas.setLineCap(1)  # round caps for smoother arc ends
    canvas.arc(cx - radius, cy - radius, cx + radius, cy + radius,
               startAng=start_angle, extent=-sweep)

    # Score number
    canvas.setFillColorRGB(0.1, 0.1, 0.18)
    canvas.setFont('Helvetica-Bold', score_font_size)
    
    if max_score == 100 and label == 'Financial Health':
        canvas.drawCentredString(cx, cy + 4, str(int(score)))
        canvas.setFont('Helvetica', score_font_size * 0.35)
        canvas.setFillColorRGB(0.5, 0.5, 0.5)
        canvas.drawCentredString(cx, cy - 12, "/100")
    else:
        canvas.drawCentredString(cx, cy + 8, str(int(score)))
        if label:
            canvas.setFillColorRGB(0.5, 0.5, 0.5)
            canvas.setFont('Helvetica', 9)
            canvas.drawCentredString(cx, cy - 8, label)


def draw_coverage_bar(canvas, x, y, width, coverage_pct, inner_text="", bar_color=None, height=12):
    canvas.setFillColorRGB(0.88, 0.90, 0.88)
    h = height
    canvas.roundRect(x, y, width, h, radius=h/2, fill=1, stroke=0)
    if coverage_pct > 0:
        fill_w = min(coverage_pct / 100, 1.0) * width
        if bar_color:
            canvas.setFillColorRGB(*bar_color)
        else:
            if coverage_pct < 50:
                canvas.setFillColorRGB(0.75, 0.22, 0.17)
            elif coverage_pct < 80:
                canvas.setFillColorRGB(0.9, 0.49, 0.13)
            else:
                canvas.setFillColorRGB(0.15, 0.68, 0.38)
        fill_w = max(fill_w, h)
        canvas.roundRect(x, y, fill_w, h, radius=h/2, fill=1, stroke=0)
    
    if inner_text:
        canvas.setFillColorRGB(1, 1, 1)
        canvas.setFont('Helvetica-Bold', 8)
        canvas.drawString(x + 8, y + 3, inner_text)


def draw_equity_gauges(canvas, cx_left, cx_right, cy, radius, current_pct, target_pct, band_low, band_high):
    for cx, pct, is_current in [(cx_left, current_pct, True), (cx_right, target_pct, False)]:
        canvas.setStrokeColorRGB(0.87, 0.87, 0.87)
        canvas.setLineWidth(6)
        canvas.setLineCap(1)
        canvas.arc(cx - radius, cy - radius, cx + radius, cy + radius, startAng=-45, extent=270)
        if is_current and (_num(pct, 0) < _num(band_low, 0) or _num(pct, 0) > _num(band_high, 0)):
            canvas.setStrokeColorRGB(0.75, 0.22, 0.17)
        else:
            canvas.setStrokeColorRGB(0.15, 0.68, 0.38)
        canvas.setLineWidth(9)
        canvas.setLineCap(1)
        sweep = (_num(pct, 0) / 100.0) * 270
        canvas.arc(cx - radius, cy - radius, cx + radius, cy + radius, startAng=-45, extent=sweep)
        canvas.setFillColorRGB(0.1, 0.1, 0.18)
        canvas.setFont("Helvetica-Bold", 18)
        canvas.drawCentredString(cx, cy + 4, f"{int(_num(pct, 0))}%")
        canvas.setFillColorRGB(0.5, 0.5, 0.5)
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(cx, cy - 12, "Current Equity" if is_current else "Target Equity")
    canvas.setFillColorRGB(0.6, 0.6, 0.6)
    canvas.setFont("Helvetica", 16)
    canvas.drawCentredString((cx_left + cx_right) / 2, cy, "→")


def _tag(label, bg_tint=False):
    return ("__TAG__", label, bg_tint)


def _page_header(section_no, title, sublabel):
    styles = _report_styles()
    return [
        Paragraph(str(sublabel or "").upper(), styles["label"]),
        Spacer(1, 3),
        draw_section_badge(section_no, title),
    ]


def draw_section_badge(section_no, title, width=500):
    styles = _report_styles()
    table = Table(
        [[Paragraph(section_no, styles["table_head"]), Paragraph(title.upper(), styles["table_head"])]],
        colWidths=[38, width - 38],
        hAlign="LEFT",
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), MEERKAT_NAVY),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#777777")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _styled_table(rows, col_widths, header=True, style_type="default"):
    styles = _report_styles()
    data = []
    for r, row in enumerate(rows):
        rendered = []
        for cell in row:
            if isinstance(cell, Flowable):
                rendered.append(cell)
            elif isinstance(cell, tuple) and len(cell) >= 2 and cell[0] == "__TAG__":
                bg_tint = cell[2] if len(cell) > 2 else False
                rendered.append(TagFlowable(cell[1], bg_tint=bg_tint))
            else:
                if style_type == "light":
                    style = styles["label"] if (header and r == 0) else styles["table"]
                    if header and r == 0:
                        rendered.append(Paragraph(f"<font color='#A8813C'>{sanitize_pdf_text(str(cell))}</font>", style))
                    else:
                        rendered.append(Paragraph(sanitize_pdf_text(str(cell)), style))
                else:
                    rendered.append(Paragraph(sanitize_pdf_text(str(cell)), styles["table_head" if header and r == 0 else "table"]))
        data.append(rendered)
    table = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1 if header else 0)
    
    if style_type == "light":
        styles_list = [
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#EEEEEE")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]
        if header:
            styles_list.append(("LINEABOVE", (0, 0), (-1, 0), 0.5, colors.HexColor("#EAE2D6")))
            styles_list.append(("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#EAE2D6")))
            styles_list.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FAFAFA")))
        styles_list.append(("LEFTPADDING", (0, 0), (0, -1), 16))
        styles_list.append(("RIGHTPADDING", (-1, 0), (-1, -1), 16))
        table.setStyle(TableStyle(styles_list))
    elif style_type == "compact_light":
        table.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#EEEEEE")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
    elif style_type == "light_right":
        table.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#EEEEEE")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ]))

    else:
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), MEERKAT_NAVY if header else MEERKAT_CARD),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white if header else colors.black),
            ("GRID", (0, 0), (-1, -1), 0.45, MEERKAT_LINE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
    return table


def _llm_paragraph(narratives, key):
    sec = ((narratives or {}).get(key) or {})
    for text in (sec.get("paragraphs") or []) + (sec.get("bullets") or []):
        cleaned = _strip_in_app_report_text(str(text or ""))
        if cleaned:
            return cleaned
    return ""


def _allocation_output(client_facts):
    analysis = client_facts.get("analysis") or {}
    income = client_facts.get("income") or {}
    insurance = client_facts.get("insurance") or {}
    personal = client_facts.get("personal") or {}
    portfolio = client_facts.get("portfolio") or {}
    monthly_income = _num(income.get("annualIncome"), 0) / 12
    monthly_surplus = _display_monthly_surplus(client_facts)
    required_life = _num((analysis.get("_diagnostics") or {}).get("requiredLifeCover"), monthly_income * 12 * 10)
    current_life = _num(insurance.get("lifeCover"), 0)
    current_health = _num(insurance.get("healthCover"), 0)
    recommended_health = 1500000 if _num(personal.get("dependents_count"), 0) > 0 else 1000000
    goals = []
    for g in client_facts.get("goals") or []:
        ideal = compute_goal_sip(g.get("target_amount"), g.get("horizon_years"), g.get("risk_tolerance") or (analysis.get("advancedRisk") or {}).get("finalCategory") or "moderate") or 0
        goals.append({
            "name": g.get("name") or "Goal",
            "target_amount": _num(g.get("target_amount"), 0),
            "horizon_years": g.get("horizon_years"),
            "ideal_sip": ideal,
            "risk_category": g.get("risk_tolerance") or "moderate",
        })
    try:
        out = PriorityAllocationEngine.compute_allocation(
            monthly_surplus=max(0, monthly_surplus),
            term_insurance_gap=max(0, required_life - current_life),
            health_insurance_gap=max(0, recommended_health - current_health),
            goals=goals,
            age=int(_num(personal.get("age"), 35)),
            has_dependents=_num(personal.get("dependents_count"), 0) > 0,
            emergency_fund_target=_num(income.get("monthlyExpenses"), 0) * 6,
            existing_sip_commitments=_num(portfolio.get("total_monthly_sip") or portfolio.get("monthly_sip"), 0),
        )
    except Exception:
        out = {}
    goal_rows = out.get("goal_sip_table") or []
    combined = sum(_num(g.get("shortfall"), 0) for g in goal_rows) or sum(_num(g.get("ideal_sip"), 0) for g in goals)
    out.setdefault("goal_sip_table", goal_rows)
    out.setdefault("combined_shortfall", combined)
    out.setdefault("insurance_provision", out.get("insurance_sip_monthly") or 0)
    out.setdefault("available_for_goals", out.get("remaining_for_goals") or 0)
    if not out.get("goal_sip_table") and goals:
        available = _num(out.get("available_for_goals"), 0)
        total_ideal = sum(_num(g.get("ideal_sip"), 0) for g in goals)
        fallback_rows = []
        for g in goals:
            ideal = _num(g.get("ideal_sip"), 0)
            allocated = available * ideal / total_ideal if total_ideal > 0 else 0
            fallback_rows.append({
                "name": g.get("name"),
                "target_amount": g.get("target_amount"),
                "horizon_years": g.get("horizon_years"),
                "ideal_sip": ideal,
                "allocated_sip": allocated,
                "shortfall": max(0, ideal - allocated),
                "coverage_pct": (allocated / ideal * 100) if ideal else 100,
                "status": "Gap" if allocated < ideal else "Funded",
            })
        out["goal_sip_table"] = fallback_rows
        out["combined_shortfall"] = sum(_num(g.get("shortfall"), 0) for g in fallback_rows)
    return out


def build_page_cover(client_facts, allocation_output, narratives=None):
    styles = _report_styles()
    analysis = client_facts.get("analysis") or {}
    ihs = analysis.get("ihs") or {}
    breakdown = ihs.get("breakdown") or {}
    name = (client_facts.get("personal") or {}).get("name") or "Client"
    alerts = sorted(breakdown.items(), key=lambda kv: _num((kv[1] or {}).get("score"), 0))[:3]
    story = []
    top = []
    if os.path.exists(LOGO_PATH):
        logo = Image(LOGO_PATH, width=1.0 * inch, height=0.42 * inch)
        
        def draw_badge(c, x, y, w, h):
            c.saveState()
            c.setStrokeColor(colors.HexColor("#E3D8C7"))
            c.setFillColor(colors.HexColor("#FDF8E7"))
            c.roundRect(x, y, w, h, 8, stroke=1, fill=1)
            c.setFillColor(colors.HexColor("#A8813C"))
            c.setFont("Helvetica", 8)
            c.drawCentredString(x + w/2, y + h/2 - 3, "Confidential")
            c.restoreState()
            
        badge = CanvasBlock(60, 16, draw_badge)
        date_str = f"<font size='9' color='#2C3E50'><b>{datetime.now().strftime('%d %B %Y')}</b></font><br/><font size='8' color='#6E8094'>{datetime.now().strftime('%I:%M %p IST')}</font>"
        date_para = Paragraph(date_str, ParagraphStyle("mk_cover_date", parent=styles["small"], alignment=TA_RIGHT))
        
        right_table = Table([[date_para, badge]], colWidths=[120, 60])
        right_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (0, 0), "RIGHT"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ]))
        
        top.append([logo, right_table])
        table = Table(top, colWidths=[300, 190])
        table.setStyle(TableStyle([("ALIGN", (1, 0), (1, 0), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(table)
        story.append(Spacer(1, 40))
        
    subtitle = "A comprehensive analysis across <font color='#2C3E50'><b>Six Critical Dimensions</b></font> of your financial wellbeing — designed to help you take clear, prioritised action."
    title = "Your Financial<br/><font name='Times-Italic' color='#B8860B'>Health Report</font>"
    story += [
        Paragraph("<font size='9' color='#88A0B9' name='Helvetica-Bold'>PREPARED FOR</font>", ParagraphStyle("mk_pf_left", parent=styles["cover_subtitle"], alignment=TA_LEFT, spaceAfter=4)),
        Paragraph(f"{sanitize_pdf_text(name)}", ParagraphStyle("mk_cover_name_left", parent=styles["cover_name"], alignment=TA_LEFT)),
        Spacer(1, 10),
        CanvasBlock(500, 1, lambda c, x, y, w, h: [c.setStrokeColor(colors.HexColor("#E3D8C7")), c.setLineWidth(0.8), c.line(0, 0, w, 0)]),
        Spacer(1, 18),
        Paragraph(f"{title}", ParagraphStyle("mk_cover_title_left", parent=styles["cover_title"], alignment=TA_LEFT)),
        Spacer(1, 8),
        Paragraph(f"{subtitle}", ParagraphStyle("mk_cover_sub_left", parent=styles["cover_subtitle"], alignment=TA_LEFT)),
        Spacer(1, 30),
    ]
    status_label = sanitize_pdf_text(str(ihs.get("band") or "Needs Attention"))
    alert_style = ParagraphStyle("mk_cover_alert", parent=styles["body"], alignment=TA_LEFT, textColor=colors.HexColor("#5E738A"), leading=14)
    
    alert_block = [
        Paragraph("<font size='9' color='#88A0B9' name='Helvetica-Bold'>OVERALL STATUS</font>", ParagraphStyle("mk_status_label", parent=styles["small"], spaceAfter=6)),
        Paragraph(status_label, ParagraphStyle("mk_cover_status_left", parent=styles["cover_status"], alignment=TA_LEFT))
    ]
    
    for key, _ in alerts:
        # Drawing a small red exclamation in a circle
        def draw_alert_icon(c, x, y, w, h):
            c.saveState()
            c.setFillColor(colors.HexColor("#FDF1F0"))
            c.setStrokeColor(colors.HexColor("#F1B2AD"))
            c.circle(x + 6, y + 6, 6, fill=1, stroke=1)
            c.setFillColor(colors.HexColor("#CB3E2D"))
            c.setFont("Helvetica-Bold", 8)
            c.drawCentredString(x + 6, y + 3, "!")
            c.restoreState()
            
        icon = CanvasBlock(16, 12, draw_alert_icon)
        text = Paragraph(f"{key.replace('_', ' ').capitalize()} needs attention", alert_style)
        alert_table = Table([[icon, text]], colWidths=[20, 260])
        alert_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 0)]))
        alert_block.append(alert_table)
        alert_block.append(Spacer(1, 4))
        
    status_table = Table(
        [[CanvasBlock(180, 160, lambda c, x, y, w, h: draw_arc_gauge(c, 70, 90, 56, ihs.get("score"), score_font_size=36)), Table([[x] for x in alert_block], colWidths=[280])]],
        colWidths=[180, 310],
    )
    status_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(status_table)
    story += [Spacer(1, 58), Paragraph("11-page analysis covering protection, portfolio, liquidity, goals, tax and action planning", ParagraphStyle("mk_cover_foot", parent=styles["small"], alignment=TA_LEFT))]
    return story


def build_page_snapshot(client_facts, allocation_output):
    styles = _report_styles()
    analysis = client_facts.get("analysis") or {}
    income = client_facts.get("income") or {}
    insurance = client_facts.get("insurance") or {}
    bank = client_facts.get("bank") or {}
    portfolio = client_facts.get("portfolio") or {}
    diag = analysis.get("_diagnostics") or {}
    ar = analysis.get("advancedRisk") or {}
    expenses = _num(income.get("monthlyExpenses"), 0)
    monthly_surplus = _display_monthly_surplus(client_facts)
    ef_target = expenses * 6
    s_body = styles["table"]
    green = colors.HexColor("#1FA55B")
    red = colors.HexColor("#CB3E2D")
    orange = colors.HexColor("#E68A1F")
    blue = colors.HexColor("#2A557E")
    def ctext(txt, col, right_align=False):
        style = ParagraphStyle("ctext", parent=s_body, alignment=TA_RIGHT if right_align else TA_LEFT)
        return Paragraph(f"<font color='{col.hexval()}'><b>{sanitize_pdf_text(str(txt))}</b></font>", style)
    
    def kpi_fmt(s):
        return str(s).replace("Rs. ", "Rs.").replace(" L", "L").replace(" Cr", "Cr")

    rows = [["AREA", "CURRENT", "IDEAL", "PRIORITY"]]
    rows += [
        ["Life Cover", ctext(kpi_fmt(_fmt_rs(insurance.get("lifeCover"))), red), ctext(kpi_fmt(_fmt_rs(diag.get("requiredLifeCover"))), green), _tag(_urgency(((analysis.get("ihs") or {}).get("breakdown") or {}).get("protection", {}).get("score")))],
        ["Health Cover", ctext(kpi_fmt(_fmt_rs(insurance.get("healthCover"))), orange), ctext("Rs.10-15L", green), _tag(analysis.get("insuranceGap") or "-")],
        ["Emergency Fund", ctext(f"{_num(diag.get('liquidityMonths'), 0):.1f} months", blue), ctext(f"6 months expenses ({kpi_fmt(_fmt_rs(ef_target))})", green), _tag(analysis.get("liquidity") or "-")],
        ["Equity Allocation", ctext(f"{_portfolio_equity(portfolio):.0f}%", red), ctext(_recommended_band_text(ar), green), _tag("HIGH")],
        ["EMI / Income", ctext(f"{_num(diag.get('emiPct'), 0):.1f}%", blue), ctext("<40%", green), _tag(analysis.get("debtStress") or "-")],
        ["Active SIP", ctext("Rs.0/mo", red), ctext(f"Need {kpi_fmt(_fmt_rs(allocation_output.get('combined_shortfall'), False))}/mo", green), _tag("HIGH")],
        ["Goals Funded", ctext(f"0 of {len(client_facts.get('goals') or [])}", red), ctext(f"{len(client_facts.get('goals') or [])} of {len(client_facts.get('goals') or [])}", green), _tag("HIGH")],
    ]
    profile = [
        ["Risk Profile", ctext(analysis.get("riskProfile"), colors.HexColor("#2C3E50"), True)],
        ["Surplus Level", ctext(analysis.get("surplusBand") or "-", green if (analysis.get("surplusBand") or "").lower() == "comfortable" or "adequate" in (analysis.get("surplusBand") or "").lower() else red, True)],
        ["Insurance Status", ctext(analysis.get("insuranceGap") or "-", red if "under" in (analysis.get("insuranceGap") or "").lower() else green, True)],
        ["Debt Position", ctext(analysis.get("debtStress") or "-", orange if "mod" in (analysis.get("debtStress") or "").lower() else green, True)],
        ["Liquidity", ctext(analysis.get("liquidity") or "-", red if "low" in (analysis.get("liquidity") or "").lower() or "insuf" in (analysis.get("liquidity") or "").lower() else green, True)],
        ["IHS Band", ctext((analysis.get("ihs") or {}).get("band") or "-", orange, True)],
    ]
    adv = [
        ["Calculated Risk Score", ctext(f"{ar.get('riskScore') or ar.get('score') or '-'} / 5.0", colors.HexColor("#2C3E50"), True)],
        ["Risk Appetite", ctext(ar.get("riskAppetite") or ar.get("appetiteCategory") or "-", colors.HexColor("#2C3E50"), True)],
        ["Tenure Limit", ctext(ar.get("tenureLimit") or ar.get("tenureLimitCategory") or "-", colors.HexColor("#2C3E50"), True)],
        ["Final Category", ctext(ar.get("finalCategory") or "-", orange, True)],
        ["Recommended Equity Band", ctext(_recommended_band_text(ar), green, True)],
    ]
    snapshot_kpis = [
        {"label": "Annual Income (Gross)", "value": kpi_fmt(_fmt_rs(income.get("annualIncome"))), "note": "Rs.2.0L gross / month"},
        {"label": "Monthly Surplus", "value": kpi_fmt(_fmt_rs(monthly_surplus)), "note": "Savings rate: 45%"},
        {"label": "Current Portfolio", "value": kpi_fmt(_fmt_rs(portfolio.get("total_value") or portfolio.get("current_value"))), "note": f"72% equity - Gains Rs.2.1L"},
        {"label": "Active SIP", "value": "Rs.0", "note": f"Need {kpi_fmt(_fmt_rs(allocation_output.get('combined_shortfall'), False))}/mo", "note_color": "orange"},
    ]
    current_vs_ideal = _styled_table(rows, [74, 56, 76, 104], style_type="light")
    profile_tbl = _styled_table(profile, [56, 86], header=False, style_type="light_right")
    adv_tbl = _styled_table(adv, [80, 62], header=False, style_type="light_right")

    profile_card = Table([[Paragraph("PROFILE AT A GLANCE", ParagraphStyle("paag", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=8))], [profile_tbl]], colWidths=[166])
    profile_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#EAE2D6")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))

    adv_card = Table([[Paragraph("ADVANCED RISK ASSESSMENT", ParagraphStyle("ara", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=8))], [adv_tbl]], colWidths=[166])
    adv_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#EAE2D6")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))

    right_stack = Table([[profile_card], [Spacer(1, 12)], [adv_card]], colWidths=[190])
    right_stack.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    
    left_side = Table([
        [Paragraph("CURRENT VS IDEAL", ParagraphStyle("cvsi", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=8))],
        [current_vs_ideal]
    ], colWidths=[310])
    left_side.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#EAE2D6")),
        ("LEFTPADDING", (0, 0), (-1, 0), 12),
        ("RIGHTPADDING", (0, 0), (-1, 0), 12),
        ("TOPPADDING", (0, 0), (-1, 0), 12),
        ("LEFTPADDING", (0, 1), (-1, -1), 0),
        ("RIGHTPADDING", (0, 1), (-1, -1), 0),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 12),
    ]))

    
    outer_table = Table([[left_side, Spacer(10, 1), right_stack]], colWidths=[310, 10, 190])
    outer_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
    ]))
    
    return [
        Paragraph("FINANCIAL SNAPSHOT", ParagraphStyle("mk_snap_label", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=4)),
        Paragraph("Where You Stand Today", ParagraphStyle("mk_snap_title", parent=styles["h2"], fontName="Times-Roman", fontSize=26, leading=30)),
        Spacer(1, 16),
        DrawingFlowable(
            lambda c, x, y, tiles: KPIFlowable(tiles, width=510, height=85, card_height=75, gap=10, card_y=5).drawOn(c, x, y),
            510, 85, snapshot_kpis
        ),
        Spacer(1, 16),
        outer_table,
    ]


def build_page_executive_summary(client_facts, allocation_output, narratives=None):
    styles = _report_styles()
    breakdown = (((client_facts.get("analysis") or {}).get("ihs") or {}).get("breakdown") or {})
    why = {
        "portfolio_health": "Equity allocation far outside your risk band",
        "goal_readiness": "SIP shortfall across all financial goals",
        "protection": "Life cover gap is a critical family risk",
        "liquidity": "Emergency fund unknown or insufficient",
        "tax_efficiency": "Potential regime saving not yet captured",
        "debt_management": "EMI/income ratio vs 40% benchmark",
    }
    canonical = [
        ("portfolio_health", "Portfolio Health"),
        ("goal_readiness", "Goal Readiness"),
        ("protection", "Protection"),
        ("liquidity", "Liquidity"),
        ("tax_efficiency", "Tax Efficiency"),
        ("debt_management", "Debt Management"),
    ]
    scored = []
    for key, label in canonical:
        item = breakdown.get(key) or {}
        scored.append((key, label, _num(item.get("score"), 0)))
    dimension_rows = []
    for _, label, score in sorted(scored, key=lambda item: item[2]):
        dimension_rows.append((label, _num(score, 0), _urgency(score)))
    priority_rows = [["AREA", "SCORE", "ACTION REQUIRED", "WHY IT MATTERS"]]
    for key, label, score in sorted(scored, key=lambda item: item[2]):
        score_val = int(_num(score, 0))
        color_hex = "#C0392B" if score_val < 40 else "#E67E22" if score_val < 75 else "#27AE60"
        priority_rows.append([Paragraph(f"<b>{label}</b>", styles["table"]), Paragraph(f"<b><font color='{color_hex}'>{score_val}/100</font></b>", styles["table"]), _tag(_urgency(score), bg_tint=True), why.get(key, "-")])
    overall = int(_num((((client_facts.get("analysis") or {}).get("ihs") or {}).get("score")), 0))
    overall_label = (("Needs Attention" if overall < 75 else "Healthy") if overall > 0 else "-")
    overall_color = "#E67E22" if overall < 75 else "#27AE60"
    priority_rows.append([Paragraph("<b>Overall Score</b>", styles["table"]), Paragraph(f"<b><font color='{overall_color}'>{overall}/100</font></b>", styles["table"]), _tag(overall_label.title() if overall_label != "-" else "-", bg_tint=True), "3 areas require immediate action before deploying capital into goals"])

    blocks = [
        Paragraph("SIX CRITICAL DIMENSIONS", ParagraphStyle("six_dim", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=4)),
        Paragraph("Your Financial Health Score", ParagraphStyle("mk_exec_title", parent=styles["h2"], fontName="Times-Roman", fontSize=32, leading=38)),
        Spacer(1, 16),
        DrawingFlowable(lambda c, x, y, rows=dimension_rows: _draw_dimension_score_bars(c, 0, 140, 500, rows), 500, 160),
        Spacer(1, 12),
        Paragraph("Color key: <font color='#C0392B'>■</font> Critical &lt;40 &nbsp;&nbsp;&middot;&nbsp;&nbsp; <font color='#E67E22'>■</font> Needs Attention 40-74 &nbsp;&nbsp;&middot;&nbsp;&nbsp; <font color='#27AE60'>■</font> Good &ge;75", ParagraphStyle("ckey", parent=styles["small"], textColor=colors.HexColor("#888888"))),
        Spacer(1, 24),
        Paragraph("PRIORITY RANKING", ParagraphStyle("pr", parent=styles["label"], textColor=colors.HexColor("#A8813C"))),
        Spacer(1, 8),
        _styled_table(priority_rows, [130, 65, 105, 200], style_type="light"),
        Spacer(1, 16),
    ]
    logic_box = Table(
        [[
            Paragraph(
                "<b>Urgency Tag Logic</b><br/>"
                "Score &lt;40 → <b>IMMEDIATE</b>  ·  "
                "Score 40-74 → <b>HIGH</b>  ·  "
                "Score ≥75 → <b>GOOD</b><br/>"
                "Protection can be overridden to IMMEDIATE when family risk is critical.",
                styles["body"],
            )
        ]],
        colWidths=[500],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F0E8")),
            ("LEFTPADDING", (0, 0), (-1, -1), 16),
            ("RIGHTPADDING", (0, 0), (-1, -1), 16),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ]),
    )
    blocks.append(logic_box)
    return blocks


def build_page_protection(client_facts, allocation_output):
    styles = _report_styles()
    analysis = client_facts.get("analysis") or {}
    insurance = client_facts.get("insurance") or {}
    diag = analysis.get("_diagnostics") or {}
    current_life = _num(insurance.get("lifeCover"), 0)
    required_life = _num(diag.get("requiredLifeCover"), 0)
    gap = max(0, required_life - current_life)
    coverage_pct = (current_life / required_life * 100) if required_life > 0 else 100

    life_status = _urgency((((analysis.get("ihs") or {}).get("breakdown") or {}).get("protection") or {}).get("score"))
    health_tag = "UPGRADE RECOMMENDED"
    est_premium = _num(allocation_output.get("insurance_provision"), 0)
    life_gap_pct = (gap / required_life * 100) if required_life > 0 else 0
    life_header = Table(
        [[Paragraph("LIFE INSURANCE", ParagraphStyle("li", parent=styles["label"], textColor=colors.HexColor("#C0392B"))),
          TagFlowable(life_status, bg_tint=True)]],
        colWidths=[130, 90],
        style=TableStyle([
            ("ALIGN", (1, 0), (1, 0), "RIGHT"), 
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#E5CBC1")),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8)
        ]),
    )
    life_card_content = [
        life_header,
        Spacer(1, 12),
        Table(
            [[Paragraph(f"<font color='#555555'>Current:</font> <b>{_fmt_rs(current_life)}</b>", styles["body"]), Paragraph(f"<font color='#555555'>Required:</font> <b><font color='#27AE60'>{_fmt_rs(required_life)}</font></b>", styles["body"])]],
            colWidths=[110, 110],
            style=TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)])
        ),
        Spacer(1, 8),
        CanvasBlock(220, 14, lambda c, x, y, w, h: draw_coverage_bar(c, 0, 0, 220, coverage_pct, _fmt_rs(current_life), bar_color=(0.75, 0.22, 0.17))),
        Spacer(1, 8),
        Paragraph(f"Coverage gap: <b><font color='#C0392B'>{life_gap_pct:.0f}% underinsured</font></b> · Basis: 20x annual expenses", styles["small"]),
        Spacer(1, 16),
        Table(
            [[ [Paragraph("PROTECTION GAP", ParagraphStyle("gap_lbl", parent=styles["label"], textColor=colors.HexColor("#C0392B"), spaceAfter=6)),
                Paragraph(f"<b>{_fmt_rs(gap)}</b>", ParagraphStyle("gap_val", parent=styles["h2"], fontSize=26, textColor=colors.HexColor("#C0392B"), leading=30, spaceAfter=4)),
                Paragraph(f"Est. premium ~{_fmt_rs(est_premium, False)}/month · Tenure until age 55", ParagraphStyle("gap_sub", parent=styles["small"], textColor=colors.HexColor("#4A5568")))] ]],
            colWidths=[200],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F6EAE6")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E5CBC1")),
                ("ROUNDEDCORNERS", [6, 6, 6, 6]),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 14),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ])
        ),
        Spacer(1, 16),
        Paragraph("WHAT TO LOOK FOR", ParagraphStyle("wtlf", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=8)),
        _styled_table([
            [Paragraph("<font color='#27AE60'>✓</font> &nbsp; Pure term insurance — no investment component", styles["body"])],
            [Paragraph("<font color='#27AE60'>✓</font> &nbsp; Coverage tenure until age 55-60 minimum", styles["body"])],
            [Paragraph("<font color='#27AE60'>✓</font> &nbsp; Sum assured benchmark: ~20x annual expenses", styles["body"])],
            [Paragraph("<font color='#27AE60'>✓</font> &nbsp; Compare: HDFC Life, ICICI Prudential, Max Life", styles["body"])],
        ], [220], header=False, style_type="light"),
    ]
    health_header = Table(
        [[Paragraph("HEALTH INSURANCE", ParagraphStyle("hi", parent=styles["label"], textColor=colors.HexColor("#27AE60"))),
          TagFlowable(health_tag, bg_tint=True)]],
        colWidths=[110, 110],
        style=TableStyle([
            ("ALIGN", (1, 0), (1, 0), "RIGHT"), 
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#CBE2D4")),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8)
        ]),
    )
    
    health_card_content = [
        health_header,
        Spacer(1, 12),
        Table(
            [[Paragraph(f"<font color='#555555'>Current:</font> <b>{_fmt_rs(insurance.get('healthCover'))}</b>", styles["body"]), Paragraph("Recommended: <b><font color='#27AE60'>Rs. 10-15 L</font></b>", styles["body"])]],
            colWidths=[110, 110],
            style=TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)])
        ),
        Spacer(1, 8),
        CanvasBlock(220, 14, lambda c, x, y, w, h: draw_coverage_bar(c, 0, 0, 220, min(200, (_num(insurance.get("healthCover"), 0) / 1000000 * 100) if _num(insurance.get("healthCover"), 0) > 0 else 0), _fmt_rs(insurance.get("healthCover")), bar_color=(0.9, 0.49, 0.13))),
        Spacer(1, 8),
        Paragraph("Adequate base — <b><font color='#E67E22'>upgrade recommended</font></b>", styles["small"]),
        Spacer(1, 16),
        Table(
            [[ [Paragraph("STATUS", ParagraphStyle("stat_lbl", parent=styles["label"], textColor=colors.HexColor("#27AE60"), spaceAfter=6)), 
                Paragraph("Adequate Base Cover", ParagraphStyle("stat_val", parent=styles["h2"], fontSize=20, textColor=colors.HexColor("#27AE60"), leading=24, spaceAfter=4)),
                Paragraph("Upgrade to Rs. 10-15L family floater recommended", ParagraphStyle("stat_sub", parent=styles["body"], textColor=colors.HexColor("#4A5568")))] ]],
            colWidths=[200],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EDF7F0")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#C6E4CF")),
                ("ROUNDEDCORNERS", [6, 6, 6, 6]),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 14),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ]),
        ),
        Spacer(1, 16),
        Paragraph("WHAT TO LOOK FOR", ParagraphStyle("wtlf", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=8)),
        _styled_table([
            [Paragraph("<font color='#27AE60'>✓</font> &nbsp; Family floater plan covering all dependents", styles["body"])],
            [Paragraph("<font color='#27AE60'>✓</font> &nbsp; Rs. 10-15 lakh sum insured minimum", styles["body"])],
            [Paragraph("<font color='#27AE60'>✓</font> &nbsp; Cashless facility at major hospitals", styles["body"])],
            [Paragraph("<font color='#27AE60'>✓</font> &nbsp; No room rent capping", styles["body"])],
        ], [220], header=False, style_type="light"),
    ]

    left_cell = Table([[life_card_content]], colWidths=[240])
    right_cell = Table([[health_card_content]], colWidths=[240])
    
    left_cell.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FDF5F2")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E5CBC1")),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
    ]))
    right_cell.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F2F9F4")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBE2D4")),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
    ]))

    cards = Table([[left_cell, Spacer(10, 1), right_cell]], colWidths=[240, 10, 240])
    cards.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
    ]))
    
    return [
        Paragraph("PROTECTION GAP ANALYSIS", ParagraphStyle("pga", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=4)),
        Paragraph("Insurance — Actual vs Recommended", ParagraphStyle("iavr", parent=styles["h2"], fontName="Times-Roman", fontSize=26, leading=30)),
        Spacer(1, 12),
        cards
    ]


def build_page_portfolio_debt(client_facts, allocation_output):
    styles = _report_styles()
    analysis = client_facts.get("analysis") or {}
    portfolio = client_facts.get("portfolio") or {}
    income = client_facts.get("income") or {}
    diag = analysis.get("_diagnostics") or {}
    ar = analysis.get("advancedRisk") or {}
    debt_score = (((analysis.get("ihs") or {}).get("breakdown") or {}).get("debt_management") or {}).get("score")
    emi_pct = _num(diag.get("emiPct"), 0)
    
    current_eq = _portfolio_equity(portfolio)
    target_eq = _recommended_band_mid(ar)
    b_min = _num((ar.get("recommendedEquityBand") or {}).get("min"), 40)
    b_max = _num((ar.get("recommendedEquityBand") or {}).get("max"), 60)
    
    current_color = "#C0392B" if (current_eq < b_min or current_eq > b_max) else "#27AE60"
    target_color = "#27AE60"
    
    def _arc(c, x, y, w, h, pct, col):
        c.setStrokeColorRGB(0.55, 0.65, 0.75)
        c.setLineWidth(10)
        c.setLineCap(1)
        c.arc(x+10, y+10, x+w-10, y+h-10, startAng=-45, extent=270)
        ch = colors.HexColor(col)
        c.setStrokeColorRGB(ch.red, ch.green, ch.blue)
        c.setLineWidth(10)
        c.setLineCap(1)
        c.arc(x+10, y+10, x+w-10, y+h-10, startAng=-45, extent=(_num(pct, 0) / 100.0) * 270)
        c.setFillColorRGB(ch.red, ch.green, ch.blue)
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(x + w/2, y + h/2 - 6, f"{int(_num(pct, 0))}%")
    
    left_block = [
        Paragraph("CURRENT ALLOCATION", ParagraphStyle("ca", parent=styles["label"], alignment=TA_CENTER, textColor=colors.HexColor("#7F8C8D"))),
        Spacer(1, 4),
        CanvasBlock(100, 100, lambda c,x,y,w,h: _arc(c, x, y, w, h, current_eq, current_color)),
        Spacer(1, 4),
        Paragraph(f"<font color='{current_color}'>● <b>{int(current_eq)}% Equity</b></font> &nbsp;&nbsp; <font color='#8CA2B5'>● {100-int(current_eq)}% Debt</font>", ParagraphStyle("sub1", parent=styles["body"], alignment=TA_CENTER)),
        Spacer(1, 2),
        Paragraph(f"Portfolio {_fmt_rs(portfolio.get('total'), False)} · Gains {_fmt_rs(portfolio.get('unrealised_gains'), False)}", ParagraphStyle("sub2", parent=styles["small"], alignment=TA_CENTER, textColor=colors.HexColor("#7F8C8D"))),
    ]

    right_block = [
        Paragraph("TARGET ALLOCATION", ParagraphStyle("ta", parent=styles["label"], alignment=TA_CENTER, textColor=colors.HexColor("#7F8C8D"))),
        Spacer(1, 4),
        CanvasBlock(100, 100, lambda c,x,y,w,h: _arc(c, x, y, w, h, target_eq, target_color)),
        Spacer(1, 4),
        Paragraph(f"<font color='{target_color}'>● <b>{int(target_eq)}% Equity</b></font> &nbsp;&nbsp; <font color='#8CA2B5'>● {100-int(target_eq)}% Debt</font>", ParagraphStyle("sub3", parent=styles["body"], alignment=TA_CENTER)),
        Spacer(1, 2),
        Paragraph(f"Conservative · Band {int(b_min)}–{int(b_max)}% · Risk-aligned", ParagraphStyle("sub4", parent=styles["small"], alignment=TA_CENTER, textColor=colors.HexColor("#7F8C8D"))),
    ]

    arrow_block = [
        Spacer(1, 30),
        CanvasBlock(40, 30, lambda c,x,y,w,h: [c.setStrokeColorRGB(0.8, 0.7, 0.5), c.setLineWidth(1), c.line(x, y+15, x+w, y+15), c.setFillColorRGB(0.8, 0.7, 0.5), c.setFont("Helvetica-Bold", 14), c.drawCentredString(x+w/2, y, "→")]),
        Spacer(1, 2),
        Paragraph("REBALANCE", ParagraphStyle("reb", parent=styles["label"], alignment=TA_CENTER, textColor=colors.HexColor("#A8813C"))),
    ]

    gauge_card = Table(
        [[left_block, arrow_block, right_block]],
        colWidths=[200, 80, 200],
        style=TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E2E8F0")),
            ("ROUNDEDCORNERS", [8, 8, 8, 8]),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ])
    )

    bullets = [
        "Which funds/instruments to use for rebalancing", 
        "When to execute (consider LTCG tax implications)", 
        "How to split equity vs debt per goal horizon", 
        "Which goals to prioritise if all aren't affordable"
    ]
    bullet_rows = [
        [
            Paragraph(f"<font color='#27AE60'>✓</font> &nbsp;{bullets[0]}", styles["body"]),
            Paragraph(f"<font color='#27AE60'>✓</font> &nbsp;{bullets[1]}", styles["body"])
        ],
        [
            Paragraph(f"<font color='#27AE60'>✓</font> &nbsp;{bullets[2]}", styles["body"]),
            Paragraph(f"<font color='#27AE60'>✓</font> &nbsp;{bullets[3]}", styles["body"])
        ]
    ]
    bullet_table = Table(bullet_rows, colWidths=[230, 230], style=TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#E5D9B1")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))

    control_card = Table(
        [[ [Paragraph("WHAT YOU CONTROL", ParagraphStyle("wyc", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=10)),
            bullet_table,
            Spacer(1, 12),
            CanvasBlock(460, 1, lambda c,x,y,w,h: [c.setStrokeColorRGB(0.9, 0.85, 0.7), c.setLineWidth(0.5), c.line(x,y,x+w,y)]),
            Spacer(1, 8),
            Paragraph("Powered by RISE · 25yr Nifty History · 6,250 Data Points · ~12.5% Long-term CAGR", ParagraphStyle("pow", parent=styles["small"], textColor=colors.HexColor("#95A5A6")))] ]],
        colWidths=[480],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FDFBF7")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E5D9B1")),
            ("ROUNDEDCORNERS", [8, 8, 8, 8]),
            ("LEFTPADDING", (0, 0), (-1, -1), 16),
            ("RIGHTPADDING", (0, 0), (-1, -1), 16),
            ("TOPPADDING", (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ])
    )


    debt_rows = [
        ["METRIC", "YOUR VALUE", "BENCHMARK", "STATUS", "NOTE"],
        ["EMI / Income Ratio", Paragraph(f"<font color='#27AE60'><b>~{emi_pct:.0f}%</b></font>", styles["table"]) if emi_pct < 40 else Paragraph(f"<font color='#C0392B'><b>~{emi_pct:.0f}%</b></font>", styles["table"]), "<40%", TagFlowable("GOOD", bg_tint=True) if emi_pct < 40 else TagFlowable("HIGH", bg_tint=True), "Healthy discipline" if emi_pct < 40 else "Reduce EMI"],
        ["Total Outstanding Loans", "Moderate", "—", TagFlowable("REVIEW", bg_tint=True), "Confirm with client"],
        ["Loan Types", "Home / Car / Personal", "—", "—", "Identify highest rate"],
        ["Debt Score", Paragraph(f"<font color='#27AE60'><b>{int(_num(debt_score, 0))}/100</b></font>", styles["table"]) if _num(debt_score, 0) >= 75 else Paragraph(f"<font color='#C0392B'><b>{int(_num(debt_score, 0))}/100</b></font>", styles["table"]), "≥75 = Good", TagFlowable("GOOD", bg_tint=True) if _num(debt_score, 0) >= 75 else TagFlowable("NEEDS ATTENTION", bg_tint=True), "Well managed" if _num(debt_score, 0) >= 75 else "Needs attention"],
    ]
    debt_table = _styled_table(debt_rows, [110, 100, 100, 80, 110], style_type="light")

    return [
        Paragraph("PORTFOLIO REBALANCING", ParagraphStyle("pga", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=2)),
        Paragraph("Equity Allocation — Actual vs Target", ParagraphStyle("iavr", parent=styles["h2"], fontName="Times-Roman", fontSize=26, leading=30)),
        Spacer(1, 4),
        gauge_card,
        Spacer(1, 6),
        control_card,
        Spacer(1, 10),
        Paragraph("DEBT MANAGEMENT", ParagraphStyle("pga", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=2)),
        Spacer(1, 2),
        debt_table
    ]



def build_page_liquidity(client_facts, allocation_output):
    styles = _report_styles()
    analysis = client_facts.get("analysis") or {}
    income = client_facts.get("income") or {}
    diag = analysis.get("_diagnostics") or {}
    expenses = _num(income.get("monthlyExpenses"), 0)
    target = expenses * 6
    score = (((analysis.get("ihs") or {}).get("breakdown") or {}).get("liquidity") or {}).get("score")
    
    # Left Card: Parameters
    params = [
        ["Liquidity Score", Paragraph(f"<font color='#E67E22'><b>{_num(score, 0):.0f}/100</b></font>", styles["body"])],
        ["Basis", Paragraph("<b>6 months</b> essential expenses", styles["body"])],
        ["Monthly Expenses (est.)", Paragraph(f"<b>{_fmt_rs(expenses, False)}/month</b>", styles["body"])],
        ["Emergency Fund Target", Paragraph(f"<font color='#27AE60'><b>~{_fmt_rs(target, False)}</b></font>", styles["body"])],
        ["Current Status", Paragraph("<font color='#E67E22'><b>Unknown — assess now</b></font>", styles["body"])],
    ]
    params_table = _styled_table(params, [110, 80], header=False, style_type="compact_light")

    
    advisory_note = Table(
        [[Paragraph("<b><font color='#2C3E50'>Advisory Note</font></b><br/><font color='#4A5568'>Build emergency fund BEFORE investing in long-term goals. This is non-negotiable financial hygiene.</font>", styles["small"])]],
        colWidths=[173],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1E9DA")),
            ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#A8813C")),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ])
    )
    
    left = Table(
        [[ [Paragraph("EMERGENCY FUND PARAMETERS", ParagraphStyle("efp", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=12)),
            params_table,
            Spacer(1, 24),
            advisory_note] ]],
        colWidths=[205],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FDFBF7")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E5D9B1")),
            ("ROUNDEDCORNERS", [8, 8, 8, 8]),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 16),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ])
    )
    
    # Right Card: Strategy
    strategy = [
        ["OPTION", "DURATION", "MONTHLY", "BEST FOR"],
        [Paragraph("<font color='#27AE60'><b>A — Fast</b></font>", styles["body"]), "12 months", Paragraph(f"<font color='#E67E22'><b>{_fmt_rs(target / 12, False)}</b></font>", styles["body"]), Paragraph("<font color='#4A5568'>High surplus period</font>", styles["small"])],
        [Paragraph("<font color='#A8813C'><b>B — Balanced<br/>★</b></font>", styles["body"]), "18 months", Paragraph(f"<font color='#E67E22'><b>{_fmt_rs(target / 18, False)}</b></font>", styles["body"]), Paragraph("<font color='#4A5568'>Recommended</font>", styles["small"])],
        [Paragraph("<font color='#7F8C8D'><b>C — Gentle</b></font>", styles["body"]), "24 months", Paragraph(f"<font color='#E67E22'><b>{_fmt_rs(target / 24, False)}</b></font>", styles["body"]), Paragraph("<font color='#4A5568'>If goals compete</font>", styles["small"])],
    ]
    strategy_table = _styled_table(strategy, [60, 58, 60, 63], style_type="compact_light")

    strategy_table.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#FDFBF7")),
        ("TOPPADDING", (0, 1), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 10),
    ]))
    
    right = Table(
        [[ [Paragraph("BUILDING STRATEGY — FROM ZERO", ParagraphStyle("bs", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=12)),
            strategy_table] ]],
        colWidths=[265],
        style=TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E2E8F0")),
            ("ROUNDEDCORNERS", [8, 8, 8, 8]),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 16),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ])
    )
    
    pair = Table([[left, Spacer(10, 1), right]], colWidths=[205, 10, 265])
    pair.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0)
    ]))
    
    # Tier Cards
    t1_amt = _fmt_rs(target * 0.4, False)
    t2_amt = _fmt_rs(target * 0.4, False)
    t3_amt = _fmt_rs(target * 0.2, False)
    
    def _tier_card(code, code_color, title, subtitle, desc):
        tbl = Table(
            [[Paragraph(f"<font color='{code_color}'><b>{code}</b></font>", ParagraphStyle("tc", fontName="Helvetica-Bold", fontSize=14, leading=14, alignment=TA_CENTER)),
              Paragraph(f"<b><font color='#2C3E50'>{title}</font></b><br/><font color='#A8813C'>{subtitle}</font><br/><br/><font color='#4A5568'>{desc}</font>", styles["small"])]],
            colWidths=[30, 115]
        )
        tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EFE5D3")),
            ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#FDFBF7")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E5D9B1")),
            ("ROUNDEDCORNERS", [8, 8, 8, 8]),
            ("LEFTPADDING", (0, 0), (0, -1), 0),
            ("RIGHTPADDING", (0, 0), (0, -1), 0),
            ("LEFTPADDING", (1, 0), (1, -1), 10),
            ("RIGHTPADDING", (1, 0), (1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ]))
        return tbl
        
    t1 = _tier_card("T1", "#C0392B", "Instant Access", f"40% · ~{t1_amt}", "Savings bank account — zero friction. High-yield savings (AU, IDFC).")
    t2 = _tier_card("T2", "#E67E22", "Quick Access", f"40% · ~{t2_amt}", "Liquid mutual funds or sweep-in FD. 1-2 day redemption. 6-7% return.")
    t3 = _tier_card("T3", "#A8813C", "Short-Term FD", f"20% · ~{t3_amt}", "3-6 month fixed deposits. Slightly higher return. Last-resort buffer.")
    
    tier_cards = Table([[t1, Spacer(8, 1), t2, Spacer(8, 1), t3]], colWidths=[145, 8, 145, 8, 145])
    tier_cards.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0)
    ]))
    
    return [
        Paragraph("LIQUIDITY ANALYSIS", ParagraphStyle("pga", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=2)),
        Paragraph("Emergency Fund Assessment", ParagraphStyle("iavr", parent=styles["h2"], fontName="Times-Roman", fontSize=26, leading=30)),
        Spacer(1, 16),
        pair,
        Spacer(1, 24),
        Paragraph("WHERE TO PARK THE EMERGENCY FUND", ParagraphStyle("wtp", parent=styles["label"], textColor=colors.HexColor("#A8813C"))),
        Spacer(1, 6),
        tier_cards
    ]


def build_page_goal_feasibility(client_facts, allocation_output):
    styles = _report_styles()
    
    total_goals = max(1, len(allocation_output.get("goal_sip_table") or []))
    total_shortfall = _num(allocation_output.get("combined_shortfall"), 0)
    
    cards = []
    for g in allocation_output.get("goal_sip_table") or []:
        name = g.get("name") or "Goal"
        target_val = _num(g.get("target_amount"), 0)
        horizon = f"{g.get('horizon_years') or '-'} yrs"
        ideal = _num(g.get("ideal_sip"), 0)
        curr = 0 # Currently assumed Rs. 0
        gap = _num(g.get("shortfall"), 0)
        coverage = (_num(g.get("allocated_sip"), 0) / ideal * 100) if ideal > 0 else 0
        
        # formatting
        def _fmt(val):
            return _fmt_rs(val, compact=True)
            
        target_str = _fmt(target_val)
        curr_str = _fmt(curr)
        ideal_str = _fmt(ideal)
        gap_str = _fmt(gap)
        
        goal_text = f"<font color='#666666' size='7'>GOAL</font><br/><font face='Times-Roman' size='11' color='#2C3E50'>{name}</font>"

        target_text = f"<font color='#666666' size='7'>TARGET</font><br/><font color='#2C3E50'>{target_str}</font>"
        horizon_text = f"<font color='#666666' size='7'>HORIZON</font><br/><font color='#2C3E50'>{horizon}</font>"
        curr_sip_text = f"<font color='#666666' size='7'>CURR. SIP</font><br/><font color='#C0392B'>{curr_str}</font>"
        req_sip_text = f"<font color='#666666' size='7'>REQ. SIP</font><br/><font color='#D35400'>{ideal_str}</font>"
        gap_text = f"<font color='#666666' size='7'>GAP/MO</font><br/><font color='#C0392B'>{gap_str}</font>"
        
        p_style = ParagraphStyle("card_p", fontName="Helvetica-Bold", fontSize=9, leading=12)
        
        inner_data = [[
            Paragraph(target_text, p_style),
            Paragraph(horizon_text, p_style),
            Paragraph(curr_sip_text, p_style),
            Paragraph(req_sip_text, p_style),
            Paragraph(gap_text, p_style),
        ]]
        inner_table = Table(inner_data, colWidths=[55, 45, 45, 55, 50])
        inner_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
        ]))
        
        # Coverage section
        cov_val = f"{coverage:.0f}%"
        cov_color = '#E67E22' if coverage < 80 else '#27AE60'
        if coverage < 50: cov_color = '#C0392B'
        
        cov_tbl = Table(
            [[CoverageFlowable(min(coverage, 100), width=85, height=8)],
             [Spacer(1, 4)],
             [Table([[Paragraph("<font color='#666666' size='7'>Coverage</font>"), Paragraph(f"<b><font color='{cov_color}' size='8'>{cov_val}</font></b>", ParagraphStyle("r", alignment=TA_RIGHT))]], colWidths=[45, 40], style=TableStyle([("PADDING", (0, 0), (-1, -1), 0)]))]],
            colWidths=[85]
        )
        cov_tbl.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        
        card = Table(
            [[
                Paragraph(goal_text, p_style),
                inner_table,
                cov_tbl
            ]],
            colWidths=[100, 250, 100]
        )
        card.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E2E8F0")),
            ("ROUNDEDCORNERS", [8, 8, 8, 8]),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("LEFTPADDING", (0, 0), (0, -1), 16),
            ("RIGHTPADDING", (-1, 0), (-1, -1), 16),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ]))

        
        cards.append(card)
        cards.append(Spacer(1, 10))

    goal_colors = [
        colors.HexColor("#C0392B"), # Red
        colors.HexColor("#E67E22"), # Orange
        colors.HexColor("#A8813C"), # Gold
        colors.HexColor("#2C3E50"), # Navy
        colors.HexColor("#27AE60"), # Green
        colors.HexColor("#7F8C8D"), # Grey
    ]
    
    goals_data = []
    for i, g in enumerate(allocation_output.get("goal_sip_table") or []):
        shortfall = _num(g.get("shortfall"), 0)
        if shortfall > 0:
            goals_data.append({
                "name": g.get("name") or f"Goal {i+1}",
                "shortfall": shortfall,
                "color": goal_colors[i % len(goal_colors)]
            })

    def draw_stacked_bar(c, x, y, w, h):
        if not total_shortfall or total_shortfall <= 0: return
        c.saveState()
        path = c.beginPath()
        path.roundRect(x, y, w, h, h/2.0)
        c.clipPath(path, stroke=0)
        
        current_x = x
        for item in goals_data:
            seg_w = (item["shortfall"] / total_shortfall) * w
            if seg_w > 0:
                c.setFillColor(item["color"])
                c.rect(current_x, y, seg_w, h, fill=1, stroke=0)
                current_x += seg_w
        c.restoreState()

    stacked_bar_block = CanvasBlock(220, 14, draw_stacked_bar)
    
    legend_cells = []
    for item in goals_data:
        amt = item["shortfall"]
        if amt >= 100000:
            amt_str = f"Rs. {amt/100000:.1f}L"
        else:
            amt_str = f"Rs. {amt/1000:.1f}K"
            
        color_hex = item["color"].hexval()[2:] # rrggbb
        legend_cells.append(Paragraph(f"<font color='#{color_hex}'>■</font> <font color='#4A5568' size='7'>{item['name']} {amt_str}</font>", styles["small"]))
    
    legend_rows = []
    for i in range(0, len(legend_cells), 2):
        legend_rows.append(legend_cells[i:i+2])
    if legend_rows and len(legend_rows[-1]) == 1:
        legend_rows[-1].append(Paragraph("", styles["small"]))
        
    if legend_rows:
        legend_table = Table(legend_rows, colWidths=[110, 110])
        legend_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    else:
        legend_table = Spacer(1, 1)

    right_side = Table(
        [[stacked_bar_block],
         [Spacer(1, 6)],
         [legend_table]],
        colWidths=[214]
    )
    right_side.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    
    total_short_str = _fmt_rs(total_shortfall, False)
    left_side = Table(
        [[Paragraph("COMBINED MONTHLY SIP SHORTFALL", ParagraphStyle("cms", parent=styles["small"], textColor=colors.HexColor("#D35400"), fontSize=7, spaceAfter=2))],
         [Paragraph(f"<font color='#D35400'>{total_short_str}</font><font color='#D35400' size='12'>/month</font>", ParagraphStyle("cmsh", fontName="Times-Roman", fontSize=26, leading=26))],
         [Spacer(1, 4)],
         [Paragraph(f"<font color='#666666' size='7'>Total additional SIP needed across all {total_goals} goals to achieve full coverage</font>", styles["small"])]],
        colWidths=[224]
    )
    left_side.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    
    bottom_card = Table(
        [[left_side, Spacer(10, 1), right_side]],
        colWidths=[240, 10, 230]
    )
    bottom_card.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F9F1EB")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E5C5B5")),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (0, -1), 16),
        ("RIGHTPADDING", (-1, 0), (-1, -1), 16),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
    ]))

    return [
        Paragraph("FINANCIAL GOALS", ParagraphStyle("fg", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=2)),
        Paragraph(f"All {total_goals} Goals — Current vs Required", ParagraphStyle("mk_goal_title", parent=styles["h2"], fontName="Times-Roman", fontSize=26, leading=30)),
        Spacer(1, 16),
    ] + cards + [
        Spacer(1, 8),
        bottom_card
    ]


def build_page_cashflow_sip(client_facts, allocation_output, narratives=None):
    styles = _report_styles()
    income = client_facts.get("income") or {}
    monthly_income = _num(income.get("annualIncome"), 0) / 12
    monthly_surplus = _display_monthly_surplus(client_facts)
    total_req = _num(allocation_output.get("combined_shortfall"), 0)
    available = _num(allocation_output.get("available_for_goals"), 0)
    coverage = available / total_req * 100 if total_req else 100
    
    cashflow_kpis = [
        {"label": "Monthly Income", "value": _fmt_rs(monthly_income)},
        {"label": "Monthly Expenses", "value": _fmt_rs(income.get("monthlyExpenses"))},
        {"label": "Monthly Surplus", "value": _fmt_rs(monthly_surplus)},
        {"label": "Insurance Provision", "value": _fmt_rs(allocation_output.get("insurance_provision"), False)},
        {"label": "Available for Goals", "value": _fmt_rs(available, False)},
    ]
    
    cf_label = Paragraph("MONTHLY CASH FLOW", ParagraphStyle("mcf", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=12))
    kpi_drawing = KPIFlowable(cashflow_kpis, width=438, height=54, card_height=50, gap=6, card_y=0)
    
    summary_text = f"Total Goal Requirement: <b>{_fmt_rs(total_req, False)}/month</b>"
    coverage_text = f"Coverage: <font color='#D35400'><b>{coverage:.1f}%</b></font>"
    summary_table = Table([[Paragraph(summary_text, styles["small"]), Paragraph(coverage_text, ParagraphStyle("cov", parent=styles["small"], alignment=TA_RIGHT))]], colWidths=[219, 219])
    summary_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    
    cf_card_inner = Table([[cf_label], [kpi_drawing], [Spacer(1, 12)], [summary_table]], colWidths=[438])
    cf_card_inner.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    
    cf_card = Table([[cf_card_inner]], colWidths=[470])
    cf_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E2E8F0")),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
    ]))

    plan_rows = [[
        Paragraph("GOAL", styles["table_head_dark"]), 
        Paragraph("REQUIRED SIP", styles["table_head_dark"]), 
        Paragraph("ALLOCATED SIP", styles["table_head_dark"]), 
        Paragraph("MONTHLY GAP", styles["table_head_dark"]), 
        Paragraph("STATUS", styles["table_head_dark"])
    ]]
    for g in allocation_output.get("goal_sip_table") or []:
        shortfall = _num(g.get("shortfall"), 0)
        status_label = "Funded" if shortfall <= 0 else "Partial"
        if _num(g.get("allocated_sip"), 0) <= 0 and shortfall > 0:
            status_label = "Pending"
            
        plan_rows.append([
            Paragraph(str(g.get("name") or "Goal"), styles["table"]), 
            Paragraph(_fmt_rs(g.get("ideal_sip"), False), styles["table"]), 
            Paragraph(_fmt_rs(g.get("allocated_sip"), False), styles["table"]), 
            Paragraph(_fmt_rs(shortfall, False), styles["table"]), 
            TagFlowable(status_label, bg_tint=True)
        ])
    
    plan_rows.append([
        Paragraph("<b>Total</b>", styles["table"]), 
        Paragraph(f"<b>{_fmt_rs(total_req, False)}</b>", styles["table"]), 
        Paragraph(f"<b>{_fmt_rs(available, False)}</b>", styles["table"]), 
        Paragraph(f"<b>{_fmt_rs(max(0, total_req - available), False)}</b>", styles["table"]), 
        Paragraph("", styles["table"])
    ])
    
    plan_table = Table(plan_rows, colWidths=[160, 90, 90, 90, 70])
    plan_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#475569")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 7),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#F8FAFC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))

    insight = _llm_paragraph(narratives, "cashflow") or "Start with the available surplus, protect the household first, then scale SIPs as income grows."

    return [
        Paragraph("REALITY CHECK", ParagraphStyle("rc", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=2)),
        Paragraph("Can You Afford All This?", ParagraphStyle("mk_cf_title", parent=styles["h2"], fontName="Times-Bold", fontSize=28, leading=34)),
        Spacer(1, 16),
        cf_card,
        Spacer(1, 24),
        Paragraph("GOAL-WISE SIP ALLOCATION PLAN", ParagraphStyle("gwap", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=8)),
        plan_table,
        Spacer(1, 24),
        InsightFlowable("Key Insight", sanitize_pdf_text(insight))
    ]


def build_page_tax(client_facts, allocation_output):
    styles = _report_styles()
    itr = client_facts.get("itr") or {}
    gross = _num(itr.get("gross_total_income") or (client_facts.get("income") or {}).get("annualIncome"), 0)
    deductions = itr.get("deductions_claimed") or []
    claimed = {"80C": 0, "80D": 0, "80CCD_1B": 0}
    for d in deductions:
        sec = str(d.get("section") or "").upper()
        amt = _num(d.get("amount"), 0)
        if "80CCD" in sec or "NPS" in sec:
            claimed["80CCD_1B"] += amt
        elif "80D" in sec:
            claimed["80D"] += amt
        elif "80C" in sec:
            claimed["80C"] += amt
            
    current = compute_regime_comparison(gross, claimed["80C"], claimed["80D"], claimed["80CCD_1B"])
    optimized = compute_regime_comparison(gross, 150000, 75000, 50000)
    
    def _tax_cell(val, better):
        col = "#27AE60" if better else "#C0392B"
        return Paragraph(f"<b><font color='{col}'>{_fmt_rs(val, False)}</font></b>", styles["table"])

    # Top Tables and Card
    comparison_rows = [[
        Paragraph("SCENARIO", styles["table_head_dark"]), 
        Paragraph("OLD REGIME", styles["table_head_dark"]), 
        Paragraph("NEW REGIME", styles["table_head_dark"]), 
        Paragraph("BETTER", styles["table_head_dark"])
    ]]
    
    for label, res in [("Current ITR", current), ("Max Deductions", optimized)]:
        comparison_rows.append([
            Paragraph(label, styles["table"]),
            _tax_cell(res["old_regime"]["tax_liability"], res["better_regime"] == "old"),
            _tax_cell(res["new_regime"]["tax_liability"], res["better_regime"] == "new"),
            TagFlowable(f"✓ {res['better_regime'].title()}", bg_tint=True)
        ])
        
    comp_table = Table(comparison_rows, colWidths=[90, 80, 80, 70])
    comp_table.setStyle(TableStyle([
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F8FAFC")),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#E2E8F0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    
    savings = max(_num(current.get('savings'), 0), _num(optimized.get('savings'), 0))
    saving_card = SavingCardFlowable(savings, width=225, height=120)
    
    top_split = Table([[comp_table, Spacer(1, 1), saving_card]], colWidths=[320, 20, 225])
    top_split.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    
    # Bottom Actions Table
    action_rows = [[
        Paragraph("#", styles["table_head_dark"]), 
        Paragraph("ACTION", styles["table_head_dark"]), 
        Paragraph("WHY", styles["table_head_dark"]), 
        Paragraph("HOW / NOTE", styles["table_head_dark"])
    ]]
    
    raw_actions = [
        ("Switch to New Regime", f"Saves {_fmt_rs(savings, False)}/year vs Old Regime", "Inform employer before April; applies from new FY"),
        ("Maximise Employer NPS", "Only deduction still valid under New Regime", "Up to 14% of Basic via Section 80CCD(2)"),
        ("Hold Equity Funds >1 Year", "LTCG 12.5% vs STCG 20%", "Avoid redemptions before 12-month mark"),
        ("Claim Standard Deduction", f"{_fmt_rs(75000, False)} automatically applied", "Confirm with payroll / CA before filing")
    ]
    
    for i, (act, why, how) in enumerate(raw_actions):
        action_rows.append([
            Paragraph(str(i+1), styles["table"]),
            Paragraph(f"<b>{act}</b>", styles["table"]),
            Paragraph(why, styles["table"]),
            Paragraph(how, styles["table"])
        ])
        
    actions_table = Table(action_rows, colWidths=[30, 130, 150, 160])
    actions_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))

    return [
        Spacer(1, 16),
        Paragraph("TAX ANALYSIS", ParagraphStyle("ta", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=2)),
        Paragraph("Regime Comparison & Structured Actions", ParagraphStyle("mk_tax_title", parent=styles["h2"], fontName="Times-Bold", fontSize=26, leading=30)),
        Spacer(1, 24),
        Paragraph("OLD VS NEW REGIME", ParagraphStyle("ovn", parent=styles["label"], textColor=colors.HexColor("#64748B"), fontSize=8)),
        Spacer(1, 8),
        top_split,
        Spacer(1, 32),
        Paragraph("KEY ACTIONS — WHAT, WHY & HOW", ParagraphStyle("ka", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=8)),
        actions_table
    ]


def build_page_action_plan(client_facts, allocation_output):
    styles = _report_styles()
    monthly_surplus = _display_monthly_surplus(client_facts)
    insurance_prov = _num(allocation_output.get("insurance_provision"), 0)
    surplus_after_ins = monthly_surplus - insurance_prov
    
    # Phase 1 Card
    p1_items = [
        Paragraph("PHASE 1 — MONTHS 1-6", ParagraphStyle("p1", parent=styles["label"], textColor=colors.HexColor("#A8813C"), fontSize=7)),
        Paragraph("Protection First", ParagraphStyle("p1t", parent=styles["h2"], fontSize=14, spaceBefore=4, spaceAfter=8)),
        Paragraph("Secure your family before deploying capital into goals.", ParagraphStyle("p1b", parent=styles["body"], fontSize=9, textColor=colors.HexColor("#64748B"), spaceAfter=12)),
    ]
    
    for p in (allocation_output.get("priority_breakdown") or []):
        if "Insurance" in p.get("name", ""):
            p1_items.append(Paragraph(f"→ {p['name']}: <b>{_fmt_rs(p['monthly_amount'], False)}/month</b>", styles["body"]))
            
    p1_items.append(Spacer(1, 12))
    p1_items.append(Paragraph(f"→ Surplus after insurance: <b>{_fmt_rs(surplus_after_ins, False)}/month</b> available for goals", styles["body"]))
    
    p1_card = Table([[item] for item in p1_items], colWidths=[240])
    p1_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FDF2F2")),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))

    # Phase 2 Card
    p2_head = [
        Paragraph("PHASE 2 — MONTH 7 ONWARDS", ParagraphStyle("p2", parent=styles["label"], textColor=colors.HexColor("#27AE60"), fontSize=7)),
        Paragraph("Goal-Based SIPs", ParagraphStyle("p2t", parent=styles["h2"], fontSize=14, spaceBefore=4, spaceAfter=8)),
        Paragraph(f"Deploy {_fmt_rs(surplus_after_ins, False)}/month across goals proportionally.", ParagraphStyle("p2b", parent=styles["body"], fontSize=9, textColor=colors.HexColor("#64748B"), spaceAfter=12)),
    ]
    
    p2_rows = [[
        Paragraph("GOAL", styles["table_head_dark"]), 
        Paragraph("ALLOC", styles["table_head_dark"]), 
        Paragraph("REQ", styles["table_head_dark"])
    ]]
    for g in (allocation_output.get("goal_sip_table") or [])[:5]:
        p2_rows.append([
            Paragraph(g.get("name") or "Goal", styles["table"]),
            Paragraph(f"<b>{_fmt_rs(g.get('allocated_sip'), False)}</b>", ParagraphStyle("ts", parent=styles["table"], textColor=colors.HexColor("#27AE60"))),
            Paragraph(_fmt_rs(g.get("ideal_sip"), False), styles["table"])
        ])
    
    p2_table = Table(p2_rows, colWidths=[90, 55, 55])
    p2_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    
    p2_items = p2_head + [p2_table]
    p2_card = Table([[item] for item in p2_items], colWidths=[240])
    p2_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F9F1")),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    
    top_split = Table([[p1_card, Spacer(1, 1), p2_card]], colWidths=[245, 10, 245])
    top_split.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    
    # Timeline
    def _timeline_item(week, title, items):
        res = [
            Paragraph(f"<font color='#A8813C'><b>{week} — {title.upper()}</b></font>", styles["label"]),
            Paragraph(title, ParagraphStyle("tt", parent=styles["h2"], fontSize=11, spaceBefore=2, spaceAfter=4)),
        ]
        for it in items:
            res.append(Paragraph(f"→ {it}", ParagraphStyle("ti", parent=styles["body"], fontSize=8, leftIndent=8)))
        return res

    col1 = [
        _timeline_item("WEEK 1", "Protection", ["Get term insurance — Rs. 4.4 Cr cover", "Upgrade health to Rs. 10-15L floater"]),
        Spacer(1, 8),
        _timeline_item("WEEK 2", "Liquidity", ["Assess savings + FD vs target", "Set up Tier 1/2/3 structure"]),
        Spacer(1, 8),
        _timeline_item("WEEK 3", "Portfolio", ["Plan LTCG-optimal exit", "Set up goal-wise SIPs"]),
    ]
    
    col2 = [
        _timeline_item("WEEK 4", "Tax", ["Switch to New Regime", "Maximise employer NPS"]),
        Spacer(1, 8),
        _timeline_item("DAY 90", "Full Review", ["Review all actions", "Recalculate Health Score"]),
        Spacer(1, 8),
        _timeline_item("QUARTERLY", "Ongoing", ["Ongoing Review", "Update plan for income changes"]),
    ]
    
    roadmap_split = Table([[
        Table([[item] for item in col1], colWidths=[230]),
        Table([[item] for item in col2], colWidths=[230])
    ]], colWidths=[245, 245])
    roadmap_split.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))

    return [
        Spacer(1, 8),
        Paragraph("YOUR ACTION PLAN", ParagraphStyle("yap", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=2)),
        Paragraph("Phase-wise Roadmap", ParagraphStyle("mk_ap_title", parent=styles["h2"], fontName="Times-Bold", fontSize=24, leading=28)),
        Spacer(1, 12),
        top_split,
        Spacer(1, 12),
        Paragraph("90-DAY ROADMAP", ParagraphStyle("90d", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=8)),
        roadmap_split
    ]


def build_page_review(client_facts, allocation_output):
    styles = _report_styles()
    
    # Left Card
    recalc_rows = [
        ("Quarterly", "Every 3 months — standard review cycle"),
        ("Income Change", "New job, salary revision, bonus, business income change"),
        ("Expense Change", "Major new EMI, lifestyle shift, new dependent"),
        ("Asset / Liability", "New loan, property purchase, large redemption"),
        ("Life Event", "Marriage, child birth, death in family, retirement"),
        ("Big Decision", "Before buying a home, taking a large loan, major investment"),
    ]
    recalc_data = []
    for k, v in recalc_rows:
        recalc_data.append([
            Paragraph(f"<b>{k}</b>", ParagraphStyle("rk", parent=styles["table"], textColor=colors.HexColor("#A8813C"))),
            Paragraph(v, styles["table"])
        ])
    recalc_table = Table(recalc_data, colWidths=[80, 140])
    recalc_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#EEEEEE")),
    ]))
    
    left_items = [
        Paragraph("WHEN TO RECALCULATE YOUR SCORE", ParagraphStyle("wt", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=12)),
        recalc_table,
        Spacer(1, 16),
        Table([[Paragraph("<b>Product Note</b><br/>Consider in-app nudges: \"Did you change jobs? Time to recalculate your Meerkat score.\"", ParagraphStyle("pn", parent=styles["body"], fontSize=8))]], colWidths=[220], style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")), ("PADDING", (0, 0), (-1, -1), 12)]))
    ]
    left_card = Table([[item] for item in left_items], colWidths=[240])
    left_card.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("PADDING", (0, 0), (-1, -1), 16),
    ]))
    
    # Right Card
    schedule_rows = [
        ("IMMEDIATE", "Term insurance + Emergency Fund setup", "#C0392B"),
        ("30 DAYS", "Portfolio rebalancing + Tax regime switch + SIP setup", "#A8813C"),
        ("90 DAYS", "Full progress review + recalculate Health Score", "#A8813C"),
        ("QUARTERLY", "Score review + SIP check + rebalance if equity drifts >5%", "#27AE60"),
        ("ANNUALLY", "Comprehensive plan update with advisor", "#64748B"),
    ]
    schedule_data = []
    for k, v, c in schedule_rows:
        schedule_data.append([
            Paragraph(f"<b>{k}</b>", ParagraphStyle("sk", parent=styles["table"], textColor=colors.HexColor(c))),
            Paragraph(v, styles["table"])
        ])
    schedule_table = Table(schedule_data, colWidths=[80, 140])
    schedule_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#EEEEEE")),
    ]))
    
    right_items = [
        Paragraph("REVIEW SCHEDULE", ParagraphStyle("rs", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=12)),
        schedule_table
    ]
    right_card = Table([[item] for item in right_items], colWidths=[240])
    right_card.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("PADDING", (0, 0), (-1, -1), 16),
    ]))
    
    top_split = Table([[left_card, Spacer(1, 1), right_card]], colWidths=[245, 10, 245])
    top_split.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    
    # Checklist Grid
    checklist = [
        "Financial Health Score across 6 dimensions", "Snapshot — Current vs Ideal with priorities",
        "Advanced Risk Assessment + Profile at a Glance", "Protection gap (Life + Health with guidance)",
        "Portfolio rebalancing plan", "Debt Management data (EMI ratio + benchmarks)",
        "Liquidity & Emergency Fund full assessment", "Goal feasibility for all 5 goals + SIP column",
        "Reality Check — cash flow vs goal requirement", "Goal-wise SIP Allocation plan",
        "Tax optimisation + structured action table", "90-Day Roadmap + Phase-wise Action Plan"
    ]
    grid_data = []
    for i in range(0, len(checklist), 2):
        row = []
        for j in range(2):
            if i + j < len(checklist):
                row.append(Paragraph(f"<font color='#27AE60'>✓</font> {checklist[i+j]}", styles["small"]))
            else:
                row.append("")
        grid_data.append(row)
        
    grid_table = Table(grid_data, colWidths=[245, 245])
    grid_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 10),
    ]))

    return [
        Spacer(1, 16),
        Paragraph("STAY ON TRACK", ParagraphStyle("sot", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=2)),
        Paragraph("When to Recalculate & Review Schedule", ParagraphStyle("mk_rev_title", parent=styles["h2"], fontName="Times-Bold", fontSize=26, leading=30)),
        Spacer(1, 24),
        top_split,
        Spacer(1, 32),
        Paragraph("WHAT THIS REPORT CONTAINS", ParagraphStyle("wtrc", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=12)),
        grid_table
    ]


def _meerkat_page_background(c, doc):
    c.saveState()
    c.setFillColor(MEERKAT_BG)
    page_w, page_h = A4
    c.rect(0, 0, page_w, page_h, stroke=0, fill=1)
    
    if doc.page > 1:
        if os.path.exists(LOGO_PATH):
            try:
                c.drawImage(LOGO_PATH, doc.leftMargin, page_h - 0.65 * inch, width=18, height=18, preserveAspectRatio=True, mask='auto')
                c.setFillColor(MEERKAT_NAVY)
                c.setFont("Times-Roman", 12)
                c.drawString(doc.leftMargin + 24, page_h - 0.58 * inch, "M E E R K A T")
            except Exception:
                pass
                
        titles = {
            2: "01 — Financial Snapshot",
            3: "02 — Executive Summary",
            4: "03 — Protection Gap",
            5: "04 — Portfolio & Debt",
            6: "05 — Liquidity Analysis",
            7: "06 — Goal Feasibility",
            8: "07 — Cash Flow & SIP Plan",
            9: "08 — Tax Optimisation",
            10: "09 — Action Plan",
            11: "10 — Review & Next Steps"
        }
        title_str = titles.get(doc.page, "")
        
        c.setFillColor(MEERKAT_NAVY)
        c.setFont("Helvetica-Bold", 10)
        if title_str:
            c.drawRightString(page_w - doc.rightMargin, page_h - 0.58 * inch, title_str)
            
        c.setFillColor(colors.HexColor("#7F8C8D"))
        c.setFont("Helvetica", 8)
        c.drawRightString(page_w - doc.rightMargin, page_h - 0.70 * inch, f"Page {doc.page} of 11")
        
        c.setFillColor(colors.HexColor("#95A5A6"))
        c.setFont("Helvetica", 7)
        c.drawString(doc.leftMargin, 0.4 * inch, "Educational analysis tool · Not financial advice · Consult a SEBI-registered Investment Advisor")
        c.drawRightString(page_w - doc.rightMargin, 0.4 * inch, f"Page {doc.page} of 11")
        
    c.restoreState()


def generate_financial_plan_pdf(q: dict, analysis: dict, output_path: str, doc_insights=None, narratives=None):
    """Generate the 11-page Meerkat Financial Health Report."""
    client_facts = _build_client_facts(q, analysis, doc_insights)
    narratives = _strip_in_app_report_text(narratives or {})
    allocation = _allocation_output(client_facts)
    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=0.65 * inch, leftMargin=0.65 * inch, topMargin=0.85 * inch, bottomMargin=0.55 * inch)
    builders = [
        lambda: build_page_cover(client_facts, allocation, narratives),
        lambda: build_page_snapshot(client_facts, allocation),
        lambda: build_page_executive_summary(client_facts, allocation, narratives),
        lambda: build_page_protection(client_facts, allocation),
        lambda: build_page_portfolio_debt(client_facts, allocation),
        lambda: build_page_liquidity(client_facts, allocation),
        lambda: build_page_goal_feasibility(client_facts, allocation),
        lambda: build_page_cashflow_sip(client_facts, allocation, narratives),
        lambda: build_page_tax(client_facts, allocation),
        lambda: build_page_action_plan(client_facts, allocation),
        lambda: build_page_review(client_facts, allocation),
    ]
    story = []
    for idx, builder in enumerate(builders):
        story.extend(builder())
        if idx < len(builders) - 1:
            story.append(PageBreak())
    doc.build(story, onFirstPage=_meerkat_page_background, onLaterPages=_meerkat_page_background)

@app.route("/report/generate", methods=["POST"])
@require_auth
@require_payment
@consume_credit
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
    
    # Include remaining credits from consume_credit decorator
    remaining_credits = getattr(g, 'remaining_credits', 0)
    
    return jsonify({
        "financial_plan_pdf_url": url,
        "summary_pdf_url": url,
        "report_type": "financial_plan",
        "questionnaire_id": questionnaire_id,
        "analysis": analysis,
        "docInsights": doc_insights,
        "sections": sections,
        "remaining_credits": remaining_credits
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
        "This report is for information purposes only. "
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

# =============================================================================
# AUTH AND PAYMENT ROUTES
# =============================================================================

@app.route("/auth/verify", methods=["POST"])
def auth_verify():
    """
    Verify Firebase ID token and create/update user in database.
    Called after client-side Firebase login.
    
    Request body:
        - id_token: Firebase ID token
    
    Returns:
        - User info and payment status
    """
    data = request.get_json(force=True) or {}
    id_token = data.get("id_token")
    
    if not id_token:
        return jsonify({"error": "id_token required"}), 400
    
    decoded = verify_firebase_token(id_token)
    if not decoded:
        return jsonify({"error": "Invalid or expired token"}), 401
    
    # Create or update user in database
    firebase_uid = decoded["uid"]
    user_id = create_or_update_user(
        firebase_uid=firebase_uid,
        email=decoded.get("email"),
        display_name=decoded.get("name")
    )
    
    # Get full user info
    user = get_user_by_firebase_uid(firebase_uid)
    
    return jsonify({
        "success": True,
        "user": {
            "firebase_uid": user["firebase_uid"],
            "email": user["email"],
            "display_name": user["display_name"],
            "has_paid": user["has_paid"],
            "payment_date": user["payment_date"],
            "report_credits": user.get("report_credits", 0),
        }
    }), 200


@app.route("/auth/status", methods=["GET"])
@require_auth
def auth_status():
    """
    Get current user's authentication and payment status.
    Requires valid Firebase ID token in Authorization header.
    """
    from flask import g
    user = g.current_user
    
    return jsonify({
        "authenticated": True,
        "user": {
            "firebase_uid": user["firebase_uid"],
            "email": user["email"],
            "display_name": user["display_name"],
            "has_paid": user["has_paid"],
            "report_credits": user.get("report_credits", 0),
            "payment_date": user["payment_date"],
        },
        "report_price_paise": get_report_price_paise(),
        "credits_per_payment": 3,
    }), 200


@app.route("/payment/create-order", methods=["POST"])
@require_auth
def payment_create_order():
    """
    Create a Razorpay order for payment.
    Requires valid Firebase ID token in Authorization header.
    
    Users can purchase multiple times to stack credits.
    Each payment grants 3 report credits.
    
    Returns:
        - Razorpay order details for client-side payment
    """
    from flask import g
    user = g.current_user
    
    # Note: We no longer block "already paid" users - they can buy more credits
    
    try:
        order = create_razorpay_order(firebase_uid=user["firebase_uid"])
        return jsonify({
            "success": True,
            "order": order,
            "user_email": user.get("email"),
            "current_credits": user.get("report_credits", 0),
            "credits_per_payment": 3,
        }), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Error creating Razorpay order: {e}")
        return jsonify({"error": "Failed to create payment order"}), 500


@app.route("/payment/verify", methods=["POST"])
@require_auth
def payment_verify():
    """
    Verify payment after successful Razorpay checkout.
    Requires valid Firebase ID token in Authorization header.
    
    Request body:
        - order_id: Razorpay order ID
        - payment_id: Razorpay payment ID
        - signature: Razorpay signature
    
    Returns:
        - Payment verification status and updated credit count
    """
    from flask import g
    from db import mark_user_as_paid, get_user_credits
    
    user = g.current_user
    data = request.get_json(force=True) or {}
    
    order_id = data.get("order_id")
    payment_id = data.get("payment_id")
    signature = data.get("signature")
    
    if not all([order_id, payment_id, signature]):
        return jsonify({"error": "Missing payment verification data"}), 400
    
    # Verify signature
    if not verify_razorpay_signature(order_id, payment_id, signature):
        return jsonify({"error": "Invalid payment signature"}), 400
    
    # Mark user as paid and add credits (default 3 credits per payment)
    success = mark_user_as_paid(
        firebase_uid=user["firebase_uid"],
        payment_id=payment_id,
        order_id=order_id,
        amount_paise=get_report_price_paise(),
        credits_to_add=3
    )
    
    if success:
        # Get updated credit count
        remaining_credits = get_user_credits(user["firebase_uid"])
        return jsonify({
            "success": True,
            "message": "Payment verified successfully. 3 report credits added.",
            "has_paid": True,
            "remaining_credits": remaining_credits,
            "credits_added": 3
        }), 200
    else:
        return jsonify({
            "error": "Failed to update payment status",
            "message": "Please contact support"
        }), 500


@app.route("/payment/webhook", methods=["POST"])
def payment_webhook():
    """
    Razorpay webhook handler for payment events.
    Verifies webhook signature and processes payment.captured events.
    
    IMPORTANT: Returns 4xx/5xx on errors so Razorpay will retry the webhook.
    """
    # Get raw payload for signature verification
    payload = request.get_data()
    signature = request.headers.get("X-Razorpay-Signature", "")
    
    if not signature:
        logger.warning("Webhook received without signature - rejecting")
        return jsonify({"error": "Missing signature"}), 400  # 400 so Razorpay knows it failed
    
    # Verify webhook signature
    if not verify_webhook_signature(payload, signature):
        logger.warning("Webhook signature verification failed")
        return jsonify({"error": "Invalid signature"}), 400
    
    # Parse payload
    try:
        data = request.get_json(force=True)
    except Exception as e:
        logger.error(f"Webhook JSON parse error: {e}")
        return jsonify({"error": "Invalid JSON"}), 400
    
    event = data.get("event")
    logger.info(f"Processing Razorpay webhook event: {event}")
    
    if event == "payment.captured":
        payment_entity = data.get("payload", {}).get("payment", {}).get("entity", {})
        if payment_entity:
            success = process_payment_captured(payment_entity)
            if not success:
                # Return 500 so Razorpay retries the webhook
                logger.error(f"Failed to process payment.captured for payment {payment_entity.get('id')}")
                return jsonify({"error": "Processing failed", "retry": True}), 500
        else:
            logger.warning("payment.captured event received but no payment entity found")
        return jsonify({"status": "processed"}), 200
    
    # Acknowledge other events
    logger.info(f"Acknowledged Razorpay webhook event: {event}")
    return jsonify({"status": "acknowledged"}), 200


@app.route("/payment/status", methods=["GET"])
@require_auth
def payment_status():
    """
    Get current user's payment status and credit information.
    """
    from flask import g
    user = g.current_user
    remaining_credits = user.get("report_credits", 0)
    
    return jsonify({
        "has_paid": user.get("has_paid", False),
        "payment_id": user.get("payment_id"),
        "payment_date": user.get("payment_date"),
        "report_price_paise": get_report_price_paise(),
        "remaining_credits": remaining_credits,
        "can_generate_report": remaining_credits > 0,
        "credits_per_payment": 3,
    }), 200



@app.route("/payment/reconcile", methods=["POST"])
@require_auth
def payment_reconcile():
    """
    Reconcile payment status by checking Razorpay directly.
    
    Use this when a user claims they paid but their status shows unpaid.
    This can happen if:
    - /payment/verify call failed/timed out
    - Webhook didn't fire or failed
    - Database wasn't updated for some reason
    
    This endpoint queries Razorpay for orders with the user's firebase_uid
    and marks them as paid if a completed payment is found.
    """
    from flask import g
    from db import mark_user_as_paid
    from payment import _get_razorpay_client
    
    user = g.current_user
    firebase_uid = user["firebase_uid"]
    
    # If already marked as paid in our system, no need to reconcile
    if user.get("has_paid"):
        return jsonify({
            "status": "already_paid",
            "message": "User already marked as paid in our system",
            "has_paid": True
        }), 200
    
    try:
        client = _get_razorpay_client()
        
        # Fetch recent orders from Razorpay (last 100)
        # We look for orders with this user's firebase_uid in notes
        orders = client.order.all({"count": 100})
        
        for order in orders.get("items", []):
            notes = order.get("notes", {})
            
            # Check if this order belongs to the current user
            if notes.get("firebase_uid") != firebase_uid:
                continue
            
            # Check if the order is paid
            if order.get("status") == "paid":
                order_id = order.get("id")
                
                # Get the payments for this order to find the payment_id
                try:
                    payments = client.order.payments(order_id)
                    payment_items = payments.get("items", [])
                    
                    if payment_items:
                        # Use the first captured/authorized payment
                        payment = payment_items[0]
                        payment_id = payment.get("id")
                        amount = order.get("amount")
                        
                        # Mark user as paid
                        success = mark_user_as_paid(
                            firebase_uid=firebase_uid,
                            payment_id=payment_id,
                            order_id=order_id,
                            amount_paise=amount
                        )
                        
                        if success:
                            logger.info(
                                f"Reconciliation successful: User {firebase_uid} marked as paid. "
                                f"Order: {order_id}, Payment: {payment_id}"
                            )
                            return jsonify({
                                "status": "reconciled",
                                "message": "Found paid order in Razorpay, user marked as paid",
                                "has_paid": True,
                                "order_id": order_id,
                                "payment_id": payment_id
                            }), 200
                        else:
                            logger.error(f"Reconciliation failed: Could not update user {firebase_uid}")
                            
                except Exception as e:
                    logger.error(f"Error fetching payments for order {order_id}: {e}")
                    continue
        
        # No paid orders found for this user
        logger.info(f"Reconciliation complete: No paid orders found for user {firebase_uid}")
        return jsonify({
            "status": "no_payment_found",
            "message": "No completed payment found in Razorpay for this user",
            "has_paid": False
        }), 200
        
    except Exception as e:
        logger.exception(f"Error during payment reconciliation for user {firebase_uid}: {e}")
        return jsonify({
            "error": "Reconciliation failed",
            "message": "Could not check payment status with Razorpay"
        }), 500

# =============================================================================
# ADMIN ROUTES
# =============================================================================

@app.route("/api/admin/stats", methods=["GET"])
@require_auth
@require_admin
def admin_stats():
    """Get dashboard statistics: total users, paid users, users with credits."""
    counts = get_user_count()
    return jsonify({"success": True, "stats": counts}), 200


@app.route("/api/admin/users", methods=["GET"])
@require_auth
@require_admin
def admin_list_users():
    """
    List all users with pagination and optional search.
    Query params: page (default 1), per_page (default 25), search (optional)
    """
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 25, type=int)
    search = request.args.get("search", "").strip() or None

    # Clamp per_page to prevent abuse
    per_page = min(max(per_page, 1), 100)

    users, total = list_all_users(page=page, per_page=per_page, search=search)
    return jsonify({
        "success": True,
        "users": users,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": (total + per_page - 1) // per_page if per_page else 1,
        },
    }), 200


@app.route("/api/admin/users/<firebase_uid>", methods=["GET"])
@require_auth
@require_admin
def admin_get_user(firebase_uid):
    """Get single user details including payment history."""
    user = get_user_by_firebase_uid(firebase_uid)
    if not user:
        return jsonify({"error": "User not found"}), 404

    payments = get_payment_history(firebase_uid)
    return jsonify({
        "success": True,
        "user": user,
        "payments": payments,
    }), 200


@app.route("/api/admin/users/<firebase_uid>", methods=["DELETE"])
@require_auth
@require_admin
def admin_delete_user(firebase_uid):
    """Delete a user and their payment history."""
    deleted = delete_user(firebase_uid)
    if not deleted:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"success": True, "message": "User deleted"}), 200


@app.route("/api/admin/users/<firebase_uid>/credits", methods=["POST"])
@require_auth
@require_admin
def admin_set_credits(firebase_uid):
    """
    Set or add credits for a user.
    Request body: { "credits": int, "mode": "set" | "add" }
    """
    data = request.get_json(force=True) or {}
    credits = data.get("credits")
    mode = data.get("mode", "set")

    if credits is None or not isinstance(credits, int) or credits < 0:
        return jsonify({"error": "Valid non-negative integer 'credits' required"}), 400

    user = get_user_by_firebase_uid(firebase_uid)
    if not user:
        return jsonify({"error": "User not found"}), 404

    if mode == "add":
        from db import add_user_credits
        success = add_user_credits(firebase_uid, credits)
    else:
        success = set_user_credits(firebase_uid, credits)

    if not success:
        return jsonify({"error": "Failed to update credits"}), 500

    updated_user = get_user_by_firebase_uid(firebase_uid)
    return jsonify({
        "success": True,
        "message": f"Credits {'added' if mode == 'add' else 'set'} successfully",
        "report_credits": updated_user.get("report_credits", 0),
    }), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
