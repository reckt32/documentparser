from flask import Flask, request, jsonify, url_for, send_from_directory
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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Frame, PageTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
from reportlab.pdfgen import canvas
from extractors import index_and_extract
from db import list_sections, list_metrics, get_document_by_sha
# --- Initialization ---
load_dotenv()

app = Flask(__name__)
CORS(app)
# Respect reverse proxy headers on Render (scheme/host) for correct external URLs
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
app.config['PREFERRED_URL_SCHEME'] = 'https'
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
                {"role": "system", "content": "You are a financial analyst expert at extracting structured data from bank statements. Return valid JSON only."},
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
                {"role": "system", "content": "You are a tax expert analyzing ITR documents. Extract all financial data accurately. Return valid JSON only."},
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
                {"role": "system", "content": "You are an insurance expert analyzing insurance policies. Extract all details accurately based on the insurance type. Return valid JSON only."},
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
                {"role": "system", "content": "You are a mutual fund analysis expert. Extract structured data into a valid JSON format."},
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
        flags.append(f"Underinsured: Cover Rs. {int(round(life_cover))} vs Required Rs. {int(round(required))}")
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
    rp = results.get("riskProfile")
    if rp == "Conservative":
        recs.append("Focus on debt-heavy allocation and low-volatility funds.")
    elif rp == "Balanced":
        recs.append("Aim for a 60:40 equity-debt mix.")
    elif rp == "Aggressive":
        recs.append("Target 70-80% equity allocation; consider adding gold as a hedge.")

    sb = results.get("surplusBand")
    if sb == "Low":
        recs.append("Cut discretionary spending; automate an SIP.")
    elif sb == "Adequate":
        recs.append("Maintain discipline and allocate savings across your goals.")
    elif sb == "Strong":
        recs.append("Accelerate retirement savings and explore growth instruments.")

    ig = results.get("insuranceGap")
    if ig == "Underinsured":
        recs.append("Buy term life insurance for 10-12x your income and add/increase health cover.")
    else:
        recs.append("Review your cover every 2-3 years.")

    ds = results.get("debtStress")
    if ds == "Stressed":
        recs.append("Refinance high-cost debt; repay high-cost debt first; avoid new borrowing.")
    elif ds == "Moderate":
        recs.append("Prioritize repaying unsecured debt.")
    else:
        recs.append("You can leverage strategically if needed.")

    liq = results.get("liquidity")
    if liq == "Insufficient":
        recs.append("Build a liquid buffer (cash/liquid funds) until 6 months of expenses are covered.")
    else:
        recs.append("Invest your incremental surplus into growth assets.")

    ihs_band = ihs.get("band")
    if ihs_band == "Poor":
        recs.append("Increase your savings rate, exit unsuitable products, and diversify your portfolio.")
    elif ihs_band == "Average":
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

    risk_profile, risk_score = compute_risk_profile(age, tolerance, horizon)
    surplus_band = compute_surplus_band(savings_percent)
    insurance_gap, required_cover = compute_insurance_gap(income.get("annualIncome"), insurance.get("lifeCover"))
    debt_stress, emi_ratio_pct = compute_debt_stress(income.get("monthlyEmi"), income.get("annualIncome"))
    liquidity, liquidity_months = compute_liquidity(income.get("monthlyExpenses"), emergency_fund_amount)
    ihs = compute_ihs(savings_percent, current_products, allocation)

    results = {
        "riskProfile": risk_profile,
        "surplusBand": surplus_band,
        "insuranceGap": insurance_gap,
        "debtStress": debt_stress,
        "liquidity": liquidity,
        "ihs": ihs,
        # extra diagnostics
        "_diagnostics": {
            "riskScore": risk_score,
            "emiPct": round(emi_ratio_pct, 2),
            "liquidityMonths": round(liquidity_months, 2),
            "requiredLifeCover": required_cover
        }
    }

    flags, recs = generate_flags_and_recommendations(results, payload)
    results["flags"] = flags
    results["recommendations"] = recs
    return results

# --- Flask Routes ---

@app.route('/financial-health/analyze', methods=['POST'])
def financial_health_analyze():
    """Analyzes financial health questionnaire and returns computed metrics, flags, and recommendations."""
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400
    try:
        result = analyze_financial_health(data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

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

    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    extracted_data = {}
    debug_shas = []
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
                    # Build structured transactions and persist as JSON
                    tx_payload = extract_bank_statement_transactions(io.BytesIO(file_bytes))
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
                    extracted_data[f"{name} {idx+1}"] = other_data
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

    # Optionally include questionnaire analysis if provided (multipart field 'questionnaire' as JSON)
    try:
        q_str = request.form.get('questionnaire')
        if q_str:
            q_payload = json.loads(q_str)
            fha = analyze_financial_health(q_payload)
            # Attach original inputs for charting in PDF
            fha["_inputs"] = q_payload
            extracted_data["Financial Health Analysis"] = fha
    except Exception as _:
        pass


    pdf_filename = f"PortfolioSummary_{os.urandom(8).hex()}.pdf"
    pdf_path = os.path.join(OUTPUT_DIR, pdf_filename)
    generate_pdf_summary(extracted_data, pdf_path)

    download_url = url_for('download_file', filename=pdf_filename, _external=True)
    return jsonify({
        "summary_pdf_url": download_url,
        "debug_shas": debug_shas
    }), 200

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

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


# --- PDF Generation Logic (Enhanced) ---

def get_custom_styles():
    """Returns a dictionary of custom ReportLab paragraph styles."""
    styles = getSampleStyleSheet()

    # --- Modify existing styles using their default names ---

    # Modify 'Title'
    title_style = styles['Title']
    title_style.fontSize = 22
    title_style.alignment = TA_CENTER
    title_style.spaceAfter = 20
    title_style.textColor = colors.HexColor('#1D3557')

    # Modify 'h1'
    h1_style = styles['h1']
    h1_style.fontSize = 16
    h1_style.leading = 22
    h1_style.spaceAfter = 12
    h1_style.textColor = colors.HexColor('#457B9D')

    # Modify 'h2'
    h2_style = styles['h2']
    h2_style.fontSize = 12
    h2_style.leading = 18
    h2_style.spaceBefore = 10
    h2_style.spaceAfter = 6
    h2_style.textColor = colors.HexColor('#E63946')

    # Modify 'BodyText'
    body_style = styles['BodyText']
    body_style.fontSize = 10
    body_style.leading = 14
    body_style.spaceAfter = 6
    
    # --- Add ONLY new, unique styles ---
    styles.add(ParagraphStyle(name='Header', fontSize=8, alignment=TA_RIGHT, textColor=colors.grey))
    styles.add(ParagraphStyle(name='Footer', fontSize=8, alignment=TA_CENTER, textColor=colors.grey))
    styles.add(ParagraphStyle(name='TableHead', fontSize=9, fontName='Helvetica-Bold', alignment=TA_LEFT, textColor=colors.white))
    styles.add(ParagraphStyle(name='TableCell', fontSize=9, alignment=TA_LEFT))

    return styles

def header(canvas, doc):
    """Adds a header to each page."""
    canvas.saveState()
    styles = get_custom_styles()
    p = Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Header'])
    w, h = p.wrap(doc.width, doc.topMargin)
    p.drawOn(canvas, doc.leftMargin, doc.height + doc.topMargin - h)
    canvas.restoreState()

def footer(canvas, doc):
    """Adds a footer with page number to each page."""
    canvas.saveState()
    styles = get_custom_styles()
    p = Paragraph(f"Page {doc.page}", styles['Footer'])
    w, h = p.wrap(doc.width, doc.bottomMargin)
    p.drawOn(canvas, doc.leftMargin, h)
    canvas.restoreState()

def build_pdf_story(story, data, doc, styles):
    """Recursively builds the PDF story from the extracted data dictionary."""
    empty_values = [None, "", "N/A", [], {}]
    if not isinstance(data, dict):
        return

    for key, value in data.items():
        # Skip attaching raw transactions JSON link into nested PDF content
        if key in ["transactions", "transactions_json_path"]:
            if key == "transactions_json_path" and isinstance(value, str):
                # Print a short line with the link instead
                story.append(Paragraph(f"Transactions JSON: {value}", styles["BodyText"]))
                story.append(Spacer(1, 4))
            continue
        if value in empty_values or key in ['extraction_error', 'llm_error']:
            continue
        
        key_title = key.replace('_', ' ').title()
        
        if isinstance(value, list) and value and isinstance(value[0], dict):
            # It's a list of dictionaries, create a table
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph(key_title, styles['h2']))
            add_dynamic_table(story, value, doc, styles)
        elif isinstance(value, dict):
            # It's a nested dictionary, recurse
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph(key_title, styles['h2']))
            build_pdf_story(story, value, doc, styles)
        else:
            # It's a simple key-value pair
            formatted_value = f"₹ {value:,.2f}" if isinstance(value, (int, float)) else value
            p_text = f"<b>{key_title}:</b> {formatted_value}"
            story.append(Paragraph(p_text, styles["BodyText"]))
            story.append(Spacer(1, 4))

def add_dynamic_table(story, data_list, doc, styles):
    """Creates a ReportLab table from a list of dictionaries."""
    filtered_data = [row for row in data_list if any(v not in [None, "", "N/A"] for v in row.values())]
    if not filtered_data:
        return

    headers = list(filtered_data[0].keys())
    header_row = [Paragraph(h.replace('_', ' ').title(), styles['TableHead']) for h in headers]
    
    table_data = [header_row]
    for row_dict in filtered_data:
        row = []
        for h in headers:
            cell_value = row_dict.get(h, "")
            # Format numbers in table cells
            if isinstance(cell_value, (int, float)):
                cell_value = f"₹ {cell_value:,.2f}"
            row.append(Paragraph(str(cell_value), styles['TableCell']))
        table_data.append(row)

    table = Table(table_data, hAlign='LEFT', repeatRows=1)
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
    """Generates the final PDF summary report."""
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            rightMargin=inch*0.75, leftMargin=inch*0.75,
                            topMargin=inch, bottomMargin=inch)
    styles = get_custom_styles()
    story = [Paragraph("Holistic Financial Report", styles['Title']), Spacer(1, 0.25 * inch)]

    for section_key, section_content in summary_data.items():
        if not section_content or section_content.get("error"):
            story.append(Paragraph(f"Could not process: {section_key}", styles['h1']))
            if isinstance(section_content, dict) and section_content.get("error"):
                story.append(Paragraph(f"Reason: {section_content['error']}", styles['BodyText']))
            story.append(Spacer(1, 0.2 * inch))
            continue

        story.append(Paragraph(section_key.replace('_', ' ').title(), styles['h1']))
        story.append(Spacer(1, 0.15 * inch))
        # Charts & insights
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

            # Bank Statement: cash flows
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
                            story.append(Paragraph(f"{note}: ₹ {abs(net):,.0f}", styles['BodyText']))
                            story.append(Spacer(1, 0.1 * inch))
                    except Exception:
                        pass

            # Mutual Fund CAS: asset allocation
            if "mutual fund cas" in key_l:
                alloc = (section_content or {}).get("asset_allocation") or {}
                if any(k in alloc for k in ["equity_percentage", "debt_percentage", "hybrid_percentage"]):
                    _add_pie_chart("Asset Allocation (CAS)", {
                        "Equity": alloc.get("equity_percentage", 0),
                        "Debt": alloc.get("debt_percentage", 0),
                        "Hybrid": alloc.get("hybrid_percentage", 0),
                    })

            # ITR: income sources
            if key_l.startswith("itr"):
                inc = (section_content or {}).get("income_sources") or {}
                if inc:
                    _add_pie_chart("Income Sources (ITR)", inc)

            # Financial Health Analysis: user's allocation from questionnaire
            if "financial health analysis" in key_l:
                q_inputs = (section_content or {}).get("_inputs") or {}
                inv = (q_inputs.get("investments") or {})
                alloc = inv.get("allocation") or {}
                if alloc:
                    _add_pie_chart("Portfolio Allocation (Questionnaire)", {
                        "Equity": alloc.get("equity", 0),
                        "Debt": alloc.get("debt", 0),
                        "Gold": alloc.get("gold", 0),
                        "Real Estate": alloc.get("realEstate", 0),
                        "Insurance-linked": alloc.get("insuranceLinked", 0),
                        "Cash": alloc.get("cash", 0),
                    })
        except Exception:
            # Keep PDF generation resilient
            pass

        build_pdf_story(story, section_content, doc, styles)
        story.append(PageBreak())

    # Remove the last page break if it exists
    if story and isinstance(story[-1], PageBreak):
        story.pop()

    # --- FIX IS HERE ---
    # 1. Define the class before it is used.
    class CustomCanvas(canvas.Canvas):
        """A custom canvas class that knows how to draw the page footer."""
        page_fn = footer

    # 2. Use the EXACT same name (CustomCanvas) as the canvasmaker.
    doc.build(story, onFirstPage=header, onLaterPages=header, canvasmaker=CustomCanvas)


# --- Main Execution ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on the network; respect PORT for Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
