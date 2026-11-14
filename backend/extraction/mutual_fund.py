"""Mutual fund CAS / portfolio statement extraction."""

import json
import logging
import re

from backend.config import client
from backend.extraction.metadata import (
    _add_extraction_metadata,
    _create_extraction_meta,
    _merge_llm_data,
)
from backend.extraction.pdf_text import _select_text_for_llm
from backend.utils import clean_and_convert_to_float

logger = logging.getLogger(__name__)


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
    
    Mutual Fund CAS Text (head and tail of document):
    {_select_text_for_llm(text, 20000)}
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
        # Regex-extracted identity fields are authoritative; LLM fills gaps only.
        _merge_llm_data(data, llm_data, protected=(
            "investor_name", "pan", "email", "folio_numbers",
        ))
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
