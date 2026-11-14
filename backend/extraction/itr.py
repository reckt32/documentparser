"""Income tax return (ITR) extraction."""

import json
import logging
import re

from backend.config import client
from backend.extraction.metadata import _add_extraction_metadata, _create_extraction_meta
from backend.extraction.pdf_text import _select_text_for_llm
from backend.utils import clean_and_convert_to_float

logger = logging.getLogger(__name__)


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
    
    ITR Document Text (head and tail of document):
    {_select_text_for_llm(text, 20000)}
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
