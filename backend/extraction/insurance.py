"""Insurance policy extraction."""

import json
import logging
import re

from backend.config import client
from backend.extraction.metadata import (
    _add_extraction_metadata,
    _create_extraction_meta,
    _log_extraction,
    _merge_llm_data,
)
from backend.extraction.pdf_text import _select_text_for_llm
from backend.utils import clean_and_convert_to_float

logger = logging.getLogger(__name__)


def extract_insurance_hybrid(text):
    data = {}
    
    # Score each type by keyword frequency instead of life-first ordering:
    # health policies routinely mention "life" in insurer names/footers, which
    # used to misclassify them (and route health cover into life cover).
    life_score = len(re.findall(r"(?i)(life\s+insurance|term\s+plan|term\s+insurance|endowment|ULIP|whole\s+life|sum\s+assured|death\s+benefit|maturity\s+benefit)", text))
    health_score = len(re.findall(r"(?i)(health\s+insurance|mediclaim|medical\s+insurance|hospitali[sz]ation|health\s*cover|in-?patient\s+treatment|annual\s+sum\s+insured|sum\s+insured|health\s+advantedge|health\s+plan|health\s+policy|medicare\s*premier|tata\s*aig\s*medicare|room\s+rent|cashless|day\s*care\s+procedure|pre-?existing\s+disease)", text))
    general_score = len(re.findall(r"(?i)(motor\s+insurance|vehicle\s+insurance|property\s+insurance|home\s+insurance|fire\s+insurance|own\s+damage|third\s+party|IDV)", text))

    logger.debug(f"Insurance type scores - life: {life_score}, health: {health_score}, general: {general_score}")

    best_score = max(life_score, health_score, general_score)
    if best_score == 0:
        data["insurance_type"] = "Unknown"
    elif health_score == best_score:
        # Ties favor health: life phrases leak into health policies far more
        # often than the reverse
        data["insurance_type"] = "Health Insurance"
    elif life_score == best_score:
        data["insurance_type"] = "Life Insurance"
    else:
        data["insurance_type"] = "General Insurance"

    is_life_insurance = data["insurance_type"] == "Life Insurance"

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
    
    Insurance Document Text (head and tail of document):
    {_select_text_for_llm(text, 20000)}
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
        # Regex-extracted policy facts are authoritative; LLM fills gaps only.
        _merge_llm_data(data, llm_data, protected=(
            "insurance_type", "policy_number", "insurer_name", "policy_holder",
            "policy_term", "policy_start_date", "policy_end_date", "nominee",
            "date_of_birth", "sum_assured_or_insured", "premium_amount",
            "premium_frequency",
        ))
    except Exception as e:
        logger.error(f"LLM extraction error for insurance: {str(e)}")
        data['extraction_error'] = str(e)
        _add_extraction_metadata(data, {
            "_llm_extraction": {"confidence": 0.0, "source": "llm", "needs_review": True, "reason": str(e)}
        })
        
    return data
