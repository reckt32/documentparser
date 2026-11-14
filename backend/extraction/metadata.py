"""Extraction logging and per-field provenance metadata."""

import logging
import re

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


def _merge_llm_data(data: dict, llm_data: dict, protected: tuple = ()) -> None:
    """
    Merge LLM-extracted fields into `data` with explicit precedence:
    deterministic/regex values win, the LLM only fills gaps.

    For keys in `protected` that already hold a real value (not None/""/"N/A"),
    a conflicting LLM value is logged and discarded. All other LLM keys are
    merged as before, so the output shape is unchanged.
    """
    if not isinstance(llm_data, dict):
        return
    for key, value in llm_data.items():
        existing = data.get(key)
        if key in protected and existing not in (None, "", "N/A"):
            if value not in (None, "", "N/A") and value != existing:
                logger.info(
                    f"LLM value for '{key}' discarded (regex wins): "
                    f"regex={_redact_pii(str(existing)[:80])} llm={_redact_pii(str(value)[:80])}"
                )
                _add_extraction_metadata(data, {
                    key: {**data.get("_extraction_meta", {}).get(key, {}),
                          "llm_disagrees": True}
                })
            continue
        data[key] = value
