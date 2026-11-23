"""Small numeric / formatting helpers shared across the backend."""

import math
import re


def inr(amount, decimals: int = 0) -> str:
    """Indian digit grouping: 1234567.8 -> '12,34,568' (decimals=0) or '12,34,567.80'."""
    s = f"{abs(float(amount or 0)):.{decimals}f}"
    whole, dot, frac = s.partition(".")
    head, tail = whole[:-3], whole[-3:]
    while len(head) > 2:
        tail = f"{head[-2:]},{tail}"
        head = head[:-2]
    whole = f"{head},{tail}" if head else tail
    return ("-" if float(amount or 0) < 0 and s.strip("0.") else "") + whole + dot + frac


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


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            f = float(x)
            return f if math.isfinite(f) else default
        s = str(x).strip()
        v = clean_and_convert_to_float(s)
        if v == "N/A":
            return default
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))
