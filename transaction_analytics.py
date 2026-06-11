"""
Deterministic analytics over structured bank-statement transactions.

Works on the rows produced by app.extract_bank_statement_transactions():
  {date, description, reference, debit, credit, amount, type, balance}

Outputs recurring groups in the SAME shape the LLM extraction emits
({description, amount, frequency, dates, is_emi}) so downstream consumers
(questionnaire upload metadata, prefill monthly_emi) are unaffected.
"""

import re
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Single source for EMI keyword rules (mirrors build_prefill_from_insights).
EMI_KEYWORDS = [
    "emi", "loan", "mortgage", "instalment", "installment", "repayment",
    "home loan", "car loan", "personal loan", "vehicle loan", "housing loan",
    "education loan",
    "elm",            # ICICI EMI/Loan
    "hdfc ln", "sbi ln", "icici ln", "axis ln", "kotak ln",
    "emi ded",        # SBI pattern
    "bil/inft/elm",   # ICICI pattern
]
# Weaker signals: auto-debit rails used for EMIs but also for SIPs/bills.
EMI_WEAK_KEYWORDS = ["nach", "ecs", "si/", "standing instruction", "auto debit", "autodebit"]


def is_emi_description(desc: str, recurring_monthly: bool = False) -> bool:
    """True if a transaction description looks like an EMI/loan repayment.

    Weak rail keywords (NACH/ECS/SI) only count when the transaction is also
    known to recur monthly, since those rails carry SIPs and bill payments too.
    """
    d = (desc or "").lower()
    if not d:
        return False
    if any(kw in d for kw in EMI_KEYWORDS):
        return True
    if recurring_monthly and any(kw in d for kw in EMI_WEAK_KEYWORDS):
        return True
    return False


def _parse_iso(s) -> Optional[datetime]:
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d")
    except Exception:
        return None


_REF_TOKEN = re.compile(r"\d{3,}")          # long digit runs are refs/UTRs
_DATE_TOKEN = re.compile(r"\b\d{1,2}[-/][a-z0-9]{2,3}[-/]\d{2,4}\b", re.I)
_SEPARATORS = re.compile(r"[\s/\\\-_:.,*#]+")


def normalize_description(desc: str) -> str:
    """Collapse a narration into a stable group key (refs/dates stripped)."""
    d = (desc or "").lower()
    d = _DATE_TOKEN.sub(" ", d)
    d = _REF_TOKEN.sub(" ", d)
    d = _SEPARATORS.sub(" ", d).strip()
    tokens = d.split()
    return " ".join(tokens[:5])


def _classify_frequency(dates: List[datetime]) -> str:
    if len(dates) < 2:
        return "Ad-hoc"
    ds = sorted(dates)
    gaps = [(b - a).days for a, b in zip(ds, ds[1:]) if (b - a).days > 0]
    if not gaps:
        return "Ad-hoc"
    gaps.sort()
    median_gap = gaps[len(gaps) // 2]
    if 6 <= median_gap <= 8:
        return "Weekly"
    if 25 <= median_gap <= 35:
        return "Monthly"
    if 80 <= median_gap <= 100:
        return "Quarterly"
    if 350 <= median_gap <= 380:
        return "Yearly"
    return "Ad-hoc"


def _txn_amount(txn: dict, side: str) -> Optional[float]:
    """Amount of a txn on the given side ('debit'/'credit'), else None."""
    v = txn.get(side)
    if isinstance(v, (int, float)) and v > 0:
        return float(v)
    amt = txn.get("amount")
    if isinstance(amt, (int, float)) and amt > 0 and txn.get("type") == side:
        return float(amt)
    return None


def detect_recurring(transactions: List[dict], side: str, min_occurrences: int = 2) -> List[Dict]:
    """Group same-description, similar-amount transactions into recurring entries.

    Returns entries shaped exactly like the LLM output consumed downstream:
      {description, amount, frequency, dates, is_emi}
    (is_emi only present for debits.)
    """
    groups = defaultdict(list)
    for t in transactions or []:
        amt = _txn_amount(t, side)
        if amt is None:
            continue
        key = normalize_description(t.get("description") or "")
        if not key:
            continue
        groups[key].append((t, amt))

    out = []
    for key, items in groups.items():
        # Split a description group into amount clusters (±2% or ±10 Rs)
        items.sort(key=lambda x: x[1])
        clusters: List[List] = []
        for t, amt in items:
            placed = False
            for cl in clusters:
                ref = cl[0][1]
                if abs(amt - ref) <= max(10.0, 0.02 * ref):
                    cl.append((t, amt))
                    placed = True
                    break
            if not placed:
                clusters.append([(t, amt)])

        for cl in clusters:
            if len(cl) < min_occurrences:
                continue
            dates_dt = [d for d in (_parse_iso(t.get("date")) for t, _ in cl) if d]
            # Need real dates on most rows to call it recurring
            if len(dates_dt) < min_occurrences:
                continue
            freq = _classify_frequency(dates_dt)
            if freq == "Ad-hoc" and len(cl) < 3:
                continue  # two same-amount txns with odd spacing: not recurring
            amounts = sorted(a for _, a in cl)
            amount = amounts[len(amounts) // 2]  # median
            # Representative (longest) raw description for readability
            description = max((t.get("description") or "" for t, _ in cl), key=len)
            entry = {
                "description": description,
                "amount": round(amount, 2),
                "frequency": freq,
                "dates": sorted(d.strftime("%Y-%m-%d") for d in dates_dt),
                "source": "deterministic",
            }
            if side == "debit":
                entry["is_emi"] = is_emi_description(description, recurring_monthly=(freq == "Monthly"))
            out.append(entry)

    out.sort(key=lambda e: -e["amount"])
    return out


def detect_recurring_transactions(transactions: List[dict]) -> Dict[str, List[Dict]]:
    """Recurring debits and credits from structured transaction rows."""
    return {
        "recurring_debits": detect_recurring(transactions, "debit"),
        "recurring_credits": detect_recurring(transactions, "credit"),
    }


def compute_average_monthly_balance(transactions: List[dict]) -> Optional[float]:
    """Mean of month-end balances, from rows that carry a running balance."""
    month_last = {}
    for t in transactions or []:
        bal = t.get("balance")
        d = _parse_iso(t.get("date"))
        if d is None or not isinstance(bal, (int, float)):
            continue
        key = (d.year, d.month)
        prev = month_last.get(key)
        if prev is None or d >= prev[0]:
            month_last[key] = (d, float(bal))
    if not month_last:
        return None
    vals = [v for _, v in month_last.values()]
    return round(sum(vals) / len(vals), 2)


def statement_months(transactions: List[dict]) -> Optional[float]:
    """Approximate statement span in months from transaction dates (>= ~0.25)."""
    dates = [d for d in (_parse_iso(t.get("date")) for t in (transactions or [])) if d]
    if len(dates) < 2:
        return None
    span_days = (max(dates) - min(dates)).days
    if span_days <= 0:
        return None
    return _snap_months(max(0.25, span_days / 30.44))


_PERIOD_DATE_FMTS = [
    "%d-%b-%Y", "%d %b %Y", "%d/%b/%Y", "%d-%b-%y", "%d/%b/%y",
    "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%y", "%d/%m/%y",
    "%Y-%m-%d",
]


def _parse_flex_date(s) -> Optional[datetime]:
    s = str(s or "").strip()
    for f in _PERIOD_DATE_FMTS:
        try:
            return datetime.strptime(s, f)
        except ValueError:
            continue
    return None


def _snap_months(months: float) -> float:
    """Snap near-integer month counts (e.g. 2.99 -> 3.0, 12.02 -> 12.0)."""
    nearest = round(months)
    if nearest >= 1 and abs(months - nearest) <= 0.25:
        return float(nearest)
    return round(months, 2)


def months_from_period(period_str) -> Optional[float]:
    """Months covered by a declared statement period like
    '01-Apr-2024 to 30-Jun-2024' (the format the extractor stores)."""
    if not period_str:
        return None
    parts = re.split(r"\s+to\s+", str(period_str), flags=re.I)
    if len(parts) != 2:
        return None
    d1, d2 = _parse_flex_date(parts[0]), _parse_flex_date(parts[1])
    if d1 is None or d2 is None or d2 <= d1:
        return None
    days = (d2 - d1).days + 1  # inclusive period
    return _snap_months(max(0.25, days / 30.44))


def _coerce_amount(v) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    try:
        s = re.sub(r"[₹$,\s]", "", str(v))
        return float(s) if s else None
    except Exception:
        return None


def _clean_llm_entry(e: Dict) -> Dict:
    """Coerce an LLM recurring entry's amount to float (downstream consumers
    skip non-numeric amounts silently)."""
    amt = _coerce_amount(e.get("amount"))
    if amt is not None and not isinstance(e.get("amount"), (int, float)):
        e = dict(e)
        e["amount"] = amt
    return e


def merge_recurring(deterministic: List[Dict], llm_entries: List[Dict]) -> List[Dict]:
    """Deterministic groups win; keep LLM entries whose description doesn't
    overlap any deterministic group (covers txn-table extraction failures and
    single-occurrence EMIs the keyword rules catch from raw text)."""
    llm_clean = [_clean_llm_entry(e) for e in (llm_entries or []) if isinstance(e, dict)]
    if not deterministic:
        return llm_clean
    seen = {normalize_description(e.get("description") or "") for e in deterministic}
    merged = list(deterministic)
    for e in llm_clean:
        key = normalize_description(e.get("description") or "")
        if key and key in seen:
            continue
        merged.append(e)
    return merged
