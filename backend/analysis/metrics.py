"""Core financial-health metrics (savings, insurance, debt, liquidity, retirement)."""

from backend.assumptions import LIFE_COVER_MULTIPLE, WITHDRAWAL_RATE_RETIREMENT
from backend.utils import _clamp, _safe_float


def _band_midpoint(band: str):
    if not band:
        return None
    b = str(band).strip().lower().replace("%", "").replace(" ", "")
    if b.startswith("<") or b in ("0-10", "under10", "below10"):
        return 5.0
    if "10-20" in b:
        return 15.0
    if "20-30" in b:
        return 25.0
    # ">30", "30+", "30andabove", "above30" — any "more than 30" phrasing
    if b.startswith(">") or b.endswith("+") or b.startswith("30") or "above" in b:
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


def declared_savings_percent(income: dict):
    """
    Savings rate from the client's own declared cashflow:
    (monthly income − expenses − EMI) / monthly income.

    This is the same arithmetic the report's snapshot page prints as
    "Savings rate", so the Surplus Level band can never contradict it.
    Returns None when income or expenses are missing.
    """
    income = income or {}
    ai = _safe_float(income.get("annualIncome"), 0.0)
    me = _safe_float(income.get("monthlyExpenses"), 0.0)
    emi = _safe_float(income.get("monthlyEmi"), 0.0)
    if ai <= 0 or me <= 0:
        return None
    monthly_income = ai / 12.0
    sp = (monthly_income - me - emi) / monthly_income * 100.0
    return _clamp(sp, 0.0, 100.0)


def effective_savings_percent(payload: dict):
    """
    Single source of truth for the savings rate used by surplus band, IHS and
    flags. Declared cashflow wins (it's complete and what the report displays);
    doc-derived savingsPercent and the self-reported band are fallbacks.
    """
    payload = payload or {}
    sp = declared_savings_percent(payload.get("income") or {})
    if sp is not None:
        return sp
    savings = payload.get("savings") or {}
    return resolve_savings_percent(savings.get("savingsPercent"), savings.get("savingsBand"))


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
    required = LIFE_COVER_MULTIPLE * ai
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


def compute_retirement_corpus(age, monthly_income, desired_monthly_pension=None, withdrawal_rate=WITHDRAWAL_RATE_RETIREMENT, retirement_age=60):
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
            # Ideal retirement corpus = annual pension / withdrawal rate (perpetuity formula)
            pension_corpus = annual_pension / withdrawal_rate
            
            result["desired_monthly_pension"] = round(pension, 0)
            result["pension_annual"] = round(annual_pension, 0)
            result["pension_corpus"] = round(pension_corpus, 0)
            result["withdrawal_rate_percent"] = withdrawal_rate * 100
    
    return result


def compute_term_insurance_need(age, monthly_income, retirement_age=60):
    """
    Calculate term insurance requirement.
    Formula: LIFE_COVER_MULTIPLE x annual income (single source of truth in
    assumptions.py), so this matches compute_insurance_gap, the protection
    score, and the allocation engine's term gap.

    For example, at a 10x multiple: ₹30,000/month = ₹3.6 lakh/year
    → Required cover = ₹36 lakh.

    Returns the required term cover amount.
    """
    mi = _safe_float(monthly_income, 0.0)
    annual_income = mi * 12

    required_cover = annual_income * LIFE_COVER_MULTIPLE
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
