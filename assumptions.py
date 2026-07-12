"""
Single source of truth for financial-planning assumptions.

Every module that needs a life-cover multiple, a safe-withdrawal rate, an
inflation assumption, or an expected-return figure imports it from here. The
goal is that no two pages of the same report can contradict each other because
they each hard-coded their own number. See backend/docs/ASSESSMENT.md §2.2.

Override at runtime via environment variables where noted.
"""

import os


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.getenv(name)
        return float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


# Life insurance: required cover = LIFE_COVER_MULTIPLE x annual income.
# Standard income-replacement rule of thumb (commonly 10-12x). This single
# value drives the "Underinsured" flag, the protection score, the term-need
# shown on the Protection page, and the allocation engine's term gap.
LIFE_COVER_MULTIPLE = _env_float("LIFE_COVER_MULTIPLE", 10.0)

# Retirement: target corpus = annual_expense / WITHDRAWAL_RATE_RETIREMENT.
# A safe-withdrawal / perpetuity rate; lower rate => larger required corpus.
WITHDRAWAL_RATE_RETIREMENT = _env_float("WITHDRAWAL_RATE_RETIREMENT", 0.05)

# Inflation used to grow future expenses and goal targets.
INFLATION_RATE = _env_float("INFLATION_RATE", 0.07)

# Annual step-up assumed for step-up SIP calculations.
STEP_UP_RATE = _env_float("STEP_UP_RATE", 0.10)

# Flat expected annual return used by the free-tier calculators.
DEFAULT_ANNUAL_RETURN = _env_float("DEFAULT_ANNUAL_RETURN", 0.10)

# Tiered expected returns used by the report engine, keyed by risk category.
# Kept here so the engine and any display code share one table.
ASSUMED_RETURNS = {
    "aggressive": {"annual": 0.14, "monthly": 0.011, "label": "14% p.a. (Aggressive Equity)"},
    "growth": {"annual": 0.114, "monthly": 0.009, "label": "11.4% p.a. (Growth/Balanced)"},
    "moderate": {"annual": 0.114, "monthly": 0.009, "label": "11.4% p.a. (Moderate)"},
    "conservative": {"annual": 0.074, "monthly": 0.006, "label": "7.4% p.a. (Conservative/Debt)"},
}

# Health insurance: recommended sum insured scales with the client's profile
# instead of a fixed "10-15L" for everyone. Base cover 10L (individual) /
# 15L (family floater when dependents exist), lifted to ~50% of annual income
# for higher earners, capped at HEALTH_COVER_MAX.
HEALTH_COVER_BASE_INDIVIDUAL = _env_float("HEALTH_COVER_BASE_INDIVIDUAL", 1000000.0)
HEALTH_COVER_BASE_FAMILY = _env_float("HEALTH_COVER_BASE_FAMILY", 1500000.0)
HEALTH_COVER_INCOME_FACTOR = _env_float("HEALTH_COVER_INCOME_FACTOR", 0.5)
HEALTH_COVER_MAX = _env_float("HEALTH_COVER_MAX", 10000000.0)

# Cover at or above OVER_INSURED_THRESHOLD x requirement is tagged
# "Over-insured" (informational — the client pays premium for cover beyond
# the benchmark, not a protection defect).
OVER_INSURED_THRESHOLD = _env_float("OVER_INSURED_THRESHOLD", 1.25)


def recommended_health_cover(annual_income, dependents_count=0) -> float:
    """Recommended health sum insured for this profile, rounded to the lakh."""
    try:
        income = float(annual_income or 0)
    except (TypeError, ValueError):
        income = 0.0
    try:
        deps = float(dependents_count or 0)
    except (TypeError, ValueError):
        deps = 0.0
    base = HEALTH_COVER_BASE_FAMILY if deps > 0 else HEALTH_COVER_BASE_INDIVIDUAL
    scaled = max(base, income * HEALTH_COVER_INCOME_FACTOR)
    lakh = 100000.0
    return min(round(scaled / lakh) * lakh, HEALTH_COVER_MAX)


def health_cover_status(current_cover, recommended_cover) -> str:
    """UI status for health cover: NOT COVERED / UPGRADE RECOMMENDED / ADEQUATE / OVER-INSURED."""
    try:
        current = float(current_cover or 0)
    except (TypeError, ValueError):
        current = 0.0
    try:
        rec = float(recommended_cover or 0)
    except (TypeError, ValueError):
        rec = 0.0
    if current <= 0:
        return "NOT COVERED"
    if rec > 0 and current >= rec * 2.0:
        return "OVER-INSURED"
    if current >= rec:
        return "ADEQUATE"
    return "UPGRADE RECOMMENDED"


def life_cover_status(current_cover, required_cover) -> str:
    """UI status for term cover: NOT COVERED / UNDERINSURED / ADEQUATE / OVER-INSURED."""
    try:
        current = float(current_cover or 0)
    except (TypeError, ValueError):
        current = 0.0
    try:
        required = float(required_cover or 0)
    except (TypeError, ValueError):
        required = 0.0
    if required <= 0:
        return "ADEQUATE"
    if current <= 0:
        return "NOT COVERED"
    if current >= required * OVER_INSURED_THRESHOLD:
        return "OVER-INSURED"
    if current >= required:
        return "ADEQUATE"
    return "UNDERINSURED"
