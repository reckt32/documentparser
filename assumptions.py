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
