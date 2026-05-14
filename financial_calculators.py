"""
Financial Calculator Utilities for the Meerkat Free-Tier Planning Layer.

Pure-math functions for:
- Spend Right (Golden Number + Status Badge)
- Retirement Gap Analysis (target corpus, FV, gap, step-up SIP)
- Step-Up SIP computation using geometric series

All functions are stateless and require no LLM or database access.
"""

import math
from typing import Dict, Any, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_INFLATION_RATE = 0.07       # 7% p.a.
DEFAULT_ANNUAL_RETURN = 0.10        # 10% p.a.
DEFAULT_STEP_UP_RATE = 0.10         # 10% annual SIP increase
DEFAULT_WITHDRAWAL_RATE = 0.05      # 5% safe withdrawal rate


# ──────────────────────────────────────────────────────────────────────────────
# Spend Right Calculator
# ──────────────────────────────────────────────────────────────────────────────

def compute_spend_right(
    income: float,
    rent: float,
    basic_spends: float,
    comfort_spends: float,
) -> Dict[str, Any]:
    """
    Compute the Spend Right analysis — Golden Number, surplus, and status badge.

    Args:
        income: Monthly take-home income (post-tax).
        rent: Monthly rent / housing EMI.
        basic_spends: Monthly essential spends (groceries, utilities, transport).
        comfort_spends: Monthly discretionary spends (dining, shopping, entertainment).

    Returns:
        Dict with:
            golden_number: Total discretionary ("wants") spending.
            total_needs: rent + basic_spends.
            surplus: income - total_needs - golden_number.
            surplus_pct: surplus as % of income.
            status_badge: "Saver" | "Balanced" | "Spender".
            needs_pct: total_needs as % of income.
            wants_pct: golden_number as % of income.
    """
    if income <= 0:
        return {
            "golden_number": 0,
            "total_needs": 0,
            "surplus": 0,
            "surplus_pct": 0,
            "status_badge": "Spender",
            "needs_pct": 0,
            "wants_pct": 0,
            "error": "Income must be greater than zero.",
        }

    golden_number = comfort_spends
    total_needs = rent + basic_spends
    surplus = income - total_needs - golden_number
    surplus_pct = (surplus / income) * 100.0

    needs_pct = (total_needs / income) * 100.0
    wants_pct = (golden_number / income) * 100.0

    # Status badge thresholds
    if surplus_pct > 30:
        status_badge = "Saver"
    elif surplus_pct >= 10:
        status_badge = "Balanced"
    else:
        status_badge = "Spender"

    return {
        "golden_number": round(golden_number, 2),
        "total_needs": round(total_needs, 2),
        "surplus": round(surplus, 2),
        "surplus_pct": round(surplus_pct, 2),
        "status_badge": status_badge,
        "needs_pct": round(needs_pct, 2),
        "wants_pct": round(wants_pct, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Step-Up SIP Formula
# ──────────────────────────────────────────────────────────────────────────────

def compute_step_up_sip(
    target: float,
    years: float,
    annual_return: float = DEFAULT_ANNUAL_RETURN,
    step_up_rate: float = DEFAULT_STEP_UP_RATE,
) -> Optional[float]:
    """
    Calculate the starting monthly SIP required to reach a target corpus,
    assuming an annual step-up in the SIP amount.

    Each year, the monthly SIP increases by `step_up_rate` (e.g. 10%).
    Year 0 SIP = P, Year 1 SIP = P*(1+step_up), Year 2 = P*(1+step_up)^2, ...

    Uses the geometric series multiplier:
        FV_factor = Σ_{k=0}^{N-1} (1+step_up)^k * annuity_factor_12(r_monthly)
        where annuity_factor_12(r) = ((1+r)^12 - 1) / r  [FV of 12 monthly payments]
        Each year's contribution is then compounded for the remaining years.

    Args:
        target: Target corpus amount.
        years: Number of years to invest.
        annual_return: Expected annual return rate (decimal, e.g. 0.10 for 10%).
        step_up_rate: Annual SIP step-up rate (decimal, e.g. 0.10 for 10%).

    Returns:
        Starting monthly SIP amount, or None if inputs are invalid.
    """
    try:
        if target <= 0 or years <= 0:
            return None

        n = int(round(years))
        if n <= 0:
            return None

        r_monthly = (1 + annual_return) ** (1 / 12) - 1  # effective monthly rate
        if r_monthly <= 0:
            return None

        # FV of 12 level monthly payments at rate r_monthly
        annuity_12 = ((1 + r_monthly) ** 12 - 1) / r_monthly

        # Accumulate: for year k (0-indexed), SIP = P * (1+step_up)^k
        # Contributions in year k compound for (n - k - 1) remaining whole years
        # plus the annuity within year k itself.
        fv_factor = 0.0
        for k in range(n):
            step_multiplier = (1 + step_up_rate) ** k
            remaining_years = n - k - 1
            compounding = (1 + annual_return) ** remaining_years
            fv_factor += step_multiplier * annuity_12 * compounding

        if fv_factor <= 0:
            return None

        starting_sip = target / fv_factor
        return round(starting_sip, 2)

    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Retirement Gap Analysis
# ──────────────────────────────────────────────────────────────────────────────

def compute_retirement_gap(
    monthly_expense: float,
    years_to_retire: float,
    existing_corpus: float = 0.0,
    ongoing_sip: float = 0.0,
    inflation: float = DEFAULT_INFLATION_RATE,
    annual_return: float = DEFAULT_ANNUAL_RETURN,
    withdrawal_rate: float = DEFAULT_WITHDRAWAL_RATE,
) -> Dict[str, Any]:
    """
    Compute the retirement corpus gap analysis.

    Formulas:
        Target Corpus = (monthly_expense * (1 + inflation)^years * 12) / withdrawal_rate
        FV of Existing Corpus = existing_corpus * (1 + annual_return)^years
        FV of Ongoing SIP = ongoing_sip * [((1 + r_monthly)^(years*12) - 1) / r_monthly]
        Gap = max(0, Target Corpus - FV_existing - FV_sip)
        Required Step-Up SIP = step_up_sip(gap, years)

    Args:
        monthly_expense: Current monthly household expense.
        years_to_retire: Number of years until retirement.
        existing_corpus: Current investment corpus value.
        ongoing_sip: Current monthly SIP amount.
        inflation: Annual inflation rate (decimal).
        annual_return: Expected annual return rate (decimal).
        withdrawal_rate: Safe annual withdrawal rate in retirement (decimal).

    Returns:
        Dict with target_corpus, fv_existing_corpus, fv_ongoing_sip, gap,
        required_step_up_sip, and all assumptions used.
    """
    if monthly_expense <= 0 or years_to_retire <= 0:
        return {
            "target_corpus": 0,
            "fv_existing_corpus": 0,
            "fv_ongoing_sip": 0,
            "gap": 0,
            "required_step_up_sip": 0,
            "error": "Monthly expense and years to retire must be positive.",
        }

    # 1. Target Corpus
    inflated_monthly = monthly_expense * ((1 + inflation) ** years_to_retire)
    target_corpus = (inflated_monthly * 12) / withdrawal_rate

    # 2. FV of Existing Corpus (lump-sum compounding)
    fv_existing = existing_corpus * ((1 + annual_return) ** years_to_retire)

    # 3. FV of Ongoing SIP (level SIP, monthly compounding)
    r_monthly = (1 + annual_return) ** (1 / 12) - 1
    n_months = int(round(years_to_retire * 12))
    if r_monthly > 0 and n_months > 0:
        fv_sip = ongoing_sip * (((1 + r_monthly) ** n_months - 1) / r_monthly)
    else:
        fv_sip = ongoing_sip * n_months  # fallback: no return

    # 4. Gap
    gap = max(0, target_corpus - fv_existing - fv_sip)

    # 5. Required Step-Up SIP to fill the gap
    if gap > 0:
        required_sip = compute_step_up_sip(
            target=gap,
            years=years_to_retire,
            annual_return=annual_return,
            step_up_rate=DEFAULT_STEP_UP_RATE,
        )
    else:
        required_sip = 0.0

    return {
        "target_corpus": round(target_corpus, 2),
        "fv_existing_corpus": round(fv_existing, 2),
        "fv_ongoing_sip": round(fv_sip, 2),
        "gap": round(gap, 2),
        "required_step_up_sip": round(required_sip, 2) if required_sip else 0,
        "assumptions": {
            "inflation_pct": round(inflation * 100, 1),
            "return_pct": round(annual_return * 100, 1),
            "step_up_pct": round(DEFAULT_STEP_UP_RATE * 100, 1),
            "withdrawal_rate_pct": round(withdrawal_rate * 100, 1),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Indian Number Formatting (server-side for API responses)
# ──────────────────────────────────────────────────────────────────────────────

def format_indian_compact(amount: float) -> str:
    """Format amount in Indian Lakhs/Crores notation for display."""
    if amount is None or amount == 0:
        return "₹0"
    is_negative = amount < 0
    amount = abs(amount)
    prefix = "-" if is_negative else ""
    if amount >= 1_00_00_000:
        return f"{prefix}₹{amount / 1_00_00_000:.2f} Cr"
    elif amount >= 1_00_000:
        return f"{prefix}₹{amount / 1_00_000:.2f} L"
    elif amount >= 1000:
        return f"{prefix}₹{amount / 1000:.1f} K"
    else:
        return f"{prefix}₹{int(round(amount))}"
