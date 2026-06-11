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

# Single source of truth lives in assumptions.py; re-exported here under the
# names this module already uses.
from assumptions import (
    INFLATION_RATE as DEFAULT_INFLATION_RATE,
    DEFAULT_ANNUAL_RETURN,
    STEP_UP_RATE as DEFAULT_STEP_UP_RATE,
    WITHDRAWAL_RATE_RETIREMENT as DEFAULT_WITHDRAWAL_RATE,
)


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
# Golden Number Calculator (4-Question Version)
# ──────────────────────────────────────────────────────────────────────────────

GOLDEN_NUMBER_WANTS_PCT = 0.38  # Wants should be ~38% of ideal income
GADGET_AMORTIZATION_MONTHS = 36  # 3-year replacement cycle

def compute_golden_number(
    annual_clothing: float,
    annual_travel: float,
    monthly_lifestyle: float,
    total_gadget_value: float,
    actual_income: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute the Golden Number (ideal monthly take-home income) from 4 spending inputs.

    The methodology: users tell us how much they SPEND on wants/lifestyle across
    4 categories. We convert everything to monthly equivalents, sum them up, and
    reverse-engineer the ideal income at which these wants would represent 38% of
    income.

    Formula:
        clothing_monthly  = annual_clothing / 12
        travel_monthly    = annual_travel / 12
        lifestyle_monthly = monthly_lifestyle  (already monthly)
        gadgets_monthly   = total_gadget_value / 36  (3-year amortization)
        total_monthly_wants = sum of above
        golden_number = round(total_monthly_wants / 0.38, -3)  # nearest ₹1,000

    If actual_income is provided (> 0), also computes:
        surplus_deficit = actual_income - total_monthly_wants
        status:
            actual >= golden_number        → "GOLDEN SPENDER"
            actual >= 80% golden_number    → "ON TRACK"
            actual <  80% golden_number    → "ADVENTURE SPENDER"

    Args:
        annual_clothing: Annual spend on clothing (jeans, T-shirts, shoes, blazer).
        annual_travel: Annual travel / vacation cost.
        monthly_lifestyle: Monthly lifestyle spend (entertainment, Netflix, dining, parties).
        total_gadget_value: Total current gadget value (phone + laptop + headphones).
        actual_income: Optional actual monthly take-home income for status check.

    Returns:
        Dict with golden_number, total_monthly_wants, category breakdown,
        and optional status/surplus info if actual_income provided.
    """
    # Monthly equivalents
    clothing_monthly = annual_clothing / 12.0 if annual_clothing > 0 else 0.0
    travel_monthly = annual_travel / 12.0 if annual_travel > 0 else 0.0
    lifestyle_monthly = max(0.0, monthly_lifestyle)
    gadgets_monthly = total_gadget_value / GADGET_AMORTIZATION_MONTHS if total_gadget_value > 0 else 0.0

    total_monthly_wants = clothing_monthly + travel_monthly + lifestyle_monthly + gadgets_monthly

    # Golden Number: ideal income where wants = 38%
    if total_monthly_wants > 0:
        raw_golden = total_monthly_wants / GOLDEN_NUMBER_WANTS_PCT
        golden_number = round(raw_golden / 1000) * 1000  # round to nearest ₹1,000
    else:
        golden_number = 0

    result = {
        "golden_number": golden_number,
        "total_monthly_wants": round(total_monthly_wants, 2),
        "breakdown": {
            "clothing_monthly": round(clothing_monthly, 2),
            "travel_monthly": round(travel_monthly, 2),
            "lifestyle_monthly": round(lifestyle_monthly, 2),
            "gadgets_monthly": round(gadgets_monthly, 2),
        },
        "methodology": {
            "wants_pct_assumed": round(GOLDEN_NUMBER_WANTS_PCT * 100, 1),
            "gadget_amortization_years": GADGET_AMORTIZATION_MONTHS // 12,
        },
    }

    # Optional: status classification if actual income provided
    if actual_income > 0 and golden_number > 0:
        surplus_deficit = actual_income - total_monthly_wants
        if actual_income >= golden_number:
            status = "GOLDEN SPENDER"
        elif actual_income >= golden_number * 0.8:
            status = "ON TRACK"
        else:
            status = "ADVENTURE SPENDER"

        result["actual_income"] = actual_income
        result["surplus_deficit"] = round(surplus_deficit, 2)
        result["status"] = status

    return result


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
    expected_pension: float = 0.0,
    inflation: float = DEFAULT_INFLATION_RATE,
    annual_return: float = DEFAULT_ANNUAL_RETURN,
    withdrawal_rate: float = DEFAULT_WITHDRAWAL_RATE,
) -> Dict[str, Any]:
    """
    Compute the retirement corpus gap analysis.

    Formulas:
        Net Monthly Expense = max(0, monthly_expense - expected_pension)
        Target Corpus = (Net Monthly Expense * (1 + inflation)^years * 12) / withdrawal_rate
        FV of Existing Corpus = existing_corpus * (1 + annual_return)^years
        FV of Ongoing SIP = ongoing_sip * [((1 + r_monthly)^(years*12) - 1) / r_monthly]
        Gap = max(0, Target Corpus - FV_existing - FV_sip)
        Required Step-Up SIP = step_up_sip(gap, years)

    Args:
        monthly_expense: Current monthly household expense.
        years_to_retire: Number of years until retirement.
        existing_corpus: Current investment corpus value.
        ongoing_sip: Current monthly SIP amount.
        expected_pension: Expected monthly pension (policy, govt, NPS).
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

    # 1. Target Corpus (Net of Pension)
    net_monthly_expense = max(0.0, monthly_expense - expected_pension)
    inflated_monthly = net_monthly_expense * ((1 + inflation) ** years_to_retire)
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
        "net_monthly_expense": round(net_monthly_expense, 2),
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
