"""Flags, recommendations and the top-level financial health analysis."""

from backend.analysis.metrics import (
    compute_debt_stress,
    compute_ihs,
    compute_insurance_gap,
    compute_liquidity,
    compute_surplus_band,
    effective_savings_percent,
)
from backend.analysis.risk import compute_advanced_risk, compute_risk_profile
from backend.assumptions import LIFE_COVER_MULTIPLE
from backend.utils import _format_indian_amount, _safe_float


def _generate_unified_equity_recommendation(results, inputs):
    """
    Generate unified equity recommendation by comparing current allocation
    vs recommended band from advanced risk assessment.
    Prevents contradictory advice.
    """
    allocation = (inputs.get("investments") or {}).get("allocation") or {}
    current_equity = _safe_float(allocation.get("equity"), 0.0)

    advanced_risk = results.get("advancedRisk") or {}
    rec_band = advanced_risk.get("recommendedEquityBand") or {}
    rec_min = rec_band.get("min")
    rec_max = rec_band.get("max")
    rec_mid = advanced_risk.get("recommendedEquityMid")

    ihs_band = results.get("ihs", {}).get("band")

    # Compare current vs recommended
    if rec_min is not None and rec_max is not None:
        if current_equity < rec_min - 5:
            # Below recommended range
            gap = rec_min - current_equity
            if gap > 20:
                return f"Increase equity from {round(current_equity, 1)}% to at least {rec_min}% (target: {rec_mid}%). Consider gradual rebalancing via SIP."
            else:
                return f"Increase equity from {round(current_equity, 1)}% towards {rec_min}-{rec_max}% range (target: {rec_mid}%)."

        elif current_equity > rec_max + 5:
            # Above recommended range
            gap = current_equity - rec_max
            if gap > 20:
                return f"Reduce equity from {round(current_equity, 1)}% to {rec_max}% or below. Consider booking profits and rebalancing to debt."
            else:
                return f"Consider rebalancing: current equity {round(current_equity, 1)}% exceeds recommended {rec_min}-{rec_max}%."

        else:
            # Within recommended range
            if ihs_band == "Excellent":
                return f"Maintain allocation (equity: {round(current_equity, 1)}%). Rebalance annually and add international exposure."
            elif ihs_band == "Good":
                return f"Equity allocation ({round(current_equity, 1)}%) is well-aligned. Consider international exposure and tax optimization."
            else:
                return f"Equity allocation ({round(current_equity, 1)}%) is within recommended range ({rec_min}-{rec_max}%). Focus on fund quality and diversification."

    # Fallback if no advanced risk data
    rp = results.get("riskProfile")
    if rp == "Conservative":
        return "Focus on debt-heavy allocation (25-40% equity) with low-volatility funds."
    elif rp == "Balanced":
        return "Aim for balanced 40-55% equity allocation with quality debt funds."
    elif rp == "Aggressive":
        return "Target 70-85% equity allocation; add gold (5-10%) as hedge."

    return "Review asset allocation with advisor to align with risk profile and goals."


def generate_flags_and_recommendations(results, inputs):
    flags = []
    recs = []

    # Convenience
    annual_income = _safe_float(inputs.get("income", {}).get("annualIncome"), 0.0)
    monthly_expenses = _safe_float(inputs.get("income", {}).get("monthlyExpenses"), 0.0)
    monthly_emi = _safe_float(inputs.get("income", {}).get("monthlyEmi"), 0.0)
    life_cover = _safe_float(inputs.get("insurance", {}).get("lifeCover"), 0.0)
    savings_percent = effective_savings_percent(inputs)
    allocation = (inputs.get("investments") or {}).get("allocation") or {}
    equity = _safe_float(allocation.get("equity"), 0.0)
    debt = _safe_float(allocation.get("debt"), 0.0)
    gold = _safe_float(allocation.get("gold"), 0.0)
    real_estate = _safe_float(allocation.get("realEstate"), 0.0)

    # Flags
    if results.get("surplusBand") == "Low":
        flags.append(f"Low Surplus: Saving {round(savings_percent,1)}% vs Benchmark 20%+")
    if results.get("insuranceGap") == "Underinsured":
        required = LIFE_COVER_MULTIPLE * annual_income
        flags.append(f"Underinsured: Cover Rs. {_format_indian_amount(life_cover)} vs Required Rs. {_format_indian_amount(required)}")
    if results.get("debtStress") == "Stressed":
        # Recompute ratio for message
        monthly_income = annual_income / 12.0 if annual_income > 0 else 0.0
        ratio_pct = (monthly_emi / monthly_income * 100.0) if monthly_income > 0 else 0.0
        flags.append(f"Debt Stress: EMI {round(ratio_pct,1)}% of Income vs Benchmark <=40%")
    if results.get("liquidity") == "Insufficient":
        # Recompute months for message
        ef = _safe_float(inputs.get("emergencyFundAmount"), 0.0)
        months = (ef / monthly_expenses) if monthly_expenses > 0 else 0.0
        flags.append(f"Insufficient Liquidity: Emergency fund {round(months,1)} months vs Benchmark 6+")
    ihs = results.get("ihs") or {}
    if ihs.get("band") in ["Poor", "Average", "Good"]:
        skew = ""
        if (debt + gold) > 60.0 and equity < 20.0:
            skew = "skewed allocation to FD/Gold"
        elif equity > 80.0:
            skew = "high equity concentration"
        elif real_estate > 50.0:
            skew = "concentration in Real Estate"
        extra = f", {skew}" if skew else ""
        flags.append(f"Score = {ihs.get('score')} -> {ihs.get('band')}: Saving {round(savings_percent,1)}% income{extra}")

    # Recommendations
    # 1. Unified Equity Recommendation (replaces old risk profile + IHS equity recommendations)
    equity_rec = _generate_unified_equity_recommendation(results, inputs)
    recs.append(equity_rec)

    # 2. Surplus recommendations
    sb = results.get("surplusBand")
    if sb == "Low":
        recs.append("Cut discretionary spending; automate an SIP.")
    elif sb == "Adequate":
        recs.append("Maintain discipline and allocate savings across your goals.")
    elif sb == "Strong":
        recs.append("Accelerate retirement savings and explore growth instruments.")

    # 3. Insurance recommendations
    ig = results.get("insuranceGap")
    if ig == "Underinsured":
        recs.append("Buy term life insurance for 10-12x your income and add/increase health cover.")
    else:
        recs.append("Review your cover every 2-3 years.")

    # 4. Debt recommendations
    ds = results.get("debtStress")
    if ds == "Stressed":
        recs.append("Refinance high-cost debt; repay high-cost debt first; avoid new borrowing.")
    elif ds == "Moderate":
        recs.append("Prioritize repaying unsecured debt.")
    else:
        recs.append("You can leverage strategically if needed.")

    # 5. Liquidity recommendations
    liq = results.get("liquidity")
    if liq == "Insufficient":
        recs.append("Build a liquid buffer (cash/liquid funds) until 6 months of expenses are covered.")
    else:
        recs.append("Invest your incremental surplus into growth assets.")

    # 6. IHS portfolio quality recommendations (non-equity advice only)
    ihs_band = ihs.get("band")
    if ihs_band == "Poor":
        recs.append("Increase savings rate, exit unsuitable products, and improve portfolio diversification.")
    elif ihs_band == "Average":
        # Check if equity is already high before suggesting to increase it
        if equity > 60.0:
            recs.append("Your portfolio is equity-heavy; consider rebalancing towards debt/hybrid to reduce risk while aligning with your goals.")
        else:
            recs.append("Increase your equity allocation and align investments with your goals.")
    elif ihs_band == "Good":
        recs.append("Conduct an annual review, add international exposure, and optimize for tax.")
    elif ihs_band == "Excellent":
        recs.append("Maintain your strategy, rebalance annually, and explore advanced strategies.")

    return flags, recs


def analyze_financial_health(payload: dict):
    payload = payload or {}
    personal = payload.get("personal") or {}
    income = payload.get("income") or {}
    goals = payload.get("goals") or {}
    risk = payload.get("risk") or {}
    insurance = payload.get("insurance") or {}
    savings = payload.get("savings") or {}
    investments = payload.get("investments") or {}
    allocation = investments.get("allocation") or {}
    current_products = investments.get("current") or []
    emergency_fund_amount = payload.get("emergencyFundAmount", 0)

    # Resolve inputs
    age = personal.get("age")
    tolerance = risk.get("tolerance")
    horizon = goals.get("goalHorizon")
    savings_percent = effective_savings_percent(payload)

    basic_risk_profile, basic_risk_score = compute_risk_profile(age, tolerance, horizon)
    surplus_band = compute_surplus_band(savings_percent)
    try:
        advanced_risk = compute_advanced_risk(payload, age)
    except Exception:
        advanced_risk = None
    
    has_financial_dependents = personal.get("has_financial_dependents", True)
    if not has_financial_dependents:
        insurance_gap = "Adequate"
        required_cover = 0.0
    else:
        insurance_gap, required_cover = compute_insurance_gap(income.get("annualIncome"), insurance.get("lifeCover"))
    debt_stress, emi_ratio_pct = compute_debt_stress(income.get("monthlyEmi"), income.get("annualIncome"))
    liquidity, liquidity_months = compute_liquidity(income.get("monthlyExpenses"), emergency_fund_amount)
    ihs = compute_ihs(savings_percent, current_products, allocation)

    results = {
        "riskProfile": (advanced_risk["finalCategory"] if advanced_risk else basic_risk_profile),
        "advancedRisk": advanced_risk,
        "surplusBand": surplus_band,
        "insuranceGap": insurance_gap,
        "debtStress": debt_stress,
        "liquidity": liquidity,
        "ihs": ihs,
        # extra diagnostics
        "_diagnostics": {
            "riskScore": basic_risk_score,
            "emiPct": round(emi_ratio_pct, 2),
            "liquidityMonths": round(liquidity_months, 2),
            "requiredLifeCover": required_cover
        }
    }

    flags, recs = generate_flags_and_recommendations(results, payload)
    results["flags"] = flags
    results["recommendations"] = recs
    return results
