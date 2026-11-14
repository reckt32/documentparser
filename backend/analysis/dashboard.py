"""Dashboard snapshot and action-item payloads derived from a report."""

from datetime import datetime

from backend.analysis.client_facts import _build_client_facts
from backend.analysis.health_score import compute_financial_health_score
from backend.analysis.llm_sections import compute_regime_comparison
from backend.assumptions import recommended_health_cover
from backend.utils import _safe_float


# --- Dashboard snapshot / action item builders ---

def _dimension_urgency(score):
    """Map a 0-100 dimension score to an urgency tag used on the dashboard."""
    try:
        s = float(score)
    except Exception:
        s = 0.0
    if s < 40:
        return "IMMEDIATE"
    if s < 75:
        return "HIGH"
    return "MAINTAIN"


def _build_dashboard_snapshot(analysis: dict, questionnaire: dict, doc_insights: dict, client_facts: dict = None, allocation_output: dict = None) -> dict:
    """
    Build the snapshot payload persisted as snapshot_json on a dashboard report.
    Captures detailed per-client scoring, actual values, and allocation data
    so the dashboard API can serve per-client data without re-running analysis.
    """
    analysis = analysis or {}
    ihs = analysis.get("ihs") or {}
    diagnostics = analysis.get("_diagnostics") or {}
    personal = (questionnaire or {}).get("personal_info") or {}

    # Use pre-built client_facts if provided, otherwise build fresh
    if client_facts is None:
        try:
            client_facts = _build_client_facts(questionnaire or {}, analysis, doc_insights or {})
        except Exception:
            client_facts = {"analysis": analysis}

    try:
        health_score = compute_financial_health_score(analysis, client_facts)
    except Exception:
        health_score = {}

    components = health_score.get("components") or {}
    dimensions = {
        key: {
            "score": comp.get("score"),
            "label": comp.get("label"),
            "priority": comp.get("priority"),
            "weight": comp.get("weight"),
            "weighted": comp.get("weighted"),
        }
        for key, comp in components.items()
    }

    # Extract actual financial values for the dashboard
    facts_income = client_facts.get("income") or {}
    facts_insurance = client_facts.get("insurance") or {}
    facts_personal = client_facts.get("personal") or {}
    term_insurance = client_facts.get("term_insurance") or {}
    alloc = allocation_output or {}

    # Per-client insurance details with GAP calculation
    life_cover_current = _safe_float(facts_insurance.get("lifeCover"), 0)
    life_cover_required = _safe_float(diagnostics.get("requiredLifeCover"), 0)
    life_cover_gap = max(0, life_cover_required - life_cover_current)
    health_cover_current = _safe_float(facts_insurance.get("healthCover"), 0)
    recommended_health = recommended_health_cover(facts_income.get("annualIncome"), facts_personal.get("dependents_count"))
    health_cover_gap = max(0, recommended_health - health_cover_current)

    # Emergency fund details in rupees
    monthly_expenses = _safe_float(facts_income.get("monthlyExpenses"), 0)
    emergency_fund_target = monthly_expenses * 6
    emergency_fund_current = _safe_float((client_facts.get("lifestyle") or {}).get("emergency_fund"), 0)
    emergency_fund_gap_inr = max(0, emergency_fund_target - emergency_fund_current)

    # Goal/SIP summary from allocation
    goal_sip_table = alloc.get("goal_sip_table") or []
    goal_summary = []
    for g in goal_sip_table:
        goal_summary.append({
            "name": g.get("name"),
            "allocated_sip": g.get("allocated_sip"),
            "ideal_sip": g.get("ideal_sip"),
            "shortfall": g.get("shortfall"),
            "target_amount": g.get("target_amount"),
            "horizon_years": g.get("horizon_years"),
            "fund_type": g.get("fund_type"),
            "risk_category": g.get("risk_category"),
        })

    return {
        "client_name": personal.get("name"),
        "client_age": personal.get("age"),
        "risk_profile": analysis.get("riskProfile"),
        "ihs": {
            "score": ihs.get("score"),
            "band": ihs.get("band"),
        },
        "overall_health": {
            "score": health_score.get("overall"),
            "label": health_score.get("overall_label"),
        },
        "headline_indicators": {
            "surplus_band": analysis.get("surplusBand"),
            "insurance_gap": analysis.get("insuranceGap"),
            "debt_stress": analysis.get("debtStress"),
            "liquidity": analysis.get("liquidity"),
        },
        "financials": {
            "annual_income": _safe_float(facts_income.get("annualIncome"), 0),
            "monthly_expenses": monthly_expenses,
            "monthly_emi": _safe_float(facts_income.get("monthlyEmi"), 0),
            "monthly_surplus": _safe_float(alloc.get("monthly_surplus"), 0),
        },
        "protection": {
            "life_cover_current": life_cover_current,
            "life_cover_required": life_cover_required,
            "life_cover_gap": life_cover_gap,
            "health_cover_current": health_cover_current,
            "health_cover_recommended": recommended_health,
            "health_cover_gap": health_cover_gap,
            "insurance_provision_monthly": _safe_float(alloc.get("insurance_provision") or alloc.get("insurance_sip_monthly"), 0),
        },
        "liquidity_detail": {
            "months_covered": diagnostics.get("liquidityMonths"),
            "emergency_fund_target_inr": emergency_fund_target,
            "emergency_fund_current_inr": emergency_fund_current,
            "emergency_fund_gap_inr": emergency_fund_gap_inr,
        },
        "allocation_summary": {
            "total_ideal_sip": _safe_float(alloc.get("total_ideal_sip_needed"), 0),
            "total_allocated_sip": _safe_float(alloc.get("total_allocated_sip"), 0),
            "goal_achievement_pct": _safe_float(alloc.get("goal_achievement_percent"), 0),
            "existing_sip_running": _safe_float(alloc.get("existing_sip_running"), 0),
            "total_investing": _safe_float(alloc.get("total_investing"), 0),
        },
        "goal_summary": goal_summary,
        "diagnostics": diagnostics,
        "dimensions": dimensions,
        "flags": analysis.get("flags") or [],
        "recommendations": analysis.get("recommendations") or [],
        "generated_at": datetime.utcnow().isoformat(),
    }


def _build_dashboard_action_items(analysis: dict, client_facts: dict = None, allocation_output: dict = None) -> list:
    """
    Translate the full analysis + client_facts + allocation_output into
    detailed, per-client action items. Each item captures the actual
    rupee values, gaps, and SIP amounts specific to this client.
    """
    analysis = analysis or {}
    diagnostics = analysis.get("_diagnostics") or {}
    facts = client_facts or {}
    alloc = allocation_output or {}
    items = []

    facts_insurance = facts.get("insurance") or {}
    facts_income = facts.get("income") or {}
    facts_personal = facts.get("personal") or {}
    term_ins = facts.get("term_insurance") or {}

    # ---- 1. PROTECTION: Life Insurance (GAP-based, not total cover) ----
    life_cover_current = _safe_float(facts_insurance.get("lifeCover"), 0)
    life_cover_required = _safe_float(diagnostics.get("requiredLifeCover"),
                                       _safe_float(term_ins.get("required_cover"), 0))
    life_cover_gap = max(0, life_cover_required - life_cover_current)
    has_dependents = facts_personal.get("has_financial_dependents", True)

    if life_cover_gap > 0 and has_dependents:
        # Estimate annual premium for the gap
        age = int(_safe_float(facts_personal.get("age"), 35))
        rate_per_lakh = 500 if age < 35 else (700 if age < 45 else 1200)
        est_premium_yearly = (life_cover_gap / 100000) * rate_per_lakh
        est_premium_monthly = est_premium_yearly / 12

        urgency = "IMMEDIATE" if life_cover_gap >= life_cover_required * 0.5 else "HIGH"
        items.append({
            "item_id": "protection_life_cover_gap",
            "dimension": "protection",
            "urgency": urgency,
            "value_type": "life_premium_monthly_inr",
            "value_num": est_premium_monthly,
            "title": "Increase life insurance cover",
            "description": (
                f"Current cover: Rs. {life_cover_current:,.0f}. Required: Rs. {life_cover_required:,.0f}. "
                f"Get additional Rs. {life_cover_gap:,.0f} term cover. "
                f"Estimated premium: ~Rs. {est_premium_monthly:,.0f}/month."
            ),
        })

    # ---- 2. PROTECTION: Health Insurance ----
    health_cover_current = _safe_float(facts_insurance.get("healthCover"), 0)
    recommended_health = recommended_health_cover(facts_income.get("annualIncome"), facts_personal.get("dependents_count"))
    health_cover_gap = max(0, recommended_health - health_cover_current)

    if health_cover_gap > 0:
        age = int(_safe_float(facts_personal.get("age"), 35))
        health_rate = 2500 if age < 35 else (3500 if age < 45 else 5000)
        health_premium_yearly = (health_cover_gap / 100000) * health_rate
        health_premium_monthly = health_premium_yearly / 12

        urgency = "IMMEDIATE" if health_cover_current < recommended_health * 0.5 else "HIGH"
        items.append({
            "item_id": "protection_health_cover_gap",
            "dimension": "protection",
            "urgency": urgency,
            "value_type": "health_premium_monthly_inr",
            "value_num": health_premium_monthly,
            "title": "Get adequate health insurance",
            "description": (
                f"Current health cover: Rs. {health_cover_current:,.0f}. "
                f"Recommended: Rs. {recommended_health:,.0f}. "
                f"Gap: Rs. {health_cover_gap:,.0f}. "
                f"Estimated premium: ~Rs. {health_premium_monthly:,.0f}/month."
            ),
        })

    # ---- 3. LIQUIDITY: Emergency Fund (in RUPEES, not months) ----
    liquidity = analysis.get("liquidity")
    liquidity_months = _safe_float(diagnostics.get("liquidityMonths"), 0)
    monthly_expenses = _safe_float(facts_income.get("monthlyExpenses"), 0)
    ef_target = monthly_expenses * 6
    ef_current = _safe_float((facts.get("lifestyle") or {}).get("emergency_fund"), 0)
    ef_gap_inr = max(0, ef_target - ef_current)

    if liquidity in ("Insufficient", "Low", "Critical") and ef_gap_inr > 0:
        items.append({
            "item_id": "liquidity_emergency_fund",
            "dimension": "liquidity",
            "urgency": "IMMEDIATE" if liquidity_months < 3 else "HIGH",
            "value_type": "emergency_fund_gap_inr",
            "value_num": ef_gap_inr,
            "title": "Build emergency fund",
            "description": (
                f"You have {liquidity_months:.1f} months of expenses covered. "
                f"Target: 6 months (Rs. {ef_target:,.0f}). "
                f"Gap: Rs. {ef_gap_inr:,.0f}. "
                f"Park in liquid fund or high-yield savings."
            ),
        })

    # ---- 4. DEBT MANAGEMENT ----
    debt_stress = analysis.get("debtStress")
    emi_pct = _safe_float(diagnostics.get("emiPct"), 0)
    monthly_emi = _safe_float(facts_income.get("monthlyEmi"), 0)

    if debt_stress in ("Stressed", "High"):
        items.append({
            "item_id": "debt_stress_high",
            "dimension": "debt_management",
            "urgency": "IMMEDIATE",
            "value_type": "monthly_emi_inr",
            "value_num": monthly_emi,
            "title": "Reduce debt burden",
            "description": (
                f"EMI-to-income ratio: {emi_pct:.1f}% (above safe limit of 40%). "
                f"Monthly EMI: Rs. {monthly_emi:,.0f}. "
                f"Refinance or accelerate repayment of high-cost debt."
            ),
        })
    elif debt_stress == "Moderate":
        items.append({
            "item_id": "debt_stress_moderate",
            "dimension": "debt_management",
            "urgency": "HIGH",
            "value_type": "monthly_emi_inr",
            "value_num": monthly_emi,
            "title": "Prioritise unsecured debt repayment",
            "description": (
                f"EMI-to-income ratio: {emi_pct:.1f}%. "
                f"Monthly EMI: Rs. {monthly_emi:,.0f}. "
                f"Start reducing unsecured loans before they compound."
            ),
        })

    # ---- 5. SAVINGS RATE ----
    surplus_band = analysis.get("surplusBand")
    monthly_surplus = _safe_float(alloc.get("monthly_surplus"), 0)
    if surplus_band == "Low":
        items.append({
            "item_id": "savings_low_surplus",
            "dimension": "savings",
            "urgency": "HIGH",
            "value_type": "monthly_surplus_inr",
            "value_num": monthly_surplus,
            "title": "Lift your savings rate",
            "description": (
                f"Current monthly surplus: Rs. {monthly_surplus:,.0f}. "
                f"Cut discretionary spending and automate an SIP to push savings above 20% of income."
            ),
        })

    # ---- 6. GOAL-BASED SIP OPPORTUNITIES ----
    goal_sip_table = alloc.get("goal_sip_table") or []
    for idx, goal in enumerate(goal_sip_table):
        goal_name = goal.get("name") or "Goal"
        is_emergency = goal.get("is_emergency", False)
        if is_emergency:
            continue  # Already handled above in liquidity

        allocated_sip = _safe_float(goal.get("allocated_sip"), 0)
        ideal_sip = _safe_float(goal.get("ideal_sip"), 0)
        shortfall = _safe_float(goal.get("shortfall"), 0)
        target_amount = _safe_float(goal.get("target_amount"), 0)
        horizon = goal.get("horizon_years")
        fund_type = goal.get("fund_type") or "Diversified Funds"
        risk_cat = goal.get("risk_category") or "Moderate"

        # Create item_id from goal name (sanitized)
        safe_name = (goal_name or "").lower().replace(" ", "_")[:30]
        item_id = f"goal_sip_{safe_name}_{idx}"

        if shortfall > 0:
            urgency = "IMMEDIATE" if (horizon and float(horizon) <= 3) else "HIGH"
            items.append({
                "item_id": item_id,
                "dimension": "goal_sip",
                "urgency": urgency,
                "value_type": "sip_amount_inr",
                "value_num": shortfall,
                "title": f"SIP for {goal_name}",
                "description": (
                    f"Target: Rs. {target_amount:,.0f}" +
                    (f" in {horizon} years" if horizon else "") +
                    f". Ideal SIP: Rs. {ideal_sip:,.0f}/month. "
                    f"Currently allocated: Rs. {allocated_sip:,.0f}/month. "
                    f"Shortfall: Rs. {shortfall:,.0f}/month. "
                    f"Recommended: {fund_type} ({risk_cat} risk)."
                ),
            })
        else:
            items.append({
                "item_id": item_id,
                "dimension": "goal_sip",
                "urgency": "MAINTAIN",
                "value_type": "sip_amount_inr",
                "value_num": allocated_sip,
                "title": f"SIP for {goal_name}",
                "description": (
                    f"Target: Rs. {target_amount:,.0f}" +
                    (f" in {horizon} years" if horizon else "") +
                    f". Allocated SIP: Rs. {allocated_sip:,.0f}/month ({fund_type}). "
                    f"On track."
                ),
            })

    # ---- 7. PORTFOLIO HEALTH ----
    ihs = analysis.get("ihs") or {}
    ihs_band = ihs.get("band")
    ihs_score = ihs.get("score")
    if ihs_band in ("Poor", "Average"):
        items.append({
            "item_id": "portfolio_quality_low",
            "dimension": "portfolio_health",
            "urgency": _dimension_urgency(ihs_score),
            "value_type": "ihs_score",
            "value_num": float(ihs_score) if isinstance(ihs_score, (int, float)) else None,
            "title": "Improve portfolio quality",
            "description": (
                f"Portfolio health score: {ihs_score}/100 ({ihs_band}). "
                f"Diversify allocation and remove unsuitable products."
            ),
        })

    # ---- 7b. PORTFOLIO REBALANCING ----
    portfolio = facts.get("portfolio") or {}
    total_portfolio_value = _safe_float(portfolio.get("total_value") or portfolio.get("current_value"), 0)
    current_equity = _safe_float(portfolio.get("equity"), 0)
    advanced_risk = analysis.get("advancedRisk") or {}
    rec_band = advanced_risk.get("recommendedEquityBand") or {}
    rec_min = _safe_float(rec_band.get("min"), 40)
    rec_max = _safe_float(rec_band.get("max"), 60)
    rec_mid = _safe_float(advanced_risk.get("recommendedEquityMid"), 50)

    if current_equity > 0:
        if current_equity > rec_max:
            shift_pct = current_equity - rec_mid
            shift_amount = (shift_pct / 100.0) * total_portfolio_value
            items.append({
                "item_id": "portfolio_rebalance_reduce_equity",
                "dimension": "portfolio_health",
                "urgency": "HIGH",
                "value_type": "rebalance_shift_inr",
                "value_num": shift_amount,
                "title": "Rebalance portfolio: Reduce equity",
                "description": (
                    f"Current equity allocation: {current_equity:.1f}% is above the recommended range of {rec_min:.0f}%-{rec_max:.0f}%. "
                    f"Consider shifting ~{shift_pct:.1f}% of your portfolio (approx. Rs. {shift_amount:,.0f}) from equity to debt."
                )
            })
        elif current_equity < rec_min:
            shift_pct = rec_mid - current_equity
            shift_amount = (shift_pct / 100.0) * total_portfolio_value
            items.append({
                "item_id": "portfolio_rebalance_increase_equity",
                "dimension": "portfolio_health",
                "urgency": "HIGH",
                "value_type": "rebalance_shift_inr",
                "value_num": shift_amount,
                "title": "Rebalance portfolio: Increase equity",
                "description": (
                    f"Current equity allocation: {current_equity:.1f}% is below the recommended range of {rec_min:.0f}%-{rec_max:.0f}%. "
                    f"Consider shifting ~{shift_pct:.1f}% of your portfolio (approx. Rs. {shift_amount:,.0f}) from debt to equity."
                )
            })

    # ---- 8. TAX OPTIMISATION ----
    itr = facts.get("itr") or {}
    deductions = itr.get("deductions_claimed") or []
    total_deductions = sum(_safe_float(d.get("amount"), 0) for d in deductions)
    if total_deductions < 150000 and _safe_float(facts_income.get("annualIncome"), 0) > 500000:
        unused = 150000 - total_deductions
        items.append({
            "item_id": "tax_deduction_gap",
            "dimension": "tax_efficiency",
            "urgency": "HIGH" if unused > 100000 else "MAINTAIN",
            "value_type": "tax_saving_potential_inr",
            "value_num": unused * 0.3,  # Approximate tax saving at 30% bracket
            "title": "Maximise tax deductions",
            "description": (
                f"Currently claiming Rs. {total_deductions:,.0f} in deductions. "
                f"Unutilised 80C limit: Rs. {unused:,.0f}. "
                f"Potential tax saving: ~Rs. {unused * 0.3:,.0f}/year via ELSS, PPF, or NPS."
            ),
        })

    # ---- 8b. TAX REGIME SWITCH ----
    gross = _safe_float(facts_income.get("annualIncome"), 0)
    if gross > 0:
        tax_info = facts.get("tax") or {}
        regime = tax_info.get("tax_regime", "old").lower()
        d_80c = _safe_float(tax_info.get("deductions_80c"), 0)
        d_80d = _safe_float(tax_info.get("deductions_80d"), 0)
        d_24c = _safe_float(tax_info.get("deductions_24c"), 0)
        d_80dd = _safe_float(tax_info.get("deductions_80dd"), 0)
        
        comp = compute_regime_comparison(
            gross_income=gross,
            deductions_80c=d_80c,
            deductions_80d=d_80d,
            other_deductions=d_24c + d_80dd
        )
        better_regime = comp.get("better_regime")
        savings = _safe_float(comp.get("savings"), 0)
        if savings > 0 and better_regime != regime:
            items.append({
                "item_id": "tax_regime_switch",
                "dimension": "tax_efficiency",
                "urgency": "HIGH",
                "value_type": "tax_saving_potential_inr",
                "value_num": savings,
                "title": f"Switch to {better_regime.upper()} tax regime",
                "description": (
                    f"Your current regime is {regime.upper()}. By switching to the {better_regime.upper()} regime, "
                    f"you could save approx. Rs. {savings:,.0f}/year in taxes based on your current income and deductions."
                )
            })

    # ---- 8c. TAX FILING COMPLIANCE ----
    if gross > 250000 and not facts.get("itr"):
        items.append({
            "item_id": "tax_filing_compliance",
            "dimension": "tax_efficiency",
            "urgency": "HIGH",
            "value_type": "tax_filing_status",
            "value_num": None,
            "title": "File ITR & upload tax return",
            "description": "Upload your latest ITR to verify tax compliance, claim refunds, and identify further tax optimization opportunities."
        })

    # ---- 8d. ESTATE PLANNING: Will & Nominations ----
    estate = facts.get("estate") or {}
    will_status = str(estate.get("will_status") or "").strip().lower()
    nominees = estate.get("nominees") or []
    
    if will_status in ("no", ""):
        items.append({
            "item_id": "estate_will_planning",
            "dimension": "estate_planning",
            "urgency": "HIGH",
            "value_type": "will_planning_status",
            "value_num": None,
            "title": "Create a Will",
            "description": "You do not have a registered Will. Create a Will to ensure hassle-free distribution of your estate and assets to your loved ones."
        })
    
    if len(nominees) == 0:
        items.append({
            "item_id": "estate_nominee_registration",
            "dimension": "estate_planning",
            "urgency": "HIGH",
            "value_type": "nominee_status",
            "value_num": None,
            "title": "Register Nominees",
            "description": "No nominee details are added to your estate profile. Register nominees across all bank accounts, mutual funds, and insurance policies."
        })

    # ---- 9. RETIREMENT PLANNING (if applicable) ----
    retirement = facts.get("retirement_planning")
    if retirement and retirement.get("enabled"):
        corpus_required = _safe_float(retirement.get("corpus_required"), 0)
        monthly_sip_needed = _safe_float(retirement.get("monthly_sip_needed"), 0)
        if corpus_required > 0 and monthly_sip_needed > 0:
            items.append({
                "item_id": "retirement_planning_sip",
                "dimension": "goal_sip",
                "urgency": "HIGH",
                "value_type": "sip_amount_inr",
                "value_num": monthly_sip_needed,
                "title": "Retirement corpus SIP",
                "description": (
                    f"Retirement corpus needed: Rs. {corpus_required:,.0f}. "
                    f"Required SIP: Rs. {monthly_sip_needed:,.0f}/month. "
                    f"Start early to benefit from compounding."
                ),
            })

    return items
