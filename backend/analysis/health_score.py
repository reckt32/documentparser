"""Composite financial health score."""

from backend.utils import _safe_float


def compute_financial_health_score(analysis: dict, facts: dict) -> dict:
    """
    Compute overall financial health score (0-100) and component scores.
    
    Score = (Protection × 0.30) + (Liquidity × 0.20) + (PortfolioHealth × 0.20)
            + (DebtManagement × 0.15) + (GoalReadiness × 0.10) + (TaxEfficiency × 0.05)
    
    Returns:
        {
            "overall": 56,
            "overall_label": "Needs Attention",
            "components": {
                "protection": {"score": 0, "weight": 30, "weighted": 0, "label": "No Coverage"},
                ...
            },
            "priority_areas": [("protection", "URGENT"), ("liquidity", "HIGH"), ...]
        }
    """
    components = {}
    
    # 1. Protection Score (30% weight) - based on insurance gap
    insurance_gap = analysis.get("insuranceGap", "Unknown")
    if insurance_gap == "Adequate":
        protection_score = 100
        protection_label = "Adequate"
    elif insurance_gap == "Underinsured":
        protection_score = 50
        protection_label = "Underinsured"
    elif insurance_gap == "Critical":
        protection_score = 20
        protection_label = "Critical Gap"
    else:
        protection_score = 0
        protection_label = "No Coverage"
    
    components["protection"] = {
        "score": protection_score,
        "weight": 30,
        "weighted": protection_score * 0.30,
        "label": protection_label,
        "priority": "URGENT" if protection_score < 50 else ("MEDIUM" if protection_score < 80 else "GOOD")
    }
    
    # 2. Liquidity Score (20% weight) - based on liquidity assessment
    # compute_liquidity emits "Adequate" / "Insufficient"; grade the gap by
    # months covered (from diagnostics) so we keep a Critical vs Low distinction.
    liquidity = analysis.get("liquidity", "Unknown")
    liquidity_months = _safe_float((analysis.get("_diagnostics") or {}).get("liquidityMonths"), None)
    if liquidity == "Adequate":
        liquidity_score = 100
        liquidity_label = "Adequate"
    elif liquidity == "Insufficient":
        if liquidity_months is not None and liquidity_months < 3:
            liquidity_score = 20
            liquidity_label = "Critical"
        else:
            liquidity_score = 50
            liquidity_label = "Low"
    else:
        liquidity_score = 50  # Unknown defaults to middle
        liquidity_label = "Unknown"
    
    components["liquidity"] = {
        "score": liquidity_score,
        "weight": 20,
        "weighted": liquidity_score * 0.20,
        "label": liquidity_label,
        "priority": "HIGH" if liquidity_score < 50 else ("MEDIUM" if liquidity_score < 80 else "GOOD")
    }
    
    # 3. Portfolio Health Score (20% weight) - based on risk alignment
    advanced_risk = analysis.get("advancedRisk") or {}
    portfolio = facts.get("portfolio") or {}
    current_equity = _safe_float(portfolio.get("equity"), 50)
    rec_band = advanced_risk.get("recommendedEquityBand") or {}
    rec_min = rec_band.get("min", 40)
    rec_max = rec_band.get("max", 60)
    
    # Score based on how close to recommended band
    if rec_min <= current_equity <= rec_max:
        portfolio_score = 100
        portfolio_label = "Well Balanced"
    elif abs(current_equity - (rec_min + rec_max) / 2) <= 15:
        portfolio_score = 70
        portfolio_label = "Slightly Off"
    elif abs(current_equity - (rec_min + rec_max) / 2) <= 30:
        portfolio_score = 50
        portfolio_label = "Needs Rebalancing"
    else:
        portfolio_score = 30
        portfolio_label = "High Risk Deviation"
    
    components["portfolio_health"] = {
        "score": portfolio_score,
        "weight": 20,
        "weighted": portfolio_score * 0.20,
        "label": portfolio_label,
        "current_equity": current_equity,
        "recommended_range": f"{rec_min}%-{rec_max}%",
        "priority": "MEDIUM" if portfolio_score < 70 else "GOOD"
    }
    
    # 4. Debt Management Score (15% weight) - based on debt stress
    # compute_debt_stress emits "Healthy" / "Moderate" / "Stressed".
    debt_stress = analysis.get("debtStress", "Healthy")
    if debt_stress == "Healthy":
        debt_score = 100
        debt_label = "Healthy"
    elif debt_stress == "Moderate":
        debt_score = 60
        debt_label = "Moderate"
    elif debt_stress == "Stressed":
        debt_score = 30
        debt_label = "High Burden"
    else:
        debt_score = 80
        debt_label = "Unknown"
    
    components["debt_management"] = {
        "score": debt_score,
        "weight": 15,
        "weighted": debt_score * 0.15,
        "label": debt_label,
        "priority": "HIGH" if debt_score < 50 else ("MEDIUM" if debt_score < 70 else "GOOD")
    }
    
    # 5. Goal Readiness Score (10% weight) - based on goal funding status
    goals = facts.get("goals") or []
    total_goals = len(goals)
    # Use IHS or estimate from surplus
    ihs = analysis.get("ihs") or {}
    ihs_score = ihs.get("score", 50)
    
    # Approximate goal readiness from IHS and surplus band
    # compute_surplus_band emits "Low" / "Adequate" / "Strong".
    surplus_band = analysis.get("surplusBand", "Unknown")
    if surplus_band == "Strong":
        goal_score = min(80, ihs_score)
        goal_label = "Good Progress"
    elif surplus_band == "Adequate":
        goal_score = min(60, ihs_score)
        goal_label = "Moderate"
    elif surplus_band == "Low":
        goal_score = min(40, ihs_score)
        goal_label = "Needs Work"
    else:
        goal_score = 30
        goal_label = "At Risk"
    
    components["goal_readiness"] = {
        "score": goal_score,
        "weight": 10,
        "weighted": goal_score * 0.10,
        "label": goal_label,
        "total_goals": total_goals,
        "priority": "MEDIUM" if goal_score < 60 else "GOOD"
    }
    
    # 6. Tax Efficiency Score (5% weight) - based on ITR data if available
    itr = facts.get("itr") or {}
    if itr:
        # If significantly using deductions, good score
        deductions = itr.get("deductions_claimed") or []
        total_deductions = sum(d.get("amount", 0) for d in deductions)
        if total_deductions >= 150000:
            tax_score = 80
            tax_label = "Well Utilized"
        elif total_deductions >= 50000:
            tax_score = 50
            tax_label = "Partial Utilization"
        else:
            tax_score = 30
            tax_label = "Room for Improvement"
    else:
        tax_score = 50
        tax_label = "Not Assessed"
    
    components["tax_efficiency"] = {
        "score": tax_score,
        "weight": 5,
        "weighted": tax_score * 0.05,
        "label": tax_label,
        "priority": "MEDIUM" if tax_score < 60 else "GOOD"
    }
    
    # Calculate overall score
    overall = sum(c["weighted"] for c in components.values())
    overall = round(overall, 0)
    
    # Overall label
    if overall >= 80:
        overall_label = "Excellent"
    elif overall >= 60:
        overall_label = "Good"
    elif overall >= 40:
        overall_label = "Needs Attention"
    else:
        overall_label = "Critical"
    
    # Priority areas (sorted by score ascending)
    priority_areas = sorted(
        [(k, v["priority"], v["score"]) for k, v in components.items()],
        key=lambda x: x[2]
    )[:3]  # Top 3 priority areas
    
    return {
        "overall": int(overall),
        "overall_label": overall_label,
        "components": components,
        "priority_areas": [(p[0], p[1]) for p in priority_areas]
    }
