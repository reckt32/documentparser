"""Data accessors and formatting helpers used by the report pages."""

from backend.analysis.llm_sections import PriorityAllocationEngine, compute_goal_sip
from backend.assumptions import (
    LIFE_COVER_MULTIPLE,
    SIP_ROUND_STEP,
    health_cover_status,
    life_cover_status,
    recommended_health_cover,
)
from backend.utils import _format_indian_amount, _safe_float


def _format_status(value) -> str:
    """Render any status / band / category label in STRICTLY CAPS for the PDF report."""
    if value is None:
        return "-"
    return str(value).upper()


def _fmt_rs(amount, compact=True):
    try:
        val = float(amount or 0)
    except Exception:
        val = 0
    return f"Rs. {_format_indian_amount(val)}" if compact else f"Rs. {val:,.0f}"


def _num(value, default=0.0):
    return _safe_float(value, default)


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_list(value):
    return value if isinstance(value, list) else []


def _score_value(value, default=0.0):
    if isinstance(value, dict):
        return _num(value.get("score"), default)
    return _num(value, default)


def _ihs_breakdown(analysis):
    ihs = _as_dict(_as_dict(analysis).get("ihs"))
    return _as_dict(ihs.get("breakdown"))


def _ihs_component_score(analysis, key, default=0.0):
    return _score_value(_ihs_breakdown(analysis).get(key), default)


def _recommended_equity_band(advanced_risk):
    band = _as_dict(advanced_risk).get("recommendedEquityBand")
    if isinstance(band, dict):
        return _num(band.get("min"), 40), _num(band.get("max"), 60)
    return 40, 60


def _portfolio_equity(portfolio):
    return _num(portfolio.get("equity_pct") or portfolio.get("equity") or portfolio.get("equity_percentage"), 0)


def _portfolio_value(portfolio, lifestyle=None):
    """Return the normalized portfolio corpus used on every report page."""
    portfolio = _as_dict(portfolio)
    lifestyle = _as_dict(lifestyle)
    for key in ("total_value", "current_value", "total"):
        value = _num(portfolio.get(key), 0)
        if value > 0:
            return value
    return _num(lifestyle.get("manual_corpus"), 0)


def _emergency_fund_metrics(client_facts):
    """Preserve the distinction between an explicit zero and missing data."""
    income = _as_dict(_as_dict(client_facts).get("income"))
    lifestyle = _as_dict(_as_dict(client_facts).get("lifestyle"))
    expenses = _num(income.get("monthlyExpenses"), 0)
    raw_current = lifestyle.get("emergency_fund")
    known = raw_current not in (None, "")
    current = _num(raw_current, 0) if known else 0.0
    months = (current / expenses) if known and expenses > 0 else None
    target = expenses * 6
    return {
        "known": known,
        "current": current,
        "months": months,
        "target": target,
        "gap": max(0, target - current),
    }


def _priority_monthly_amount(allocation_output, priority_name):
    """Read one premium from the allocation engine's canonical split."""
    for item in _as_list(_as_dict(allocation_output).get("priority_breakdown")):
        item = _as_dict(item)
        if str(item.get("name") or "").strip().lower() == priority_name.strip().lower():
            return _num(item.get("monthly_amount"), 0)
    return 0.0


def _display_monthly_surplus(client_facts):
    income = client_facts.get("income") or {}
    bank = client_facts.get("bank") or {}
    monthly_income = _num(income.get("annualIncome"), 0) / 12
    computed = monthly_income - _num(income.get("monthlyExpenses"), 0) - _num(income.get("monthlyEmi"), 0)
    bank_net = _num(bank.get("net_cashflow"), 0)
    has_bank_cashflow = any(_num(bank.get(k), 0) > 0 for k in ("total_inflows", "total_outflows"))
    if has_bank_cashflow and monthly_income > 0 and abs(bank_net) <= monthly_income * 1.5:
        return bank_net
    return computed


def _recommended_band_text(advanced_risk):
    band = _as_dict(advanced_risk).get("recommendedEquityBand") or {}
    if isinstance(band, dict):
        b_min, b_max = _recommended_equity_band(advanced_risk)
        return f"{b_min:.0f}-{b_max:.0f}%"
    return str(band or "40-60%")


def _recommended_band_mid(advanced_risk):
    advanced_risk = _as_dict(advanced_risk)
    if advanced_risk.get("recommendedEquityMid") is not None:
        return _num(advanced_risk.get("recommendedEquityMid"), 50)
    b_min, b_max = _recommended_equity_band(advanced_risk)
    return (b_min + b_max) / 2


def _urgency(score):
    score = _num(score, 0)
    if score < 40:
        return "IMMEDIATE"
    if score < 75:
        return "HIGH"
    return "MAINTAIN"


def _allocation_output(client_facts):
    analysis = client_facts.get("analysis") or {}
    advanced_risk = _as_dict(analysis.get("advancedRisk"))
    income = client_facts.get("income") or {}
    insurance = client_facts.get("insurance") or {}
    personal = client_facts.get("personal") or {}
    portfolio = client_facts.get("portfolio") or {}
    monthly_income = _num(income.get("annualIncome"), 0) / 12
    monthly_surplus = _display_monthly_surplus(client_facts)
    required_life = _num((analysis.get("_diagnostics") or {}).get("requiredLifeCover"), monthly_income * 12 * LIFE_COVER_MULTIPLE)
    current_life = _num(insurance.get("lifeCover"), 0)
    current_health = _num(insurance.get("healthCover"), 0)
    recommended_health = recommended_health_cover(income.get("annualIncome"), personal.get("dependents_count"))
    goals = []
    for g in _as_list(client_facts.get("goals")):
        g = _as_dict(g)
        ideal = compute_goal_sip(g.get("target_amount"), g.get("horizon_years"), g.get("risk_tolerance") or advanced_risk.get("finalCategory") or "moderate") or 0
        goals.append({
            "name": g.get("name") or "Goal",
            "target_amount": _num(g.get("target_amount"), 0),
            "horizon_years": g.get("horizon_years"),
            "ideal_sip": ideal,
            "risk_category": g.get("risk_tolerance") or "moderate",
        })
    # Avoid a duplicate Emergency Fund: if the client already listed one as a goal
    # (it's a selectable goal type), don't let the engine inject its own EF bucket.
    lifestyle_facts = client_facts.get("lifestyle") or {}
    ef_current = _num(lifestyle_facts.get("emergency_fund"), 0)
    has_ef_goal = any("emergency" in (str(g.get("name") or "")).lower() for g in goals)
    ef_target = 0 if has_ef_goal else _num(income.get("monthlyExpenses"), 0) * 6
    try:
        out = PriorityAllocationEngine.compute_allocation(
            monthly_surplus=max(0, monthly_surplus),
            term_insurance_gap=max(0, required_life - current_life),
            health_insurance_gap=max(0, recommended_health - current_health),
            goals=goals,
            age=int(_num(personal.get("age"), 35)),
            has_dependents=_num(personal.get("dependents_count"), 0) > 0,
            emergency_fund_target=ef_target,
            emergency_fund_current=ef_current,
            # Client's current balance — a one-time lump sum kept separate from the
            # waterfall. Premiums are provisioned monthly from surplus, not from this.
            available_savings=_num(lifestyle_facts.get("available_savings"), 0),
            existing_sip_commitments=_num(portfolio.get("total_monthly_sip") or portfolio.get("monthly_sip"), 0),
            manual_sip=_num((client_facts.get("lifestyle") or {}).get("manual_sip"), 0),
            manual_corpus=_num((client_facts.get("lifestyle") or {}).get("manual_corpus"), 0),
            expected_pension=_num((client_facts.get("lifestyle") or {}).get("expected_pension"), 0),
            monthly_expenses=_num(income.get("monthlyExpenses"), 0),
        )
    except Exception:
        out = {}
    existing_sip_running = _num(portfolio.get("total_monthly_sip") or portfolio.get("monthly_sip"), 0)
    if existing_sip_running <= 0:
        existing_sip_running = _num((client_facts.get("lifestyle") or {}).get("manual_sip"), 0)
    goal_rows = out.get("goal_sip_table") or []
    combined = sum(_num(g.get("shortfall"), 0) for g in goal_rows) or sum(_num(g.get("ideal_sip"), 0) for g in goals)
    out.setdefault("goal_sip_table", goal_rows)
    out.setdefault("combined_shortfall", combined)
    out.setdefault("insurance_provision", out.get("insurance_sip_monthly") or 0)
    out.setdefault("existing_sip_running", existing_sip_running)
    out.setdefault("monthly_surplus", round(max(0, monthly_surplus), 0))
    if _num(out.get("monthly_surplus"), 0) <= 0 and monthly_surplus > 0:
        out["monthly_surplus"] = round(monthly_surplus, 0)
    out.setdefault("available_for_goals", out.get("remaining_for_goals") or max(0, monthly_surplus - existing_sip_running - _num(out.get("insurance_provision"), 0)))
    if _num(out.get("available_for_goals"), 0) <= 0 and monthly_surplus > existing_sip_running:
        out["available_for_goals"] = max(0, monthly_surplus - existing_sip_running - _num(out.get("insurance_provision"), 0))
    out.setdefault("total_investing", existing_sip_running + _num(out.get("available_for_goals"), 0))
    if not out.get("goal_sip_table") and goals:
        available = _num(out.get("available_for_goals"), 0)
        total_ideal = sum(_num(g.get("ideal_sip"), 0) for g in goals)
        fallback_rows = []
        for g in goals:
            ideal = _num(g.get("ideal_sip"), 0)
            allocated = available * ideal / total_ideal if total_ideal > 0 else 0
            fallback_rows.append({
                "name": g.get("name"),
                "target_amount": g.get("target_amount"),
                "horizon_years": g.get("horizon_years"),
                "ideal_sip": ideal,
                "allocated_sip": allocated,
                "shortfall": max(0, ideal - allocated),
                "coverage_pct": (allocated / ideal * 100) if ideal else 100,
                "status": "Gap" if allocated < ideal else "Funded",
            })
        out["goal_sip_table"] = fallback_rows
        out["combined_shortfall"] = sum(_num(g.get("shortfall"), 0) for g in fallback_rows)

    # ── Single source of truth for the goal cards / cashflow plan ──────────────
    # Per goal: Current SIP (existing SIP shared by need), Gap = Required − Current,
    # and Coverage = the new money the waterfall directed at that goal (70% of the
    # pool to EF + Retirement, rest nearest-horizon-first, capped at each need and
    # locked to Rs.500 steps — see PriorityAllocationEngine). "Extra savings
    # needed" = whatever gaps survive the waterfall; leftover pool money stays in
    # the savings account (unallocated_to_savings).
    rows = out.get("goal_sip_table") or []
    existing = _num(out.get("existing_sip_running"), 0)
    total_ideal = sum(_num(r.get("ideal_sip"), 0) for r in rows)
    total_gap = 0.0
    total_cov = 0.0
    total_savings_increase = 0.0
    for r in rows:
        ideal = _num(r.get("ideal_sip"), 0)
        curr = existing * ideal / total_ideal if existing > 0 and total_ideal > 0 and ideal > 0 else 0
        gap = max(0, ideal - curr)
        cov = min(_num(r.get("allocated_sip"), 0), gap)
        r["current_sip"] = round(curr, 0)
        r["gap_mo"] = round(gap, 0)
        # Coverage stays the Rs.500-locked figure the client should actually set
        # up; a sub-step residue lives in the savings buffer, so the goal still
        # counts as covered for status purposes.
        r["coverage_used"] = round(cov, 0)
        r["surplus_covers_gap"] = (gap - cov) < SIP_ROUND_STEP
        total_gap += gap
        total_cov += cov
        if not r["surplus_covers_gap"]:
            total_savings_increase += gap - cov
    out["goal_total_gap"] = round(total_gap, 0)
    out["goal_total_coverage"] = round(total_cov, 0)
    out["goal_savings_increase"] = round(total_savings_increase, 0)
    # Existing SIP + newly allocated coverage is the amount the roadmap calls
    # "Deploy". Store it once so the cash-flow and roadmap pages cannot drift.
    out["goal_total_deployment"] = round(existing + total_cov, 0)
    # Total required SIP across all rows — the "requirement" the cashflow page
    # compares against (combined_shortfall is the unmet remainder, not the need).
    out["goal_total_required"] = round(sum(_num(r.get("ideal_sip"), 0) for r in rows), 0)
    out.setdefault("unallocated_to_savings", 0)

    # Insurance verdicts shared by the snapshot, protection page and roadmap so
    # "recommended" amounts and adequate/over-insured tags can't disagree.
    out["recommended_health_cover"] = recommended_health
    out["health_cover_status"] = health_cover_status(current_health, recommended_health)
    out["required_life_cover"] = required_life
    out["life_cover_status"] = life_cover_status(current_life, required_life)
    return out
