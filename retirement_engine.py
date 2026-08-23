"""Deterministic retirement analysis for the corpus-deployment MVP.

The module deliberately has no Flask, database, or PDF dependencies.  The same
normalized result is consumed by the API, report generator, dashboard, and
tests so financial figures cannot drift between surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class RetirementAssumptions:
    planning_age: int = 85
    liquidity_months: int = 6
    scss_rate: float = 0.082
    fd_rate: float = 0.075
    inflation_rate: float = 0.06
    sustainable_withdrawal_rate: float = 0.05
    health_cover_multiple: int = 30
    swp_step_up: float = 0.07
    swp_growth_rate: float = 0.10


ASSUMPTIONS = RetirementAssumptions()
SWP_RATES: Tuple[Tuple[str, float], ...] = (
    ("gold_standard", 0.035),
    ("healthy", 0.05),
    ("caution", 0.06),
    ("critical", 0.07),
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number and number not in (float("inf"), float("-inf")) else default


def _money(value: Any) -> float:
    return round(max(0.0, _number(value)), 2)


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _mapping(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _records(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def validate_retirement_input(payload: Dict[str, Any]) -> List[str]:
    """Return stable field-path errors; an empty list means the payload is valid."""
    errors: List[str] = []
    profile = _mapping(payload.get("profile"))
    expenses = _mapping(payload.get("expenses"))
    insurance = _mapping(payload.get("insurance"))

    if not str(profile.get("client_name") or "").strip():
        errors.append("profile.client_name:required")
    pan = str(profile.get("pan") or "").strip().upper()
    if len(pan) != 10 or not (pan[:5].isalpha() and pan[5:9].isdigit() and pan[9:].isalpha()):
        errors.append("profile.pan:invalid")
    age = _integer(profile.get("age"), -1)
    if age < 55 or age >= ASSUMPTIONS.planning_age:
        errors.append(f"profile.age:must_be_55_to_{ASSUMPTIONS.planning_age - 1}")
    if _money(expenses.get("monthly_core")) <= 0:
        errors.append("expenses.monthly_core:required")

    for idx, goal in enumerate(_records(payload.get("goals"))):
        if not str(goal.get("name") or "").strip():
            errors.append(f"goals.{idx}.name:required")
        if _money(goal.get("target_amount")) <= 0:
            errors.append(f"goals.{idx}.target_amount:required")
        if _integer(goal.get("years_from_now"), 0) < 1:
            errors.append(f"goals.{idx}.years_from_now:required")

    if insurance.get("has_health_insurance") is True and _money(insurance.get("health_cover")) <= 0:
        errors.append("insurance.health_cover:required")
    return errors


def _old_age_care_percent(age: int) -> float:
    if age >= 76:
        return 0.20
    if age >= 70:
        return 0.15
    if age >= 65:
        return 0.12
    return 0.10


def _goal_return(years: int) -> Tuple[float, str, int]:
    if years <= 3:
        return 0.06, "Capital preservation", 10
    if years <= 7:
        return 0.08, "Balanced growth", 40
    return 0.10, "Inflation-beating growth", 70


def _depletion_age(corpus: float, monthly_withdrawal: float, start_age: int) -> int | None:
    if corpus <= 0 or monthly_withdrawal <= 0:
        return start_age
    balance = corpus
    annual_withdrawal = monthly_withdrawal * 12
    for year in range(1, ASSUMPTIONS.planning_age - start_age + 31):
        balance = balance * (1 + ASSUMPTIONS.swp_growth_rate) - annual_withdrawal
        if balance <= 0:
            return start_age + year
        annual_withdrawal *= 1 + ASSUMPTIONS.swp_step_up
    return None


def _score_band(score: int) -> str:
    if score >= 75:
        return "well_structured"
    if score >= 40:
        return "needs_attention"
    return "immediate_action"


def analyze_retirement(payload: Dict[str, Any]) -> Dict[str, Any]:
    errors = validate_retirement_input(payload)
    if errors:
        raise ValueError(";".join(errors))

    profile = _mapping(payload.get("profile"))
    assets = _mapping(payload.get("assets"))
    income = _mapping(payload.get("income"))
    expenses = _mapping(payload.get("expenses"))
    dependents = _records(payload.get("dependents"))
    insurance = _mapping(payload.get("insurance"))
    goals_input = _records(payload.get("goals"))

    age = _integer(profile.get("age"))
    planning_years = ASSUMPTIONS.planning_age - age

    other_investments = _records(assets.get("other_investments"))
    other_total = sum(_money(item.get("amount")) for item in other_investments)
    other_deployable = sum(
        _money(item.get("amount")) for item in other_investments if item.get("deployable", True)
    )
    asset_values = {
        "primary_residence": _money(assets.get("primary_residence")),
        "investment_property": _money(assets.get("investment_property")),
        "scss": _money(assets.get("scss")),
        "fixed_deposits": _money(assets.get("fixed_deposits")),
        "gold_silver": _money(assets.get("gold_silver")),
        "insurance_surrender": _money(assets.get("insurance_surrender")),
        "equity_investments": _money(assets.get("equity_investments")),
        "debt_investments": _money(assets.get("debt_investments")),
        "liquid_savings": _money(assets.get("liquid_savings")),
        "pension_lump_sum": _money(assets.get("pension_lump_sum")),
        "other_investments": round(other_total, 2),
    }
    total_assets = round(sum(asset_values.values()), 2)
    liabilities = _money(assets.get("outstanding_liabilities"))
    net_worth = round(max(0.0, total_assets - liabilities), 2)
    usable_corpus = round(
        asset_values["fixed_deposits"]
        + asset_values["equity_investments"]
        + asset_values["debt_investments"]
        + asset_values["liquid_savings"]
        + asset_values["pension_lump_sum"]
        + other_deployable,
        2,
    )

    monthly_core = _money(expenses.get("monthly_core"))
    annual_items = _records(expenses.get("annual_items"))
    annual_expenses = sum(_money(item.get("amount")) for item in annual_items)
    dependent_cost = sum(_money(item.get("monthly_cost")) for item in dependents)
    monthly_emi = _money(expenses.get("monthly_emi"))
    effective_monthly_expense = round(monthly_core + annual_expenses / 12 + dependent_cost, 2)

    scss_interest = round(asset_values["scss"] * ASSUMPTIONS.scss_rate / 12, 2)
    fd_interest = round(asset_values["fixed_deposits"] * ASSUMPTIONS.fd_rate / 12, 2)
    pension_sources = {
        "government": _money(income.get("government_pension")),
        "employer": _money(income.get("employer_pension")),
        "nps_annuity": _money(income.get("nps_annuity")),
        "other_annuity": _money(income.get("other_annuity")),
    }
    rental_income = _money(income.get("rental_income"))
    other_income_rows = _records(income.get("other_sources"))
    permanent_other = sum(
        _money(item.get("monthly_amount"))
        for item in other_income_rows
        if not _integer(item.get("years_remaining"), 0)
    )
    temporary_other = sum(
        _money(item.get("monthly_amount"))
        for item in other_income_rows
        if _integer(item.get("years_remaining"), 0) > 0
    )
    permanent_income = round(
        sum(pension_sources.values()) + rental_income + scss_interest + fd_interest + permanent_other,
        2,
    )
    total_current_income = round(permanent_income + temporary_other, 2)
    current_gap = round(max(0.0, effective_monthly_expense + monthly_emi - total_current_income), 2)
    permanent_gap = round(max(0.0, effective_monthly_expense + monthly_emi - permanent_income), 2)

    liquidity_buffer = round(effective_monthly_expense * ASSUMPTIONS.liquidity_months, 2)
    insurance_premium = _money(insurance.get("annual_health_premium")) + _money(
        insurance.get("annual_term_premium")
    )
    insurance_supply = round(insurance_premium * planning_years, 2)
    old_age_care_pct = _old_age_care_percent(age)
    old_age_care = round(usable_corpus * old_age_care_pct, 2)
    pension_corpus_available = round(
        max(0.0, usable_corpus - liquidity_buffer - insurance_supply - old_age_care), 2
    )
    pension_corpus_required = round(
        permanent_gap * 12 / ASSUMPTIONS.sustainable_withdrawal_rate, 2
    )
    corpus_adequacy_pct = (
        100.0
        if pension_corpus_required <= 0
        else round(pension_corpus_available / pension_corpus_required * 100, 1)
    )

    spectrum: List[Dict[str, Any]] = []
    swp_status = "no_swp_needed" if permanent_gap <= 0 else "gap"
    recommended_rate = 0.0
    for label, rate in SWP_RATES:
        monthly = round(pension_corpus_available * rate / 12, 2)
        covers_gap = permanent_gap > 0 and monthly >= permanent_gap
        if swp_status == "gap" and covers_gap:
            swp_status = label
            recommended_rate = rate
        spectrum.append(
            {
                "status": label,
                "rate": rate,
                "monthly_income": monthly,
                "covers_gap": covers_gap,
                "depletion_age": _depletion_age(pension_corpus_available, monthly, age)
                if rate >= 0.06
                else None,
            }
        )
    if swp_status == "gap":
        recommended_rate = 0.07

    ideal_corpus = round(permanent_gap * 12 / 0.035, 2) if permanent_gap > 0 else 0.0
    natural_swp_rate = (
        round(permanent_gap * 12 / pension_corpus_available, 4)
        if pension_corpus_available > 0
        else 0.0
    )
    annual_recommended = pension_corpus_available * recommended_rate
    swp_projection: List[Dict[str, Any]] = []
    cumulative = 0.0
    for year in (1, 3, 5, 10):
        monthly = annual_recommended * ((1 + ASSUMPTIONS.swp_step_up) ** (year - 1)) / 12
        cumulative = sum(
            annual_recommended * ((1 + ASSUMPTIONS.swp_step_up) ** offset)
            for offset in range(year)
        )
        swp_projection.append(
            {"year": year, "age": age + year, "monthly_income": round(monthly, 2), "cumulative": round(cumulative, 2)}
        )

    pension_surplus = round(max(0.0, pension_corpus_available - pension_corpus_required), 2)
    goals: List[Dict[str, Any]] = []
    remaining_goal_corpus = pension_surplus
    for goal in sorted(goals_input, key=lambda item: _integer(item.get("years_from_now"), 99)):
        years = max(1, _integer(goal.get("years_from_now"), 1))
        target_today = _money(goal.get("target_amount"))
        inflation_linked = goal.get("inflation_linked", True) is not False
        future_cost = target_today * ((1 + ASSUMPTIONS.inflation_rate) ** years) if inflation_linked else target_today
        expected_return, strategy, growth_allocation = _goal_return(years)
        corpus_needed_today = future_cost / ((1 + expected_return) ** years)
        allocation = min(remaining_goal_corpus, corpus_needed_today)
        remaining_goal_corpus = max(0.0, remaining_goal_corpus - allocation)
        goals.append(
            {
                "name": str(goal.get("name") or "Goal").strip(),
                "type": str(goal.get("type") or "other").strip().lower(),
                "years_from_now": years,
                "target_today": round(target_today, 2),
                "future_cost": round(future_cost, 2),
                "corpus_needed_today": round(corpus_needed_today, 2),
                "allocated_corpus": round(allocation, 2),
                "funding_pct": round(allocation / corpus_needed_today * 100, 1) if corpus_needed_today else 100.0,
                "strategy": strategy,
                "growth_allocation_pct": growth_allocation,
                "expected_return": expected_return,
            }
        )

    recommended_health_cover = round(effective_monthly_expense * ASSUMPTIONS.health_cover_multiple, 2)
    health_cover = _money(insurance.get("health_cover"))
    health_gap = round(max(0.0, recommended_health_cover - health_cover), 2)
    has_health = insurance.get("has_health_insurance") is True
    spouse_covered = insurance.get("spouse_covered") is not False
    has_term = insurance.get("has_term_insurance") is True
    has_dependents = bool(dependents)

    corpus_score = 25 if corpus_adequacy_pct >= 100 else 20 if corpus_adequacy_pct >= 80 else 15 if corpus_adequacy_pct >= 60 else 10 if corpus_adequacy_pct >= 40 else 5
    income_cover_pct = (
        min(100.0, permanent_income / (effective_monthly_expense + monthly_emi) * 100)
        if effective_monthly_expense + monthly_emi > 0
        else 100.0
    )
    income_score = 25 if income_cover_pct >= 100 else 20 if income_cover_pct >= 75 else 15 if income_cover_pct >= 50 else 10 if income_cover_pct >= 25 else 5
    if not has_health:
        protection_score = 0
    else:
        protection_score = 12
        if health_gap <= 0:
            protection_score += 7
        if spouse_covered:
            protection_score += 3
        if not has_dependents or has_term:
            protection_score += 3
    goal_funding_pct = (
        round(sum(goal["allocated_corpus"] for goal in goals) / sum(goal["corpus_needed_today"] for goal in goals) * 100, 1)
        if goals and sum(goal["corpus_needed_today"] for goal in goals) > 0
        else 100.0
    )
    goal_score = 25 if goal_funding_pct >= 100 else 18 if goal_funding_pct >= 75 else 12 if goal_funding_pct >= 40 else 5
    total_score = min(100, corpus_score + income_score + protection_score + goal_score)

    flags: List[Dict[str, Any]] = []

    def flag(priority: str, code: str, message: str, action: str, value: float = 0.0) -> None:
        flags.append({"priority": priority, "code": code, "message": message, "action": action, "value": round(value, 2)})

    if profile.get("will_in_place") is not True:
        flag("critical", "will_missing", "A current will is not in place.", "Complete a will and nomination review with a qualified lawyer.")
    if not has_health:
        flag("critical", "health_missing", "No health insurance is recorded.", "Arrange health cover before deploying the retirement corpus.", recommended_health_cover)
    elif health_gap > 0:
        flag("high", "health_gap", "Health cover is below the planning benchmark.", "Review a top-up or base-cover increase with the insurer.", health_gap)
    if not spouse_covered:
        flag("critical", "spouse_uncovered", "The spouse is not covered by the health policy.", "Add spouse protection or arrange a separate policy.", recommended_health_cover)
    if swp_status in ("critical", "gap"):
        flag("critical", "swp_gap", "The current corpus requires an aggressive or unsustainable withdrawal rate.", "Reduce the monthly gap, add corpus, or defer non-essential goals.", max(0.0, ideal_corpus - pension_corpus_available))
    elif swp_status == "caution":
        flag("high", "swp_stretched", "The sustainable withdrawal plan is stretched.", "Review expenses and SWP performance at least annually.")
    if assets.get("fixed_deposits") and usable_corpus > 0 and asset_values["fixed_deposits"] / usable_corpus > 0.30:
        flag("high", "fd_concentration", "More than 30% of usable corpus is held in fixed deposits.", "Review post-tax real returns and rebalance excess FD exposure.", asset_values["fixed_deposits"])
    if asset_values["insurance_surrender"] > 0:
        flag("high", "insurance_investment", "Insurance-investment surrender value is illiquid and may lag inflation.", "Request a surrender-versus-hold analysis before changing the policy.", asset_values["insurance_surrender"])
    if has_dependents and not has_term:
        flag("high", "term_missing", "Financial dependents are recorded but no term cover is available.", "Review whether term cover is still required for dependent obligations.")
    if goal_funding_pct < 100:
        flag("high", "goals_underfunded", "Goals exceed the surplus available after pension security.", "Prioritise pension, then phase or resize lower-priority goals.", sum(max(0.0, g["corpus_needed_today"] - g["allocated_corpus"]) for g in goals))
    if any(g["type"] == "property" for g in goals):
        flag("high", "property_liquidity", "A post-retirement property goal can materially reduce liquidity.", "Fund it only from surplus after pension, protection, and liquidity reserves.")

    priority_order = {"critical": 0, "high": 1, "maintain": 2}
    flags.sort(key=lambda item: (priority_order.get(item["priority"], 9), -item.get("value", 0)))

    return {
        "module": "retirement",
        "profile": {
            "client_name": str(profile.get("client_name") or "").strip(),
            "pan": str(profile.get("pan") or "").strip().upper(),
            "age": age,
            "planning_age": ASSUMPTIONS.planning_age,
            "planning_years": planning_years,
            "will_in_place": profile.get("will_in_place") is True,
        },
        "net_worth": {
            "assets": asset_values,
            "total_assets": total_assets,
            "liabilities": liabilities,
            "net_worth": net_worth,
            "usable_corpus": usable_corpus,
            "protected_scss": asset_values["scss"],
        },
        "cashflow": {
            "monthly_core_expense": monthly_core,
            "annual_expenses": round(annual_expenses, 2),
            "annual_items": annual_items,
            "dependent_cost": round(dependent_cost, 2),
            "monthly_emi": monthly_emi,
            "effective_monthly_expense": effective_monthly_expense,
            "pension_sources": pension_sources,
            "rental_income": rental_income,
            "scss_interest": scss_interest,
            "fd_interest": fd_interest,
            "other_income_sources": other_income_rows,
            "permanent_income": permanent_income,
            "temporary_income": round(temporary_other, 2),
            "total_current_income": total_current_income,
            "current_gap": current_gap,
            "permanent_gap": permanent_gap,
        },
        "allocation": {
            "liquidity_buffer": liquidity_buffer,
            "insurance_supply": insurance_supply,
            "old_age_care": old_age_care,
            "old_age_care_pct": old_age_care_pct,
            "pension_corpus_available": pension_corpus_available,
            "pension_corpus_required": pension_corpus_required,
            "pension_surplus": pension_surplus,
            "unallocated_surplus_after_goals": round(remaining_goal_corpus, 2),
            "corpus_adequacy_pct": corpus_adequacy_pct,
        },
        "swp": {
            "status": swp_status,
            "recommended_rate": recommended_rate,
            "natural_rate": natural_swp_rate,
            "spectrum": spectrum,
            "ideal_corpus_35": ideal_corpus,
            "ideal_gap": round(max(0.0, ideal_corpus - pension_corpus_available), 2),
            "projection": swp_projection,
        },
        "protection": {
            "has_health_insurance": has_health,
            "health_cover": health_cover,
            "recommended_health_cover": recommended_health_cover,
            "health_cover_gap": health_gap,
            "spouse_covered": spouse_covered,
            "has_term_insurance": has_term,
            "term_cover": _money(insurance.get("term_cover")),
            "insurance_supply_corpus": insurance_supply,
        },
        "goals": goals,
        "scores": {
            "overall": total_score,
            "band": _score_band(total_score),
            "corpus": corpus_score,
            "income": income_score,
            "protection": protection_score,
            "goals": goal_score,
            "corpus_adequacy_pct": corpus_adequacy_pct,
            "income_coverage_pct": round(income_cover_pct, 1),
            "goal_funding_pct": goal_funding_pct,
        },
        "flags": flags,
        "assumptions": {
            "planning_age": ASSUMPTIONS.planning_age,
            "liquidity_months": ASSUMPTIONS.liquidity_months,
            "scss_rate": ASSUMPTIONS.scss_rate,
            "fd_rate": ASSUMPTIONS.fd_rate,
            "inflation_rate": ASSUMPTIONS.inflation_rate,
            "sustainable_withdrawal_rate": ASSUMPTIONS.sustainable_withdrawal_rate,
            "swp_step_up": ASSUMPTIONS.swp_step_up,
            "swp_growth_rate": ASSUMPTIONS.swp_growth_rate,
        },
    }


def dashboard_action_items(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    dimension_map = {
        "will_missing": "estate",
        "health_missing": "protection",
        "health_gap": "protection",
        "spouse_uncovered": "protection",
        "term_missing": "protection",
        "swp_gap": "retirement_income",
        "swp_stretched": "retirement_income",
        "fd_concentration": "portfolio",
        "insurance_investment": "portfolio",
        "goals_underfunded": "goals",
        "property_liquidity": "goals",
    }
    items = []
    for index, item in enumerate(analysis.get("flags") or []):
        code = item.get("code") or f"retirement_{index + 1}"
        items.append(
            {
                "item_id": f"retirement_{code}",
                "dimension": dimension_map.get(code, "retirement"),
                "urgency": "IMMEDIATE" if item.get("priority") == "critical" else "HIGH",
                "value_type": "INR" if _money(item.get("value")) > 0 else "NONE",
                "value_num": _money(item.get("value")),
                "description": item.get("action"),
                "final_status": "PENDING",
                "is_converted": False,
            }
        )
    return items
