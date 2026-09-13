"""Deterministic adviser-side retirement plan solver.

The engine owns every financial number used by the Plan Builder, PDF and
dashboard. It does not select schemes, score the client or remove goals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class RetirementAssumptions:
    version: str = "retirement_solver_v4_2026_09"
    planning_age: int = 85
    expense_inflation: float = 0.06
    pension_indexation: float = 0.06
    emergency_months: int = 6
    premium_reserve_multiple: int = 10
    opportunity_default_pct: float = 0.05
    withdrawal_floor: float = 0.035
    withdrawal_cap: float = 0.07
    scss_rate: float = 0.082
    fd_rate: float = 0.075
    interest_tax_rate: float = 0.22


ASSUMPTIONS = RetirementAssumptions()

RISK_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "no_risk": {"label": "No Risk", "fund_type": "Money market", "return": 0.06},
    "low": {"label": "Low Risk", "fund_type": "Long-term debt / arbitrage / corporate bond", "return": 0.07},
    "medium": {"label": "Medium Risk", "fund_type": "Multi-asset / balanced advantage", "return": 0.09},
    "aggressive_medium": {"label": "Aggressive Medium", "fund_type": "Equity-tilted hybrid", "return": 0.10},
    "high": {"label": "High Risk", "fund_type": "Flexi cap", "return": 0.12},
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


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


def _bounded(value: Any, minimum: float, maximum: float, default: float) -> float:
    return min(maximum, max(minimum, _number(value, default)))


def _slug(value: Any, fallback: str) -> str:
    text = "_".join(str(value or "").strip().lower().replace("-", " ").split())
    return text or fallback


def retirement_plan_is_accepted(payload: Dict[str, Any]) -> bool:
    """Return the explicit adviser approval submitted with the current plan."""
    return _mapping(payload.get("planning")).get("adviser_accepted") is True


def _uses_interest_tax_assumption(instrument_type: Any) -> bool:
    return _slug(instrument_type, "") in {"scss", "fixed_deposit", "fd"}


def validate_retirement_input(payload: Dict[str, Any]) -> List[str]:
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
    planning_age = _integer(profile.get("planning_age"), ASSUMPTIONS.planning_age)
    if age < 55 or age >= planning_age:
        errors.append(f"profile.age:must_be_55_to_{planning_age - 1}")
    if _money(expenses.get("monthly_core")) <= 0:
        errors.append("expenses.monthly_core:required")
    for index, goal in enumerate(_records(payload.get("goals"))):
        if goal.get("active", True) is False:
            continue
        if not str(goal.get("name") or "").strip():
            errors.append(f"goals.{index}.name:required")
        goal_type = _slug(goal.get("type"), "other")
        amount = _money(goal.get("annual_amount")) if goal_type == "vacation" else _money(goal.get("target_amount"))
        if amount <= 0:
            errors.append(f"goals.{index}.target_amount:required")
        if _integer(goal.get("years_from_now"), 0) < 1:
            errors.append(f"goals.{index}.years_from_now:required")
    if insurance.get("has_health_insurance") is True and _money(insurance.get("health_cover")) <= 0:
        errors.append("insurance.health_cover:required")
    return errors


def _risk_for_goal(goal_type: str, years: int) -> str:
    goal_type = goal_type.replace("childs_", "child_").replace("grandchilds_", "grandchild_")
    if goal_type in {"old_age", "old_age_care", "old_age_healthcare", "healthcare"}:
        return "aggressive_medium"
    if goal_type in {"vacation", "home_renovation", "renovation"}:
        return "medium"
    if goal_type == "legacy":
        return "high"
    if goal_type in {"business", "business_fund"}:
        return "medium" if years < 3 else "high"
    if goal_type in {"car", "new_car", "vehicle"}:
        return "low" if years < 3 else "medium"
    if years < 3:
        return "low"
    if years <= 5:
        return "medium"
    return "aggressive_medium"


def _asset_row(
    asset_id: str,
    name: str,
    instrument_type: str,
    value: float,
    deployable_amount: float,
    **extra: Any,
) -> Dict[str, Any]:
    deployable_amount = min(value, max(0.0, deployable_amount))
    instrument_rate = max(0.0, _number(extra.get("rate")))
    assumed_tax_rate = ASSUMPTIONS.interest_tax_rate if _uses_interest_tax_assumption(instrument_type) else 0.0
    return {
        "id": asset_id,
        "name": name,
        "instrument_type": instrument_type,
        "current_value": round(value, 2),
        "deployable_amount": round(deployable_amount, 2),
        "retained_amount": round(value - deployable_amount, 2),
        "protected": extra.get("protected", False),
        "held_by": extra.get("held_by", "client"),
        "rate": instrument_rate,
        "assumed_tax_rate": assumed_tax_rate,
        "post_tax_rate": round(instrument_rate * (1 - assumed_tax_rate), 6),
        "goal_id": extra.get("goal_id"),
        "maturity_years": max(0, _integer(extra.get("maturity_years"))),
        "lock_in_years": max(0, _integer(extra.get("lock_in_years"))),
    }


def _legacy_asset_rows(assets: Dict[str, Any]) -> List[Dict[str, Any]]:
    definitions = (
        ("primary_residence", "Primary residence", "primary_residence", False, False),
        ("investment_property", "Investment property", "property", False, False),
        ("scss", "SCSS", "scss", False, True),
        ("fixed_deposits", "Fixed deposits", "fixed_deposit", True, False),
        ("gold_silver", "Gold / silver", "gold", False, False),
        ("insurance_surrender", "Insurance surrender value", "insurance", False, False),
        ("equity_investments", "Equity investments", "equity", True, False),
        ("debt_investments", "Debt investments", "debt", True, False),
        ("liquid_savings", "Liquid savings", "cash", True, False),
        ("pension_lump_sum", "Retirement proceeds", "retirement_proceeds", True, False),
    )
    rows: List[Dict[str, Any]] = []
    for key, name, instrument_type, deployable_default, protected in definitions:
        value = _money(assets.get(key))
        if value <= 0:
            continue
        deployable = assets.get(f"{key}_deployable", deployable_default) is True
        rate = ASSUMPTIONS.scss_rate if key == "scss" else ASSUMPTIONS.fd_rate if key == "fixed_deposits" else 0.0
        rows.append(
            _asset_row(
                key,
                name,
                instrument_type,
                value,
                value if deployable else 0.0,
                protected=protected,
                rate=rate,
            )
        )
    for index, item in enumerate(_records(assets.get("other_investments"))):
        value = _money(item.get("amount", item.get("current_value")))
        if value <= 0:
            continue
        explicit = item.get("deployable_amount")
        deployable_amount = (
            min(value, _money(explicit))
            if explicit is not None
            else value if item.get("deployable", True) is True else 0.0
        )
        rows.append(
            _asset_row(
                str(item.get("id") or f"other_{index + 1}"),
                str(item.get("name") or f"Other investment {index + 1}").strip(),
                _slug(item.get("instrument_type"), "other"),
                value,
                deployable_amount,
                protected=item.get("protected") is True,
                held_by=str(item.get("held_by") or "client"),
                rate=item.get("rate"),
                goal_id=item.get("goal_id"),
                maturity_years=item.get("maturity_years"),
                lock_in_years=item.get("lock_in_years"),
            )
        )
    return rows


def _normalized_assets(assets: Dict[str, Any]) -> List[Dict[str, Any]]:
    explicit = _records(assets.get("holdings"))
    if not explicit:
        return _legacy_asset_rows(assets)
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(explicit):
        value = _money(item.get("current_value", item.get("amount")))
        if value <= 0:
            continue
        default_deployable = item.get("deployable", True) is True
        deployable_amount = min(value, _money(item.get("deployable_amount", value if default_deployable else 0.0)))
        rows.append(
            _asset_row(
                str(item.get("id") or f"holding_{index + 1}"),
                str(item.get("name") or f"Holding {index + 1}").strip(),
                _slug(item.get("instrument_type"), "other"),
                value,
                deployable_amount,
                protected=item.get("protected") is True,
                held_by=str(item.get("held_by") or "client"),
                rate=item.get("rate"),
                goal_id=item.get("goal_id"),
                maturity_years=item.get("maturity_years"),
                lock_in_years=item.get("lock_in_years"),
            )
        )
    return rows


def _normalized_income(income: Dict[str, Any], assets: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    exclusions: List[str] = []
    fixed_sources = (
        ("government_pension", "Government pension", True),
        ("employer_pension", "Employer pension", False),
        ("family_pension", "Family pension", False),
        ("nps_annuity", "NPS annuity", False),
        ("other_annuity", "Other annuity", False),
        ("rental_income", "Rental income", False),
        ("spouse_pension", "Spouse pension", False),
    )
    for key, name, indexed_default in fixed_sources:
        amount = _money(income.get(key))
        if amount <= 0:
            continue
        rows.append(
            {
                "id": key,
                "name": name,
                "monthly_amount": amount,
                "nature": "lifelong",
                "years_remaining": 0,
                "indexed": income.get(f"{key}_indexed", indexed_default) is True,
                "index_rate": max(0.0, _number(income.get(f"{key}_index_rate"), ASSUMPTIONS.pension_indexation)),
                "linked_asset_id": None,
            }
        )
    for index, item in enumerate(_records(income.get("other_sources"))):
        amount = _money(item.get("monthly_amount"))
        if amount <= 0:
            continue
        years = max(0, _integer(item.get("years_remaining")))
        rows.append(
            {
                "id": str(item.get("id") or f"income_{index + 1}"),
                "name": str(item.get("name") or f"Other income {index + 1}").strip(),
                "monthly_amount": amount,
                "nature": _slug(item.get("nature"), "fixed_term" if years > 0 else "lifelong"),
                "years_remaining": years,
                "indexed": item.get("indexed_to_inflation", item.get("indexed", False)) is True,
                "index_rate": max(0.0, _number(item.get("index_rate"), ASSUMPTIONS.pension_indexation)),
                "linked_asset_id": item.get("linked_asset_id"),
            }
        )
    assets_by_id = {item["id"]: item for item in assets}
    included: List[Dict[str, Any]] = []
    linked_assets: set[str] = set()
    for row in rows:
        linked_id = str(row.get("linked_asset_id") or "")
        if linked_id:
            linked_assets.add(linked_id)
            asset = assets_by_id.get(linked_id)
            if asset and asset["retained_amount"] <= 0:
                exclusions.append(f"{row['name']} excluded because {asset['name']} is fully deployable")
                continue
            if asset and _uses_interest_tax_assumption(asset["instrument_type"]):
                gross_amount = row["monthly_amount"]
                row["gross_monthly_amount"] = gross_amount
                row["assumed_tax_rate"] = ASSUMPTIONS.interest_tax_rate
                row["monthly_amount"] = round(gross_amount * (1 - ASSUMPTIONS.interest_tax_rate), 2)
            else:
                row["gross_monthly_amount"] = row["monthly_amount"]
                row["assumed_tax_rate"] = 0.0
        else:
            row["gross_monthly_amount"] = row["monthly_amount"]
            row["assumed_tax_rate"] = 0.0
        included.append(row)
    for asset in assets:
        if asset["id"] in linked_assets or asset["retained_amount"] <= 0:
            continue
        if asset["instrument_type"] not in {"scss", "fixed_deposit", "fd"}:
            continue
        rate = asset["rate"] or (ASSUMPTIONS.scss_rate if asset["instrument_type"] == "scss" else ASSUMPTIONS.fd_rate)
        gross_monthly = round(asset["retained_amount"] * rate / 12, 2)
        assumed_tax_rate = ASSUMPTIONS.interest_tax_rate
        included.append(
            {
                "id": f"asset_income_{asset['id']}",
                "name": f"{asset['name']} interest",
                "gross_monthly_amount": gross_monthly,
                "monthly_amount": round(gross_monthly * (1 - assumed_tax_rate), 2),
                "assumed_tax_rate": assumed_tax_rate,
                "nature": "lifelong",
                "years_remaining": 0,
                "indexed": False,
                "index_rate": 0.0,
                "linked_asset_id": asset["id"],
            }
        )
    return included, exclusions


def _normalized_liabilities(payload: Dict[str, Any], assets: Dict[str, Any], expenses: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(_records(payload.get("liabilities"))):
        outstanding = _money(item.get("outstanding"))
        emi = _money(item.get("monthly_emi", item.get("emi")))
        if outstanding <= 0 and emi <= 0:
            continue
        rows.append(
            {
                "id": str(item.get("id") or f"liability_{index + 1}"),
                "name": str(item.get("name") or item.get("type") or f"Liability {index + 1}").strip(),
                "outstanding": outstanding,
                "monthly_emi": emi,
                "months_remaining": max(0, _integer(item.get("months_remaining"))),
                "rate": max(0.0, _number(item.get("rate"))),
                "treatment": _slug(item.get("treatment"), "settle"),
            }
        )
    legacy_outstanding = _money(assets.get("outstanding_liabilities"))
    legacy_emi = _money(expenses.get("monthly_emi"))
    if not rows and (legacy_outstanding > 0 or legacy_emi > 0):
        rows.append(
            {
                "id": "legacy_liability",
                "name": "Recorded liabilities",
                "outstanding": legacy_outstanding,
                "monthly_emi": legacy_emi,
                "months_remaining": max(0, _integer(expenses.get("monthly_emi_months"))),
                "rate": 0.0,
                "treatment": _slug(assets.get("liability_treatment"), "settle"),
            }
        )
    return rows


def _income_for_year(row: Dict[str, Any], year: int) -> float:
    amount = row["monthly_amount"]
    nature = row["nature"]
    years = row["years_remaining"]
    if nature == "fixed_term" and years > 0 and year >= years:
        return 0.0
    if nature == "reducing" and years > 0:
        amount *= max(0.0, 1 - year / years)
    if row["indexed"]:
        amount *= (1 + row["index_rate"]) ** year
    return round(amount, 2)


def _emi_for_year(liability: Dict[str, Any], year: int, planning_years: int) -> float:
    if liability["treatment"] == "settle":
        return 0.0
    months = liability["months_remaining"] or planning_years * 12
    return liability["monthly_emi"] if year * 12 < months else 0.0


def _normalize_goals(goals_input: List[Dict[str, Any]], assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    assigned_by_goal: Dict[str, float] = {}
    for asset in assets:
        goal_id = str(asset.get("goal_id") or "")
        if goal_id:
            assigned_by_goal[goal_id] = assigned_by_goal.get(goal_id, 0.0) + asset["deployable_amount"]
    goals: List[Dict[str, Any]] = []
    for index, item in enumerate(goals_input):
        if item.get("active", True) is False:
            continue
        goal_id = str(item.get("id") or f"goal_{index + 1}")
        goal_type = _slug(item.get("type"), "other")
        years = max(1, _integer(item.get("years_from_now"), 1))
        buffer_years = max(1, _integer(item.get("buffer_years"), 5))
        annual_amount = _money(item.get("annual_amount"))
        nominal_requirement = (
            round(annual_amount * buffer_years, 2)
            if goal_type == "vacation" and annual_amount > 0
            else _money(item.get("target_amount"))
        )
        risk_key = _risk_for_goal(goal_type, years)
        risk = RISK_CATEGORIES[risk_key]
        corpus_today = round(nominal_requirement / ((1 + risk["return"]) ** years), 2)
        assigned = min(corpus_today, assigned_by_goal.get(goal_id, 0.0))
        goals.append(
            {
                "id": goal_id,
                "name": str(item.get("name") or "Goal").strip(),
                "type": goal_type,
                "active": True,
                "years_from_now": years,
                "nominal_requirement": nominal_requirement,
                "annual_amount": annual_amount,
                "buffer_years": buffer_years if goal_type == "vacation" else None,
                "corpus_today": corpus_today,
                "assigned_existing_corpus": round(assigned, 2),
                "fresh_corpus_needed": round(max(0.0, corpus_today - assigned), 2),
                "risk_key": risk_key,
                "risk_category": risk["label"],
                "fund_type": risk["fund_type"],
                "expected_return": risk["return"],
            }
        )
    return sorted(goals, key=lambda goal: (goal["years_from_now"], goal["name"]))


def _solve_rate(annual_gap: float, income_capacity: float) -> float | None:
    if annual_gap <= 0:
        return 0.0
    if income_capacity <= 0:
        return None
    return annual_gap / income_capacity


def analyze_retirement(payload: Dict[str, Any]) -> Dict[str, Any]:
    errors = validate_retirement_input(payload)
    if errors:
        raise ValueError(";".join(errors))
    profile = _mapping(payload.get("profile"))
    assets_input = _mapping(payload.get("assets"))
    income_input = _mapping(payload.get("income"))
    expenses = _mapping(payload.get("expenses"))
    insurance = _mapping(payload.get("insurance"))
    planning = _mapping(payload.get("planning"))
    dependents = _records(payload.get("dependents"))

    age = _integer(profile.get("age"))
    planning_age = _integer(profile.get("planning_age"), ASSUMPTIONS.planning_age)
    planning_years = max(1, planning_age - age)
    expense_inflation = max(0.0, _number(planning.get("expense_inflation"), ASSUMPTIONS.expense_inflation))
    assets = _normalized_assets(assets_input)
    income_rows, income_exclusions = _normalized_income(income_input, assets)
    liabilities = _normalized_liabilities(payload, assets_input, expenses)

    total_assets = round(sum(item["current_value"] for item in assets), 2)
    total_liabilities = round(sum(item["outstanding"] for item in liabilities), 2)
    settle_liabilities = round(sum(item["outstanding"] for item in liabilities if item["treatment"] == "settle"), 2)
    gross_deployable = round(sum(item["deployable_amount"] for item in assets), 2)
    available_corpus = round(max(0.0, gross_deployable - settle_liabilities), 2)
    net_worth = round(max(0.0, total_assets - total_liabilities), 2)

    monthly_core = _money(expenses.get("monthly_core"))
    annual_items = _records(expenses.get("annual_items"))
    annual_item_expenses = round(sum(_money(item.get("amount")) for item in annual_items), 2)
    annual_term_premium = _money(insurance.get("annual_term_premium"))
    annual_motor_premium = _money(insurance.get("annual_motor_premium"))
    annual_insurance_expenses = round(annual_term_premium + annual_motor_premium, 2)
    annual_expenses = round(annual_item_expenses + annual_insurance_expenses, 2)
    dependent_cost = round(sum(_money(item.get("monthly_cost")) for item in dependents), 2)
    effective_monthly = round(monthly_core + annual_expenses / 12 + dependent_cost, 2)
    timeline: List[Dict[str, Any]] = []
    for year in range(planning_years):
        expense_for_year = effective_monthly * ((1 + expense_inflation) ** year)
        income_for_year = sum(_income_for_year(row, year) for row in income_rows)
        emi_for_year = sum(_emi_for_year(item, year, planning_years) for item in liabilities)
        gap = max(0.0, expense_for_year + emi_for_year - income_for_year)
        timeline.append(
            {
                "year": year,
                "age": age + year,
                "monthly_expense": round(expense_for_year, 2),
                "monthly_income": round(income_for_year, 2),
                "monthly_emi": round(emi_for_year, 2),
                "monthly_gap": round(gap, 2),
            }
        )
    design_row = max(timeline, key=lambda row: row["monthly_gap"])
    current_gap = timeline[0]["monthly_gap"]
    design_gap = design_row["monthly_gap"]

    annual_health_premium = _money(insurance.get("annual_health_premium"))
    emergency_months = min(6, max(3, _integer(planning.get("emergency_months"), ASSUMPTIONS.emergency_months)))
    emergency_reserve = round(effective_monthly * emergency_months, 2)
    premium_reserve = round(annual_health_premium * ASSUMPTIONS.premium_reserve_multiple, 2)
    opportunity_enabled = planning.get("opportunity_enabled", True) is not False
    opportunity_pct = _bounded(planning.get("opportunity_pct"), 0.0, 0.10, ASSUMPTIONS.opportunity_default_pct)
    if not opportunity_enabled:
        opportunity_pct = 0.0
    opportunity_bucket = round(available_corpus * opportunity_pct, 2)

    goals = _normalize_goals(_records(payload.get("goals")), assets)
    other_goal_corpus = round(sum(goal["corpus_today"] for goal in goals), 2)
    income_capacity = round(available_corpus - emergency_reserve - premium_reserve - opportunity_bucket - other_goal_corpus, 2)
    annual_design_gap = round(design_gap * 12, 2)
    natural_rate = _solve_rate(annual_design_gap, income_capacity)
    floor = ASSUMPTIONS.withdrawal_floor
    cap = ASSUMPTIONS.withdrawal_cap
    if annual_design_gap <= 0:
        status, minimum_rate, default_rate = "no_income_required", 0.0, 0.0
    elif natural_rate is None or natural_rate > cap:
        status, minimum_rate, default_rate = "infeasible", natural_rate, cap
    else:
        status = "feasible_with_surplus" if natural_rate < floor else "feasible"
        minimum_rate = max(floor, natural_rate)
        default_rate = minimum_rate
    selected_raw = planning.get("selected_withdrawal_rate")
    if annual_design_gap <= 0:
        selected_rate = 0.0
    elif selected_raw is None:
        selected_rate = default_rate
    else:
        selected_rate = _bounded(selected_raw, floor, cap, default_rate)
    selected_income_corpus = round(annual_design_gap / selected_rate, 2) if annual_design_gap > 0 and selected_rate > 0 else 0.0
    selected_shortfall = round(max(0.0, selected_income_corpus - max(0.0, income_capacity)), 2)
    legacy_residual = round(max(0.0, income_capacity - selected_income_corpus), 2)
    selected_feasible = selected_shortfall <= 0.01
    cap_required_corpus = round(annual_design_gap / cap, 2) if annual_design_gap > 0 else 0.0
    shortfall_at_cap = round(max(0.0, cap_required_corpus - max(0.0, income_capacity)), 2)

    baseline_rate = natural_rate if natural_rate is not None else 1.0
    for goal in goals:
        rate_without = _solve_rate(annual_design_gap, income_capacity + goal["corpus_today"])
        display_without = 0.0 if rate_without == 0 else max(floor, rate_without or cap)
        goal["rate_without_goal"] = round(display_without, 4)
        goal["rate_impact"] = round(max(0.0, baseline_rate - (rate_without or 0.0)), 4)

    risk_allocations: Dict[str, float] = {key: 0.0 for key in RISK_CATEGORIES}
    risk_allocations["no_risk"] += emergency_reserve + premium_reserve
    risk_allocations["low"] += opportunity_bucket
    risk_allocations["aggressive_medium"] += selected_income_corpus
    risk_allocations["high"] += legacy_residual
    for goal in goals:
        risk_allocations[goal["risk_key"]] += goal["corpus_today"]
    by_risk = []
    allocated_total = sum(risk_allocations.values())
    for key, value in risk_allocations.items():
        if value <= 0:
            continue
        risk = RISK_CATEGORIES[key]
        by_risk.append(
            {
                "risk_key": key,
                "category": risk["label"],
                "fund_type": risk["fund_type"],
                "expected_return": risk["return"],
                "amount": round(value, 2),
                "percentage": round(value / allocated_total * 100, 1) if allocated_total else 0.0,
            }
        )
    blended_return = round(sum(item["amount"] * item["expected_return"] for item in by_risk) / allocated_total, 4) if allocated_total else 0.0

    flags: List[Dict[str, Any]] = []

    def flag(priority: str, code: str, message: str, action: str, value: float = 0.0) -> None:
        flags.append({"priority": priority, "code": code, "message": message, "action": action, "value": round(value, 2)})

    if status == "infeasible" or not selected_feasible:
        rate_text = "undefined" if natural_rate is None else f"{natural_rate * 100:.2f}%"
        flag("critical", "plan_infeasible", f"The settled goals require a {rate_text} withdrawal rate, above the 7% cap.", "Discuss resizing or deferring a goal, reducing expenses, adding corpus or extending earned income.", shortfall_at_cap)
    if design_gap > current_gap + 1:
        flag("critical", "design_gap_step_up", f"The monthly gap rises from Rs {current_gap:,.0f} to Rs {design_gap:,.0f} in year {design_row['year']}.", "Build the income bucket on the design gap, not only the plan-date gap.", design_gap - current_gap)
    if profile.get("will_in_place") is not True:
        flag("high", "will_missing", "A current will is not recorded.", "Complete a will and nomination review with a qualified lawyer.")
    has_health = insurance.get("has_health_insurance") is True
    if not has_health:
        flag("critical", "health_missing", "No health insurance is recorded.", "Complete an adviser-led health-cover review before deployment.")
    if dependents and insurance.get("has_term_insurance") is not True:
        flag("high", "term_missing", "Dependants are recorded but no term cover is available.", "Review whether term cover remains necessary for outstanding obligations.")
    retained_fd = sum(item["retained_amount"] for item in assets if item["instrument_type"] in {"fixed_deposit", "fd"})
    if retained_fd > 0:
        flag("high", "fd_tax_review", "Retained fixed-deposit income is reduced by the agreed 22% tax assumption.", "Review whether the retained FD remains appropriate after the assumed post-tax return.", retained_fd)
    sip_monthly = _money(assets_input.get("existing_sip_monthly"))
    sip_count = max(0, _integer(assets_input.get("existing_sip_count")))
    if sip_monthly > 0 or sip_count > 0:
        flag("high", "sip_review", "Existing SIPs are still recorded for this retirement client.", "Review and stop accumulation SIPs when retirement deployment begins.", sip_monthly)
    if opportunity_bucket <= 0:
        flag("maintain", "opportunity_declined", "No market-opportunity bucket is included.", "Confirm that the omission is intentional and record the adviser decision.")
    for message in income_exclusions:
        flag("maintain", "asset_income_excluded", message, "Retain either the asset income or the deployable corpus, never both.")
    pension_monthly = sum(row["monthly_amount"] for row in income_rows if "pension" in row["name"].lower() or "annuity" in row["name"].lower())
    serviced_emi = sum(item["monthly_emi"] for item in liabilities if item["treatment"] != "settle")
    if serviced_emi > 0 and abs(pension_monthly - serviced_emi) / serviced_emi < 0.10:
        flag("maintain", "pension_emi_match", "Monthly pension closely matches the serviced EMI.", "Consider aligning the EMI debit date immediately after pension credit.")
    priority_order = {"critical": 0, "high": 1, "maintain": 2}
    flags.sort(key=lambda item: (priority_order.get(item["priority"], 9), -item["value"]))

    reconciliation = [
        {"label": "Assets available for deployment", "amount": gross_deployable, "operation": "add"},
        {"label": "Liabilities settled", "amount": settle_liabilities, "operation": "subtract"},
        {"label": "Emergency fund", "amount": emergency_reserve, "operation": "subtract"},
        {"label": "Insurance premium reserve", "amount": premium_reserve, "operation": "subtract"},
        {"label": "Opportunity bucket", "amount": opportunity_bucket, "operation": "subtract"},
        {"label": "Other goal corpora", "amount": other_goal_corpus, "operation": "subtract"},
        {"label": "Income bucket", "amount": selected_income_corpus, "operation": "subtract"},
        {"label": "Legacy residual", "amount": legacy_residual, "operation": "balance"},
    ]
    actionables = []
    if available_corpus > 0:
        actionables.append({"type": "fresh_deployment", "amount": available_corpus, "description": "Deploy the settled retirement corpus by goal and risk category."})
    if retained_fd > 0:
        actionables.append({"type": "restructure", "amount": retained_fd, "description": "Review retained fixed deposits for category-level tax drag and goal fit."})
    if not has_health:
        actionables.append({"type": "insurance", "amount": 0.0, "description": "Complete the health-cover review."})
    if sip_monthly > 0 or sip_count > 0:
        actionables.append({"type": "sip_stop", "amount": sip_monthly, "description": "Review accumulation SIPs at retirement deployment."})

    return {
        "module": "retirement",
        "profile": {"client_name": str(profile.get("client_name") or "").strip(), "pan": str(profile.get("pan") or "").strip().upper(), "age": age, "planning_age": planning_age, "planning_years": planning_years, "will_in_place": profile.get("will_in_place") is True},
        "net_worth": {"holdings": assets, "total_assets": total_assets, "total_liabilities": total_liabilities, "net_worth": net_worth, "gross_deployable": gross_deployable, "settle_liabilities": settle_liabilities, "available_corpus": available_corpus},
        "cashflow": {"monthly_core_expense": monthly_core, "annual_item_expenses": annual_item_expenses, "annual_insurance_expenses": annual_insurance_expenses, "annual_expenses": annual_expenses, "annual_items": annual_items, "dependent_cost": dependent_cost, "effective_monthly_expense": effective_monthly, "income_sources": income_rows, "income_exclusions": income_exclusions, "liabilities": liabilities, "current_gap": current_gap, "design_gap": design_gap, "design_gap_year": design_row["year"], "design_gap_age": design_row["age"], "timeline": timeline},
        "reserves": {"emergency_months": emergency_months, "emergency_fund": emergency_reserve, "annual_health_premium": annual_health_premium, "premium_multiple": ASSUMPTIONS.premium_reserve_multiple, "premium_reserve": premium_reserve, "opportunity_pct": opportunity_pct, "opportunity_bucket": opportunity_bucket},
        "goals": goals,
        "solver": {"status": status, "current_gap": current_gap, "design_gap": design_gap, "design_gap_year": design_row["year"], "annual_design_gap": annual_design_gap, "income_capacity": income_capacity, "natural_rate": round(natural_rate, 4) if natural_rate is not None else None, "minimum_feasible_rate": round(minimum_rate, 4) if minimum_rate is not None else None, "selected_rate": round(selected_rate, 4), "withdrawal_floor": floor, "withdrawal_cap": cap, "selected_feasible": selected_feasible, "income_bucket": selected_income_corpus, "monthly_swp": design_gap, "shortfall_at_selected_rate": selected_shortfall, "shortfall_at_cap": shortfall_at_cap, "legacy_residual": legacy_residual},
        "allocation": {"reconciliation": reconciliation, "by_risk": by_risk, "blended_return": blended_return, "allocated_total": round(allocated_total, 2)},
        "protection": {"has_health_insurance": has_health, "health_cover": _money(insurance.get("health_cover")), "spouse_covered": insurance.get("spouse_covered") is True, "annual_health_premium": annual_health_premium, "has_term_insurance": insurance.get("has_term_insurance") is True, "term_cover": _money(insurance.get("term_cover")), "annual_term_premium": annual_term_premium, "annual_motor_premium": annual_motor_premium},
        "decision": {"adviser_accepted": retirement_plan_is_accepted(payload)},
        "flags": flags,
        "actionables": actionables,
        "assumptions": {
            **asdict(ASSUMPTIONS),
            "planning_age": planning_age,
            "expense_inflation": expense_inflation,
            "emergency_months": emergency_months,
            "opportunity_selected_pct": opportunity_pct,
            "risk_categories": RISK_CATEGORIES,
            "goal_amounts_inflated": False,
        },
    }


def dashboard_action_items(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    dimension_map = {
        "plan_infeasible": "retirement_income",
        "design_gap_step_up": "retirement_income",
        "will_missing": "estate",
        "health_missing": "protection",
        "term_missing": "protection",
        "fd_tax_review": "portfolio",
        "sip_review": "portfolio",
        "opportunity_declined": "portfolio",
        "asset_income_excluded": "portfolio",
        "pension_emi_match": "cashflow",
    }
    items = []
    for index, item in enumerate(analysis.get("flags") or []):
        code = item.get("code") or f"retirement_{index + 1}"
        items.append(
            {
                "item_id": f"retirement_{code}_{index + 1}",
                "dimension": dimension_map.get(code, "retirement"),
                "urgency": "IMMEDIATE" if item.get("priority") == "critical" else "HIGH" if item.get("priority") == "high" else "MAINTAIN",
                "value_type": "INR" if _money(item.get("value")) > 0 else "NONE",
                "value_num": _money(item.get("value")),
                "description": item.get("action"),
                "final_status": "PENDING",
                "is_converted": False,
            }
        )
    return items
