"""Generate fictional, internally reconciled retirement-module sample reports."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import fitz

BACKEND_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = BACKEND_DIR.parent
OUTPUT_DIR = WORKSPACE_DIR / "retirement module" / "sample reports"
sys.path.insert(0, str(BACKEND_DIR))

from retirement_engine import analyze_retirement  # noqa: E402
from retirement_report import generate_retirement_pdf  # noqa: E402


def _holding(
    holding_id: str,
    name: str,
    instrument_type: str,
    value: float,
    deployable: float = 0,
    rate: float = 0,
) -> dict[str, Any]:
    return {
        "id": holding_id,
        "name": name,
        "instrument_type": instrument_type,
        "current_value": value,
        "deployable_amount": deployable,
        "rate": rate,
    }


def _base_profile(name: str, pan: str, age: int) -> dict[str, Any]:
    return {
        "profile": {
            "client_name": name,
            "pan": pan,
            "age": age,
            "planning_age": 85,
            "will_in_place": True,
        },
        "dependents": [],
        "liabilities": [],
        "planning": {
            "expense_inflation": 0.05,
            "emergency_months": 6,
            "opportunity_enabled": True,
            "opportunity_pct": 0.05,
            "selected_withdrawal_rate": 0.035,
            "adviser_accepted": True,
        },
    }


def _comfortable_plan() -> dict[str, Any]:
    payload = _base_profile("Sample Client - Stable Pension", "SAMPL1001A", 61)
    payload.update(
        {
            "assets": {
                "holdings": [
                    _holding("home", "Primary residence", "primary_residence", 20_000_000),
                    _holding("scss", "SCSS holding", "scss", 3_000_000, rate=0.082),
                    _holding("fd", "Retained fixed deposits", "fixed_deposit", 2_500_000, rate=0.075),
                    _holding("proceeds", "Retirement proceeds", "retirement_proceeds", 1, 1),
                ]
            },
            "income": {
                "government_pension": 55_000,
                "government_pension_indexed": True,
                "rental_income": 25_000,
            },
            "expenses": {
                "monthly_core": 70_000,
                "annual_items": [
                    {"name": "Property tax and maintenance", "amount": 120_000},
                    {"name": "Travel", "amount": 180_000},
                ],
            },
            "insurance": {
                "has_health_insurance": True,
                "health_cover": 2_500_000,
                "spouse_covered": True,
                "annual_health_premium": 45_000,
                "has_term_insurance": True,
                "term_cover": 5_000_000,
                "annual_term_premium": 22_000,
                "annual_motor_premium": 15_000,
            },
            "goals": [
                {"id": "vacation", "name": "Annual vacations", "type": "vacation", "annual_amount": 200_000, "buffer_years": 5, "years_from_now": 1},
                {"id": "car", "name": "Replacement car", "type": "vehicle", "target_amount": 1_500_000, "years_from_now": 4},
                {"id": "legacy", "name": "Family legacy", "type": "legacy", "target_amount": 2_500_000, "years_from_now": 15},
            ],
        }
    )
    payload["planning"].update({"opportunity_pct": 0.08, "selected_withdrawal_rate": 0.035})
    return _calibrate(payload, natural_rate=0.032)


def _goal_heavy_plan() -> dict[str, Any]:
    payload = _base_profile("Sample Client - Goal Review", "SAMPL1002B", 64)
    payload["profile"]["will_in_place"] = False
    payload.update(
        {
            "assets": {
                "holdings": [
                    _holding("home", "Primary residence", "primary_residence", 18_000_000),
                    _holding("fd", "Retained fixed deposits", "fixed_deposit", 4_000_000, rate=0.075),
                    _holding("equity", "Equity investments", "equity", 3_000_000, 3_000_000),
                    _holding("proceeds", "Retirement proceeds", "retirement_proceeds", 1, 1),
                ],
                "existing_sip_count": 3,
                "existing_sip_monthly": 25_000,
            },
            "income": {
                "employer_pension": 65_000,
                "other_sources": [
                    {"id": "consulting", "name": "Advisory income", "monthly_amount": 40_000, "nature": "fixed_term", "years_remaining": 3}
                ],
            },
            "expenses": {
                "monthly_core": 95_000,
                "annual_items": [
                    {"name": "Property and vehicle costs", "amount": 180_000},
                    {"name": "Travel", "amount": 240_000},
                ],
            },
            "insurance": {
                "has_health_insurance": True,
                "health_cover": 2_000_000,
                "spouse_covered": True,
                "annual_health_premium": 52_000,
                "has_term_insurance": True,
                "term_cover": 4_000_000,
                "annual_term_premium": 24_000,
                "annual_motor_premium": 18_000,
            },
            "goals": [
                {"id": "wedding", "name": "Child wedding", "type": "wedding", "target_amount": 4_000_000, "years_from_now": 4},
                {"id": "education", "name": "Grandchild education", "type": "education", "target_amount": 2_500_000, "years_from_now": 7},
                {"id": "business", "name": "Family business fund", "type": "business", "target_amount": 2_000_000, "years_from_now": 8},
                {"id": "care", "name": "Old-age care reserve", "type": "old_age_healthcare", "target_amount": 3_000_000, "years_from_now": 10},
            ],
        }
    )
    payload["planning"].update({"emergency_months": 4, "selected_withdrawal_rate": 0.065})
    return _calibrate(payload, natural_rate=0.062)


def _income_step_down_plan() -> dict[str, Any]:
    payload = _base_profile("Sample Client - Income Step-down", "SAMPL1003C", 60)
    payload.update(
        {
            "assets": {
                "holdings": [
                    _holding("home", "Primary residence", "primary_residence", 15_000_000),
                    _holding("split_fd", "Partially deployed FD", "fixed_deposit", 4_000_000, 1_500_000, 0.075),
                    _holding("debt", "Debt investments", "debt", 2_000_000, 2_000_000),
                    _holding("proceeds", "Retirement proceeds", "retirement_proceeds", 1, 1),
                ]
            },
            "income": {
                "employer_pension": 30_000,
                "other_sources": [
                    {"id": "consulting", "name": "Consulting income", "monthly_amount": 80_000, "nature": "fixed_term", "years_remaining": 5},
                    {"id": "fd_income", "name": "Retained FD interest", "monthly_amount": 15_625, "nature": "lifelong", "linked_asset_id": "split_fd"},
                ],
            },
            "expenses": {
                "monthly_core": 80_000,
                "annual_items": [
                    {"name": "Property tax and repairs", "amount": 150_000},
                    {"name": "Travel", "amount": 150_000},
                ],
            },
            "liabilities": [
                {"id": "home_loan", "name": "Home loan", "outstanding": 1_600_000, "monthly_emi": 30_000, "months_remaining": 60, "treatment": "service"}
            ],
            "insurance": {
                "has_health_insurance": True,
                "health_cover": 3_000_000,
                "spouse_covered": True,
                "annual_health_premium": 58_000,
                "has_term_insurance": True,
                "term_cover": 6_000_000,
                "annual_term_premium": 28_000,
                "annual_motor_premium": 17_000,
            },
            "goals": [
                {"id": "renovation", "name": "Home renovation", "type": "home_renovation", "target_amount": 2_000_000, "years_from_now": 5},
                {"id": "care", "name": "Long-term care reserve", "type": "old_age_healthcare", "target_amount": 2_500_000, "years_from_now": 12},
            ],
        }
    )
    payload["planning"].update({"expense_inflation": 0.04, "emergency_months": 3, "opportunity_pct": 0.10, "selected_withdrawal_rate": 0.055})
    return _calibrate(payload, natural_rate=0.052)


def _calibrate(payload: dict[str, Any], natural_rate: float) -> dict[str, Any]:
    calibrated = copy.deepcopy(payload)
    first = analyze_retirement(calibrated)
    opportunity_pct = calibrated["planning"]["opportunity_pct"] if calibrated["planning"]["opportunity_enabled"] else 0
    fixed_needs = (
        first["reserves"]["emergency_fund"]
        + first["reserves"]["premium_reserve"]
        + sum(goal["corpus_today"] for goal in first["goals"])
    )
    target_available = (first["solver"]["annual_design_gap"] / natural_rate + fixed_needs) / (1 - opportunity_pct)
    holdings = calibrated["assets"]["holdings"]
    other_deployable = sum(item["deployable_amount"] for item in holdings if item["id"] != "proceeds")
    settled_liabilities = sum(
        item.get("outstanding", 0)
        for item in calibrated.get("liabilities", [])
        if item.get("treatment", "settle") == "settle"
    )
    proceeds = next(item for item in holdings if item["id"] == "proceeds")
    proceeds["current_value"] = round(target_available + settled_liabilities - other_deployable, 2)
    proceeds["deployable_amount"] = proceeds["current_value"]
    result = analyze_retirement(calibrated)
    if not result["solver"]["selected_feasible"]:
        raise RuntimeError(f"Sample plan is not feasible: {result['profile']['client_name']}")
    return calibrated


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = [
        ("01_comfortable_plan.pdf", _comfortable_plan()),
        ("02_goal_heavy_plan.pdf", _goal_heavy_plan()),
        ("03_income_step_down_plan.pdf", _income_step_down_plan()),
    ]
    manifest = []
    for filename, payload in scenarios:
        analysis = analyze_retirement(payload)
        output_path = OUTPUT_DIR / filename
        generate_retirement_pdf(analysis, str(output_path), logo_path=str(BACKEND_DIR / "logo.png"))
        with fitz.open(output_path) as document:
            pages = len(document)
            text = "\n".join(page.get_text() for page in document)
        if not 6 <= pages <= 8:
            raise RuntimeError(f"{filename} generated {pages} pages")
        required_text = ["Adviser decision", "Accepted for report generation", "22% assumed tax", "Health premium reserve"]
        missing = [item for item in required_text if item not in text]
        if missing:
            raise RuntimeError(f"{filename} is missing expected text: {missing}")
        manifest.append(
            {
                "file": filename,
                "client": analysis["profile"]["client_name"],
                "pages": pages,
                "selected_rate_pct": round(analysis["solver"]["selected_rate"] * 100, 2),
                "natural_rate_pct": round((analysis["solver"]["natural_rate"] or 0) * 100, 2),
                "design_gap": analysis["cashflow"]["design_gap"],
                "available_corpus": analysis["net_worth"]["available_corpus"],
                "flags": [item["code"] for item in analysis["flags"]],
            }
        )
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
