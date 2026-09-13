import os
import sys

import pdfplumber
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from retirement_engine import (
    analyze_retirement,
    dashboard_action_items,
    retirement_plan_is_accepted,
    validate_retirement_input,
)
from retirement_report import generate_retirement_pdf


def retirement_payload(**overrides):
    payload = {
        "profile": {
            "client_name": "Solver Client",
            "pan": "ABCDE1234F",
            "age": 62,
            "planning_age": 70,
            "will_in_place": False,
        },
        "assets": {
            "primary_residence": 20_000_000,
            "scss": 3_000_000,
            "fixed_deposits": 2_000_000,
            "equity_investments": 8_000_000,
            "debt_investments": 5_000_000,
            "liquid_savings": 2_000_000,
            "pension_lump_sum": 8_000_000,
            "outstanding_liabilities": 1_000_000,
            "liability_treatment": "settle",
            "existing_sip_count": 2,
            "existing_sip_monthly": 15_000,
            "other_investments": [
                {"id": "bonds", "name": "Tax-free bonds", "amount": 1_000_000, "deployable": True},
                {"id": "private", "name": "Private holding", "amount": 500_000, "deployable": False},
            ],
        },
        "income": {
            "government_pension": 25_000,
            "government_pension_indexed": False,
            "employer_pension": 10_000,
            "nps_annuity": 5_000,
            "rental_income": 20_000,
            "other_sources": [
                {
                    "id": "consulting",
                    "name": "Consulting",
                    "monthly_amount": 30_000,
                    "nature": "fixed_term",
                    "years_remaining": 3,
                },
                {"id": "royalty", "name": "Royalty", "monthly_amount": 5_000, "nature": "lifelong"},
            ],
        },
        "expenses": {
            "monthly_core": 75_000,
            "annual_items": [
                {"name": "Property tax", "amount": 60_000},
                {"name": "Travel", "amount": 120_000},
            ],
        },
        "dependents": [{"type": "spouse", "name": "Spouse", "monthly_cost": 10_000}],
        "insurance": {
            "has_health_insurance": True,
            "health_cover": 2_000_000,
            "spouse_covered": True,
            "annual_health_premium": 45_000,
            "has_term_insurance": True,
            "term_cover": 5_000_000,
            "annual_term_premium": 20_000,
            "annual_motor_premium": 15_000,
        },
        "goals": [
            {"id": "wedding", "name": "Daughter wedding", "type": "wedding", "target_amount": 2_500_000, "years_from_now": 4},
            {"id": "education", "name": "Grandchild education", "type": "education", "target_amount": 1_500_000, "years_from_now": 7},
        ],
        "planning": {
            "expense_inflation": 0.0,
            "emergency_months": 6,
            "opportunity_enabled": True,
            "opportunity_pct": 0.05,
        },
    }
    payload.update(overrides)
    return payload


def test_deployable_fd_is_not_also_counted_as_income():
    payload = retirement_payload()
    payload["income"]["other_sources"].append(
        {
            "id": "fd_income",
            "name": "FD interest",
            "monthly_amount": 12_500,
            "nature": "lifelong",
            "linked_asset_id": "fixed_deposits",
        }
    )

    result = analyze_retirement(payload)

    names = [row["name"] for row in result["cashflow"]["income_sources"]]
    assert "FD interest" not in names
    assert result["cashflow"]["income_exclusions"] == [
        "FD interest excluded because Fixed deposits is fully deployable"
    ]


def test_partial_asset_deployment_retains_linked_income_after_tax():
    payload = retirement_payload()
    payload["assets"] = {
        "holdings": [
            {
                "id": "split_fd",
                "name": "Split FD",
                "instrument_type": "fixed_deposit",
                "current_value": 2_000_000,
                "deployable_amount": 1_000_000,
                "rate": 0.075,
            },
            {
                "id": "proceeds",
                "name": "Retirement proceeds",
                "instrument_type": "retirement_proceeds",
                "current_value": 30_000_000,
                "deployable_amount": 30_000_000,
            },
        ]
    }
    payload["income"]["other_sources"].append(
        {
            "id": "split_fd_income",
            "name": "Retained FD interest",
            "monthly_amount": 6_250,
            "nature": "lifelong",
            "linked_asset_id": "split_fd",
        }
    )

    result = analyze_retirement(payload)

    names = [row["name"] for row in result["cashflow"]["income_sources"]]
    assert "Retained FD interest" in names
    retained_income = next(row for row in result["cashflow"]["income_sources"] if row["name"] == "Retained FD interest")
    assert retained_income["gross_monthly_amount"] == 6_250
    assert retained_income["monthly_amount"] == 4_875
    assert retained_income["assumed_tax_rate"] == 0.22
    split_fd = next(row for row in result["net_worth"]["holdings"] if row["id"] == "split_fd")
    assert split_fd["deployable_amount"] == 1_000_000
    assert split_fd["retained_amount"] == 1_000_000


def test_design_gap_uses_maximum_year_after_fixed_income_ends():
    payload = retirement_payload()
    payload["profile"]["planning_age"] = 66
    payload["income"] = {
        "employer_pension": 40_000,
        "other_sources": [
            {
                "name": "Consulting",
                "monthly_amount": 30_000,
                "nature": "fixed_term",
                "years_remaining": 2,
            }
        ],
    }
    payload["expenses"] = {"monthly_core": 80_000, "annual_items": []}
    payload["insurance"]["annual_term_premium"] = 0
    payload["insurance"]["annual_motor_premium"] = 0
    payload["dependents"] = []
    payload["assets"] = {
        "holdings": [
            {
                "id": "proceeds",
                "name": "Retirement proceeds",
                "instrument_type": "retirement_proceeds",
                "current_value": 30_000_000,
                "deployable_amount": 30_000_000,
            }
        ]
    }
    payload["planning"]["expense_inflation"] = 0.0

    result = analyze_retirement(payload)

    assert result["cashflow"]["current_gap"] == 10_000
    assert result["cashflow"]["design_gap"] == 40_000
    assert result["cashflow"]["design_gap_year"] == 2


def test_goal_amounts_are_not_inflated_and_follow_type_tenure_mapping():
    result = analyze_retirement(retirement_payload())
    wedding = next(goal for goal in result["goals"] if goal["id"] == "wedding")
    education = next(goal for goal in result["goals"] if goal["id"] == "education")

    assert wedding["nominal_requirement"] == 2_500_000
    assert wedding["risk_category"] == "Medium Risk"
    assert wedding["expected_return"] == 0.09
    assert wedding["corpus_today"] == pytest.approx(2_500_000 / (1.09**4), abs=0.01)
    assert education["nominal_requirement"] == 1_500_000
    assert education["risk_category"] == "Aggressive Medium"
    assert education["expected_return"] == 0.10


def test_only_health_premium_receives_reserve_and_other_premiums_are_expenses():
    result = analyze_retirement(retirement_payload())

    assert result["reserves"]["annual_health_premium"] == 45_000
    assert result["reserves"]["premium_reserve"] == 450_000
    assert result["reserves"]["premium_multiple"] == 10
    assert result["cashflow"]["annual_expenses"] == 215_000
    assert result["cashflow"]["annual_insurance_expenses"] == 35_000
    assert result["cashflow"]["effective_monthly_expense"] == pytest.approx(102_916.67, abs=0.01)


def test_retained_scss_interest_uses_flat_twenty_two_percent_tax_assumption():
    result = analyze_retirement(retirement_payload())
    scss = next(row for row in result["cashflow"]["income_sources"] if row["name"] == "SCSS interest")

    assert scss["gross_monthly_amount"] == 20_500
    assert scss["monthly_amount"] == 15_990
    assert scss["assumed_tax_rate"] == 0.22


def test_client_income_at_or_below_twelve_lakh_has_no_tax_adjustment():
    payload = retirement_payload()
    payload["assets"] = {
        "holdings": [
            {
                "id": "proceeds",
                "name": "Retirement proceeds",
                "instrument_type": "retirement_proceeds",
                "current_value": 30_000_000,
                "deployable_amount": 30_000_000,
            }
        ]
    }
    payload["income"] = {"employer_pension": 100_000}

    result = analyze_retirement(payload)
    pension = result["cashflow"]["income_sources"][0]

    assert result["tax"]["current_client_gross_annual_income"] == 1_200_000
    assert result["tax"]["current_threshold_exceeded"] is False
    assert pension["monthly_amount"] == 100_000
    assert pension["tax_adjustment_monthly"] == 0


def test_above_threshold_adjustment_applies_to_all_client_income_sources():
    payload = retirement_payload()
    payload["assets"] = {
        "holdings": [
            {
                "id": "proceeds",
                "name": "Retirement proceeds",
                "instrument_type": "retirement_proceeds",
                "current_value": 30_000_000,
                "deployable_amount": 30_000_000,
            }
        ]
    }
    payload["income"] = {
        "employer_pension": 80_000,
        "other_sources": [
            {"id": "consulting", "name": "Consulting", "monthly_amount": 30_000, "nature": "lifelong"}
        ],
    }

    result = analyze_retirement(payload)
    sources = {row["id"]: row for row in result["cashflow"]["income_sources"]}

    assert result["tax"]["current_client_gross_annual_income"] == 1_320_000
    assert result["tax"]["current_threshold_exceeded"] is True
    assert sources["employer_pension"]["monthly_amount"] == 62_400
    assert sources["consulting"]["monthly_amount"] == 23_400
    assert result["tax"]["current_estimated_annual_adjustment"] == 290_400


def test_spouse_income_is_separate_from_client_threshold():
    payload = retirement_payload()
    payload["assets"] = {
        "holdings": [
            {
                "id": "proceeds",
                "name": "Retirement proceeds",
                "instrument_type": "retirement_proceeds",
                "current_value": 30_000_000,
                "deployable_amount": 30_000_000,
            }
        ]
    }
    payload["income"] = {"employer_pension": 60_000, "spouse_pension": 100_000}

    result = analyze_retirement(payload)
    sources = {row["id"]: row for row in result["cashflow"]["income_sources"]}

    assert result["tax"]["current_client_gross_annual_income"] == 720_000
    assert result["tax"]["current_threshold_exceeded"] is False
    assert sources["employer_pension"]["monthly_amount"] == 60_000
    assert sources["spouse_pension"]["owner"] == "spouse"
    assert sources["spouse_pension"]["monthly_amount"] == 100_000


def test_adviser_acceptance_is_explicit():
    payload = retirement_payload()
    assert retirement_plan_is_accepted(payload) is False

    payload["planning"]["adviser_accepted"] = True
    assert retirement_plan_is_accepted(payload) is True
    assert analyze_retirement(payload)["decision"]["adviser_accepted"] is True


def test_exact_rate_solver_and_selected_rate_create_legacy_residual():
    payload = retirement_payload()
    baseline = analyze_retirement(payload)
    natural_rate = baseline["solver"]["natural_rate"]

    assert natural_rate is not None
    expected_rate = baseline["solver"]["annual_design_gap"] / baseline["solver"]["income_capacity"]
    assert natural_rate == pytest.approx(expected_rate, abs=0.0001)

    payload["planning"]["selected_withdrawal_rate"] = min(0.07, max(0.06, natural_rate + 0.01))
    adjusted = analyze_retirement(payload)
    assert "planning" not in adjusted
    if payload["planning"]["selected_withdrawal_rate"] >= natural_rate:
        assert adjusted["solver"]["legacy_residual"] >= 0


def test_engine_flags_but_never_removes_or_partially_funds_goals():
    payload = retirement_payload()
    payload["assets"] = {
        "holdings": [
            {
                "id": "small_corpus",
                "name": "Small retirement corpus",
                "instrument_type": "retirement_proceeds",
                "current_value": 1_000_000,
                "deployable_amount": 1_000_000,
            }
        ]
    }
    result = analyze_retirement(payload)

    assert result["solver"]["status"] == "infeasible"
    assert result["solver"]["shortfall_at_cap"] > 0
    assert len(result["goals"]) == len(payload["goals"])
    assert all(goal["corpus_today"] > 0 for goal in result["goals"])
    assert result["flags"][0]["code"] == "plan_infeasible"


def test_diagnostic_score_and_swp_spectrum_are_removed():
    result = analyze_retirement(retirement_payload())

    assert "scores" not in result
    assert "swp" not in result
    assert "corpus_adequacy_pct" not in result["allocation"]
    assert "pension_surplus" not in result["allocation"]


def test_dashboard_items_use_solver_flags():
    payload = retirement_payload()
    payload["insurance"]["has_health_insurance"] = False
    payload["insurance"]["health_cover"] = 0
    result = analyze_retirement(payload)

    items = dashboard_action_items(result)
    assert items
    assert all(item["item_id"].startswith("retirement_") for item in items)
    assert any(item["dimension"] == "protection" for item in items)


def test_validation_reports_stable_field_paths():
    payload = retirement_payload()
    payload["profile"]["pan"] = "bad"
    payload["expenses"]["monthly_core"] = 0
    payload["goals"][0]["target_amount"] = 0

    errors = validate_retirement_input(payload)
    assert "profile.pan:invalid" in errors
    assert "expenses.monthly_core:required" in errors
    assert "goals.0.target_amount:required" in errors


def test_retirement_pdf_is_eight_readable_pages(tmp_path):
    payload = retirement_payload()
    payload["planning"]["adviser_accepted"] = True
    result = analyze_retirement(payload)
    output = tmp_path / "retirement-plan.pdf"
    generate_retirement_pdf(result, str(output))

    assert output.exists() and output.stat().st_size > 10_000
    with pdfplumber.open(output) as pdf:
        assert 6 <= len(pdf.pages) <= 8
        text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    assert "Retirement Advisory Plan" in text
    assert "Current Status" in text
    assert "Corpus Reconciliation" in text
    assert "Accepted for report generation" in text
    assert "Tax Implications" in text
    assert "22% indicative" in text
    assert "Rs 12 lakh test uses only client-owned recurring gross income" in text
    assert "SWP withdrawals are not entirely treated as income" in text
    assert "Term insurance cover" in text
    assert "10 times annual health premium" in text
    assert "Viability Score" not in text
    assert "SWP Adequacy Spectrum" not in text
