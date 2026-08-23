import os
import sys

import pdfplumber
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from retirement_engine import analyze_retirement, dashboard_action_items, validate_retirement_input
from retirement_report import generate_retirement_pdf


def retirement_payload(**overrides):
    payload = {
        "profile": {
            "client_name": "MVP Client",
            "pan": "ABCDE1234F",
            "age": 62,
            "will_in_place": False,
        },
        "assets": {
            "primary_residence": 20_000_000,
            "investment_property": 8_000_000,
            "scss": 3_000_000,
            "fixed_deposits": 2_000_000,
            "gold_silver": 1_500_000,
            "insurance_surrender": 500_000,
            "equity_investments": 7_000_000,
            "debt_investments": 4_000_000,
            "liquid_savings": 1_000_000,
            "pension_lump_sum": 6_000_000,
            "outstanding_liabilities": 1_000_000,
            "other_investments": [
                {"name": "Tax-free bonds", "amount": 1_000_000, "deployable": True},
                {"name": "Private holding", "amount": 500_000, "deployable": False},
            ],
        },
        "income": {
            "government_pension": 25_000,
            "employer_pension": 10_000,
            "nps_annuity": 5_000,
            "other_annuity": 0,
            "rental_income": 20_000,
            "other_sources": [
                {"name": "Consulting", "monthly_amount": 30_000, "years_remaining": 3},
                {"name": "Royalty", "monthly_amount": 5_000, "years_remaining": 0},
            ],
        },
        "expenses": {
            "monthly_core": 75_000,
            "monthly_emi": 10_000,
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
        },
        "goals": [
            {"name": "Daughter wedding", "type": "wedding", "target_amount": 2_500_000, "years_from_now": 4, "inflation_linked": True},
            {"name": "Grandchild education", "type": "education", "target_amount": 1_500_000, "years_from_now": 10, "inflation_linked": True},
        ],
    }
    payload.update(overrides)
    return payload


def test_net_worth_and_usable_corpus_keep_non_deployable_assets_separate():
    result = analyze_retirement(retirement_payload())

    assert result["net_worth"]["net_worth"] == 53_500_000
    assert result["net_worth"]["usable_corpus"] == 21_000_000
    assert result["net_worth"]["protected_scss"] == 3_000_000


def test_income_and_expenses_include_client_requested_sources():
    result = analyze_retirement(retirement_payload())
    cashflow = result["cashflow"]

    assert cashflow["scss_interest"] == pytest.approx(20_500)
    assert cashflow["fd_interest"] == pytest.approx(12_500)
    assert cashflow["permanent_income"] == pytest.approx(98_000)
    assert cashflow["temporary_income"] == 30_000
    assert cashflow["effective_monthly_expense"] == 100_000
    assert cashflow["permanent_gap"] == 12_000


def test_goals_are_precise_inflation_linked_and_pension_first():
    result = analyze_retirement(retirement_payload())
    wedding, education = result["goals"]

    assert wedding["future_cost"] > wedding["target_today"]
    assert wedding["growth_allocation_pct"] == 40
    assert education["growth_allocation_pct"] == 70
    assert sum(goal["allocated_corpus"] for goal in result["goals"]) <= result["allocation"]["pension_surplus"]


def test_score_swp_flags_and_dashboard_items_are_stable():
    payload = retirement_payload()
    payload["insurance"]["has_health_insurance"] = False
    payload["insurance"]["health_cover"] = 0
    result = analyze_retirement(payload)

    assert 0 <= result["scores"]["overall"] <= 100
    assert result["swp"]["status"] in {"gold_standard", "healthy", "caution", "critical", "gap", "no_swp_needed"}
    assert result["flags"][0]["priority"] == "critical"
    items = dashboard_action_items(result)
    assert items
    assert all(item["item_id"].startswith("retirement_") for item in items)


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
    result = analyze_retirement(retirement_payload())
    output = tmp_path / "retirement-plan.pdf"
    generate_retirement_pdf(result, str(output))

    assert output.exists() and output.stat().st_size > 10_000
    with pdfplumber.open(output) as pdf:
        assert len(pdf.pages) == 8
        text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    assert "Retirement Advisory Plan" in text
    assert "SWP Adequacy Spectrum" in text
    assert "Implementation Roadmap" in text
