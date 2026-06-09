import os

os.environ.setdefault("OPENAI_API_KEY", "test")

from app import (
    _allocation_output,
    _assemble_financial_inputs,
    _build_client_facts,
    _build_dashboard_snapshot,
    analyze_financial_health,
)


def _client_questionnaire():
    return {
        "id": None,
        "personal_info": {"name": "Revamp Regression", "age": 35},
        "family_info": {
            "spouse": {"name": "Spouse"},
            "children": [{"age": 5}],
            "dependents": [{"relation": "father"}],
        },
        "goals": {
            "wants_retirement_planning": True,
            "expected_pension": 150000,
            "items": [
                {
                    "name": "Child education",
                    "target_amount": 1500000,
                    "horizon_years": 10,
                    "risk_tolerance": "high",
                    "goal_importance": "important",
                    "goal_flexibility": "flexible",
                    "behavior": "buy",
                },
                {
                    "name": "Vacation",
                    "target_amount": 800000,
                    "horizon_years": 3,
                    "risk_tolerance": "low",
                    "goal_importance": "lifestyle",
                    "goal_flexibility": "fixed",
                    "behavior": "aggressive buy",
                },
                {
                    "name": "Retirement corpus",
                    "horizon_years": 25,
                    "risk_tolerance": "high",
                    "goal_importance": "essential",
                    "goal_flexibility": "flexible",
                    "behavior": "buy",
                },
            ],
        },
        "risk_profile": {
            "tolerance": "high",
            "loss_tolerance_percent": 10,
            "behavior": "buy",
            "emergency_fund_months": 4,
            "equity_allocation_percent": 90,
            "primary_horizon_years": 25,
        },
        "insurance": {
            "life_cover": 15000000,
            "health_cover": 800000,
            "insurer_name": "Client-entered insurer",
            "insurance_type": "Term + Health",
        },
        "lifestyle": {
            "annual_income": 2100000,
            "monthly_expenses": 45000,
            "monthly_emi": 35000,
            "emergency_fund": 250000,
            "available_savings": 115000,
            "savings_band": "10-20%",
            "products": ["mf", "stocks"],
            "allocation": {"equity": 90, "debt": 10},
            "manual_sip": 50000,
            "manual_corpus": 3200000,
            "expected_pension": 150000,
        },
        "tax_info": {"tax_regime": "new"},
        "estate": {"will_status": "no"},
    }


def test_revamped_report_preserves_manual_sip_allocation_and_pension():
    q = _client_questionnaire()
    inputs = _assemble_financial_inputs(q, doc_insights={})
    analysis = analyze_financial_health(inputs)
    facts = _build_client_facts(q, analysis, doc_insights={})
    allocation = _allocation_output(facts)
    snapshot = _build_dashboard_snapshot(analysis, q, {}, facts, allocation)

    assert inputs["personal"]["has_financial_dependents"] is True
    assert inputs["investments"]["allocation"]["equity"] == 90
    assert analysis["advancedRisk"]["raw"]["equityAllocationPercent"] == 90

    assert facts["portfolio"]["equity"] == 90
    assert facts["portfolio"]["debt"] == 10
    assert facts["portfolio"]["total_monthly_sip"] == 50000
    assert facts["portfolio"]["current_value"] == 3200000
    assert facts["insurance"]["insurerName"] == "Client-entered insurer"
    assert facts["itr"]["tax_regime"] == "new"

    assert facts["retirement_planning"]["enabled"] is True
    assert facts["retirement_planning"]["desired_monthly_pension"] == 150000

    assert allocation["existing_sip_running"] == 50000
    assert allocation["monthly_surplus"] == 95000
    assert snapshot["allocation_summary"]["existing_sip_running"] == 50000
    assert snapshot["financials"]["monthly_surplus"] == 95000
