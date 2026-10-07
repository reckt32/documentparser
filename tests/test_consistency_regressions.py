import os

os.environ.setdefault("OPENAI_API_KEY", "test")

import app
import db


def _facts(*, emergency_fund=None):
    return {
        "income": {
            "annualIncome": 1_200_000,
            "monthlyExpenses": 43_333,
            "monthlyEmi": 0,
        },
        "insurance": {"lifeCover": 0, "healthCover": 0},
        "lifestyle": {
            "emergency_fund": emergency_fund,
            "manual_corpus": 50_000,
        },
        "portfolio": {
            "equity": 70,
            "debt": 30,
            "current_value": 50_000,
            "total_monthly_sip": 8_500,
        },
        "goals": [],
        "analysis": {
            "insuranceGap": "Underinsured",
            "liquidity": "Insufficient",
            "debtStress": "Healthy",
            "surplusBand": "Strong",
            "advancedRisk": {
                "recommendedEquityBand": {"min": 55, "max": 70},
                "recommendedEquityMid": 62.5,
            },
            "_diagnostics": {
                "requiredLifeCover": 8_000_000,
                "liquidityMonths": 0,
                "emiPct": 0,
            },
            "ihs": {
                "score": 80,
                "breakdown": {
                    "portfolio_health": {"score": 100},
                    "goal_readiness": {"score": 100},
                    "protection": {"score": 20},
                    "liquidity": {"score": 20},
                    "tax_efficiency": {"score": 80},
                    "debt_management": {"score": 100},
                },
            },
        },
    }


def _allocation():
    return {
        "insurance_provision": 4_375,
        "priority_breakdown": [
            {"name": "Term Insurance", "monthly_amount": 3_333},
            {"name": "Health Insurance", "monthly_amount": 1_042},
        ],
        "goal_total_required": 60_000,
        "goal_total_deployment": 52_292,
        "goal_total_coverage": 43_792,
        "goal_savings_increase": 0,
        "goal_sip_table": [
            {
                "name": "Retirement",
                "ideal_sip": 60_000,
                "current_sip": 8_500,
                "gap_mo": 51_500,
                "coverage_used": 43_792,
            }
        ],
    }


def _paragraph_text(flowables):
    parts = []

    def visit(value):
        if isinstance(value, app.Paragraph):
            parts.append(value.getPlainText())
        elif isinstance(value, app.TagFlowable):
            parts.append(value.label)
        elif isinstance(value, app.KPIFlowable):
            for tile in value.tiles:
                parts.extend([str(tile.get("label") or ""), str(tile.get("value") or "")])
        elif isinstance(value, app.Table):
            visit(value._cellvalues)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(flowables)
    return " | ".join(parts)


def test_annual_identified_is_exactly_converted_plus_pending(monkeypatch):
    monkeypatch.setattr(
        db,
        "_query",
        lambda *_args, **_kwargs: [{
            "total_identified_count": 195,
            "total_identified_value": 25_900_000,
            "converted_value": 7_836_000,
            "converted_count": 17,
            "pending_value": 6_848_000,
            "pending_count": 178,
        }],
    )

    result = db.get_aggregate_metrics_for_period("mfd", "start", "end")

    assert result["total_identified_value"] == 14_684_000
    assert result["total_identified_count"] == 195
    assert result["conversion_pct"] == round(7_836_000 / 14_684_000 * 100, 2)


def test_report_uses_canonical_premiums_goal_plan_and_portfolio_value():
    facts = _facts()
    allocation = _allocation()

    protection_text = _paragraph_text(app.build_page_protection(facts, allocation))
    cashflow_text = _paragraph_text(app.build_page_cashflow_sip(facts, allocation))
    roadmap_text = _paragraph_text(app.build_page_action_plan(facts, allocation))
    portfolio_text = _paragraph_text(app.build_page_portfolio_debt(facts, allocation))

    assert "Rs. 3,333/month" in protection_text
    assert "Rs. 4,375/month" not in protection_text
    assert "Term + Health" in cashflow_text
    assert "Total Goal SIP Plan" in cashflow_text
    assert "Rs. 52,292" in cashflow_text
    assert "Deploy Rs. 52,292/month" in roadmap_text
    assert "Portfolio Rs. 50,000" in portfolio_text
    assert "MAINTAIN" in portfolio_text


def test_missing_and_zero_emergency_funds_are_not_conflated():
    missing = app._emergency_fund_metrics(_facts(emergency_fund=None))
    zero = app._emergency_fund_metrics(_facts(emergency_fund=0))

    assert missing["known"] is False
    assert missing["months"] is None
    assert zero["known"] is True
    assert zero["months"] == 0

    missing_snapshot = _paragraph_text(app.build_page_snapshot(_facts(emergency_fund=None), _allocation()))
    missing_liquidity = _paragraph_text(app.build_page_liquidity(_facts(emergency_fund=None), _allocation()))
    zero_snapshot = _paragraph_text(app.build_page_snapshot(_facts(emergency_fund=0), _allocation()))
    zero_liquidity = _paragraph_text(app.build_page_liquidity(_facts(emergency_fund=0), _allocation()))

    assert "Unknown" in missing_snapshot
    assert "Unknown" in missing_liquidity
    assert "0.0 months" in zero_snapshot
    assert "0.0 months" in zero_liquidity


def test_in_band_equity_gets_positive_consistent_copy():
    facts = _facts()
    snapshot_text = _paragraph_text(app.build_page_snapshot(facts, _allocation()))
    executive_text = _paragraph_text(app.build_page_executive_summary(facts, _allocation()))

    assert "70%" in snapshot_text
    assert "GOOD" in snapshot_text
    assert "70% equity is within the recommended 55-70% band" in executive_text
    assert "Equity allocation far outside your risk band" not in executive_text
