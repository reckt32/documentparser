"""
Regression tests for compute_financial_health_score.

These lock the contract between the analysis layer (which emits status
strings like "Stressed" / "Insufficient" / "Strong") and the scoring layer
that consumes them. A prior revamp left the consumer matching a different
vocabulary, which silently mis-scored debt (inverted), liquidity, and goal
readiness. See backend/docs/ASSESSMENT.md section 2.1.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test")

from app import compute_financial_health_score


def _facts(equity=50):
    return {
        "portfolio": {"equity": equity},
        "goals": [],
        "itr": {},
    }


def _analysis(**overrides):
    base = {
        "insuranceGap": "Adequate",
        "liquidity": "Adequate",
        "debtStress": "Healthy",
        "surplusBand": "Adequate",
        "ihs": {"score": 70},
        "advancedRisk": {"recommendedEquityBand": {"min": 40, "max": 60}},
        "_diagnostics": {"liquidityMonths": 8},
    }
    base.update(overrides)
    return base


# --- Debt: the previously inverted component -------------------------------

def test_debt_healthy_scores_full():
    comp = compute_financial_health_score(_analysis(debtStress="Healthy"), _facts())["components"]
    assert comp["debt_management"]["score"] == 100
    assert comp["debt_management"]["label"] == "Healthy"


def test_debt_stressed_scores_low():
    comp = compute_financial_health_score(_analysis(debtStress="Stressed"), _facts())["components"]
    assert comp["debt_management"]["score"] == 30
    assert comp["debt_management"]["priority"] == "HIGH"


def test_debt_moderate_scores_mid():
    comp = compute_financial_health_score(_analysis(debtStress="Moderate"), _facts())["components"]
    assert comp["debt_management"]["score"] == 60


def test_debt_healthy_beats_stressed_overall():
    healthy = compute_financial_health_score(_analysis(debtStress="Healthy"), _facts())["overall"]
    stressed = compute_financial_health_score(_analysis(debtStress="Stressed"), _facts())["overall"]
    assert healthy > stressed


# --- Liquidity -------------------------------------------------------------

def test_liquidity_adequate_full():
    comp = compute_financial_health_score(_analysis(liquidity="Adequate"), _facts())["components"]
    assert comp["liquidity"]["score"] == 100


def test_liquidity_insufficient_low_band():
    a = _analysis(liquidity="Insufficient", _diagnostics={"liquidityMonths": 4})
    comp = compute_financial_health_score(a, _facts())["components"]
    assert comp["liquidity"]["score"] == 50
    assert comp["liquidity"]["label"] == "Low"


def test_liquidity_insufficient_critical_band():
    a = _analysis(liquidity="Insufficient", _diagnostics={"liquidityMonths": 1})
    comp = compute_financial_health_score(a, _facts())["components"]
    assert comp["liquidity"]["score"] == 20
    assert comp["liquidity"]["label"] == "Critical"


# --- Surplus / goal readiness ----------------------------------------------

def test_surplus_strong_is_good_progress():
    comp = compute_financial_health_score(_analysis(surplusBand="Strong"), _facts())["components"]
    assert comp["goal_readiness"]["label"] == "Good Progress"
    assert comp["goal_readiness"]["score"] > 30


def test_surplus_low_needs_work():
    comp = compute_financial_health_score(_analysis(surplusBand="Low"), _facts())["components"]
    assert comp["goal_readiness"]["label"] == "Needs Work"


def test_surplus_strong_beats_low_for_goals():
    strong = compute_financial_health_score(_analysis(surplusBand="Strong"), _facts())["components"]["goal_readiness"]["score"]
    low = compute_financial_health_score(_analysis(surplusBand="Low"), _facts())["components"]["goal_readiness"]["score"]
    assert strong > low


# --- End-to-end sanity: a strong client outranks a weak one ----------------

def test_strong_client_outranks_weak_client():
    strong = compute_financial_health_score(
        _analysis(insuranceGap="Adequate", liquidity="Adequate", debtStress="Healthy", surplusBand="Strong"),
        _facts(equity=50),
    )["overall"]
    weak = compute_financial_health_score(
        _analysis(
            insuranceGap="Underinsured",
            liquidity="Insufficient",
            debtStress="Stressed",
            surplusBand="Low",
            _diagnostics={"liquidityMonths": 1},
        ),
        _facts(equity=95),
    )["overall"]
    assert strong > weak
