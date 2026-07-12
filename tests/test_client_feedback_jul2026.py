"""
Regression tests for the July 2026 client-feedback fixes (3rd batch).

Client-reported defects on report FinancialPlan_3a18d37b613dfa27:
  1. Goals with no running SIP were tagged "Funded" because surplus *could*
     cover them — they must read "Feasible" instead.
  2. Surplus Level showed "Low" for a client with a declared 53% savings
     rate — the band must derive from declared cashflow when present.
  3. Health insurance recommendation was locked at "10-15L" for every
     profile and recommended an upgrade even when current cover exceeded it.
  4. Term insurance: a 4.0Cr policy vs a 2.8Cr requirement still produced a
     "buy 2.8Cr" action; such policies must be marked over-insured instead.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test")

from assumptions import (
    recommended_health_cover,
    health_cover_status,
    life_cover_status,
)
from app import (
    _band_midpoint,
    declared_savings_percent,
    effective_savings_percent,
    analyze_financial_health,
)


# --- Surplus band from declared cashflow ------------------------------------

def _combo2_payload():
    """Combination 2: 28L income, 85k expenses, 25k EMI -> ~53% savings rate."""
    return {
        "personal": {"age": 29},
        "income": {"annualIncome": 2800000, "monthlyExpenses": 85000, "monthlyEmi": 25000},
        "goals": {"goalHorizon": "long"},
        "risk": {"tolerance": "high"},
        "insurance": {"lifeCover": 40000000, "healthCover": 2000000},
        "savings": {"savingsPercent": None, "savingsBand": ">30%"},
        "investments": {"allocation": {"equity": 85, "debt": 15}},
        "emergencyFundAmount": 1000000,
    }


def test_declared_savings_percent_matches_snapshot_rate():
    sp = declared_savings_percent({"annualIncome": 2800000, "monthlyExpenses": 85000, "monthlyEmi": 25000})
    assert sp is not None
    assert 52 < sp < 54  # (233333 - 85000 - 25000) / 233333 = 52.86%


def test_declared_savings_percent_missing_inputs():
    assert declared_savings_percent({}) is None
    assert declared_savings_percent({"annualIncome": 2800000}) is None
    assert declared_savings_percent({"monthlyExpenses": 85000}) is None


def test_surplus_band_strong_for_53pct_saver():
    results = analyze_financial_health(_combo2_payload())
    assert results["surplusBand"] == "Strong"


def test_declared_cashflow_beats_stale_doc_percent():
    payload = _combo2_payload()
    # A partial bank statement once seeded a misleading low percent
    payload["savings"]["savingsPercent"] = 4.0
    assert effective_savings_percent(payload) > 50


def test_band_fallback_when_no_declared_cashflow():
    payload = _combo2_payload()
    payload["income"] = {}
    assert effective_savings_percent(payload) == 35.0  # >30% band midpoint


def test_band_midpoint_variants():
    for band in (">30%", "30%+", "30+", "> 30 %", "Above 30%"):
        assert _band_midpoint(band) == 35.0, band
    assert _band_midpoint("<10%") == 5.0
    assert _band_midpoint("10-20%") == 15.0
    assert _band_midpoint("20-30%") == 25.0
    assert _band_midpoint("") is None


# --- Health cover: profile-scaled recommendation ----------------------------

def test_health_recommendation_moves_with_income():
    # 28L family -> base 15L wins over 14L income-scaled
    assert recommended_health_cover(2800000, 1) == 1500000
    # 60L family -> 30L, no longer locked at 15L
    assert recommended_health_cover(6000000, 1) == 3000000
    # Individual base is 10L
    assert recommended_health_cover(0, 0) == 1000000
    # Capped at 1Cr
    assert recommended_health_cover(100000000, 2) == 10000000


def test_health_20L_vs_15L_requirement_is_adequate():
    rec = recommended_health_cover(2800000, 1)
    assert health_cover_status(2000000, rec) == "ADEQUATE"


def test_health_statuses():
    assert health_cover_status(0, 1500000) == "NOT COVERED"
    assert health_cover_status(500000, 1500000) == "UPGRADE RECOMMENDED"
    assert health_cover_status(3000000, 1500000) == "OVER-INSURED"


# --- Life cover: over-insured marking ---------------------------------------

def test_life_4cr_vs_2p8cr_required_is_over_insured():
    assert life_cover_status(40000000, 28000000) == "OVER-INSURED"


def test_life_statuses():
    assert life_cover_status(0, 28000000) == "NOT COVERED"
    assert life_cover_status(20000000, 28000000) == "UNDERINSURED"
    assert life_cover_status(29000000, 28000000) == "ADEQUATE"
    assert life_cover_status(0, 0) == "ADEQUATE"  # no requirement (no dependents)
