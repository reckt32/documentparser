"""
Lock the financial assumptions to a single source of truth (assumptions.py)
and assert the figures that used to diverge across modules now agree.
See backend/docs/ASSESSMENT.md section 2.2.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test")

import assumptions
import financial_calculators
import llm_sections
from app import (
    compute_insurance_gap,
    compute_term_insurance_need,
    compute_retirement_corpus,
)


# --- Withdrawal rate: one value everywhere ---------------------------------

def test_withdrawal_rate_single_source():
    assert financial_calculators.DEFAULT_WITHDRAWAL_RATE == assumptions.WITHDRAWAL_RATE_RETIREMENT
    assert llm_sections.WITHDRAWAL_RATE_RETIREMENT == assumptions.WITHDRAWAL_RATE_RETIREMENT


def test_inflation_rate_single_source():
    assert financial_calculators.DEFAULT_INFLATION_RATE == assumptions.INFLATION_RATE
    assert llm_sections.ASSUMED_INFLATION_RATE == assumptions.INFLATION_RATE


def test_retirement_corpus_uses_central_withdrawal_rate():
    result = compute_retirement_corpus(
        age=35, monthly_income=100000, desired_monthly_pension=50000
    )
    # Perpetuity corpus = annual pension / withdrawal rate.
    expected = (50000 * 12) / assumptions.WITHDRAWAL_RATE_RETIREMENT
    assert result["pension_corpus"] == round(expected, 0)
    assert result["withdrawal_rate_percent"] == assumptions.WITHDRAWAL_RATE_RETIREMENT * 100


# --- Life cover: the flag, the term need, and the multiple all agree -------

def test_insurance_gap_uses_central_multiple():
    annual_income = 1200000
    _status, required = compute_insurance_gap(annual_income, life_cover=0)
    assert required == assumptions.LIFE_COVER_MULTIPLE * annual_income


def test_term_need_matches_insurance_gap_required():
    monthly_income = 100000
    annual_income = monthly_income * 12
    term_need = compute_term_insurance_need(age=35, monthly_income=monthly_income)
    _status, gap_required = compute_insurance_gap(annual_income, life_cover=0)
    # Protection-page required cover must equal the analysis-layer required cover.
    assert term_need == gap_required
