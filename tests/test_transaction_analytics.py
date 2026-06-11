"""
Unit tests for transaction_analytics: deterministic recurring/EMI detection
over structured bank-statement rows, and the merge rules that decide when
LLM-detected entries survive.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transaction_analytics import (
    detect_recurring,
    detect_recurring_transactions,
    merge_recurring,
    compute_average_monthly_balance,
    statement_months,
    months_from_period,
    normalize_description,
    is_emi_description,
)


def _txn(date, desc, debit=None, credit=None, balance=None):
    amount = debit if debit is not None else credit
    ttype = "debit" if debit is not None else ("credit" if credit is not None else None)
    return {
        "date": date, "description": desc, "reference": None,
        "debit": debit, "credit": credit, "amount": amount,
        "type": ttype, "balance": balance,
    }


EMI_TXNS = [
    _txn("2024-04-05", "NACH/HDFC LTD/HOME LOAN/482910", debit=15000.0, balance=85000.0),
    _txn("2024-05-05", "NACH/HDFC LTD/HOME LOAN/513288", debit=15000.0, balance=70000.0),
    _txn("2024-06-05", "NACH/HDFC LTD/HOME LOAN/559104", debit=15000.0, balance=55000.0),
]

SALARY_TXNS = [
    _txn("2024-04-01", "NEFT SALARY ACME CORP", credit=100000.0, balance=100000.0),
    _txn("2024-05-01", "NEFT SALARY ACME CORP", credit=100000.0, balance=85000.0),
    _txn("2024-06-01", "NEFT SALARY ACME CORP", credit=100000.0, balance=70000.0),
]


# --- normalization ----------------------------------------------------------

def test_normalize_strips_reference_numbers():
    a = normalize_description("NACH/HDFC LTD/HOME LOAN/482910")
    b = normalize_description("NACH/HDFC LTD/HOME LOAN/559104")
    assert a == b == "nach hdfc ltd home loan"


def test_is_emi_strong_keyword():
    assert is_emi_description("HDFC LN 12345 INSTALMENT")
    assert not is_emi_description("AMAZON PURCHASE")


def test_is_emi_weak_rail_needs_monthly_recurrence():
    # NACH alone is a rail (SIPs, bills); only EMI when known monthly
    assert not is_emi_description("NACH RAZORPAY SOMething", recurring_monthly=False)
    assert is_emi_description("NACH RAZORPAY SOMething", recurring_monthly=True)


# --- recurring detection ----------------------------------------------------

def test_detects_monthly_emi_group():
    debits = detect_recurring(EMI_TXNS, "debit")
    assert len(debits) == 1
    e = debits[0]
    assert e["amount"] == 15000.0
    assert e["frequency"] == "Monthly"
    assert e["is_emi"] is True
    assert e["dates"] == ["2024-04-05", "2024-05-05", "2024-06-05"]
    # Shape matches what the LLM emits + prefill consumes
    assert set(e) >= {"description", "amount", "frequency", "dates", "is_emi"}


def test_detects_monthly_salary_credit():
    credits = detect_recurring(SALARY_TXNS, "credit")
    assert len(credits) == 1
    assert credits[0]["frequency"] == "Monthly"
    assert "is_emi" not in credits[0]


def test_single_occurrence_is_not_recurring():
    one = [_txn("2024-04-05", "NACH/HDFC LTD/HOME LOAN/482910", debit=15000.0)]
    assert detect_recurring(one, "debit") == []


def test_same_description_different_amounts_split_into_clusters():
    txns = EMI_TXNS + [
        _txn("2024-04-20", "NACH/HDFC LTD/HOME LOAN/990001", debit=40000.0),
        _txn("2024-05-20", "NACH/HDFC LTD/HOME LOAN/990002", debit=40000.0),
        _txn("2024-06-20", "NACH/HDFC LTD/HOME LOAN/990003", debit=40000.0),
    ]
    debits = detect_recurring(txns, "debit")
    amounts = sorted(e["amount"] for e in debits)
    assert amounts == [15000.0, 40000.0]


def test_undated_rows_are_ignored():
    txns = [
        _txn(None, "NACH/HDFC LTD/HOME LOAN/1", debit=15000.0),
        _txn(None, "NACH/HDFC LTD/HOME LOAN/2", debit=15000.0),
    ]
    assert detect_recurring(txns, "debit") == []


def test_detect_recurring_transactions_shape():
    out = detect_recurring_transactions(EMI_TXNS + SALARY_TXNS)
    assert set(out) == {"recurring_debits", "recurring_credits"}
    assert len(out["recurring_debits"]) == 1
    assert len(out["recurring_credits"]) == 1


# --- merge rules (deterministic wins, LLM fills gaps) ------------------------

def test_merge_prefers_deterministic_on_overlap():
    det = detect_recurring(EMI_TXNS, "debit")
    llm = [
        # Same group, hallucinated amount -> must be discarded
        {"description": "NACH/HDFC LTD/HOME LOAN/482910", "amount": 99999.0,
         "frequency": "Monthly", "dates": [], "is_emi": True},
        # Non-overlapping description -> must survive
        {"description": "NETFLIX SUBSCRIPTION", "amount": 649.0,
         "frequency": "Monthly", "dates": [], "is_emi": False},
    ]
    merged = merge_recurring(det, llm)
    by_amount = {e["amount"] for e in merged}
    assert 15000.0 in by_amount and 649.0 in by_amount
    assert 99999.0 not in by_amount


def test_merge_falls_back_to_llm_when_deterministic_empty():
    llm = [{"description": "EMI X", "amount": 5000.0, "frequency": "Monthly",
            "dates": [], "is_emi": True}]
    assert merge_recurring([], llm) == llm


# --- balance / span helpers --------------------------------------------------

def test_average_monthly_balance_uses_month_end_rows():
    txns = [
        _txn("2024-04-02", "A", debit=1.0, balance=500.0),
        _txn("2024-04-28", "B", debit=1.0, balance=1000.0),   # April month-end
        _txn("2024-05-30", "C", debit=1.0, balance=3000.0),   # May month-end
    ]
    assert compute_average_monthly_balance(txns) == 2000.0


def test_average_monthly_balance_none_without_balances():
    assert compute_average_monthly_balance(EMI_TXNS_NO_BAL) is None


EMI_TXNS_NO_BAL = [dict(t, balance=None) for t in EMI_TXNS]


def test_statement_months_span():
    months = statement_months(EMI_TXNS)  # 2024-04-05 .. 2024-06-05
    assert months is not None
    assert 1.8 <= months <= 2.2


# --- declared statement period -> months ------------------------------------

def test_months_from_period_quarter():
    assert months_from_period("01-Apr-2024 to 30-Jun-2024") == 3.0


def test_months_from_period_full_year_snaps_to_12():
    assert months_from_period("01-Apr-2023 to 31-Mar-2024") == 12.0


def test_months_from_period_numeric_dates():
    assert months_from_period("01/01/2024 to 31/03/2024") == 3.0


def test_months_from_period_invalid():
    assert months_from_period(None) is None
    assert months_from_period("N/A") is None
    assert months_from_period("garbage to nonsense") is None
    # reversed range
    assert months_from_period("30-Jun-2024 to 01-Apr-2024") is None


def test_merge_coerces_llm_string_amounts():
    llm = [{"description": "EMI X", "amount": "₹15,000", "frequency": "Monthly",
            "dates": [], "is_emi": True}]
    merged = merge_recurring([], llm)
    assert merged[0]["amount"] == 15000.0
