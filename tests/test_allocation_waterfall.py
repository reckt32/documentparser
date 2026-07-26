"""
Regression tests for the v3 surplus-allocation waterfall (founder spec, 14 Jul 2026).

Spec (WhatsApp, Atharva):
  - Recurring premiums are provisioned in full monthly from the surplus and
    deducted before the split; the current balance is never spent on them.
  - 70% of the post-insurance pool -> Emergency Fund + Retirement, each capped
    at its required monthly amount ("whichever is first"); EF fills before
    retirement; excess flows down to the other goals.
  - Remaining 30% -> other goals prioritised time-wise: fill the nearest-horizon
    goal to its requirement, then the next.
  - Every SIP locked down to Rs.500 steps (3,653 -> 3,500).
  - Leftover after all allocations stays in the savings bank account.

Replaces the Jan 2026 "divide remaining surplus EQUALLY" behaviour.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test")

from llm_sections import PriorityAllocationEngine


def _row(alloc, name):
    return next(r for r in alloc["goal_sip_table"] if r["name"] == name)


def _allocate(**overrides):
    params = dict(
        monthly_surplus=100000,
        term_insurance_gap=0,
        health_insurance_gap=0,
        goals=[
            {"name": "Car", "target_amount": 1000000, "horizon_years": 3, "ideal_sip": 24000, "risk_category": "moderate"},
            {"name": "Education", "target_amount": 4000000, "horizon_years": 10, "ideal_sip": 17653, "risk_category": "growth"},
            {"name": "House", "target_amount": 8000000, "horizon_years": 15, "ideal_sip": 20000, "risk_category": "growth"},
            {"name": "Retirement Corpus", "target_amount": 30000000, "horizon_years": 25, "ideal_sip": 30000, "risk_category": "growth"},
        ],
        age=30,
        has_dependents=True,
        emergency_fund_target=510000,
        emergency_fund_current=0,
        available_savings=500000,
        monthly_expenses=0,  # keep the goal ideal_sips as given (no retirement re-derivation)
    )
    params.update(overrides)
    return PriorityAllocationEngine.compute_allocation(**params)


def test_priority_bucket_gets_70pct_ef_before_retirement():
    # Pool = 1,00,000 (no insurance gaps, savings absorb nothing).
    # P1 budget = 70,000. EF need = 510000/18 = 28,333 -> locked 28,000.
    # Retirement need 30,000 -> fits in the remaining 42,000.
    alloc = _allocate()
    ef = _row(alloc, "Emergency Fund")
    ret = _row(alloc, "Retirement Corpus")
    assert ef["allocated_sip"] == 28000
    assert ret["allocated_sip"] == 30000


def test_p1_excess_spills_to_term_goals_nearest_first():
    # P1 used 58,000 of 70,000 -> 12,000 spills down. P2 budget = 30,000 + 12,000.
    # Nearest goal (Car, 3y, need 24,000) fills first, then Education (10y).
    alloc = _allocate()
    car = _row(alloc, "Car")
    edu = _row(alloc, "Education")
    house = _row(alloc, "House")
    assert car["allocated_sip"] == 24000          # filled to need
    assert edu["allocated_sip"] == 17500          # 17,653 locked down to 500
    # 42,000 - 24,000 - 17,500 = 500 left for House
    assert house["allocated_sip"] == 500


def test_allocations_locked_to_500_steps():
    alloc = _allocate()
    for r in alloc["goal_sip_table"]:
        assert r["allocated_sip"] % 500 == 0, r


def test_leftover_parked_in_savings_never_lost():
    alloc = _allocate()
    total = sum(r["allocated_sip"] for r in alloc["goal_sip_table"])
    assert total + alloc["unallocated_to_savings"] == 100000


def test_no_goal_allocated_beyond_need():
    alloc = _allocate(monthly_surplus=500000)  # plenty of money
    for r in alloc["goal_sip_table"]:
        assert r["allocated_sip"] <= r["ideal_sip"], r
        assert r["shortfall"] == 0, r
    # Everything above the total requirement stays in savings
    assert alloc["unallocated_to_savings"] > 0


def test_retirement_takes_full_70_when_ef_already_funded():
    # Founder Q1: "yes & yes" — EF full -> retirement alone owns the 70% bucket.
    alloc = _allocate(emergency_fund_current=510000)
    names = [r["name"] for r in alloc["goal_sip_table"]]
    assert "Emergency Fund" not in names
    assert _row(alloc, "Retirement Corpus")["allocated_sip"] == 30000  # capped at need


def test_term_goal_leftover_tops_up_priority_bucket():
    # Tiny term goals -> the 30% bucket has spare, which must flow back to
    # retirement instead of sitting idle while a P1 gap remains.
    alloc = _allocate(
        goals=[
            {"name": "Gadget", "target_amount": 60000, "horizon_years": 1, "ideal_sip": 5000, "risk_category": "low"},
            {"name": "Retirement Corpus", "target_amount": 30000000, "horizon_years": 25, "ideal_sip": 80000, "risk_category": "growth"},
        ],
    )
    ret = _row(alloc, "Retirement Corpus")
    gadget = _row(alloc, "Gadget")
    assert gadget["allocated_sip"] == 5000
    # P1 = 70,000: EF 28,000 + retirement 42,000; then 30,000-5,000 = 25,000
    # spills back up to retirement -> 67,000 of its 80,000 need.
    assert ret["allocated_sip"] == 67000


def test_insurance_premium_provisioned_monthly_from_surplus():
    # Founder update (Jul 2026): the full premium is provisioned monthly from the
    # surplus (deducted before the 70/30 split), never from the current balance.
    # A 1Cr term gap costs ~50,000/yr at age 30 -> 4,167/month out of the pool;
    # the 5L current balance is left untouched.
    alloc = _allocate(term_insurance_gap=10000000)
    assert alloc["insurance_from_savings"] == 0
    assert alloc["insurance_sip_monthly"] == round(50000 / 12)  # 4,167
    assert alloc["available_savings"] == 500000                 # current balance untouched
    assert alloc["remaining_for_goals"] == 95833                # pool = surplus - provision
    total = sum(r["allocated_sip"] for r in alloc["goal_sip_table"])
    assert total + alloc["unallocated_to_savings"] == alloc["remaining_for_goals"]


def test_existing_sip_reduces_need_not_double_counted():
    # A running 50k SIP is prorated across goals; the waterfall only funds the
    # remaining need, and the pool itself shrinks by the existing commitment.
    alloc = _allocate(existing_sip_commitments=50000)
    pool = 100000 - 50000
    total = sum(r["allocated_sip"] for r in alloc["goal_sip_table"])
    assert total + alloc["unallocated_to_savings"] == pool
    for r in alloc["goal_sip_table"]:
        assert r["allocated_sip"] <= r["ideal_sip"]
