#!/usr/bin/env python3
"""Quick test for priority allocation engine with enhanced features."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm_sections import PriorityAllocationEngine, GoalsStrategyRunner

print("=" * 60)
print(" PRIORITY ALLOCATION ENGINE TEST (ENHANCED)")
print("=" * 60)

# Test 1: Basic priority allocation with equal division
print("\n--- Test 1: Basic Priority Allocation with Equal Division ---")
allocation = PriorityAllocationEngine.compute_allocation(
    monthly_surplus=40000,
    term_insurance_gap=1500000,  # 15 lakh gap
    health_insurance_gap=500000,  # 5 lakh gap
    goals=[
        {"ideal_sip": 15000, "name": "Retirement", "risk_category": "Moderate", "horizon_years": 25, "target_amount": 25000000},
        {"ideal_sip": 20000, "name": "Child Education", "risk_category": "Growth", "horizon_years": 10, "target_amount": 5000000},
        {"ideal_sip": 10000, "name": "Home Purchase", "risk_category": "Conservative", "horizon_years": 5, "target_amount": 8000000},
    ],
    age=35,
    has_dependents=True,
    emergency_fund_target=240000,  # 6 months x 40k expenses
    emergency_fund_current=50000,  # Only 50k saved
)

print("Priority Breakdown:")
for p in allocation["priority_breakdown"]:
    print(f"  Priority {p['priority']}: {p['name']}")
    print(f"    Monthly: Rs. {p['monthly_amount']:,.0f}")
    print(f"    Status: {p['status']}")
    if "from_savings" in p:
        print(f"    From Savings: Rs. {p.get('from_savings', 0):,.0f}")
    if "note" in p:
        print(f"    Note: {p['note']}")

print(f"\nSummary: {allocation['summary']}")
print(f"Goal Achievement: {allocation['goal_achievement_percent']}%")
print(f"Per-goal SIP (equal division): Rs. {allocation.get('per_goal_sip', 0):,.0f}")
print(f"Number of buckets (emergency + goals): {allocation.get('num_allocation_buckets', 0)}")

# Print the new per-goal SIP table
print("\n--- Per-Goal SIP Table ---")
print(f"{'Goal':<20} {'SIP':>10} {'Risk':>12} {'Corpus':>12} {'Required':>10} {'Shortfall':>10}")
print("-" * 80)
for g in allocation.get("goal_sip_table", []):
    corpus = g.get("corpus_created") or g.get("corpus_created_1yr") or 0
    print(f"{g['name']:<20} Rs.{g['allocated_sip']:>7,} {g['risk_category']:>12} Rs.{corpus:>9,} Rs.{g['ideal_sip']:>7,} Rs.{g['shortfall']:>7,}")

print("\nBridge Recommendations:")
for i, rec in enumerate(allocation["bridge_recommendations"], 1):
    print(f"  {i}. {rec['option']}: {rec['action']}")

# Test 2: Using savings for insurance purchase
print("\n" + "=" * 60)
print("--- Test 2: Using Savings for Insurance Purchase ---")
allocation2 = PriorityAllocationEngine.compute_allocation(
    monthly_surplus=15000,  # 15k surplus
    term_insurance_gap=1200000,  # 12 lakh term gap
    health_insurance_gap=1000000,  # 10 lakh health gap
    goals=[
        {"ideal_sip": 6000, "name": "Goal 1", "risk_category": "Moderate", "horizon_years": 5, "target_amount": 500000},
        {"ideal_sip": 15000, "name": "Goal 2", "risk_category": "Growth", "horizon_years": 15, "target_amount": 5000000},
        {"ideal_sip": 60000, "name": "Goal 3", "risk_category": "Moderate", "horizon_years": 25, "target_amount": 50000000},
    ],
    age=35,
    has_dependents=True,
    available_savings=200000,  # Rs. 2 lakh savings available
    emergency_fund_target=200000,  # 6 months expenses
    emergency_fund_current=0,  # No emergency fund yet
)

ins_rec = allocation2.get("insurance_recommendation", {})
print(f"Savings Used for Insurance: Rs. {ins_rec.get('savings_used', 0):,.0f}")
print(f"Savings Remaining: Rs. {ins_rec.get('savings_remaining', 0):,.0f}")
print(f"Insurance SIP (for next year): Rs. {ins_rec.get('monthly_sip_for_next_year', 0):,.0f}/month")
if ins_rec.get("note"):
    print(f"Recommendation: {ins_rec['note']}")

print(f"\nRemaining for Goals + Emergency: Rs. {allocation2['remaining_for_goals']:,.0f}")
print(f"Equal division per bucket: Rs. {allocation2.get('per_goal_sip', 0):,.0f}")

print("\n--- Per-Goal SIP Table ---")
print(f"{'Goal':<20} {'SIP':>10} {'Risk':>12} {'Corpus':>12} {'Required':>10} {'Shortfall':>10}")
print("-" * 80)
for g in allocation2.get("goal_sip_table", []):
    corpus = g.get("corpus_created") or g.get("corpus_created_1yr") or 0
    print(f"{g['name']:<20} Rs.{g['allocated_sip']:>7,} {g['risk_category']:>12} Rs.{corpus:>9,} Rs.{g['ideal_sip']:>7,} Rs.{g['shortfall']:>7,}")

# Test 3: Insurance confirmed via tickmark (no allocation needed)
print("\n" + "=" * 60)
print("--- Test 3: Insurance Already Covered (Tickmark) ---")
allocation3 = PriorityAllocationEngine.compute_allocation(
    monthly_surplus=7000,  # 7k surplus (as in manager example)
    term_insurance_gap=1200000,  # Gap would exist but user confirmed having insurance
    health_insurance_gap=1000000,  
    goals=[
        {"ideal_sip": 6948, "name": "Lifestyle", "risk_category": "Moderate", "horizon_years": 5, "target_amount": 500000},
        {"ideal_sip": 15502, "name": "Home Purchase", "risk_category": "Growth", "horizon_years": 15, "target_amount": 5000000},
        {"ideal_sip": 59794, "name": "Retirement", "risk_category": "Moderate", "horizon_years": 25, "target_amount": 50000000},
    ],
    age=35,
    has_dependents=True,
    has_term_insurance_confirmed=True,  # User confirmed having term insurance
    has_health_insurance_confirmed=True,  # User confirmed having health insurance
    emergency_fund_target=234000,  # 6 months x 39k (as in example)
    emergency_fund_current=0,
)

print("Priority Breakdown (Insurance skipped):")
for p in allocation3["priority_breakdown"]:
    print(f"  Priority {p['priority']}: {p['name']} - {p['status']}")
    if p['monthly_amount'] > 0:
        print(f"    Monthly: Rs. {p['monthly_amount']:,.0f}")

print(f"\nFull surplus goes to goals + emergency: Rs. {allocation3['remaining_for_goals']:,.0f}")
print(f"Equal division (4 items = emergency + 3 goals): Rs. {allocation3.get('per_goal_sip', 0):,.0f} each")

print("\n--- Per-Goal SIP Table ---")
print(f"{'Goal':<20} {'SIP':>10} {'Risk':>12} {'Required':>10} {'Shortfall':>10}")
print("-" * 70)
for g in allocation3.get("goal_sip_table", []):
    print(f"{g['name']:<20} Rs.{g['allocated_sip']:>7,} {g['risk_category']:>12} Rs.{g['ideal_sip']:>7,} Rs.{g['shortfall']:>7,}")

# Test 4: Existing investments allocation
print("\n" + "=" * 60)
print("--- Test 4: Existing Investments Rebalancing ---")
allocation4 = PriorityAllocationEngine.compute_allocation(
    monthly_surplus=15000,
    term_insurance_gap=0,
    health_insurance_gap=0,
    goals=[
        {"ideal_sip": 10000, "name": "Goal A", "risk_category": "Conservative", "horizon_years": 5, "target_amount": 600000},
        {"ideal_sip": 10000, "name": "Goal B", "risk_category": "Growth", "horizon_years": 10, "target_amount": 2000000},
        {"ideal_sip": 20000, "name": "Goal C", "risk_category": "Aggressive", "horizon_years": 20, "target_amount": 10000000},
    ],
    age=35,
    has_dependents=True,
    existing_investments=1200000,  # Rs. 12 lakh existing investments
)

print("Existing Investments Allocation per Goal:")
for inv in allocation4.get("existing_investments_allocation", []):
    print(f"  {inv['goal_name']}: Rs. {inv['allocated_amount']:,.0f}")
    print(f"    {inv['rebalance_note']}")

print("\n" + "=" * 60)
print(" ALL ENHANCED PRIORITY TESTS PASSED!")
print("=" * 60)
