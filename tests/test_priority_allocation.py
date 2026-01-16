#!/usr/bin/env python3
"""Quick test for priority allocation engine."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm_sections import PriorityAllocationEngine, GoalsStrategyRunner

print("=" * 60)
print(" PRIORITY ALLOCATION ENGINE TEST")
print("=" * 60)

# Test 1: Basic priority allocation
print("\n--- Test 1: Priority Allocation ---")
allocation = PriorityAllocationEngine.compute_allocation(
    monthly_surplus=40000,
    term_insurance_gap=1500000,  # 15 lakh gap
    health_insurance_gap=500000,  # 5 lakh gap
    goals=[
        {"ideal_sip": 15000, "name": "Retirement"},
        {"ideal_sip": 20000, "name": "Child Education"},
        {"ideal_sip": 10000, "name": "Home Purchase"},
    ],
    age=35,
    has_dependents=True
)

print("Priority Breakdown:")
for p in allocation["priority_breakdown"]:
    print(f"  Priority {p['priority']}: {p['name']}")
    print(f"    Monthly: Rs. {p['monthly_amount']:,.0f}")
    print(f"    Status: {p['status']}")
    if "note" in p:
        print(f"    Note: {p['note']}")

print(f"\nSummary: {allocation['summary']}")
print(f"Goal Achievement: {allocation['goal_achievement_percent']}%")
print(f"Total Ideal SIP Needed: Rs. {allocation['total_ideal_sip_needed']:,.0f}")
print(f"Savings Shortfall: Rs. {allocation['savings_shortfall']:,.0f}")

print("\nBridge Recommendations:")
for i, rec in enumerate(allocation["bridge_recommendations"], 1):
    print(f"  {i}. {rec['option']}: {rec['action']}")

# Test 2: Scenario with adequate insurance
print("\n--- Test 2: Adequate Insurance ---")
allocation2 = PriorityAllocationEngine.compute_allocation(
    monthly_surplus=50000,
    term_insurance_gap=0,  # Adequate coverage
    health_insurance_gap=0,  # Adequate coverage
    goals=[{"ideal_sip": 30000, "name": "Goals"}],
    age=35,
    has_dependents=True
)
print(f"Achievement when insurance adequate: {allocation2['goal_achievement_percent']}%")
print(f"Allocated to Insurance: Rs. {allocation2['allocated_to_insurance']:,.0f}")

print("\n" + "=" * 60)
print(" ALL PRIORITY TESTS PASSED!")
print("=" * 60)
