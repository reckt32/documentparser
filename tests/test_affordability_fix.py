#!/usr/bin/env python3
"""Quick test for affordability summary verification."""
import sys
sys.path.insert(0, r'd:\flutter-projects\document-parser\backend')

from llm_sections import GoalsStrategyRunner

class DummyLLM: model = 'dummy'

# Test with sample data similar to the reported issue
SAMPLE_FACTS = {
    'personal': {'age': 32, 'dependents_count': 0},
    'income': {
        'annualIncome': 922254,  # Annual income from report
        'monthlyExpenses': 38965,  # Approximate monthly expenses (467588/12)
        'monthlyEmi': 0,
    },
    'insurance': {'lifeCover': 0, 'healthCover': 0},
    'goals': [
        {'name': 'Retirement Corpus', 'target_amount': 50000000, 'horizon_years': 25},
        {'name': 'Wealth Creation', 'target_amount': 15000000, 'horizon_years': 10},
        {'name': 'Lifestyle', 'target_amount': 500000, 'horizon_years': 5},
    ],
    'portfolio': {'equity': 100, 'debt': 0, 'monthly_sip': 1500},
    'analysis': {
        'riskProfile': 'Moderate',
        'advancedRisk': {'finalCategory': 'Moderate', 'recommendedEquityBand': {'min': 40, 'max': 55}},
        '_diagnostics': {'requiredLifeCover': 3840330},
    },
}

runner = GoalsStrategyRunner(DummyLLM(), 9999, '/tmp')
digest = runner.digest(SAMPLE_FACTS)

print("=" * 60)
print(" AFFORDABILITY CHECK - VERIFYING FIX")
print("=" * 60)

fc = digest['financial_capacity']
print(f"\n=== FINANCIAL CAPACITY ===")
print(f"  Monthly Income: Rs. {fc['monthly_income']:,.0f}")
print(f"  Monthly Expenses: Rs. {fc['monthly_expenses']:,.0f}")
print(f"  Monthly EMI: Rs. {fc['monthly_emi']:,.0f}")
print(f"  Available Surplus: Rs. {fc['available_surplus']:,.0f}")
print(f"  Existing SIP: Rs. {fc['existing_sip_commitments']:,.0f}")
print(f"  Net Available for Goals: Rs. {fc['net_available_for_new_goals']:,.0f}")
print(f"  Total Ideal SIP Required: Rs. {fc['total_ideal_sip_required']:,.0f}")

aff = digest['affordability_summary']
print(f"\n=== AFFORDABILITY SUMMARY (NEW) ===")
print(f"  Total Ideal SIP Required: Rs. {aff['total_ideal_sip_required']:,.0f}")
print(f"  Total Affordable SIP: Rs. {aff['total_affordable_sip']:,.0f}")
print(f"  Remaining Budget for Goals: Rs. {aff['remaining_budget_for_goals']:,.0f}")
print(f"  Funding Gap: Rs. {aff['funding_gap']:,.0f}")
print(f"  Fundable %: {aff['fundable_percentage']:.1f}%")
print(f"  Constraint Violated: {aff['constraint_violated']}")
print(f"  Budget Exceeded By: Rs. {aff['budget_exceeded_by']:,.0f}")

print(f"\n=== PER-GOAL SIPs ===")
for g in digest['goals']:
    ideal = g.get('ideal_sip', 0) or 0
    affordable = g.get('affordable_sip', 0) or 0
    gap = g.get('gap_exists', False)
    print(f"  {g['name']}: ideal=Rs.{ideal:,.0f} | affordable=Rs.{affordable:,.0f} | gap={gap}")

# VALIDATION
total_recommended = sum((g.get('affordable_sip', 0) or 0) for g in digest['goals'])
available = fc['net_available_for_new_goals']
print(f"\n=== VALIDATION ===")
print(f"  Total Recommended (sum of affordable_sip): Rs. {total_recommended:,.0f}")
print(f"  Available Budget: Rs. {available:,.0f}")
if total_recommended <= available:
    print(f"  [PASS] Recommendations within budget!")
else:
    print(f"  [FAIL] Recommendations exceed budget by Rs. {total_recommended - available:,.0f}")

