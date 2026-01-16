#!/usr/bin/env python3
"""Quick test script for tax optimization digest."""
import sys
import json
sys.path.insert(0, '.')

from llm_sections import TaxOptimizationRunner
from tests.test_report_generation import SAMPLE_CLIENT_FACTS

# Create dummy LLM
class DummyLLM:
    model = 'dummy'

runner = TaxOptimizationRunner(DummyLLM(), 9999, '/tmp')
digest = runner.digest(SAMPLE_CLIENT_FACTS)

print("=" * 60)
print(" TAX OPTIMIZATION DIGEST TEST")
print("=" * 60)

print("\n=== TAX PROFILE ===")
tp = digest['tax_profile']
print(f"Gross Income: Rs. {tp['gross_income']:,.0f}")
print(f"Taxable Income: Rs. {tp['taxable_income']:,.0f}")
print(f"Tax Paid: Rs. {tp['total_tax_paid']:,.0f}")
print(f"Effective Rate: {tp['effective_tax_rate']}%")
print(f"Marginal Rate: {tp['marginal_tax_rate']}%")
print(f"Regime: {tp['detected_regime']}")

print("\n=== DEDUCTION GAPS ===")
for sec, gap in digest['deduction_gaps'].items():
    print(f"{sec}:")
    print(f"  Limit: Rs. {gap.get('limit', gap.get('total_limit', 0)):,.0f}")
    print(f"  Current: Rs. {gap['current_utilization']:,.0f}")
    print(f"  Gap: Rs. {gap['gap']:,.0f}")
    print(f"  Tax Saving Potential: Rs. {gap['tax_saving_potential']:,.0f}")

print("\n=== LTCG OPTIMIZATION ===")
ltcg = digest.get('ltcg_optimization')
if ltcg:
    print(f"Type: {ltcg['type']}")
    print(f"Recommendation: {ltcg['recommendation']}")
else:
    print("No LTCG optimization data")

print("\n=== NEXT YEAR ROADMAP ===")
for item in digest['next_year_roadmap']['action_items']:
    print(f"P{item['priority']}: {item['action']}")
    print(f"        Save: {item['tax_saving']}")

print(f"\n*** TOTAL POTENTIAL SAVING: Rs. {digest['next_year_roadmap']['total_potential_saving']:,.0f} ***")

print("\n" + "=" * 60)
print(" TEST PASSED - Tax digest generated successfully!")
print("=" * 60)
