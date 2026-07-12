"""
End-to-end repro for the July 2026 client feedback (Combination 2 profile).

Rebuilds the report that produced FinancialPlan_3a18d37b613dfa27.pdf —
29yo, 28L income, 85k expenses, 25k EMI, 4Cr term / 20L health cover,
10L emergency fund, savings band >30%, no uploaded documents, no running
SIP — and prints the lines the client complained about.

Run: venv/Scripts/python.exe repro_combo2.py
"""

import os
import sys
import types

os.environ.setdefault("OPENAI_API_KEY", "test")

class MockModule(types.ModuleType):
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

sys.modules.setdefault('firebase_admin', MockModule('firebase_admin'))
sys.modules.setdefault('firebase_admin.auth', MockModule('firebase_admin.auth'))
sys.modules.setdefault('firebase_admin.credentials', MockModule('firebase_admin.credentials'))

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from app import _assemble_financial_inputs, analyze_financial_health, generate_financial_plan_pdf

q = {
    "id": 999,
    "personal_info": {"name": "combination 2", "age": 29},
    "family_info": {"spouse": "Wife", "children": ["Kid1"], "has_financial_dependents": True},
    "lifestyle": {
        "annual_income": 2800000,
        "monthly_expenses": 85000,
        "monthly_emi": 25000,
        "emergency_fund": 1000000,
        "available_savings": 120000,
        "savings_band": ">30%",
        "allocation": {"equity": 85, "debt": 15},
        "products": ["Mutual Funds", "Stocks"],
    },
    "insurance": {"life_cover": 40000000, "health_cover": 2000000},
    "risk_profile": {
        "tolerance": "high",
        "primary_horizon": "long",
        "primary_horizon_years": "31",
        "loss_tolerance_percent": "25",
        "goal_importance": "essential",
        "goal_flexibility": "flexible",
        "behavior": "aggressive buy",
        "income_stability": "stable",
        "emergency_fund_months": "9",
        "equity_allocation_percent": "85",
    },
    "goals": {
        "wants_retirement_planning": True,
        "expected_pension": "0",
        "items": [
            {"name": "Children's Education", "target_amount": 4000000, "horizon_years": 17,
             "risk_tolerance": "high", "goal_importance": "essential", "goal_flexibility": "flexible", "behavior": "aggressive buy"},
            {"name": "Vacation", "target_amount": 1200000, "horizon_years": 4,
             "risk_tolerance": "medium", "goal_importance": "lifestyle", "goal_flexibility": "flexible", "behavior": "buy"},
            {"name": "Retirement Corpus", "target_amount": None, "horizon_years": 31,
             "risk_tolerance": "high", "goal_importance": "essential", "goal_flexibility": "flexible", "behavior": "aggressive buy"},
        ],
    },
    "tax_info": {"tax_regime": "new"},
    "estate": {"will_status": "No"},
}

inputs = _assemble_financial_inputs(q)
analysis = analyze_financial_health(inputs)
print("surplusBand:", analysis.get("surplusBand"))
print("insuranceGap:", analysis.get("insuranceGap"))
print("liquidity:", analysis.get("liquidity"), "| months:", (analysis.get("_diagnostics") or {}).get("liquidityMonths"))

out_path = os.path.join(os.path.dirname(__file__), "output", "repro_combo2.pdf")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
generate_financial_plan_pdf(q, analysis, out_path)
print("PDF written:", out_path)

import pdfplumber
with pdfplumber.open(out_path) as pdf:
    for i, p in enumerate(pdf.pages):
        text = p.extract_text() or ""
        print(f"\n===== PAGE {i+1} =====")
        print(text)
