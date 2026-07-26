"""
Generate two test reports for the v3 allocation waterfall.

PDF 1 — Combination 2 exactly as the founder listed it, INCLUDING the manual
        investment profile (Rs.90,000/month SIP, Rs.22L corpus) that the
        client's original submission lacked. Exercises: existing-SIP proration,
        full EF, over-insured term, adequate health, leftover-to-savings.

PDF 2 — "Stretched family" stress profile designed to hit every other branch:
        EF gap (builds at gap/18), term+health insurance gaps partially funded
        from savings (monthly insurance SIP shrinks the pool), tight surplus so
        the 30% bucket can't cover all goals -> nearest-horizon goal fills
        first, farthest goal goes Pending.

Run: venv/Scripts/python.exe repro_waterfall_pdfs.py
"""

import os
import sys
import types

os.environ.setdefault("OPENAI_API_KEY", "test")

class MockModule(types.ModuleType):
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

for m in ("firebase_admin", "firebase_admin.auth", "firebase_admin.credentials"):
    sys.modules.setdefault(m, MockModule(m))

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from app import _assemble_financial_inputs, analyze_financial_health, generate_financial_plan_pdf

COMBO2 = {
    "id": 1001,
    "personal_info": {"name": "combination 2 (full inputs)", "age": 29},
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
        "use_manual_overrides": True,
        "manual_sip": 90000,
        "manual_corpus": 2200000,
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

STRESS = {
    "id": 1002,
    "personal_info": {"name": "stretched family", "age": 36},
    "family_info": {"spouse": "Husband", "children": ["K1", "K2"], "has_financial_dependents": True},
    "lifestyle": {
        "annual_income": 1200000,      # 1L/month
        "monthly_expenses": 55000,
        "monthly_emi": 15000,          # surplus = 30,000/month
        "emergency_fund": 50000,       # target 3.3L -> gap 2.8L -> EF row appears
        "available_savings": 100000,   # partially covers insurance premiums
        "savings_band": "20-30%",
        "allocation": {"equity": 40, "debt": 60},
        "products": ["Mutual Funds"],
    },
    "insurance": {"life_cover": 2500000, "health_cover": 200000},  # both underinsured
    "risk_profile": {
        "tolerance": "medium",
        "primary_horizon": "medium",
        "primary_horizon_years": "12",
        "loss_tolerance_percent": "10",
        "goal_importance": "important",
        "goal_flexibility": "flexible",
        "behavior": "hold",
        "income_stability": "stable",
        "emergency_fund_months": "6",
        "equity_allocation_percent": "40",
    },
    "goals": {
        "wants_retirement_planning": True,
        "expected_pension": "0",
        "items": [
            {"name": "Car Upgrade", "target_amount": 600000, "horizon_years": 3,
             "risk_tolerance": "low", "goal_importance": "lifestyle", "goal_flexibility": "flexible", "behavior": "hold"},
            {"name": "Child Education", "target_amount": 3000000, "horizon_years": 12,
             "risk_tolerance": "medium", "goal_importance": "essential", "goal_flexibility": "fixed", "behavior": "hold"},
            {"name": "Retirement Corpus", "target_amount": None, "horizon_years": 24,
             "risk_tolerance": "medium", "goal_importance": "essential", "goal_flexibility": "flexible", "behavior": "hold"},
        ],
    },
    "tax_info": {"tax_regime": "old"},
    "estate": {"will_status": "No"},
}


def run(q, out_name):
    inputs = _assemble_financial_inputs(q)
    analysis = analyze_financial_health(inputs)
    out_path = os.path.join(os.path.dirname(__file__), "output", out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    generate_financial_plan_pdf(q, analysis, out_path)
    print(f"\n################ {q['personal_info']['name']} -> {out_name} ################")
    print("surplusBand:", analysis.get("surplusBand"), "| insuranceGap:", analysis.get("insuranceGap"),
          "| liquidity:", analysis.get("liquidity"))
    import pdfplumber
    with pdfplumber.open(out_path) as pdf:
        for i in (1, 3, 6, 7, 9):   # snapshot, protection, goals, cashflow, action plan
            print(f"----- PAGE {i+1} -----")
            print(pdf.pages[i].extract_text() or "")
    return out_path


if __name__ == "__main__":
    run(COMBO2, "waterfall_combo2_full.pdf")
    run(STRESS, "waterfall_stress_family.pdf")
