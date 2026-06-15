"""Reproduce the reported equity case (no CAS; grid 90/10; equity field 80%)."""
import app

q = {
    "id": None,
    "personal_info": {"name": "Test Client", "age": 40, "pan": "ABCDE1234F"},
    "family_info": {"spouse": True, "children": [{"name": "Kid", "age": 8}],
                    "dependents": [{"relation": "parent"}, {"relation": "parent"}]},
    "risk_profile": {
        "tolerance": "high",
        "primary_horizon": "long",
        "loss_tolerance_percent": "15",
        "behavior": "buy",
        "emergency_fund_months": "6",
        "equity_allocation_percent": "80",   # standalone field
    },
    "insurance": {"life_cover": "20000000", "health_cover": "1200000"},
    "lifestyle": {
        "annual_income": "2000000",
        "monthly_expenses": "67000",
        "monthly_emi": "15000",
        "emergency_fund": "150000",
        "available_savings": "100000",
        "savings_band": "10-20%",
        "products": ["Mutual Funds", "Stocks"],
        "allocation": {"equity": "90", "debt": "10"},   # the grid
        "manual_sip": "70000",
        "manual_corpus": "4000000",
        "expected_pension": "150000",
    },
    "goals": {"wants_retirement_planning": True, "items": [
        {"name": "Child Education", "target_amount": 2500000, "horizon_years": 15, "risk_tolerance": "high"},
        {"name": "Vacation", "target_amount": 1000000, "horizon_years": 3, "risk_tolerance": "medium"},
        {"name": "Retirement", "target_amount": 0, "horizon_years": 15, "risk_tolerance": "high"},
    ]},
    "tax_info": {"tax_regime": "new"},
    "estate": {"will_status": "No"},
}

CAS = {"portfolio": {"equity": 30, "debt": 70}}  # inaccurate CAS extraction (~30%)

def show(label, q_, di):
    cf = app._build_client_facts(q_, {}, doc_insights=di)
    p = cf["portfolio"]
    report_eq = app._portfolio_equity(p)
    # Analysis-pipeline equity (feeds risk band + recommendation text)
    inputs = app._assemble_financial_inputs(q_, di)
    analysis_eq = app._safe_float((inputs.get("investments") or {}).get("allocation", {}).get("equity"), 0)
    match = "OK" if abs(report_eq - analysis_eq) < 0.5 else "MISMATCH!"
    print(f"{label}: report={report_eq:.0f}%  analysis={analysis_eq:.0f}%  [{match}]")

import copy
q_on = copy.deepcopy(q); q_on["lifestyle"]["use_manual_overrides"] = True
q_off = copy.deepcopy(q); q_off["lifestyle"]["use_manual_overrides"] = False

print("Inputs: grid equity=90, equity field=80, manual SIP=70000, corpus=4000000; CAS equity=30\n")
show("CAS + override ON  (expect 90% / 70000 / 4000000)", q_on, CAS)
show("CAS + override OFF (expect 30% / CAS-or-fallback)", q_off, CAS)
show("No CAS + override ON  (expect 90%)", q_on, None)
show("No CAS + override OFF (expect 90%, grid used as-is)", q_off, None)
