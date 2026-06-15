"""Ad-hoc layout check: render the Meerkat report with sample data to eyeball
the equity gauge + goal/gap card changes. Not part of the test suite."""
import os
import app

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

q = {
    "id": None,  # no DB / CAS lookup
    "personal_info": {"name": "Sam Sharma", "age": 38, "pan": "ABCDE1234F"},
    "family_info": {"spouse": True, "children": [{"name": "Aarav", "age": 6}]},
    "risk_profile": {
        "tolerance": "conservative",
        "primary_horizon": "long",
        "emergency_fund_months": "6",
        "equity_allocation_percent": "",
    },
    "lifestyle": {
        "annual_income": "2400000",
        "monthly_expenses": "90000",
        "monthly_emi": "20000",
        "manual_sip": "15000",          # -> existing/current SIP
        "emergency_fund": "200000",
        "available_savings": "300000",
        # Client typed 80/20 in the grid; CAS (below) says 29/71 -> grid must win
        "allocation": {"equity": "80", "debt": "20"},
        "use_manual_overrides": True,
    },
    "insurance": {"life_cover": "5000000", "health_cover": "500000"},
    "goals": {"items": [
        {"name": "Child Education", "target_amount": 4000000, "horizon_years": 12},
        {"name": "Home Purchase", "target_amount": 8000000, "horizon_years": 7},
        {"name": "Retirement", "target_amount": 50000000, "horizon_years": 22},
    ]},
    "tax_info": {"tax_regime": "new"},
    "estate": {},
}

analysis = {
    "advancedRisk": {
        "recommendedEquityBand": {"min": 40, "max": 55},
        "recommendedEquityMid": 47,
        "finalCategory": "Conservative",
        "baselineCategory": "Conservative",
    },
    "_diagnostics": {"emiPct": 18, "requiredLifeCover": 24000000},
    "recommendations": ["Build emergency fund", "Increase term cover"],
}

doc_insights = {"portfolio": {"equity": 29, "debt": 71, "gold": 5}}

out_path = os.path.join(OUT_DIR, "layout_check.pdf")

try:
    app.generate_financial_plan_pdf(q, dict(analysis), out_path, doc_insights=doc_insights)
    print("FULL report generated ->", out_path)
except Exception as e:
    import traceback
    print("Full report failed (%s); rendering the 3 changed pages only." % e)
    traceback.print_exc()
    cf = app._build_client_facts(q, dict(analysis), doc_insights)
    try:
        hs = app.compute_financial_health_score(cf.get("analysis") or {}, cf)
        cf.setdefault("analysis", {}).setdefault("ihs", {})
        cf["analysis"]["ihs"].update({"score": hs.get("overall", 0), "band": hs.get("overall_label", ""), "breakdown": hs.get("components") or {}})
    except Exception:
        pass
    alloc = app._allocation_output(cf)
    story = []
    story += app.build_page_portfolio_debt(cf, alloc)
    story.append(app.PageBreak())
    story += app.build_page_goal_feasibility(cf, alloc)
    story.append(app.PageBreak())
    story += app.build_page_cashflow_sip(cf, alloc)
    doc = app.SimpleDocTemplate(out_path, pagesize=app.A4, rightMargin=0.65*app.inch,
                                leftMargin=0.65*app.inch, topMargin=0.85*app.inch, bottomMargin=0.55*app.inch)
    doc.build(story, onFirstPage=app._meerkat_page_background, onLaterPages=app._meerkat_page_background)
    print("SUBSET (3 pages) generated ->", out_path)

# Show the key computed numbers for sanity
cf = app._build_client_facts(q, dict(analysis), doc_insights)
print("portfolio equity now:", app._portfolio_equity(cf["portfolio"]), "(CAS said 29, grid said 80)")
alloc = app._allocation_output(cf)
print("existing_sip_running:", alloc.get("existing_sip_running"))
for g in alloc.get("goal_sip_table", []):
    print("  goal:", g.get("name"), "ideal:", g.get("ideal_sip"), "allocated(coverage):", g.get("allocated_sip"))
