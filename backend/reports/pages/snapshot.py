"""Report page: snapshot."""

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from backend.reports.flowables import DrawingFlowable, KPIFlowable, _styled_table, _tag
from backend.reports.helpers import (
    _as_dict,
    _as_list,
    _display_monthly_surplus,
    _emergency_fund_metrics,
    _fmt_rs,
    _format_status,
    _ihs_component_score,
    _num,
    _portfolio_equity,
    _portfolio_value,
    _recommended_band_text,
    _recommended_equity_band,
    _urgency,
)
from backend.reports.theme import _report_styles, sanitize_pdf_text


def build_page_snapshot(client_facts, allocation_output):
    styles = _report_styles()
    analysis = client_facts.get("analysis") or {}
    income = client_facts.get("income") or {}
    insurance = client_facts.get("insurance") or {}
    bank = client_facts.get("bank") or {}
    portfolio = client_facts.get("portfolio") or {}
    diag = analysis.get("_diagnostics") or {}
    ar = _as_dict(analysis.get("advancedRisk"))
    expenses = _num(income.get("monthlyExpenses"), 0)
    monthly_surplus = _display_monthly_surplus(client_facts)
    ef_target = expenses * 6
    ef_metrics = _emergency_fund_metrics(client_facts)
    s_body = styles["table"]
    green = colors.HexColor("#1FA55B")
    red = colors.HexColor("#CB3E2D")
    orange = colors.HexColor("#E68A1F")
    blue = colors.HexColor("#2A557E")
    def ctext(txt, col, right_align=False):
        style = ParagraphStyle("ctext", parent=s_body, alignment=TA_RIGHT if right_align else TA_LEFT)
        return Paragraph(f"<font color='{col.hexval()}'><b>{sanitize_pdf_text(str(txt))}</b></font>", style)
    
    def kpi_fmt(s):
        return str(s).replace("Rs. ", "Rs.").replace(" L", "L").replace(" Cr", "Cr")

    # Compute actual SIP amount from normalized CAS/manual facts.
    active_sip = _num(portfolio.get("total_monthly_sip"), 0)
    active_sip_text = f"Rs.{active_sip:,.0f}/mo" if active_sip > 0 else "Rs.0/mo"
    active_sip_color = green if active_sip > 0 else red

    # Goal status from allocation output. "Funded" means the current SIP covers
    # the requirement; a gap the surplus could absorb is only "feasible".
    goal_sip_rows = _as_list(allocation_output.get("goal_sip_table"))
    total_goals = max(len(goal_sip_rows), len(client_facts.get("goals") or []))
    funded_goals = 0
    feasible_goals = 0
    for g in goal_sip_rows:
        g = _as_dict(g)
        g_gap = _num(g.get("gap_mo"), 0)
        if g_gap <= 0:
            funded_goals += 1
        elif g.get("surplus_covers_gap") or _num(g.get("coverage_used"), 0) >= g_gap:
            feasible_goals += 1
    all_funded = funded_goals >= total_goals and total_goals > 0
    all_feasible = (funded_goals + feasible_goals) >= total_goals and total_goals > 0
    funded_color = green if all_funded else (blue if all_feasible else (orange if funded_goals > 0 else red))
    goals_tag = "GOOD" if all_funded else ("FEASIBLE" if all_feasible else "HIGH")

    # Insurance verdicts shared with the protection page
    life_status = str(allocation_output.get("life_cover_status") or "").upper()
    health_status = str(allocation_output.get("health_cover_status") or "").upper()
    rec_health = _num(allocation_output.get("recommended_health_cover"), 0)
    life_ok = life_status in ("ADEQUATE", "OVER-INSURED")
    health_ok = health_status in ("ADEQUATE", "OVER-INSURED")
    life_tag = life_status if life_ok else _urgency(_ihs_component_score(analysis, "protection"))
    health_display_tag = "UPGRADE" if health_status == "UPGRADE RECOMMENDED" else health_status
    equity = _portfolio_equity(portfolio)
    band_min, band_max = _recommended_equity_band(ar)
    equity_in_band = band_min <= equity <= band_max
    equity_color = green if equity_in_band else red
    equity_tag = "GOOD" if equity_in_band else _urgency(_ihs_component_score(analysis, "portfolio_health"))
    emergency_text = "Unknown" if not ef_metrics["known"] else f"{ef_metrics['months'] or 0:.1f} months"
    emergency_tag = "ASSESS" if not ef_metrics["known"] else (analysis.get("liquidity") or "-")

    rows = [["AREA", "CURRENT", "IDEAL", "PRIORITY"]]
    rows += [
        ["Life Cover", ctext(kpi_fmt(_fmt_rs(insurance.get("lifeCover"))), green if life_ok else red), ctext(kpi_fmt(_fmt_rs(diag.get("requiredLifeCover"))), green), _tag(life_tag)],
        ["Health Cover", ctext(kpi_fmt(_fmt_rs(insurance.get("healthCover"))), green if health_ok else orange), ctext(kpi_fmt(_fmt_rs(rec_health)), green), _tag(health_display_tag)],
        ["Emergency Fund", ctext(emergency_text, orange if not ef_metrics["known"] else blue), ctext(f"6 months expenses ({kpi_fmt(_fmt_rs(ef_target))})", green), _tag(emergency_tag)],
        ["Equity Allocation", ctext(f"{equity:.0f}%", equity_color), ctext(_recommended_band_text(ar), green), _tag(equity_tag)],
        ["EMI / Income", ctext(f"{_num(diag.get('emiPct'), 0):.1f}%", blue), ctext("<40%", green), _tag(analysis.get("debtStress") or "-")],
        ["Active SIP", ctext(active_sip_text, active_sip_color), ctext(f"Need {kpi_fmt(_fmt_rs(allocation_output.get('goal_total_required'), False))}/mo", green), _tag("HIGH" if active_sip <= 0 else "GOOD")],
        ["Goals Funded", ctext(f"{funded_goals} of {total_goals}", funded_color), ctext(f"{total_goals} of {total_goals}", green), _tag(goals_tag)],
    ]
    profile = [
        ["Risk Profile", ctext(_format_status(analysis.get("riskProfile")), colors.HexColor("#2C3E50"), True)],
        ["Surplus Level", ctext(_format_status(analysis.get("surplusBand") or "-"), green if (analysis.get("surplusBand") or "").lower() == "comfortable" or "adequate" in (analysis.get("surplusBand") or "").lower() else red, True)],
        ["Insurance Status", ctext(_format_status(analysis.get("insuranceGap") or "-"), red if "under" in (analysis.get("insuranceGap") or "").lower() else green, True)],
        ["Debt Position", ctext(_format_status(analysis.get("debtStress") or "-"), orange if "mod" in (analysis.get("debtStress") or "").lower() else green, True)],
        ["Liquidity", ctext(_format_status(analysis.get("liquidity") or "-"), red if "low" in (analysis.get("liquidity") or "").lower() or "insuf" in (analysis.get("liquidity") or "").lower() else green, True)],
        ["IHS Band", ctext(_format_status((analysis.get("ihs") or {}).get("band") or "-"), orange, True)],
    ]
    adv = [
        ["Calculated Risk Score", ctext(f"{ar.get('riskScore') or ar.get('score') or '-'} / 5.0", colors.HexColor("#2C3E50"), True)],
        ["Risk Appetite", ctext(_format_status(ar.get("riskAppetite") or ar.get("appetiteCategory") or "-"), colors.HexColor("#2C3E50"), True)],
        ["Tenure Limit", ctext(_format_status(ar.get("tenureLimit") or ar.get("tenureLimitCategory") or "-"), colors.HexColor("#2C3E50"), True)],
        ["Final Category", ctext(_format_status(ar.get("finalCategory") or "-"), orange, True)],
        ["Recommended Equity Band", ctext(_recommended_band_text(ar), green, True)],
    ]
    # Compute dynamic KPI notes
    annual_inc = _num(income.get("annualIncome"), 0)
    monthly_inc = annual_inc / 12 if annual_inc > 0 else 0
    savings_rate = (monthly_surplus / monthly_inc * 100) if monthly_inc > 0 else 0

    # Portfolio value: CAS data first, then manual_corpus fallback
    portfolio_val = _portfolio_value(portfolio, client_facts.get("lifestyle"))
    equity_pct = _portfolio_equity(portfolio)
    portfolio_note = f"{equity_pct:.0f}% equity" if portfolio_val > 0 else "No data"

    # Active SIP: normalized CAS/manual value
    active_sip_val = _num(portfolio.get("total_monthly_sip"), 0)

    snapshot_kpis = [
        {"label": "Annual Income (Gross)", "value": kpi_fmt(_fmt_rs(income.get("annualIncome"))), "note": f"{kpi_fmt(_fmt_rs(monthly_inc, False))}/month" if monthly_inc > 0 else ""},
        {"label": "Monthly Surplus", "value": kpi_fmt(_fmt_rs(monthly_surplus)), "note": f"Savings rate: {savings_rate:.0f}%" if savings_rate > 0 else ""},
        {"label": "Current Portfolio", "value": kpi_fmt(_fmt_rs(portfolio_val)) if portfolio_val > 0 else "—", "note": portfolio_note},
        {"label": "Active SIP", "value": f"Rs.{active_sip_val:,.0f}/mo" if active_sip_val > 0 else "Rs.0", "note": f"Need {kpi_fmt(_fmt_rs(allocation_output.get('goal_total_required'), False))}/mo" if active_sip_val <= 0 else "On track", "note_color": "orange" if active_sip_val <= 0 else "green"},
    ]
    current_vs_ideal = _styled_table(rows, [74, 56, 76, 104], style_type="light")
    profile_tbl = _styled_table(profile, [64, 96], header=False, style_type="light_right")
    adv_tbl = _styled_table(adv, [85, 75], header=False, style_type="light_right")

    profile_card = Table([[Paragraph("PROFILE AT A GLANCE", ParagraphStyle("paag", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=8))], [profile_tbl]], colWidths=[184])
    profile_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#EAE2D6")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))

    adv_card = Table([[Paragraph("ADVANCED RISK ASSESSMENT", ParagraphStyle("ara", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=8))], [adv_tbl]], colWidths=[184])
    adv_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#EAE2D6")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))

    right_stack = Table([[profile_card], [Spacer(1, 12)], [adv_card]], colWidths=[190])
    right_stack.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    
    left_side = Table([
        [Paragraph("CURRENT VS IDEAL", ParagraphStyle("cvsi", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=8))],
        [current_vs_ideal]
    ], colWidths=[310])
    left_side.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#EAE2D6")),
        ("LEFTPADDING", (0, 0), (-1, 0), 12),
        ("RIGHTPADDING", (0, 0), (-1, 0), 12),
        ("TOPPADDING", (0, 0), (-1, 0), 12),
        ("LEFTPADDING", (0, 1), (-1, -1), 0),
        ("RIGHTPADDING", (0, 1), (-1, -1), 0),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 12),
    ]))

    
    outer_table = Table([[left_side, Spacer(10, 1), right_stack]], colWidths=[310, 10, 190])
    outer_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
    ]))
    
    return [
        Paragraph("FINANCIAL SNAPSHOT", ParagraphStyle("mk_snap_label", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=4)),
        Paragraph("Where You Stand Today", ParagraphStyle("mk_snap_title", parent=styles["h2"], fontName="Times-Roman", fontSize=26, leading=30)),
        Spacer(1, 16),
        DrawingFlowable(
            lambda c, x, y, tiles: KPIFlowable(tiles, width=510, height=85, card_height=75, gap=10, card_y=5).drawOn(c, x, y),
            510, 85, snapshot_kpis
        ),
        Spacer(1, 16),
        outer_table,
    ]
