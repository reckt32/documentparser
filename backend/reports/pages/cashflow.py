"""Report page: cashflow."""

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from backend.reports.flowables import InsightFlowable, KPIFlowable, TagFlowable, _llm_paragraph
from backend.reports.helpers import _as_dict, _as_list, _display_monthly_surplus, _fmt_rs, _num
from backend.reports.theme import _report_styles, sanitize_pdf_text


def build_page_cashflow_sip(client_facts, allocation_output, narratives=None):
    styles = _report_styles()
    income = client_facts.get("income") or {}
    monthly_income = _num(income.get("annualIncome"), 0) / 12
    monthly_surplus = _display_monthly_surplus(client_facts)
    total_req = _num(allocation_output.get("goal_total_required"), 0) or _num(allocation_output.get("combined_shortfall"), 0)
    goal_plan = _num(allocation_output.get("goal_total_deployment"), 0)
    coverage = goal_plan / total_req * 100 if total_req else 100
    
    cashflow_kpis = [
        {"label": "Monthly Income", "value": _fmt_rs(monthly_income)},
        {"label": "Monthly Expenses", "value": _fmt_rs(income.get("monthlyExpenses"))},
        {"label": "Monthly Surplus", "value": _fmt_rs(monthly_surplus)},
        {"label": "Term + Health", "value": _fmt_rs(allocation_output.get("insurance_provision"), False)},
        {"label": "Total Goal SIP Plan", "value": _fmt_rs(goal_plan, False)},
    ]
    
    cf_label = Paragraph("MONTHLY CASH FLOW", ParagraphStyle("mcf", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=12))
    kpi_drawing = KPIFlowable(cashflow_kpis, width=438, height=54, card_height=50, gap=6, card_y=0)
    
    summary_text = f"Total Goal Requirement: <b>{_fmt_rs(total_req, False)}/month</b>"
    coverage_text = f"Coverage: <font color='#D35400'><b>{coverage:.1f}%</b></font>"
    summary_table = Table([[Paragraph(summary_text, styles["small"]), Paragraph(coverage_text, ParagraphStyle("cov", parent=styles["small"], alignment=TA_RIGHT))]], colWidths=[219, 219])
    summary_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    
    cf_card_inner = Table([[cf_label], [kpi_drawing], [Spacer(1, 12)], [summary_table]], colWidths=[438])
    cf_card_inner.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    
    cf_card = Table([[cf_card_inner]], colWidths=[470])
    cf_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E2E8F0")),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
    ]))

    plan_rows = [[
        Paragraph("GOAL", styles["table_head_dark"]),
        Paragraph("REQUIRED SIP", styles["table_head_dark"]),
        Paragraph("CURRENT SIP", styles["table_head_dark"]),
        Paragraph("COVERAGE", styles["table_head_dark"]),
        Paragraph("GAP/MO", styles["table_head_dark"]),
        Paragraph("STATUS", styles["table_head_dark"])
    ]]
    goal_rows_cf = _as_list(allocation_output.get("goal_sip_table"))
    tot_req = tot_curr = tot_cov = tot_gap = 0.0
    for g in goal_rows_cf:
        g = _as_dict(g)
        # Current SIP / Gap / Coverage centralised in _allocation_output (same
        # numbers the Goals page shows). Coverage is the gap-proportional share
        # of surplus, capped at the gap.
        ideal = _num(g.get("ideal_sip"), 0)
        curr = _num(g.get("current_sip"), 0)
        gap = _num(g.get("gap_mo"), 0)
        coverage_used = _num(g.get("coverage_used"), 0)
        tot_req += ideal; tot_curr += curr; tot_cov += coverage_used; tot_gap += gap

        if gap <= 0:
            status_label = "Funded"           # current SIP already covers it
        elif g.get("surplus_covers_gap") or coverage_used >= gap:
            status_label = "Feasible"         # surplus can cover it, SIP not set up yet
        elif coverage_used > 0:
            status_label = "Partial"
        else:
            status_label = "Pending"

        plan_rows.append([
            Paragraph(str(g.get("name") or "Goal"), styles["table"]),
            Paragraph(_fmt_rs(ideal, False), styles["table"]),
            Paragraph(_fmt_rs(curr, False), styles["table"]),
            Paragraph(_fmt_rs(coverage_used, False), styles["table"]),
            Paragraph(_fmt_rs(gap, False), styles["table"]),
            TagFlowable(status_label, bg_tint=True)
        ])

    plan_rows.append([
        Paragraph("<b>Total</b>", styles["table"]),
        Paragraph(f"<b>{_fmt_rs(tot_req, False)}</b>", styles["table"]),
        Paragraph(f"<b>{_fmt_rs(tot_curr, False)}</b>", styles["table"]),
        Paragraph(f"<b>{_fmt_rs(tot_cov, False)}</b>", styles["table"]),
        Paragraph(f"<b>{_fmt_rs(tot_gap, False)}</b>", styles["table"]),
        Paragraph("", styles["table"])
    ])

    plan_table = Table(plan_rows, colWidths=[120, 76, 76, 76, 76, 76])
    plan_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#475569")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 7),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#F8FAFC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))

    insight = _llm_paragraph(narratives, "cashflow") or "Start with the available surplus, protect the household first, then scale SIPs as income grows."

    return [
        Paragraph("REALITY CHECK", ParagraphStyle("rc", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=2)),
        Paragraph("Can You Afford All This?", ParagraphStyle("mk_cf_title", parent=styles["h2"], fontName="Times-Bold", fontSize=28, leading=34)),
        Spacer(1, 16),
        cf_card,
        Spacer(1, 24),
        Paragraph("GOAL-WISE SIP ALLOCATION PLAN", ParagraphStyle("gwap", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=8)),
        plan_table,
        Spacer(1, 24),
        InsightFlowable("Key Insight", sanitize_pdf_text(insight))
    ]
