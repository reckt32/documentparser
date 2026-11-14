"""Report page: liquidity."""

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from backend.reports.flowables import _styled_table
from backend.reports.helpers import _emergency_fund_metrics, _fmt_rs, _ihs_component_score, _num
from backend.reports.theme import _report_styles


def build_page_liquidity(client_facts, allocation_output):
    styles = _report_styles()
    analysis = client_facts.get("analysis") or {}
    income = client_facts.get("income") or {}
    diag = analysis.get("_diagnostics") or {}
    expenses = _num(income.get("monthlyExpenses"), 0)
    target = expenses * 6
    score = _ihs_component_score(analysis, "liquidity")

    # Current emergency fund from the questionnaire (Rs.). When provided, show the
    # actual funded status instead of a blanket "Unknown".
    ef_metrics = _emergency_fund_metrics(client_facts)
    ef_current = ef_metrics["current"]
    ef_months = ef_metrics["months"]
    ef_gap = ef_metrics["gap"]
    if not ef_metrics["known"]:
        status_markup = "<font color='#E67E22'><b>Unknown — assess now</b></font>"
    elif ef_current <= 0:
        status_markup = "<font color='#C0392B'><b>0.0 months · not funded</b></font>"
    elif ef_current >= target and target > 0:
        status_markup = f"<font color='#27AE60'><b>Fully funded · {_fmt_rs(ef_current, False)} (~{ef_months:.1f} months)</b></font>"
    else:
        status_markup = f"<font color='#E67E22'><b>{_fmt_rs(ef_current, False)} · ~{ef_months:.1f} of 6 months</b></font>"

    # Left Card: Parameters
    params = [
        ["Liquidity Score", Paragraph(f"<font color='#E67E22'><b>{_num(score, 0):.0f}/100</b></font>", styles["body"])],
        ["Basis", Paragraph("<b>6 months</b> essential expenses", styles["body"])],
        ["Monthly Expenses (est.)", Paragraph(f"<b>{_fmt_rs(expenses, False)}/month</b>", styles["body"])],
        ["Emergency Fund Target", Paragraph(f"<font color='#27AE60'><b>~{_fmt_rs(target, False)}</b></font>", styles["body"])],
        ["Current Status", Paragraph(status_markup, styles["body"])],
    ]
    params_table = _styled_table(params, [110, 80], header=False, style_type="compact_light")

    
    advisory_note = Table(
        [[Paragraph("<b><font color='#2C3E50'>Advisory Note</font></b><br/><font color='#4A5568'>Build emergency fund BEFORE investing in long-term goals. This is non-negotiable financial hygiene.</font>", styles["small"])]],
        colWidths=[173],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1E9DA")),
            ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#A8813C")),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ])
    )
    
    left = Table(
        [[ [Paragraph("EMERGENCY FUND PARAMETERS", ParagraphStyle("efp", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=12)),
            params_table,
            Spacer(1, 24),
            advisory_note] ]],
        colWidths=[205],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FDFBF7")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E5D9B1")),
            ("ROUNDEDCORNERS", [8, 8, 8, 8]),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 16),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ])
    )
    
    # Right Card: Strategy — build the *remaining* gap when some fund already exists.
    build_base = ef_gap if ef_metrics["known"] else target
    strategy = [
        ["OPTION", "DURATION", "MONTHLY", "BEST FOR"],
        [Paragraph("<font color='#27AE60'><b>A — Fast</b></font>", styles["body"]), "12 months", Paragraph(f"<font color='#E67E22'><b>{_fmt_rs(build_base / 12, False)}</b></font>", styles["body"]), Paragraph("<font color='#4A5568'>High surplus period</font>", styles["small"])],
        [Paragraph("<font color='#A8813C'><b>B — Balanced<br/>★</b></font>", styles["body"]), "18 months", Paragraph(f"<font color='#E67E22'><b>{_fmt_rs(build_base / 18, False)}</b></font>", styles["body"]), Paragraph("<font color='#4A5568'>Recommended</font>", styles["small"])],
        [Paragraph("<font color='#7F8C8D'><b>C — Gentle</b></font>", styles["body"]), "24 months", Paragraph(f"<font color='#E67E22'><b>{_fmt_rs(build_base / 24, False)}</b></font>", styles["body"]), Paragraph("<font color='#4A5568'>If goals compete</font>", styles["small"])],
    ]
    strategy_table = _styled_table(strategy, [60, 58, 60, 63], style_type="compact_light")

    strategy_table.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#FDFBF7")),
        ("TOPPADDING", (0, 1), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 10),
    ]))
    
    right = Table(
        [[ [Paragraph("BUILDING STRATEGY — TO CLOSE GAP" if ef_current > 0 else "BUILDING STRATEGY — FROM ZERO", ParagraphStyle("bs", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=12)),
            strategy_table] ]],
        colWidths=[265],
        style=TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E2E8F0")),
            ("ROUNDEDCORNERS", [8, 8, 8, 8]),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 16),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ])
    )
    
    pair = Table([[left, Spacer(10, 1), right]], colWidths=[205, 10, 265])
    pair.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0)
    ]))
    
    # Tier Cards
    t1_amt = _fmt_rs(target * 0.4, False)
    t2_amt = _fmt_rs(target * 0.4, False)
    t3_amt = _fmt_rs(target * 0.2, False)
    
    def _tier_card(code, code_color, title, subtitle, desc):
        tbl = Table(
            [[Paragraph(f"<font color='{code_color}'><b>{code}</b></font>", ParagraphStyle("tc", fontName="Helvetica-Bold", fontSize=14, leading=14, alignment=TA_CENTER)),
              Paragraph(f"<b><font color='#2C3E50'>{title}</font></b><br/><font color='#A8813C'>{subtitle}</font><br/><br/><font color='#4A5568'>{desc}</font>", styles["small"])]],
            colWidths=[30, 115]
        )
        tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EFE5D3")),
            ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#FDFBF7")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E5D9B1")),
            ("ROUNDEDCORNERS", [8, 8, 8, 8]),
            ("LEFTPADDING", (0, 0), (0, -1), 0),
            ("RIGHTPADDING", (0, 0), (0, -1), 0),
            ("LEFTPADDING", (1, 0), (1, -1), 10),
            ("RIGHTPADDING", (1, 0), (1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ]))
        return tbl
        
    t1 = _tier_card("T1", "#C0392B", "Instant Access", f"40% · ~{t1_amt}", "Savings bank account — zero friction. High-yield savings accounts.")
    t2 = _tier_card("T2", "#E67E22", "Quick Access", f"40% · ~{t2_amt}", "Liquid mutual funds or sweep-in FD. 1-2 day redemption. 6-7% return.")
    t3 = _tier_card("T3", "#A8813C", "Short-Term FD", f"20% · ~{t3_amt}", "3-6 month fixed deposits. Slightly higher return. Last-resort buffer.")
    
    tier_cards = Table([[t1, Spacer(8, 1), t2, Spacer(8, 1), t3]], colWidths=[145, 8, 145, 8, 145])
    tier_cards.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0)
    ]))
    
    return [
        Paragraph("LIQUIDITY ANALYSIS", ParagraphStyle("pga", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=2)),
        Paragraph("Emergency Fund Assessment", ParagraphStyle("iavr", parent=styles["h2"], fontName="Times-Roman", fontSize=26, leading=30)),
        Spacer(1, 16),
        pair,
        Spacer(1, 24),
        Paragraph("WHERE TO PARK THE EMERGENCY FUND", ParagraphStyle("wtp", parent=styles["label"], textColor=colors.HexColor("#A8813C"))),
        Spacer(1, 6),
        tier_cards
    ]
