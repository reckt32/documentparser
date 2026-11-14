"""Report page: portfolio debt."""

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from backend.reports.drawing import _safe_canvas_arc
from backend.reports.flowables import CanvasBlock, TagFlowable, _styled_table
from backend.reports.helpers import (
    _as_dict,
    _fmt_rs,
    _ihs_component_score,
    _num,
    _portfolio_equity,
    _portfolio_value,
    _recommended_band_mid,
    _recommended_equity_band,
)
from backend.reports.theme import _report_styles


def build_page_portfolio_debt(client_facts, allocation_output):
    styles = _report_styles()
    analysis = client_facts.get("analysis") or {}
    portfolio = client_facts.get("portfolio") or {}
    income = client_facts.get("income") or {}
    diag = analysis.get("_diagnostics") or {}
    ar = _as_dict(analysis.get("advancedRisk"))
    debt_score = _ihs_component_score(analysis, "debt_management")
    emi_pct = _num(diag.get("emiPct"), 0)
    
    current_eq = _portfolio_equity(portfolio)
    target_eq = _recommended_band_mid(ar)
    b_min, b_max = _recommended_equity_band(ar)
    equity_in_band = b_min <= current_eq <= b_max
    
    current_color = "#C0392B" if (current_eq < b_min or current_eq > b_max) else "#27AE60"
    target_color = "#27AE60"
    
    def _arc(c, x, y, w, h, pct, col):
        c.setStrokeColorRGB(0.55, 0.65, 0.75)
        c.setLineWidth(10)
        c.setLineCap(1)
        _safe_canvas_arc(c, x+10, y+10, x+w-10, y+h-10, -45, 270)
        ch = colors.HexColor(col)
        c.setStrokeColorRGB(ch.red, ch.green, ch.blue)
        c.setLineWidth(10)
        c.setLineCap(1)
        sweep = max(0.0, min(270.0, (_num(pct, 0) / 100.0) * 270))
        _safe_canvas_arc(c, x+10, y+10, x+w-10, y+h-10, -45, sweep)
        c.setFillColorRGB(ch.red, ch.green, ch.blue)
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(x + w/2, y + h/2 - 6, f"{int(_num(pct, 0))}%")
    
    left_block = [
        Paragraph("CURRENT ALLOCATION", ParagraphStyle("ca", parent=styles["label"], alignment=TA_CENTER, textColor=colors.HexColor("#7F8C8D"))),
        Spacer(1, 4),
        CanvasBlock(100, 100, lambda c,x,y,w,h: _arc(c, x, y, w, h, current_eq, current_color)),
        Spacer(1, 4),
        Paragraph(f"<font color='{current_color}'>● <b>{int(current_eq)}% Equity</b></font> &nbsp;&nbsp; <font color='#8CA2B5'>● {100-int(current_eq)}% Debt</font>", ParagraphStyle("sub1", parent=styles["body"], alignment=TA_CENTER)),
        Spacer(1, 2),
        Paragraph(f"Portfolio {_fmt_rs(_portfolio_value(portfolio, client_facts.get('lifestyle')), False)} · Gains {_fmt_rs(portfolio.get('unrealised_gains'), False)}", ParagraphStyle("sub2", parent=styles["small"], alignment=TA_CENTER, textColor=colors.HexColor("#7F8C8D"))),
    ]

    right_block = [
        Paragraph("TARGET ALLOCATION", ParagraphStyle("ta", parent=styles["label"], alignment=TA_CENTER, textColor=colors.HexColor("#7F8C8D"))),
        Spacer(1, 4),
        CanvasBlock(100, 100, lambda c,x,y,w,h: _arc(c, x, y, w, h, target_eq, target_color)),
        Spacer(1, 4),
        Paragraph(f"<font color='{target_color}'>● <b>{int(target_eq)}% Equity</b></font> &nbsp;&nbsp; <font color='#8CA2B5'>● {100-int(target_eq)}% Debt</font>", ParagraphStyle("sub3", parent=styles["body"], alignment=TA_CENTER)),
        Spacer(1, 2),
        Paragraph(f"{_as_dict(ar).get('finalCategory') or 'Moderate'} · Band {int(b_min)}–{int(b_max)}% · Risk-aligned", ParagraphStyle("sub4", parent=styles["small"], alignment=TA_CENTER, textColor=colors.HexColor("#7F8C8D"))),
    ]

    arrow_block = [
        Spacer(1, 30),
        CanvasBlock(40, 30, lambda c,x,y,w,h: [c.setStrokeColorRGB(0.8, 0.7, 0.5), c.setLineWidth(1), c.line(x, y+15, x+w, y+15), c.setFillColorRGB(0.8, 0.7, 0.5), c.setFont("Helvetica-Bold", 14), c.drawCentredString(x+w/2, y, "→")]),
        Spacer(1, 2),
        Paragraph("MAINTAIN" if equity_in_band else "REBALANCE", ParagraphStyle("reb", parent=styles["label"], alignment=TA_CENTER, textColor=colors.HexColor("#27AE60" if equity_in_band else "#A8813C"))),
    ]

    gauge_card = Table(
        [[left_block, arrow_block, right_block]],
        colWidths=[200, 80, 200],
        style=TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E2E8F0")),
            ("ROUNDEDCORNERS", [8, 8, 8, 8]),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ])
    )

    bullets = [
        "Which funds/instruments to use for rebalancing", 
        "When to execute (consider LTCG tax implications)", 
        "How to split equity vs debt per goal horizon", 
        "Which goals to prioritise if all aren't affordable"
    ]
    bullet_rows = [
        [
            Paragraph(f"<font color='#27AE60'>✓</font> &nbsp;{bullets[0]}", styles["body"]),
            Paragraph(f"<font color='#27AE60'>✓</font> &nbsp;{bullets[1]}", styles["body"])
        ],
        [
            Paragraph(f"<font color='#27AE60'>✓</font> &nbsp;{bullets[2]}", styles["body"]),
            Paragraph(f"<font color='#27AE60'>✓</font> &nbsp;{bullets[3]}", styles["body"])
        ]
    ]
    bullet_table = Table(bullet_rows, colWidths=[230, 230], style=TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#E5D9B1")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))

    control_card = Table(
        [[ [Paragraph("WHAT YOU CONTROL", ParagraphStyle("wyc", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=10)),
            bullet_table,
            Spacer(1, 12),
            CanvasBlock(460, 1, lambda c,x,y,w,h: [c.setStrokeColorRGB(0.9, 0.85, 0.7), c.setLineWidth(0.5), c.line(x,y,x+w,y)]),
            Spacer(1, 8),
            Paragraph("Powered by RISE · 25yr Nifty History · 6,250 Data Points · ~12.5% Long-term CAGR", ParagraphStyle("pow", parent=styles["small"], textColor=colors.HexColor("#95A5A6")))] ]],
        colWidths=[480],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FDFBF7")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E5D9B1")),
            ("ROUNDEDCORNERS", [8, 8, 8, 8]),
            ("LEFTPADDING", (0, 0), (-1, -1), 16),
            ("RIGHTPADDING", (0, 0), (-1, -1), 16),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ])
    )


    debt_rows = [
        ["METRIC", "YOUR VALUE", "BENCHMARK", "STATUS", "NOTE"],
        ["EMI / Income Ratio", Paragraph(f"<font color='#27AE60'><b>~{emi_pct:.0f}%</b></font>", styles["table"]) if emi_pct < 40 else Paragraph(f"<font color='#C0392B'><b>~{emi_pct:.0f}%</b></font>", styles["table"]), "<40%", TagFlowable("GOOD", bg_tint=True) if emi_pct < 40 else TagFlowable("HIGH", bg_tint=True), "Healthy discipline" if emi_pct < 40 else "Reduce EMI"],
        ["Total Outstanding Loans", "Moderate", "—", TagFlowable("REVIEW", bg_tint=True), "Confirm with client"],
        ["Loan Types", "Home / Car / Personal", "—", "—", "Identify highest rate"],
        ["Debt Score", Paragraph(f"<font color='#27AE60'><b>{int(_num(debt_score, 0))}/100</b></font>", styles["table"]) if _num(debt_score, 0) >= 75 else Paragraph(f"<font color='#C0392B'><b>{int(_num(debt_score, 0))}/100</b></font>", styles["table"]), "≥75 = Good", TagFlowable("GOOD", bg_tint=True) if _num(debt_score, 0) >= 75 else TagFlowable("NEEDS ATTENTION", bg_tint=True), "Well managed" if _num(debt_score, 0) >= 75 else "Needs attention"],
    ]
    debt_table = _styled_table(debt_rows, [110, 85, 95, 100, 110], style_type="light")
    debt_table.setStyle(TableStyle([
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    return [
        Paragraph("PORTFOLIO REBALANCING", ParagraphStyle("pga", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=2)),
        Paragraph("Equity Allocation — Actual vs Target", ParagraphStyle("iavr", parent=styles["h2"], fontName="Times-Roman", fontSize=26, leading=30)),
        Spacer(1, 4),
        gauge_card,
        Spacer(1, 6),
        control_card,
        Spacer(1, 6),
        Paragraph("DEBT MANAGEMENT", ParagraphStyle("pga", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=2)),
        Spacer(1, 2),
        debt_table
    ]
