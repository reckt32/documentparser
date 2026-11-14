"""Report page: executive summary."""

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from backend.reports.drawing import _draw_dimension_score_bars
from backend.reports.flowables import DrawingFlowable, _styled_table, _tag
from backend.reports.helpers import (
    _as_dict,
    _emergency_fund_metrics,
    _fmt_rs,
    _format_status,
    _ihs_breakdown,
    _num,
    _portfolio_equity,
    _recommended_equity_band,
    _score_value,
    _urgency,
)
from backend.reports.theme import _report_styles


def build_page_executive_summary(client_facts, allocation_output, narratives=None):
    styles = _report_styles()
    analysis = client_facts.get("analysis") or {}
    breakdown = _ihs_breakdown(analysis)
    portfolio = _as_dict(client_facts.get("portfolio"))
    ar = _as_dict(analysis.get("advancedRisk"))
    equity = _portfolio_equity(portfolio)
    band_min, band_max = _recommended_equity_band(ar)
    ef_metrics = _emergency_fund_metrics(client_facts)
    emi_pct = _num(_as_dict(analysis.get("_diagnostics")).get("emiPct"), 0)
    why = {
        "portfolio_health": (
            f"{equity:.0f}% equity is within the recommended {band_min:.0f}-{band_max:.0f}% band"
            if band_min <= equity <= band_max else
            f"{equity:.0f}% equity is outside the recommended {band_min:.0f}-{band_max:.0f}% band"
        ),
        "goal_readiness": (
            "Current SIPs and available surplus cover the goal plan"
            if _num(allocation_output.get("goal_savings_increase"), 0) <= 0 else
            f"Additional monthly savings of {_fmt_rs(allocation_output.get('goal_savings_increase'), False)} are needed"
        ),
        "protection": f"Protection status: {_format_status(analysis.get('insuranceGap') or 'Unknown')}",
        "liquidity": (
            "Emergency-fund amount has not been provided"
            if not ef_metrics["known"] else
            f"Emergency fund covers {(ef_metrics['months'] or 0):.1f} months versus the 6-month target"
        ),
        "tax_efficiency": "Tax-efficiency score reflects the available filing and deduction data",
        "debt_management": f"EMIs use {emi_pct:.1f}% of monthly income versus the 40% benchmark",
    }
    canonical = [
        ("portfolio_health", "Portfolio Health"),
        ("goal_readiness", "Goal Readiness"),
        ("protection", "Protection"),
        ("liquidity", "Liquidity"),
        ("tax_efficiency", "Tax Efficiency"),
        ("debt_management", "Debt Management"),
    ]
    scored = []
    for key, label in canonical:
        scored.append((key, label, _score_value(breakdown.get(key), 0)))
    dimension_rows = []
    for _, label, score in sorted(scored, key=lambda item: item[2]):
        dimension_rows.append((label, _num(score, 0), _urgency(score)))
    priority_rows = [["AREA", "SCORE", "ACTION REQUIRED", "WHY IT MATTERS"]]
    for key, label, score in sorted(scored, key=lambda item: item[2]):
        score_val = int(_num(score, 0))
        color_hex = "#C0392B" if score_val < 40 else "#E67E22" if score_val < 75 else "#27AE60"
        priority_rows.append([Paragraph(f"<b>{label}</b>", styles["table"]), Paragraph(f"<b><font color='{color_hex}'>{score_val}/100</font></b>", styles["table"]), _tag(_urgency(score), bg_tint=True), why.get(key, "-")])
    overall = int(_num((((client_facts.get("analysis") or {}).get("ihs") or {}).get("score")), 0))
    overall_label = (("Needs Attention" if overall < 75 else "Healthy") if overall > 0 else "-")
    overall_color = "#E67E22" if overall < 75 else "#27AE60"
    immediate_count = sum(1 for _, _, score in scored if _num(score, 0) < 40)
    high_count = sum(1 for _, _, score in scored if 40 <= _num(score, 0) < 75)
    if immediate_count > 0:
        overall_note = f"{immediate_count} area{'s' if immediate_count != 1 else ''} require{'s' if immediate_count == 1 else ''} immediate action before deploying capital into goals"
    elif high_count > 0:
        overall_note = f"{high_count} area{'s' if high_count != 1 else ''} need{'s' if high_count == 1 else ''} attention — address alongside goal investing"
    else:
        overall_note = "All six dimensions healthy — maintain discipline and review quarterly"
    priority_rows.append([Paragraph("<b>Overall Score</b>", styles["table"]), Paragraph(f"<b><font color='{overall_color}'>{overall}/100</font></b>", styles["table"]), _tag(overall_label.title() if overall_label != "-" else "-", bg_tint=True), overall_note])
    priority_table = _styled_table(priority_rows, [130, 65, 105, 200], style_type="light")
    priority_table.setStyle(TableStyle([
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    blocks = [
        Paragraph("SIX CRITICAL DIMENSIONS", ParagraphStyle("six_dim", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=4)),
        Paragraph("Your Financial Health Score", ParagraphStyle("mk_exec_title", parent=styles["h2"], fontName="Times-Roman", fontSize=32, leading=38)),
        Spacer(1, 16),
        DrawingFlowable(lambda c, x, y, rows=dimension_rows: _draw_dimension_score_bars(c, 0, 140, 500, rows), 500, 160),
        Spacer(1, 12),
        Paragraph("Color key: <font color='#C0392B'>■</font> Critical &lt;40 &nbsp;&nbsp;&middot;&nbsp;&nbsp; <font color='#E67E22'>■</font> Needs Attention 40-74 &nbsp;&nbsp;&middot;&nbsp;&nbsp; <font color='#27AE60'>■</font> Good &ge;75", ParagraphStyle("ckey", parent=styles["small"], textColor=colors.HexColor("#888888"))),
        Spacer(1, 12),
        Paragraph("PRIORITY RANKING", ParagraphStyle("pr", parent=styles["label"], textColor=colors.HexColor("#A8813C"))),
        Spacer(1, 8),
        priority_table,
        Spacer(1, 6),
    ]
    logic_box = Table(
        [[
            Paragraph(
                "<b>Urgency Tag Logic</b><br/>"
                "Score &lt;40 → <b>IMMEDIATE</b>  ·  "
                "Score 40-74 → <b>HIGH</b>  ·  "
                "Score ≥75 → <b>GOOD</b><br/>"
                "Protection can be overridden to IMMEDIATE when family risk is critical.",
                ParagraphStyle("urgency_logic", parent=styles["small"], fontSize=7.5, leading=9),
            )
        ]],
        colWidths=[500],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F0E8")),
            ("LEFTPADDING", (0, 0), (-1, -1), 16),
            ("RIGHTPADDING", (0, 0), (-1, -1), 16),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]),
    )
    blocks.append(logic_box)
    return blocks
