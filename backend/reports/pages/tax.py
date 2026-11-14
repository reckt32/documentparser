"""Report page: tax."""

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from backend.analysis.llm_sections import compute_regime_comparison
from backend.reports.flowables import SavingCardFlowable, TagFlowable
from backend.reports.helpers import _as_dict, _as_list, _fmt_rs, _num
from backend.reports.theme import _report_styles


def build_page_tax(client_facts, allocation_output):
    styles = _report_styles()
    itr = client_facts.get("itr") or {}
    gross = _num(itr.get("gross_total_income") or (client_facts.get("income") or {}).get("annualIncome"), 0)
    deductions = _as_list(itr.get("deductions_claimed"))
    claimed = {"80C": 0, "80D": 0, "80CCD_1B": 0}
    for d in deductions:
        d = _as_dict(d)
        sec = str(d.get("section") or "").upper()
        amt = _num(d.get("amount"), 0)
        if "80CCD" in sec or "NPS" in sec:
            claimed["80CCD_1B"] += amt
        elif "80D" in sec:
            claimed["80D"] += amt
        elif "80C" in sec:
            claimed["80C"] += amt
            
    current = compute_regime_comparison(gross, claimed["80C"], claimed["80D"], claimed["80CCD_1B"])
    optimized = compute_regime_comparison(gross, 150000, 75000, 50000)
    
    def _tax_cell(val, better):
        col = "#27AE60" if better else "#C0392B"
        return Paragraph(f"<b><font color='{col}'>{_fmt_rs(val, False)}</font></b>", styles["table"])

    # Top Tables and Card
    comparison_rows = [[
        Paragraph("SCENARIO", styles["table_head_dark"]), 
        Paragraph("OLD REGIME", styles["table_head_dark"]), 
        Paragraph("NEW REGIME", styles["table_head_dark"]), 
        Paragraph("BETTER", styles["table_head_dark"])
    ]]
    
    for label, res in [("Current ITR", current), ("Max Deductions", optimized)]:
        comparison_rows.append([
            Paragraph(label, styles["table"]),
            _tax_cell(res["old_regime"]["tax_liability"], res["better_regime"] == "old"),
            _tax_cell(res["new_regime"]["tax_liability"], res["better_regime"] == "new"),
            TagFlowable(f"✓ {res['better_regime'].title()}", bg_tint=True)
        ])
        
    comp_table = Table(comparison_rows, colWidths=[90, 80, 80, 70])
    comp_table.setStyle(TableStyle([
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F8FAFC")),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#E2E8F0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    
    savings = max(_num(current.get('savings'), 0), _num(optimized.get('savings'), 0))
    saving_card = SavingCardFlowable(savings, width=225, height=120)
    
    top_split = Table([[comp_table, Spacer(1, 1), saving_card]], colWidths=[320, 20, 225])
    top_split.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    
    # Bottom Actions Table
    action_rows = [[
        Paragraph("#", styles["table_head_dark"]), 
        Paragraph("ACTION", styles["table_head_dark"]), 
        Paragraph("WHY", styles["table_head_dark"]), 
        Paragraph("HOW / NOTE", styles["table_head_dark"])
    ]]
    
    raw_actions = [
        ("Switch to New Regime", f"Saves {_fmt_rs(savings, False)}/year vs Old Regime", "Inform employer before April; applies from new FY"),
        ("Maximise Employer NPS", "Only deduction still valid under New Regime", "Up to 14% of Basic via Section 80CCD(2)"),
        ("Hold Equity Funds >1 Year", "LTCG 12.5% vs STCG 20%", "Avoid redemptions before 12-month mark"),
        ("Claim Standard Deduction", f"{_fmt_rs(75000, False)} automatically applied", "Confirm with payroll / CA before filing")
    ]
    
    for i, (act, why, how) in enumerate(raw_actions):
        action_rows.append([
            Paragraph(str(i+1), styles["table"]),
            Paragraph(f"<b>{act}</b>", styles["table"]),
            Paragraph(why, styles["table"]),
            Paragraph(how, styles["table"])
        ])
        
    actions_table = Table(action_rows, colWidths=[30, 130, 150, 160])
    actions_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))

    return [
        Spacer(1, 16),
        Paragraph("TAX ANALYSIS", ParagraphStyle("ta", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=2)),
        Paragraph("Regime Comparison & Structured Actions", ParagraphStyle("mk_tax_title", parent=styles["h2"], fontName="Times-Bold", fontSize=26, leading=30)),
        Spacer(1, 24),
        Paragraph("OLD VS NEW REGIME", ParagraphStyle("ovn", parent=styles["label"], textColor=colors.HexColor("#64748B"), fontSize=8)),
        Spacer(1, 8),
        top_split,
        Spacer(1, 32),
        Paragraph("KEY ACTIONS — WHAT, WHY & HOW", ParagraphStyle("ka", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=8)),
        actions_table
    ]
