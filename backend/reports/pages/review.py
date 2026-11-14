"""Report page: review."""

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from backend.reports.helpers import _as_dict
from backend.reports.theme import _report_styles


def build_page_review(client_facts, allocation_output):
    styles = _report_styles()
    
    # Left Card
    recalc_rows = [
        ("Quarterly", "Every 3 months — standard review cycle"),
        ("Income Change", "New job, salary revision, bonus, business income change"),
        ("Expense Change", "Major new EMI, lifestyle shift, new dependent"),
        ("Asset / Liability", "New loan, property purchase, large redemption"),
        ("Life Event", "Marriage, child birth, death in family, retirement"),
        ("Big Decision", "Before buying a home, taking a large loan, major investment"),
    ]
    recalc_data = []
    for k, v in recalc_rows:
        recalc_data.append([
            Paragraph(f"<b>{k}</b>", ParagraphStyle("rk", parent=styles["table"], textColor=colors.HexColor("#A8813C"))),
            Paragraph(v, styles["table"])
        ])
    recalc_table = Table(recalc_data, colWidths=[80, 140])
    recalc_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#EEEEEE")),
    ]))
    
    left_items = [
        Paragraph("WHEN TO RECALCULATE YOUR SCORE", ParagraphStyle("wt", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=12)),
        recalc_table
    ]
    left_card = Table([[item] for item in left_items], colWidths=[240])
    left_card.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("PADDING", (0, 0), (-1, -1), 16),
    ]))
    
    # Right Card. IMMEDIATE line reflects the actual open gaps — a client with
    # adequate cover and a full emergency fund shouldn't be told to set them up.
    immediate_items = []
    life_ok = str(allocation_output.get("life_cover_status") or "").upper() in ("ADEQUATE", "OVER-INSURED")
    health_ok = str(allocation_output.get("health_cover_status") or "").upper() in ("ADEQUATE", "OVER-INSURED")
    if not life_ok:
        immediate_items.append("Term insurance")
    if not health_ok:
        immediate_items.append("Health cover")
    if str((_as_dict(client_facts.get("analysis")).get("liquidity") or "")).lower() != "adequate":
        immediate_items.append("Emergency Fund setup")
    immediate_text = " + ".join(immediate_items) if immediate_items else "Goal-wise SIP setup from surplus"
    schedule_rows = [
        ("IMMEDIATE", immediate_text, "#C0392B"),
        ("30 DAYS", "Portfolio rebalancing + Tax regime switch + SIP setup", "#A8813C"),
        ("90 DAYS", "Full progress review + recalculate Health Score", "#A8813C"),
        ("QUARTERLY", "Score review + SIP check + rebalance if equity drifts >5%", "#27AE60"),
        ("ANNUALLY", "Comprehensive plan update with advisor", "#64748B"),
    ]
    schedule_data = []
    for k, v, c in schedule_rows:
        schedule_data.append([
            Paragraph(f"<b>{k}</b>", ParagraphStyle("sk", parent=styles["table"], textColor=colors.HexColor(c))),
            Paragraph(v, styles["table"])
        ])
    schedule_table = Table(schedule_data, colWidths=[80, 140])
    schedule_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#EEEEEE")),
    ]))
    
    right_items = [
        Paragraph("REVIEW SCHEDULE", ParagraphStyle("rs", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=12)),
        schedule_table
    ]
    right_card = Table([[item] for item in right_items], colWidths=[240])
    right_card.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("PADDING", (0, 0), (-1, -1), 16),
    ]))
    
    top_split = Table([[left_card, Spacer(1, 1), right_card]], colWidths=[245, 10, 245])
    top_split.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    
    # Checklist Grid
    checklist = [
        "Financial Health Score across 6 dimensions", "Snapshot — Current vs Ideal with priorities",
        "Advanced Risk Assessment + Profile at a Glance", "Protection gap (Life + Health with guidance)",
        "Portfolio rebalancing plan", "Debt Management data (EMI ratio + benchmarks)",
        "Liquidity & Emergency Fund full assessment", "Goal feasibility for all your goals + SIP column",
        "Reality Check — cash flow vs goal requirement", "Goal-wise SIP Allocation plan",
        "Tax optimisation + structured action table", "90-Day Roadmap + Phase-wise Action Plan"
    ]
    grid_data = []
    for i in range(0, len(checklist), 2):
        row = []
        for j in range(2):
            if i + j < len(checklist):
                row.append(Paragraph(f"<font color='#27AE60'>✓</font> {checklist[i+j]}", styles["small"]))
            else:
                row.append("")
        grid_data.append(row)
        
    grid_table = Table(grid_data, colWidths=[245, 245])
    grid_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 10),
    ]))

    return [
        Spacer(1, 16),
        Paragraph("STAY ON TRACK", ParagraphStyle("sot", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=2)),
        Paragraph("When to Recalculate & Review Schedule", ParagraphStyle("mk_rev_title", parent=styles["h2"], fontName="Times-Bold", fontSize=26, leading=30)),
        Spacer(1, 24),
        top_split,
        Spacer(1, 32),
        Paragraph("WHAT THIS REPORT CONTAINS", ParagraphStyle("wtrc", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=12)),
        grid_table
    ]
