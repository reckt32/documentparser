"""Report page: goals."""

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from backend.reports.flowables import CanvasBlock, CoverageFlowable
from backend.reports.helpers import _as_dict, _as_list, _fmt_rs, _num
from backend.reports.theme import _report_styles


def build_page_goal_feasibility(client_facts, allocation_output):
    styles = _report_styles()
    
    goal_sip_rows = _as_list(allocation_output.get("goal_sip_table"))
    total_goals = max(1, len(goal_sip_rows))
    total_gap = _num(allocation_output.get("goal_total_gap"), 0)        # Σ (Required − Current)
    total_raisable = _num(allocation_output.get("goal_total_coverage"), 0)  # Σ coverage (= min(pool, gap))

    def _fmt(val):
        return _fmt_rs(val, compact=True)

    hint_style = ParagraphStyle("goal_hint", fontName="Helvetica", fontSize=7, leading=9,
                                textColor=colors.HexColor("#7F8C8D"), leftIndent=16, spaceBefore=3)

    cards = []
    goal_metrics = []          # per-goal {name, gap} for the stacked bar
    for g in goal_sip_rows:
        g = _as_dict(g)
        name = g.get("name") or "Goal"
        target_val = _num(g.get("target_amount"), 0)
        horizon = f"{g.get('horizon_years') or '-'} yrs"
        ideal = _num(g.get("ideal_sip"), 0)
        # Current SIP / Gap / Coverage are centralised in _allocation_output so the
        # goals page and the cashflow plan can't drift apart. Coverage is already
        # the gap-proportional share of surplus (capped at the gap).
        curr = _num(g.get("current_sip"), 0)
        gap = _num(g.get("gap_mo"), 0)
        coverage_used = _num(g.get("coverage_used"), 0)
        surplus_covers = bool(g.get("surplus_covers_gap")) or coverage_used >= gap
        savings_gap = 0 if surplus_covers else max(0, gap - coverage_used)
        # Bar = share of Required already fundable from Current + Coverage.
        funded_pct = min(100, (curr + coverage_used) / ideal * 100) if ideal > 0 else 100
        goal_metrics.append({"name": name, "gap": gap})

        target_str = _fmt(target_val)
        curr_str = _fmt(curr)
        ideal_str = _fmt(ideal)
        cov_str = _fmt(coverage_used)
        gap_str = _fmt(gap)

        goal_text = f"<font color='#666666' size='7'>GOAL</font><br/><font face='Times-Roman' size='11' color='#2C3E50'>{name}</font>"

        target_text = f"<font color='#666666' size='7'>TARGET</font><br/><font color='#2C3E50'>{target_str}</font>"
        horizon_text = f"<font color='#666666' size='7'>HORIZON</font><br/><font color='#2C3E50'>{horizon}</font>"
        req_sip_text = f"<font color='#666666' size='7'>REQ. SIP</font><br/><font color='#D35400'>{ideal_str}</font>"
        curr_sip_text = f"<font color='#666666' size='7'>CURR. SIP</font><br/><font color='#2C3E50'>{curr_str}</font>"
        cov_sip_text = f"<font color='#666666' size='7'>COVERAGE</font><br/><font color='#27AE60'>{cov_str}</font>"
        gap_text = f"<font color='#666666' size='7'>GAP/MO</font><br/><font color='{'#27AE60' if gap <= 0 else '#C0392B'}'>{gap_str}</font>"

        p_style = ParagraphStyle("card_p", fontName="Helvetica-Bold", fontSize=9, leading=12)
        val_style = ParagraphStyle("card_val", fontName="Helvetica-Bold", fontSize=8, leading=11)

        inner_data = [[
            Paragraph(target_text, val_style),
            Paragraph(horizon_text, val_style),
            Paragraph(req_sip_text, val_style),
            Paragraph(curr_sip_text, val_style),
            Paragraph(cov_sip_text, val_style),
            Paragraph(gap_text, val_style),
        ]]
        inner_table = Table(inner_data, colWidths=[44, 36, 46, 44, 50, 50])
        inner_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
        ]))

        # Bar = share of Required met by Current SIP + surplus Coverage. "Funded"
        # only when the running SIP alone covers the goal; a gap the surplus can
        # absorb is "Feasible" — money that isn't flowing yet.
        if gap <= 0:
            fund_label, fund_color = "Funded", '#27AE60'
        elif surplus_covers:
            fund_label, fund_color = "Feasible", '#2A557E'
        else:
            fund_label = "Partial" if (curr + coverage_used) > 0 else "Pending"
            fund_color = '#E67E22' if funded_pct >= 50 else '#C0392B'
        fund_val = f"{funded_pct:.0f}%"

        cov_tbl = Table(
            [[CoverageFlowable(min(funded_pct, 100), width=74, height=8)],
             [Spacer(1, 4)],
             [Table([[Paragraph(f"<font color='#666666' size='7'>{fund_label}</font>"), Paragraph(f"<b><font color='{fund_color}' size='8'>{fund_val}</font></b>", ParagraphStyle("r", alignment=TA_RIGHT))]], colWidths=[40, 34], style=TableStyle([("PADDING", (0, 0), (-1, -1), 0)]))]],
            colWidths=[74]
        )
        cov_tbl.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("VALIGN", (0, 0), (-1, -1), "TOP")]))

        card = Table(
            [[
                Paragraph(goal_text, p_style),
                inner_table,
                cov_tbl
            ]],
            colWidths=[104, 270, 76]
        )
        card.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E2E8F0")),
            ("ROUNDEDCORNERS", [8, 8, 8, 8]),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("LEFTPADDING", (0, 0), (0, -1), 16),
            ("RIGHTPADDING", (0, 0), (0, -1), 8),
            ("RIGHTPADDING", (-1, 0), (-1, -1), 16),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ]))

        # Per-goal one-line action: raise SIP now (from surplus) + savings to add later.
        if gap <= 0:
            hint = "Fully funded by current SIP."
        elif coverage_used > 0 and savings_gap > 0:
            hint = f"Raise SIP by {_fmt(coverage_used)} now from surplus · increase savings by {_fmt(savings_gap)} to fully close the gap."
        elif savings_gap <= 0:
            hint = f"Raise SIP by {_fmt(coverage_used)} now from surplus to fully close the gap."
        else:
            hint = f"Increase savings by {_fmt(savings_gap)} to close the gap."

        cards.append(card)
        cards.append(Paragraph(hint, hint_style))
        cards.append(Spacer(1, 10))

    goal_colors = [
        colors.HexColor("#C0392B"), # Red
        colors.HexColor("#E67E22"), # Orange
        colors.HexColor("#A8813C"), # Gold
        colors.HexColor("#2C3E50"), # Navy
        colors.HexColor("#27AE60"), # Green
        colors.HexColor("#7F8C8D"), # Grey
    ]
    
    goals_data = []
    for i, m in enumerate(goal_metrics):
        gap_amt = _num(m.get("gap"), 0)
        if gap_amt > 0:
            goals_data.append({
                "name": m.get("name") or f"Goal {i+1}",
                "shortfall": gap_amt,
                "color": goal_colors[i % len(goal_colors)]
            })

    def draw_stacked_bar(c, x, y, w, h):
        if not total_gap or total_gap <= 0: return
        c.saveState()
        path = c.beginPath()
        path.roundRect(x, y, w, h, h/2.0)
        c.clipPath(path, stroke=0)

        current_x = x
        for item in goals_data:
            seg_w = (item["shortfall"] / total_gap) * w
            if seg_w > 0:
                c.setFillColor(item["color"])
                c.rect(current_x, y, seg_w, h, fill=1, stroke=0)
                current_x += seg_w
        c.restoreState()

    stacked_bar_block = CanvasBlock(220, 14, draw_stacked_bar)

    legend_cells = []
    for item in goals_data:
        amt = item["shortfall"]
        if amt >= 100000:
            amt_str = f"Rs. {amt/100000:.1f}L"
        else:
            amt_str = f"Rs. {amt/1000:.1f}K"

        color_hex = item["color"].hexval()[2:] # rrggbb
        legend_cells.append(Paragraph(f"<font color='#{color_hex}'>■</font> <font color='#4A5568' size='7'>{item['name']} {amt_str}</font>", styles["small"]))

    legend_rows = []
    for i in range(0, len(legend_cells), 2):
        legend_rows.append(legend_cells[i:i+2])
    if legend_rows and len(legend_rows[-1]) == 1:
        legend_rows[-1].append(Paragraph("", styles["small"]))

    if legend_rows:
        legend_table = Table(legend_rows, colWidths=[110, 110])
        legend_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    else:
        legend_table = Spacer(1, 1)

    right_side = Table(
        [[stacked_bar_block],
         [Spacer(1, 6)],
         [legend_table]],
        colWidths=[214]
    )
    right_side.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))

    # Action plan: raise SIP by what surplus allows now, then grow savings to
    # close the rest. goal_savings_increase already ignores sub-500 rounding
    # residue (that money sits in the savings buffer by design).
    savings_increase = _num(allocation_output.get("goal_savings_increase"), 0)
    if savings_increase > 0:
        plan_label = "ADDITIONAL MONTHLY SAVINGS NEEDED"
        plan_amount = savings_increase
        plan_sub = (f"Raise SIP by {_fmt_rs(total_raisable, False)} now from your surplus, then grow monthly "
                    f"savings by {_fmt_rs(savings_increase, False)} to fully close the {_fmt_rs(total_gap, False)} "
                    f"gap across all {total_goals} goals.")
    elif total_raisable > 0:
        plan_label = "RAISE SIP FROM CURRENT SURPLUS"
        plan_amount = total_raisable
        plan_sub = (f"Raise SIP by {_fmt_rs(total_raisable, False)} from your surplus to fully fund all "
                    f"{total_goals} goals — no extra savings needed.")
    else:
        plan_label = "GOAL FUNDING"
        plan_amount = 0
        plan_sub = f"All {total_goals} goals are on track with current SIPs."
    left_side = Table(
        [[Paragraph(plan_label, ParagraphStyle("cms", parent=styles["small"], textColor=colors.HexColor("#D35400"), fontSize=7, spaceAfter=2))],
         [Paragraph(f"<font color='#D35400'>{_fmt_rs(plan_amount, False)}</font><font color='#D35400' size='12'>/month</font>", ParagraphStyle("cmsh", fontName="Times-Roman", fontSize=26, leading=26))],
         [Spacer(1, 4)],
         [Paragraph(f"<font color='#666666' size='7'>{plan_sub}</font>", styles["small"])]],
        colWidths=[224]
    )
    left_side.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    
    bottom_card = Table(
        [[left_side, Spacer(10, 1), right_side]],
        colWidths=[240, 10, 230]
    )
    bottom_card.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F9F1EB")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E5C5B5")),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (0, -1), 16),
        ("RIGHTPADDING", (-1, 0), (-1, -1), 16),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
    ]))

    return [
        Paragraph("FINANCIAL GOALS", ParagraphStyle("fg", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=2)),
        Paragraph(f"All {total_goals} Goals — Current vs Required", ParagraphStyle("mk_goal_title", parent=styles["h2"], fontName="Times-Roman", fontSize=26, leading=30)),
        Spacer(1, 16),
    ] + cards + [
        Spacer(1, 8),
        bottom_card
    ]
