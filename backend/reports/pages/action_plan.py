"""Report page: action plan."""

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from backend.reports.helpers import _as_dict, _as_list, _display_monthly_surplus, _fmt_rs, _num
from backend.reports.theme import _report_styles


def build_page_action_plan(client_facts, allocation_output):
    styles = _report_styles()
    monthly_surplus = _display_monthly_surplus(client_facts)
    insurance_prov = _num(allocation_output.get("insurance_provision"), 0)
    surplus_after_ins = monthly_surplus - insurance_prov
    
    # Phase 1 Card
    p1_items = [
        Paragraph("PHASE 1 — MONTHS 1-6", ParagraphStyle("p1", parent=styles["label"], textColor=colors.HexColor("#A8813C"), fontSize=7)),
        Paragraph("Protection First", ParagraphStyle("p1t", parent=styles["h2"], fontSize=14, spaceBefore=4, spaceAfter=8)),
        Paragraph("Secure your family before deploying capital into goals.", ParagraphStyle("p1b", parent=styles["body"], fontSize=9, textColor=colors.HexColor("#64748B"), spaceAfter=12)),
    ]
    
    ins_verdicts = {
        "Term Insurance": str(allocation_output.get("life_cover_status") or "").upper(),
        "Health Insurance": str(allocation_output.get("health_cover_status") or "").upper(),
    }
    for p in _as_list(allocation_output.get("priority_breakdown")):
        p = _as_dict(p)
        name = p.get("name", "")
        if "Insurance" in name:
            monthly = _num(p.get("monthly_amount"), 0)
            if monthly > 0:
                p1_items.append(Paragraph(f"→ {name}: <b>{_fmt_rs(monthly, False)}/month</b>", styles["body"]))
            else:
                verdict = ins_verdicts.get(name, "")
                if verdict == "OVER-INSURED":
                    note = "over-insured — no new premium"
                elif verdict in ("ADEQUATE", ""):
                    note = "adequate — no new premium"
                elif str(p.get("status") or "").lower().startswith("provision"):
                    note = "provision premium monthly from surplus"
                else:
                    note = "cover gap — see protection page"
                p1_items.append(Paragraph(f"→ {name}: <b>{note}</b>", styles["body"]))

    p1_items.append(Spacer(1, 12))
    p1_items.append(Paragraph(f"→ Surplus after insurance: <b>{_fmt_rs(surplus_after_ins, False)}/month</b> available for goals", styles["body"]))
    
    p1_card = Table([[item] for item in p1_items], colWidths=[240])
    p1_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FDF2F2")),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))

    # Phase 2 Card. Allocation per goal = Current SIP + gap-proportional surplus
    # coverage (capped at the goal's requirement) — the same numbers as the Goals
    # and Cash Flow pages, never an equal split that overshoots a small goal.
    p2_goal_rows = _as_list(allocation_output.get("goal_sip_table"))[:5]
    total_deploy = _num(allocation_output.get("goal_total_deployment"), 0)
    has_running_sip = any(_num(_as_dict(g).get("current_sip"), 0) > 0 for g in p2_goal_rows)
    if has_running_sip:
        # ALLOC includes running SIPs, which may legitimately exceed a goal's
        # requirement — only NEW money is capped at need.
        p2_sub = (f"Deploy {_fmt_rs(total_deploy, False)}/month — continue existing SIPs; "
                  "new money fills emergency & retirement first, then nearest goals, up to each goal's need.")
    else:
        p2_sub = (f"Deploy {_fmt_rs(total_deploy, False)}/month — emergency & retirement first, "
                  "then nearest goals — capped at each goal's requirement.")
    p2_head = [
        Paragraph("PHASE 2 — MONTH 7 ONWARDS", ParagraphStyle("p2", parent=styles["label"], textColor=colors.HexColor("#27AE60"), fontSize=7)),
        Paragraph("Goal-Based SIPs", ParagraphStyle("p2t", parent=styles["h2"], fontSize=14, spaceBefore=4, spaceAfter=8)),
        Paragraph(p2_sub, ParagraphStyle("p2b", parent=styles["body"], fontSize=9, textColor=colors.HexColor("#64748B"), spaceAfter=12)),
    ]

    p2_rows = [[
        Paragraph("GOAL", styles["table_head_dark"]),
        Paragraph("ALLOC", styles["table_head_dark"]),
        Paragraph("REQ", styles["table_head_dark"])
    ]]
    for g in p2_goal_rows:
        g = _as_dict(g)
        alloc = _num(g.get("current_sip"), 0) + _num(g.get("coverage_used"), 0)
        p2_rows.append([
            Paragraph(g.get("name") or "Goal", styles["table"]),
            Paragraph(f"<b>{_fmt_rs(alloc, False)}</b>", ParagraphStyle("ts", parent=styles["table"], textColor=colors.HexColor("#27AE60"))),
            Paragraph(_fmt_rs(g.get("ideal_sip"), False), styles["table"])
        ])
    
    p2_table = Table(p2_rows, colWidths=[90, 55, 55])
    p2_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    
    p2_items = p2_head + [p2_table]
    unallocated = _num(allocation_output.get("unallocated_to_savings"), 0)
    if unallocated > 0:
        p2_items.append(Spacer(1, 8))
        p2_items.append(Paragraph(
            f"Leftover {_fmt_rs(unallocated, False)}/month stays in your savings account.",
            ParagraphStyle("p2left", parent=styles["small"], textColor=colors.HexColor("#64748B"))))
    p2_card = Table([[item] for item in p2_items], colWidths=[240])
    p2_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F9F1")),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    
    top_split = Table([[p1_card, Spacer(1, 1), p2_card]], colWidths=[245, 10, 245])
    top_split.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    
    # Timeline
    def _timeline_item(week, title, items):
        res = [
            Paragraph(f"<font color='#A8813C'><b>{week} — {title.upper()}</b></font>", styles["label"]),
            Paragraph(title, ParagraphStyle("tt", parent=styles["h2"], fontSize=11, spaceBefore=2, spaceAfter=4)),
        ]
        for it in items:
            res.append(Paragraph(f"→ {it}", ParagraphStyle("ti", parent=styles["body"], fontSize=8, leftIndent=8)))
        return res

    required_life = _num((_as_dict(client_facts.get("analysis")).get("_diagnostics") or {}).get("requiredLifeCover"), 0)
    current_life = _num((client_facts.get("insurance") or {}).get("lifeCover"), 0)
    life_verdict = str(allocation_output.get("life_cover_status") or "").upper()
    if life_verdict == "OVER-INSURED":
        term_action = f"Term cover {_fmt_rs(current_life)} exceeds {_fmt_rs(required_life)} needed — over-insured; review premium at renewal"
    elif life_verdict == "ADEQUATE" and current_life > 0:
        term_action = f"Maintain term cover — {_fmt_rs(current_life)} meets the {_fmt_rs(required_life)} requirement"
    elif required_life > 0 and current_life > 0:
        term_action = f"Top up term cover by {_fmt_rs(max(0, required_life - current_life))} to reach {_fmt_rs(required_life)}"
    elif required_life > 0:
        term_action = f"Get term insurance — {_fmt_rs(required_life)} cover"
    else:
        term_action = "Review term insurance need with an advisor"

    rec_health = _num(allocation_output.get("recommended_health_cover"), 0)
    current_health = _num((client_facts.get("insurance") or {}).get("healthCover"), 0)
    health_verdict = str(allocation_output.get("health_cover_status") or "").upper()
    if health_verdict in ("ADEQUATE", "OVER-INSURED"):
        health_action = f"Health cover adequate — maintain the {_fmt_rs(current_health)} floater"
    else:
        health_action = f"Upgrade health to a {_fmt_rs(rec_health)} family floater"
    col1 = [
        _timeline_item("WEEK 1", "Protection", [term_action, health_action]),
        Spacer(1, 8),
        _timeline_item("WEEK 2", "Liquidity", ["Assess savings + FD vs target", "Set up Tier 1/2/3 structure"]),
        Spacer(1, 8),
        _timeline_item("WEEK 3", "Portfolio", ["Plan LTCG-optimal exit", "Set up goal-wise SIPs"]),
    ]
    
    col2 = [
        _timeline_item("WEEK 4", "Tax", ["Switch to New Regime", "Maximise employer NPS"]),
        Spacer(1, 8),
        _timeline_item("DAY 90", "Full Review", ["Review all actions", "Recalculate Health Score"]),
        Spacer(1, 8),
        _timeline_item("QUARTERLY", "Ongoing", ["Ongoing Review", "Update plan for income changes"]),
    ]
    
    roadmap_split = Table([[
        Table([[item] for item in col1], colWidths=[230]),
        Table([[item] for item in col2], colWidths=[230])
    ]], colWidths=[245, 245])
    roadmap_split.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))

    return [
        Spacer(1, 8),
        Paragraph("YOUR ACTION PLAN", ParagraphStyle("yap", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=2)),
        Paragraph("Phase-wise Roadmap", ParagraphStyle("mk_ap_title", parent=styles["h2"], fontName="Times-Bold", fontSize=24, leading=28)),
        Spacer(1, 12),
        top_split,
        Spacer(1, 12),
        Paragraph("90-DAY ROADMAP", ParagraphStyle("90d", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=8)),
        roadmap_split
    ]
