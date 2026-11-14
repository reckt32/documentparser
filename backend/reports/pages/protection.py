"""Report page: protection."""

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from backend.assumptions import (
    LIFE_COVER_MULTIPLE,
    health_cover_status,
    life_cover_status,
    recommended_health_cover,
)
from backend.reports.drawing import draw_coverage_bar
from backend.reports.flowables import CanvasBlock, TagFlowable, _styled_table
from backend.reports.helpers import (
    _fmt_rs,
    _ihs_component_score,
    _num,
    _priority_monthly_amount,
    _urgency,
)
from backend.reports.theme import _report_styles


def build_page_protection(client_facts, allocation_output):
    styles = _report_styles()
    analysis = client_facts.get("analysis") or {}
    insurance = client_facts.get("insurance") or {}
    diag = analysis.get("_diagnostics") or {}
    current_life = _num(insurance.get("lifeCover"), 0)
    required_life = _num(diag.get("requiredLifeCover"), 0)
    gap = max(0, required_life - current_life)
    coverage_pct = (current_life / required_life * 100) if required_life > 0 else 100

    life_verdict = str(allocation_output.get("life_cover_status") or life_cover_status(current_life, required_life)).upper()
    life_status = life_verdict if life_verdict in ("ADEQUATE", "OVER-INSURED") else _urgency(_ihs_component_score(analysis, "protection"))
    current_health = _num(insurance.get("healthCover"), 0)
    rec_health = _num(allocation_output.get("recommended_health_cover"), 0) or recommended_health_cover(
        (client_facts.get("income") or {}).get("annualIncome"), (client_facts.get("personal") or {}).get("dependents_count"))
    health_tag = str(allocation_output.get("health_cover_status") or health_cover_status(current_health, rec_health)).upper()
    health_ok = health_tag in ("ADEQUATE", "OVER-INSURED")
    # This card is specifically for life cover; the combined provision also
    # contains health insurance and is shown only on the cash-flow page.
    est_premium = _priority_monthly_amount(allocation_output, "Term Insurance")
    life_gap_pct = (gap / required_life * 100) if required_life > 0 else 0
    life_surplus = max(0, current_life - required_life)
    life_header = Table(
        [[Paragraph("LIFE INSURANCE", ParagraphStyle("li", parent=styles["label"], textColor=colors.HexColor("#C0392B"))),
          TagFlowable(life_status, bg_tint=True)]],
        colWidths=[130, 90],
        style=TableStyle([
            ("ALIGN", (1, 0), (1, 0), "RIGHT"), 
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#E5CBC1")),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8)
        ]),
    )
    life_card_content = [
        life_header,
        Spacer(1, 12),
        Table(
            [[Paragraph(f"<font color='#555555'>Current:</font> <b>{_fmt_rs(current_life)}</b>", styles["body"]), Paragraph(f"<font color='#555555'>Required:</font> <b><font color='#27AE60'>{_fmt_rs(required_life)}</font></b>", styles["body"])]],
            colWidths=[110, 110],
            style=TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)])
        ),
        Spacer(1, 8),
        CanvasBlock(220, 14, lambda c, x, y, w, h: draw_coverage_bar(c, 0, 0, 220, coverage_pct, _fmt_rs(current_life), bar_color=(0.75, 0.22, 0.17))),
        Spacer(1, 8),
        Paragraph(
            (f"Coverage gap: <b><font color='#C0392B'>{life_gap_pct:.0f}% underinsured</font></b> · Basis: {int(LIFE_COVER_MULTIPLE)}x annual income")
            if gap > 0 else
            (f"Cover exceeds the {int(LIFE_COVER_MULTIPLE)}x-income benchmark by <b>{_fmt_rs(life_surplus)}</b>"
             + (" — <b><font color='#2A557E'>over-insured</font></b>" if life_verdict == "OVER-INSURED" else "")),
            styles["small"]),
        Spacer(1, 16),
        Table(
            [[ [Paragraph("PROTECTION GAP", ParagraphStyle("gap_lbl", parent=styles["label"], textColor=colors.HexColor("#C0392B"), spaceAfter=6)),
                Paragraph(f"<b>{_fmt_rs(gap)}</b>", ParagraphStyle("gap_val", parent=styles["h2"], fontSize=26, textColor=colors.HexColor("#C0392B"), leading=30, spaceAfter=4)),
                Paragraph(
                    (f"Est. premium ~{_fmt_rs(est_premium, False)}/month · Tenure until age 55") if gap > 0 else
                    ("No additional term cover needed — existing cover already exceeds the benchmark."
                     + (" Review sum assured at renewal; the extra cover carries extra premium." if life_verdict == "OVER-INSURED" else "")),
                    ParagraphStyle("gap_sub", parent=styles["small"], textColor=colors.HexColor("#4A5568")))] ]],
            colWidths=[200],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F6EAE6")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E5CBC1")),
                ("ROUNDEDCORNERS", [6, 6, 6, 6]),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 14),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ])
        ),
        Spacer(1, 16),
        Paragraph("WHAT TO LOOK FOR", ParagraphStyle("wtlf", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=8)),
        _styled_table([
            [Paragraph("<font color='#27AE60'>✓</font> &nbsp; Pure term insurance — no investment component", styles["body"])],
            [Paragraph("<font color='#27AE60'>✓</font> &nbsp; Coverage tenure until age 55-60 minimum", styles["body"])],
            [Paragraph(f"<font color='#27AE60'>✓</font> &nbsp; Sum assured benchmark: ~{int(LIFE_COVER_MULTIPLE)}x annual income", styles["body"])],
            [Paragraph("<font color='#27AE60'>✓</font> &nbsp; Compare claim-settlement ratio, exclusions and premium stability", styles["body"])],
        ], [220], header=False, style_type="light"),
    ]
    health_header = Table(
        [[Paragraph("HEALTH INSURANCE", ParagraphStyle("hi", parent=styles["label"], textColor=colors.HexColor("#27AE60"))),
          TagFlowable("UPGRADE" if health_tag == "UPGRADE RECOMMENDED" else health_tag, bg_tint=True)]],
        colWidths=[110, 110],
        style=TableStyle([
            ("ALIGN", (1, 0), (1, 0), "RIGHT"), 
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#CBE2D4")),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8)
        ]),
    )
    
    if health_tag == "OVER-INSURED":
        health_note = f"Cover exceeds the <b>{_fmt_rs(rec_health)}</b> recommendation — <b><font color='#2A557E'>over-insured</font></b>"
        health_status_line = "Cover Above Requirement"
        health_status_sub = f"Current cover exceeds the recommended {_fmt_rs(rec_health)} — review premium vs benefit at renewal"
    elif health_tag == "ADEQUATE":
        health_note = f"Meets the <b>{_fmt_rs(rec_health)}</b> recommendation — <b><font color='#27AE60'>no upgrade needed</font></b>"
        health_status_line = "Cover Adequate"
        health_status_sub = f"Current cover meets the recommended {_fmt_rs(rec_health)} for your profile — maintain and review annually"
    elif health_tag == "UPGRADE RECOMMENDED":
        health_note = f"Below the <b>{_fmt_rs(rec_health)}</b> recommendation — <b><font color='#E67E22'>upgrade recommended</font></b>"
        health_status_line = "Upgrade Recommended"
        health_status_sub = f"Upgrade to a {_fmt_rs(rec_health)} family floater for your income and dependents"
    else:  # NOT COVERED
        health_note = f"No health cover found — <b><font color='#C0392B'>get a {_fmt_rs(rec_health)} floater now</font></b>"
        health_status_line = "Not Covered"
        health_status_sub = f"Buy a {_fmt_rs(rec_health)} family floater before deploying capital into goals"
    health_bar_pct = min(200, (_num(insurance.get("healthCover"), 0) / rec_health * 100) if rec_health > 0 and _num(insurance.get("healthCover"), 0) > 0 else 0)

    health_card_content = [
        health_header,
        Spacer(1, 12),
        Table(
            [[Paragraph(f"<font color='#555555'>Current:</font> <b>{_fmt_rs(insurance.get('healthCover'))}</b>", styles["body"]), Paragraph(f"Recommended: <b><font color='#27AE60'>{_fmt_rs(rec_health)}</font></b>", styles["body"])]],
            colWidths=[110, 110],
            style=TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)])
        ),
        Spacer(1, 8),
        CanvasBlock(220, 14, lambda c, x, y, w, h: draw_coverage_bar(c, 0, 0, 220, health_bar_pct, _fmt_rs(insurance.get("healthCover")), bar_color=(0.15, 0.65, 0.38) if health_ok else (0.9, 0.49, 0.13))),
        Spacer(1, 8),
        Paragraph(health_note, styles["small"]),
        Spacer(1, 16),
        Table(
            [[ [Paragraph("STATUS", ParagraphStyle("stat_lbl", parent=styles["label"], textColor=colors.HexColor("#27AE60"), spaceAfter=6)),
                Paragraph(health_status_line, ParagraphStyle("stat_val", parent=styles["h2"], fontSize=20, textColor=colors.HexColor("#27AE60" if health_ok else "#E67E22"), leading=24, spaceAfter=4)),
                Paragraph(health_status_sub, ParagraphStyle("stat_sub", parent=styles["body"], textColor=colors.HexColor("#4A5568")))] ]],
            colWidths=[200],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EDF7F0")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#C6E4CF")),
                ("ROUNDEDCORNERS", [6, 6, 6, 6]),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 14),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ]),
        ),
        Spacer(1, 16),
        Paragraph("WHAT TO LOOK FOR", ParagraphStyle("wtlf", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=8)),
        _styled_table([
            [Paragraph("<font color='#27AE60'>✓</font> &nbsp; Family floater plan covering all dependents", styles["body"])],
            [Paragraph(f"<font color='#27AE60'>✓</font> &nbsp; {_fmt_rs(rec_health)} sum insured for your profile", styles["body"])],
            [Paragraph("<font color='#27AE60'>✓</font> &nbsp; Cashless facility at major hospitals", styles["body"])],
            [Paragraph("<font color='#27AE60'>✓</font> &nbsp; No room rent capping", styles["body"])],
        ], [220], header=False, style_type="light"),
    ]

    left_cell = Table([[life_card_content]], colWidths=[240])
    right_cell = Table([[health_card_content]], colWidths=[240])
    
    left_cell.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FDF5F2")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E5CBC1")),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
    ]))
    right_cell.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F2F9F4")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBE2D4")),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
    ]))

    cards = Table([[left_cell, Spacer(10, 1), right_cell]], colWidths=[240, 10, 240])
    cards.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
    ]))
    
    return [
        Paragraph("PROTECTION GAP ANALYSIS", ParagraphStyle("pga", parent=styles["label"], textColor=colors.HexColor("#A8813C"), spaceAfter=4)),
        Paragraph("Insurance — Actual vs Recommended", ParagraphStyle("iavr", parent=styles["h2"], fontName="Times-Roman", fontSize=26, leading=30)),
        Spacer(1, 12),
        cards
    ]
