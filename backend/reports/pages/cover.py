"""Report page: cover."""

import os
from datetime import datetime, timedelta, timezone

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, Spacer, Table, TableStyle

from backend.assumptions import LIFE_COVER_MULTIPLE
from backend.config import LOGO_PATH
from backend.reports.drawing import draw_arc_gauge
from backend.reports.flowables import CanvasBlock
from backend.reports.helpers import _format_status, _ihs_breakdown, _score_value
from backend.reports.theme import _report_styles, sanitize_pdf_text


def build_page_cover(client_facts, allocation_output, narratives=None):
    styles = _report_styles()
    analysis = client_facts.get("analysis") or {}
    ihs = analysis.get("ihs") or {}
    breakdown = _ihs_breakdown(analysis)
    name = (client_facts.get("personal") or {}).get("name") or "Client"
    alerts = sorted(breakdown.items(), key=lambda kv: _score_value(kv[1], 0))[:3]
    story = []
    top = []
    if os.path.exists(LOGO_PATH):
        logo = Image(LOGO_PATH, width=1.0 * inch, height=0.42 * inch)
        
        def draw_badge(c, x, y, w, h):
            c.saveState()
            c.setStrokeColor(colors.HexColor("#E3D8C7"))
            c.setFillColor(colors.HexColor("#FDF8E7"))
            c.roundRect(x, y, w, h, 8, stroke=1, fill=1)
            c.setFillColor(colors.HexColor("#A8813C"))
            c.setFont("Helvetica", 8)
            c.drawCentredString(x + w/2, y + h/2 - 3, "Confidential")
            c.restoreState()
            
        badge = CanvasBlock(60, 16, draw_badge)
        generated_ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
        date_str = f"<font size='9' color='#2C3E50'><b>{generated_ist.strftime('%d %B %Y')}</b></font><br/><font size='8' color='#6E8094'>{generated_ist.strftime('%I:%M %p IST')}</font>"
        date_para = Paragraph(date_str, ParagraphStyle("mk_cover_date", parent=styles["small"], alignment=TA_RIGHT))
        
        right_table = Table([[date_para, badge]], colWidths=[120, 60])
        right_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (0, 0), "RIGHT"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ]))
        
        top.append([logo, right_table])
        table = Table(top, colWidths=[300, 190])
        table.setStyle(TableStyle([("ALIGN", (1, 0), (1, 0), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(table)
        story.append(Spacer(1, 40))
        
    subtitle = "A comprehensive analysis across <font color='#2C3E50'><b>Six Critical Dimensions</b></font> of your financial wellbeing — designed to help you take clear, prioritised action."
    title = "Your Financial<br/><font name='Times-Italic' color='#B8860B'>Health Report</font>"
    story += [
        Paragraph("<font size='9' color='#88A0B9' name='Helvetica-Bold'>PREPARED FOR</font>", ParagraphStyle("mk_pf_left", parent=styles["cover_subtitle"], alignment=TA_LEFT, spaceAfter=4)),
        Paragraph(f"{sanitize_pdf_text(name)}", ParagraphStyle("mk_cover_name_left", parent=styles["cover_name"], alignment=TA_LEFT)),
        Spacer(1, 10),
        CanvasBlock(500, 1, lambda c, x, y, w, h: [c.setStrokeColor(colors.HexColor("#E3D8C7")), c.setLineWidth(0.8), c.line(0, 0, w, 0)]),
        Spacer(1, 18),
        Paragraph(f"{title}", ParagraphStyle("mk_cover_title_left", parent=styles["cover_title"], alignment=TA_LEFT)),
        Spacer(1, 8),
        Paragraph(f"{subtitle}", ParagraphStyle("mk_cover_sub_left", parent=styles["cover_subtitle"], alignment=TA_LEFT)),
        Spacer(1, 30),
    ]
    status_label = sanitize_pdf_text(_format_status(ihs.get("band") or "Needs Attention"))
    alert_style = ParagraphStyle("mk_cover_alert", parent=styles["body"], alignment=TA_LEFT, textColor=colors.HexColor("#5E738A"), leading=14)
    
    alert_block = [
        Paragraph("<font size='9' color='#88A0B9' name='Helvetica-Bold'>OVERALL STATUS</font>", ParagraphStyle("mk_status_label", parent=styles["small"], spaceAfter=6)),
        Paragraph(status_label, ParagraphStyle("mk_cover_status_left", parent=styles["cover_status"], alignment=TA_LEFT))
    ]
    
    for key, _ in alerts:
        # Drawing a small red exclamation in a circle
        def draw_alert_icon(c, x, y, w, h):
            c.saveState()
            c.setFillColor(colors.HexColor("#FDF1F0"))
            c.setStrokeColor(colors.HexColor("#F1B2AD"))
            c.circle(x + 6, y + 6, 6, fill=1, stroke=1)
            c.setFillColor(colors.HexColor("#CB3E2D"))
            c.setFont("Helvetica-Bold", 8)
            c.drawCentredString(x + 6, y + 3, "!")
            c.restoreState()
            
        icon = CanvasBlock(16, 12, draw_alert_icon)
        text = Paragraph(f"{key.replace('_', ' ').capitalize()} needs attention", alert_style)
        alert_table = Table([[icon, text]], colWidths=[20, 260])
        alert_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 0)]))
        alert_block.append(alert_table)
        alert_block.append(Spacer(1, 4))
        
    status_table = Table(
        [[CanvasBlock(180, 160, lambda c, x, y, w, h: draw_arc_gauge(c, 70, 90, 56, ihs.get("score"), score_font_size=36)), Table([[x] for x in alert_block], colWidths=[280])]],
        colWidths=[180, 310],
    )
    status_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(status_table)
    story += [Spacer(1, 20), Paragraph("11-page analysis covering protection, portfolio, liquidity, goals, tax and action planning", ParagraphStyle("mk_cover_foot", parent=styles["small"], alignment=TA_LEFT))]
    # Prominent disclaimer block on the cover page
    disclaimer_block = Table(
        [[Paragraph(
            "<b><font color='#A8813C'>IMPORTANT DISCLAIMERS</font></b><br/>"
            "<font color='#4A5568' size='8'>"
            "• This is an <b>educational financial analysis tool</b>, not personalised investment advice.<br/>"
            "• All recommendations are <b>indicative</b> and based on the data and assumptions you provided.<br/>"
            f"• Calculations use <b>standard benchmarks</b> (6-month emergency fund, {int(LIFE_COVER_MULTIPLE)}× income term cover, &lt;40% EMI/income, 12% long-term equity CAGR).<br/>"
            "• <b>Market and regulatory conditions change</b>; revisit this plan at least annually or after major life events.<br/>"
            "• <b>Product selection and execution are your responsibility.</b> Consult a SEBI-registered Investment Advisor before acting.<br/>"
            "• We do <b>not recommend specific securities or products</b> and do not earn commissions from any provider."
            "</font>",
            styles["small"],
        )]],
        colWidths=[490],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FDF8E7")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E3D8C7")),
            ("LEFTPADDING", (0, 0), (-1, -1), 14),
            ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ]),
    )
    story.append(disclaimer_block)
    return story
