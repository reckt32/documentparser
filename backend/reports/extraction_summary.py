"""Generic PDF summary of the data extracted from an uploaded document."""

import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.config import LOGO_PATH
from backend.reports.theme import sanitize_pdf_text


def get_custom_styles():
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    title_style.fontSize = 22
    title_style.alignment = TA_CENTER
    title_style.spaceAfter = 20
    title_style.textColor = colors.HexColor('#1D3557')
    h1_style = styles['h1']
    h1_style.fontSize = 16
    h1_style.leading = 22
    h1_style.spaceAfter = 12
    h1_style.textColor = colors.HexColor('#457B9D')
    h2_style = styles['h2']
    h2_style.fontSize = 12
    h2_style.leading = 18
    h2_style.spaceBefore = 10
    h2_style.spaceAfter = 6
    h2_style.textColor = colors.HexColor('#E63946')
    body_style = styles['BodyText']
    body_style.fontSize = 10
    body_style.leading = 14
    body_style.spaceAfter = 6
    body_style.wordWrap = 'CJK'
    styles.add(ParagraphStyle(name='Header', fontSize=8, alignment=TA_RIGHT, textColor=colors.grey))
    styles.add(ParagraphStyle(name='Footer', fontSize=8, alignment=TA_CENTER, textColor=colors.grey))
    styles.add(ParagraphStyle(name='TableHead', fontSize=9, fontName='Helvetica-Bold', alignment=TA_LEFT, textColor=colors.white))
    styles.add(ParagraphStyle(name='TableCell', fontSize=9, alignment=TA_LEFT, wordWrap='CJK'))
    return styles


def header(canvas, doc):
    canvas.saveState()
    # Draw logo at top-left in the margin area (above content)
    if os.path.exists(LOGO_PATH):
        try:
            logo_width = 60
            logo_height = 30
            # Position logo in the top margin, well above the content area
            logo_y = doc.height + doc.topMargin + 15
            canvas.drawImage(LOGO_PATH, doc.leftMargin, logo_y, 
                           width=logo_width, height=logo_height, preserveAspectRatio=True, mask='auto')
        except Exception:
            pass  # Skip logo if there's an error loading it
    # Draw date at top-right in the margin
    styles = get_custom_styles()
    p = Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Header'])
    w, h = p.wrap(doc.width - 80, doc.topMargin)
    p.drawOn(canvas, doc.leftMargin + 80, doc.height + doc.topMargin + 20)
    canvas.restoreState()


def footer(canvas, doc):
    canvas.saveState()
    styles = get_custom_styles()
    
    # Educational disclaimer - displayed on every page
    disclaimer_text = (
        "This is an educational analysis tool, not financial advice. "
        "This report is for information purposes only. "
        "Consult a SEBI-registered Investment Advisor before making decisions. "
        "We do not recommend specific securities or products."
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        fontSize=6,
        alignment=TA_CENTER,
        textColor=colors.grey,
        leading=8
    )
    disclaimer = Paragraph(disclaimer_text, disclaimer_style)
    dw, dh = disclaimer.wrap(doc.width, doc.bottomMargin)
    disclaimer.drawOn(canvas, doc.leftMargin, dh + 15)
    
    # Page number
    p = Paragraph(f"Page {doc.page}", styles['Footer'])
    w, h = p.wrap(doc.width, doc.bottomMargin)
    p.drawOn(canvas, doc.leftMargin, h)
    canvas.restoreState()


def build_pdf_story(story, data, doc, styles):
    empty_values = [None, "", "N/A", [], {}]
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        if key in ["transactions", "transactions_json_path"]:
            if key == "transactions_json_path" and isinstance(value, str):
                story.append(Paragraph(sanitize_pdf_text(f"Transactions JSON: {value}"), styles["BodyText"]))
                story.append(Spacer(1, 4))
            continue
        if value in empty_values or key in ['extraction_error', 'llm_error']:
            continue
        key_title = key.replace('_', ' ').title()
        if isinstance(value, list) and value and isinstance(value[0], dict):
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph(key_title, styles['h2']))
            add_dynamic_table(story, value, doc, styles)
        elif isinstance(value, dict):
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph(key_title, styles['h2']))
            build_pdf_story(story, value, doc, styles)
        else:
            formatted_value = f"Rs. {value:,.2f}" if isinstance(value, (int, float)) else value
            p_text = f"<b>{key_title}:</b> {formatted_value}"
            story.append(Paragraph(sanitize_pdf_text(p_text), styles["BodyText"]))
            story.append(Spacer(1, 4))


def add_dynamic_table(story, data_list, doc, styles):
    filtered_data = [row for row in data_list if any(v not in [None, "", "N/A"] for v in row.values())]
    if not filtered_data:
        return
    headers = list(filtered_data[0].keys())
    header_row = [Paragraph(sanitize_pdf_text(h.replace('_', ' ').title()), styles['TableHead']) for h in headers]
    table_data = [header_row]
    for row_dict in filtered_data:
        row = []
        for h in headers:
            cell_value = row_dict.get(h, "")
            if isinstance(cell_value, (int, float)):
                cell_value = f"Rs. {cell_value:,.2f}"
            row.append(Paragraph(sanitize_pdf_text(str(cell_value)), styles['TableCell']))
        table_data.append(row)
    col_count = len(headers) if headers else 1
    col_widths = [doc.width / col_count] * col_count
    table = Table(table_data, hAlign='LEFT', repeatRows=1, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#457B9D')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(table)


def generate_pdf_summary(summary_data, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            rightMargin=inch*0.75, leftMargin=inch*0.75,
                            topMargin=inch, bottomMargin=inch)
    styles = get_custom_styles()
    story = [Paragraph("Holistic Financial Report", styles['Title']), Spacer(1, 0.25 * inch)]
    for section_key, section_content in summary_data.items():
        if not section_content or section_content.get("error"):
            story.append(Paragraph(sanitize_pdf_text(f"Could not process: {section_key}"), styles['h1']))
            if isinstance(section_content, dict) and section_content.get("error"):
                story.append(Paragraph(f"Reason: {section_content['error']}", styles['BodyText']))
            story.append(Spacer(1, 0.2 * inch))
            continue
        story.append(Paragraph(sanitize_pdf_text(section_key.replace('_', ' ').title()), styles['h1']))
        story.append(Spacer(1, 0.15 * inch))
        try:
            from reportlab.graphics.shapes import Drawing
            from reportlab.graphics.charts.piecharts import Pie
            key_l = str(section_key).lower()
            def _add_pie_chart(title, data_map):
                items = []
                for k, v in (data_map or {}).items():
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if fv > 0:
                        items.append((str(k), fv))
                if not items:
                    return
                story.append(Paragraph(title, styles['h2']))
                d = Drawing(300, 220)
                pie = Pie()
                pie.x = 50
                pie.y = 10
                pie.width = 200
                pie.height = 200
                pie.data = [v for _, v in items]
                pie.labels = [k for k, _ in items]
                pie.simpleLabels = True
                pie.slices.strokeWidth = 0.5
                d.add(pie)
                story.append(d)
                story.append(Spacer(1, 0.1 * inch))
            if "bank statement" in key_l:
                acct = (section_content or {}).get("account_summary") or {}
                inflows = acct.get("total_inflows")
                outflows = acct.get("total_outflows")
                if inflows or outflows:
                    _add_pie_chart("Cash Flows (Inflows vs Outflows)", {
                        "Inflows": inflows or 0,
                        "Outflows": outflows or 0,
                    })
                    try:
                        if isinstance(inflows, (int, float)) and isinstance(outflows, (int, float)):
                            net = float(inflows) - float(outflows)
                            note = "Net Cash Surplus" if net >= 0 else "Net Cash Deficit"
                            story.append(Paragraph(f"{note}: Rs. {abs(net):,.0f}", styles['BodyText']))
                            story.append(Spacer(1, 0.1 * inch))
                    except Exception:
                        pass
            if "mutual fund cas" in key_l:
                alloc = (section_content or {}).get("asset_allocation") or {}
                if any(k in alloc for k in ["equity_percentage", "debt_percentage", "hybrid_percentage"]):
                    _add_pie_chart("Asset Allocation (CAS)", {
                        "Equity": alloc.get("equity_percentage", 0),
                        "Debt": alloc.get("debt_percentage", 0),
                        "Hybrid": alloc.get("hybrid_percentage", 0),
                    })
            if key_l.startswith("itr"):
                inc = (section_content or {}).get("income_sources") or {}
                if inc:
                    _add_pie_chart("Income Sources (ITR)", inc)
        except Exception:
            pass
        build_pdf_story(story, section_content, doc, styles)
        story.append(PageBreak())
    if story and isinstance(story[-1], PageBreak):
        story.pop()
    class CustomCanvas(canvas.Canvas):
        page_fn = footer
    doc.build(story, onFirstPage=header, onLaterPages=header, canvasmaker=CustomCanvas)
