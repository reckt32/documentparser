"""Low-level canvas drawing primitives (tags, KPI cards, gauges, bars)."""

import math

from reportlab.lib import colors

from backend.reports.helpers import _fmt_rs, _num


def draw_saving_card(canvas, x, y, w, h, amount):
    canvas.saveState()
    # Background
    canvas.setFillColor(colors.HexColor("#F9F7F2"))
    canvas.roundRect(x, y, w, h, radius=8, fill=1, stroke=0)
    canvas.setStrokeColor(colors.HexColor("#E2E8F0"))
    canvas.setLineWidth(0.5)
    canvas.roundRect(x, y, w, h, radius=8, fill=0, stroke=1)
    
    # Label
    canvas.setFillColor(colors.HexColor("#A8813C"))
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawCentredString(x + w/2, y + h - 24, "POTENTIAL ANNUAL SAVING")
    
    # Value
    canvas.setFillColor(colors.HexColor("#27AE60"))
    canvas.setFont("Times-Bold", 32)
    canvas.drawCentredString(x + w/2, y + h/2 - 10, f"{_fmt_rs(amount, False)}")
    
    # Subtext
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(x + w/2, y + 20, "by switching to the New Tax Regime")
    
    canvas.restoreState()


def _safe_canvas_arc(canvas_obj, x1, y1, x2, y2, start_ang, extent, min_extent=1e-6):
    """Draw arc only when geometry and extent are valid to avoid ReportLab runtime failures."""
    try:
        fx1, fy1, fx2, fy2 = float(x1), float(y1), float(x2), float(y2)
        fstart = float(start_ang)
        fextent = float(extent)
    except Exception:
        return
    if not all(math.isfinite(v) for v in (fx1, fy1, fx2, fy2, fstart, fextent)):
        return
    if fx2 <= fx1 or fy2 <= fy1:
        return
    if abs(fextent) <= float(min_extent):
        return
    canvas_obj.arc(fx1, fy1, fx2, fy2, startAng=fstart, extent=fextent)


def draw_tag(canvas, x, y, label, font_size=8, bg_tint=False):
    # y is baseline of text
    pad_x, pad_y = 8, 3
    text_width = canvas.stringWidth(label, 'Helvetica-Bold', font_size)
    rect_w = text_width + pad_x * 2 + 2
    rect_h = font_size + pad_y * 2
    colour_map = {
        'IMMEDIATE': '#C0392B', 'CRITICAL': '#C0392B', 'UNDERINSURED': '#C0392B', 'URGENT': '#C0392B', 'INSUFFICIENT': '#C0392B',
        'STRESSED': '#E67E22', 'NEEDS ATTENTION': '#E67E22', 'ATTENTION': '#E67E22', 'ASSESS & IMPROVE': '#E67E22', 'MODERATE': '#E67E22', 'GAP': '#E67E22', 'LOW': '#E67E22', 'UPGRADE': '#E67E22', 'UPGRADE RECOMMENDED': '#E67E22', 'REVIEW': '#E67E22', 'AVERAGE': '#E67E22', 'POOR': '#E67E22',
        'GOOD': '#27AE60', 'MAINTAIN': '#27AE60', 'WELL OPTIMISED': '#27AE60', 'ADEQUATE': '#27AE60', 'HEALTHY': '#27AE60', 'FUNDED': '#27AE60', 'COMFORTABLE': '#27AE60', 'EXCELLENT': '#27AE60',
        'FEASIBLE': '#2A557E', 'OVER-INSURED': '#2A557E', 'NOT COVERED': '#C0392B', 'PENDING': '#C0392B', 'PARTIAL': '#E67E22',
    }
    hex_col = colour_map.get(label.upper(), '#888888')
    r, g, b = int(hex_col[1:3],16)/255, int(hex_col[3:5],16)/255, int(hex_col[5:7],16)/255
    canvas.setStrokeColorRGB(r, g, b)
    if bg_tint:
        canvas.setFillColorRGB(r * 0.15 + 0.85, g * 0.15 + 0.85, b * 0.15 + 0.85)
    else:
        canvas.setFillColorRGB(1, 1, 1)
    canvas.setLineWidth(0.8)
    canvas.roundRect(x, y - pad_y, rect_w, rect_h, radius=rect_h/2, fill=1, stroke=1)
    canvas.setFillColorRGB(r, g, b)
    canvas.setFont('Helvetica-Bold', font_size)
    canvas.drawString(x + pad_x, y + 1, label)


def draw_kpi_card(canvas, x, y, w, h, label, value, sub_value=None, sub_colour=None):
    is_active_sip = ("ACTIVE SIP" in label.upper())
    is_red = ("EXPENSES" in label.upper())
    is_tan = ("INSURANCE" in label.upper())
    
    if is_active_sip:
        bg_color = (0.99, 0.95, 0.94)  # Very light red/pink
        border_color = (0.95, 0.8, 0.77)
    else:
        bg_color = (1, 1, 1)
        border_color = (0.92, 0.92, 0.92)
        
    canvas.setFillColorRGB(*bg_color)
    canvas.setStrokeColorRGB(*border_color)
    canvas.setLineWidth(0.8)
    canvas.roundRect(x, y, w, h, radius=6, fill=1, stroke=1)
    
    # Label
    canvas.setFillColorRGB(0.5, 0.5, 0.5)
    canvas.setFont('Helvetica-Bold', 6)
    canvas.drawCentredString(x + w/2, y + h - 14, label.upper())
    
    # Value
    if is_active_sip or is_red:
        canvas.setFillColor(colors.HexColor("#C0392B")) # Meerkat Red
    elif is_tan:
        canvas.setFillColor(colors.HexColor("#A8813C")) # Meerkat Gold
    elif "SURPLUS" in label.upper() or "AVAILABLE" in label.upper() or "INCOME" in label.upper():
        canvas.setFillColor(colors.HexColor("#27AE60")) # Meerkat Green
    elif "PORTFOLIO" in label.upper():
        canvas.setFillColor(colors.HexColor("#2C3E50")) # Meerkat Navy
    else:
        canvas.setFillColor(colors.HexColor("#27AE60")) # Meerkat Green
        
    canvas.setFont('Helvetica-Bold', 12)
    canvas.drawCentredString(x + w/2, y + h - 34, value)
    
    # Sub value
    if sub_value:
        r, g, b = (0.9, 0.49, 0.13) if sub_colour == 'orange' else (0.4, 0.4, 0.4)
        if is_active_sip:
            r, g, b = (0.8, 0.4, 0.2)
        canvas.setFillColorRGB(r, g, b)
        canvas.setFont('Helvetica', 8)
        canvas.drawString(x + 12, y + h - 56, sub_value)


def draw_score_bar(canvas, x, y, bar_width, label, score, tag_label):
    label_w = 130
    score_w = 40
    actual_bar_w = bar_width - label_w - score_w - 20
    # Label
    canvas.setFillColorRGB(0.1, 0.1, 0.18)
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(x, y + 2, label)
    bx = x + label_w
    # Background bar
    canvas.setFillColorRGB(0.91, 0.91, 0.91)
    canvas.rect(bx, y, actual_bar_w, 10, fill=1, stroke=0)
    # Foreground bar
    fill_w = (score / 100) * actual_bar_w
    if score < 40:
        canvas.setFillColorRGB(0.75, 0.22, 0.17)
    elif score < 75:
        canvas.setFillColorRGB(0.9, 0.49, 0.13)
    else:
        canvas.setFillColorRGB(0.15, 0.68, 0.38)
    canvas.rect(bx, y, fill_w, 10, fill=1, stroke=0)
    # Score text
    canvas.setFillColorRGB(0.1, 0.1, 0.18)
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(bx + actual_bar_w + 6, y + 2, f'{score}/100')
    # Tag
    draw_tag(canvas, bx + actual_bar_w + 50, y, tag_label)


def _draw_dimension_score_bars(canvas, left_margin, starting_y, content_width, dimension_rows):
    bar_x = left_margin + 160
    bar_w = content_width - 160 - 80
    y = starting_y
    for label, score, tag in dimension_rows:
        canvas.setFillColorRGB(0.2, 0.2, 0.2)
        canvas.setFont("Helvetica", 10)
        canvas.drawString(left_margin, y + 2, str(label))
        canvas.setFillColorRGB(0.92, 0.92, 0.92)
        canvas.roundRect(bar_x, y, bar_w, 8, radius=4, fill=1, stroke=0)
        score_val = _num(score, 0)
        fill_w = (score_val / 100.0) * bar_w
        if score_val < 40:
            color = (0.75, 0.22, 0.17)
            tag_label = "Critical"
        elif score_val < 75:
            color = (0.9, 0.49, 0.13)
            tag_label = "Attention"
        else:
            color = (0.15, 0.68, 0.38)
            tag_label = "Good"
        canvas.setFillColorRGB(*color)
        canvas.roundRect(bar_x, y, fill_w, 8, radius=4, fill=1, stroke=0)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(bar_x + bar_w + 12, y + 1, str(int(score_val)))
        draw_tag(canvas, bar_x + bar_w + 40, y, tag_label, bg_tint=True)
        y -= 28


def draw_arc_gauge(canvas, cx, cy, radius, score, max_score=100,
                   label='Financial Health', score_font_size=28, colour=None):
    stroke_bg = 6      # background arc stroke width
    stroke_fg = 9      # foreground arc stroke width
    start_angle = 225
    total_sweep = 270
    min_sweep = 1e-6
    score_val = _num(score, 0)
    max_score_val = _num(max_score, 100)
    if max_score_val <= 0:
        max_score_val = 100

    # Background arc
    canvas.setStrokeColorRGB(0.87, 0.87, 0.87)
    canvas.setLineWidth(stroke_bg)
    canvas.setLineCap(1)  # round caps for smoother arc ends
    _safe_canvas_arc(
        canvas,
        cx - radius, cy - radius, cx + radius, cy + radius,
        start_angle, -total_sweep
    )

    # Foreground arc
    sweep = (score_val / max_score_val) * total_sweep
    sweep = max(0.0, min(float(total_sweep), float(sweep)))
    if colour:
        canvas.setStrokeColor(colour)
    else:
        if score_val < 40:
            canvas.setStrokeColorRGB(0.75, 0.22, 0.17)
        elif score_val < 75:
            canvas.setStrokeColorRGB(0.9, 0.49, 0.13)
        else:
            canvas.setStrokeColorRGB(0.15, 0.68, 0.38)
    canvas.setLineWidth(stroke_fg)
    canvas.setLineCap(1)  # round caps for smoother arc ends
    if sweep > min_sweep:
        _safe_canvas_arc(
            canvas,
            cx - radius, cy - radius, cx + radius, cy + radius,
            start_angle, -sweep
        )

    # Score number
    canvas.setFillColorRGB(0.1, 0.1, 0.18)
    canvas.setFont('Helvetica-Bold', score_font_size)
    
    if max_score == 100 and label == 'Financial Health':
        canvas.drawCentredString(cx, cy + 4, str(int(score_val)))
        canvas.setFont('Helvetica', score_font_size * 0.35)
        canvas.setFillColorRGB(0.5, 0.5, 0.5)
        canvas.drawCentredString(cx, cy - 12, "/100")
    else:
        canvas.drawCentredString(cx, cy + 8, str(int(score_val)))
        if label:
            canvas.setFillColorRGB(0.5, 0.5, 0.5)
            canvas.setFont('Helvetica', 9)
            canvas.drawCentredString(cx, cy - 8, label)


def draw_coverage_bar(canvas, x, y, width, coverage_pct, inner_text="", bar_color=None, height=12):
    coverage_val = _num(coverage_pct, 0)
    canvas.setFillColorRGB(0.88, 0.90, 0.88)
    h = height
    canvas.roundRect(x, y, width, h, radius=h/2, fill=1, stroke=0)
    if coverage_val > 0:
        fill_w = min(coverage_val / 100, 1.0) * width
        if bar_color:
            canvas.setFillColorRGB(*bar_color)
        else:
            if coverage_val < 50:
                canvas.setFillColorRGB(0.75, 0.22, 0.17)
            elif coverage_val < 80:
                canvas.setFillColorRGB(0.9, 0.49, 0.13)
            else:
                canvas.setFillColorRGB(0.15, 0.68, 0.38)
        fill_w = max(fill_w, h)
        canvas.roundRect(x, y, fill_w, h, radius=h/2, fill=1, stroke=0)
    
    if inner_text:
        canvas.setFillColorRGB(1, 1, 1)
        canvas.setFont('Helvetica-Bold', 8)
        canvas.drawString(x + 8, y + 3, inner_text)


def draw_equity_gauges(canvas, cx_left, cx_right, cy, radius, current_pct, target_pct, band_low, band_high):
    min_sweep = 1e-6
    for cx, pct, is_current in [(cx_left, current_pct, True), (cx_right, target_pct, False)]:
        canvas.setStrokeColorRGB(0.87, 0.87, 0.87)
        canvas.setLineWidth(6)
        canvas.setLineCap(1)
        _safe_canvas_arc(canvas, cx - radius, cy - radius, cx + radius, cy + radius, -45, 270)
        if is_current and (_num(pct, 0) < _num(band_low, 0) or _num(pct, 0) > _num(band_high, 0)):
            canvas.setStrokeColorRGB(0.75, 0.22, 0.17)
        else:
            canvas.setStrokeColorRGB(0.15, 0.68, 0.38)
        canvas.setLineWidth(9)
        canvas.setLineCap(1)
        sweep = (_num(pct, 0) / 100.0) * 270
        sweep = max(0.0, min(270.0, float(sweep)))
        if sweep > min_sweep:
            _safe_canvas_arc(canvas, cx - radius, cy - radius, cx + radius, cy + radius, -45, sweep)
        canvas.setFillColorRGB(0.1, 0.1, 0.18)
        canvas.setFont("Helvetica-Bold", 18)
        canvas.drawCentredString(cx, cy + 4, f"{int(_num(pct, 0))}%")
        canvas.setFillColorRGB(0.5, 0.5, 0.5)
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(cx, cy - 12, "Current Equity" if is_current else "Target Equity")
    canvas.setFillColorRGB(0.6, 0.6, 0.6)
    canvas.setFont("Helvetica", 16)
    canvas.drawCentredString((cx_left + cx_right) / 2, cy, "→")
