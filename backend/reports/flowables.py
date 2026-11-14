"""Custom ReportLab flowables and table/tag builders."""

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Flowable, Paragraph, Spacer, Table, TableStyle

from backend.reports.drawing import (
    draw_coverage_bar,
    draw_equity_gauges,
    draw_kpi_card,
    draw_saving_card,
    draw_tag,
)
from backend.reports.helpers import _num, _urgency
from backend.reports.theme import (
    MEERKAT_CARD,
    MEERKAT_GREEN,
    MEERKAT_LINE,
    MEERKAT_NAVY,
    MEERKAT_ORANGE,
    MEERKAT_RED,
    _report_styles,
    _strip_in_app_report_text,
    sanitize_pdf_text,
)


class CanvasBlock(Flowable):
    def __init__(self, width, height, draw_fn):
        super().__init__()
        self.width = width
        self.height = height
        self.draw_fn = draw_fn

    def wrap(self, avail_width, avail_height):
        return self.width, self.height

    def draw(self):
        self.draw_fn(self.canv, 0, 0, self.width, self.height)


class DrawingFlowable(Flowable):
    def __init__(self, draw_fn, width, height, *args, **kwargs):
        super().__init__()
        self.draw_fn = draw_fn
        self.width = width
        self.height = height
        self.args = args
        self.kwargs = kwargs

    def wrap(self, avail_width, avail_height):
        return self.width, self.height

    def draw(self):
        self.draw_fn(self.canv, 0, 0, *self.args, **self.kwargs)


class TagFlowable(Flowable):
    def __init__(self, label, height=15, bg_tint=False):
        super().__init__()
        self.label = str(label or "-").upper().strip()
        self.bg_tint = bg_tint
        self.width = len(self.label) * 6.0 + 20
        self.height = height

    def wrap(self, avail_width, avail_height):
        return self.width, self.height

    def draw(self):
        draw_tag(self.canv, 0, 1, self.label, bg_tint=self.bg_tint)


class KPIFlowable(Flowable):
    def __init__(self, tiles, width=500, height=70, card_height=54, gap=9, card_y=8):
        super().__init__()
        self.tiles = tiles
        self.width = width
        self.height = height
        self.card_height = card_height
        self.gap = gap
        self.card_y = card_y

    def wrap(self, avail_width, avail_height):
        return self.width, self.height

    def draw(self):
        count = max(1, len(self.tiles))
        tile_w = (self.width - self.gap * (count - 1)) / count
        for idx, tile in enumerate(self.tiles):
            x = idx * (tile_w + self.gap)
            draw_kpi_card(
                self.canv,
                x, self.card_y, tile_w, self.card_height,
                tile.get("label"),
                tile.get("value"),
                sub_value=tile.get("note"),
                sub_colour=tile.get("note_color")
            )


class ScoreBarFlowable(Flowable):
    def __init__(self, label, score, band=None, width=500, height=24):
        super().__init__()
        self.label = label
        self.score = max(0, min(100, _num(score, 0)))
        self.band = band or _urgency(self.score)
        self.width = width
        self.height = height

    def wrap(self, avail_width, avail_height):
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.setFillColor(MEERKAT_NAVY)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(0, 13, self.label)
        bar_x = 140
        bar_w = 245
        colour = MEERKAT_RED if self.score < 40 else MEERKAT_ORANGE if self.score < 75 else MEERKAT_GREEN
        c.setFillColor(colors.HexColor("#E8E8E8"))
        c.rect(bar_x, 9, bar_w, 8, stroke=0, fill=1)
        c.setFillColor(colour)
        c.rect(bar_x, 9, bar_w * self.score / 100, 8, stroke=0, fill=1)
        c.setFillColor(MEERKAT_NAVY)
        c.setFont("Helvetica-Bold", 8)
        c.drawRightString(self.width - 72, 13, f"{self.score:.0f}/100")
        draw_tag(c, self.width - 62, 7, self.band)


class GaugePairFlowable(Flowable):
    def __init__(self, current, target, ar=None, width=500, height=132):
        super().__init__()
        self.current = current
        self.target = target
        self.ar = ar
        self.width = width
        self.height = height

    def wrap(self, avail_width, avail_height):
        return self.width, self.height

    def draw(self):
        c = self.canv
        band = (self.ar or {}).get("recommendedEquityBand") or {}
        b_min = _num(band.get("min"), 40)
        b_max = _num(band.get("max"), 60)
        draw_equity_gauges(c, 130, 370, 60, 35, _num(self.current, 0), _num(self.target, 0), b_min, b_max)


class CoverageFlowable(Flowable):
    def __init__(self, pct, width=92, height=12):
        super().__init__()
        self.pct = pct
        self.width = width
        self.height = height

    def wrap(self, avail_width, avail_height):
        return self.width, self.height

    def draw(self):
        draw_coverage_bar(self.canv, 0, 0, self.width, self.pct, height=self.height)


class SavingCardFlowable(Flowable):
    def __init__(self, amount, width=220, height=140):
        super().__init__()
        self.amount = amount
        self.width = width
        self.height = height
    def wrap(self, aw, ah): return self.width, self.height
    def draw(self): draw_saving_card(self.canv, 0, 0, self.width, self.height, self.amount)


class InsightFlowable(Flowable):
    def __init__(self, title, text, width=500, height=80):
        super().__init__()
        self.title = title
        self.text = text
        self.width = width
        self.height = height

    def wrap(self, avail_width, avail_height):
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.saveState()
        
        # Background
        c.setFillColor(colors.HexColor("#F9F7F2"))
        c.roundRect(0, 0, self.width, self.height, radius=6, fill=1, stroke=0)
        
        # Gold bar on left
        c.setFillColor(colors.HexColor("#A8813C"))
        c.rect(0, 0, 4, self.height, fill=1, stroke=0)
        
        # Title
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(16, self.height - 24, self.title)
        
        # Text
        styles = _report_styles()
        p = Paragraph(self.text, ParagraphStyle("insight", parent=styles["small"], fontSize=9, leading=14))
        w, h = p.wrap(self.width - 40, self.height - 40)
        p.drawOn(c, 16, self.height - 32 - h)
        
        c.restoreState()


def _tag(label, bg_tint=False):
    return ("__TAG__", label, bg_tint)


def _page_header(section_no, title, sublabel):
    styles = _report_styles()
    return [
        Paragraph(str(sublabel or "").upper(), styles["label"]),
        Spacer(1, 3),
        draw_section_badge(section_no, title),
    ]


def draw_section_badge(section_no, title, width=500):
    styles = _report_styles()
    table = Table(
        [[Paragraph(section_no, styles["table_head"]), Paragraph(title.upper(), styles["table_head"])]],
        colWidths=[38, width - 38],
        hAlign="LEFT",
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), MEERKAT_NAVY),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#777777")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _styled_table(rows, col_widths, header=True, style_type="default"):
    styles = _report_styles()
    data = []
    for r, row in enumerate(rows):
        rendered = []
        for cell in row:
            if isinstance(cell, Flowable):
                rendered.append(cell)
            elif isinstance(cell, tuple) and len(cell) >= 2 and cell[0] == "__TAG__":
                bg_tint = cell[2] if len(cell) > 2 else False
                rendered.append(TagFlowable(cell[1], bg_tint=bg_tint))
            else:
                if style_type == "light":
                    style = styles["label"] if (header and r == 0) else styles["table"]
                    if header and r == 0:
                        rendered.append(Paragraph(f"<font color='#A8813C'>{sanitize_pdf_text(str(cell))}</font>", style))
                    else:
                        rendered.append(Paragraph(sanitize_pdf_text(str(cell)), style))
                else:
                    rendered.append(Paragraph(sanitize_pdf_text(str(cell)), styles["table_head" if header and r == 0 else "table"]))
        data.append(rendered)
    table = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1 if header else 0)
    
    if style_type == "light":
        styles_list = [
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#EEEEEE")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]
        if header:
            styles_list.append(("LINEABOVE", (0, 0), (-1, 0), 0.5, colors.HexColor("#EAE2D6")))
            styles_list.append(("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#EAE2D6")))
            styles_list.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FAFAFA")))
        styles_list.append(("LEFTPADDING", (0, 0), (0, -1), 16))
        styles_list.append(("RIGHTPADDING", (-1, 0), (-1, -1), 16))
        table.setStyle(TableStyle(styles_list))
    elif style_type == "compact_light":
        table.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#EEEEEE")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
    elif style_type == "light_right":
        table.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#EEEEEE")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ]))

    else:
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), MEERKAT_NAVY if header else MEERKAT_CARD),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white if header else colors.black),
            ("GRID", (0, 0), (-1, -1), 0.45, MEERKAT_LINE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
    return table


def _llm_paragraph(narratives, key):
    sec = ((narratives or {}).get(key) or {})
    for text in (sec.get("paragraphs") or []) + (sec.get("bullets") or []):
        cleaned = _strip_in_app_report_text(str(text or ""))
        if cleaned:
            return cleaned
    return ""
