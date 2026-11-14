"""Colours, text sanitising and paragraph styles for the financial health report."""

import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet


# PDF text sanitization helper to avoid unsupported glyphs
def sanitize_pdf_text(s):
    try:
        if s is None:
            return ""
        t = str(s)
        # Normalize common unicode symbols to ASCII
        t = (t.replace("•", "-")
               .replace("–", "-")
               .replace("—", "-")
               .replace("×", "x")
               .replace("₹", "Rs.")
               .replace("“", '"')
               .replace("”", '"')
               .replace("’", "'")
               .replace("‘", "'")
               .replace("≈", "~"))
        # Remove non-printable/non-ASCII chars (preserve tab/newline/carriage return + printable ASCII 0x20-0x7E)
        t = re.sub(r"[^\t\n\r -~]", "", t)
        # Collapse repeated dashes from earlier bullet normalization
        t = re.sub(r"-{2,}", "-", t)
        # Soft-break very long unbroken tokens to allow wrapping in PDF tables/paras
        t = re.sub(r"([A-Za-z0-9:/\.\-_]{30})(?=[A-Za-z0-9:/\.\-_])", r"\1 ", t)
        # Collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t
    except Exception:
        return str(s) if s is not None else ""


# =============================================================================
# Meerkat Financial Health Report v5 - presentation layer
# =============================================================================

MEERKAT_BG = colors.HexColor("#F5F0E8")


MEERKAT_NAVY = colors.HexColor("#1A1A2E")


MEERKAT_RED = colors.HexColor("#C0392B")


MEERKAT_ORANGE = colors.HexColor("#E67E22")


MEERKAT_GREEN = colors.HexColor("#27AE60")


MEERKAT_CARD = colors.HexColor("#FAFAFA")


MEERKAT_GOLD = colors.HexColor("#B8860B")


MEERKAT_LINE = colors.HexColor("#DDDDDD")


IN_APP_REPORT_BLOCKLIST = ("in-app", "nudge", "push notification", "mobile alert")


def _strip_in_app_report_text(value):
    if isinstance(value, str):
        return "\n".join(
            line for line in value.splitlines()
            if not any(term in line.lower() for term in IN_APP_REPORT_BLOCKLIST)
        ).strip()
    if isinstance(value, list):
        return [_strip_in_app_report_text(v) for v in value if _strip_in_app_report_text(v)]
    if isinstance(value, dict):
        return {k: _strip_in_app_report_text(v) for k, v in value.items()}
    return value


def _report_styles():
    base = getSampleStyleSheet()
    return {
        "cover_name": ParagraphStyle("mk_cover_name", parent=base["Title"], fontName="Times-Roman", fontSize=34, leading=40, textColor=MEERKAT_NAVY, spaceAfter=8, alignment=TA_LEFT),
        "title": ParagraphStyle("mk_title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=MEERKAT_NAVY, alignment=TA_LEFT),
        "cover_title": ParagraphStyle("mk_cover_title", parent=base["Title"], fontName="Times-Roman", fontSize=32, leading=38, textColor=MEERKAT_NAVY, alignment=TA_LEFT),
        "cover_subtitle": ParagraphStyle("mk_cover_subtitle", parent=base["BodyText"], fontName="Times-Italic", fontSize=11, leading=16, textColor=colors.HexColor("#555555"), alignment=TA_LEFT),
        "section": ParagraphStyle("mk_section", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=colors.HexColor("#666666"), uppercase=True),
        "h2": ParagraphStyle("mk_h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=MEERKAT_NAVY, spaceBefore=4, spaceAfter=6),
        "cover_status": ParagraphStyle("mk_cover_status", parent=base["Heading2"], fontName="Times-Roman", fontSize=24, leading=28, textColor=MEERKAT_ORANGE, alignment=TA_LEFT, spaceBefore=4, spaceAfter=12),
        "body": ParagraphStyle("mk_body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5, leading=12, textColor=colors.HexColor("#222222")),
        "small": ParagraphStyle("mk_small", parent=base["BodyText"], fontName="Helvetica", fontSize=8, leading=10, textColor=colors.HexColor("#666666")),
        "label": ParagraphStyle("mk_label", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=colors.HexColor("#666666")),
        "table": ParagraphStyle("mk_table", parent=base["BodyText"], fontName="Helvetica", fontSize=8.2, leading=10),
        "table_head": ParagraphStyle("mk_table_head", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=8.2, leading=10, textColor=colors.white),
        "table_head_dark": ParagraphStyle("mk_table_head_dark", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=8.2, leading=10, textColor=colors.HexColor("#475569")),
    }
