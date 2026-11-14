"""Assemble the financial health report PDF."""

import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, SimpleDocTemplate

from backend.analysis.client_facts import _build_client_facts
from backend.analysis.health_score import compute_financial_health_score
from backend.config import LOGO_PATH
from backend.reports.helpers import _allocation_output
from backend.reports.pages.action_plan import build_page_action_plan
from backend.reports.pages.cashflow import build_page_cashflow_sip
from backend.reports.pages.cover import build_page_cover
from backend.reports.pages.executive_summary import build_page_executive_summary
from backend.reports.pages.goals import build_page_goal_feasibility
from backend.reports.pages.liquidity import build_page_liquidity
from backend.reports.pages.portfolio_debt import build_page_portfolio_debt
from backend.reports.pages.protection import build_page_protection
from backend.reports.pages.review import build_page_review
from backend.reports.pages.snapshot import build_page_snapshot
from backend.reports.pages.tax import build_page_tax
from backend.reports.theme import MEERKAT_BG, MEERKAT_NAVY, _strip_in_app_report_text


def _meerkat_page_background(c, doc):
    c.saveState()
    c.setFillColor(MEERKAT_BG)
    page_w, page_h = A4
    c.rect(0, 0, page_w, page_h, stroke=0, fill=1)
    
    if doc.page > 1:
        if os.path.exists(LOGO_PATH):
            try:
                c.drawImage(LOGO_PATH, doc.leftMargin, page_h - 0.65 * inch, width=18, height=18, preserveAspectRatio=True, mask='auto')
                c.setFillColor(MEERKAT_NAVY)
                c.setFont("Times-Roman", 12)
                c.drawString(doc.leftMargin + 24, page_h - 0.58 * inch, "M E E R K A T")
            except Exception:
                pass
                
        titles = {
            2: "01 — Financial Snapshot",
            3: "02 — Executive Summary",
            4: "03 — Protection Gap",
            5: "04 — Portfolio & Debt",
            6: "05 — Liquidity Analysis",
            7: "06 — Goal Feasibility",
            8: "07 — Cash Flow & SIP Plan",
            9: "08 — Tax Optimisation",
            10: "09 — Action Plan",
            11: "10 — Review & Next Steps"
        }
        title_str = titles.get(doc.page, "")
        
        c.setFillColor(MEERKAT_NAVY)
        c.setFont("Helvetica-Bold", 10)
        if title_str:
            c.drawRightString(page_w - doc.rightMargin, page_h - 0.58 * inch, title_str)
            
        c.setFillColor(colors.HexColor("#7F8C8D"))
        c.setFont("Helvetica", 8)
        c.drawRightString(page_w - doc.rightMargin, page_h - 0.70 * inch, f"Page {doc.page} of 11")
        
        c.setFillColor(colors.HexColor("#95A5A6"))
        c.setFont("Helvetica", 7)
        c.drawString(doc.leftMargin, 0.4 * inch, "Educational analysis tool · Not financial advice · Consult a SEBI-registered Investment Advisor")
        c.drawRightString(page_w - doc.rightMargin, 0.4 * inch, f"Page {doc.page} of 11")
        
    c.restoreState()


def generate_financial_plan_pdf(q: dict, analysis: dict, output_path: str, doc_insights=None, narratives=None):
    """Generate the 11-page Meerkat Financial Health Report."""
    client_facts = _build_client_facts(q, analysis, doc_insights)
    try:
        hs = compute_financial_health_score(analysis, client_facts)
        if "ihs" not in analysis:
            analysis["ihs"] = {}
        analysis["ihs"]["score"] = hs.get("overall", 0)
        analysis["ihs"]["band"] = hs.get("overall_label", "")
        # Inject component breakdown so _ihs_component_score() can find
        # protection, liquidity, debt_management, etc. for urgency tags.
        analysis["ihs"]["breakdown"] = hs.get("components") or {}
        if "analysis" not in client_facts:
            client_facts["analysis"] = {}
        if "ihs" not in client_facts["analysis"]:
            client_facts["analysis"]["ihs"] = {}
        client_facts["analysis"]["ihs"]["score"] = hs.get("overall", 0)
        client_facts["analysis"]["ihs"]["band"] = hs.get("overall_label", "")
        client_facts["analysis"]["ihs"]["breakdown"] = hs.get("components") or {}
    except Exception:
        pass
    narratives = _strip_in_app_report_text(narratives or {})
    allocation = _allocation_output(client_facts)
    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=0.65 * inch, leftMargin=0.65 * inch, topMargin=0.85 * inch, bottomMargin=0.55 * inch)
    builders = [
        lambda: build_page_cover(client_facts, allocation, narratives),
        lambda: build_page_snapshot(client_facts, allocation),
        lambda: build_page_executive_summary(client_facts, allocation, narratives),
        lambda: build_page_protection(client_facts, allocation),
        lambda: build_page_portfolio_debt(client_facts, allocation),
        lambda: build_page_liquidity(client_facts, allocation),
        lambda: build_page_goal_feasibility(client_facts, allocation),
        lambda: build_page_cashflow_sip(client_facts, allocation, narratives),
        lambda: build_page_tax(client_facts, allocation),
        lambda: build_page_action_plan(client_facts, allocation),
        lambda: build_page_review(client_facts, allocation),
    ]
    story = []
    for idx, builder in enumerate(builders):
        story.extend(builder())
        if idx < len(builders) - 1:
            story.append(PageBreak())
    doc.build(story, onFirstPage=_meerkat_page_background, onLaterPages=_meerkat_page_background)
