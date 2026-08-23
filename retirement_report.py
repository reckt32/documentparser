"""Seven-page retirement MVP report, styled after the supplied ASFS samples."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Iterable, List
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BLUE = colors.HexColor("#0752B8")
NAVY = colors.HexColor("#113466")
GREEN = colors.HexColor("#2BC878")
RED = colors.HexColor("#C93B36")
AMBER = colors.HexColor("#D89021")
INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#7D8797")
LINE = colors.HexColor("#D7E0EC")
PALE_BLUE = colors.HexColor("#F2F6FC")
PALE_GREEN = colors.HexColor("#EAF8F0")
PALE_RED = colors.HexColor("#FCEFED")
WHITE = colors.white


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "RetirementBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.1,
            leading=13.2,
            textColor=INK,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "RetirementSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=10.2,
            textColor=MUTED,
        ),
        "section_label": ParagraphStyle(
            "RetirementSectionLabel",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.2,
            leading=10,
            textColor=GREEN,
            spaceAfter=2,
        ),
        "h1": ParagraphStyle(
            "RetirementH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=BLUE,
            spaceBefore=0,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "RetirementH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=5,
        ),
        "metric": ParagraphStyle(
            "RetirementMetric",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=17,
            textColor=NAVY,
        ),
        "metric_label": ParagraphStyle(
            "RetirementMetricLabel",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.4,
            leading=9,
            textColor=MUTED,
        ),
        "cover_title": ParagraphStyle(
            "RetirementCoverTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=33,
            alignment=TA_CENTER,
            textColor=NAVY,
            spaceAfter=8,
        ),
        "cover_subtitle": ParagraphStyle(
            "RetirementCoverSubtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=11.5,
            leading=15,
            alignment=TA_CENTER,
            textColor=MUTED,
        ),
        "table_header": ParagraphStyle(
            "RetirementTableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.8,
            leading=9.5,
            textColor=WHITE,
        ),
        "table": ParagraphStyle(
            "RetirementTable",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10.4,
            textColor=INK,
        ),
        "table_bold": ParagraphStyle(
            "RetirementTableBold",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.8,
            leading=10.4,
            textColor=INK,
        ),
        "action": ParagraphStyle(
            "RetirementAction",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=INK,
        ),
    }


def _inr(value: Any, compact: bool = False) -> str:
    try:
        amount = float(value or 0)
    except (TypeError, ValueError):
        amount = 0.0
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    if compact and amount >= 10_000_000:
        return f"{sign}Rs {amount / 10_000_000:.2f} Cr"
    if compact and amount >= 100_000:
        return f"{sign}Rs {amount / 100_000:.2f} L"
    return f"{sign}Rs {amount:,.0f}"


def _pct(value: Any, decimals: int = 1) -> str:
    try:
        return f"{float(value or 0):.{decimals}f}%"
    except (TypeError, ValueError):
        return "0.0%"


def _p(value: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(value if value is not None else "")), style)


def _rich(value: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(value, style)


def _section(story: List[Any], number: str, title: str, styles: Dict[str, ParagraphStyle]) -> None:
    story.append(_rich(f"<font color='#2BC878'>■</font> SECTION {escape(number)}", styles["section_label"]))
    story.append(_p(title, styles["h1"]))
    story.append(Table([[""]], colWidths=[174 * mm], rowHeights=[0.6 * mm], style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), LINE)])))
    story.append(Spacer(1, 4 * mm))


def _table(
    rows: List[List[Any]],
    widths: Iterable[float],
    styles: Dict[str, ParagraphStyle],
    header: bool = True,
    highlight_last: bool = False,
) -> Table:
    cooked: List[List[Any]] = []
    for r_index, row in enumerate(rows):
        cooked.append(
            [
                cell
                if isinstance(cell, Paragraph)
                else _p(cell, styles["table_header"] if header and r_index == 0 else styles["table"])
                for cell in row
            ]
        )
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [WHITE, PALE_BLUE]),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), BLUE))
    if highlight_last and len(rows) > 1:
        commands.append(("BACKGROUND", (0, -1), (-1, -1), PALE_GREEN))
    table = Table(cooked, colWidths=list(widths), repeatRows=1 if header else 0, hAlign="LEFT")
    table.setStyle(TableStyle(commands))
    return table


def _metrics(values: List[tuple[str, str]], styles: Dict[str, ParagraphStyle]) -> Table:
    cells = []
    for label, value in values:
        cell = Table(
            [[_p(label.upper(), styles["metric_label"])], [_p(value, styles["metric"])]],
            colWidths=[174 * mm / len(values) - 14],
        )
        cell.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
        cells.append(cell)
    table = Table([cells], colWidths=[174 * mm / len(cells)] * len(cells), rowHeights=[19 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.45, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _status_label(value: str) -> str:
    return value.replace("_", " ").title()


def _page_decor(canvas, doc) -> None:
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(BLUE)
    canvas.rect(0, height - 6 * mm, width * 0.72, 6 * mm, fill=1, stroke=0)
    canvas.setFillColor(GREEN)
    canvas.rect(width * 0.72, height - 6 * mm, width * 0.28, 6 * mm, fill=1, stroke=0)
    if doc.page > 1:
        canvas.setFont("Helvetica-Bold", 7)
        canvas.setFillColor(BLUE)
        canvas.drawString(16 * mm, height - 12.5 * mm, "RETIREMENT ADVISORY PLAN")
        canvas.setStrokeColor(LINE)
        canvas.line(16 * mm, height - 15 * mm, width - 16 * mm, height - 15 * mm)
    canvas.setStrokeColor(LINE)
    canvas.line(16 * mm, 13 * mm, width - 16 * mm, 13 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 6.8)
    canvas.drawString(16 * mm, 8.5 * mm, "Meerkat Wealth Management  |  Private & Confidential")
    canvas.drawRightString(width - 16 * mm, 8.5 * mm, f"Page {doc.page} of 8")
    canvas.restoreState()


def _cover(story: List[Any], analysis: Dict[str, Any], styles: Dict[str, ParagraphStyle], logo_path: str | None) -> None:
    profile = analysis["profile"]
    scores = analysis["scores"]
    net = analysis["net_worth"]
    story.append(Spacer(1, 28 * mm))
    if logo_path and os.path.exists(logo_path):
        logo = Image(logo_path, width=38 * mm, height=38 * mm)
        logo.hAlign = "CENTER"
        story.append(logo)
        story.append(Spacer(1, 8 * mm))
    story.append(_p("Retirement Advisory Plan", styles["cover_title"]))
    story.append(_p("A practical corpus, income and protection roadmap", styles["cover_subtitle"]))
    story.append(Spacer(1, 16 * mm))
    score_color = GREEN if scores["overall"] >= 75 else AMBER if scores["overall"] >= 40 else RED
    score_box = Table(
        [
            [_rich("RETIREMENT VIABILITY", styles["metric_label"]), _rich(f"<font color='{score_color.hexval()}'><b>{scores['overall']}/100</b></font>", styles["metric"])],
            [_p("STATUS", styles["metric_label"]), _p(_status_label(scores["band"]), styles["table_bold"])],
            [_p("USABLE CORPUS", styles["metric_label"]), _p(_inr(net["usable_corpus"], True), styles["table_bold"])],
        ],
        colWidths=[54 * mm, 80 * mm],
    )
    score_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.8, LINE),
                ("LINEBEFORE", (0, 0), (0, -1), 3, GREEN),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    score_box.hAlign = "CENTER"
    story.append(score_box)
    story.append(Spacer(1, 14 * mm))
    info = _table(
        [
            ["Prepared for", profile["client_name"]],
            ["PAN", profile["pan"]],
            ["Plan date", datetime.now().strftime("%d %B %Y")],
            ["Planning horizon", f"Age {profile['age']} to {profile['planning_age']} ({profile['planning_years']} years)"],
        ],
        [43 * mm, 91 * mm],
        styles,
        header=False,
    )
    info.hAlign = "CENTER"
    story.append(info)
    if net.get("protected_scss"):
        story.append(Spacer(1, 8 * mm))
        story.append(_rich(f"<b>Protected legacy nest:</b> {_inr(net['protected_scss'])} in SCSS is preserved outside deployable corpus.", styles["body"]))


def _snapshot_page(story: List[Any], analysis: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> None:
    net = analysis["net_worth"]
    cash = analysis["cashflow"]
    scores = analysis["scores"]
    swp = analysis["swp"]
    _section(story, "01", "Retirement Snapshot", styles)
    story.append(
        _metrics(
            [
                ("Net worth", _inr(net["net_worth"], True)),
                ("Usable corpus", _inr(net["usable_corpus"], True)),
                ("Monthly expense", _inr(cash["effective_monthly_expense"], True)),
                ("Permanent gap", _inr(cash["permanent_gap"], True)),
            ],
            styles,
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(_p("Viability Score", styles["h2"]))
    score_rows = [["Dimension", "Score", "Planning indicator"]]
    score_rows.extend(
        [
            ["Corpus adequacy", f"{scores['corpus']}/25", _pct(scores["corpus_adequacy_pct"])],
            ["Income security", f"{scores['income']}/25", f"{_pct(scores['income_coverage_pct'])} covered by permanent income"],
            ["Protection", f"{scores['protection']}/25", "Health, spouse and dependent cover"],
            ["Goal readiness", f"{scores['goals']}/25", f"{_pct(scores['goal_funding_pct'])} provisioned from surplus"],
            ["Overall", f"{scores['overall']}/100", _status_label(scores["band"])],
        ]
    )
    story.append(_table(score_rows, [58 * mm, 30 * mm, 86 * mm], styles, highlight_last=True))
    story.append(Spacer(1, 6 * mm))
    story.append(_p("Immediate Reading", styles["h2"]))
    top_flags = analysis.get("flags", [])[:4]
    if top_flags:
        for item in top_flags:
            colour = "#C93B36" if item["priority"] == "critical" else "#D89021"
            story.append(_rich(f"<font color='{colour}'>■</font> <b>{escape(item['message'])}</b> {escape(item['action'])}", styles["body"]))
    else:
        story.append(_rich("<font color='#2BC878'>■</font> The plan has no critical flags. Maintain annual review discipline.", styles["body"]))
    story.append(Spacer(1, 5 * mm))
    story.append(_rich(f"At the current corpus, the first SWP level that covers the permanent monthly gap is <b>{_status_label(swp['status'])}</b>.", styles["body"]))


def _allocation_page(story: List[Any], analysis: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> None:
    net = analysis["net_worth"]
    alloc = analysis["allocation"]
    _section(story, "02", "Corpus Deployment Architecture", styles)
    story.append(_rich(f"The model starts with <b>{_inr(net['usable_corpus'])}</b> of deployable assets. Primary residence, investment property, physical gold, insurance surrender value and SCSS remain outside this amount.", styles["body"]))
    story.append(Spacer(1, 4 * mm))
    rows = [
        ["Priority", "Bucket", "Amount", "% of usable", "Purpose"],
        ["1", "Liquidity buffer", _inr(alloc["liquidity_buffer"]), _pct(alloc["liquidity_buffer"] / net["usable_corpus"] * 100 if net["usable_corpus"] else 0), "Six months of effective expenses"],
        ["2", "Insurance supply", _inr(alloc["insurance_supply"]), _pct(alloc["insurance_supply"] / net["usable_corpus"] * 100 if net["usable_corpus"] else 0), "Future health and term premiums"],
        ["3", "Old-age care", _inr(alloc["old_age_care"]), _pct(alloc["old_age_care_pct"] * 100), "Late-life medical and care reserve"],
        ["4", "Pension corpus", _inr(alloc["pension_corpus_available"]), _pct(alloc["pension_corpus_available"] / net["usable_corpus"] * 100 if net["usable_corpus"] else 0), "Monthly SWP and inflation step-up"],
        ["5", "Pension surplus", _inr(alloc["pension_surplus"]), _pct(alloc["pension_surplus"] / net["usable_corpus"] * 100 if net["usable_corpus"] else 0), "Available for goals only after retirement income"],
    ]
    story.append(_table(rows, [14 * mm, 37 * mm, 32 * mm, 26 * mm, 65 * mm], styles))
    story.append(Spacer(1, 7 * mm))
    story.append(_p("Net Worth Classification", styles["h2"]))
    labels = {
        "primary_residence": "Primary residence",
        "investment_property": "Investment property",
        "scss": "SCSS protected corpus",
        "fixed_deposits": "Fixed deposits",
        "gold_silver": "Gold and silver",
        "insurance_surrender": "Insurance surrender value",
        "equity_investments": "Equity investments",
        "debt_investments": "Debt investments",
        "liquid_savings": "Liquid savings",
        "pension_lump_sum": "Pension lump sum",
        "other_investments": "Other investments",
    }
    deployable = {"fixed_deposits", "equity_investments", "debt_investments", "liquid_savings", "pension_lump_sum", "other_investments"}
    protected = {"scss"}
    asset_rows = [["Asset", "Value", "Planning class"]]
    for key, value in net["assets"].items():
        if value <= 0:
            continue
        asset_rows.append([labels.get(key, key.replace("_", " ").title()), _inr(value), "Deployable" if key in deployable else "Protected" if key in protected else "Non-deployable"])
    asset_rows.append(["Total net worth", _inr(net["net_worth"]), "After recorded liabilities"])
    story.append(_table(asset_rows, [70 * mm, 45 * mm, 59 * mm], styles, highlight_last=True))


def _income_page(story: List[Any], analysis: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> None:
    cash = analysis["cashflow"]
    swp = analysis["swp"]
    alloc = analysis["allocation"]
    _section(story, "03", "Monthly Income and SWP", styles)
    story.append(
        _metrics(
            [
                ("Permanent income", _inr(cash["permanent_income"], True)),
                ("Current income", _inr(cash["total_current_income"], True)),
                ("Expense + EMI", _inr(cash["effective_monthly_expense"] + cash["monthly_emi"], True)),
                ("Permanent gap", _inr(cash["permanent_gap"], True)),
            ],
            styles,
        )
    )
    story.append(Spacer(1, 6 * mm))
    income_rows = [["Income source", "Monthly amount", "Nature"]]
    for key, value in cash["pension_sources"].items():
        if value:
            income_rows.append([key.replace("_", " ").title(), _inr(value), "Permanent pension"])
    for label, value in (("Rental income", cash["rental_income"]), ("SCSS interest", cash["scss_interest"]), ("FD interest", cash["fd_interest"])):
        if value:
            income_rows.append([label, _inr(value), "Permanent income"])
    for item in cash["other_income_sources"]:
        amount = float(item.get("monthly_amount") or 0)
        if amount:
            years = int(float(item.get("years_remaining") or 0))
            income_rows.append([item.get("name") or "Other income", _inr(amount), f"{years} years remaining" if years else "Permanent income"])
    income_rows.append(["Total current income", _inr(cash["total_current_income"]), "Before expenses"])
    story.append(_table(income_rows, [72 * mm, 43 * mm, 59 * mm], styles, highlight_last=True))
    story.append(Spacer(1, 6 * mm))
    story.append(_p("SWP Adequacy Spectrum", styles["h2"]))
    spectrum_rows = [["Rate", "Monthly SWP", "Covers gap?", "Interpretation"]]
    for row in swp["spectrum"]:
        note = _status_label(row["status"])
        if row.get("depletion_age"):
            note += f"; modelled depletion age {row['depletion_age']}"
        spectrum_rows.append([_pct(row["rate"] * 100), _inr(row["monthly_income"]), "Yes" if row["covers_gap"] else "No", note])
    story.append(_table(spectrum_rows, [24 * mm, 43 * mm, 30 * mm, 77 * mm], styles))
    story.append(Spacer(1, 5 * mm))
    story.append(_rich(f"<b>Recommended reading:</b> {_status_label(swp['status'])}. The 3.5% ideal corpus is {_inr(swp['ideal_corpus_35'])}; the gap to that ideal is {_inr(swp['ideal_gap'])}. Corpus adequacy at the 5% planning benchmark is {_pct(alloc['corpus_adequacy_pct'])}.", styles["body"]))


def _expense_protection_page(story: List[Any], analysis: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> None:
    cash = analysis["cashflow"]
    protection = analysis["protection"]
    _section(story, "04", "Expenses, Liquidity and Protection", styles)
    expense_rows = [["Expense", "Annual amount", "Monthly equivalent"]]
    expense_rows.append(["Core household expenses", _inr(cash["monthly_core_expense"] * 12), _inr(cash["monthly_core_expense"])])
    for item in cash["annual_items"]:
        amount = float(item.get("amount") or 0)
        if amount:
            expense_rows.append([item.get("name") or "Annual expense", _inr(amount), _inr(amount / 12)])
    if cash["dependent_cost"]:
        expense_rows.append(["Dependent support", _inr(cash["dependent_cost"] * 12), _inr(cash["dependent_cost"])])
    if cash["monthly_emi"]:
        expense_rows.append(["Loan EMI", _inr(cash["monthly_emi"] * 12), _inr(cash["monthly_emi"])])
    expense_rows.append(["Effective monthly expense", _inr(cash["effective_monthly_expense"] * 12), _inr(cash["effective_monthly_expense"])])
    story.append(_table(expense_rows, [78 * mm, 48 * mm, 48 * mm], styles, highlight_last=True))
    story.append(Spacer(1, 7 * mm))
    story.append(_p("Protection Check", styles["h2"]))
    protection_rows = [
        ["Item", "Current", "Planning benchmark", "Status"],
        ["Health insurance", _inr(protection["health_cover"]), _inr(protection["recommended_health_cover"]), "Adequate" if protection["health_cover_gap"] <= 0 else f"Gap {_inr(protection['health_cover_gap'])}"],
        ["Spouse covered", "Yes" if protection["spouse_covered"] else "No", "Yes, where applicable", "Covered" if protection["spouse_covered"] else "Immediate action"],
        ["Term insurance", _inr(protection["term_cover"]) if protection["has_term_insurance"] else "Not recorded", "Needs-based in retirement", "Review dependents and liabilities"],
        ["Premium reserve", _inr(protection["insurance_supply_corpus"]), "Premiums through planning age", "Dedicated corpus"],
    ]
    story.append(_table(protection_rows, [44 * mm, 42 * mm, 48 * mm, 40 * mm], styles))
    story.append(Spacer(1, 7 * mm))
    story.append(_rich("<b>Expense definition:</b> Core monthly expenses should include groceries, utilities, society maintenance, domestic help, transport, dining, subscriptions, mobile, internet and other regular living costs. Annual or periodic expenses are converted to a monthly equivalent; EMIs remain visible separately.", styles["body"]))


def _goals_page(story: List[Any], analysis: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> None:
    goals = analysis["goals"]
    profile = analysis["profile"]
    _section(story, "05", "Goal Funding and Estate Readiness", styles)
    if goals:
        goal_rows = [["Goal", "Due", "Future cost", "Corpus today", "Funded", "Strategy"]]
        for goal in goals:
            goal_rows.append(
                [
                    goal["name"],
                    f"{goal['years_from_now']} yrs",
                    _inr(goal["future_cost"], True),
                    _inr(goal["corpus_needed_today"], True),
                    _pct(goal["funding_pct"]),
                    f"{goal['strategy']} ({goal['growth_allocation_pct']}% growth assets)",
                ]
            )
        story.append(_table(goal_rows, [35 * mm, 17 * mm, 29 * mm, 29 * mm, 20 * mm, 44 * mm], styles))
        story.append(Spacer(1, 5 * mm))
        story.append(_rich("Goal amounts are made explicit, inflated to the due date where selected, then discounted using a horizon-based return assumption. The growth allocation reduces as the goal approaches so capital stability gradually takes priority over beating inflation.", styles["body"]))
    else:
        story.append(_rich("<font color='#2BC878'>■</font> No separate retirement goals were entered. All surplus remains available for income resilience and legacy planning.", styles["body"]))
    story.append(Spacer(1, 8 * mm))
    story.append(_p("Pension-First Funding Rule", styles["h2"]))
    rules = [
        "Protect the liquidity, insurance and old-age care reserves.",
        "Secure the pension corpus required for the monthly gap.",
        "Fund education, wedding, vehicle, property and other goals only from the resulting surplus.",
        "As a goal approaches, reduce volatile growth exposure and move toward short-duration debt or cash equivalents.",
    ]
    for index, rule in enumerate(rules, 1):
        story.append(_rich(f"<b>{index}.</b> {escape(rule)}", styles["body"]))
    story.append(Spacer(1, 7 * mm))
    will_status = "In place" if profile["will_in_place"] else "Not in place - immediate action"
    estate_rows = [
        ["Estate item", "Status / action"],
        ["Will", will_status],
        ["Nominations", "Verify bank, demat, mutual fund, pension and insurance nominations"],
        ["Asset register", "Record institutions, folio/account details and adviser contacts"],
        ["Digital access", "Document recovery instructions without sharing live passwords"],
    ]
    story.append(_table(estate_rows, [48 * mm, 126 * mm], styles))


def _action_page(story: List[Any], analysis: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> None:
    flags = analysis.get("flags") or []
    _section(story, "06", "Implementation Roadmap", styles)
    rows: List[List[Any]] = []
    if flags:
        for index, item in enumerate(flags[:8], 1):
            label = "Immediate" if item["priority"] == "critical" else "Priority"
            rows.append(
                [
                    _rich(f"<font color='#FFFFFF'><b>{index}</b></font>", styles["action"]),
                    _rich(f"<b>{label}:</b> {escape(item['action'])}", styles["action"]),
                ]
            )
    else:
        rows.append([_rich("<font color='#FFFFFF'><b>1</b></font>", styles["action"]), _p("Maintain the current structure and complete an annual retirement review.", styles["action"])])
    table = Table(rows, colWidths=[10 * mm, 164 * mm])
    commands = [
        ("BACKGROUND", (0, 0), (0, -1), BLUE),
        ("ROWBACKGROUNDS", (1, 0), (1, -1), [PALE_BLUE, WHITE]),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]
    table.setStyle(TableStyle(commands))
    story.append(table)
    story.append(Spacer(1, 8 * mm))
    story.append(_p("Review Cadence", styles["h2"]))
    review_rows = [
        ["When", "Review"],
        ["Within 30 days", "Complete immediate protection, will and liquidity actions"],
        ["Every quarter", "Check withdrawals, cash buffer and major goal changes"],
        ["Every year", "Re-run viability, rebalance buckets and refresh insurance adequacy"],
        ["On major change", "Regenerate after retirement, bereavement, property sale, inheritance or medical event"],
    ]
    story.append(_table(review_rows, [42 * mm, 132 * mm], styles))


def _assumptions_page(story: List[Any], analysis: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> None:
    assumptions = analysis["assumptions"]
    _section(story, "07", "Assumptions and Important Notes", styles)
    rows = [
        ["Assumption", "MVP value", "Use in this report"],
        ["Planning age", str(assumptions["planning_age"]), "Defines premium reserve and review horizon"],
        ["Inflation", _pct(assumptions["inflation_rate"] * 100), "Inflates goal costs selected as inflation-linked"],
        ["SCSS rate", _pct(assumptions["scss_rate"] * 100), "Estimates permanent monthly SCSS income"],
        ["FD rate", _pct(assumptions["fd_rate"] * 100), "Estimates permanent monthly FD income"],
        ["Planning withdrawal rate", _pct(assumptions["sustainable_withdrawal_rate"] * 100), "Sizes the base pension corpus requirement"],
        ["SWP step-up", _pct(assumptions["swp_step_up"] * 100), "Models rising income over time"],
        ["SWP portfolio return", _pct(assumptions["swp_growth_rate"] * 100), "Illustrative depletion modelling only"],
        ["Liquidity reserve", f"{assumptions['liquidity_months']} months", "Protects monthly living expenses"],
    ]
    story.append(_table(rows, [47 * mm, 33 * mm, 94 * mm], styles))
    story.append(Spacer(1, 8 * mm))
    story.append(_p("Important Notes", styles["h2"]))
    notes = [
        "This is a deterministic planning report based only on the information entered. It is not an account statement or a guarantee of returns.",
        "All return, inflation, interest and withdrawal figures are assumptions. Actual market returns, taxation, expenses and longevity will vary.",
        "Instrument references are categories, not recommendations of a particular scheme, security, insurer or property.",
        "Insurance availability, premiums and claims are subject to underwriting and policy terms.",
        "Estate planning actions should be completed with a qualified lawyer and tax implications with an appropriate tax professional.",
        "Review the plan at least annually and whenever income, health, family responsibilities, assets or goals materially change.",
    ]
    for note in notes:
        story.append(_rich(f"<font color='#2BC878'>■</font> {escape(note)}", styles["body"]))
    story.append(Spacer(1, 8 * mm))
    story.append(_rich("<b>Private and confidential.</b> Prepared solely for the named client and adviser review.", styles["body"]))


def generate_retirement_pdf(analysis: Dict[str, Any], output_path: str, logo_path: str | None = None) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    styles = _styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=21 * mm,
        bottomMargin=17 * mm,
        title="Retirement Advisory Plan",
        author="Meerkat Wealth Management",
        subject="Retirement corpus deployment and SWP analysis",
    )
    story: List[Any] = []
    _cover(story, analysis, styles, logo_path)
    story.append(PageBreak())
    _snapshot_page(story, analysis, styles)
    story.append(PageBreak())
    _allocation_page(story, analysis, styles)
    story.append(PageBreak())
    _income_page(story, analysis, styles)
    story.append(PageBreak())
    _expense_protection_page(story, analysis, styles)
    story.append(PageBreak())
    _goals_page(story, analysis, styles)
    story.append(PageBreak())
    _action_page(story, analysis, styles)
    story.append(PageBreak())
    _assumptions_page(story, analysis, styles)
    doc.build(story, onFirstPage=_page_decor, onLaterPages=_page_decor)
