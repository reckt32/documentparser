"""Settled retirement advisory PDF generated from deterministic solver output."""

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
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BLUE = colors.HexColor("#0752B8")
NAVY = colors.HexColor("#113466")
GREEN = colors.HexColor("#20B86A")
RED = colors.HexColor("#C93B36")
AMBER = colors.HexColor("#D89021")
INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#6B7280")
LINE = colors.HexColor("#D7E0EC")
PALE_BLUE = colors.HexColor("#F2F6FC")
PALE_GREEN = colors.HexColor("#EAF8F0")
PALE_RED = colors.HexColor("#FCEFED")
WHITE = colors.white


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=8.8, leading=12.5, textColor=INK, spaceAfter=4),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontName="Helvetica", fontSize=7.2, leading=9.4, textColor=MUTED),
        "section_label": ParagraphStyle("SectionLabel", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=GREEN, spaceAfter=2),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=BLUE, spaceAfter=8),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=NAVY, spaceBefore=7, spaceAfter=5),
        "metric": ParagraphStyle("Metric", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=14, leading=16, textColor=NAVY),
        "metric_label": ParagraphStyle("MetricLabel", parent=base["BodyText"], fontName="Helvetica", fontSize=7.1, leading=8.5, textColor=MUTED),
        "cover_title": ParagraphStyle("CoverTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=27, leading=33, alignment=TA_CENTER, textColor=NAVY, spaceAfter=8),
        "cover_subtitle": ParagraphStyle("CoverSubtitle", parent=base["BodyText"], fontName="Helvetica", fontSize=11.5, leading=15, alignment=TA_CENTER, textColor=MUTED),
        "table_header": ParagraphStyle("TableHeader", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.2, leading=8.8, textColor=WHITE),
        "table": ParagraphStyle("Table", parent=base["BodyText"], fontName="Helvetica", fontSize=7.2, leading=9.3, textColor=INK),
        "table_bold": ParagraphStyle("TableBold", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.2, leading=9.3, textColor=INK),
        "action": ParagraphStyle("Action", parent=base["BodyText"], fontName="Helvetica", fontSize=8.2, leading=11.2, textColor=INK),
        "center": ParagraphStyle("Center", parent=base["BodyText"], alignment=TA_CENTER, fontName="Helvetica", fontSize=8, leading=11, textColor=INK),
        "right": ParagraphStyle("Right", parent=base["BodyText"], alignment=TA_RIGHT, fontName="Helvetica", fontSize=7.2, leading=9.3, textColor=INK),
    }


def _p(value: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(value if value is not None else "")), style)


def _rich(value: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(value, style)


def _inr(value: Any, compact: bool = False) -> str:
    amount = float(value or 0)
    if compact and abs(amount) >= 10_000_000:
        return f"Rs {amount / 10_000_000:.2f} Cr"
    if compact and abs(amount) >= 100_000:
        return f"Rs {amount / 100_000:.2f} L"
    return f"Rs {amount:,.0f}"


def _pct(value: Any, decimals: int = 1) -> str:
    return f"{float(value or 0) * 100:.{decimals}f}%"


def _status_label(value: Any) -> str:
    return str(value or "").replace("_", " ").strip().title()


def _table(
    rows: Iterable[Iterable[Any]],
    widths: List[float],
    styles: Dict[str, ParagraphStyle],
    *,
    highlight_last: bool = False,
    header: bool = True,
) -> Table:
    rendered = []
    for row_index, row in enumerate(rows):
        rendered.append(
            [
                cell
                if isinstance(cell, Paragraph)
                else _p(
                    cell,
                    styles["table_header"]
                    if header and row_index == 0
                    else styles["table"],
                )
                for cell in row
            ]
        )
    table = Table(
        rendered,
        colWidths=widths,
        repeatRows=1 if header else 0,
        hAlign="LEFT",
    )
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        (
            "ROWBACKGROUNDS",
            (0, 1 if header else 0),
            (-1, -1),
            [WHITE, PALE_BLUE],
        ),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ]
        )
    if highlight_last and len(rendered) > 1:
        commands.append(("BACKGROUND", (0, -1), (-1, -1), PALE_GREEN))
    table.setStyle(TableStyle(commands))
    return table


def _section(story: List[Any], number: str, title: str, styles: Dict[str, ParagraphStyle]) -> None:
    story.append(_p(f"SECTION {number}", styles["section_label"]))
    story.append(_p(title, styles["h1"]))


def _metric_strip(items: List[tuple[str, str]], styles: Dict[str, ParagraphStyle]) -> Table:
    cells = []
    for label, value in items:
        cells.append([_p(label.upper(), styles["metric_label"]), _p(value, styles["metric"])])
    table = Table([cells], colWidths=[174 * mm / len(cells)] * len(cells))
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def _cover(story: List[Any], analysis: Dict[str, Any], styles: Dict[str, ParagraphStyle], logo_path: str | None) -> None:
    profile = analysis["profile"]
    solver = analysis["solver"]
    corpus = analysis["net_worth"]
    story.append(Spacer(1, 34 * mm))
    if logo_path and os.path.exists(logo_path):
        logo = Image(logo_path, width=20 * mm, height=20 * mm)
        logo.hAlign = "CENTER"
        story.append(logo)
        story.append(Spacer(1, 8 * mm))
    story.append(_p("Retirement Advisory Plan", styles["cover_title"]))
    story.append(_p("A settled income, goal and corpus-deployment roadmap", styles["cover_subtitle"]))
    story.append(Spacer(1, 18 * mm))
    feasible = solver["selected_feasible"]
    status_color = GREEN if feasible else RED
    status_box = Table(
        [
            [_p("PLAN STATUS", styles["metric_label"]), _rich(f"<font color='{status_color.hexval()}'><b>{'FEASIBLE' if feasible else 'REQUIRES ADJUSTMENT'}</b></font>", styles["metric"])],
            [_p("SELECTED WITHDRAWAL RATE", styles["metric_label"]), _p(_pct(solver["selected_rate"]), styles["metric"])],
            [_p("AVAILABLE CORPUS", styles["metric_label"]), _p(_inr(corpus["available_corpus"], True), styles["metric"])],
        ],
        colWidths=[72 * mm, 72 * mm],
    )
    status_box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE), ("BOX", (0, 0), (-1, -1), 0.5, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    status_box.hAlign = "CENTER"
    story.append(status_box)
    story.append(Spacer(1, 17 * mm))
    details = [
        ["Prepared for", profile["client_name"]],
        ["PAN", profile["pan"]],
        ["Plan date", datetime.now().strftime("%d %B %Y")],
        ["Planning horizon", f"Age {profile['age']} to {profile['planning_age']}"],
        ["Adviser decision", "Accepted for report generation" if analysis.get("decision", {}).get("adviser_accepted") else "Draft scenario"],
    ]
    story.append(_table(details, [47 * mm, 93 * mm], styles, header=False))
    story.append(Spacer(1, 9 * mm))
    story.append(_p("Private and confidential. Category returns are planning assumptions, not guaranteed outcomes or scheme recommendations.", styles["center"]))
    story.append(PageBreak())


def _current_status(story: List[Any], analysis: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> None:
    cash = analysis["cashflow"]
    solver = analysis["solver"]
    protection = analysis["protection"]
    reserves = analysis["reserves"]
    _section(story, "01", "Current Status", styles)
    story.append(
        _metric_strip(
            [
                ("Effective monthly expense", _inr(cash["effective_monthly_expense"], True)),
                ("Current gap", _inr(cash["current_gap"], True)),
                ("Design gap", _inr(cash["design_gap"], True)),
                ("Gap peaks", f"Year {cash['design_gap_year']}"),
            ],
            styles,
        )
    )
    story.append(_p("Income Timeline", styles["h2"]))
    income_rows = [["Source", "Net monthly", "Nature", "Ends / tax basis"]]
    for source in cash["income_sources"]:
        end = "Lifelong" if source["nature"] == "lifelong" else f"{source['years_remaining']} yrs"
        if source["indexed"]:
            end += f"; indexed {_pct(source['index_rate'])}"
        if source.get("assumed_tax_rate"):
            end += f"; {_pct(source['assumed_tax_rate'], 0)} assumed tax"
        income_rows.append([source["name"], _inr(source["monthly_amount"]), _status_label(source["nature"]), end])
    if len(income_rows) == 1:
        income_rows.append(["No recorded income", _inr(0), "-", "-"])
    story.append(_table(income_rows, [55 * mm, 30 * mm, 35 * mm, 54 * mm], styles))
    story.append(_p("Expense and Protection Position", styles["h2"]))
    status_rows = [
        ["Item", "Current position", "Planning treatment"],
        ["Core monthly expenses", _inr(cash["monthly_core_expense"]), "Inflated through planning age"],
        ["Annual / periodic expenses", _inr(cash["annual_item_expenses"]), "Converted to monthly equivalent"],
        ["Term and motor premiums", _inr(cash["annual_insurance_expenses"]), "Included as recurring annual expenses"],
        ["Dependent support", _inr(cash["dependent_cost"]), "Included in effective expense"],
        ["Health insurance cover", _inr(protection["health_cover"]), f"Annual premium {_inr(protection['annual_health_premium'])}"],
        ["Term insurance cover", _inr(protection["term_cover"]), f"Annual premium {_inr(protection['annual_term_premium'])}"],
        ["Health premium reserve", _inr(reserves["premium_reserve"]), "10 times annual health premium only"],
        ["Selected SWP", _inr(solver["monthly_swp"]), f"{_pct(solver['selected_rate'])} from income bucket"],
    ]
    story.append(_table(status_rows, [55 * mm, 40 * mm, 79 * mm], styles))
    if cash["liabilities"]:
        story.append(_p("Liabilities", styles["h2"]))
        liability_rows = [["Liability", "Outstanding", "EMI", "Treatment"]]
        for item in cash["liabilities"]:
            liability_rows.append([item["name"], _inr(item["outstanding"]), _inr(item["monthly_emi"]), _status_label(item["treatment"])])
        story.append(_table(liability_rows, [62 * mm, 38 * mm, 34 * mm, 40 * mm], styles))
    story.append(PageBreak())


def _goals(story: List[Any], analysis: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> None:
    goals = analysis["goals"]
    reserves = analysis["reserves"]
    _section(story, "02", "Goals", styles)
    goal_rows = [["Goal", "Due", "Requirement", "Corpus now", "Risk / return", "Rate impact"]]
    for goal in goals:
        goal_rows.append(
            [
                goal["name"],
                f"{goal['years_from_now']} yrs",
                _inr(goal["nominal_requirement"], True),
                _inr(goal["corpus_today"], True),
                f"{goal['risk_category']} / {_pct(goal['expected_return'], 0)}",
                f"{goal['rate_impact'] * 100:.2f} pp",
            ]
        )
    if len(goal_rows) == 1:
        goal_rows.append(["No additional goals", "-", "-", "-", "-", "-"])
    story.append(_table(goal_rows, [43 * mm, 18 * mm, 28 * mm, 28 * mm, 38 * mm, 19 * mm], styles))
    story.append(_rich("Goal requirements are treated as fixed nominal amounts and are <b>not inflated</b>. Each amount is discounted by the expected return of its goal category to determine the corpus required today.", styles["body"]))
    story.append(_p("Protected Reserves", styles["h2"]))
    reserve_rows = [
        ["Bucket", "Rule", "Corpus", "Category"],
        ["Emergency fund", f"{reserves['emergency_months']} months of effective expense", _inr(reserves["emergency_fund"]), "No Risk"],
        ["Health premium reserve", f"{reserves['premium_multiple']} x annual health premium", _inr(reserves["premium_reserve"]), "No Risk"],
        ["Market-opportunity bucket", f"{reserves['opportunity_pct'] * 100:.1f}% of available corpus", _inr(reserves["opportunity_bucket"]), "Low Risk"],
    ]
    story.append(_table(reserve_rows, [52 * mm, 64 * mm, 34 * mm, 24 * mm], styles))
    story.append(_p("Adviser Decision Rule", styles["h2"]))
    story.append(_rich("The engine does not remove or resize a goal. If the required rate exceeds 7%, the adviser changes goal amount, goal date, inclusion or selected withdrawal rate with the client and reruns the plan.", styles["body"]))
    story.append(PageBreak())


def _investments(story: List[Any], analysis: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> None:
    net = analysis["net_worth"]
    allocation = analysis["allocation"]
    _section(story, "03", "Investments", styles)
    story.append(
        _metric_strip(
            [
                ("Total assets", _inr(net["total_assets"], True)),
                ("Net worth", _inr(net["net_worth"], True)),
                ("Gross deployable", _inr(net["gross_deployable"], True)),
                ("Available after settlement", _inr(net["available_corpus"], True)),
            ],
            styles,
        )
    )
    story.append(_p("Existing Holding Classification", styles["h2"]))
    holding_rows = [["Category", "Current", "Deployable", "Retained", "Goal", "Tax basis"]]
    for holding in net["holdings"]:
        holding_rows.append(
            [
                _status_label(holding["instrument_type"]),
                _inr(holding["current_value"], True),
                _inr(holding["deployable_amount"], True),
                _inr(holding["retained_amount"], True),
                holding.get("goal_id") or "Unassigned",
                f"{_pct(holding['assumed_tax_rate'], 0)} on interest" if holding.get("assumed_tax_rate") else "No engine tax rule",
            ]
        )
    story.append(_table(holding_rows, [40 * mm, 27 * mm, 27 * mm, 27 * mm, 25 * mm, 28 * mm], styles))
    story.append(_rich("Existing holdings are shown only by category. The adviser selects schemes and reviews whether each holding's risk profile remains appropriate; the engine makes no scheme-level judgement.", styles["body"]))
    story.append(_p("Settled Allocation by Risk Category", styles["h2"]))
    risk_rows = [["Category", "Fund type", "Amount", "Share", "Expected return"]]
    for item in allocation["by_risk"]:
        risk_rows.append([item["category"], item["fund_type"], _inr(item["amount"], True), f"{item['percentage']:.1f}%", _pct(item["expected_return"], 0)])
    risk_rows.append(["Blended", "Category-level allocation", _inr(allocation["allocated_total"], True), "100%", _pct(allocation["blended_return"])])
    story.append(_table(risk_rows, [36 * mm, 64 * mm, 32 * mm, 20 * mm, 22 * mm], styles, highlight_last=True))
    story.append(PageBreak())


def _plan(story: List[Any], analysis: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> None:
    solver = analysis["solver"]
    cash = analysis["cashflow"]
    allocation = analysis["allocation"]
    _section(story, "04", "Plan", styles)
    status = "Feasible" if solver["selected_feasible"] else "Requires adjustment"
    story.append(
        _metric_strip(
            [
                ("Plan status", status),
                ("Selected rate", _pct(solver["selected_rate"])),
                ("Income bucket", _inr(solver["income_bucket"], True)),
                ("Legacy residual", _inr(solver["legacy_residual"], True)),
            ],
            styles,
        )
    )
    story.append(_p("Corpus Reconciliation", styles["h2"]))
    waterfall_rows = [["Step", "Treatment", "Amount"]]
    for item in allocation["reconciliation"]:
        treatment = "+" if item["operation"] == "add" else "-" if item["operation"] == "subtract" else "="
        waterfall_rows.append([item["label"], treatment, _inr(item["amount"])])
    story.append(_table(waterfall_rows, [104 * mm, 20 * mm, 50 * mm], styles, highlight_last=True))
    story.append(_p("Withdrawal Solve", styles["h2"]))
    solve_rows = [
        ["Parameter", "Settled value", "Meaning"],
        ["Current monthly gap", _inr(cash["current_gap"]), "Gap on plan date"],
        ["Design monthly gap", _inr(cash["design_gap"]), f"Maximum gap in year {cash['design_gap_year']}"],
        ["Natural required rate", "Not defined" if solver["natural_rate"] is None else _pct(solver["natural_rate"]), "Exact rate after reserves and goals"],
        ["Permitted range", "3.5% to 7.0%", "7% is a hard cap"],
        ["Selected monthly SWP", _inr(solver["monthly_swp"]), "Built on the design gap"],
        ["Shortfall at 7%", _inr(solver["shortfall_at_cap"]), "Must be resolved before implementation"],
    ]
    story.append(_table(solve_rows, [56 * mm, 40 * mm, 78 * mm], styles))
    story.append(_p("Cash-flow Milestones", styles["h2"]))
    timeline = cash["timeline"]
    milestone_indexes = sorted(set([0, cash["design_gap_year"], min(len(timeline) - 1, 3), min(len(timeline) - 1, 5), len(timeline) - 1]))
    timeline_rows = [["Year", "Age", "Expense", "Income", "EMI", "Gap"]]
    for index in milestone_indexes:
        row = timeline[index]
        timeline_rows.append([row["year"], row["age"], _inr(row["monthly_expense"], True), _inr(row["monthly_income"], True), _inr(row["monthly_emi"], True), _inr(row["monthly_gap"], True)])
    story.append(_table(timeline_rows, [20 * mm, 20 * mm, 34 * mm, 34 * mm, 32 * mm, 34 * mm], styles))
    story.append(PageBreak())


def _actionables(story: List[Any], analysis: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> None:
    flags = analysis["flags"]
    actions = analysis["actionables"]
    _section(story, "05", "Actionables", styles)
    if flags:
        flag_rows = [["Priority", "Finding", "Adviser action"]]
        for item in flags:
            flag_rows.append([item["priority"].upper(), item["message"], item["action"]])
        story.append(_table(flag_rows, [25 * mm, 68 * mm, 81 * mm], styles))
    else:
        story.append(_rich("<font color='#20B86A'><b>No structural flags remain in the settled plan.</b></font>", styles["body"]))
    story.append(_p("Implementation Roadmap", styles["h2"]))
    roadmap = []
    for index, item in enumerate(actions, 1):
        roadmap.append([str(index), item["description"], _inr(item["amount"], True) if item["amount"] else "Review"])
    if not roadmap:
        roadmap = [["1", "Complete the agreed category-level deployment and schedule the annual review.", "Ongoing"]]
    roadmap_table = Table(
        [[_p("#", styles["table_header"]), _p("Action", styles["table_header"]), _p("Value", styles["table_header"])]]
        + [[_p(number, styles["table_bold"]), _p(description, styles["action"]), _p(value, styles["table"])] for number, description, value in roadmap],
        colWidths=[13 * mm, 125 * mm, 36 * mm],
    )
    roadmap_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), BLUE), ("GRID", (0, 0), (-1, -1), 0.35, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("BACKGROUND", (0, 1), (0, -1), NAVY), ("TEXTCOLOR", (0, 1), (0, -1), WHITE), ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(roadmap_table)
    story.append(_p("Review Cadence", styles["h2"]))
    cadence = [
        ["When", "Review"],
        ["Before deployment", "Confirm liabilities, reserve amounts, selected rate and every active goal"],
        ["Within 30 days", "Complete deployment, protection and estate actions"],
        ["Every quarter", "Check withdrawals, reserve balances and material goal changes"],
        ["Every year", "Re-run the full timeline and produce a newly settled plan"],
    ]
    story.append(_table(cadence, [45 * mm, 129 * mm], styles))
    story.append(PageBreak())


def _assumptions(story: List[Any], analysis: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> None:
    assumptions = analysis["assumptions"]
    _section(story, "06", "Assumptions and Important Notes", styles)
    rows = [
        ["Assumption", "Value", "Use"],
        ["Assumption version", assumptions["version"], "Reproduces the calculation contract"],
        ["Planning age", assumptions["planning_age"], "Timeline endpoint unless adviser overrides"],
        ["Expense inflation", _pct(assumptions["expense_inflation"]), "Builds the year-by-year design gap"],
        ["Withdrawal floor", _pct(assumptions["withdrawal_floor"]), "Minimum selected planning rate"],
        ["Withdrawal cap", _pct(assumptions["withdrawal_cap"]), "Hard ceiling"],
        ["Emergency reserve", f"{assumptions['emergency_months']} months", "Liquid No Risk reserve"],
        ["Premium reserve", f"{assumptions['premium_reserve_multiple']} times", "Annual health premium only"],
        ["SCSS / FD tax", _pct(assumptions["interest_tax_rate"], 0), "Gross interest multiplied by 78%"],
        ["Opportunity selection", _pct(assumptions["opportunity_selected_pct"]), "Adviser-adjustable from zero to 10%"],
        ["Goal inflation", "None", "Goal inputs are fixed nominal requirements"],
    ]
    story.append(_table(rows, [55 * mm, 43 * mm, 76 * mm], styles))
    story.append(_p("Risk Category Assumptions", styles["h2"]))
    risk_rows = [["Category", "Fund type", "Expected return"]]
    for item in assumptions["risk_categories"].values():
        risk_rows.append([item["label"], item["fund_type"], _pct(item["return"], 0)])
    story.append(_table(risk_rows, [44 * mm, 96 * mm, 34 * mm], styles))
    story.append(_p("Important Notes", styles["h2"]))
    notes = [
        "This is a deterministic planning output based only on the inputs and selected adviser decisions. It is not an account statement, tax opinion or guarantee.",
        "Risk labels are plain-English relative indicators within this portfolio and are not SEBI riskometer categories.",
        "No mutual fund scheme is selected or recommended. Exact schemes remain exclusively within the adviser's purview.",
        "The only tax calculation used is the agreed 22% assumption on SCSS and fixed-deposit interest. No personal tax liability is calculated.",
        "Insurance placement remains subject to underwriting, waiting periods, exclusions and full disclosure.",
        "Review this plan at least annually and whenever income, expenses, liabilities, health, family responsibilities or goals materially change.",
    ]
    for note in notes:
        story.append(_rich(f"<font color='#20B86A'>■</font> {escape(note)}", styles["body"]))


def generate_retirement_pdf(analysis: Dict[str, Any], output_path: str, logo_path: str | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    styles = _styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=16 * mm,
        title="Retirement Advisory Plan",
        author="Meerkat",
    )
    story: List[Any] = []
    _cover(story, analysis, styles, logo_path)
    _current_status(story, analysis, styles)
    _goals(story, analysis, styles)
    _investments(story, analysis, styles)
    _plan(story, analysis, styles)
    _actionables(story, analysis, styles)
    _assumptions(story, analysis, styles)

    def decorate(canvas, document):
        canvas.saveState()
        width, height = A4
        canvas.setFillColor(BLUE)
        canvas.rect(0, height - 4 * mm, width * 0.72, 4 * mm, fill=1, stroke=0)
        canvas.setFillColor(GREEN)
        canvas.rect(width * 0.72, height - 4 * mm, width * 0.28, 4 * mm, fill=1, stroke=0)
        canvas.setStrokeColor(LINE)
        canvas.line(18 * mm, 12 * mm, width - 18 * mm, 12 * mm)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 6.5)
        canvas.drawString(18 * mm, 8 * mm, "Meerkat Wealth Management | Private & Confidential")
        canvas.drawRightString(width - 18 * mm, 8 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
