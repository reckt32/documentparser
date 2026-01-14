"""
Sentinel Service - Financial Validation Pipeline

This module implements a three-layer validation pipeline:
1. Validation Layer: Critical data checks with interactive question generation
2. Cashflow Layer: Waterfall calculations and allocation priorities
3. Tax Efficiency Layer: FY25-26 tax optimization and missed savings

Usage:
    from sentinel import SentinelOrchestrator
    
    sentinel = SentinelOrchestrator()
    result = sentinel.run(payload)
    
    if result.has_critical_failures:
        return {"status": "needs_input", "questions": result.get_questions(), ...}
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
import math

# =============================================================================
# FY25-26 TAX LIMITS CONFIGURATION
# =============================================================================

TAX_LIMITS_FY2526 = {
    "80C": 150000,              # PPF, ELSS, LIC, NSC, Tuition, Housing Principal
    "80CCD_1B": 50000,          # NPS additional deduction
    "80D": {
        "self_below_60": 25000,
        "self_above_60": 50000,
        "parents_below_60": 25000,
        "parents_above_60": 50000,
        "max_total": 100000,
    },
    "section_24b_housing": 200000,  # Housing loan interest
    "LTCG_equity_exemption": 125000,  # Long-term capital gains exemption
    "standard_deduction": 75000,  # Updated for FY25-26
}

# =============================================================================
# VALIDATION SCHEMA - Critical vs Warning checks
# =============================================================================

VALIDATION_QUESTIONS = {
    "annual_income": {
        "field_path": "income.annualIncome",
        "question": "What is your annual income?",
        "type": "number",
        "required": True,
        "hint": "Include salary, business income, and other regular sources (in ₹)"
    },
    "monthly_expenses": {
        "field_path": "income.monthlyExpenses",
        "question": "What are your average monthly expenses?",
        "type": "number",
        "required": True,
        "hint": "Include rent, utilities, groceries, transport, etc. (in ₹)"
    },
    "age": {
        "field_path": "personal.age",
        "question": "What is your age?",
        "type": "number",
        "required": True,
        "hint": "Your current age in years"
    },
    "emergency_fund": {
        "field_path": "emergencyFundAmount",
        "question": "How much do you have in emergency savings?",
        "type": "number",
        "required": False,
        "hint": "Liquid savings you can access immediately (in ₹)"
    },
    "life_cover": {
        "field_path": "insurance.lifeCover",
        "question": "What is your current life insurance cover amount?",
        "type": "number",
        "required": False,
        "hint": "Sum assured from all term/life policies (in ₹)"
    },
    "health_cover": {
        "field_path": "insurance.healthCover",
        "question": "What is your health insurance cover amount?",
        "type": "number",
        "required": False,
        "hint": "Sum insured for health/mediclaim (in ₹)"
    },
    "monthly_emi": {
        "field_path": "income.monthlyEmi",
        "question": "What is your total monthly EMI amount?",
        "type": "number",
        "required": False,
        "hint": "Include home loan, car loan, personal loan EMIs (in ₹)"
    },
    "dependents_count": {
        "field_path": "personal.dependents_count",
        "question": "How many dependents do you have?",
        "type": "number",
        "required": False,
        "hint": "Number of people financially dependent on you"
    },
    # Advanced validation questions
    "income_source_clarification": {
        "field_path": "_clarification.income_source",
        "question": "Your bank credits significantly exceed your declared income. Please clarify the source of additional funds.",
        "type": "select",
        "required": True,
        "options": [
            {"value": "business_income", "label": "Business income not yet declared in ITR"},
            {"value": "gifts_received", "label": "Gifts or inheritance received"},
            {"value": "loan_proceeds", "label": "Loan or credit facility proceeds"},
            {"value": "investment_redemption", "label": "Investment redemptions"},
            {"value": "other", "label": "Other sources"}
        ],
        "hint": "This helps ensure accurate financial planning"
    },
    "other_bank_accounts": {
        "field_path": "_clarification.other_bank_accounts",
        "question": "Your bank credits are lower than your ITR income. Do you have other bank accounts or receive part of your salary in cash?",
        "type": "select",
        "required": True,
        "options": [
            {"value": "other_accounts", "label": "Yes, I have other bank accounts"},
            {"value": "cash_salary", "label": "Part of salary is in cash"},
            {"value": "both", "label": "Both - other accounts and cash"},
            {"value": "none", "label": "No, this is my only account"}
        ],
        "hint": "This helps reconcile income figures"
    },
    "child_education_goal_clarification": {
        "field_path": "_clarification.child_education_intent",
        "question": "You have a child education goal but no children declared. Please clarify:",
        "type": "select",
        "required": True,
        "options": [
            {"value": "future_child", "label": "Planning for a future child's education"},
            {"value": "relative_child", "label": "Planning for a relative's child (niece/nephew)"},
            {"value": "error", "label": "This goal was added by mistake"}
        ],
        "hint": "We will adjust the goal accordingly"
    },
    "children_count": {
        "field_path": "personal.children_count",
        "question": "How many children do you have?",
        "type": "number",
        "required": False,
        "hint": "Number of children you are financially planning for"
    },
    "retirement_age": {
        "field_path": "personal.retirement_age",
        "question": "At what age do you plan to retire?",
        "type": "number",
        "required": False,
        "hint": "Your target retirement age (typically 58-65)"
    },
    "emi_restructuring_interest": {
        "field_path": "_clarification.emi_restructuring",
        "question": "Your EMI exceeds 50% of income. Would you like guidance on debt restructuring?",
        "type": "select",
        "required": True,
        "options": [
            {"value": "yes_urgent", "label": "Yes, this is urgent - struggling with payments"},
            {"value": "yes_improve", "label": "Yes, want to improve cash flow"},
            {"value": "temporary", "label": "This is temporary - expecting income increase"},
            {"value": "no_managing", "label": "No, I'm managing fine"}
        ],
        "hint": "We'll prioritize debt restructuring in your plan"
    },
}

# =============================================================================
# CASHFLOW MATRIX
# =============================================================================

CASHFLOW_WATERFALL_STEPS = [
    {"name": "gross_inflows", "label": "GROSS INFLOWS", "operation": "add"},
    {"name": "mandatory_outflows", "label": "(-) Mandatory Outflows", "operation": "subtract", 
     "sub_items": ["emi", "taxes"]},
    {"name": "living_expenses", "label": "(-) Living Expenses", "operation": "subtract"},
    {"name": "existing_investments", "label": "(-) Existing SIPs", "operation": "subtract"},
]

ALLOCATION_PRIORITIES = [
    {
        "priority": 1,
        "name": "emergency_fund",
        "label": "Emergency Fund",
        "condition": "liquidity_months < 6",
        "allocation_rule": "min(net_available * 0.3, emergency_fund_gap)",
    },
    {
        "priority": 2,
        "name": "insurance_gap",
        "label": "Insurance Premium",
        "condition": "life_insurance_gap > 0",
        "allocation_rule": "estimated_annual_premium / 12",
    },
    {
        "priority": 3,
        "name": "high_interest_debt",
        "label": "High Interest Debt Prepayment",
        "condition": "has_debt_above_12_percent",
        "allocation_rule": "min(net_available * 0.4, monthly_debt_target)",
    },
]

# =============================================================================
# TAX EFFICIENCY MATRIX
# =============================================================================

TAX_EFFICIENCY_RULES = {
    "capital_gains_strategy": {
        "equity_mutual_funds": {
            "ltcg_threshold": 125000,
            "fy_end_months": [1, 2, 3],  # Jan, Feb, Mar - alert window
            "recommendation_template": "Book gains of up to ₹{headroom:,.0f} tax-free before March 31st"
        }
    },
    "missed_savings_sections": [
        {"section": "80C", "limit": 150000, "description": "ELSS, PPF, LIC, NSC, Tuition Fees"},
        {"section": "80D", "limit": 25000, "description": "Health Insurance Premium (self/family)"},
        {"section": "80CCD(1B)", "limit": 50000, "description": "NPS Contribution"},
    ]
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ValidationIssue:
    """Represents a validation issue (critical or warning)."""
    field: str
    message: str
    severity: str  # "critical" or "warning"
    question_id: Optional[str] = None


@dataclass
class CashflowWaterfallItem:
    """Represents a line item in the cashflow waterfall."""
    name: str
    label: str
    amount: float
    operation: str  # "add" or "subtract"
    sub_items: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class AllocationPriority:
    """Represents an allocation priority with amount."""
    priority: int
    name: str
    label: str
    condition_met: bool
    allocated_amount: float
    reasoning: str


@dataclass
class TaxRecommendation:
    """Represents a tax efficiency recommendation."""
    section: str
    gap_amount: float
    potential_tax_saved: float
    recommendation: str
    deadline: str


@dataclass
class SentinelResult:
    """Result object from Sentinel pipeline run."""
    has_critical_failures: bool = False
    critical_issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    
    # Cashflow data
    cashflow_waterfall: List[Dict[str, Any]] = field(default_factory=list)
    net_available: float = 0.0
    allocation_priorities: List[Dict[str, Any]] = field(default_factory=list)
    
    # Tax efficiency data
    tax_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    total_tax_alpha: float = 0.0
    ltcg_harvest_recommendation: Optional[Dict[str, Any]] = None
    
    def get_questions(self) -> List[Dict[str, Any]]:
        """Generate structured questions for missing critical data."""
        questions = []
        for field_id in self.missing_fields:
            if field_id in VALIDATION_QUESTIONS:
                q = VALIDATION_QUESTIONS[field_id].copy()
                q["id"] = field_id
                questions.append(q)
        return questions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "has_critical_failures": self.has_critical_failures,
            "critical_issues": [{"field": i.field, "message": i.message} for i in self.critical_issues],
            "warnings": [{"field": i.field, "message": i.message} for i in self.warnings],
            "missing_fields": self.missing_fields,
            "cashflow_waterfall": self.cashflow_waterfall,
            "net_available": self.net_available,
            "allocation_priorities": self.allocation_priorities,
            "tax_recommendations": self.tax_recommendations,
            "total_tax_alpha": self.total_tax_alpha,
            "ltcg_harvest_recommendation": self.ltcg_harvest_recommendation,
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        # Handle strings with commas or currency symbols
        s = str(value).strip().replace(",", "").replace("₹", "").replace("Rs.", "").replace("Rs", "")
        return float(s) if s and s != "N/A" else default
    except (ValueError, TypeError):
        return default


def _get_nested(data: Dict, path: str, default: Any = None) -> Any:
    """Get nested dictionary value by dot-separated path."""
    keys = path.split(".")
    result = data
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key)
        else:
            return default
        if result is None:
            return default
    return result


def _set_nested(data: Dict, path: str, value: Any) -> None:
    """Set nested dictionary value by dot-separated path."""
    keys = path.split(".")
    for key in keys[:-1]:
        data = data.setdefault(key, {})
    data[keys[-1]] = value


def _is_fy_ending_soon(months_ahead: int = 3) -> bool:
    """Check if financial year is ending within specified months."""
    today = date.today()
    # FY ends March 31st
    fy_end_month = 3
    current_month = today.month
    
    # Check if we're in the alert window (Jan, Feb, Mar)
    return current_month in [1, 2, 3]


def _calculate_tax_saved(deduction_amount: float, tax_slab_rate: float = 0.30) -> float:
    """Estimate tax saved from a deduction amount."""
    # Using 30% marginal rate as default for higher income brackets
    return deduction_amount * tax_slab_rate


def _format_indian_amount(amount: float) -> str:
    """Format amount in Indian number format (lakhs/crores)."""
    if amount >= 10000000:  # 1 crore
        return f"₹{amount/10000000:.2f} Cr"
    elif amount >= 100000:  # 1 lakh
        return f"₹{amount/100000:.2f} L"
    else:
        return f"₹{amount:,.0f}"


# =============================================================================
# VALIDATOR CLASS
# =============================================================================

class SentinelValidator:
    """
    Validation Layer - Checks critical data completeness and consistency.
    Returns questions for frontend when data is missing or inconsistent.
    
    Implements validation rules:
    1. Basic field validation (income, age)
    2. Income plausibility (ITR vs bank credits)
    3. Goal-family validation (child education with no children)
    4. EMI burden validation (>50% of income)
    5. Retirement horizon validation (less than 10 years)
    """
    
    def __init__(self):
        self.critical_fields = [
            ("annual_income", "income.annualIncome", lambda x: _safe_float(x) > 0),
            ("age", "personal.age", lambda x: 18 <= _safe_float(x) <= 100),
        ]
        self.warning_fields = [
            ("emergency_fund", "emergencyFundAmount", lambda x: _safe_float(x) > 0),
            ("life_cover", "insurance.lifeCover", lambda x: _safe_float(x) > 0),
            ("monthly_expenses", "income.monthlyExpenses", lambda x: _safe_float(x) > 0),
        ]
    
    def validate(self, payload: Dict[str, Any]) -> Tuple[List[ValidationIssue], List[ValidationIssue], List[str]]:
        """
        Validate payload and return (critical_issues, warnings, missing_fields).
        """
        critical_issues = []
        warnings = []
        missing_fields = []
        
        # =====================================================================
        # BASIC FIELD VALIDATION
        # =====================================================================
        
        # Check critical fields
        for field_id, path, validator in self.critical_fields:
            value = _get_nested(payload, path)
            if value is None or not validator(value):
                issue = ValidationIssue(
                    field=path,
                    message=VALIDATION_QUESTIONS.get(field_id, {}).get("hint", f"Missing or invalid: {field_id}"),
                    severity="critical",
                    question_id=field_id
                )
                critical_issues.append(issue)
                missing_fields.append(field_id)
        
        # Get common values for subsequent checks
        income = _safe_float(_get_nested(payload, "income.annualIncome"))
        monthly_income = income / 12 if income > 0 else 0
        expenses = _safe_float(_get_nested(payload, "income.monthlyExpenses")) * 12
        emi = _safe_float(_get_nested(payload, "income.monthlyEmi")) * 12
        age = _safe_float(_get_nested(payload, "personal.age"))
        
        # Check for negative cashflow (critical)
        if income > 0 and (expenses + emi) > income:
            critical_issues.append(ValidationIssue(
                field="cashflow",
                message="Your expenses and EMI exceed your income. Please review the figures.",
                severity="critical",
                question_id=None
            ))
        
        # =====================================================================
        # INCOME PLAUSIBILITY: ITR vs Bank Credits
        # =====================================================================
        
        bank_data = payload.get("bank", {})
        itr_data = payload.get("itr", {})
        
        bank_credits = _safe_float(bank_data.get("total_inflows") or bank_data.get("total_credits"))
        itr_income = _safe_float(itr_data.get("gross_total_income"))
        
        if bank_credits > 0 and itr_income > 0:
            # Bank credits significantly exceed declared income (>150%)
            if bank_credits > (itr_income * 1.5):
                critical_issues.append(ValidationIssue(
                    field="income_plausibility",
                    message=f"Bank credits (₹{bank_credits:,.0f}) significantly exceed declared ITR income (₹{itr_income:,.0f}). Possible undeclared income or business receipts.",
                    severity="critical",
                    question_id="income_source_clarification"
                ))
                missing_fields.append("income_source_clarification")
            
            # Bank credits lower than expected (<70% of ITR income)
            elif bank_credits < (itr_income * 0.7):
                warnings.append(ValidationIssue(
                    field="income_plausibility",
                    message=f"Bank credits (₹{bank_credits:,.0f}) are lower than ITR income (₹{itr_income:,.0f}). Possible multiple bank accounts or cash salary.",
                    severity="warning",
                    question_id="other_bank_accounts"
                ))
                # This is a warning, so we add to missing_fields to prompt the question
                if "other_bank_accounts" not in missing_fields:
                    missing_fields.append("other_bank_accounts")
        
        # =====================================================================
        # GOAL-FAMILY VALIDATION: Child Education Goal with No Children
        # =====================================================================
        
        goals_list = payload.get("goals", {}).get("goalsList", [])
        children_count = _safe_float(_get_nested(payload, "personal.children_count"))
        
        # Check for child education goals
        child_education_goals = [
            g for g in goals_list 
            if g.get("goalType", "").lower() in ["child_education", "childeducation", "education"]
            or "child" in g.get("goalName", "").lower()
            or "education" in g.get("goalName", "").lower()
        ]
        
        if len(child_education_goals) > 0 and children_count == 0:
            critical_issues.append(ValidationIssue(
                field="goals.child_education",
                message=f"Child education goal exists ('{child_education_goals[0].get('goalName', 'Child Education')}') but no children declared.",
                severity="critical",
                question_id="child_education_goal_clarification"
            ))
            missing_fields.append("child_education_goal_clarification")
        
        # =====================================================================
        # EMI BURDEN VALIDATION: >50% of Income
        # =====================================================================
        
        if income > 0:
            emi_pct = (emi / income) * 100
            
            if emi_pct > 50:
                # Critical - EMI exceeds 50%, needs debt restructuring
                critical_issues.append(ValidationIssue(
                    field="income.monthlyEmi",
                    message=f"EMI exceeds 50% of income ({emi_pct:.1f}%). High debt stress. Debt restructuring becomes Priority 1.",
                    severity="critical",
                    question_id="emi_restructuring_interest"
                ))
                missing_fields.append("emi_restructuring_interest")
            elif emi_pct > 40:
                # Warning - EMI between 40-50%
                warnings.append(ValidationIssue(
                    field="income.monthlyEmi",
                    message=f"EMI is {emi_pct:.1f}% of income, which is high (>40%). Consider reducing debt.",
                    severity="warning"
                ))
        
        # =====================================================================
        # RETIREMENT HORIZON VALIDATION
        # =====================================================================
        
        retirement_age = _safe_float(_get_nested(payload, "personal.retirement_age"))
        
        # Check for retirement goals to infer retirement age
        if retirement_age == 0:
            retirement_goals = [
                g for g in goals_list
                if g.get("goalType", "").lower() in ["retirement", "retire"]
                or "retire" in g.get("goalName", "").lower()
            ]
            if retirement_goals:
                # Try to infer retirement age from goal horizon
                horizon = _safe_float(retirement_goals[0].get("horizon") or retirement_goals[0].get("timeframe"))
                if horizon > 0 and age > 0:
                    retirement_age = age + horizon
        
        # Default retirement age if not specified
        if retirement_age == 0 and age > 0:
            retirement_age = 60  # Default assumption
        
        if retirement_age > 0 and age > 0:
            years_to_retirement = retirement_age - age
            
            if years_to_retirement < 10 and years_to_retirement > 0:
                warnings.append(ValidationIssue(
                    field="personal.retirement_age",
                    message=f"Retirement in less than 10 years ({years_to_retirement:.0f} years). Limited time for wealth accumulation. Adjusting equity allocation downward and focusing on capital preservation.",
                    severity="warning",
                    question_id="retirement_age"
                ))
        
        # =====================================================================
        # WARNING FIELD CHECKS
        # =====================================================================
        
        for field_id, path, validator in self.warning_fields:
            value = _get_nested(payload, path)
            if value is None or not validator(value):
                # Check if dependents exist for insurance warning
                if field_id == "life_cover":
                    dependents = _safe_float(_get_nested(payload, "personal.dependents_count"))
                    if dependents > 0:
                        warnings.append(ValidationIssue(
                            field=path,
                            message="You have dependents but no life insurance coverage declared.",
                            severity="warning",
                            question_id=field_id
                        ))
                else:
                    warnings.append(ValidationIssue(
                        field=path,
                        message=VALIDATION_QUESTIONS.get(field_id, {}).get("hint", f"Consider providing: {field_id}"),
                        severity="warning",
                        question_id=field_id
                    ))
        
        return critical_issues, warnings, missing_fields


# =============================================================================
# CASHFLOW CLASS
# =============================================================================

class SentinelCashflow:
    """
    Cashflow Layer - Calculates waterfall and allocation priorities.
    """
    
    def calculate_waterfall(self, payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], float]:
        """
        Calculate cashflow waterfall and return (waterfall_items, net_available).
        """
        income = payload.get("income", {})
        
        # Get monthly values
        annual_income = _safe_float(income.get("annualIncome"))
        monthly_income = annual_income / 12 if annual_income > 0 else 0
        
        monthly_expenses = _safe_float(income.get("monthlyExpenses"))
        monthly_emi = _safe_float(income.get("monthlyEmi"))
        
        # Get existing SIP from portfolio if available
        portfolio = payload.get("portfolio", {})
        existing_sip = _safe_float(portfolio.get("monthly_sip") or portfolio.get("total_monthly_sip"))
        
        # Estimate taxes (rough: 20% effective for middle income)
        # This is a simplification; real tax calc would need ITR data
        monthly_tax = monthly_income * 0.15 if monthly_income > 50000 else monthly_income * 0.05
        
        waterfall = []
        running_total = 0.0
        
        # Gross Inflows
        waterfall.append({
            "name": "gross_inflows",
            "label": "GROSS INFLOWS",
            "amount": monthly_income,
            "operation": "add",
            "formatted": _format_indian_amount(monthly_income)
        })
        running_total = monthly_income
        
        # Mandatory Outflows (EMI + Taxes)
        mandatory_total = monthly_emi + monthly_tax
        waterfall.append({
            "name": "mandatory_outflows",
            "label": "(-) Mandatory Outflows",
            "amount": mandatory_total,
            "operation": "subtract",
            "sub_items": [
                {"name": "EMI", "amount": monthly_emi, "formatted": _format_indian_amount(monthly_emi)},
                {"name": "Taxes (est.)", "amount": monthly_tax, "formatted": _format_indian_amount(monthly_tax)},
            ],
            "formatted": _format_indian_amount(mandatory_total)
        })
        running_total -= mandatory_total
        
        # Living Expenses
        waterfall.append({
            "name": "living_expenses",
            "label": "(-) Living Expenses",
            "amount": monthly_expenses,
            "operation": "subtract",
            "formatted": _format_indian_amount(monthly_expenses)
        })
        running_total -= monthly_expenses
        
        # Existing SIPs
        if existing_sip > 0:
            waterfall.append({
                "name": "existing_investments",
                "label": "(-) Existing SIPs",
                "amount": existing_sip,
                "operation": "subtract",
                "formatted": _format_indian_amount(existing_sip)
            })
            running_total -= existing_sip
        
        # Net Available
        net_available = max(0, running_total)
        waterfall.append({
            "name": "net_available",
            "label": "= NET AVAILABLE",
            "amount": net_available,
            "operation": "result",
            "formatted": _format_indian_amount(net_available),
            "is_negative": running_total < 0
        })
        
        return waterfall, net_available
    
    def calculate_allocation_priorities(
        self, 
        payload: Dict[str, Any], 
        net_available: float
    ) -> List[Dict[str, Any]]:
        """
        Calculate allocation priorities based on financial gaps.
        """
        priorities = []
        remaining = net_available
        
        # Priority 1: Emergency Fund
        monthly_expenses = _safe_float(_get_nested(payload, "income.monthlyExpenses"))
        emergency_fund = _safe_float(payload.get("emergencyFundAmount"))
        target_emergency = monthly_expenses * 6  # 6 months
        emergency_gap = max(0, target_emergency - emergency_fund)
        liquidity_months = (emergency_fund / monthly_expenses) if monthly_expenses > 0 else 0
        
        if liquidity_months < 6 and emergency_gap > 0:
            allocation = min(remaining * 0.3, emergency_gap / 12)  # Monthly allocation
            priorities.append({
                "priority": 1,
                "name": "emergency_fund",
                "label": "Emergency Fund",
                "condition_met": True,
                "allocated_amount": allocation,
                "reasoning": f"Current: {liquidity_months:.1f} months, Target: 6 months",
                "formatted": _format_indian_amount(allocation)
            })
            remaining -= allocation
        
        # Priority 2: Insurance Gap
        annual_income = _safe_float(_get_nested(payload, "income.annualIncome"))
        life_cover = _safe_float(_get_nested(payload, "insurance.lifeCover"))
        required_cover = annual_income * 10  # 10x income rule
        insurance_gap = max(0, required_cover - life_cover)
        
        if insurance_gap > 0:
            # Estimate premium: ~₹600-800 per lakh for age 30-40
            age = _safe_float(_get_nested(payload, "personal.age"), 35)
            premium_per_lakh = 500 if age < 35 else (700 if age < 45 else 1200)
            annual_premium = (insurance_gap / 100000) * premium_per_lakh
            monthly_premium = annual_premium / 12
            
            priorities.append({
                "priority": 2,
                "name": "insurance_gap",
                "label": "Term Insurance Premium",
                "condition_met": True,
                "allocated_amount": monthly_premium,
                "reasoning": f"Need additional cover of {_format_indian_amount(insurance_gap)}",
                "formatted": _format_indian_amount(monthly_premium)
            })
            remaining -= monthly_premium
        
        # Priority 3: High Interest Debt
        monthly_emi = _safe_float(_get_nested(payload, "income.monthlyEmi"))
        if monthly_emi > 0:
            # Suggest prepayment if EMI > 30% of income
            monthly_income = annual_income / 12 if annual_income > 0 else 0
            emi_pct = (monthly_emi / monthly_income * 100) if monthly_income > 0 else 0
            
            if emi_pct > 30 and remaining > 0:
                prepay_amount = min(remaining * 0.4, monthly_emi * 0.5)
                priorities.append({
                    "priority": 3,
                    "name": "high_interest_debt",
                    "label": "Debt Prepayment",
                    "condition_met": True,
                    "allocated_amount": prepay_amount,
                    "reasoning": f"EMI is {emi_pct:.1f}% of income; prepay high-interest debt first",
                    "formatted": _format_indian_amount(prepay_amount)
                })
                remaining -= prepay_amount
        
        # Priority 4: Goal Investments (remaining surplus)
        if remaining > 0:
            priorities.append({
                "priority": 4,
                "name": "goal_investments",
                "label": "Goal-based Investments",
                "condition_met": True,
                "allocated_amount": remaining,
                "reasoning": "Remaining surplus for goal-based SIPs",
                "formatted": _format_indian_amount(remaining)
            })
        
        return priorities


# =============================================================================
# TAX EFFICIENCY CLASS
# =============================================================================

class SentinelTax:
    """
    Tax Efficiency Layer - Calculates missed savings and LTCG harvesting opportunities.
    """
    
    def calculate_tax_efficiency(self, payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], float, Optional[Dict[str, Any]]]:
        """
        Calculate tax efficiency recommendations.
        Returns (recommendations, total_tax_alpha, ltcg_harvest_recommendation)
        """
        recommendations = []
        total_tax_alpha = 0.0
        
        # Get ITR/deduction data if available
        itr_data = payload.get("itr", {})
        deductions_claimed = itr_data.get("deductions_claimed", [])
        
        # Create a lookup of claimed deductions
        claimed = {}
        for d in deductions_claimed:
            section = d.get("section", "").upper().replace("-", "").replace(" ", "")
            claimed[section] = _safe_float(d.get("amount"))
        
        # Calculate FY end date
        today = date.today()
        if today.month <= 3:
            fy_end = date(today.year, 3, 31)
        else:
            fy_end = date(today.year + 1, 3, 31)
        days_to_fy_end = (fy_end - today).days
        deadline = fy_end.strftime("%B %d, %Y")
        
        # Section 80C Gap
        claimed_80c = claimed.get("80C", 0)
        limit_80c = TAX_LIMITS_FY2526["80C"]
        gap_80c = max(0, limit_80c - claimed_80c)
        
        if gap_80c > 0:
            tax_saved = _calculate_tax_saved(gap_80c)
            recommendations.append({
                "section": "80C",
                "gap_amount": gap_80c,
                "potential_tax_saved": tax_saved,
                "recommendation": f"Invest ₹{gap_80c:,.0f} in ELSS/PPF by {deadline}",
                "deadline": deadline,
                "formatted_gap": _format_indian_amount(gap_80c),
                "formatted_savings": _format_indian_amount(tax_saved)
            })
            total_tax_alpha += tax_saved
        
        # Section 80D Gap (Health Insurance)
        health_premium = _safe_float(_get_nested(payload, "insurance.healthPremium"))
        claimed_80d = claimed.get("80D", health_premium)
        age = _safe_float(_get_nested(payload, "personal.age"), 35)
        limit_80d = TAX_LIMITS_FY2526["80D"]["self_above_60" if age >= 60 else "self_below_60"]
        gap_80d = max(0, limit_80d - claimed_80d)
        
        if gap_80d > 0:
            tax_saved = _calculate_tax_saved(gap_80d)
            recommendations.append({
                "section": "80D",
                "gap_amount": gap_80d,
                "potential_tax_saved": tax_saved,
                "recommendation": f"Pay health premium of ₹{gap_80d:,.0f} or top-up policy by {deadline}",
                "deadline": deadline,
                "formatted_gap": _format_indian_amount(gap_80d),
                "formatted_savings": _format_indian_amount(tax_saved)
            })
            total_tax_alpha += tax_saved
        
        # Section 80CCD(1B) - NPS
        claimed_nps = claimed.get("80CCD1B", claimed.get("80CCD", 0))
        limit_nps = TAX_LIMITS_FY2526["80CCD_1B"]
        gap_nps = max(0, limit_nps - claimed_nps)
        
        if gap_nps > 0:
            tax_saved = _calculate_tax_saved(gap_nps)
            recommendations.append({
                "section": "80CCD(1B)",
                "gap_amount": gap_nps,
                "potential_tax_saved": tax_saved,
                "recommendation": f"Contribute ₹{gap_nps:,.0f} to NPS by {deadline}",
                "deadline": deadline,
                "formatted_gap": _format_indian_amount(gap_nps),
                "formatted_savings": _format_indian_amount(tax_saved)
            })
            total_tax_alpha += tax_saved
        
        # LTCG Harvesting Check
        ltcg_recommendation = self._check_ltcg_harvesting(payload, deadline, days_to_fy_end)
        
        return recommendations, total_tax_alpha, ltcg_recommendation
    
    def _check_ltcg_harvesting(
        self, 
        payload: Dict[str, Any], 
        deadline: str,
        days_to_fy_end: int
    ) -> Optional[Dict[str, Any]]:
        """Check for LTCG harvesting opportunity."""
        # Only suggest if FY is ending in 3 months
        if days_to_fy_end > 90:
            return None
        
        # Get unrealized gains from portfolio
        portfolio = payload.get("portfolio", {})
        cas_data = payload.get("cas", {})
        
        # Try multiple sources for unrealized gains
        unrealized_gain = _safe_float(
            portfolio.get("unrealized_gain") or 
            portfolio.get("total_unrealized_gain") or
            cas_data.get("transaction_summary", {}).get("total_unrealized_gain")
        )
        
        if unrealized_gain <= 0:
            return None
        
        ltcg_threshold = TAX_LIMITS_FY2526["LTCG_equity_exemption"]
        
        # Check if there's headroom for tax-free harvesting
        if unrealized_gain < ltcg_threshold:
            remaining_headroom = ltcg_threshold - unrealized_gain
            return {
                "current_unrealized_ltcg": unrealized_gain,
                "ltcg_threshold": ltcg_threshold,
                "remaining_headroom": remaining_headroom,
                "recommendation": f"Book gains of up to ₹{remaining_headroom:,.0f} tax-free before {deadline}",
                "action": "HARVEST_GAINS",
                "benefit": "Reset cost base tax-free, use ₹1.25L annual exemption"
            }
        else:
            # Already exceeds threshold - no harvesting opportunity
            return {
                "current_unrealized_ltcg": unrealized_gain,
                "ltcg_threshold": ltcg_threshold,
                "remaining_headroom": 0,
                "recommendation": f"Unrealized gains (₹{unrealized_gain:,.0f}) exceed ₹1.25L exemption. Plan redemptions carefully.",
                "action": "PLAN_REDEMPTIONS",
                "benefit": "Spread redemptions across FYs to minimize tax"
            }


# =============================================================================
# ORCHESTRATOR CLASS
# =============================================================================

class SentinelOrchestrator:
    """
    Main orchestrator that runs the three-layer pipeline:
    Validation → Cashflow → Tax Efficiency
    """
    
    def __init__(self):
        self.validator = SentinelValidator()
        self.cashflow = SentinelCashflow()
        self.tax = SentinelTax()
    
    def run(self, payload: Dict[str, Any]) -> SentinelResult:
        """
        Run the complete Sentinel pipeline.
        
        Returns SentinelResult with:
        - has_critical_failures: True if frontend needs to collect more data
        - questions: Structured questions for missing data
        - cashflow_waterfall: Visual waterfall breakdown
        - tax_recommendations: Actionable tax savings
        """
        result = SentinelResult()
        
        # Layer 1: Validation
        critical_issues, warnings, missing_fields = self.validator.validate(payload)
        result.critical_issues = critical_issues
        result.warnings = warnings
        result.missing_fields = missing_fields
        result.has_critical_failures = len(critical_issues) > 0
        
        # If critical failures, return early with questions
        if result.has_critical_failures:
            return result
        
        # Layer 2: Cashflow
        waterfall, net_available = self.cashflow.calculate_waterfall(payload)
        result.cashflow_waterfall = waterfall
        result.net_available = net_available
        
        allocation_priorities = self.cashflow.calculate_allocation_priorities(payload, net_available)
        result.allocation_priorities = allocation_priorities
        
        # Layer 3: Tax Efficiency
        tax_recs, total_alpha, ltcg_rec = self.tax.calculate_tax_efficiency(payload)
        result.tax_recommendations = tax_recs
        result.total_tax_alpha = total_alpha
        result.ltcg_harvest_recommendation = ltcg_rec
        
        return result


# =============================================================================
# PUBLIC API
# =============================================================================

def run_sentinel_pipeline(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Public function to run the Sentinel pipeline.
    
    Returns dict with either:
    - {"status": "needs_input", "questions": [...], ...} if data is missing
    - {"status": "complete", "cashflow_waterfall": [...], "tax_efficiency": {...}, ...} if successful
    """
    sentinel = SentinelOrchestrator()
    result = sentinel.run(payload)
    
    if result.has_critical_failures:
        return {
            "status": "needs_input",
            "questions": result.get_questions(),
            "preserve_payload": payload,
            "missing_fields": result.missing_fields,
            "critical_issues": [{"field": i.field, "message": i.message} for i in result.critical_issues],
        }
    
    return {
        "status": "complete",
        "warnings": [{"field": i.field, "message": i.message} for i in result.warnings],
        "cashflow_waterfall": result.cashflow_waterfall,
        "net_available": result.net_available,
        "allocation_priorities": result.allocation_priorities,
        "tax_recommendations": result.tax_recommendations,
        "total_tax_alpha": result.total_tax_alpha,
        "ltcg_harvest_recommendation": result.ltcg_harvest_recommendation,
    }
