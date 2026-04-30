import os
import sys
import types
import json

# Mock out dependencies
class MockModule(types.ModuleType):
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

sys.modules['firebase_admin'] = MockModule('firebase_admin')
sys.modules['firebase_admin.auth'] = MockModule('firebase_admin.auth')
sys.modules['firebase_admin.credentials'] = MockModule('firebase_admin.credentials')

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from app import generate_financial_plan_pdf

# Mock questionnaire data
q = {
    "id": 999,
    "personal_info": {
        "name": "Arjun Test",
        "age": 35
    },
    "family_info": {
        "spouse": "Wife",
        "children": ["Child1"],
        "has_financial_dependents": True
    },
    "lifestyle": {
        "annual_income": 3000000,
        "monthly_expenses": 100000,
        "monthly_emi": 25000,
        "savings_percent": 40
    },
    "insurance": {
        "life_cover": 5000000,
        "health_cover": 1000000
    },
    "goals": {
        "wants_retirement_planning": True,
        "desired_monthly_pension": 50000,
        "items": [
            {
                "name": "Child Education",
                "target_amount": 5000000,
                "horizon_years": 15
            },
            {
                "name": "Car Purchase",
                "target_amount": 1000000,
                "horizon_years": 3
            }
        ]
    }
}

# Mock analysis data
analysis = {
    "riskProfile": "Aggressive",
    "advancedRisk": {
        "score": 85,
        "appetiteCategory": "High",
        "recommendedEquityBand": {"min": 60, "max": 80},
        "recommendedEquityMid": 70
    },
    "surplusBand": "Comfortable",
    "insuranceGap": "UNDERINSURED",
    "debtStress": "Healthy",
    "liquidity": "Low",
    "_diagnostics": {
        "liquidityMonths": 1.5,
        "requiredLifeCover": 20000000,
        "emiPct": 10.0
    },
    "ihs": {
        "score": 55,
        "band": "Needs Attention",
        "breakdown": {
            "portfolio_health": {"score": 35},
            "goal_readiness": {"score": 45},
            "protection": {"score": 25},
            "liquidity": {"score": 30},
            "tax_efficiency": {"score": 80},
            "debt_management": {"score": 95}
        }
    }
}

# Mock document insights
doc_insights = {
    "portfolio": {
        "current_value": 2500000,
        "equity_percentage": 90,
        "debt_percentage": 10
    },
    "bank": {
        "total_inflows": 250000,
        "total_outflows": 150000,
        "net_cashflow": 100000,
        "opening_balance": 50000,
        "closing_balance": 150000
    },
    "itr": {
        "gross_total_income": 3000000,
        "deductions_claimed": [
            {"section": "80C", "amount": 150000},
            {"section": "80D", "amount": 25000}
        ]
    }
}

# Mock sections (LLM outputs)
narratives = {
    "cashflow": {
        "paragraphs": ["You have a strong monthly surplus but need to channel it towards your goals and insurance gaps."],
        "bullets": []
    }
}

# Generate PDF
output_path = "test_report_v2.pdf"
generate_financial_plan_pdf(q, analysis, output_path, doc_insights, narratives)
print(f"Generated successfully to {output_path}")
