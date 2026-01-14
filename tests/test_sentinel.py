"""
Test suite for Sentinel Service

Tests the three-layer validation pipeline:
1. Validation - critical/warning checks and question generation
2. Cashflow - waterfall calculation and allocation priorities
3. Tax Efficiency - missed savings and LTCG harvesting
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from sentinel import (
    SentinelOrchestrator,
    SentinelValidator,
    SentinelCashflow,
    SentinelTax,
    run_sentinel_pipeline,
    TAX_LIMITS_FY2526,
    VALIDATION_QUESTIONS,
)


# =============================================================================
# SAMPLE PAYLOADS
# =============================================================================

VALID_PAYLOAD = {
    "personal": {
        "age": 35,
        "dependents_count": 2,
    },
    "income": {
        "annualIncome": 1800000,  # 18L
        "monthlyExpenses": 60000,
        "monthlyEmi": 30000,
    },
    "insurance": {
        "lifeCover": 5000000,  # 50L
        "healthCover": 500000,  # 5L
    },
    "emergencyFundAmount": 200000,  # 2L
    "savings": {
        "savingsPercent": 25,
    },
    "investments": {
        "allocation": {
            "equity": 60,
            "debt": 30,
            "gold": 10,
        },
        "current": ["mf", "fd"],
    },
    "portfolio": {
        "monthly_sip": 25000,
        "unrealized_gain": 80000,  # For LTCG testing
    },
    "itr": {
        "deductions_claimed": [
            {"section": "80C", "amount": 100000},
            {"section": "80D", "amount": 10000},
        ],
    },
    "risk": {
        "tolerance": "medium",
    },
    "goals": {
        "goalHorizon": 10,
    },
}

MISSING_INCOME_PAYLOAD = {
    "personal": {
        "age": 35,
    },
    "income": {
        "annualIncome": 0,  # Missing income
        "monthlyExpenses": 50000,
    },
}

MISSING_AGE_PAYLOAD = {
    "personal": {},  # No age
    "income": {
        "annualIncome": 1200000,
        "monthlyExpenses": 40000,
    },
}

NEGATIVE_CASHFLOW_PAYLOAD = {
    "personal": {
        "age": 30,
    },
    "income": {
        "annualIncome": 600000,  # 50k/month
        "monthlyExpenses": 45000,
        "monthlyEmi": 25000,  # Total 70k > 50k income
    },
}


# =============================================================================
# VALIDATION LAYER TESTS
# =============================================================================

class TestSentinelValidator:
    """Test the validation layer."""
    
    def test_valid_payload_passes(self):
        """Valid payload should have no critical issues."""
        validator = SentinelValidator()
        critical, warnings, missing = validator.validate(VALID_PAYLOAD)
        
        assert len(critical) == 0, f"Expected no critical issues, got: {critical}"
        assert len(missing) == 0, f"Expected no missing fields, got: {missing}"
    
    def test_missing_income_returns_question(self):
        """Missing income should trigger critical issue with question."""
        validator = SentinelValidator()
        critical, warnings, missing = validator.validate(MISSING_INCOME_PAYLOAD)
        
        assert len(critical) > 0, "Expected critical issues for missing income"
        assert "annual_income" in missing, "Expected annual_income in missing fields"
    
    def test_missing_age_returns_question(self):
        """Missing age should trigger critical issue with question."""
        validator = SentinelValidator()
        critical, warnings, missing = validator.validate(MISSING_AGE_PAYLOAD)
        
        assert len(critical) > 0, "Expected critical issues for missing age"
        assert "age" in missing, "Expected age in missing fields"
    
    def test_negative_cashflow_critical(self):
        """Expenses exceeding income should be a critical issue."""
        validator = SentinelValidator()
        critical, warnings, missing = validator.validate(NEGATIVE_CASHFLOW_PAYLOAD)
        
        # Find the cashflow issue
        cashflow_issues = [c for c in critical if c.field == "cashflow"]
        assert len(cashflow_issues) > 0, "Expected cashflow critical issue"
    
    def test_warning_for_no_emergency_fund(self):
        """No emergency fund should trigger a warning."""
        payload = VALID_PAYLOAD.copy()
        payload["emergencyFundAmount"] = 0
        
        validator = SentinelValidator()
        critical, warnings, missing = validator.validate(payload)
        
        assert len(critical) == 0, "Should not have critical issues"
        assert len(warnings) > 0, "Expected warnings for missing emergency fund"


# =============================================================================
# CASHFLOW LAYER TESTS
# =============================================================================

class TestSentinelCashflow:
    """Test the cashflow layer."""
    
    def test_waterfall_calculation(self):
        """Waterfall should correctly calculate net available."""
        cashflow = SentinelCashflow()
        waterfall, net_available = cashflow.calculate_waterfall(VALID_PAYLOAD)
        
        # Check waterfall structure
        assert len(waterfall) >= 4, "Waterfall should have at least 4 items"
        
        # Find net available item
        net_item = next((w for w in waterfall if w["name"] == "net_available"), None)
        assert net_item is not None, "Waterfall should have net_available item"
        
        # Net available should be positive for valid payload
        assert net_available >= 0, f"Net available should be >= 0, got {net_available}"
    
    def test_allocation_priorities(self):
        """Allocation priorities should follow correct order."""
        cashflow = SentinelCashflow()
        _, net_available = cashflow.calculate_waterfall(VALID_PAYLOAD)
        priorities = cashflow.calculate_allocation_priorities(VALID_PAYLOAD, net_available)
        
        # Should have some priorities
        assert len(priorities) > 0, "Should have allocation priorities"
        
        # Emergency fund should be priority 1 if liquidity < 6 months
        ef_priority = next((p for p in priorities if p["name"] == "emergency_fund"), None)
        if ef_priority:
            assert ef_priority["priority"] == 1, "Emergency fund should be priority 1"
    
    def test_insurance_gap_allocation(self):
        """Should allocate for insurance gap when underinsured."""
        payload = VALID_PAYLOAD.copy()
        payload["insurance"] = {"lifeCover": 0, "healthCover": 0}  # Underinsured
        
        cashflow = SentinelCashflow()
        _, net_available = cashflow.calculate_waterfall(payload)
        priorities = cashflow.calculate_allocation_priorities(payload, net_available)
        
        # Should have insurance gap priority
        ins_priority = next((p for p in priorities if p["name"] == "insurance_gap"), None)
        assert ins_priority is not None, "Should have insurance gap priority when underinsured"


# =============================================================================
# TAX EFFICIENCY LAYER TESTS
# =============================================================================

class TestSentinelTax:
    """Test the tax efficiency layer."""
    
    def test_80c_gap_calculation(self):
        """Should calculate 80C gap correctly."""
        tax = SentinelTax()
        recommendations, total_alpha, ltcg = tax.calculate_tax_efficiency(VALID_PAYLOAD)
        
        # Check for 80C recommendation
        rec_80c = next((r for r in recommendations if r["section"] == "80C"), None)
        assert rec_80c is not None, "Should have 80C recommendation"
        
        # Gap should be 150000 - 100000 = 50000
        assert rec_80c["gap_amount"] == 50000, f"80C gap should be 50000, got {rec_80c['gap_amount']}"
    
    def test_80d_gap_calculation(self):
        """Should calculate 80D gap correctly."""
        tax = SentinelTax()
        recommendations, total_alpha, ltcg = tax.calculate_tax_efficiency(VALID_PAYLOAD)
        
        # Check for 80D recommendation
        rec_80d = next((r for r in recommendations if r["section"] == "80D"), None)
        assert rec_80d is not None, "Should have 80D recommendation"
        
        # Gap should be 25000 - 10000 = 15000
        assert rec_80d["gap_amount"] == 15000, f"80D gap should be 15000, got {rec_80d['gap_amount']}"
    
    def test_ltcg_harvest_recommendation(self):
        """Should recommend LTCG harvesting when FY ending soon."""
        tax = SentinelTax()
        recommendations, total_alpha, ltcg = tax.calculate_tax_efficiency(VALID_PAYLOAD)
        
        # LTCG recommendation depends on current month
        # Just check the structure if it exists
        if ltcg is not None:
            assert "current_unrealized_ltcg" in ltcg
            assert "ltcg_threshold" in ltcg
            assert "recommendation" in ltcg
    
    def test_tax_limits_fy2526(self):
        """Verify FY25-26 tax limits are configured correctly."""
        assert TAX_LIMITS_FY2526["80C"] == 150000
        assert TAX_LIMITS_FY2526["80CCD_1B"] == 50000
        assert TAX_LIMITS_FY2526["LTCG_equity_exemption"] == 125000
        assert TAX_LIMITS_FY2526["80D"]["self_below_60"] == 25000
        assert TAX_LIMITS_FY2526["80D"]["self_above_60"] == 50000


# =============================================================================
# ORCHESTRATOR TESTS
# =============================================================================

class TestSentinelOrchestrator:
    """Test the complete Sentinel pipeline."""
    
    def test_valid_payload_completes(self):
        """Valid payload should complete the full pipeline."""
        sentinel = SentinelOrchestrator()
        result = sentinel.run(VALID_PAYLOAD)
        
        assert not result.has_critical_failures, f"Should not have critical failures: {result.critical_issues}"
        assert len(result.cashflow_waterfall) > 0, "Should have cashflow waterfall"
        assert len(result.tax_recommendations) > 0, "Should have tax recommendations"
    
    def test_missing_data_returns_questions(self):
        """Missing critical data should return questions."""
        sentinel = SentinelOrchestrator()
        result = sentinel.run(MISSING_INCOME_PAYLOAD)
        
        assert result.has_critical_failures, "Should have critical failures"
        
        questions = result.get_questions()
        assert len(questions) > 0, "Should return questions"
        
        # Check question structure
        q = questions[0]
        assert "id" in q
        assert "field_path" in q
        assert "question" in q
        assert "type" in q
    
    def test_sentinel_result_to_dict(self):
        """Result should serialize to dict correctly."""
        sentinel = SentinelOrchestrator()
        result = sentinel.run(VALID_PAYLOAD)
        
        result_dict = result.to_dict()
        
        assert "has_critical_failures" in result_dict
        assert "cashflow_waterfall" in result_dict
        assert "tax_recommendations" in result_dict
        assert "total_tax_alpha" in result_dict


# =============================================================================
# PUBLIC API TESTS
# =============================================================================

class TestPublicAPI:
    """Test the public API function."""
    
    def test_run_sentinel_pipeline_complete(self):
        """Complete pipeline should return status=complete."""
        result = run_sentinel_pipeline(VALID_PAYLOAD)
        
        assert result["status"] == "complete", f"Expected status=complete, got {result['status']}"
        assert "cashflow_waterfall" in result
        assert "tax_recommendations" in result
    
    def test_run_sentinel_pipeline_needs_input(self):
        """Missing data should return status=needs_input with questions."""
        result = run_sentinel_pipeline(MISSING_INCOME_PAYLOAD)
        
        assert result["status"] == "needs_input", f"Expected status=needs_input, got {result['status']}"
        assert "questions" in result
        assert "preserve_payload" in result
        assert len(result["questions"]) > 0


# =============================================================================
# VALIDATION QUESTIONS TESTS
# =============================================================================

class TestValidationQuestions:
    """Test the validation questions schema."""
    
    def test_all_critical_fields_have_questions(self):
        """All critical fields should have corresponding questions defined."""
        critical_fields = ["annual_income", "age"]
        
        for field_id in critical_fields:
            assert field_id in VALIDATION_QUESTIONS, f"Missing question for {field_id}"
            q = VALIDATION_QUESTIONS[field_id]
            assert "field_path" in q
            assert "question" in q
            assert "type" in q
    
    def test_question_types_are_valid(self):
        """Question types should be valid."""
        valid_types = ["number", "text", "boolean", "select"]
        
        for field_id, q in VALIDATION_QUESTIONS.items():
            assert q["type"] in valid_types, f"Invalid type for {field_id}: {q['type']}"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
