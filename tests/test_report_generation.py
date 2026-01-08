#!/usr/bin/env python3
"""
Local test script for financial report generation.

This script allows you to test the LLM report sections with sample client data
without going through the full document upload flow.

Usage:
    python test_report_generation.py [--no-llm]
    
Options:
    --no-llm    Skip actual LLM calls, just test the digest generation (faster)

Example:
    cd backend
    python tests/test_report_generation.py
"""

import os
import sys
import json
import argparse

# Ensure imports resolve
here = os.path.dirname(__file__)
root = os.path.dirname(here)
if root not in sys.path:
    sys.path.insert(0, root)

# Load env before importing app modules
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(root, ".env"))
except ImportError:
    # dotenv not installed, try to load from environment
    pass

# Import the report generation modules
from llm_sections import (
    GoalsStrategyRunner,
    ProtectionPlanRunner,
    DebtStrategyRunner,
    PortfolioRebalanceRunner,
    LLMClient,
    compute_goal_sip,
    compute_goal_priority,
    _get_return_assumptions,
    ASSUMED_RETURNS,
    ASSUMED_INFLATION_RATE,
)


# ============== SAMPLE CLIENT DATA ==============
# Modify this to test different scenarios

SAMPLE_CLIENT_FACTS = {
    "questionnaire_id": 9999,
    "personal": {
        "name": "Test Client",
        "age": 35,
        "dependents_count": 2,
    },
    "income": {
        "annualIncome": 1800000,      # Rs. 18 lakh/year
        "monthlyExpenses": 80000,      # Rs. 80k/month
        "monthlyEmi": 30000,           # Rs. 30k/month EMI
    },
    "insurance": {
        "lifeCover": 2500000,          # Rs. 25 lakh current cover
        "healthCover": 500000,         # Rs. 5 lakh health cover
    },
    "savings": {
        "savingsPercent": 25,
    },
    "goals": [
        {
            "name": "Retirement",
            "target_amount": 25000000,   # Rs. 2.5 crore
            "horizon_years": 25,
            "goal_importance": "essential",
            "goal_flexibility": "fixed",
            "risk_tolerance": "medium",
        },
        {
            "name": "Child's Education",
            "target_amount": 5000000,    # Rs. 50 lakh
            "horizon_years": 10,
            "goal_importance": "important",
            "goal_flexibility": "flexible",
            "risk_tolerance": "medium",
        },
        {
            "name": "Home Purchase",
            "target_amount": 8000000,    # Rs. 80 lakh
            "horizon_years": 5,
            "goal_importance": "important",
            "goal_flexibility": "fixed",
            "risk_tolerance": "low",
        },
    ],
    "bank": {
        "total_inflows": 1800000,
        "total_outflows": 1200000,
        "net_cashflow": 600000,
    },
    "portfolio": {
        "equity": 100,                  # 100% equity - to test rebalancing advice
        "debt": 0,
        "gold": 0,
        "realEstate": 0,
        "monthly_sip": 1500,           # Existing SIP commitment
    },
    "analysis": {
        "riskProfile": "Moderate",
        "advancedRisk": {
            "finalCategory": "Moderate",
            "recommendedEquityBand": {"min": 40, "max": 55},
            "recommendedEquityMid": 47.5,
        },
        "surplusBand": "Adequate",
        "insuranceGap": "Underinsured",
        "debtStress": "Moderate",
        "liquidity": "Adequate",
        "_diagnostics": {
            "emiPct": 20.0,
            "liquidityMonths": 4,
            "requiredLifeCover": 18000000,  # 10x income
        },
    },
}


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60 + "\n")


def test_assumptions_display():
    """Test that return assumptions are correctly calculated."""
    print_separator("Testing Return Assumptions")
    
    print(f"Global Inflation Rate: {ASSUMED_INFLATION_RATE * 100}%")
    print(f"\nAssumed Returns by Category:")
    for cat, rates in ASSUMED_RETURNS.items():
        print(f"  {cat}: {rates['label']}")
    
    # Test for different scenarios
    test_cases = [
        ("Aggressive", 25),   # Long horizon, aggressive
        ("Moderate", 10),     # Medium horizon
        ("Conservative", 3),  # Short horizon
    ]
    
    print(f"\nEffective Return Assumptions for Sample Goals:")
    for risk_cat, horizon in test_cases:
        assumptions = _get_return_assumptions(risk_cat, horizon)
        print(f"  {risk_cat} ({horizon} years): "
              f"{assumptions['expected_return_percent']}% expected, "
              f"{assumptions['inflation_percent']}% inflation, "
              f"{assumptions['real_return_percent']}% real")


def test_sip_calculations():
    """Test that SIP calculations work correctly."""
    print_separator("Testing SIP Calculations")
    
    for goal in SAMPLE_CLIENT_FACTS["goals"]:
        sip = compute_goal_sip(
            goal["target_amount"],
            goal["horizon_years"],
            goal.get("risk_tolerance", "medium")
        )
        print(f"Goal: {goal['name']}")
        print(f"  Target: Rs. {goal['target_amount']:,.0f}")
        print(f"  Horizon: {goal['horizon_years']} years")
        print(f"  Required SIP: Rs. {sip:,.0f}/month" if sip else "  SIP: Not calculable")
        print()


def test_priority_ranking():
    """Test goal priority ranking logic."""
    print_separator("Testing Goal Priority Ranking")
    
    goals_with_priority = []
    for goal in SAMPLE_CLIENT_FACTS["goals"]:
        # Simulate gap_exists (assume gap for testing)
        score, tier = compute_goal_priority(
            horizon_years=goal["horizon_years"],
            importance=goal.get("goal_importance", "important"),
            gap_exists=True,  # Assume gap exists
            shortfall_percent=30.0,  # Assume 30% shortfall
        )
        goals_with_priority.append({
            "name": goal["name"],
            "horizon": goal["horizon_years"],
            "importance": goal.get("goal_importance"),
            "score": score,
            "tier": tier,
        })
    
    # Sort by priority score (lower = higher priority)
    goals_with_priority.sort(key=lambda x: x["score"])
    
    print("Priority Order (1 = highest priority):")
    for i, g in enumerate(goals_with_priority, 1):
        print(f"  {i}. {g['name']} - Score: {g['score']}, Tier: {g['tier']}")
        print(f"     (Horizon: {g['horizon']} yrs, Importance: {g['importance']})")


def test_digest_generation(runner_class, runner_name: str):
    """Test digest generation for a specific runner."""
    print_separator(f"Testing {runner_name} Digest")
    
    # Create runner with dummy LLM client (won't actually call)
    class DummyLLM:
        model = "dummy"
    
    runner = runner_class(DummyLLM(), 9999, "/tmp/test")
    digest = runner.digest(SAMPLE_CLIENT_FACTS)
    
    if digest is None:
        print("Digest returned None (no data available)")
        return None
    
    print(json.dumps(digest, indent=2, ensure_ascii=False, default=str))
    return digest


def test_full_report_section(runner_class, runner_name: str, llm_client):
    """Test full report section generation with actual LLM call."""
    print_separator(f"Generating {runner_name} Section (LLM)")
    
    runner = runner_class(llm_client, 9999, os.path.join(root, "output", "test_sections"))
    
    try:
        result = runner.run(SAMPLE_CLIENT_FACTS)
        if result:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("No result returned")
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test financial report generation")
    parser.add_argument("--no-llm", action="store_true", 
                        help="Skip LLM calls, only test digest generation")
    parser.add_argument("--section", type=str, default="all",
                        choices=["all", "goals", "protection", "debt", "portfolio"],
                        help="Which section to test")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print(" FINANCIAL REPORT GENERATION - LOCAL TEST")
    print("=" * 60)
    
    # Show sample data summary
    print_separator("Sample Client Summary")
    income = SAMPLE_CLIENT_FACTS["income"]
    monthly_income = income["annualIncome"] / 12
    surplus = monthly_income - income["monthlyExpenses"] - income["monthlyEmi"]
    print(f"Monthly Income: Rs. {monthly_income:,.0f}")
    print(f"Monthly Expenses: Rs. {income['monthlyExpenses']:,.0f}")
    print(f"Monthly EMI: Rs. {income['monthlyEmi']:,.0f}")
    print(f"Available Surplus: Rs. {surplus:,.0f}/month")
    print(f"Existing SIPs: Rs. {SAMPLE_CLIENT_FACTS['portfolio'].get('monthly_sip', 0):,.0f}")
    print(f"Current Equity Allocation: {SAMPLE_CLIENT_FACTS['portfolio']['equity']}%")
    print(f"Recommended Equity Band: {SAMPLE_CLIENT_FACTS['analysis']['advancedRisk']['recommendedEquityBand']}")
    print(f"\nGoals: {len(SAMPLE_CLIENT_FACTS['goals'])}")
    for g in SAMPLE_CLIENT_FACTS["goals"]:
        print(f"  - {g['name']}: Rs. {g['target_amount']:,.0f} in {g['horizon_years']} years")
    
    # Test component functions
    test_assumptions_display()
    test_sip_calculations()
    test_priority_ranking()
    
    # Test digest generation (no LLM calls)
    runners = {
        "goals": (GoalsStrategyRunner, "Goals Strategy"),
        "protection": (ProtectionPlanRunner, "Protection Plan"),
        "debt": (DebtStrategyRunner, "Debt Strategy"),
        "portfolio": (PortfolioRebalanceRunner, "Portfolio Rebalance"),
    }
    
    if args.section == "all":
        for key, (runner_class, name) in runners.items():
            test_digest_generation(runner_class, name)
    else:
        runner_class, name = runners[args.section]
        test_digest_generation(runner_class, name)
    
    # Test full LLM generation if not skipped
    if not args.no_llm:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "test":
            print_separator("⚠️ LLM Testing Skipped")
            print("OPENAI_API_KEY not set or is 'test'. Set a valid API key to test LLM generation.")
            print("Run with --no-llm to skip LLM testing.")
        else:
            llm_client = LLMClient(api_key=api_key)
            
            if args.section == "all":
                for key, (runner_class, name) in runners.items():
                    test_full_report_section(runner_class, name, llm_client)
            else:
                runner_class, name = runners[args.section]
                test_full_report_section(runner_class, name, llm_client)
    
    print_separator("Test Complete")
    print("To modify test data, edit SAMPLE_CLIENT_FACTS at the top of this script.")
    print("\nUsage tips:")
    print("  python tests/test_report_generation.py --no-llm     # Test digests only (fast)")
    print("  python tests/test_report_generation.py --section goals  # Test specific section")
    print("  python tests/test_report_generation.py              # Full test with LLM calls")


if __name__ == "__main__":
    main()
