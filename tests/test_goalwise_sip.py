#!/usr/bin/env python3
"""Test script to verify Goal-wise SIP table and Reality Check changes in PDF generation."""
import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import generate_financial_plan_pdf, compute_term_insurance_need
import fitz  # PyMuPDF - to read PDF text

print("=" * 70)
print(" TESTING GOAL-WISE SIP TABLE AND REALITY CHECK CHANGES")
print("=" * 70)

# Test 1: Verify life cover formula (19.5x)
print("\n--- Test 1: Life Cover Formula (15x * 1.3 = 19.5x) ---")
monthly_income = 30000  # 30k/month
expected_cover = 30000 * 12 * 15 * 1.3  # 3.6L * 19.5 = 70.2L
actual_cover = compute_term_insurance_need(30, monthly_income)
print(f"Monthly Income: Rs. {monthly_income:,}")
print(f"Expected Cover (19.5x annual): Rs. {expected_cover:,.0f}")
print(f"Actual Cover from function: Rs. {actual_cover:,.0f}")
if abs(actual_cover - expected_cover) < 100:
    print("[PASS] Life cover formula is correct!")
else:
    print(f"[FAIL] Expected {expected_cover:,.0f} but got {actual_cover:,.0f}")

# Test 2: Generate a PDF and check content
print("\n--- Test 2: PDF Content Verification ---")

# Sample questionnaire data - using values from sample_docs bank statement
# Bank statement shows: Credits=922254, Debits=467588.70
sample_q = {
    "id": 9999,
    "personal_info": {"name": "Test User", "age": 32},
    "family_info": {"spouse": None, "children": [], "dependents": []},
    "goals": {
        "items": [
            {"name": "Retirement Corpus", "target_amount": 50000000, "horizon_years": 25},
            {"name": "Lifestyle", "target_amount": 500000, "horizon_years": 5},
            {"name": "Wealth Creation", "target_amount": 10000000, "horizon_years": 15},
        ]
    },
    "lifestyle": {
        # From sample_docs bank statement: Credits=922254 (annual income)
        "annual_income": 922254,
        # From sample_docs bank statement: Debits=467588.70 / 12 = 38965.72
        "monthly_expenses": 38966,
        "monthly_emi": 0,
    },
    "insurance": {
        "life_cover": 0,
        "health_cover": 2500000,
    },
}

sample_analysis = {
    "advancedRisk": {
        "finalCategory": "Moderate",
        "recommendedEquityBand": {"min": 40, "max": 55},
    },
    "recommendations": [],
    "ihs": {"band": "Average"},
}

# Generate PDF to temp file
with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
    output_path = f.name

try:
    generate_financial_plan_pdf(sample_q, sample_analysis, output_path)
    print(f"PDF generated at: {output_path}")
    
    # Read PDF text
    doc = fitz.open(output_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    
    # Check for key content
    checks = [
        ("15x annual income", "Life cover basis text"),
        ("Goal-wise SIP Allocation", "Goal-wise SIP table header"),
        ("Required SIP", "SIP table column"),
        ("Allocated SIP", "SIP table column"),
        ("Current SIP", "Existing SIP context"),
        ("TOTAL Investing", "Total investing row"),
        ("Insurance Provision", "Insurance provision row"),
    ]
    
    print("\nContent Checks:")
    all_passed = True
    for search_text, description in checks:
        if search_text.lower() in full_text.lower():
            print(f"  [PASS] {description}: Found '{search_text}'")
        else:
            print(f"  [FAIL] {description}: Missing '{search_text}'")
            all_passed = False
    
    if all_passed:
        print("\n[SUCCESS] All content checks passed!")
    else:
        print("\n[WARNING] Some content is missing. Check PDF generation code.")
        print("\n--- Relevant PDF Text Snippet (Page 6) ---")
        if len(doc) >= 6:
            doc = fitz.open(output_path)
            page6_text = doc[5].get_text() if len(doc) > 5 else "Page 6 not found"
            doc.close()
            print(page6_text[:2000])
        
finally:
    # Clean up
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"\nCleaned up temp file: {output_path}")

print("\n" + "=" * 70)
print(" TEST COMPLETE")
print("=" * 70)
