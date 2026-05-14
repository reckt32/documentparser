"""
Unit tests for financial_calculators.py

Tests cover:
- Spend Right (golden number, surplus %, status badge boundaries)
- Step-Up SIP (known values, edge cases)
- Retirement Gap (zero corpus, partial corpus, full coverage)
- Indian number formatting
"""

import sys
import os
import math
import unittest

# Ensure backend root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from financial_calculators import (
    compute_spend_right,
    compute_step_up_sip,
    compute_retirement_gap,
    format_indian_compact,
)


class TestSpendRight(unittest.TestCase):
    """Tests for the Spend Right / Golden Number calculator."""

    def test_saver_badge(self):
        """Surplus > 30% => Saver."""
        result = compute_spend_right(
            income=100_000,
            rent=15_000,
            basic_spends=15_000,
            comfort_spends=10_000,
        )
        # surplus = 100k - 30k - 10k = 60k => 60%
        self.assertEqual(result["status_badge"], "Saver")
        self.assertAlmostEqual(result["surplus_pct"], 60.0, places=1)
        self.assertEqual(result["golden_number"], 10_000)

    def test_balanced_badge(self):
        """Surplus between 10-30% => Balanced."""
        result = compute_spend_right(
            income=100_000,
            rent=30_000,
            basic_spends=25_000,
            comfort_spends=25_000,
        )
        # surplus = 100k - 55k - 25k = 20k => 20%
        self.assertEqual(result["status_badge"], "Balanced")
        self.assertAlmostEqual(result["surplus_pct"], 20.0, places=1)

    def test_spender_badge(self):
        """Surplus < 10% => Spender."""
        result = compute_spend_right(
            income=100_000,
            rent=40_000,
            basic_spends=30_000,
            comfort_spends=25_000,
        )
        # surplus = 100k - 70k - 25k = 5k => 5%
        self.assertEqual(result["status_badge"], "Spender")
        self.assertAlmostEqual(result["surplus_pct"], 5.0, places=1)

    def test_boundary_exactly_10_percent(self):
        """Surplus == 10% => Balanced (boundary inclusive)."""
        result = compute_spend_right(
            income=100_000,
            rent=40_000,
            basic_spends=20_000,
            comfort_spends=30_000,
        )
        # surplus = 100k - 60k - 30k = 10k => 10%
        self.assertEqual(result["status_badge"], "Balanced")

    def test_boundary_exactly_30_percent(self):
        """Surplus == 30% => Balanced (30% is not > 30%)."""
        result = compute_spend_right(
            income=100_000,
            rent=25_000,
            basic_spends=15_000,
            comfort_spends=30_000,
        )
        # surplus = 100k - 40k - 30k = 30k => 30%
        self.assertEqual(result["status_badge"], "Balanced")

    def test_boundary_30_point_1_percent(self):
        """Surplus = 30.1% => Saver."""
        result = compute_spend_right(
            income=100_000,
            rent=25_000,
            basic_spends=14_900,
            comfort_spends=30_000,
        )
        # surplus = 100k - 39.9k - 30k = 30.1k => 30.1%
        self.assertEqual(result["status_badge"], "Saver")

    def test_zero_income(self):
        """Zero income returns Spender with error."""
        result = compute_spend_right(income=0, rent=0, basic_spends=0, comfort_spends=0)
        self.assertEqual(result["status_badge"], "Spender")
        self.assertIn("error", result)

    def test_negative_surplus(self):
        """Expenses > Income => negative surplus => Spender."""
        result = compute_spend_right(
            income=50_000,
            rent=30_000,
            basic_spends=20_000,
            comfort_spends=15_000,
        )
        # surplus = 50k - 50k - 15k = -15k => -30%
        self.assertEqual(result["status_badge"], "Spender")
        self.assertLess(result["surplus"], 0)

    def test_golden_number_is_comfort_spends(self):
        """Golden number should equal comfort_spends."""
        result = compute_spend_right(
            income=150_000,
            rent=20_000,
            basic_spends=30_000,
            comfort_spends=42_500,
        )
        self.assertEqual(result["golden_number"], 42_500)


class TestStepUpSIP(unittest.TestCase):
    """Tests for step-up SIP calculation."""

    def test_basic_calculation(self):
        """Step-up SIP should return a positive number for valid inputs."""
        sip = compute_step_up_sip(target=5_00_00_000, years=25)
        self.assertIsNotNone(sip)
        self.assertGreater(sip, 0)
        # With 10% step-up and 10% return over 25 years,
        # starting SIP for ₹5 Cr should be roughly ₹3,000-8,000 range
        self.assertGreater(sip, 1_000)
        self.assertLess(sip, 50_000)

    def test_short_horizon(self):
        """Short horizon => higher starting SIP."""
        sip_5yr = compute_step_up_sip(target=1_00_00_000, years=5)
        sip_25yr = compute_step_up_sip(target=1_00_00_000, years=25)
        self.assertIsNotNone(sip_5yr)
        self.assertIsNotNone(sip_25yr)
        self.assertGreater(sip_5yr, sip_25yr)

    def test_zero_target(self):
        """Zero target => None."""
        self.assertIsNone(compute_step_up_sip(target=0, years=25))

    def test_zero_years(self):
        """Zero years => None."""
        self.assertIsNone(compute_step_up_sip(target=1_00_00_000, years=0))

    def test_negative_target(self):
        """Negative target => None."""
        self.assertIsNone(compute_step_up_sip(target=-100, years=10))

    def test_one_year(self):
        """1 year horizon: FV = SIP * annuity_12(r_monthly), no step-up needed."""
        target = 1_00_000
        sip = compute_step_up_sip(target=target, years=1)
        self.assertIsNotNone(sip)
        # Verify: SIP * annuity_12 ≈ target
        r_monthly = (1.10) ** (1/12) - 1
        annuity_12 = ((1 + r_monthly) ** 12 - 1) / r_monthly
        expected_sip = target / annuity_12
        self.assertAlmostEqual(sip, round(expected_sip, 2), delta=1.0)


class TestRetirementGap(unittest.TestCase):
    """Tests for retirement gap analysis."""

    def test_zero_existing_corpus(self):
        """With zero existing investments, gap equals target corpus."""
        result = compute_retirement_gap(
            monthly_expense=50_000,
            years_to_retire=25,
            existing_corpus=0,
            ongoing_sip=0,
        )
        self.assertGreater(result["target_corpus"], 0)
        self.assertEqual(result["fv_existing_corpus"], 0)
        self.assertEqual(result["fv_ongoing_sip"], 0)
        self.assertAlmostEqual(result["gap"], result["target_corpus"], places=0)
        self.assertGreater(result["required_step_up_sip"], 0)

    def test_target_corpus_formula(self):
        """Verify target corpus = (monthly * 1.07^years * 12) / 0.05."""
        result = compute_retirement_gap(
            monthly_expense=50_000,
            years_to_retire=25,
        )
        expected = (50_000 * (1.07 ** 25) * 12) / 0.05
        self.assertAlmostEqual(result["target_corpus"], expected, delta=1.0)

    def test_fv_existing_corpus(self):
        """FV of existing = corpus * 1.10^years."""
        result = compute_retirement_gap(
            monthly_expense=50_000,
            years_to_retire=20,
            existing_corpus=10_00_000,
            ongoing_sip=0,
        )
        expected_fv = 10_00_000 * (1.10 ** 20)
        self.assertAlmostEqual(result["fv_existing_corpus"], expected_fv, delta=1.0)

    def test_partial_coverage(self):
        """With some existing corpus + SIP, gap should be less than target."""
        result = compute_retirement_gap(
            monthly_expense=50_000,
            years_to_retire=25,
            existing_corpus=5_00_000,
            ongoing_sip=5_000,
        )
        self.assertGreater(result["gap"], 0)
        self.assertLess(result["gap"], result["target_corpus"])

    def test_full_coverage(self):
        """When existing resources exceed target, gap should be zero."""
        result = compute_retirement_gap(
            monthly_expense=10_000,
            years_to_retire=5,
            existing_corpus=10_00_00_000,  # 10 Cr existing - more than enough
            ongoing_sip=50_000,
        )
        self.assertEqual(result["gap"], 0)
        self.assertEqual(result["required_step_up_sip"], 0)

    def test_zero_monthly_expense(self):
        """Zero expense => error case."""
        result = compute_retirement_gap(monthly_expense=0, years_to_retire=25)
        self.assertIn("error", result)

    def test_assumptions_present(self):
        """Result should include assumptions dict."""
        result = compute_retirement_gap(
            monthly_expense=50_000,
            years_to_retire=25,
        )
        self.assertIn("assumptions", result)
        self.assertEqual(result["assumptions"]["inflation_pct"], 7.0)
        self.assertEqual(result["assumptions"]["return_pct"], 10.0)
        self.assertEqual(result["assumptions"]["step_up_pct"], 10.0)
        self.assertEqual(result["assumptions"]["withdrawal_rate_pct"], 5.0)


class TestIndianFormatting(unittest.TestCase):
    """Tests for Indian number formatting."""

    def test_crore(self):
        self.assertIn("Cr", format_indian_compact(1_50_00_000))

    def test_lakh(self):
        self.assertIn("L", format_indian_compact(5_50_000))

    def test_thousand(self):
        self.assertIn("K", format_indian_compact(5_000))

    def test_small_number(self):
        result = format_indian_compact(500)
        self.assertIn("500", result)

    def test_zero(self):
        self.assertEqual(format_indian_compact(0), "₹0")

    def test_negative(self):
        result = format_indian_compact(-5_00_000)
        self.assertTrue(result.startswith("-"))


if __name__ == "__main__":
    unittest.main()
