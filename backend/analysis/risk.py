"""Risk profiling: basic profile and the advanced multi-factor risk score."""

from backend.utils import _safe_float


def compute_risk_profile(age, tolerance, horizon):
    a = _safe_float(age, 0.0)
    # age points
    if a < 35:
        age_pts = 2
    elif a <= 50:
        age_pts = 1
    else:
        age_pts = 0
    # tolerance points
    tol_map = {"low": 0, "medium": 1, "med": 1, "high": 2}
    tol_pts = tol_map.get(str(tolerance).strip().lower(), 0)
    # horizon points
    hor_map = {"short": 0, "medium": 1, "med": 1, "long": 2}
    hor_pts = hor_map.get(str(horizon).strip().lower(), 0)
    score = age_pts + tol_pts + hor_pts  # 0-6
    if score <= 2:
        return "Conservative", score
    if score <= 4:
        return "Balanced", score
    return "Aggressive", score


# --- Advanced Risk Logic Engine (Phases 1–3) ---

_RISK_CATEGORIES = [
    "Ultra Conservative",
    "Conservative",
    "Moderate",
    "Growth",
    "Aggressive",
    "Very Aggressive",
]


_RISK_EQUITY_BANDS = {
    "Ultra Conservative": (0, 25),
    "Conservative": (25, 40),
    "Moderate": (40, 55),
    "Growth": (55, 70),
    "Aggressive": (70, 85),
    "Very Aggressive": (85, 95),
}


def _tenure_limit_category(years):
    if years is None:
        return "Moderate"
    try:
        y = float(years)
    except Exception:
        return "Moderate"
    if y < 3:
        return "Conservative"
    if 3 <= y < 5:
        return "Moderate"
    if 5 <= y < 7:
        return "Growth"
    if 7 <= y < 10:
        return "Aggressive"
    return "Very Aggressive"


def _score_to_category(score):
    if score < 1.8:
        return "Ultra Conservative"
    if score < 2.4:
        return "Conservative"
    if score < 3.2:
        return "Moderate"
    if score < 4.0:
        return "Growth"
    if score < 4.6:
        return "Aggressive"
    return "Very Aggressive"


def _bucket_loss_tolerance(percent):
    if percent is None:
        return 2
    p = max(0.0, float(percent))
    if p <= 5: return 1
    if p <= 10: return 2
    if p <= 15: return 3
    if p <= 25: return 4
    if p <= 35: return 5
    return 6


def _bucket_savings(percent):
    if percent is None:
        return 2
    p = max(0.0, float(percent))
    if p < 10: return 1
    if p < 20: return 2
    if p < 30: return 3
    if p < 40: return 4
    return 5


def _bucket_income_stability(label):
    if not label:
        return 3
    mapping = {
        "very unstable": 1,
        "unstable": 2,
        "average": 3,
        "stable": 4,
        "very stable": 5,
    }
    return mapping.get(label.strip().lower(), 3)


def _bucket_emergency_fund(months):
    if months is None:
        return 2
    m = max(0.0, float(months))
    if m < 1: return 1
    if m < 3: return 2
    if m < 6: return 3
    if m < 12: return 4
    return 5


def _bucket_behavior(label):
    if not label:
        return 3
    mapping = {
        "sell": 1,
        "reduce": 2,
        "hold": 3,
        "buy": 4,
        "aggressive buy": 5,
        "aggressive_buy": 5,
    }
    return mapping.get(label.strip().lower(), 3)


def _category_index(cat):
    try:
        return _RISK_CATEGORIES.index(cat)
    except Exception:
        return _RISK_CATEGORIES.index("Moderate")


def _index_to_category(idx):
    idx = max(0, min(len(_RISK_CATEGORIES) - 1, idx))
    return _RISK_CATEGORIES[idx]


def compute_advanced_risk(payload, age):
    risk = (payload.get("risk") or {})
    goals = (payload.get("goals") or {})
    lifestyle = (payload.get("savings") or {})
    investments = (payload.get("investments") or {})
    allocation = investments.get("allocation") or {}

    tenure_years = risk.get("goal_tenure_years") or risk.get("primary_horizon_years")
    if tenure_years is None:
        horizon_text = risk.get("primary_horizon")
        if isinstance(horizon_text, str):
            ht = horizon_text.lower()
            if ht.startswith("short"):
                tenure_years = 3
            elif ht.startswith("med"):
                tenure_years = 5
            else:
                tenure_years = 10
    try:
        tenure_years = float(tenure_years) if tenure_years not in (None, "") else None
    except Exception:
        tenure_years = None

    flexibility = risk.get("goal_flexibility") or goals.get("flexibility")
    importance = risk.get("goal_importance") or goals.get("importance")
    behavior = risk.get("behavior")
    loss_pct = risk.get("loss_tolerance_percent")
    try:
        loss_pct = float(loss_pct) if loss_pct not in (None, "") else None
    except Exception:
        loss_pct = None
    savings_percent = lifestyle.get("savingsPercent") or lifestyle.get("savings_percent")
    try:
        savings_percent = float(savings_percent) if savings_percent not in (None, "") else None
    except Exception:
        savings_percent = None
    income_stability = risk.get("income_stability")
    emergency_months = risk.get("emergency_fund_months")
    if emergency_months in (None, ""):
        ef_amt = payload.get("emergencyFundAmount")
        mexp = (payload.get("income") or {}).get("monthlyExpenses")
        try:
            ef_amt_f = float(ef_amt) if ef_amt not in (None, "") else None
            mexp_f = float(mexp) if mexp not in (None, "") else None
            if ef_amt_f and mexp_f and mexp_f > 0:
                emergency_months = ef_amt_f / mexp_f
        except Exception:
            emergency_months = None
    try:
        emergency_months = float(emergency_months) if emergency_months not in (None, "") else None
    except Exception:
        emergency_months = None

    tenure_limit = _tenure_limit_category(tenure_years)

    b_behavior = _bucket_behavior(behavior)
    b_loss = _bucket_loss_tolerance(loss_pct)
    b_savings = _bucket_savings(savings_percent)
    b_income = _bucket_income_stability(income_stability)
    b_emergency = _bucket_emergency_fund(emergency_months)

    score = (b_behavior * 0.25) + (b_loss * 0.25) + (b_savings * 0.20) + (b_income * 0.15) + (b_emergency * 0.15)
    appetite_category = _score_to_category(score)

    baseline_category = _index_to_category(min(_category_index(appetite_category), _category_index(tenure_limit)))
    adjustments = []

    def _goal_adj(value, mapping):
        if not value:
            return 0
        return mapping.get(str(value).strip().lower(), 0)

    flex_adj = _goal_adj(flexibility, {"critical": -1, "fixed": 0, "flexible": 1})
    imp_adj = _goal_adj(importance, {"essential": -1, "important": 0, "lifestyle": 1})
    net_goal_adj = flex_adj + imp_adj
    if net_goal_adj != 0:
        new_idx = _category_index(baseline_category) + net_goal_adj
        new_cat = _index_to_category(new_idx)
        if _category_index(new_cat) > _category_index(tenure_limit):
            if flex_adj == 1 and imp_adj == 1 and net_goal_adj == 2:
                new_cat = _index_to_category(_category_index(tenure_limit) + 1)
            else:
                new_cat = tenure_limit
        adjustments.append(f"GoalAdjustment: {baseline_category} -> {new_cat} (net {net_goal_adj})")
        baseline_category = new_cat

    equity_pct = None
    try:
        eq_raw = allocation.get("equity")
        equity_pct = float(eq_raw) if eq_raw not in (None, "") else None
    except Exception:
        equity_pct = None

    # The recommended band is driven by the client's risk profile (answers +
    # tenure + goal adjustments), NOT by their current allocation. Current
    # equity only measures the rebalancing distance — letting it mutate the
    # category made the target chase the holdings and pinned every profile
    # with a drifted portfolio to Moderate (the "47% for all cases" bug).
    if equity_pct is not None:
        band = _RISK_EQUITY_BANDS.get(baseline_category)
        if band:
            band_min, band_max = band
            nearest_edge_dist = 0
            if equity_pct < band_min:
                nearest_edge_dist = band_min - equity_pct
            elif equity_pct > band_max:
                nearest_edge_dist = equity_pct - band_max
            if nearest_edge_dist > 0:
                adjustments.append(
                    f"AllocationDrift: current equity {equity_pct:.1f}% is {nearest_edge_dist:.1f}pp outside the {band_min}-{band_max}% band (rebalance, target unchanged)"
                )

    final_category = baseline_category
    if _category_index(final_category) > _category_index(tenure_limit):
        final_category = tenure_limit

    final_band = _RISK_EQUITY_BANDS.get(final_category)
    band_mid = None
    if final_band:
        band_mid = round((final_band[0] + final_band[1]) / 2, 1)

    reasoning_parts = [
        f"Score {score:.2f} -> Appetite {appetite_category}",
        f"Tenure limit {tenure_limit}",
        f"Baseline after adjustments {baseline_category}",
    ]
    if adjustments:
        reasoning_parts.append("Adjustments: " + "; ".join(adjustments))
    reasoning_parts.append(f"Final {final_category}")
    reasoning_text = " | ".join(reasoning_parts)

    return {
        "score": round(score, 2),
        "tenureLimitCategory": tenure_limit,
        "appetiteCategory": appetite_category,
        "baselineCategory": baseline_category,
        "finalCategory": final_category,
        "adjustmentsApplied": adjustments,
        "recommendedEquityBand": {
            "min": final_band[0] if final_band else None,
            "max": final_band[1] if final_band else None,
        } if final_band else None,
        "recommendedEquityMid": band_mid,
        "reasoningText": reasoning_text,
        "raw": {
            "tenureYears": tenure_years,
            "flexibility": flexibility,
            "importance": importance,
            "behavior": behavior,
            "lossTolerancePercent": loss_pct,
            "savingsPercent": savings_percent,
            "incomeStability": income_stability,
            "emergencyFundMonths": emergency_months,
            "equityAllocationPercent": equity_pct,
        },
    }
