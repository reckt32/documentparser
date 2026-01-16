import os
import io
import json
import time
import math
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Public entrypoint:
# run_report_sections(questionnaire_id: int, client_facts: dict, model: Optional[str]) -> Dict[str, dict]
#
# Expectations:
# - client_facts is a compact JSON with:
#   {
#     "questionnaire_id": int,
#     "personal": {"name": str, "age": number, "dependents_count": int},
#     "income": {"annualIncome": number, "monthlyExpenses": number, "monthlyEmi": number},
#     "insurance": {"lifeCover": number, "healthCover": number},
#     "savings": {"savingsPercent": number},
#     "goals": [{"name": str, "target_amount": number, "horizon_years": number}, ...],
#     "bank": {"total_inflows": number, "total_outflows": number, "net_cashflow": number},
#     "portfolio": {...}  # optional
#     "analysis": {
#        "riskProfile": str,
#        "advancedRisk": {...},
#        "surplusBand": str,
#        "insuranceGap": str,
#        "debtStress": str,
#        "liquidity": str,
#        "ihs": {"band": str, "score": number, "breakdown": {...}},
#        "_diagnostics": {"emiPct": number, "liquidityMonths": number, "requiredLifeCover": number}
#     }
#   }


OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
SECTIONS_DIR_NAME = "sections"
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# ------------------------ Financial Assumptions (Configurable) ------------------------ #
# These assumptions are used for SIP calculations and are displayed in reports
ASSUMED_INFLATION_RATE = 0.06  # 6% annual inflation
ASSUMED_RETURNS = {
    "aggressive": {"annual": 0.14, "monthly": 0.011, "label": "14% p.a. (Aggressive Equity)"},
    "growth": {"annual": 0.114, "monthly": 0.009, "label": "11.4% p.a. (Growth/Balanced)"},
    "moderate": {"annual": 0.114, "monthly": 0.009, "label": "11.4% p.a. (Moderate)"},
    "conservative": {"annual": 0.074, "monthly": 0.006, "label": "7.4% p.a. (Conservative/Debt)"},
}
WITHDRAWAL_RATE_RETIREMENT = 0.07  # 7% safe withdrawal rate for retirement corpus


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _now_ts() -> int:
    return int(time.time())


def _coerce_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        sx = str(x).strip()
        if sx == "":
            return default
        # strip commas and currency symbols
        sx = sx.replace(",", "")
        for sym in ["₹", "Rs.", "Rs", "$"]:
            sx = sx.replace(sym, "")
        if sx.endswith("-"):
            sx = "-" + sx[:-1]
        return float(sx)
    except Exception:
        return default


# ------------------------ LLM Client Wrapper ------------------------ #

class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.2, request_timeout: int = 60):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model = model or DEFAULT_MODEL
        self.temperature = temperature
        self.request_timeout = request_timeout
        self._client = OpenAI(api_key=self.api_key)

    def complete_json(self, system: str, user: str, retry_on_schema_error: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        usage = {}
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=self.temperature,
            )
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
            txt = resp.choices[0].message.content
            data = json.loads(txt)
            return data, usage
        except Exception as e:
            if retry_on_schema_error:
                # One retry with a stricter reminder
                try:
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user + "\n\nCRITICAL: Return STRICT valid JSON only. Do not include any text outside JSON."},
                        ],
                        response_format={"type": "json_object"},
                        temperature=self.temperature,
                    )
                    usage = {
                        "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                        "total_tokens": getattr(resp.usage, "total_tokens", None),
                    }
                    txt = resp.choices[0].message.content
                    data = json.loads(txt)
                    return data, usage
                except Exception as e2:
                    raise e2
            raise e


# ------------------------ Base Section Runner ------------------------ #

class SectionRunner:
    name: str = "base"

    def __init__(self, llm: LLMClient, qid: int, base_dir: str):
        self.llm = llm
        self.qid = qid
        self.base_dir = base_dir

    def digest(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        # Must be implemented by subclasses
        raise NotImplementedError

    def prompt(self, digest: Dict[str, Any]) -> Tuple[str, str]:
        # Returns (system, user)
        raise NotImplementedError

    def validate(self, obj: Dict[str, Any]) -> bool:
        # Basic schema: title: str, bullets: list[str], paragraphs: list[str], actions: list[str]
        if not isinstance(obj, dict):
            return False
        if "title" not in obj:
            return False
        for k in ["bullets", "paragraphs", "actions"]:
            if k in obj and not isinstance(obj[k], list):
                return False
        return True

    def run(self, facts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        dg = self.digest(facts)
        if dg is None:
            return None
        dg_str = _json_dumps(dg)
        dg_hash = _sha256_text(dg_str)
        sec_dir = os.path.join(self.base_dir, self.name)
        _ensure_dir(sec_dir)
        out_path = os.path.join(sec_dir, f"{self.name}.json")

        # Cache check
        if os.path.isfile(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("_meta", {}).get("facts_hash") == dg_hash:
                    return cached.get("content")
            except Exception:
                pass

        system, user = self.prompt(dg)
        data, usage = self.llm.complete_json(system, user, retry_on_schema_error=True)
        if not self.validate(data):
            raise RuntimeError(f"Schema validation failed for section {self.name}")

        payload = {
            "_meta": {
                "questionnaire_id": self.qid,
                "section": self.name,
                "model": self.llm.model,
                "ts": _now_ts(),
                "facts_hash": dg_hash,
                "prompt": {
                    "system": system,
                    "user": user[:4000],  # avoid huge files
                },
                "usage": usage,
            },
            "content": data,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return data


# ------------------------ Helpers for Deterministic Goal Math ------------------------ #

def _choose_monthly_rate(final_category: Optional[str], horizon_years: Optional[float]) -> float:
    """Select monthly expected return based on risk category and horizon."""
    cat = (final_category or "").lower()
    h = float(horizon_years) if horizon_years not in (None, "") else None
    
    # Use constants for rate selection
    if cat in ["very aggressive", "aggressive"]:
        base = ASSUMED_RETURNS["aggressive"]["monthly"]
    elif cat in ["growth", "moderate"]:
        base = ASSUMED_RETURNS["growth"]["monthly"]
    else:
        base = ASSUMED_RETURNS["conservative"]["monthly"]
    
    # Short horizons use more conservative rates
    if h is not None:
        if h <= 3:
            base = min(base, ASSUMED_RETURNS["conservative"]["monthly"])
        elif h <= 7:
            base = min(base, ASSUMED_RETURNS["growth"]["monthly"])
    return base


def _get_return_assumptions(final_category: Optional[str], horizon_years: Optional[float]) -> Dict[str, Any]:
    """Get return assumptions for display in reports."""
    cat = (final_category or "").lower()
    h = float(horizon_years) if horizon_years not in (None, "") else None
    
    if cat in ["very aggressive", "aggressive"]:
        ret = ASSUMED_RETURNS["aggressive"]
    elif cat in ["growth", "moderate"]:
        ret = ASSUMED_RETURNS["growth"]
    else:
        ret = ASSUMED_RETURNS["conservative"]
    
    # Adjust label for short horizon
    effective_annual = ret["annual"]
    if h is not None and h <= 3:
        effective_annual = ASSUMED_RETURNS["conservative"]["annual"]
    elif h is not None and h <= 7:
        effective_annual = min(ret["annual"], ASSUMED_RETURNS["growth"]["annual"])
    
    return {
        "expected_return_percent": round(effective_annual * 100, 1),
        "inflation_percent": round(ASSUMED_INFLATION_RATE * 100, 1),
        "real_return_percent": round((effective_annual - ASSUMED_INFLATION_RATE) * 100, 1),
    }


def compute_goal_priority(
    horizon_years: Optional[float],
    importance: Optional[str],
    gap_exists: bool,
    shortfall_percent: float = 0
) -> Tuple[int, str]:
    """
    Compute goal priority score and tier.
    Lower score = higher priority.
    
    Returns: (priority_score, priority_tier)
    Tiers: "Immediate" (<=3 yrs), "Short-term" (3-7 yrs), "Long-term" (>7 yrs)
    """
    base_score = 100
    
    # Importance factor (essential goals get highest priority)
    imp = str(importance or "").strip().lower()
    if imp == "essential":
        base_score -= 40
    elif imp == "important":
        base_score -= 20
    # lifestyle goals stay at base
    
    # Horizon urgency (shorter horizon = higher priority)
    tier = "Long-term"
    if horizon_years:
        h = float(horizon_years)
        if h <= 3:
            base_score -= 30
            tier = "Immediate"
        elif h <= 5:
            base_score -= 20
            tier = "Short-term"
        elif h <= 7:
            base_score -= 10
            tier = "Short-term"
    
    # Gap factor: achievable goals get slight priority boost
    # But severely underfunded goals (>50% shortfall) need attention too
    if not gap_exists:
        base_score -= 10  # Achievable goals get priority
    elif shortfall_percent > 50:
        base_score -= 5  # Severely underfunded needs attention
    
    return (base_score, tier)


def compute_goal_sip(target_amount: float, horizon_years: Optional[float], final_category: Optional[str]) -> Optional[float]:
    try:
        target = _coerce_float(target_amount, 0.0)
        if target <= 0:
            return None
        n_years = float(horizon_years) if horizon_years not in (None, "") else None
        if n_years is None or n_years <= 0:
            return None
        n = int(round(n_years * 12))
        r = _choose_monthly_rate(final_category, n_years)
        denom = (math.pow(1.0 + r, n) - 1.0)
        if denom <= 0:
            return None
        sip = target * r / denom
        return round(sip, 2)
    except Exception:
        return None


def compute_realistic_target(monthly_sip: float, horizon_years: Optional[float], final_category: Optional[str]) -> Optional[float]:
    """Calculate what target amount is achievable with available monthly SIP."""
    try:
        sip = _coerce_float(monthly_sip, 0.0)
        if sip <= 0:
            return None
        n_years = float(horizon_years) if horizon_years not in (None, "") else None
        if n_years is None or n_years <= 0:
            return None
        n = int(round(n_years * 12))
        r = _choose_monthly_rate(final_category, n_years)
        # Future Value of SIP: FV = P * [((1+r)^n - 1) / r]
        fv = sip * ((math.pow(1.0 + r, n) - 1.0) / r)
        return round(fv, 2)
    except Exception:
        return None


def compute_required_horizon(target_amount: float, monthly_sip: float, final_category: Optional[str]) -> Optional[float]:
    """Calculate years needed to reach target with given monthly SIP using iterative approximation."""
    try:
        target = _coerce_float(target_amount, 0.0)
        sip = _coerce_float(monthly_sip, 0.0)
        if target <= 0 or sip <= 0:
            return None
        
        # Binary search for the required horizon (in years)
        low, high = 0.5, 50.0  # Search between 6 months and 50 years
        
        for _ in range(50):  # Max iterations
            mid = (low + high) / 2.0
            achievable = compute_realistic_target(sip, mid, final_category)
            if achievable is None:
                return None
            
            if abs(achievable - target) < target * 0.01:  # Within 1% tolerance
                return round(mid, 1)
            
            if achievable < target:
                low = mid
            else:
                high = mid
        
        return round((low + high) / 2.0, 1)
    except Exception:
        return None

def compute_goal_risk_category(
    horizon_years: Optional[float],
    risk_tolerance: Optional[str],
    goal_importance: Optional[str],
    goal_flexibility: Optional[str],
    behavior: Optional[str] = None
) -> str:
    """
    Compute risk category for a single goal based on its specific parameters.
    Returns one of: 'Conservative', 'Moderate', 'Growth', 'Aggressive'
    """
    # Base score from horizon (0-3)
    horizon_score = 0
    if horizon_years is not None:
        try:
            h = float(horizon_years)
            if h < 3:
                horizon_score = 0
            elif h < 5:
                horizon_score = 1
            elif h < 7:
                horizon_score = 2
            else:
                horizon_score = 3
        except (ValueError, TypeError):
            pass
    
    # Tolerance score (0-2)
    tol_map = {"low": 0, "medium": 1, "med": 1, "high": 2}
    tol_score = tol_map.get(str(risk_tolerance).strip().lower(), 1)
    
    # Importance adjustment (-1 to +1)
    imp_map = {"essential": -1, "important": 0, "lifestyle": 1}
    imp_adj = imp_map.get(str(goal_importance).strip().lower(), 0)
    
    # Flexibility adjustment (-1 to +1)
    flex_map = {"critical": -1, "fixed": 0, "flexible": 1}
    flex_adj = flex_map.get(str(goal_flexibility).strip().lower(), 0)
    
    # Behavior adjustment (-1 to +1)
    behavior_map = {"sell": -1, "reduce": -1, "hold": 0, "buy": 1, "aggressive buy": 1, "aggressive_buy": 1}
    behavior_adj = behavior_map.get(str(behavior).strip().lower(), 0) if behavior else 0
    
    # Total score: horizon (0-3) + tolerance (0-2) + adjustments (-3 to +3) = range -3 to 8
    total = horizon_score + tol_score + imp_adj + flex_adj + behavior_adj
    
    # Map to category
    if total <= 1:
        return "Conservative"
    elif total <= 3:
        return "Moderate"
    elif total <= 5:
        return "Growth"
    else:
        return "Aggressive"


# ------------------------ Priority Allocation Engine ------------------------ #

def generate_bridge_recommendations(shortfall: float, current_surplus: float) -> List[Dict]:
    """
    Generate specific recommendations to bridge the goal funding gap.
    Called when user cannot afford all their goals with current surplus.
    """
    if shortfall <= 0:
        return []
    
    recs = []
    
    # Option 1: Reduce expenses
    expense_reduction = shortfall
    recs.append({
        "option": "Reduce Expenses",
        "amount": round(expense_reduction),
        "action": f"Reduce monthly expenses by Rs. {expense_reduction:,.0f} to fully fund all goals"
    })
    
    # Option 2: Increase income (need ~30% more gross income due to taxes/savings rate)
    income_increase = shortfall * 1.3
    recs.append({
        "option": "Increase Income",
        "amount": round(income_increase),
        "action": f"Increase monthly income by Rs. {income_increase:,.0f} (assuming current savings rate)"
    })
    
    # Option 3: Accept partial achievement and increase over time
    if current_surplus > 0:
        achievable_pct = (current_surplus / (current_surplus + shortfall)) * 100
        recs.append({
            "option": "Start Now, Increase Later",
            "amount": None,
            "action": f"Start SIPs now at Rs. {current_surplus:,.0f}/month ({achievable_pct:.0f}% of goals) and increase contributions annually"
        })
    
    # Option 4: Extend timelines
    recs.append({
        "option": "Extend Goal Timelines",
        "amount": None,
        "action": "Extend goal target dates to reduce monthly SIP requirement"
    })
    
    return recs


class PriorityAllocationEngine:
    """
    Allocates available monthly surplus following priority order:
    
    Priority 1: Term Insurance Premium (if gap exists & has dependents)
    Priority 2: Health Insurance Premium (if gap exists)
    Priority 3+: Goals (remaining surplus distributed by priority rank)
    
    This implements the framework:
    "First term insurance → then health → then goals → if shortfall, show % achievable"
    """
    
    @staticmethod
    def estimate_term_premium(cover_gap: float, age: int = 35) -> float:
        """Estimate yearly term insurance premium based on cover gap and age."""
        if cover_gap <= 0:
            return 0.0
        # Rough estimate: Rs. 500-1200 per lakh depending on age
        rate_per_lakh = 500 if age < 35 else (700 if age < 45 else 1200)
        return (cover_gap / 100000) * rate_per_lakh
    
    @staticmethod
    def estimate_health_premium(cover_gap: float, age: int = 35) -> float:
        """Estimate yearly health insurance premium based on cover gap and age."""
        if cover_gap <= 0:
            return 0.0
        # Rough estimate: Rs. 2500-5000 per lakh for family floater
        rate_per_lakh = 2500 if age < 35 else (3500 if age < 45 else 5000)
        return (cover_gap / 100000) * rate_per_lakh
    
    @staticmethod
    def compute_allocation(
        monthly_surplus: float,
        term_insurance_gap: float,
        health_insurance_gap: float,
        goals: List[Dict],
        age: int = 35,
        has_dependents: bool = True
    ) -> Dict[str, Any]:
        """
        Compute priority-based allocation of monthly surplus.
        
        Args:
            monthly_surplus: Available monthly savings after expenses/EMI
            term_insurance_gap: Gap in term cover (Rs.)
            health_insurance_gap: Gap in health cover (Rs.)
            goals: List of goal dicts with 'ideal_sip', 'name', 'priority_rank'
            age: Client age for premium estimation
            has_dependents: Whether client has financial dependents
        
        Returns:
            Dict with priority breakdown, achievement %, and bridge recommendations
        """
        priority_items = []
        allocated_so_far = 0.0
        
        # Priority 1: Term Insurance (if has dependents and gap exists)
        term_premium_yearly = PriorityAllocationEngine.estimate_term_premium(term_insurance_gap, age)
        term_premium_monthly = term_premium_yearly / 12 if has_dependents and term_insurance_gap > 0 else 0
        
        term_status = "Adequate"
        if term_premium_monthly > 0:
            if monthly_surplus >= term_premium_monthly:
                term_status = "Fundable"
                allocated_so_far += term_premium_monthly
            else:
                term_status = "Critical Gap"
        elif not has_dependents:
            term_status = "Not Required"
            
        priority_items.append({
            "priority": 1,
            "name": "Term Insurance",
            "monthly_amount": round(term_premium_monthly, 0),
            "yearly_amount": round(term_premium_yearly, 0),
            "status": term_status,
            "note": f"Cover gap: Rs. {term_insurance_gap:,.0f}" if term_insurance_gap > 0 else "Adequate coverage"
        })
        
        # Priority 2: Health Insurance
        health_premium_yearly = PriorityAllocationEngine.estimate_health_premium(health_insurance_gap, age)
        health_premium_monthly = health_premium_yearly / 12 if health_insurance_gap > 0 else 0
        
        remaining_after_term = max(0, monthly_surplus - allocated_so_far)
        health_status = "Adequate"
        if health_premium_monthly > 0:
            if remaining_after_term >= health_premium_monthly:
                health_status = "Fundable"
                allocated_so_far += health_premium_monthly
            else:
                health_status = "Gap Exists"
                
        priority_items.append({
            "priority": 2,
            "name": "Health Insurance",
            "monthly_amount": round(health_premium_monthly, 0),
            "yearly_amount": round(health_premium_yearly, 0),
            "status": health_status,
            "note": f"Cover gap: Rs. {health_insurance_gap:,.0f}" if health_insurance_gap > 0 else "Adequate coverage"
        })
        
        # Remaining for goals
        remaining_for_goals = max(0, monthly_surplus - allocated_so_far)
        
        # Calculate total ideal SIP needed for all goals
        total_ideal_sip = sum(_coerce_float(g.get("ideal_sip"), 0.0) for g in goals)
        
        # Goal achievement percentage
        if total_ideal_sip > 0:
            achievement_pct = min(100.0, (remaining_for_goals / total_ideal_sip) * 100)
        else:
            achievement_pct = 100.0  # No goals = 100% achieved
        
        # Shortfall amount
        shortfall = max(0, total_ideal_sip - remaining_for_goals)
        
        # Add goals as priority 3
        goal_status = "Full" if achievement_pct >= 100 else ("Partial" if achievement_pct > 0 else "Cannot Fund")
        priority_items.append({
            "priority": 3,
            "name": "Goal SIPs",
            "monthly_amount": round(remaining_for_goals, 0),
            "ideal_amount": round(total_ideal_sip, 0),
            "status": goal_status,
            "note": f"Can fund {achievement_pct:.0f}% of goal requirements"
        })
        
        # Generate bridge recommendations if shortfall exists
        bridge_recs = generate_bridge_recommendations(shortfall, remaining_for_goals) if shortfall > 0 else []
        
        return {
            "priority_breakdown": priority_items,
            "monthly_surplus": round(monthly_surplus, 0),
            "allocated_to_insurance": round(term_premium_monthly + health_premium_monthly, 0),
            "remaining_for_goals": round(remaining_for_goals, 0),
            "total_ideal_sip_needed": round(total_ideal_sip, 0),
            "goal_achievement_percent": round(achievement_pct, 1),
            "savings_shortfall": round(shortfall, 0) if shortfall > 0 else 0,
            "bridge_recommendations": bridge_recs,
            "summary": f"With Rs. {monthly_surplus:,.0f}/month surplus: Insurance needs Rs. {allocated_so_far:,.0f}, leaving Rs. {remaining_for_goals:,.0f} for goals ({achievement_pct:.0f}% of requirement)."
        }


# ------------------------ Concrete Section Runners ------------------------ #

class FlagsExplainerRunner(SectionRunner):
    name = "flags_explainer"

    def digest(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        analysis = facts.get("analysis") or {}
        flags = analysis.get("flags") or facts.get("flags") or []
        kpis = {
            "surplusBand": analysis.get("surplusBand"),
            "insuranceGap": analysis.get("insuranceGap"),
            "debtStress": analysis.get("debtStress"),
            "liquidity": analysis.get("liquidity"),
            "ihsBand": ((analysis.get("ihs") or {}).get("band")),
            "emiPct": ((analysis.get("_diagnostics") or {}).get("emiPct")),
            "liquidityMonths": ((analysis.get("_diagnostics") or {}).get("liquidityMonths")),
        }
        return {
            "flags": flags[:10],
            "kpis": kpis,
            "client": {
                "age": (facts.get("personal") or {}).get("age"),
                "dependents": (facts.get("personal") or {}).get("dependents_count"),
            },
        }

    def prompt(self, digest: Dict[str, Any]) -> Tuple[str, str]:
        system = (
            "You are a senior financial planner writing for Indian retail clients. "
            "IMPORTANT: All monetary values MUST be in Indian Rupees (₹ or Rs.). NEVER use dollars ($) or any other currency. "
            "Be precise, neutral, and practical. No guarantees or forward-looking promises. "
            "Return valid JSON with keys: title (string), bullets (array of short points), paragraphs (2-3 short paragraphs), actions (checklist items)."
        )
        user = (
            "Section: Flags and Actions\n"
            "FactsDigest JSON (<=2KB):\n"
            f"{_json_dumps(digest)}\n\n"
            "Constraints:\n"
            "- Max 6 bullets, each <= 20 words.\n"
            "- 1-2 paragraphs, each <= 80 words.\n"
            "- Max 5 actions, each actionable and specific.\n"
        )
        return system, user


class ProtectionPlanRunner(SectionRunner):
    name = "protection_plan"

    def digest(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        analysis = facts.get("analysis") or {}
        personal = facts.get("personal") or {}
        insurance = facts.get("insurance") or {}
        income = facts.get("income") or {}
        return {
            "age": personal.get("age"),
            "dependents": personal.get("dependents_count"),
            "insuranceGap": analysis.get("insuranceGap"),
            "lifeCover": insurance.get("lifeCover"),
            "healthCover": insurance.get("healthCover"),
            "annualIncome": income.get("annualIncome"),
            "requiredLifeCover": ((analysis.get("_diagnostics") or {}).get("requiredLifeCover")),
        }

    def prompt(self, digest: Dict[str, Any]) -> Tuple[str, str]:
        annual_income = _coerce_float(digest.get("annualIncome"), 0.0)
        required_cover = _coerce_float(digest.get("requiredLifeCover"), annual_income * 10)
        current_life = _coerce_float(digest.get("lifeCover"), 0.0)
        current_health = _coerce_float(digest.get("healthCover"), 0.0)
        age = _coerce_float(digest.get("age"), 35)
        
        # Estimate term premium (rough: Rs. 500-800 per lakh for 30-40 age group)
        cover_gap = max(0, required_cover - current_life)
        estimated_premium = (cover_gap / 100000) * (500 if age < 35 else 700 if age < 45 else 1200)
        
        system = (
            "You are a senior financial planner. Draft a protection (life/health) section for Indian clients. "
            "IMPORTANT: All monetary values MUST be in Indian Rupees (₹ or Rs.). NEVER use dollars ($) or any other currency. "
            "Provide SPECIFIC coverage amounts and estimates. Avoid naming specific insurers but be specific about amounts."
        )
        user = (
            "Section: Protection Plan\n"
            f"FactsDigest:\n{_json_dumps(digest)}\n\n"
            "INSTRUCTIONS:\n"
            f"1. Life Insurance Gap: Required Rs.{required_cover:,.0f}, Current Rs.{current_life:,.0f}, Gap Rs.{cover_gap:,.0f}\n"
            f"   - If gap exists, recommend term insurance with specific cover amount\n"
            f"   - Estimated premium: approx Rs.{estimated_premium:,.0f}/year (varies by age, health, insurer)\n"
            f"2. Health Insurance: Current Rs.{current_health:,.0f}\n"
            "   - Recommend Rs.10-15 lakh family floater for adequate coverage\n"
            "   - If current < Rs.5 lakh, flag as critical gap\n"
            "3. Priority Order: Term life first if dependents exist, then health cover\n\n"
            "Output JSON keys: title, bullets, paragraphs, actions.\n"
            "Length limits: <=6 bullets, <=2 paragraphs, <=5 actions."
        )
        return system, user


class CashflowRunner(SectionRunner):
    name = "cashflow"

    def digest(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        analysis = facts.get("analysis") or {}
        bank = facts.get("bank") or {}
        return {
            "surplusBand": analysis.get("surplusBand"),
            "bank": {
                "total_inflows": bank.get("total_inflows"),
                "total_outflows": bank.get("total_outflows"),
                "net_cashflow": bank.get("net_cashflow"),
            }
        }

    def prompt(self, digest: Dict[str, Any]) -> Tuple[str, str]:
        system = (
            "You are a financial planner for Indian clients. Explain cashflows clearly, with practical advice to improve savings. "
            "IMPORTANT: All monetary values MUST be in Indian Rupees (₹ or Rs.). NEVER use dollars ($) or any other currency."
        )
        user = (
            "Section: Cashflow Overview\n"
            f"FactsDigest:\n{_json_dumps(digest)}\n\n"
            "Return JSON: title, bullets, paragraphs, actions. Keep concise and actionable."
        )
        return system, user


class DebtStrategyRunner(SectionRunner):
    name = "debt_strategy"

    def digest(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        analysis = facts.get("analysis") or {}
        diag = analysis.get("_diagnostics") or {}
        income = facts.get("income") or {}
        return {
            "debtStress": analysis.get("debtStress"),
            "emiPct": diag.get("emiPct"),
            "monthlyEmi": income.get("monthlyEmi"),
            "annualIncome": income.get("annualIncome"),
        }

    def prompt(self, digest: Dict[str, Any]) -> Tuple[str, str]:
        monthly_emi = _coerce_float(digest.get("monthlyEmi"), 0.0)
        annual_income = _coerce_float(digest.get("annualIncome"), 0.0)
        emi_pct = _coerce_float(digest.get("emiPct"), 0.0)
        monthly_income = annual_income / 12 if annual_income > 0 else 0
        
        # Calculate suggested prepayment from surplus (if EMI is >30%, suggest aggressive prepayment)
        target_emi_pct = 30.0
        excess_emi = max(0, monthly_emi - (monthly_income * target_emi_pct / 100))
        
        system = (
            "You craft debt optimization advice for Indian retail clients. "
            "Be specific about which debts to pay first and prepayment amounts. "
            "IMPORTANT: All monetary values MUST be in Indian Rupees (₹ or Rs.). NEVER use dollars ($)."
        )
        user = (
            "Section: Debt Strategy\n"
            f"FactsDigest:\n{_json_dumps(digest)}\n\n"
            "INSTRUCTIONS:\n"
            f"1. Current EMI: Rs.{monthly_emi:,.0f}/month ({emi_pct:.1f}% of income)\n"
            f"   - Benchmark: Keep EMI below 30-40% of income\n"
            f"2. DEBT PRIORITY (always recommend this order):\n"
            "   - Credit card debt (18-42% interest) - Pay off FIRST\n"
            "   - Personal loans (12-20% interest) - Pay off SECOND\n"
            "   - Car loans (8-12% interest) - Pay off THIRD\n"
            "   - Home loans (7-9% interest) - Lowest priority\n"
            f"3. PREPAYMENT: If EMI exceeds 30%, suggest monthly prepayment of Rs.{excess_emi:,.0f}\n"
            "   - Focus prepayment on highest-interest debt first\n"
            "4. Avoid new unsecured debt until EMI < 30%\n\n"
            "JSON output keys: title, bullets, paragraphs, actions. "
            "Actions should include specific prepayment amounts if applicable."
        )
        return system, user


class LiquidityPlanRunner(SectionRunner):
    name = "liquidity_plan"

    def digest(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        analysis = facts.get("analysis") or {}
        diag = analysis.get("_diagnostics") or {}
        return {
            "liquidity": analysis.get("liquidity"),
            "liquidityMonths": diag.get("liquidityMonths"),
        }

    def prompt(self, digest: Dict[str, Any]) -> Tuple[str, str]:
        system = "You design emergency fund guidance aligned with Indian norms (6+ months as baseline). IMPORTANT: All monetary values MUST be in Indian Rupees (₹ or Rs.). NEVER use dollars ($) or any other currency."
        user = (
            "Section: Liquidity Plan\n"
            f"FactsDigest:\n{_json_dumps(digest)}\n\n"
            "Return JSON with title, bullets, paragraphs, actions. Keep it concise and practical."
        )
        return system, user


class RiskRationaleRunner(SectionRunner):
    name = "risk_rationale"

    def digest(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        ar = (facts.get("analysis") or {}).get("advancedRisk") or {}
        return {
            "finalCategory": ar.get("finalCategory") or (facts.get("analysis") or {}).get("riskProfile"),
            "appetiteCategory": ar.get("appetiteCategory"),
            "tenureLimitCategory": ar.get("tenureLimitCategory"),
            "recommendedEquityBand": ar.get("recommendedEquityBand"),
            "recommendedEquityMid": ar.get("recommendedEquityMid"),
            "reasoningText": ar.get("reasoningText"),
        }

    def prompt(self, digest: Dict[str, Any]) -> Tuple[str, str]:
        system = "You explain risk profile and asset allocation rationale for Indian clients simply and accurately. IMPORTANT: All monetary values MUST be in Indian Rupees (₹ or Rs.). NEVER use dollars ($) or any other currency."
        user = (
            "Section: Risk Profile & Allocation Rationale\n"
            f"FactsDigest:\n{_json_dumps(digest)}\n\n"
            "Return JSON: title, bullets, paragraphs. Optionally actions. Avoid technical jargon."
        )
        return system, user


class GoalsStrategyRunner(SectionRunner):
    name = "goals_strategy"

    def digest(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        analysis = facts.get("analysis") or {}
        income_info = facts.get("income") or {}
        bank_info = facts.get("bank") or {}
        portfolio_info = facts.get("portfolio") or {}
        
        final_cat = ((analysis.get("advancedRisk") or {}).get("finalCategory")) or analysis.get("riskProfile")
        
        # Calculate user's actual financial capacity
        annual_income = _coerce_float(income_info.get("annualIncome"), 0.0)
        # Fallback to bank inflows if no declared income
        if annual_income <= 0:
            annual_income = _coerce_float(bank_info.get("total_inflows"), 0.0)
        
        monthly_income = annual_income / 12.0 if annual_income > 0 else 0.0
        monthly_expenses = _coerce_float(income_info.get("monthlyExpenses"), 0.0)
        monthly_emi = _coerce_float(income_info.get("monthlyEmi"), 0.0)
        
        # Calculate available surplus for investments
        available_surplus = max(0, monthly_income - monthly_expenses - monthly_emi)
        
        # Get existing SIP commitments from portfolio/CAS data if available
        # Note: SIPs may already be counted in monthly_expenses in some cases
        existing_sip_raw = _coerce_float(
            portfolio_info.get("monthly_sip") or portfolio_info.get("total_monthly_sip"), 0.0
        )
        # Cap existing SIP to not exceed available surplus (prevents negative budget)
        existing_sip = min(existing_sip_raw, available_surplus * 0.9)  # Allow 10% buffer
        
        # Net available for new goals (after existing commitments)
        net_available_for_goals = max(0, available_surplus - existing_sip)
        
        # Process each goal with affordability context
        goals_list = facts.get("goals") or []
        num_goals = len(goals_list) if goals_list else 1
        
        # Distribute available surplus proportionally across goals
        per_goal_budget = net_available_for_goals / num_goals if num_goals > 0 else 0
        
        # Ensure minimum floor: if surplus exists, allocate at least 5% of income per goal
        # This prevents the Rs. 0/month issue when existing_sip data is unreliable
        minimum_per_goal = (monthly_income * 0.05) if monthly_income > 0 else 0
        if available_surplus > 0 and per_goal_budget < minimum_per_goal:
            per_goal_budget = min(minimum_per_goal, available_surplus / num_goals)
        
        items = []
        total_ideal_sip = 0.0
        
        for g in goals_list:
            nm = g.get("name") or g.get("goal") or "Goal"
            tgt = _coerce_float(g.get("target_amount"), 0.0)
            hz = g.get("horizon_years") or g.get("horizon")
            try:
                hzv = float(hz) if hz not in (None, "", "N/A") else None
            except Exception:
                hzv = None
            
            # Get per-goal risk settings (fall back to defaults if not specified)
            goal_risk_tolerance = g.get("risk_tolerance") or "medium"
            goal_importance = g.get("goal_importance") or "important"
            goal_flexibility = g.get("goal_flexibility") or "fixed"
            goal_behavior = g.get("behavior") or "hold"
            
            # Compute per-goal risk category
            goal_risk_cat = compute_goal_risk_category(
                horizon_years=hzv,
                risk_tolerance=goal_risk_tolerance,
                goal_importance=goal_importance,
                goal_flexibility=goal_flexibility,
                behavior=goal_behavior
            )
            
            # Calculate ideal SIP using per-goal risk category
            ideal_sip = compute_goal_sip(tgt, hzv, goal_risk_cat)
            if ideal_sip:
                total_ideal_sip += ideal_sip
            
            # Calculate affordable SIP (capped by available budget)
            affordable_sip = min(ideal_sip, per_goal_budget) if ideal_sip and per_goal_budget > 0 else per_goal_budget
            
            # Calculate what's achievable with affordable SIP
            achievable_amount = compute_realistic_target(affordable_sip, hzv, goal_risk_cat) if affordable_sip and hzv else None
            
            # Calculate required horizon if SIP is limited
            required_horizon = None
            if ideal_sip and affordable_sip and affordable_sip < ideal_sip and tgt > 0:
                required_horizon = compute_required_horizon(tgt, affordable_sip, goal_risk_cat)
            
            # Determine if there's an affordability gap
            gap_exists = bool(ideal_sip and affordable_sip and ideal_sip > affordable_sip * 1.1)  # >10% gap
            shortfall_pct = round(((ideal_sip - affordable_sip) / ideal_sip) * 100, 1) if gap_exists and ideal_sip else 0
            
            # Compute priority score and tier for this goal
            priority_score, priority_tier = compute_goal_priority(
                horizon_years=hzv,
                importance=goal_importance,
                gap_exists=gap_exists,
                shortfall_percent=shortfall_pct
            )
            
            # Get return assumptions for this goal's risk category
            return_assumptions = _get_return_assumptions(goal_risk_cat, hzv)
            
            items.append({
                "name": nm,
                "target_amount": tgt if tgt > 0 else None,
                "horizon_years": hzv,
                "risk_category": goal_risk_cat,
                "risk_inputs": {
                    "tolerance": goal_risk_tolerance,
                    "importance": goal_importance,
                    "flexibility": goal_flexibility,
                    "behavior": goal_behavior,
                },
                "ideal_sip": ideal_sip,
                "affordable_sip": round(affordable_sip, 2) if affordable_sip else None,
                "achievable_amount": achievable_amount,
                "required_horizon_at_affordable": required_horizon,
                "gap_exists": gap_exists,
                "shortfall_percent": shortfall_pct,
                "priority_score": priority_score,
                "priority_tier": priority_tier,
                "calculation_assumptions": return_assumptions,
            })
        # Sort goals by priority score (lower = higher priority)
        items.sort(key=lambda x: x.get("priority_score", 100))
        
        # Assign priority rank based on sorted order
        for rank, item in enumerate(items, start=1):
            item["priority_rank"] = rank
        
        # Get overall equity recommendation context for reconciliation
        advanced_risk = analysis.get("advancedRisk") or {}
        overall_equity_band = advanced_risk.get("recommendedEquityBand") or {}
        
        # --- Priority Framework Integration ---
        # Extract insurance gap data from facts
        personal = facts.get("personal") or {}
        insurance = facts.get("insurance") or {}
        term_insurance_info = facts.get("term_insurance") or {}
        
        age = _coerce_float(personal.get("age"), 35)
        has_dependents = personal.get("has_financial_dependents", False) or (_coerce_float(personal.get("dependents_count"), 0) > 0)
        
        # Calculate term insurance gap
        term_gap = _coerce_float(term_insurance_info.get("gap"), 0.0)
        if term_gap <= 0:
            # Fallback: compute from required vs current
            required_cover = _coerce_float(
                (analysis.get("_diagnostics") or {}).get("requiredLifeCover"), 
                monthly_income * 12 * 10  # Default: 10x annual income
            )
            current_life_cover = _coerce_float(insurance.get("lifeCover"), 0.0)
            term_gap = max(0, required_cover - current_life_cover)
        
        # Calculate health insurance gap (recommend Rs. 10-15 lakh family floater)
        current_health_cover = _coerce_float(insurance.get("healthCover"), 0.0)
        recommended_health = 1000000  # Rs. 10 lakh minimum recommended
        if has_dependents:
            recommended_health = 1500000  # Rs. 15 lakh for family
        health_gap = max(0, recommended_health - current_health_cover)
        
        # Compute priority-based allocation
        priority_allocation = PriorityAllocationEngine.compute_allocation(
            monthly_surplus=available_surplus,
            term_insurance_gap=term_gap,
            health_insurance_gap=health_gap,
            goals=items,
            age=int(age),
            has_dependents=has_dependents
        )
        
        return {
            "riskFinalCategory": final_cat,
            "goals": items[:12],
            "financial_capacity": {
                "monthly_income": round(monthly_income, 2),
                "monthly_expenses": round(monthly_expenses, 2),
                "monthly_emi": round(monthly_emi, 2),
                "available_surplus": round(available_surplus, 2),
                "existing_sip_commitments": round(existing_sip, 2),
                "net_available_for_new_goals": round(net_available_for_goals, 2),
                "per_goal_budget": round(per_goal_budget, 2),
                "total_ideal_sip_required": round(total_ideal_sip, 2),
            },
            "overall_gap_exists": total_ideal_sip > net_available_for_goals * 1.1 if net_available_for_goals > 0 else True,
            "calculation_assumptions_note": f"Calculations use expected returns of 7.4%-14% p.a. (based on risk category) and {ASSUMED_INFLATION_RATE*100}% inflation.",
            "overall_recommended_equity_band": {
                "min": overall_equity_band.get("min"),
                "max": overall_equity_band.get("max"),
            },
            # NEW: Priority Framework Data
            "priority_framework": priority_allocation,
            "insurance_gaps": {
                "term_insurance_gap": round(term_gap, 0),
                "health_insurance_gap": round(health_gap, 0),
                "has_dependents": has_dependents,
            },
        }

    def prompt(self, digest: Dict[str, Any]) -> Tuple[str, str]:
        # Extract priority framework summary for prompt
        priority_data = digest.get("priority_framework") or {}
        achievement_pct = priority_data.get("goal_achievement_percent", 100)
        bridge_recs = priority_data.get("bridge_recommendations") or []
        
        system = (
            "You are a senior financial planner creating realistic goal strategies for Indian retail clients. "
            "IMPORTANT: All monetary values MUST be in Indian Rupees (₹ or Rs.). NEVER use dollars ($) or any other currency. "
            "Your role is to provide HONEST and PRACTICAL advice. "
            "For each goal, compare the IDEAL SIP (what's mathematically needed) against what's AFFORDABLE. "
            "Goals are pre-sorted by priority (priority_rank=1 is highest priority). "
            "If there's a gap (gap_exists=true), provide specific bridge recommendations. "
            "Do NOT recommend SIPs that exceed the client's capacity."
        )
        user = (
            "Section: Goal-wise Strategy with Priority Framework\n"
            f"FactsDigest:\n{_json_dumps(digest)}\n\n"
            "IMPORTANT INSTRUCTIONS:\n\n"
            "**PRIORITY FRAMEWORK (MUST INCLUDE):**\n"
            "The priority_framework section shows allocation in this order:\n"
            "  Priority 1: Term Insurance (if dependents exist)\n"
            "  Priority 2: Health Insurance (if gap exists)\n"
            "  Priority 3: Goal SIPs (remaining surplus)\n\n"
            f"Current goal achievement level: {achievement_pct:.0f}% of total goal requirements.\n"
        )
        
        # Add bridge recommendations if goals aren't fully funded
        if achievement_pct < 100 and bridge_recs:
            user += (
                "Since goals cannot be fully funded, include these BRIDGE OPTIONS in your response:\n"
            )
            for rec in bridge_recs[:3]:  # Top 3 recommendations
                user += f"  - {rec.get('option')}: {rec.get('action')}\n"
            user += "\n"
        
        user += (
            "1. START WITH PRIORITY SUMMARY: Begin with a bullet stating:\n"
            "   'With Rs.[surplus]/month: Rs.[X] for insurance, Rs.[Y] remaining for goals ([Z]% of requirement)'\n\n"
            "2. GOAL PRIORITY ORDER: Discuss goals in priority order (priority_rank=1 first).\n"
            "   - 'Immediate' tier: Address within 1-2 years\n"
            "   - 'Short-term' tier: Address in years 3-5\n"
            "   - 'Long-term' tier: Can be deferred to year 5+\n\n"
            "3. For each goal, clearly state:\n"
            "   - IDEAL: 'To achieve Rs.[target] in [horizon] years, you need Rs.[ideal_sip]/month'\n"
            "   - CURRENT CAPACITY: 'You can afford Rs.[affordable_sip]/month for this goal'\n"
            "   - Use the calculation_assumptions to note the expected return used\n\n"
            "4. If gap_exists=true, provide BRIDGE RECOMMENDATIONS:\n"
            "   - 'Extend timeline to [required_horizon_at_affordable] years'\n"
            "   - 'Or reduce target to Rs.[achievable_amount]'\n"
            "   - 'Or increase savings rate and reanalyze'\n\n"
            "5. PATH TO FULL ACHIEVEMENT: If achievement < 100%, explain how to reach 100%:\n"
            "   - Specific amount to reduce expenses\n"
            "   - Or specific amount to increase income\n"
            "   - Or start with current SIP and increase annually\n\n"
            "6. ASSUMPTIONS NOTE: Mention in one of the paragraphs:\n"
            f"   '{digest.get('calculation_assumptions_note', '')}'\n\n"
            "7. EQUITY RECONCILIATION: If per-goal allocations differ from overall_recommended_equity_band, "
            "explain that goal-specific allocations are theoretical and the overall portfolio should stay within the recommended band.\n\n"
            "Output JSON with keys:\n"
            "- title: 'Goal Strategy: Prioritized Action Plan'\n"
            "- bullets: First bullet = priority summary, then one per goal in priority order (max 6)\n"
            "- paragraphs: 2-3 paragraphs covering priority framework, phased plan, and bridge strategies\n"
            "- actions: Specific phased recommendations including path to full achievement (max 5)"
        )
        return system, user


class PortfolioRebalanceRunner(SectionRunner):
    name = "portfolio_rebalance"

    def digest(self, facts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        port = facts.get("portfolio") or {}
        if not isinstance(port, dict) or not port:
            return None
        analysis = facts.get("analysis") or {}
        ihs = analysis.get("ihs") or {}
        ar = analysis.get("advancedRisk") or {}
        return {
            "portfolio": port,
            "ihsBand": ihs.get("band"),
            "riskProfile": ar.get("finalCategory") or analysis.get("riskProfile"),
            "recommendedEquityBand": ar.get("recommendedEquityBand"),
            "recommendedEquityMid": ar.get("recommendedEquityMid"),
        }

    def prompt(self, digest: Dict[str, Any]) -> Tuple[str, str]:
        portfolio = digest.get("portfolio") or {}
        rec_band = digest.get("recommendedEquityBand") or {}
        rec_mid = _coerce_float(digest.get("recommendedEquityMid"), 50.0)
        rec_min = _coerce_float(rec_band.get("min"), 40.0)
        rec_max = _coerce_float(rec_band.get("max"), 60.0)
        
        current_equity = _coerce_float(portfolio.get("equity"), 50.0)
        current_debt = _coerce_float(portfolio.get("debt"), 30.0)
        
        # Calculate rebalancing direction
        if current_equity > rec_max:
            direction = "REDUCE EQUITY"
            shift_pct = round(current_equity - rec_mid, 1)
        elif current_equity < rec_min:
            direction = "INCREASE EQUITY"
            shift_pct = round(rec_mid - current_equity, 1)
        else:
            direction = "MAINTAIN"
            shift_pct = 0
        
        system = (
            "You suggest high-level rebalancing directions for Indian clients. "
            "IMPORTANT: All monetary values MUST be in Indian Rupees (₹). NEVER use dollars ($). "
            "Provide CATEGORY-level guidance (e.g., 'large-cap index funds', 'debt mutual funds') "
            "but do NOT name specific AMCs or fund names. "
            "Be consistent with the client's risk profile."
        )
        user = (
            "Section: Portfolio Rebalancing\n"
            f"FactsDigest:\n{_json_dumps(digest)}\n\n"
            "INSTRUCTIONS:\n"
            f"1. CURRENT vs RECOMMENDED:\n"
            f"   - Current Equity: {current_equity:.1f}%\n"
            f"   - Recommended Range: {rec_min:.0f}-{rec_max:.0f}% (target: {rec_mid:.0f}%)\n"
            f"   - Action: {direction} by ~{shift_pct}% of portfolio\n\n"
            "2. CATEGORY-LEVEL GUIDANCE (use these terms, not product names):\n"
            "   TO REDUCE EQUITY:\n"
            "   - Move to short-duration debt funds or corporate bond funds\n"
            "   - Consider liquid funds for near-term needs\n"
            "   - Book profits in small/mid-cap and shift to debt\n"
            "   TO INCREASE EQUITY:\n"
            "   - Add to large-cap index funds or flexi-cap funds\n"
            "   - Use SIP route for gradual deployment\n"
            "   - Consider hybrid equity-debt funds for moderate risk\n\n"
            "3. DIVERSIFICATION SUGGESTIONS:\n"
            "   - International equity: 5-10% via US/global index funds\n"
            "   - Gold: 5-10% via sovereign gold bonds or gold ETFs\n\n"
            "Return JSON: title, bullets, paragraphs, actions. "
            "Actions should include specific percentage shifts."
        )
        return system, user


# ------------------------ Tax Optimization Constants & Helpers ------------------------ #

# Tax Deduction Limits (FY 2024-25 / AY 2025-26)
TAX_DEDUCTION_LIMITS = {
    "80C": 150000,              # PPF, ELSS, LIC, EPF, principal repayment, etc.
    "80CCD_1B": 50000,          # Additional NPS contribution (over and above 80C)
    "80D_self": 25000,          # Health insurance - self/family (<60 years)
    "80D_parents": 50000,       # Health insurance - senior citizen parents (>60 years)
    "80D_parents_non_senior": 25000,  # Health insurance - non-senior parents
    "24B_HOME_LOAN_INTEREST": 200000,  # Home loan interest deduction
    "80E_EDUCATION_LOAN": None,  # No limit (full interest deductible)
    "80G_DONATIONS": None,       # 50% or 100% of donation (varies)
    "LTCG_EXEMPTION": 125000,   # Long-term capital gains exemption (FY25-26)
}

# Tax slab rates for old regime (FY 2024-25)
TAX_SLABS_OLD_REGIME = [
    (250000, 0.00),
    (500000, 0.05),
    (1000000, 0.20),
    (float('inf'), 0.30),
]

# Tax slab rates for new regime (FY 2024-25)
TAX_SLABS_NEW_REGIME = [
    (300000, 0.00),
    (700000, 0.05),
    (1000000, 0.10),
    (1200000, 0.15),
    (1500000, 0.20),
    (float('inf'), 0.30),
]


def _get_marginal_tax_rate(taxable_income: float, regime: str = "old") -> float:
    """Get the marginal tax rate for given taxable income."""
    slabs = TAX_SLABS_OLD_REGIME if regime == "old" else TAX_SLABS_NEW_REGIME
    prev_threshold = 0
    for threshold, rate in slabs:
        if taxable_income <= threshold:
            return rate
        prev_threshold = threshold
    return 0.30  # Default to highest slab


def compute_tax_deduction_gaps(
    deductions_claimed: List[Dict[str, Any]],
    insurance_data: Dict[str, Any],
    gross_income: float,
    taxable_income: float
) -> Dict[str, Dict[str, Any]]:
    """
    Compute utilization and gaps for key tax deduction sections.
    Uses extracted ITR data for deductions already claimed.
    
    Returns dict with section -> {limit, current, gap, instruments, tax_saving_potential}
    """
    gaps = {}
    
    # Create lookup for claimed deductions
    claimed = {}
    for d in (deductions_claimed or []):
        section = str(d.get("section", "")).upper().strip()
        amount = _coerce_float(d.get("amount"), 0.0)
        # Normalize section names
        if "80C" in section and "80CCD" not in section:
            claimed["80C"] = claimed.get("80C", 0.0) + amount
        elif "80CCD" in section or "NPS" in section.upper():
            claimed["80CCD_1B"] = claimed.get("80CCD_1B", 0.0) + amount
        elif "80D" in section:
            claimed["80D"] = claimed.get("80D", 0.0) + amount
        elif "80E" in section:
            claimed["80E"] = claimed.get("80E", 0.0) + amount
        elif "80G" in section:
            claimed["80G"] = claimed.get("80G", 0.0) + amount
    
    # Calculate marginal rate for tax saving calculations
    marginal_rate = _get_marginal_tax_rate(taxable_income)
    
    # 80C Analysis
    limit_80c = TAX_DEDUCTION_LIMITS["80C"]
    current_80c = claimed.get("80C", 0.0)
    gap_80c = max(0, limit_80c - current_80c)
    if gap_80c > 0:
        gaps["80C"] = {
            "limit": limit_80c,
            "current_utilization": current_80c,
            "gap": gap_80c,
            "utilization_percent": round((current_80c / limit_80c) * 100, 1),
            "tax_saving_potential": round(gap_80c * marginal_rate, 0),
            "recommended_instruments": [
                {"name": "ELSS", "priority": 1, "amount": min(gap_80c, 50000), 
                 "reason": "Equity exposure with shortest lock-in (3 years) + tax saving"},
                {"name": "PPF", "priority": 2, "amount": min(gap_80c, 50000), 
                 "reason": "Safe, guaranteed returns, 15-year tax-free compounding"},
                {"name": "NPS Tier-1", "priority": 3, "amount": min(gap_80c, 50000), 
                 "reason": "Retirement savings + additional 80CCD(1B) benefit available"},
            ]
        }
    
    # 80CCD(1B) Analysis - Additional NPS
    limit_80ccd = TAX_DEDUCTION_LIMITS["80CCD_1B"]
    current_80ccd = claimed.get("80CCD_1B", 0.0)
    gap_80ccd = max(0, limit_80ccd - current_80ccd)
    if gap_80ccd > 0:
        gaps["80CCD_1B"] = {
            "limit": limit_80ccd,
            "current_utilization": current_80ccd,
            "gap": gap_80ccd,
            "utilization_percent": round((current_80ccd / limit_80ccd) * 100, 1),
            "tax_saving_potential": round(gap_80ccd * marginal_rate, 0),
            "recommended_instruments": [
                {"name": "NPS Tier-1", "priority": 1, "amount": gap_80ccd,
                 "reason": "Additional ₹50K deduction OVER AND ABOVE Section 80C limit"}
            ],
            "note": "This is in addition to 80C limit. Invest ₹50K in NPS for extra deduction."
        }
    
    # 80D Analysis - Health Insurance
    # Use insurance data from CAS/documents if available, otherwise use claimed
    health_premium_claimed = claimed.get("80D", 0.0)
    health_cover = _coerce_float(insurance_data.get("healthCover"), 0.0)
    
    # Estimate 80D utilization - if we have insurance docs, assume some premium was paid
    limit_80d_total = TAX_DEDUCTION_LIMITS["80D_self"] + TAX_DEDUCTION_LIMITS["80D_parents"]
    current_80d = health_premium_claimed
    gap_80d = max(0, limit_80d_total - current_80d)
    
    if gap_80d > 0:
        gaps["80D"] = {
            "limit_self_family": TAX_DEDUCTION_LIMITS["80D_self"],
            "limit_parents": TAX_DEDUCTION_LIMITS["80D_parents"],
            "total_limit": limit_80d_total,
            "current_utilization": current_80d,
            "gap": gap_80d,
            "utilization_percent": round((current_80d / limit_80d_total) * 100, 1),
            "tax_saving_potential": round(gap_80d * marginal_rate, 0),
            "recommended_actions": [
                {"action": "Top-up health insurance", "priority": 1,
                 "reason": "Medical inflation at 10-15%; adequate health cover prevents financial stress"},
                {"action": "Add parents under 80D", "priority": 2,
                 "reason": "₹50K additional deduction for senior citizen parents' health insurance"}
            ]
        }
    
    return gaps


def detect_tax_regime(deductions_claimed: List[Dict[str, Any]], total_deductions: float) -> str:
    """
    Infer tax regime based on deductions claimed.
    If significant deductions are claimed, likely old regime. Otherwise new regime.
    """
    if not deductions_claimed and total_deductions <= 0:
        return "new_regime_likely"
    
    total_claimed = sum(_coerce_float(d.get("amount"), 0.0) for d in (deductions_claimed or []))
    if total_claimed >= 50000:  # Meaningful deductions claimed
        return "old_regime"
    
    return "regime_unclear"


def compute_effective_tax_rate(tax_paid: float, gross_income: float) -> float:
    """Calculate effective tax rate as percentage."""
    if gross_income <= 0:
        return 0.0
    return round((tax_paid / gross_income) * 100, 2)


class TaxOptimizationRunner(SectionRunner):
    """
    Tax Optimization Section Runner.
    
    Uses extracted ITR data to:
    1. Analyze current tax profile and deduction utilization
    2. Identify gaps in 80C, 80CCD(1B), 80D, and other sections
    3. Calculate potential tax savings from optimizing deductions
    4. Provide actionable recommendations for next FY
    5. Include capital gains optimization strategies
    """
    name = "tax_optimization"
    
    def digest(self, facts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Get ITR data - this is the key data source
        itr_data = facts.get("itr") or {}
        
        # If no ITR data, skip this section
        if not itr_data:
            return None
        
        # Extract key ITR fields
        gross_income = _coerce_float(itr_data.get("gross_total_income"), 0.0)
        taxable_income = _coerce_float(itr_data.get("taxable_income"), 0.0)
        total_tax_paid = _coerce_float(itr_data.get("total_tax_paid"), 0.0)
        
        # If no income data, skip
        if gross_income <= 0:
            return None
        
        # Get deductions claimed from ITR
        deductions_claimed = itr_data.get("deductions_claimed") or []
        
        # Get tax computation details if available
        tax_computation = itr_data.get("tax_computation") or {}
        total_deductions = _coerce_float(
            tax_computation.get("total_deductions") or (gross_income - taxable_income),
            0.0
        )
        
        # Get income sources breakdown
        income_sources = itr_data.get("income_sources") or {}
        
        # Get capital gains data for LTCG optimization
        capital_gains = _coerce_float(income_sources.get("capital_gains"), 0.0)
        
        # Get carry forward losses for offset opportunities
        carry_forward_losses = itr_data.get("carry_forward_losses") or {}
        
        # Get insurance data from facts for 80D analysis
        insurance_data = facts.get("insurance") or {}
        
        # Detect likely tax regime
        regime = detect_tax_regime(deductions_claimed, total_deductions)
        
        # Compute effective tax rate
        effective_rate = compute_effective_tax_rate(total_tax_paid, gross_income)
        
        # Calculate marginal tax rate for savings projections
        marginal_rate = _get_marginal_tax_rate(taxable_income)
        
        # Compute deduction gaps and recommendations
        deduction_gaps = {}
        total_potential_saving = 0.0
        
        if regime != "new_regime_likely":
            deduction_gaps = compute_tax_deduction_gaps(
                deductions_claimed=deductions_claimed,
                insurance_data=insurance_data,
                gross_income=gross_income,
                taxable_income=taxable_income
            )
            # Sum up potential savings
            for section, data in deduction_gaps.items():
                total_potential_saving += _coerce_float(data.get("tax_saving_potential"), 0.0)
        
        # LTCG optimization opportunity
        ltcg_optimization = None
        if capital_gains > 0 or capital_gains == 0:
            ltcg_exemption = TAX_DEDUCTION_LIMITS["LTCG_EXEMPTION"]
            # If capital gains are less than exemption limit, suggest gain harvesting
            if 0 <= capital_gains < ltcg_exemption:
                harvestable = ltcg_exemption - capital_gains
                ltcg_optimization = {
                    "type": "GAIN_HARVESTING",
                    "current_ltcg": capital_gains,
                    "exemption_limit": ltcg_exemption,
                    "harvestable_room": harvestable,
                    "recommendation": f"You can book up to ₹{harvestable:,.0f} in additional LTCG tax-free this year",
                    "action": "Redeem equity mutual funds showing gains, then reinvest to reset cost base"
                }
            elif capital_gains > ltcg_exemption:
                excess = capital_gains - ltcg_exemption
                ltcg_optimization = {
                    "type": "STRATEGIC_REDEMPTION",
                    "current_ltcg": capital_gains,
                    "exemption_limit": ltcg_exemption,
                    "taxable_ltcg": excess,
                    "tax_liability": round(excess * 0.125, 0),  # 12.5% LTCG tax
                    "recommendation": "Spread large redemptions across financial years to optimize ₹1.25L annual exemption",
                    "action": "Plan redemptions to stay within ₹1.25L LTCG per year"
                }
        
        # Tax loss harvesting opportunity
        portfolio = facts.get("portfolio") or {}
        loss_harvesting = None
        # Note: We'd need CAS data with unrealized losses for full implementation
        # For now, flag the strategy if equity allocation exists
        if _coerce_float(portfolio.get("equity"), 0.0) > 0:
            loss_harvesting = {
                "strategy": "TAX_LOSS_HARVESTING",
                "recommendation": "Review portfolio for underperforming funds with unrealized losses",
                "benefit": "Book losses to offset against STCG/LTCG; carry forward unused losses for 8 years",
                "timing": "Execute before March 31 to claim in current FY"
            }
        
        # Build next-year roadmap
        action_items = []
        total_actions_saving = 0.0
        
        # Priority order: 80C first (biggest limit), then 80CCD, then 80D
        if "80C" in deduction_gaps:
            gap = deduction_gaps["80C"]
            action_items.append({
                "priority": 1,
                "action": f"Max out 80C (₹{gap['limit']:,.0f}) via ELSS/PPF",
                "current": f"₹{gap['current_utilization']:,.0f} used",
                "gap": f"₹{gap['gap']:,.0f} remaining",
                "tax_saving": f"₹{gap['tax_saving_potential']:,.0f}"
            })
            total_actions_saving += gap['tax_saving_potential']
        
        if "80CCD_1B" in deduction_gaps:
            gap = deduction_gaps["80CCD_1B"]
            action_items.append({
                "priority": 2,
                "action": f"Invest ₹{gap['gap']:,.0f} in NPS (80CCD-1B)",
                "current": f"₹{gap['current_utilization']:,.0f} used",
                "gap": f"₹{gap['gap']:,.0f} remaining",
                "tax_saving": f"₹{gap['tax_saving_potential']:,.0f}",
                "note": "This is ADDITIONAL to 80C limit"
            })
            total_actions_saving += gap['tax_saving_potential']
        
        if "80D" in deduction_gaps:
            gap = deduction_gaps["80D"]
            action_items.append({
                "priority": 3,
                "action": f"Top-up health insurance by ₹{gap['gap']:,.0f}",
                "current": f"₹{gap['current_utilization']:,.0f} used",
                "gap": f"₹{gap['gap']:,.0f} remaining (self + parents)",
                "tax_saving": f"₹{gap['tax_saving_potential']:,.0f}"
            })
            total_actions_saving += gap['tax_saving_potential']
        
        return {
            "tax_profile": {
                "gross_income": gross_income,
                "taxable_income": taxable_income,
                "total_tax_paid": total_tax_paid,
                "effective_tax_rate": effective_rate,
                "marginal_tax_rate": round(marginal_rate * 100, 1),
                "detected_regime": regime,
            },
            "income_sources": income_sources,
            "deductions_summary": {
                "total_claimed": total_deductions,
                "deductions_list": deductions_claimed[:10],  # Limit for prompt size
            },
            "deduction_gaps": deduction_gaps,
            "ltcg_optimization": ltcg_optimization,
            "loss_harvesting_strategy": loss_harvesting,
            "carry_forward_losses": carry_forward_losses if carry_forward_losses else None,
            "next_year_roadmap": {
                "action_items": action_items,
                "total_potential_saving": round(total_actions_saving, 0),
            },
            "regime_advice": {
                "old_regime": "Maximize deductions (80C, 80D, 80CCD, HRA) to reduce taxable income",
                "new_regime": "No deductions allowed; focus on post-tax optimization via tax-efficient products",
            }
        }
    
    def prompt(self, digest: Dict[str, Any]) -> Tuple[str, str]:
        tax_profile = digest.get("tax_profile") or {}
        deduction_gaps = digest.get("deduction_gaps") or {}
        roadmap = digest.get("next_year_roadmap") or {}
        ltcg_opt = digest.get("ltcg_optimization")
        
        gross_income = tax_profile.get("gross_income", 0)
        taxable_income = tax_profile.get("taxable_income", 0)
        effective_rate = tax_profile.get("effective_tax_rate", 0)
        marginal_rate = tax_profile.get("marginal_tax_rate", 0)
        regime = tax_profile.get("detected_regime", "old_regime")
        total_saving = roadmap.get("total_potential_saving", 0)
        
        system = (
            "You are a tax planning expert for Indian retail clients. "
            "IMPORTANT: All monetary values MUST be in Indian Rupees (₹ or Rs.). NEVER use dollars ($). "
            "Provide SPECIFIC and ACTIONABLE tax optimization advice. "
            "Reference actual sections (80C, 80CCD, 80D) and their limits. "
            "Be precise about amounts and tax savings. Do not give generic advice."
        )
        
        user = (
            "Section: Tax Optimization Strategy\n\n"
            f"Client Tax Profile:\n{_json_dumps(digest)}\n\n"
            "INSTRUCTIONS - Create a comprehensive tax optimization report:\n\n"
            "1. CURRENT TAX SUMMARY (use these exact numbers):\n"
            f"   - Gross Income: ₹{gross_income:,.0f}\n"
            f"   - Taxable Income: ₹{taxable_income:,.0f}\n"
            f"   - Effective Tax Rate: {effective_rate}%\n"
            f"   - Marginal Tax Rate: {marginal_rate}%\n"
            f"   - Detected Regime: {regime.replace('_', ' ').title()}\n\n"
            "2. DEDUCTION GAPS IDENTIFIED:\n"
        )
        
        # Add specific gap details
        for section, gap_data in deduction_gaps.items():
            if isinstance(gap_data, dict):
                user += f"   - Section {section}: Gap of ₹{gap_data.get('gap', 0):,.0f} (can save ₹{gap_data.get('tax_saving_potential', 0):,.0f} in tax)\n"
        
        user += (
            "\n3. RECOMMENDATIONS BY PRIORITY:\n"
            "   For each gap, recommend SPECIFIC instruments:\n"
            "   - 80C: ELSS (for equity + tax), PPF (for safe returns), NPS (for retirement)\n"
            "   - 80CCD(1B): NPS Tier-1 (additional ₹50K over 80C)\n"
            "   - 80D: Health insurance top-up, super top-up policies\n\n"
        )
        
        if ltcg_opt:
            user += (
                "4. CAPITAL GAINS OPTIMIZATION:\n"
                f"   - Strategy: {ltcg_opt.get('type', 'N/A')}\n"
                f"   - {ltcg_opt.get('recommendation', '')}\n"
                f"   - Action: {ltcg_opt.get('action', '')}\n\n"
            )
        
        user += (
            f"5. TOTAL POTENTIAL TAX SAVING: ₹{total_saving:,.0f}\n\n"
            "6. REGIME-SPECIFIC ADVICE:\n"
            f"   - If OLD REGIME: {digest.get('regime_advice', {}).get('old_regime', '')}\n"
            f"   - If NEW REGIME: {digest.get('regime_advice', {}).get('new_regime', '')}\n\n"
            "OUTPUT JSON STRUCTURE:\n"
            "{\n"
            '  "title": "Tax Optimization: Maximize Your Savings",\n'
            '  "bullets": [max 6 bullets with specific amounts and sections],\n'
            '  "paragraphs": [2-3 paragraphs explaining strategy and next steps],\n'
            '  "actions": [max 5 specific actions with amounts, timelines]\n'
            "}\n\n"
            "CRITICAL: Include SPECIFIC amounts from the data. Do NOT give vague advice."
        )
        
        return system, user


class ExecutiveSummaryRunner(SectionRunner):
    name = "executive_summary"

    def __init__(self, llm: LLMClient, qid: int, base_dir: str, outlines: Dict[str, Dict[str, Any]]):
        super().__init__(llm, qid, base_dir)
        self.outlines = outlines or {}

    def digest(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        # Only small outlines from previous sections, not raw facts
        def _small(sec):
            d = self.outlines.get(sec) or {}
            return {
                "title": d.get("title"),
                "bullets": (d.get("bullets") or [])[:5],
            }
        return {
            "flags_explainer": _small("flags_explainer"),
            "protection_plan": _small("protection_plan"),
            "cashflow": _small("cashflow"),
            "debt_strategy": _small("debt_strategy"),
            "liquidity_plan": _small("liquidity_plan"),
            "risk_rationale": _small("risk_rationale"),
            "goals_strategy": _small("goals_strategy"),
            "portfolio_rebalance": _small("portfolio_rebalance"),
            "tax_optimization": _small("tax_optimization"),
        }

    def prompt(self, digest: Dict[str, Any]) -> Tuple[str, str]:
        system = "You write a crisp executive summary for Indian retail clients. IMPORTANT: All monetary values MUST be in Indian Rupees (₹ or Rs.). NEVER use dollars ($) or any other currency."
        user = (
            "Section: Executive Summary\n"
            "You are given short outlines from multiple sections. Create a concise summary.\n"
            f"SectionOutlines:\n{_json_dumps(digest)}\n\n"
            "Output JSON keys: title, bullets (<=6), paragraphs (<=2). No actions."
        )
        return system, user


# ------------------------ Orchestrator ------------------------ #

class SectionsOrchestrator:
    ORDER = [
        "flags_explainer",
        "protection_plan",
        "cashflow",
        "debt_strategy",
        "liquidity_plan",
        "risk_rationale",
        "goals_strategy",
        "portfolio_rebalance",
        "tax_optimization",
        "executive_summary",
    ]

    def __init__(self, qid: int, client: Optional[LLMClient] = None):
        self.qid = qid
        self.client = client or LLMClient()
        self.base_dir = os.path.join(OUTPUT_DIR, SECTIONS_DIR_NAME, str(qid))
        _ensure_dir(self.base_dir)

    def run(self, facts: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}

        runners: Dict[str, SectionRunner] = {
            "flags_explainer": FlagsExplainerRunner(self.client, self.qid, self.base_dir),
            "protection_plan": ProtectionPlanRunner(self.client, self.qid, self.base_dir),
            "cashflow": CashflowRunner(self.client, self.qid, self.base_dir),
            "debt_strategy": DebtStrategyRunner(self.client, self.qid, self.base_dir),
            "liquidity_plan": LiquidityPlanRunner(self.client, self.qid, self.base_dir),
            "risk_rationale": RiskRationaleRunner(self.client, self.qid, self.base_dir),
            "goals_strategy": GoalsStrategyRunner(self.client, self.qid, self.base_dir),
            "portfolio_rebalance": PortfolioRebalanceRunner(self.client, self.qid, self.base_dir),
            "tax_optimization": TaxOptimizationRunner(self.client, self.qid, self.base_dir),
        }

        # Run main sections
        for key in self.ORDER:
            if key == "executive_summary":
                break
            runner = runners.get(key)
            if not runner:
                continue
            try:
                out = runner.run(facts)
            except Exception as e:
                out = {"title": f"{key.replace('_', ' ').title()} (Error)", "bullets": [], "paragraphs": [], "actions": [], "error": str(e)}
            if out:
                results[key] = out

        # Exec summary with outlines only
        try:
            exec_runner = ExecutiveSummaryRunner(self.client, self.qid, self.base_dir, outlines=results)
            out = exec_runner.run(facts)
            if out:
                results["executive_summary"] = out
        except Exception as e:
            results["executive_summary"] = {"title": "Executive Summary (Error)", "bullets": [], "paragraphs": [], "actions": [], "error": str(e)}

        # Persist aggregate
        agg_path = os.path.join(self.base_dir, "all_sections.json")
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return results


def run_report_sections(questionnaire_id: int, client_facts: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    client = LLMClient(model=model or DEFAULT_MODEL)
    orch = SectionsOrchestrator(qid=questionnaire_id, client=client)
    return orch.run(client_facts)
