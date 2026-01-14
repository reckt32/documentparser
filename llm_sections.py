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
        sentinel = analysis.get("sentinel") or {}
        bank = facts.get("bank") or {}
        return {
            "surplusBand": analysis.get("surplusBand"),
            "waterfall": analysis.get("cashflow_waterfall") or sentinel.get("cashflow_waterfall"),
            "net_available": sentinel.get("net_available"),
            "allocation_priorities": analysis.get("allocation_priorities") or sentinel.get("allocation_priorities"),
            "bank": {
                "total_inflows": bank.get("total_inflows"),
                "total_outflows": bank.get("total_outflows"),
                "net_cashflow": bank.get("net_cashflow"),
            }
        }

    def prompt(self, digest: Dict[str, Any]) -> Tuple[str, str]:
        system = (
            "You are a financial planner for Indian clients. Explain cashflows clearly, with practical advice to improve savings. "
            "IMPORTANT: All monetary values MUST be in Indian Rupees (₹ or Rs.). NEVER use dollars ($) or any other currency. "
            "Use the waterfall structure to show money flow visually."
        )
        user = (
            "Section: Cashflow Overview (Waterfall Analysis)\n"
            f"FactsDigest:\n{_json_dumps(digest)}\n\n"
            "INSTRUCTIONS:\n"
            "1. Present the cashflow as a WATERFALL showing:\n"
            "   - Gross Inflows -> Mandatory Outflows -> Living Expenses -> Investments -> Net Available\n"
            "2. Highlight the NET AVAILABLE amount prominently\n"
            "3. If allocation_priorities exist, explain the recommended order of fund deployment:\n"
            "   - Emergency fund first (if < 6 months)\n"
            "   - Insurance gaps next\n"
            "   - High-interest debt prepayment\n"
            "   - Goal investments last\n"
            "4. If net_available is low or negative, provide specific cost-cutting recommendations\n\n"
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
        }

    def prompt(self, digest: Dict[str, Any]) -> Tuple[str, str]:
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
            "Section: Goal-wise Strategy with Feasibility Analysis\n"
            f"FactsDigest:\n{_json_dumps(digest)}\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "1. PRIORITY ORDER: Goals are sorted by priority. Discuss them in priority order (priority_rank=1 first).\n"
            "   - 'Immediate' tier: Address within 1-2 years\n"
            "   - 'Short-term' tier: Address in years 3-5\n"
            "   - 'Long-term' tier: Can be deferred to year 5+\n\n"
            "2. For each goal, clearly state:\n"
            "   - IDEAL: 'To achieve Rs.[target] in [horizon] years, you need Rs.[ideal_sip]/month'\n"
            "   - CURRENT CAPACITY: 'You can afford Rs.[affordable_sip]/month for this goal'\n"
            "   - Use the calculation_assumptions to note the expected return used\n\n"
            "3. If gap_exists=true, provide BRIDGE RECOMMENDATIONS:\n"
            "   - 'Extend timeline to [required_horizon_at_affordable] years'\n"
            "   - 'Or reduce target to Rs.[achievable_amount]'\n"
            "   - 'Or increase savings rate and reanalyze'\n\n"
            "4. PHASED ACTION PLAN: Create a timeline based on priority_tier:\n"
            "   - 'Year 1-2: Focus on [Immediate priority goals] + build emergency fund + insurance'\n"
            "   - 'Year 3-5: Begin investing for [Short-term goals]'\n"
            "   - 'Year 5+: Accelerate [Long-term goals]'\n\n"
            "5. ASSUMPTIONS NOTE: Mention in one of the paragraphs:\n"
            f"   '{digest.get('calculation_assumptions_note', '')}'\n\n"
            "6. EQUITY RECONCILIATION: If per-goal allocations differ from overall_recommended_equity_band, "
            "explain that goal-specific allocations are theoretical and the overall portfolio should stay within the recommended band.\n\n"
            "Output JSON with keys:\n"
            "- title: 'Goal Strategy: Prioritized Action Plan'\n"
            "- bullets: One per goal in priority order, showing ideal vs affordable (max 6)\n"
            "- paragraphs: 2-3 paragraphs covering phased plan, assumptions, and bridge strategies\n"
            "- actions: Specific phased recommendations (max 5)"
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


class TaxEfficiencyRunner(SectionRunner):
    name = "tax_efficiency"

    def digest(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        analysis = facts.get("analysis") or {}
        tax_efficiency = analysis.get("tax_efficiency") or {}
        itr = facts.get("itr") or {}
        
        return {
            "recommendations": tax_efficiency.get("recommendations", []),
            "total_tax_alpha": tax_efficiency.get("total_tax_alpha", 0),
            "ltcg_harvest": tax_efficiency.get("ltcg_harvest"),
            "deductions_claimed": itr.get("deductions_claimed", []),
            "taxable_income": itr.get("taxable_income"),
        }

    def prompt(self, digest: Dict[str, Any]) -> Tuple[str, str]:
        recommendations = digest.get("recommendations", [])
        total_alpha = _coerce_float(digest.get("total_tax_alpha"), 0)
        ltcg = digest.get("ltcg_harvest")
        
        system = (
            "You are a tax optimization specialist for Indian clients. "
            "IMPORTANT: All monetary values MUST be in Indian Rupees (₹ or Rs.). NEVER use dollars ($). "
            "Provide ACTIONABLE recommendations with SPECIFIC amounts and DEADLINES. "
            "Focus on 80C, 80D, 80CCD(1B), and LTCG harvesting opportunities."
        )
        
        user = (
            "Section: Tax Efficiency & Savings\n"
            f"FactsDigest:\n{_json_dumps(digest)}\n\n"
            "INSTRUCTIONS:\n"
            "1. MISSED SAVINGS: For each recommendation, format as:\n"
            "   '80C: Invest ₹[gap_amount] in ELSS/PPF by [deadline] to save ₹[tax_saved] in taxes'\n"
            "2. LTCG HARVESTING: If ltcg_harvest exists, explain:\n"
            "   - Current unrealized gains vs ₹1.25L exemption limit\n"
            "   - Specific harvest recommendation with deadline\n"
            "3. PRIORITY ORDER:\n"
            "   - High-impact low-effort items first (e.g., health premium top-up)\n"
            "   - Lock-in items next (e.g., PPF, ELSS)\n"
            "   - NPS last (15-year+ lock-in)\n"
            f"4. HIGHLIGHT total potential tax savings: ₹{total_alpha:,.0f}\n\n"
            "Return JSON: title, bullets, paragraphs, actions.\n"
            "Actions MUST include specific amounts and the deadline 'by March 31st'."
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
            "tax_efficiency": _small("tax_efficiency"),
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
        "tax_efficiency",
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
            "tax_efficiency": TaxEfficiencyRunner(self.client, self.qid, self.base_dir),
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
