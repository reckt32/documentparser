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
    # Simple heuristic, monthly expected return
    cat = (final_category or "").lower()
    h = float(horizon_years) if horizon_years not in (None, "") else None
    if cat in ["very aggressive", "aggressive"]:
        base = 0.011
    elif cat in ["growth", "moderate"]:
        base = 0.009
    else:
        base = 0.006
    # Short horizons tend lower
    if h is not None:
        if h <= 3:
            base = min(base, 0.006)
        elif h <= 7:
            base = min(base, 0.009)
    return base


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
        system = (
            "You are a senior financial planner. Draft a protection (life/health) section for Indian clients. "
            "No guarantees; avoid product pushing; focus on adequacy and prioritization."
        )
        user = (
            "Section: Protection Plan\n"
            f"FactsDigest:\n{_json_dumps(digest)}\n\n"
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
            "You are a financial planner. Explain cashflows clearly, with practical advice to improve savings."
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
        system = "You craft debt optimization advice for Indian retail clients. Be specific and conservative."
        user = (
            "Section: Debt Strategy\n"
            f"FactsDigest:\n{_json_dumps(digest)}\n\n"
            "JSON output keys: title, bullets, paragraphs, actions. Avoid guarantees. Suggestions should be prioritized."
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
        system = "You design emergency fund guidance aligned with Indian norms (6+ months as baseline)."
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
        system = "You explain risk profile and asset allocation rationale simply and accurately."
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
        final_cat = ((analysis.get("advancedRisk") or {}).get("finalCategory")) or analysis.get("riskProfile")
        items = []
        for g in (facts.get("goals") or []):
            nm = g.get("name") or g.get("goal") or "Goal"
            tgt = _coerce_float(g.get("target_amount"), 0.0)
            hz = g.get("horizon_years") or g.get("horizon")
            try:
                hzv = float(hz) if hz not in (None, "", "N/A") else None
            except Exception:
                hzv = None
            sip = compute_goal_sip(tgt, hzv, final_cat)
            items.append({
                "name": nm,
                "target_amount": tgt if tgt > 0 else None,
                "horizon_years": hzv,
                "suggested_sip": sip,
            })
        return {
            "riskFinalCategory": final_cat,
            "goals": items[:12],
        }

    def prompt(self, digest: Dict[str, Any]) -> Tuple[str, str]:
        system = (
            "You map goals to strategies in simple English. Use suggested_sip if present as a numeric reference; do not recalc."
        )
        user = (
            "Section: Goal-wise Strategy\n"
            f"FactsDigest:\n{_json_dumps(digest)}\n\n"
            "For each goal, produce a short rationale. Output JSON keys: title, bullets (one per goal), paragraphs (1-2 max), actions."
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
        system = (
            "You suggest high-level rebalancing directions without naming products. "
            "IMPORTANT: Be consistent with the client's risk profile. "
            "If portfolio equity is already above the recommended band, suggest reducing equity exposure. "
            "If portfolio equity is below the recommended band, suggest increasing equity exposure. "
            "Never give advice that contradicts the risk profile recommendations."
        )
        user = (
            "Section: Portfolio Rebalancing\n"
            f"FactsDigest:\n{_json_dumps(digest)}\n\n"
            "The recommendedEquityBand shows the target equity range based on the client's risk profile. "
            "Compare current portfolio allocation against this target when suggesting changes.\n\n"
            "Return JSON: title, bullets, paragraphs, actions. Avoid product recommendations; discuss allocation directions."
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
        }

    def prompt(self, digest: Dict[str, Any]) -> Tuple[str, str]:
        system = "You write a crisp executive summary for Indian retail clients."
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
