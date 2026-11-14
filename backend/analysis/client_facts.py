"""Assemble questionnaire + document insights into analysis inputs and client facts."""

from backend.analysis.metrics import compute_retirement_corpus, compute_term_insurance_need
from backend.extraction.insights import _get_cas_data_for_questionnaire
from backend.utils import _safe_float


def _resolve_current_allocation(lifestyle: dict, risk: dict, cas_portfolio: dict) -> dict:
    """Single source of truth for the client's CURRENT asset allocation.

    Used by both the analysis pipeline (_assemble_financial_inputs) and the
    report (_build_client_facts) so the equity % is identical on every page.

    Precedence (Manual Investment Override toggle):
      - toggle ON  -> manual grid wins over CAS (override exists because CAS can
                      be inaccurate).
      - toggle OFF -> CAS holdings authoritative; manual only fills missing fields.
      - no CAS     -> manual grid used as-is.
    The standalone equity-% field is a last-resort fallback (it is prefilled with
    the recommended target, so it never overrides an actual holding).
    """
    lifestyle = lifestyle or {}
    manual = lifestyle.get("allocation") or {}
    use_manual = bool(lifestyle.get("use_manual_overrides"))
    portfolio = dict(cas_portfolio or {})
    if not portfolio:
        portfolio.update(manual)
    elif use_manual:
        for k, v in manual.items():
            if v not in (None, "", 0):
                portfolio[k] = v
    else:
        for k, v in manual.items():
            if portfolio.get(k) in (None, "", 0):
                portfolio[k] = v
    rp_equity = _safe_float((risk or {}).get("equity_allocation_percent"), 0.0)
    if rp_equity > 0 and portfolio.get("equity") in (None, "", 0):
        portfolio["equity"] = rp_equity
        if portfolio.get("debt") in (None, "", 0):
            portfolio["debt"] = max(0.0, 100.0 - rp_equity)
    return portfolio


def _assemble_financial_inputs(q: dict, doc_insights=None) -> dict:
    personal = q.get("personal_info") or {}
    family = q.get("family_info") or {}
    goals = q.get("goals") or {}
    risk = q.get("risk_profile") or {}
    insurance = q.get("insurance") or {}
    lifestyle = q.get("lifestyle") or {}

    children = family.get("children") or []
    other_dependents = family.get("dependents") or []
    has_financial_dependents = bool(
        children
        or other_dependents
        or family.get("has_financial_dependents")
        or family.get("spouse")
    )
    # Resolve current allocation with the same CAS-vs-manual precedence the report
    # uses, so the equity % feeding risk/recommendations matches every report page.
    allocation = _resolve_current_allocation(lifestyle, risk, (doc_insights or {}).get("portfolio"))

    payload = {
        "personal": {
            "age": personal.get("age"),
            "name": personal.get("name"),
            "has_financial_dependents": has_financial_dependents,
        },
        "income": {
            "annualIncome": lifestyle.get("annual_income"),
            "monthlyExpenses": lifestyle.get("monthly_expenses"),
            "monthlyEmi": lifestyle.get("monthly_emi"),
        },
        "goals": {
            "goalHorizon": risk.get("primary_horizon"),
            "items": goals.get("items") or [],
        },
        "risk": {
            "tolerance": risk.get("tolerance"),
            # Advanced risk raw inputs (persisted in risk_profile JSON)
            "primary_horizon": risk.get("primary_horizon"),
            "primary_horizon_years": risk.get("primary_horizon_years"),
            "goal_tenure_years": risk.get("goal_tenure_years"),
            "goal_flexibility": risk.get("goal_flexibility"),
            "goal_importance": risk.get("goal_importance"),
            "behavior": risk.get("behavior"),
            "loss_tolerance_percent": risk.get("loss_tolerance_percent"),
            "income_stability": risk.get("income_stability"),
            "emergency_fund_months": risk.get("emergency_fund_months"),
        },
        "insurance": {
            "lifeCover": insurance.get("life_cover"),
            "healthCover": insurance.get("health_cover"),
        },
        "savings": {
            "savingsPercent": lifestyle.get("savings_percent"),
            "savingsBand": lifestyle.get("savings_band"),
        },
        "investments": {
            "current": lifestyle.get("products") or [],
            "allocation": allocation,
        },
        "emergencyFundAmount": lifestyle.get("emergency_fund"),
        # Manual overrides (used when CAS data is absent)
        "manual_sip": lifestyle.get("manual_sip"),
        "manual_corpus": lifestyle.get("manual_corpus"),
    }

    # Merge document-derived signals if questionnaire fields are missing
    di = doc_insights or {}
    bank = di.get("bank") or {}
    try:
        # Annual income fallback from bank inflows — ONLY the period-normalized
        # annual estimate. Raw total_inflows is a statement-period sum, not annual,
        # so using it would grossly understate income for short statements.
        if not payload["income"].get("annualIncome"):
            inflow = bank.get("annual_inflows_est")
            if isinstance(inflow, (int, float)) and inflow > 0:
                payload["income"]["annualIncome"] = inflow
        # Monthly expenses fallback from bank outflows
        if not payload["income"].get("monthlyExpenses"):
            monthly_out = bank.get("monthly_outflows")
            outflow = bank.get("total_outflows")
            if isinstance(monthly_out, (int, float)) and monthly_out > 0:
                payload["income"]["monthlyExpenses"] = round(monthly_out, 2)
            elif isinstance(outflow, (int, float)) and outflow > 0:
                payload["income"]["monthlyExpenses"] = round(outflow / 12.0, 2)
        # Savings percent fallback from net cashflow
        if not payload["savings"].get("savingsPercent"):
            inflow = bank.get("total_inflows")
            outflow = bank.get("total_outflows")
            if isinstance(inflow, (int, float)) and inflow > 0 and isinstance(outflow, (int, float)):
                sp = max(0.0, (inflow - outflow)) / inflow * 100.0
                sp = max(0.0, min(100.0, sp))
                payload["savings"]["savingsPercent"] = round(sp, 2)
    except Exception:
        pass

    # Merge insurance covers from document insights when questionnaire values are absent
    try:
        ins = di.get("insurance") or {}
        sum_val = ins.get("sum_assured_or_insured")
        ins_type = str(ins.get("insurance_type") or "").lower()
        if isinstance(sum_val, (int, float)) and sum_val > 0:
            if not payload["insurance"].get("lifeCover") and ("life" in ins_type or "term" in ins_type or "ulip" in ins_type):
                payload["insurance"]["lifeCover"] = float(sum_val)
            if not payload["insurance"].get("healthCover") and ("health" in ins_type or "mediclaim" in ins_type):
                payload["insurance"]["healthCover"] = float(sum_val)
            # Unknown type: default to life cover if both missing
            if not payload["insurance"].get("lifeCover") and not payload["insurance"].get("healthCover"):
                payload["insurance"]["lifeCover"] = float(sum_val)
    except Exception:
        pass

    # Merge portfolio allocation from document insights (CAS) when missing
    try:
        port = di.get("portfolio") or {}
        alloc = payload.get("investments", {}).get("allocation") or {}
        def _set_if_missing(key, src_key):
            if alloc.get(key) in (None, "", 0):
                v = port.get(src_key)
                if isinstance(v, (int, float)) and v >= 0:
                    alloc[key] = float(v)
        _set_if_missing("equity", "equity")
        _set_if_missing("debt", "debt")
        _set_if_missing("gold", "gold")
        _set_if_missing("realEstate", "realEstate")
        _set_if_missing("insuranceLinked", "insuranceLinked")
        _set_if_missing("cash", "cash")
        payload["investments"]["allocation"] = alloc
    except Exception:
        pass

    return payload


def _build_client_facts(q: dict, analysis: dict, doc_insights=None) -> dict:
    personal = q.get("personal_info") or {}
    family = q.get("family_info") or {}
    lifestyle = q.get("lifestyle") or {}
    insurance = q.get("insurance") or {}
    tax_info = q.get("tax_info") or {}
    goals_data = q.get("goals") or {}
    goals = goals_data.get("items") if isinstance(goals_data.get("items"), list) else []

    # Calculate dependents - children automatically count as financial dependents
    children = family.get("children") or []
    other_dependents = family.get("dependents") or []
    has_spouse = bool(family.get("spouse"))
    dependents_count = (1 if has_spouse else 0) + len(children) + len(other_dependents)
    
    # Financial dependents: children or explicitly marked dependents
    # If user has children, they automatically have financial dependents
    has_financial_dependents = len(children) > 0 or len(other_dependents) > 0 or family.get("has_financial_dependents", False)
    
    di = doc_insights or {}
    bank = di.get("bank") or {}
    portfolio = dict(di.get("portfolio") or {})
    itr = di.get("itr") or {}
    
    # ITR fallback if tax_info is missing
    if not itr and tax_info:
        deductions_claimed = []
        if tax_info.get("deductions_80c"):
            deductions_claimed.append({"section": "80C", "amount": _safe_float(tax_info["deductions_80c"], 0)})
        if tax_info.get("deductions_80d"):
            deductions_claimed.append({"section": "80D", "amount": _safe_float(tax_info["deductions_80d"], 0)})
        if tax_info.get("deductions_24c"):
            deductions_claimed.append({"section": "24C", "amount": _safe_float(tax_info["deductions_24c"], 0)})
        if tax_info.get("deductions_80dd"):
            deductions_claimed.append({"section": "80DD", "amount": _safe_float(tax_info["deductions_80dd"], 0)})
            
        itr = {
            "gross_total_income": _safe_float((q.get("lifestyle") or {}).get("annual_income"), 0),
            "total_tax_paid": _safe_float(tax_info.get("total_tax_paid"), 0),
            "deductions_claimed": deductions_claimed,
            "tax_regime": tax_info.get("tax_regime", "old").lower(),
        }
    
    # Resolve current allocation with the shared precedence rule (same one the
    # analysis pipeline uses), honoring the Manual Investment Override toggle.
    use_manual_overrides = bool(lifestyle.get("use_manual_overrides"))
    portfolio = _resolve_current_allocation(lifestyle, q.get("risk_profile") or {}, portfolio)

    # Get CAS data for SIP commitments
    qid = q.get("id")
    cas_data = _get_cas_data_for_questionnaire(qid) if qid else {}
    
    # Calculate total monthly SIP from CAS SIP details
    total_monthly_sip = 0.0
    sip_details = cas_data.get("sip_details") or []
    for sip in sip_details:
        freq = (sip.get("frequency") or "Monthly").lower()
        amount = sip.get("sip_amount", 0)
        if freq == "monthly" and isinstance(amount, (int, float)):
            total_monthly_sip += float(amount)
    
    # Enrich portfolio with SIP data
    enriched_portfolio = dict(portfolio)
    manual_sip = _safe_float(lifestyle.get("manual_sip"), 0.0)
    manual_corpus = _safe_float(lifestyle.get("manual_corpus"), 0.0)
    if use_manual_overrides and manual_sip > 0:
        total_monthly_sip = manual_sip          # override toggled -> manual wins over CAS
    elif total_monthly_sip <= 0 and manual_sip > 0:
        total_monthly_sip = manual_sip          # fallback when CAS has no SIP
    enriched_portfolio["total_monthly_sip"] = total_monthly_sip
    enriched_portfolio["sip_count"] = len(sip_details)
    if manual_corpus > 0:
        if use_manual_overrides:
            enriched_portfolio["current_value"] = manual_corpus   # override toggled -> manual wins
            enriched_portfolio["total_value"] = manual_corpus
        else:
            enriched_portfolio.setdefault("current_value", manual_corpus)
            enriched_portfolio.setdefault("total_value", manual_corpus)
    
    # Calculate retirement planning if enabled
    age = personal.get("age")
    annual_income = lifestyle.get("annual_income")
    monthly_income = _safe_float(annual_income, 0.0) / 12.0 if annual_income else 0.0
    monthly_expenses = _safe_float(lifestyle.get("monthly_expenses"), 0.0)
    
    retirement_planning = None
    if goals_data.get("wants_retirement_planning"):
        desired_pension = (
            goals_data.get("desired_monthly_pension")
            or goals_data.get("expected_pension")
            or lifestyle.get("expected_pension")
        )
        retirement_planning = compute_retirement_corpus(
            age=age,
            monthly_income=monthly_income,
            desired_monthly_pension=desired_pension
        )
        retirement_planning["enabled"] = True
        retirement_planning["monthly_expenses"] = monthly_expenses
    
    # Calculate term insurance requirement if has financial dependents
    term_insurance = None
    if has_financial_dependents and monthly_income > 0:
        required_cover = compute_term_insurance_need(age, monthly_income)
        current_life_cover = _safe_float(insurance.get("life_cover"), 0.0)
        term_insurance = {
            "has_financial_dependents": True,
            "required_cover": required_cover,
            "current_cover": current_life_cover,
            "gap": max(0, required_cover - current_life_cover),
            "is_adequate": current_life_cover >= required_cover,
        }

    facts = {
        "questionnaire_id": q.get("id"),
        "personal": {
            "name": personal.get("name"),
            "age": personal.get("age"),
            "dependents_count": dependents_count,
            "has_financial_dependents": has_financial_dependents,
        },
        "income": {
            "annualIncome": lifestyle.get("annual_income"),
            "monthlyExpenses": lifestyle.get("monthly_expenses"),
            "monthlyEmi": lifestyle.get("monthly_emi"),
        },
        "insurance": {
            "lifeCover": insurance.get("life_cover"),
            "healthCover": insurance.get("health_cover"),
            "insurerName": insurance.get("insurer_name") or (di.get("insurance") or {}).get("insurer_name"),
            "insuranceType": insurance.get("insurance_type") or (di.get("insurance") or {}).get("insurance_type"),
        },
        "savings": {
            "savingsPercent": lifestyle.get("savings_percent"),
        },
        "lifestyle": {
            "expected_pension": lifestyle.get("expected_pension"),
            "manual_sip": lifestyle.get("manual_sip"),
            "manual_corpus": lifestyle.get("manual_corpus"),
            "emergency_fund": lifestyle.get("emergency_fund"),
            "available_savings": lifestyle.get("available_savings"),
        },
        "goals": [
            {
                "name": (g.get("name") or g.get("goal")),
                "target_amount": g.get("target_amount"),
                "horizon_years": g.get("horizon_years") or g.get("horizon"),
                "risk_tolerance": g.get("risk_tolerance", "medium"),
                "goal_importance": g.get("goal_importance", "important"),
                "goal_flexibility": g.get("goal_flexibility", "fixed"),
                "behavior": g.get("behavior", "hold"),
            }
            for g in goals
            if isinstance(g, dict)
        ],
        "bank": {
            "total_inflows": bank.get("total_inflows"),
            "total_outflows": bank.get("total_outflows"),
            "net_cashflow": bank.get("net_cashflow"),
            "opening_balance": bank.get("opening_balance"),
            "closing_balance": bank.get("closing_balance"),
        },
        "portfolio": enriched_portfolio,
        "extracts": di.get("raw_extracts"),
        "analysis": analysis,
        "retirement_planning": retirement_planning,
        "term_insurance": term_insurance,
        "tax": tax_info,
        "itr": itr,  # Full ITR data for tax optimization section
        "estate": q.get("estate") or {},
    }
    return facts
