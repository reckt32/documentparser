"""Aggregate per-document extractions into questionnaire-level insights and prefill."""

import json
import logging
from datetime import datetime
from typing import Dict

from backend.db import list_metrics, list_questionnaire_uploads
from backend.extraction.extractors import extract_and_store_from_indexed

logger = logging.getLogger(__name__)


# --- Document insights aggregation (from linked uploads) ---
def aggregate_doc_insights_for_questionnaire(qid: int) -> dict:
    """
    Aggregate insights using deterministic extractors from all documents linked to the questionnaire.
    Combines:
      - DB metrics (numeric aggregations for bank/CAS/ITR/insurance)
      - Indexed section/table-driven summaries via extract_and_store_from_indexed(document_id)
    Returns:
      {
        "bank": {...},                   # totals + opening/closing + net_cashflow
        "portfolio": {...},              # CAS allocation percentages if present
        "insurance": {"sum_assured_or_insured": number}?,
        "itr": {"gross_total_income": n, "taxable_income": n, "total_tax_paid": n}?,
        "raw_extracts": [                # per-document extracted summaries from index
          {
            "document_id": int,
            "summary": {
              "investment_snapshot": {...} | None,
              "account_summary": {...} | None,
              "portfolio_summary": {...} | None,
              "provenance": {...}
            }
          },
          ...
        ]
      }
    """
    # _query returns sqlite3.Row objects, which have no .get(); normalize to dicts
    uploads = [dict(u) for u in (list_questionnaire_uploads(qid) or [])]
    doc_ids = [r["document_id"] for r in uploads if r["document_id"] is not None]

    # Aggregates
    bank = {"total_inflows": 0.0, "total_outflows": 0.0, "opening_balance": None, "closing_balance": None}
    portfolio_alloc = {}
    insurance = {}
    itr = {}

    per_doc_extracts = []

    # Extract metadata_json fields that are not stored as numeric metrics.
    for upload in uploads:
        doc_type = str(upload.get("doc_type") or "").lower()
        metadata_json = upload.get("metadata_json")
        if not metadata_json:
            continue
        try:
            metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        except Exception as e:
            logger.warning(f"Error parsing upload metadata: {e}")
            continue

        if "insurance" in doc_type:
            for key in ("insurance_type", "insurer_name", "policy_number", "policy_holder"):
                value = metadata.get(key)
                if value not in (None, "", "N/A") and not insurance.get(key):
                    insurance[key] = value

        if "itr" in doc_type:
            try:
                # Merge full ITR extracted data into itr dict.
                if metadata.get("deductions_claimed"):
                    itr["deductions_claimed"] = metadata["deductions_claimed"]
                if metadata.get("income_sources"):
                    itr["income_sources"] = metadata["income_sources"]
                if metadata.get("tax_computation"):
                    itr["tax_computation"] = metadata["tax_computation"]
                if metadata.get("carry_forward_losses"):
                    itr["carry_forward_losses"] = metadata["carry_forward_losses"]
                if metadata.get("assets_and_liabilities"):
                    itr["assets_and_liabilities"] = metadata["assets_and_liabilities"]
                if metadata.get("gross_total_income") and not itr.get("gross_total_income"):
                    try:
                        itr["gross_total_income"] = float(metadata["gross_total_income"])
                    except (ValueError, TypeError):
                        pass
                if metadata.get("taxable_income") and not itr.get("taxable_income"):
                    try:
                        itr["taxable_income"] = float(metadata["taxable_income"])
                    except (ValueError, TypeError):
                        pass
                if metadata.get("total_tax_paid") and not itr.get("total_tax_paid"):
                    try:
                        itr["total_tax_paid"] = float(metadata["total_tax_paid"])
                    except (ValueError, TypeError):
                        pass
            except Exception as e:
                logger.warning(f"Error parsing ITR metadata: {e}")
                continue

    # Period-normalized cashflow estimates (each doc treated as a distinct
    # account; only emitted when every doc with flows declares its coverage)
    monthly_in_sum = 0.0
    monthly_out_sum = 0.0
    months_max = 0.0
    normalization_complete = True

    for did in doc_ids:
        # 1) Deterministic summaries from indexed sections/tables
        try:
            idx_summary = extract_and_store_from_indexed(did) or {}
            per_doc_extracts.append({"document_id": did, "summary": idx_summary})
            # Merge account summary for opening/closing only to avoid double counting with metrics
            acct = idx_summary.get("account_summary") or {}
            try:
                ob = acct.get("opening_balance")
                cb = acct.get("closing_balance")
                if bank["opening_balance"] is None and isinstance(ob, (int, float)):
                    bank["opening_balance"] = float(ob)
                if bank["closing_balance"] is None and isinstance(cb, (int, float)):
                    bank["closing_balance"] = float(cb)
            except Exception:
                pass
            # No direct allocation percentages in idx_summary; keep for narratives via facts
        except Exception:
            # If extraction fails for any doc, continue with metrics-only for that doc
            pass

        # 2) Numeric metrics for CAS allocation/ITR/insurance/bank (authoritative numeric store)
        try:
            mets = list_metrics(did) or []
        except Exception:
            mets = []
        doc_inflows = None
        doc_outflows = None
        doc_months = None
        for m in mets:
            md = dict(m)
            k = (md.get("key") or "").strip().lower()
            vnum = md.get("value_num")
            try:
                if k in ("total_inflows", "total_outflows"):
                    if vnum is not None:
                        if k == "total_inflows":
                            bank["total_inflows"] += float(vnum)
                            doc_inflows = float(vnum)
                        else:
                            bank["total_outflows"] += float(vnum)
                            doc_outflows = float(vnum)
                elif k == "statement_months":
                    if vnum is not None and float(vnum) > 0:
                        doc_months = float(vnum)
                elif k in ("opening_balance", "closing_balance"):
                    if vnum is not None:
                        bank[k] = float(vnum)
                elif k.startswith("portfolio_"):
                    if vnum is not None:
                        portfolio_alloc[k.replace("portfolio_", "")] = float(vnum)
                elif k in ("insurance_sum_assured_or_insured", "sum_assured_or_insured"):
                    if vnum is not None:
                        insurance["sum_assured_or_insured"] = float(vnum)
                elif k in ("gross_total_income", "taxable_income", "total_tax_paid"):
                    if vnum is not None:
                        itr[k] = float(vnum)
            except Exception:
                continue

        # Accumulate period-normalized flows for this doc
        if doc_months:
            if doc_inflows is not None:
                monthly_in_sum += doc_inflows / doc_months
            if doc_outflows is not None:
                monthly_out_sum += doc_outflows / doc_months
            months_max = max(months_max, doc_months)
        elif doc_inflows is not None or doc_outflows is not None:
            # A doc contributed flows without declaring coverage: estimates
            # would be wrong, so suppress them and let consumers fall back.
            normalization_complete = False

    # Compute net cashflow
    try:
        inflow = float(bank.get("total_inflows") or 0.0)
        outflow = float(bank.get("total_outflows") or 0.0)
        bank["net_cashflow"] = inflow - outflow
    except Exception:
        bank["net_cashflow"] = None

    # Period-normalized estimates (additive keys; only when coverage is known
    # for every contributing doc, so legacy consumers see no change otherwise)
    if normalization_complete and (monthly_in_sum > 0 or monthly_out_sum > 0):
        bank["months_covered"] = months_max
        if monthly_in_sum > 0:
            bank["monthly_inflows"] = round(monthly_in_sum, 2)
            bank["annual_inflows_est"] = round(monthly_in_sum * 12.0, 2)
        if monthly_out_sum > 0:
            bank["monthly_outflows"] = round(monthly_out_sum, 2)
            bank["annual_outflows_est"] = round(monthly_out_sum * 12.0, 2)

    out: Dict[str, dict] = {}
    if any(v not in (None, 0.0) for v in bank.values()):
        out["bank"] = bank
    if portfolio_alloc:
        out["portfolio"] = portfolio_alloc
    if insurance:
        out["insurance"] = insurance
    if itr:
        out["itr"] = itr
    if per_doc_extracts:
        out["raw_extracts"] = per_doc_extracts
    return out


def _get_cas_data_for_questionnaire(qid: int) -> dict:
    """
    Retrieve CAS data from uploaded Mutual Fund CAS documents.
    Returns dict with sip_details, transaction_summary, asset_allocation or empty dict.
    """
    uploads = list_questionnaire_uploads(qid) or []

    for upload in uploads:
        if upload["doc_type"] == "Mutual fund CAS (Consolidated Account Statement)":
            metadata_json = upload["metadata_json"]
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                    cas_data = metadata.get("cas_data", {})
                    if cas_data:
                        return cas_data
                except Exception as e:
                    logger.warning(f"Error parsing CAS metadata: {e}")
                    continue

    return {}


def build_prefill_from_insights(qid: int) -> dict:
    """
    Derive questionnaire prefill values from linked uploads (Bank/CAS/ITR/Insurance).
    Prefill rules:
      - Lifestyle:
          annual_income := ITR.gross_total_income if present else Bank.total_inflows
          monthly_expenses := Bank.total_outflows / 12
          savings_percent := max(0, inflows - outflows)/inflows * 100
      - Allocation:
          from CAS portfolio_* metrics (equity, debt, gold, realEstate, insuranceLinked, cash)
      - Insurance:
          life_cover := sum_assured_or_insured if insurance_type suggests life
          health_cover := sum_assured_or_insured if insurance_type suggests health
    """
    di = aggregate_doc_insights_for_questionnaire(qid) or {}
    bank = di.get("bank") or {}
    portfolio = di.get("portfolio") or {}
    itr = di.get("itr") or {}
    ins = di.get("insurance") or {}

    lifestyle = {}
    try:
        # Annual income prefers ITR; monthly expenses from bank; savings% strictly from bank flows
        itr_income = itr.get("gross_total_income")
        bank_inflow = bank.get("total_inflows")
        bank_outflow = bank.get("total_outflows")
        # Period-normalized estimates (set only when statement coverage is known)
        annual_in_est = bank.get("annual_inflows_est")
        monthly_out_est = bank.get("monthly_outflows")

        if isinstance(itr_income, (int, float)) and itr_income > 0:
            inflow = itr_income
        elif isinstance(annual_in_est, (int, float)) and annual_in_est > 0:
            inflow = annual_in_est
        else:
            # Raw statement-period inflows are NOT an annual figure — a 2-3 month
            # statement understates income several-fold. Only prefill annual income
            # from ITR or a period-normalized annual estimate; otherwise leave it
            # blank for the user to enter rather than seed a wrong value.
            inflow = None
        outflow = bank_outflow

        if isinstance(inflow, (int, float)) and inflow > 0:
            lifestyle["annual_income"] = round(float(inflow), 2)
        if isinstance(monthly_out_est, (int, float)) and monthly_out_est > 0:
            lifestyle["monthly_expenses"] = round(float(monthly_out_est), 2)
        elif isinstance(outflow, (int, float)) and outflow > 0:
            # Legacy fallback: assume a 12-month statement
            lifestyle["monthly_expenses"] = round(float(outflow) / 12.0, 2)
        if isinstance(bank_inflow, (int, float)) and bank_inflow > 0 and isinstance(bank_outflow, (int, float)):
            sp = max(0.0, (float(bank_inflow) - float(bank_outflow))) / float(bank_inflow) * 100.0
            sp = max(0.0, min(100.0, sp))
            lifestyle["savings_percent"] = round(sp, 2)
    except Exception:
        pass

    # Extract monthly EMI from bank statement recurring debits
    try:
        uploads = list_questionnaire_uploads(qid) or []
        total_monthly_emi = 0.0
        # Expanded EMI keywords including bank-specific codes
        emi_keywords = [
            "emi", "loan", "ln", "mortgage", "instalment", "installment", "repayment",
            "home loan", "car loan", "personal loan", "vehicle loan", "housing loan", "education loan",
            # Bank-specific codes
            "elm",  # ICICI EMI/Loan
            "nach",  # National Automated Clearing House (auto-debit EMIs)
            "hdfc ln", "sbi ln", "icici ln", "axis ln", "kotak ln",  # Bank loan prefixes
            "si/",  # Standing Instruction
            "standing instruction",
            "ecs",  # Electronic Clearing Service
            "auto debit", "autodebit",
            "emi ded",  # SBI pattern
            "bil/inft/elm",  # ICICI pattern
        ]
        for upload in uploads:
            if (upload["doc_type"] or "").lower() == "bank statement":
                metadata_json = upload["metadata_json"]
                if metadata_json:
                    try:
                        metadata = json.loads(metadata_json)
                        bank_data = metadata.get("bank_data") or {}
                        recurring_debits = bank_data.get("recurring_debits") or []
                        for debit in recurring_debits:
                            desc = (debit.get("description") or "").lower()
                            amount = debit.get("amount")
                            freq = (debit.get("frequency") or "").lower()
                            is_emi = debit.get("is_emi", False)
                            
                            # Check if this is an EMI payment:
                            # 1. LLM marked it as EMI (is_emi=True)
                            # 2. OR description matches EMI keywords
                            is_emi_payment = is_emi or any(kw in desc for kw in emi_keywords)
                            
                            if is_emi_payment:
                                if isinstance(amount, (int, float)) and amount > 0:
                                    # Convert to monthly if not already monthly
                                    if freq in ("monthly", "month"):
                                        total_monthly_emi += float(amount)
                                    elif freq in ("quarterly", "quarter"):
                                        total_monthly_emi += float(amount) / 3.0
                                    elif freq in ("yearly", "annual", "year"):
                                        total_monthly_emi += float(amount) / 12.0
                                    else:
                                        # Assume monthly if frequency unclear
                                        total_monthly_emi += float(amount)
                    except Exception:
                        continue
        if total_monthly_emi > 0:
            lifestyle["monthly_emi"] = round(total_monthly_emi, 2)
    except Exception as e:
        pass  # Silently handle errors to avoid breaking prefill

    allocation = {}
    try:
        for k in ["equity", "debt", "gold", "realEstate", "insuranceLinked", "cash"]:
            v = portfolio.get(k)
            if isinstance(v, (int, float)) and v >= 0:
                allocation[k] = float(v)
    except Exception:
        pass

    # Scan uploads for insurance prefill - need to read insurance_type from metadata
    insurance_prefill = {}
    try:
        uploads = list_questionnaire_uploads(qid) or []
        for upload in uploads:
            doc_type = (upload["doc_type"] or "").lower()
            if "insurance" in doc_type:
                metadata_json = upload["metadata_json"]
                print(f"[Prefill] Found insurance upload, metadata_json: {metadata_json[:500] if metadata_json else 'None'}")
                if metadata_json:
                    try:
                        metadata = json.loads(metadata_json)
                        ins_type = str(metadata.get("insurance_type") or "").lower()
                        sum_val = metadata.get("sum_assured_or_insured")
                        
                        print(f"[Prefill] Parsed insurance_type: '{ins_type}', sum_assured_or_insured: {sum_val}")
                        
                        # Also check aggregated insights if not in metadata
                        if sum_val is None or sum_val == "N/A":
                            sum_val = ins.get("sum_assured_or_insured")
                            print(f"[Prefill] Fallback to aggregated insights: {sum_val}")
                        
                        if isinstance(sum_val, (int, float)) and sum_val > 0:
                            if "health" in ins_type or "mediclaim" in ins_type:
                                # Add to health cover (may have multiple health policies)
                                existing_health = insurance_prefill.get("health_cover", 0.0)
                                insurance_prefill["health_cover"] = existing_health + float(sum_val)
                                print(f"[Prefill] Added to health_cover: {sum_val}")
                            elif "life" in ins_type or "term" in ins_type or "ulip" in ins_type:
                                # Add to life cover
                                existing_life = insurance_prefill.get("life_cover", 0.0)
                                insurance_prefill["life_cover"] = existing_life + float(sum_val)
                                print(f"[Prefill] Added to life_cover: {sum_val}")
                            else:
                                # Unknown type: default to life_cover
                                print(f"[Prefill] Unknown insurance type '{ins_type}', defaulting to life_cover")
                                existing_life = insurance_prefill.get("life_cover", 0.0)
                                insurance_prefill["life_cover"] = existing_life + float(sum_val)
                    except Exception:
                        continue
        
        # Fallback: if no metadata found, use aggregated insights
        if not insurance_prefill:
            sum_val = ins.get("sum_assured_or_insured")
            if isinstance(sum_val, (int, float)) and sum_val > 0:
                # Default to life_cover when type unknown
                insurance_prefill["life_cover"] = float(sum_val)
    except Exception:
        pass

    # Personal info extraction - priority order:
    # 1. Bank statement account_holder_name (most reliable, directly from user's bank)
    # 2. CAS investor_name
    # 3. Insurance policy_holder
    # 4. ITR assessee_name (least reliable - may pick up father's name field)
    personal_info = {}
    try:
        uploads = list_questionnaire_uploads(qid) or []
        
        # First pass: look for bank statement (highest priority for name)
        for upload in uploads:
            if personal_info.get("name"):
                break
            doc_type = (upload["doc_type"] or "").lower()
            if "bank" in doc_type:
                metadata_json = upload["metadata_json"]
                if metadata_json:
                    try:
                        metadata = json.loads(metadata_json)
                        account_holder = metadata.get("account_holder_name")
                        if account_holder and account_holder != "N/A" and len(str(account_holder)) > 2:
                            personal_info["name"] = str(account_holder).strip().title()
                    except Exception:
                        continue
        
        # Second pass: look for CAS investor_name
        if not personal_info.get("name"):
            for upload in uploads:
                if personal_info.get("name"):
                    break
                doc_type = (upload["doc_type"] or "").lower()
                if "cas" in doc_type or "mutual fund" in doc_type:
                    metadata_json = upload["metadata_json"]
                    if metadata_json:
                        try:
                            metadata = json.loads(metadata_json)
                            cas_data = metadata.get("cas_data") or {}
                            investor_name = metadata.get("investor_name") or cas_data.get("investor_name")
                            if investor_name and investor_name != "N/A" and len(investor_name) > 2:
                                personal_info["name"] = investor_name.strip().title()
                        except Exception:
                            continue
        
        # Third pass: look for Insurance policy_holder
        if not personal_info.get("name"):
            for upload in uploads:
                if personal_info.get("name"):
                    break
                doc_type = (upload["doc_type"] or "").lower()
                if "insurance" in doc_type:
                    metadata_json = upload["metadata_json"]
                    if metadata_json:
                        try:
                            metadata = json.loads(metadata_json)
                            policy_holder = metadata.get("policy_holder")
                            if policy_holder and policy_holder != "N/A" and len(policy_holder) > 2:
                                personal_info["name"] = policy_holder.strip().title()
                        except Exception:
                            continue
        
        # Fourth pass: look for ITR assessee_name (lowest priority)
        if not personal_info.get("name"):
            for upload in uploads:
                if personal_info.get("name"):
                    break
                doc_type = (upload["doc_type"] or "").lower()
                if "itr" in doc_type:
                    metadata_json = upload["metadata_json"]
                    if metadata_json:
                        try:
                            metadata = json.loads(metadata_json)
                            assessee_name = metadata.get("assessee_name")
                            if assessee_name and assessee_name != "N/A" and len(assessee_name) > 2:
                                personal_info["name"] = assessee_name.strip().title()
                        except Exception:
                            continue
        
        # Extract age from date_of_birth (from any document)
        for upload in uploads:
            if personal_info.get("age"):
                break
            metadata_json = upload["metadata_json"]
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                    dob = metadata.get("date_of_birth")
                    if dob and dob != "N/A":
                        for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"]:
                            try:
                                birth_date = datetime.strptime(dob, fmt)
                                if birth_date.year < 100:  # Handle 2-digit years
                                    birth_date = birth_date.replace(year=birth_date.year + 1900)
                                today = datetime.now()
                                age = today.year - birth_date.year
                                if (today.month, today.day) < (birth_date.month, birth_date.day):
                                    age -= 1
                                if 18 <= age <= 100:
                                    personal_info["age"] = age
                                break
                            except ValueError:
                                continue
                except Exception:
                    continue
    except Exception:
        pass

    return {
        "questionnaire_id": qid,
        "docInsights": di,
        "personal_info": personal_info,
        "lifestyle": lifestyle,
        "allocation": allocation,
        "insurance": insurance_prefill,
    }
