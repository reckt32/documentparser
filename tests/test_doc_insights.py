import os
import json
import random

# Ensure imports resolve when running this file directly
if __name__ == "__main__":
    here = os.path.dirname(__file__)
    import sys
    root = os.path.dirname(here)  # backend root
    if root not in sys.path:
        sys.path.insert(0, root)

# Ensure API key is present to satisfy app.py import-time check (LLM not used in these tests)
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "test"

from app import (
    aggregate_doc_insights_for_questionnaire,
    build_prefill_from_insights,
)
from db import (
    create_questionnaire,
    upsert_document,
    link_questionnaire_upload,
    insert_metric,
    get_questionnaire,
)

def _rand_sha():
    return os.urandom(8).hex() + os.urandom(8).hex()

def _insert_bank_metrics(doc_id, inflows, outflows, opening=None, closing=None):
    if opening is not None:
        insert_metric(doc_id, "opening_balance", float(opening), None)
    if closing is not None:
        insert_metric(doc_id, "closing_balance", float(closing), None)
    insert_metric(doc_id, "total_inflows", float(inflows), None)
    insert_metric(doc_id, "total_outflows", float(outflows), None)

def _insert_portfolio_alloc(doc_id, equity, debt, gold=0.0, real_estate=0.0, insurance_linked=0.0, cash=0.0):
    insert_metric(doc_id, "portfolio_equity", float(equity), None)
    insert_metric(doc_id, "portfolio_debt", float(debt), None)
    if gold is not None:
        insert_metric(doc_id, "portfolio_gold", float(gold), None)
    if real_estate is not None:
        insert_metric(doc_id, "portfolio_realEstate", float(real_estate), None)
    if insurance_linked is not None:
        insert_metric(doc_id, "portfolio_insuranceLinked", float(insurance_linked), None)
    if cash is not None:
        insert_metric(doc_id, "portfolio_cash", float(cash), None)

def _insert_insurance(doc_id, sum_assured, insurance_type="Life Insurance"):
    insert_metric(doc_id, "sum_assured_or_insured", float(sum_assured), None)
    # insurance_type is not stored in metrics; aggregator will only pick numeric sum

def _insert_itr(doc_id, gti, ti, tax_paid):
    insert_metric(doc_id, "gross_total_income", float(gti), None)
    insert_metric(doc_id, "taxable_income", float(ti), None)
    insert_metric(doc_id, "total_tax_paid", float(tax_paid), None)

def main():
    # 1) Create questionnaire
    user_id = "test-user-" + os.urandom(4).hex()
    qid = create_questionnaire(user_id=user_id)

    # 2) Create docs and link uploads
    # Bank statement doc
    sha_bank = _rand_sha()
    doc_bank = upsert_document(sha_bank, "bank.pdf", 1)
    link_questionnaire_upload(qid, doc_bank, sha_bank, "Bank statement", "bank.pdf", metadata={"source": "unit_test"})
    _insert_bank_metrics(doc_bank, inflows=1200000.0, outflows=900000.0, opening=10000.0, closing=13000.0)

    # CAS doc (portfolio allocation)
    sha_cas = _rand_sha()
    doc_cas = upsert_document(sha_cas, "cas.pdf", 3)
    link_questionnaire_upload(qid, doc_cas, sha_cas, "Mutual fund CAS (Consolidated Account Statement)", "cas.pdf", metadata={"source": "unit_test"})
    _insert_portfolio_alloc(doc_cas, equity=60.0, debt=30.0, gold=10.0, real_estate=0.0, insurance_linked=0.0, cash=0.0)

    # Insurance doc
    sha_ins = _rand_sha()
    doc_ins = upsert_document(sha_ins, "insurance.pdf", 2)
    link_questionnaire_upload(qid, doc_ins, sha_ins, "Insurance document", "insurance.pdf", metadata={"source": "unit_test"})
    _insert_insurance(doc_ins, sum_assured=5000000.0)

    # ITR doc
    sha_itr = _rand_sha()
    doc_itr = upsert_document(sha_itr, "itr.pdf", 5)
    link_questionnaire_upload(qid, doc_itr, sha_itr, "ITR", "itr.pdf", metadata={"source": "unit_test"})
    _insert_itr(doc_itr, gti=1500000.0, ti=1200000.0, tax_paid=150000.0)

    # 3) Aggregate insights
    di = aggregate_doc_insights_for_questionnaire(qid)

    # Basic assertions
    assert isinstance(di, dict), "docInsights should be a dict"
    assert "bank" in di and isinstance(di["bank"], dict), "bank section missing"
    assert "portfolio" in di and isinstance(di["portfolio"], dict), "portfolio section missing"
    assert "insurance" in di and isinstance(di["insurance"], dict), "insurance section missing"
    assert "itr" in di and isinstance(di["itr"], dict), "itr section missing"

    # Validate bank math
    b = di["bank"]
    assert abs(b.get("total_inflows", 0.0) - 1200000.0) < 1e-6, "inflows mismatch"
    assert abs(b.get("total_outflows", 0.0) - 900000.0) < 1e-6, "outflows mismatch"
    assert abs(b.get("net_cashflow", 0.0) - 300000.0) < 1e-6, "net cashflow mismatch"

    # Validate portfolio allocation
    p = di["portfolio"]
    assert abs(p.get("equity", 0.0) - 60.0) < 1e-6, "equity alloc mismatch"
    assert abs(p.get("debt", 0.0) - 30.0) < 1e-6, "debt alloc mismatch"
    assert abs(p.get("gold", 0.0) - 10.0) < 1e-6, "gold alloc mismatch"

    # Validate insurance
    ins = di["insurance"]
    assert abs(ins.get("sum_assured_or_insured", 0.0) - 5000000.0) < 1e-3, "sum assured mismatch"

    # Validate ITR
    itr = di["itr"]
    assert abs(itr.get("gross_total_income", 0.0) - 1500000.0) < 1e-6, "gross total income mismatch"
    assert abs(itr.get("taxable_income", 0.0) - 1200000.0) < 1e-6, "taxable income mismatch"
    assert abs(itr.get("total_tax_paid", 0.0) - 150000.0) < 1e-6, "tax paid mismatch"

    # 4) Prefill suggestions
    prefill = build_prefill_from_insights(qid)
    assert isinstance(prefill, dict), "prefill should be dict"
    lifestyle = prefill.get("lifestyle") or {}
    alloc = prefill.get("allocation") or {}
    insurance_prefill = prefill.get("insurance") or {}

    # Prefill validations
    assert abs(lifestyle.get("annual_income", 0.0) - 1500000.0) < 1e-6, "prefill annual_income from ITR mismatch"
    expected_me = 900000.0 / 12.0
    assert abs(lifestyle.get("monthly_expenses", 0.0) - expected_me) < 1e-6, "prefill monthly_expenses mismatch"
    expected_sp = (1200000.0 - 900000.0) / 1200000.0 * 100.0
    assert abs(lifestyle.get("savings_percent", 0.0) - expected_sp) < 1e-6, "prefill savings_percent mismatch"

    assert abs(alloc.get("equity", 0.0) - 60.0) < 1e-6, "prefill allocation equity mismatch"
    assert abs(alloc.get("debt", 0.0) - 30.0) < 1e-6, "prefill allocation debt mismatch"

    # Insurance prefill defaults to life_cover if type unknown
    assert abs(insurance_prefill.get("life_cover", 0.0) - 5000000.0) < 1e-3, "prefill life_cover mismatch"

    # Save outputs for inspection
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_doc_insights.json")
    with open(out_path, "w", encoding="utf-8") as wf:
        json.dump({"docInsights": di, "prefill": prefill}, wf, ensure_ascii=False, indent=2)

    print(json.dumps({"docInsights": di, "prefill": prefill}, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")
    print("All assertions passed.")

if __name__ == "__main__":
    main()
