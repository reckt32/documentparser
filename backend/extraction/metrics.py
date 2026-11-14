"""Persist extracted document values as queryable metrics."""

from backend.db import delete_metrics_for_doc_keys, insert_metric
from backend.utils import clean_and_convert_to_float


def _persist_metrics_for_doc(document_id: int, data: dict):
    """
    Persist key numeric insights from extracted data into metrics for aggregation/prefill.
    Keys stored:
      - opening_balance, closing_balance, total_inflows, total_outflows (bank)
      - gross_total_income, taxable_income, total_tax_paid (ITR)
      - sum_assured_or_insured (insurance)
      - portfolio_equity/debt/gold/realEstate/insuranceLinked/cash (CAS allocation)
    """
    if not isinstance(data, dict):
        return

    def n(v):
        try:
            if isinstance(v, (int, float)):
                return float(v)
            if v in (None, "", "N/A"):
                return None
            cv = clean_and_convert_to_float(str(v))
            return cv if cv != "N/A" else None
        except Exception:
            return None

    to_store = {}

    # Bank account summary
    acct = data.get("account_summary") or {}
    for k in ["opening_balance", "closing_balance", "total_inflows", "total_outflows", "statement_months"]:
        v = acct.get(k)
        nv = n(v)
        if nv is not None:
            to_store[k] = nv

    # ITR numbers (top-level or under tax_computation)
    itr_top = {k: data.get(k) for k in ["gross_total_income", "taxable_income", "total_tax_paid"]}
    tax_comp = data.get("tax_computation") or {}
    for k in ["gross_total_income", "taxable_income", "total_tax_paid"]:
        v = itr_top.get(k, None)
        if v in (None, "", "N/A"):
            v = tax_comp.get(k)
        nv = n(v)
        if nv is not None:
            to_store[k] = nv

    # Insurance
    ins_sum = data.get("sum_assured_or_insured")
    ins_nv = n(ins_sum)
    if ins_nv is not None:
        to_store["sum_assured_or_insured"] = ins_nv

    # CAS allocation
    alloc = data.get("asset_allocation") or {}
    mapping = {
        "equity_percentage": "portfolio_equity",
        "debt_percentage": "portfolio_debt",
        "gold_percentage": "portfolio_gold",
        "real_estate_percentage": "portfolio_realEstate",
        "insurance_linked_percentage": "portfolio_insuranceLinked",
        "cash_percentage": "portfolio_cash",
        # Support alternate keys if present
        "equity": "portfolio_equity",
        "debt": "portfolio_debt",
        "gold": "portfolio_gold",
        "realEstate": "portfolio_realEstate",
        "insuranceLinked": "portfolio_insuranceLinked",
        "cash": "portfolio_cash",
    }
    for src, dst in mapping.items():
        if src in alloc:
            nv = n(alloc.get(src))
            if nv is not None:
                to_store[dst] = nv

    if to_store:
        try:
            delete_metrics_for_doc_keys(document_id, list(to_store.keys()))
        except Exception:
            pass
        for k, v in to_store.items():
            try:
                insert_metric(document_id, k, v, None)
            except Exception:
                continue
