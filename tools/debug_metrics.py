import sys, json
sys.path.insert(0, 'backend')

import db, extractors

def main():
    docs = db._query("SELECT id, filename FROM documents ORDER BY id")
    print("docs:", [(d["id"], d["filename"]) for d in docs])

    total_before = db._query("SELECT COUNT(*) AS c FROM metrics")[0]["c"]
    print("metrics_before:", total_before)

    for d in docs:
        doc_id = d["id"]
        print("\n=== DOC", doc_id, d["filename"], "===")
        try:
            tbl = extractors._parse_statement_summary_from_tables(doc_id)
            print("table_stmt_summary:", tbl)
        except Exception as e:
            print("table_stmt_summary_err:", e)

        try:
            ptotal = extractors._parse_portfolio_grand_total(doc_id)
            print("portfolio_total:", ptotal)
        except Exception as e:
            print("portfolio_total_err:", e)

        try:
            res = extractors.extract_and_store_from_indexed(doc_id)
            print("extract_and_store:", json.dumps(res))
        except Exception as e:
            print("extract_and_store_err:", e)

        cnt = db._query("SELECT COUNT(*) AS c FROM metrics WHERE document_id=?", (doc_id,))[0]["c"]
        print("metrics_count_for_doc:", doc_id, cnt)
        rows = db._query("SELECT key, value_num FROM metrics WHERE document_id=? ORDER BY key", (doc_id,))
        print("metric_rows:", [(r["key"], r["value_num"]) for r in rows])

    total_after = db._query("SELECT COUNT(*) AS c FROM metrics")[0]["c"]
    print("\nmetrics_after:", total_after)

if __name__ == "__main__":
    main()
