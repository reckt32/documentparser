import os
import json

# Ensure imports resolve when running this file directly
if __name__ == "__main__":
    here = os.path.dirname(__file__)
    import sys
    if here not in sys.path:
        sys.path.insert(0, here)

from app import extract_mutual_fund_cas_hybrid

def main():
    here = os.path.dirname(__file__)
    sample_path = os.path.join(here, "tests", "sample_portfolio.txt")
    with open(sample_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Run CAS/Portfolio parser (offline-safe: LLM failures are caught inside)
    data = extract_mutual_fund_cas_hybrid(text)

    # Focus on the end-of-document summary and snapshot we added
    result = {
        "has_investment_snapshot": "investment_snapshot" in data,
        "investment_snapshot": data.get("investment_snapshot"),
        "account_summary": data.get("account_summary"),
        "keys": list(data.keys()),
        "extraction_error": data.get("extraction_error"),
    }

    # Save output for inspection
    out_dir = os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_portfolio_summary.json")
    with open(out_path, "w", encoding="utf-8") as wf:
        json.dump(result, wf, ensure_ascii=False, indent=2)

    # Print to stdout for terminal view
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
