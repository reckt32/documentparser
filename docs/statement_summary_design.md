# Robust end-of-document summary extraction (design)

Problem
- PDFs like Portfolio Summary/CAS place the true statement summary at the end (e.g., Investment Snapshot, Grand Total, XIRR).
- Current approach scans raw text without a reliable notion of section order/position, so earlier partial matches get picked.
- Need a deterministic, explainable pipeline that always prefers the last valid summary and is auditable.

Design overview
1) Page/Section Indexing (deterministic)
- Build a document index on first upload:
  - For each page: extract text blocks (with page number and y-position) and tables (with headers and rows).
  - Detect candidate sections by headings/keywords: 
    - “Investment Snapshot”, “Grand Total”, “Statement Summary”, “SIP Summary”
  - Persist document, sections, and tables into a lightweight SQLite DB using the document fingerprint (sha256 of bytes).
- Ordered selection uses (page asc, y asc) to compute the canonical “last occurrence” reliably.

2) Candidate detection + ranking (rules first)
- Regex-driven section classifiers:
  - Snapshot: Investment (A), Switch In (B), Switch Out (C), Redemption (D), Div. Payout/FD Interest (E), Net Investment (F), Current Value (G), Net Gain (H), XIRR.
  - Statement summary: Opening/Closing Balance, Total Inflows/Credits, Total Outflows/Debits, Grand Total lines.
- Ranking:
  - Prefer sections from the highest page number.
  - Within a page, prefer sections with the greatest y (lowest on the page).
  - If multiple candidates exist on the last page, prefer ones containing the richest set of required fields.

3) Parsing rules (deterministic)
- Apply narrowly-scoped regex extractors to the chosen final section only (not across the whole text).
- Number cleaning:
  - Existing `clean_and_convert_to_float` handles ₹, commas, parentheses negatives, and trailing hyphen negatives.
- Output schema:
  - investment_snapshot: investment, switch_in, switch_out, redemption, div_payout_fd_interest, net_investment, current_value, net_gain, xirr_percent
  - account_summary: opening_balance, closing_balance, total_inflows, total_outflows
- Each metric stores its source section id and page for traceability.

4) Caching and auditability (SQLite)
- On first parse, index and extract → store per-doc results. Subsequent uploads of the same PDF use the cache by fingerprint.
- Add a debug endpoint to dump the section index for a document and the selected final section.

SQLite schema
```sql
-- documents
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY,
  sha256 TEXT UNIQUE NOT NULL,
  filename TEXT,
  page_count INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- sections (text regions with heading classification)
CREATE TABLE IF NOT EXISTS sections (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL,
  page_number INTEGER NOT NULL,
  heading TEXT,                 -- raw heading text, if any
  section_type TEXT,            -- snapshot | statement_summary | sip_summary | other
  y_top REAL,                   -- top Y coordinate (pdf units)
  y_bottom REAL,                -- bottom Y coordinate
  text TEXT NOT NULL,
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- tables (optional, for rows used by summary detection where needed)
CREATE TABLE IF NOT EXISTS tables (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL,
  section_id INTEGER,           -- nullable if not associated to a heading block
  page_number INTEGER NOT NULL,
  header_json TEXT,             -- JSON array
  rows_json TEXT,               -- JSON array of arrays
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
  FOREIGN KEY(section_id) REFERENCES sections(id) ON DELETE SET NULL
);

-- extracted metrics, with provenance
CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL,
  key TEXT NOT NULL,            -- e.g. investment, current_value, xirr_percent, total_inflows
  value_num REAL,
  source_section_id INTEGER,
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
  FOREIGN KEY(source_section_id) REFERENCES sections(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_sections_doc_page ON sections(document_id, page_number, y_top);
CREATE INDEX IF NOT EXISTS idx_metrics_doc_key ON metrics(document_id, key);
```

Indexing algorithm
```python
# python
def index_document(pdf_bytes) -> str:
    sha = sha256(pdf_bytes).hexdigest()
    if not exists_in_db(sha):
        doc_id = insert_document(sha, filename, page_count)
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                # 1) extract text blocks with y-coordinates
                words = page.extract_words(keep_blank_chars=False, use_text_flow=True)  # words have x0, x1, top, bottom
                blocks = group_words_into_blocks(words)  # heuristic line/paragraph grouping
                # 2) classify sections by heading/keywords
                for block in blocks:
                    stype = classify_section(block.text)  # snapshot | statement_summary | sip_summary | other
                    section_id = insert_section(doc_id, i, block.heading, stype, block.top, block.bottom, block.text)
                # 3) extract tables if present
                for tbl in page.extract_tables() or []:
                    header, rows = normalize_table(tbl)
                    insert_table(doc_id, section_id_for_table_if_any, i, header, rows)
    return sha
```

Selection and extraction
```python
# python
def pick_final_section(doc_id, target_type):
    # pick last by (page desc, y_bottom desc)
    rows = query("""
      SELECT id, page_number, y_bottom, text
      FROM sections
      WHERE document_id=? AND section_type=?
      ORDER BY page_number DESC, y_bottom DESC
      LIMIT 1
    """, (doc_id, target_type))
    return rows[0] if rows else None

def extract_snapshot_from_text(text):
    # strict regexes for A..H and XIRR; return only when majority found
    ...

def extract_statement_summary_from_text(text):
    # strict regexes for opening/closing/credits/debits/grand total; prefer totals row
    ...
```

API integration
- upload flow:
  1) Read PDF bytes → compute sha256.
  2) index_document(pdf_bytes) if not cached.
  3) Pick final snapshot section and final statement summary section.
  4) Parse each with deterministic regex; persist metrics with provenance.
  5) Build response JSON from metrics (fall back to txn-derived totals only when a target metric is missing).

Failure modes handled
- Multiple “snapshot” occurrences: rank by page/y to take the last.
- Numbers inside intermediate tables: ignored because extractor reads from the chosen final section only.
- Missing headings but present totals row: fallback classifier looks for “Grand Total” and similar keywords within blocks.
- Scanned PDFs: optional OCR fallback (future) using Tesseract, store OCR text in sections with a flag.

Why SQLite
- Zero external dependency, file-based, reliable ACID semantics.
- Enables caching, reproducibility, and easy debugging via SQL.
- Schema keeps provenance (which section/page yielded each metric).

Implementation plan (staged)
1) Infra
   - Add `backend/indexer.py` (indexing, section classifier, table normalizer).
   - Add `backend/extractors.py` (regex extractors for snapshot and statement summary).
   - Add `backend/db.py` (SQLite helpers, migrations).
2) Upload integration
   - In `app.py`, compute SHA, call indexer, then extractors; assemble `investment_snapshot` and `account_summary` from DB-backed metrics.
3) Debug endpoints
   - GET `/debug/docs/<sha>` → list sections (page, type, y, 120-chars preview).
   - GET `/debug/docs/<sha>/metrics` → list metrics with source_section/page.
4) Tests
   - Unit tests with provided sample text (and add a couple more PDFs) to assert:
     - Last snapshot is chosen.
     - Grand Total-based summary is used.
     - Provenance points to the last page section.
5) Performance
   - Indexing runs once per doc; subsequent requests are O(1) with DB lookups.

Edge cases and extensions
- OCR fallback (tesseract) when `pdfplumber` returns no text.
- Template fine-tuning: add per-issuer variants if needed by adding new regex patterns without changing the pipeline.
- Optional vector ranking for section type if strict keywords fail (embeddings), retained as a last-resort fallback, behind a feature flag.

Acceptance criteria
- The selected summary comes from the final matching section by document order.
- Snapshot A..H numbers and XIRR match the last on-page values.
- Statement summary totals match last totals row (or explicit labeled fields).
- Response includes page and section id for every reported metric to aid verification.

Next steps (awaiting approval)
- Implement DB + indexer + extractors (staged PRs).
- Wire into upload endpoint.
- Add debug endpoints and tests.
