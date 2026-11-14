# Meerkat — Document Parser & Financial Planning Backend

Flask backend that turns a client's financial documents and a questionnaire
into a personalised financial health report.

1. **Extract** — bank statements, ITRs, insurance policies and mutual fund CAS
   PDFs are parsed with deterministic rules first; an LLM only fills the gaps.
2. **Analyse** — the extracted figures and questionnaire answers feed risk
   profiling, a financial health score, goal/SIP allocation and tax checks.
3. **Report** — the result is rendered as a multi-page PDF (ReportLab) and
   summarised for the dashboard.

It also serves a retirement-planning module, free-tier calculators, Firebase
authentication, Razorpay payments and an admin panel.

## Project structure

```
app.py                  WSGI entrypoint (gunicorn app:app)
expire_reports.py       Daily cron job that prunes expired reports
backend/
  config.py             Environment-driven settings and the OpenAI client
  db.py                 SQLite persistence layer
  auth.py               Firebase token verification and route guards
  payment.py            Razorpay integration
  assumptions.py        Financial assumptions shared by every calculation
  utils.py              Numeric / formatting helpers
  extraction/           PDF text extraction and per-document-type parsers
  analysis/             Risk, health score, allocation engine, LLM report sections
  reports/              Financial health report PDF (theme, flowables, pages/)
  retirement/           Retirement planning engine and its PDF report
  routes/               Flask blueprints, one per feature area
scripts/                Developer utilities
tests/                  pytest suite
```

## Getting started

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in the keys
python app.py               # http://localhost:5000/healthz
```

`OPENAI_API_KEY` is required; the Firebase and Razorpay settings are described
in `.env.example`.

## Tests

```bash
pip install pytest pymupdf
OPENAI_API_KEY=test python -m pytest tests
```

The LLM is faked in the tests, so no network access or real key is needed.

## Deployment

Pushes to `master` deploy to Azure App Service through GitHub Actions — see
[DEPLOYMENT.md](DEPLOYMENT.md).
