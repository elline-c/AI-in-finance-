# AI-in-finance-

## StatementGuard — Tampered Bank Statement Screening

A prototype that screens corporate-loan bank statements for **tampering** before a
credit analyst reviews them. It auto-clears clean files and routes suspicious ones
to a human, attaching readable evidence to every flag — aligned with MAS **FEAT**
principles (the machine never approves or rejects a customer on its own).

### The 4-agent pipeline
1. **Ingestion** — reads the statement, flags missing/messy data.
2. **Visual Forensics** — flags editing-software traces, altered metadata, dodgy supporting docs.
3. **Financial Logic** — checks balances add up; detects structuring, anomalous & pass-through transactions.
4. **Risk & Synthesis** — combines everything into a risk score (1–100) + confidence score, then decides: auto-clear or send to a human.

### Run it (no installation — pure Python 3)
```bash
python3 -m statement_guard.cli --all                              # run all samples
python3 -m statement_guard.cli samples/clean_statement.json       # one clean file
python3 -m statement_guard.cli samples/tampered_statement.json    # one tampered file
```

### Project docs
- **`PROJECT_PLAN.md`** — the full roadmap (what's built + the 6 phases in plain English).
- **`STATEMENTGUARD_README.md`** — detailed prototype notes.

> **Status:** working *prototype*. The decision engine (all 4 agents + scoring +
> routing + evidence) is complete. Real PDF/OCR reading, real image forensics, the
> analyst review screen, and production hardening are the remaining phases.
