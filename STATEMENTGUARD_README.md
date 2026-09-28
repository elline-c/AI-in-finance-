# StatementGuard — Tampered Bank Statement Screening Prototype

A working prototype of an agentic pipeline that screens corporate loan bank
statements for tampering **before** a credit analyst reviews them.

> Aligned with MAS **FEAT** principles: every decision carries readable,
> auditable evidence, and **the machine never rejects a customer on its own** —
> it only auto-clears clean files or routes suspicious ones to a human.

## The pipeline (4 worker nodes)

```
          ┌─────────────────────────────┐
  input → │ Node 1: Ingestion & Preproc │  OCR + layout parse + metadata
          └──────────────┬──────────────┘  → strict JSON contract
                         │
          ┌──────────────▼──────────────┐
          │ Node 2: Visual Forensics    │  metadata / font / format / layer
          └──────────────┬──────────────┘  anomalies + support-doc relevance
                         │
          ┌──────────────▼──────────────┐
          │ Node 3: Financial Logic     │  running balance, period continuity,
          └──────────────┬──────────────┘  structuring, anomalous flows
                         │
          ┌──────────────▼──────────────┐
          │ Node 4: Risk & Synthesis    │  risk 1-100 + confidence + report
          └──────────────┬──────────────┘
                         │
        risk band + confidence → routing decision
          ├── Low risk + high confidence  → AUTO-CLEAR (approval rec. summary)
          └── Medium/High/Very-High, OR low confidence,
              OR poor data quality        → HUMAN-IN-THE-LOOP
```

## Risk & confidence bands (from the ideation)

| Risk score | Band       |     | Confidence | Band   |
|------------|------------|-----|------------|--------|
| 1–25       | Low        |     | 76–100     | High   |
| 26–50      | Medium     |     | 51–75      | Medium |
| 51–75      | High       |     | 26–50      | Low    |
| 76–100     | Very High  |     | 1–25       | Very Low |

## Run it

```bash
cd statement_guard
python3 -m statement_guard.cli samples/clean_statement.json
python3 -m statement_guard.cli samples/tampered_statement.json
python3 -m statement_guard.cli --all          # run every sample
python3 -m statement_guard.cli samples/tampered_statement.json --json report.json
```

No external dependencies — pure Python 3 standard library.

## What is real vs. simulated

This prototype implements the **full decision pipeline, scoring math, evidence
model, and human-in-the-loop routing** with deterministic heuristics. The parts
that would require heavy ML/infra in production are represented by a clean input
contract so they can be swapped in later without changing the pipeline:

| Production component            | Prototype stand-in                                  |
|---------------------------------|-----------------------------------------------------|
| PDF→image + OCR (Node 1)        | Pre-extracted JSON input matching the OCR contract  |
| Pixel/layer image forensics     | Metadata + declared-artifact heuristics             |
| LLM narrative synthesis         | Templated, evidence-linked report generator         |

Every finding is a structured object with a severity, human-readable
explanation, and the raw evidence it was derived from — so the report is
auditable end-to-end.
