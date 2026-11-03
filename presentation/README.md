# StatementGuard — Presentation deck

A slide deck to present and demonstrate the StatementGuard project.

## Files

| File | What it is | How to use |
|------|------------|------------|
| **`statementguard-deck-offline.html`** | **Fully self-contained** slide deck (20 slides) — all CSS + JS inlined, **zero network**. | Open in any browser, even offline. `←`/`→` or click to navigate, `F` fullscreen, `Esc` for the slide overview, `Home`/`End` to jump. Works from a USB stick or air-gapped laptop. **Recommended for presenting.** |
| `statementguard-deck.html` | [reveal.js](https://revealjs.com/) slide deck that loads reveal.js from a CDN at open time. | Open in any browser **with internet**. Arrow keys to navigate, `F` fullscreen, `Esc` overview. |
| `statementguard-deck.md` | The same deck as [Marp](https://marp.app/) Markdown. | Read inline on GitHub, or export to PDF/PPTX with `marp statementguard-deck.md --pdf` (or `--pptx`). |

> **Which one should I open?** Use **`statementguard-deck-offline.html`** — it needs no internet and renders identically everywhere. The reveal.js and Marp versions are kept for teams that prefer those tools.

### PDF

A ready-made **`statementguard-deck.pdf`** (20 pages, 16:9 landscape, one slide per page) is committed here — hand it out, email it, or present from it directly, no tooling required.

To regenerate it (e.g. after editing the content):

```bash
pip install --target=./.pylibs fpdf2         # one-time, into any writable dir
PYTHONPATH=./.pylibs python3 build_pdf.py    # writes statementguard-deck.pdf
```

`build_pdf.py` renders the deck with [fpdf2](https://py-pdf.github.io/fpdf2/) — pure Python, no browser or network. You can also export from the HTML instead: open `statementguard-deck-offline.html`, then **Print → Save as PDF** (landscape); its print stylesheet lays out one slide per page.

| File | What it is |
|------|------------|
| `statementguard-deck.pdf` | The committed PDF (share / present / print). |
| `build_pdf.py` | Script that generates the PDF from the deck content. |

## What the deck covers

1. Title & one-line pitch
2. The problem (manual review misses tampering)
3. Governance alignment — MAS FEAT, PDPA, MAS TRM
4. The pipeline at a glance (Nodes 1–6 + security layer + routing)
5–9. Each node in detail (Ingestion, Visual Forensics, Financial Logic, ML/memory, Risk Synthesis)
10. Three separated roles — customer / analyst / loan officer
11–14. **Live demonstration** — run commands, a clean file (AUTO_CLEAR), a tampered file (HUMAN_REVIEW), and the auditable evidence object
15. Validation — planted-issue detection, test counts
16. GNN evaluation — measured results with honest caveats
17. What's real vs. simulated
18. Roadmap — the 6 staged phases
19. Success metrics
20. Closing summary

## Source material

Content is drawn entirely from the project's own documentation: `README.md`,
`PROJECT_PLAN.md`, `WORKFLOW.md`, `STATEMENTGUARD_README.md`,
`FRAUD_DETECT_README.md`, `CUSTOMER_PORTAL_README.md`, `BANKER_PORTAL_README.md`,
and `fraud_detect/ml/gnn/EVALUATION.md`.

> The console transcripts on the demo slides are illustrative — they show the shape
> of the output (reason codes, scoring, routing) using the project's real reason
> codes and bands. Run `python demo.py` for the live output.
