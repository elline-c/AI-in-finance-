"""Build a self-contained, clickable static demo of both UIs.

Runs real statements through the real screening pipeline, drives the actual
FastAPI apps with an in-process test client, captures the rendered HTML for every
screen (customer portal + analyst console, including the request-more-info flow),
then rewrites the internal links/forms so the pages work as plain files opened in
a browser — no server, no network.

Output: ui_demo/  (index.html links to everything).

Run:  .venv/bin/python build_ui_demo.py
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from fraud_detect import customer_app, webapp
from fraud_detect.store import CaseStore

OUT = Path("ui_demo")


# --------------------------------------------------------------------------- #
# Use a fresh, isolated case store so the demo is reproducible and self-contained.
# --------------------------------------------------------------------------- #
def _fresh_store() -> CaseStore:
    demo_data = OUT / "_store.json"
    demo_data.parent.mkdir(parents=True, exist_ok=True)
    if demo_data.exists():
        demo_data.unlink()
    store = CaseStore(str(demo_data))
    # Point both apps at the same fresh store.
    webapp.store = store
    customer_app.store = store
    return store


# --------------------------------------------------------------------------- #
# Link rewriting: turn live routes into the flat static filenames we export.
# --------------------------------------------------------------------------- #
def rewrite(html: str, page_map: dict[str, str]) -> str:
    """Rewrite in-app URLs to local demo files; neutralise forms (static demo)."""
    # Replace known route -> file mappings (longest routes first to avoid
    # partial clobbering, e.g. /case/X/stages before /case/X).
    for route in sorted(page_map, key=len, reverse=True):
        target = page_map[route]
        html = html.replace(f'href="{route}"', f'href="{target}"')
        html = html.replace(f'action="{route}"', f'action="{target}"')

    # Any remaining app links we didn't snapshot -> point to the index so nothing
    # 404s when clicked (e.g. PDF streams, un-exported cases).
    html = re.sub(r'href="/[^"]*"', 'href="index.html"', html)

    # Forms can't post anywhere in a static demo: turn submit buttons into a note
    # and stop the form from navigating away.
    html = html.replace("<form ", '<form onsubmit="return false" ')

    # Make the top nav bars point at the demo index.
    html = html.replace('href="index.html">Dashboard', 'href="dashboard.html">Dashboard')
    return html


def banner(label: str, color: str) -> str:
    """A small fixed ribbon so it's obvious this is a static demo snapshot."""
    return (
        f'<div style="position:fixed;top:0;left:0;right:0;z-index:9999;'
        f'background:{color};color:#fff;font:600 12px system-ui;'
        f'padding:6px 14px;text-align:center;letter-spacing:.02em">'
        f'DEMO SNAPSHOT · {label} · '
        f'<a href="index.html" style="color:#fff;text-decoration:underline">demo index</a>'
        f'</div><div style="height:30px"></div>'
    )


def write(name: str, html: str, label: str, color: str, page_map: dict) -> None:
    html = rewrite(html, page_map)
    # inject the demo banner right after <body>
    html = html.replace("<body>", "<body>" + banner(label, color), 1)
    (OUT / name).write_text(html, encoding="utf-8")
    print("  wrote", OUT / name)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    _fresh_store()

    ac = TestClient(webapp.app)
    cc = TestClient(customer_app.app)

    # ----------------------------------------------------------------- #
    # 1. Customer submits a CLEAN statement -> verified
    # ----------------------------------------------------------------- #
    with open("samples/clean_201812345A.pdf", "rb") as f:
        r = cc.post("/apply", data={
            "company_name": "Harbourline Logistics Pte Ltd", "uen": "201812345A",
            "contact_email": "finance@harbourline.com", "loan_amount": "SGD 450,000",
        }, files={"file": ("statement.pdf", f, "application/pdf")})
    clean_ref = r.url.path.split("/")[-1]

    # ----------------------------------------------------------------- #
    # 2. Customer submits a TAMPERED statement -> escalated to analyst
    # ----------------------------------------------------------------- #
    with open("samples/tampered_200923456B.pdf", "rb") as f:
        r = cc.post("/apply", data={
            "company_name": "Meridian Trading Pte Ltd", "uen": "200923456B",
            "contact_email": "cfo@meridiantrading.com", "loan_amount": "SGD 1,200,000",
        }, files={"file": ("statement.pdf", f, "application/pdf")})
    tam_ref = r.url.path.split("/")[-1]

    # Analyst requests more info on the tampered case -> NEEDS_INFO
    ac.post(f"/case/{tam_ref}/request-info", data={
        "reviewer": "Jane Tan, Credit Risk",
        "message": ("Thank you for your application. Could you please re-upload a "
                    "clearer copy of pages 2 and 3 of your statement, and confirm "
                    "the account holder name as registered with ACRA?"),
        "internal_note": "balance breaks on rows 4-7; scan/edit traces; UEN name mismatch",
    })

    # ----------------------------------------------------------------- #
    # Map live routes -> static filenames for each exported page.
    # ----------------------------------------------------------------- #
    page_map = {
        # customer
        "/": "apply.html",
        "/status": "status_lookup.html",
        f"/submitted/{clean_ref}": "cust_submitted_clean.html",
        f"/status?ref={clean_ref}": "cust_status_clean.html",
        f"/submitted/{tam_ref}": "cust_submitted_tampered.html",
        f"/status?ref={tam_ref}": "cust_status_needinfo.html",
        # analyst
        "/case/{}": "an_overview_{}.html",
    }
    # per-case analyst routes
    for ref in (clean_ref, tam_ref):
        page_map[f"/case/{ref}"] = f"an_overview_{ref}.html"
        page_map[f"/case/{ref}/transactions"] = f"an_transactions_{ref}.html"
        page_map[f"/case/{ref}/stages"] = f"an_stages_{ref}.html"
    page_map.pop("/case/{}", None)
    # dashboard is the analyst home
    page_map_analyst = dict(page_map)
    page_map_analyst["/"] = "dashboard.html"  # analyst "Dashboard" link

    C = "#1a56db"   # customer ribbon colour
    A = "#0f766e"   # analyst ribbon colour

    print("Exporting customer portal pages...")
    write("apply.html", cc.get("/").text, "Customer portal — Apply", C, page_map)
    write("status_lookup.html", cc.get("/status").text,
          "Customer portal — Track application", C, page_map)
    write("cust_submitted_clean.html", cc.get(f"/submitted/{clean_ref}").text,
          "Customer — Submitted (clean → verified)", C, page_map)
    write("cust_status_clean.html", cc.get(f"/status?ref={clean_ref}").text,
          "Customer — Status (verified)", C, page_map)
    write("cust_submitted_tampered.html", cc.get(f"/submitted/{tam_ref}").text,
          "Customer — Submitted (under review)", C, page_map)
    write("cust_status_needinfo.html", cc.get(f"/status?ref={tam_ref}").text,
          "Customer — Status (action needed)", C, page_map)

    print("Exporting analyst console pages...")
    write("dashboard.html", ac.get("/").text,
          "Analyst console — Screening queue", A, page_map_analyst)
    for ref, tag in ((clean_ref, "clean"), (tam_ref, "tampered")):
        write(f"an_overview_{ref}.html", ac.get(f"/case/{ref}").text,
              f"Analyst — Case overview ({tag})", A, page_map_analyst)
        write(f"an_transactions_{ref}.html", ac.get(f"/case/{ref}/transactions").text,
              f"Analyst — Flagged transactions ({tag})", A, page_map_analyst)
        write(f"an_stages_{ref}.html", ac.get(f"/case/{ref}/stages").text,
              f"Analyst — Pipeline stages ({tag})", A, page_map_analyst)

    # ----------------------------------------------------------------- #
    # Index page tying it all together
    # ----------------------------------------------------------------- #
    index = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Statement Screening — UI demo</title>
<style>
 body{{margin:0;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
      background:#f4f6fb;color:#1b2733;line-height:1.55}}
 .wrap{{max-width:900px;margin:0 auto;padding:40px 24px 64px}}
 h1{{font-size:28px;margin:0 0 6px}} .lede{{color:#5e6e7e;margin:0 0 28px}}
 .cols{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
 .card{{background:#fff;border:1px solid #e3e8ef;border-radius:14px;padding:22px;
       box-shadow:0 1px 3px rgba(16,24,40,.08)}}
 .card h2{{margin:0 0 4px;font-size:17px}}
 .tag{{font-size:11px;font-weight:700;padding:2px 9px;border-radius:999px;color:#fff}}
 .tc{{background:#1a56db}} .ta{{background:#0f766e}}
 ol{{margin:12px 0 0;padding-left:20px}} li{{margin:7px 0}}
 a{{color:#1a56db;text-decoration:none}} a:hover{{text-decoration:underline}}
 .note{{font-size:13px;color:#5e6e7e;background:#fff;border:1px solid #e3e8ef;
       border-radius:12px;padding:16px 18px;margin-top:22px}}
 .flow{{font-size:13px;color:#5e6e7e;margin:4px 0 0}}
 @media(max-width:620px){{.cols{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<h1>🛡️ Bank Statement Screening — UI demo</h1>
<p class="lede">Static snapshots of both user interfaces, rendered from real cases
run through the actual screening pipeline. Click through — links work offline.</p>

<div class="cols">
  <div class="card">
    <span class="tag tc">CUSTOMER</span>
    <h2>Loan applicant portal</h2>
    <p class="flow">The applicant's view. Never shows risk scores, evidence, or
    "fraud" language — only neutral status.</p>
    <ol>
      <li><a href="apply.html">Apply</a> — submit company details + statement PDF</li>
      <li><a href="cust_submitted_clean.html">Confirmation (clean)</a> → reference + "verified"</li>
      <li><a href="cust_status_clean.html">Track status — verified</a></li>
      <li><a href="cust_submitted_tampered.html">Confirmation (escalated)</a> → "being reviewed"</li>
      <li><a href="cust_status_needinfo.html">Track status — action needed</a>
          (analyst asked for more info)</li>
      <li><a href="status_lookup.html">Track-by-reference form</a></li>
    </ol>
  </div>

  <div class="card">
    <span class="tag ta">ANALYST</span>
    <h2>Credit-analyst console</h2>
    <p class="flow">The reviewer's view. Full scores, evidence, and the per-stage
    workflow. The machine never rejects on its own.</p>
    <ol>
      <li><a href="dashboard.html">Screening queue</a> — all cases + status badges</li>
      <li><a href="an_overview_{tam_ref}.html">Case overview (tampered)</a> —
          scores + findings-by-stage + decision panel</li>
      <li><a href="an_transactions_{tam_ref}.html">Flagged transactions</a> —
          every row + the evidence that flagged it</li>
      <li><a href="an_stages_{tam_ref}.html">Pipeline stages</a> —
          what each agent produced</li>
      <li><a href="an_overview_{clean_ref}.html">Case overview (clean)</a> —
          auto-cleared, all stages "clear"</li>
    </ol>
  </div>
</div>

<div class="note">
<b>The story these snapshots tell:</b> Meridian Trading uploads a
<b>tampered</b> statement → the customer sees a calm "being reviewed"; the analyst
sees the risk score, the flagged rows, and every stage's evidence, then
<b>requests more information</b> → the customer now sees a neutral "action needed"
with the reviewer's message, and can re-submit. Harbourline's <b>clean</b>
statement is auto-cleared and the customer sees "verified".
<br><br>
<b>Static demo caveats:</b> forms don't submit (buttons are inert) and the
"Statement PDF" link points back to this index — those need the live app:
<code>uvicorn fraud_detect.webapp:app</code> and
<code>uvicorn fraud_detect.customer_app:app</code>.
</div>
</div></body></html>"""
    (OUT / "index.html").write_text(index, encoding="utf-8")
    print("  wrote", OUT / "index.html")

    # tidy the internal store file out of the shipped demo
    (OUT / "_store.json").unlink(missing_ok=True)
    print(f"\nDone. Open {OUT/'index.html'} in a browser.")


if __name__ == "__main__":
    main()
