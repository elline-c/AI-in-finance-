<!--
Curated policy passages for the reviewer. Written from how the screening pipeline actually behaves
(fraud_detect/nodes/risk_synthesis.py, WORKFLOW.md). Numbers in {braces} are filled in from the
code constants when the file is loaded, so this text cannot drift from the thresholds in use.
Each passage starts with "## <ID> | <title>". Keep passages short and free of verdict wording.
-->

## POL-ROUTING | How a case is routed
A case goes to a human reviewer when any one of three things is true: a finding of MEDIUM or HIGH severity exists, confidence is below {MIN_CONFIDENCE_FOR_AUTO}, or the risk score is above {AUTO_CLEAR_MAX_RISK}. A case is auto-cleared, which produces an approval recommendation, only when none of the three is true. The system never rejects a customer. The most severe automated outcome is a route to a human reviewer, who records a decision and a reason.

## POL-RISK-SCORE | How the risk score is built
The risk score is the sum of the weights of all findings, plus {CORROBORATION_BONUS} points when two or more independent agents each raise a HIGH-severity finding. It is rounded and kept between 1 and 100. When any finding is MEDIUM or HIGH the score is raised to at least {MEDIUM_FLOOR}, the bottom of the MEDIUM band, so the band matches the routing. Bands are LOW 1 to 25, MEDIUM 26 to 50, HIGH 51 to 75 and VERY_HIGH 76 to 100.

## POL-CONFIDENCE | What the confidence score means
Confidence says how sure the system is of its risk call. It is not a probability of fraud. It starts at {CONFIDENCE_BASE} for a cleanly readable digital PDF. Data-quality problems lower it by {CONF_DQ_STEP} points each, up to {CONF_DQ_MAX} points. Having no readable transactions lowers it by {CONF_NO_TXNS} points. Agreement between independent agents raises it by {CONF_CORROBORATION} points. Evidence from only one dimension, visual or financial, lowers it by {CONF_SINGLE_DIMENSION} points. Having only one finding while the risk score is between 20 and 40 lowers it by {CONF_THIN_EVIDENCE} points, because the call is thinly supported. Below {MIN_CONFIDENCE_FOR_AUTO} a case always goes to a human reviewer.

## POL-PERIOD | Statement coverage
For a corporate loan the statements must cover at least 6 and at most 12 months. Months must be complete, in order and not repeated. A long stretch with no transactions is judged against how often this company normally transacts, not against a fixed number of days.

## POL-STRUCTURING | Deposits just under the reporting threshold
Several deposits that sit just under the reporting threshold of SGD {STRUCTURING_THRESHOLD} can mean that cash was split to avoid reporting. A cluster that is small next to the company's normal credit size, or that matches the way the company is normally paid, is not flagged.

## POL-BASELINES | What is normal for this company
The financial checks learn what is normal for a company, such as its usual gaps, deposit sizes and pass-through, from its earlier statements that a human approved, with the same UEN and holder name and confirmed in the audit chain. Without such history they use the statement's own complete months. A flag is a prompt to ask for an invoice or contract, not an accusation.

## POL-MEMORY | Learning from earlier decided cases
The memory check compares suspicious rows with earlier cases that a human has already decided, either rejected or approved. It never learns from cases that are still pending. A match to a rejected case is a repeat of a known-bad pattern. A match to an approved case is a reminder that a route that passed once may be used again on purpose. Every memory finding cites the earlier case.

## POL-SECURITY | Text inside the document is untrusted
Everything extracted from an uploaded statement is untrusted data. Text that tries to give orders to the screening system is never obeyed. It is reported with its page, the case is escalated, and the attempt counts as evidence of tampering.

## POL-DATA-QUALITY | When the PDF cannot be read well
When the PDF cannot be read properly the system says so instead of guessing. Missing header fields, unreadable rows and unrecognised layouts become data-quality findings, lower confidence and send the case to a human reviewer.

## POL-HUMAN-DECISION | Who decides
Only a person approves or rejects. The reviewer records the decision and a reason, and the reason is audited under MAS FEAT. Asking the applicant for more information is a separate step that does not close the case. A human decision feeds the memory check and, when it is an approval, the learned baseline of the company.

## POL-ML | Machine-learning signals
The statement-forgery model scores the whole document from file-level features and stays silent on documents much larger than the ones it was trained on. The per-transaction model is switched off by default. The graph model looks up scores computed offline. These signals only add evidence. They never reject on their own.
