"""Text scan of FRAUD_DETECT_README.md for the "Stage B: ML retraining" section (task 25.1).

Requirements 16.1-16.4, 16.6-16.9 and 17.13. These tests fail until task 25.2 adds the section.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[2] / "FRAUD_DETECT_README.md"
HEADING = re.compile(r"^#{1,6}\s+Stage B: ML retraining\s*$", re.M)

RESULT_WORDS = re.compile(
    r"corpus|result|score|recall|precision|\bauc\b|metric|accuracy|false[- ]alarm|"
    r"trained|validation|test set|dataset", re.I)
NON_REAL_SOURCES = ("generator_corpus", "forgery_dataset", "awkward_set")


def _section() -> str:
    text = README.read_text(encoding="utf-8")
    m = HEADING.search(text)
    assert m, 'FRAUD_DETECT_README.md has no heading "Stage B: ML retraining"'
    level = len(m.group(0).split()[0])
    rest = text[m.end():]
    nxt = re.search(rf"^#{{1,{level}}}\s", rest, re.M)  # next heading of same or higher level
    return rest[:nxt.start()] if nxt else rest


def _paragraphs(section: str) -> list:
    """Blank-line separated blocks, without fenced code blocks and headings."""
    section = re.sub(r"```.*?```", "", section, flags=re.S)
    blocks = [b.strip() for b in re.split(r"\n\s*\n", section)]
    return [b for b in blocks if b and not re.fullmatch(r"(#+\s.*\n?)+", b)]


def _has(text: str, *patterns: str) -> bool:
    return all(re.search(p, text, re.I) for p in patterns)


def _paras_with(*patterns: str) -> list:
    return [p for p in _paragraphs(_section()) if _has(p, *patterns)]


def test_section_exists():
    assert _section().strip()


def test_corpus_result_paragraphs_say_synthetic_data():  # 16.1
    offenders = [p[:80] for p in _paragraphs(_section())
                 if RESULT_WORDS.search(p) and not re.search(r"synthetic data", p, re.I)]
    assert _paragraphs(_section()), "section has no paragraphs"
    assert not offenders, f"paragraphs about Corpus-derived results lack 'synthetic data': {offenders}"


def test_first_results_table_paragraph_states_demo_not_real_world():  # 16.2
    paras = _paragraphs(_section())
    tables = [p for p in paras if re.search(r"^\s*\|.*\|\s*$", p, re.M)
              and re.search(r"switch|recall|precision|auc|result", p, re.I)]
    assert tables, "no results table found in the section"
    first = tables[0]
    assert _has(first, r"end-to-end demonstration"), "first results table paragraph lacks 'end-to-end demonstration'"
    assert _has(first, r"not real-world accuracy"), "first results table paragraph lacks 'not real-world accuracy'"


def test_circularity_risk_and_seven_mitigations():  # 16.3
    sec = _section()
    assert re.search(r"circular", sec, re.I), "circularity risk not stated"
    assert re.search(r"generator'?s? (own )?(output|habits)", sec, re.I), "generator's output/habits not stated"
    assert re.search(r"habits", sec, re.I)
    for name, pat in {
        "held-out layouts": r"held-out layouts",
        "held-out Master_Seeds": r"held-out Master_Seeds",
        "randomised styling": r"randomised styling",
        "Leak_Test": r"Leak_Test",
        "Real_Style_Files": r"Real_Style_Files",
        "Switch_Rule": r"Switch_Rule",
        "Awkward_Set": r"Awkward_Set",
    }.items():
        assert re.search(pat, sec), f"mitigation not named: {name}"


def test_only_two_real_style_statements():  # 16.4
    paras = _paras_with(r"statement_E", r"statement_F")
    assert any(_has(p, r"only two", r"too small", r"indicative only") for p in paras), \
        "need one paragraph naming statement_E and statement_F, saying only two exist, " \
        "that this is too small to estimate accuracy, and that results are indicative only"


@pytest.mark.parametrize("node", ["Node 6a", "Node 6b"])
def test_switch_outcome_and_model_in_use_per_node(node):  # 16.7
    ps = _paras_with(re.escape(node))
    assert any(_has(p, r"switched", r"Current model|Current_Model|Candidate model|Candidate_Model")
               and _has(p, r"because|reason|\bas\b|since|blocked|required") for p in ps), \
        f"{node}: need the Switch_Rule outcome (switched / not switched) with a reason " \
        f"and which model (Current or Candidate) it uses after Stage B"


def test_node4_unchanged():  # 16.8
    assert _paras_with(r"Node 4", r"weights", r"thresholds", r"not changed|unchanged|no change"), \
        "must state Node 4 weights and thresholds are not changed by Stage B"


@pytest.mark.parametrize("cmd", ["audit", "leak-test", "train", "compare", "switch", "restore"])
def test_each_command_documented(cmd):  # 17.13
    assert re.search(rf"python -m fraud_detect\.ml\.retraining\s+{re.escape(cmd)}\b", _section()), \
        f"command not documented as 'python -m fraud_detect.ml.retraining {cmd}'"


def test_exit_codes_documented():  # 17.13
    ps = _paras_with(r"exit code")
    assert ps, "no 'exit code' text in the section"
    joined = "\n".join(ps)
    for code in "01234":
        assert re.search(rf"(?<![\w.]){code}(?![\w.])", joined), f"exit code {code} not documented"


def test_label_stored_in_bundle_not_yet_shown_in_demo():  # 16.6 / decision 9
    assert any(_has(p, r"model trained on synthetic data", r"bundle", r"demo",
                    r"not yet shown|not shown|not yet displayed|not displayed")
               for p in _paragraphs(_section())), \
        'must state the "model trained on synthetic data" label is stored in the bundle ' \
        "and is not yet shown in the demo"


def test_non_real_style_scores_marked_synthetic_and_not_real_world():  # 16.9
    ps = [p for p in _paragraphs(_section()) if any(s in p for s in NON_REAL_SOURCES)]
    assert ps, "no paragraph reports a score on a non-real_style Source (generator_corpus, forgery_dataset, awkward_set)"
    for p in ps:
        assert _has(p, r"synthetic data", r"not real-world accuracy"), \
            f"non-real_style source paragraph lacks 'synthetic data' / 'not real-world accuracy': {p[:80]}"
