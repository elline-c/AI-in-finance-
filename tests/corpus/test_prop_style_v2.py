"""Task 3.3: property tests for ``fraud_detect/corpus/styling.py`` (Balanced_Styling).

Feature: generator-styling-variety, Property 2: The style draw is a pure function of seed and document id, and keeps the legacy values
Feature: generator-styling-variety, Property 3: Every drawn value comes from its catalogue
Feature: generator-styling-variety, Property 4: The printed address follows the template and the second-line flag
Feature: generator-styling-variety, Property 5: The normaliser removes only the label-dependent choices

Pure functions only (no PDF rendering), so every property runs at least 100 examples.
"""
from __future__ import annotations

import dataclasses
import inspect

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from fraud_detect.corpus import finish, styling
from fraud_detect.corpus.schema import ALL_TAMPER_TYPES
from fraud_detect.corpus.seeding import rng
from fraud_detect.corpus.tamper.catalog import CATALOG
from tests.corpus.helpers import load_renderer, make_test_ledger

PBT = settings(max_examples=100, deadline=None,
               suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])

seeds = st.integers(min_value=0, max_value=2**31 - 1)
doc_ids = st.from_regex(r"[A-Z]{1,3}[0-9]{1,6}", fullmatch=True)


# --------------------------------------------------------------------------
# Property 2
# --------------------------------------------------------------------------
@PBT
@given(seed=seeds, doc_id=doc_ids)
def test_property_2_pure_draw_keeps_legacy_values(seed, doc_id):
    """**Validates: Requirements 2.2, 6.1, 6.3**"""
    a = styling.make_style_v2(seed, doc_id)
    b = styling.make_style_v2(seed, doc_id)
    assert a == b
    legacy = finish.make_style(seed, doc_id)
    for name in ("bank_name", "address_lines", "creation_time", "row_step",
                 "second_address_line"):
        assert getattr(a, name) == getattr(legacy, name), name


def test_property_2_signature_has_no_label_argument():
    """**Validates: Requirements 2.2**"""
    params = list(inspect.signature(styling.make_style_v2).parameters.values())
    assert [p.name for p in params] == ["master_seed", "parts"]
    assert params[1].kind is inspect.Parameter.VAR_POSITIONAL


# --------------------------------------------------------------------------
# Property 3
# --------------------------------------------------------------------------
@PBT
@given(seed=seeds, doc_id=doc_ids)
def test_property_3_values_come_from_catalogues(seed, doc_id):
    """**Validates: Requirements 2.1, 3.2, 4.1, 5.2**"""
    s = styling.make_style_v2(seed, doc_id)
    assert s.producer_family in {"base", "export", "editor"}
    assert s.producer in styling.PRODUCER_CATALOGUE[s.producer_family]
    assert s.creator in styling.CREATOR_POOL
    assert s.font_regular in {"Helvetica", "Times-Roman"}
    assert s.font_bold == "Helvetica-Bold"
    assert s.address_wording in {0, 1, 2}
    assert s.print_address is True
    assert s.style_version == 2


# --------------------------------------------------------------------------
# Property 4
# --------------------------------------------------------------------------
@PBT
@given(seed=seeds, doc_id=doc_ids)
def test_property_4_address_follows_template_and_flag(seed, doc_id):
    """**Validates: Requirements 5.2, 5.3**"""
    s = styling.make_style_v2(seed, doc_id)
    street, postcode_line = s.address_lines[0], s.address_lines[1]
    text = styling.address_text(s)
    if not s.second_address_line:
        assert text == street
    elif s.address_wording == 0:
        assert text == f"{street}, {postcode_line}"
    elif s.address_wording == 1:
        assert text == f"{street} {postcode_line}"
    else:
        assert text == street


@PBT
@given(seed=seeds, doc_id=doc_ids, wording=st.sampled_from((0, 1, 2)),
       flag=st.booleans())
def test_property_4_every_wording_and_flag_combination(seed, doc_id, wording, flag):
    """**Validates: Requirements 5.2, 5.3**"""
    s = dataclasses.replace(styling.make_style_v2(seed, doc_id),
                            address_wording=wording, second_address_line=flag)
    street, postcode_line = s.address_lines[0], s.address_lines[1]
    expected = (f"{street}, {postcode_line}" if flag and wording == 0
                else f"{street} {postcode_line}" if flag and wording == 1
                else street)
    assert styling.address_text(s) == expected


# --------------------------------------------------------------------------
# Property 5
# --------------------------------------------------------------------------
_LAYOUT = "generator_v1"


def _tamper_result(name: str, seed: int):
    renderer = load_renderer(_LAYOUT)
    assert renderer is not None
    ledger = make_test_ledger(1, seed % 5000)
    try:
        return CATALOG[name].plan(ledger, renderer, rng(seed, "prop-style-v2", name))
    except (ValueError, IndexError):   # a planner may not fit a 1-month ledger
        return None


def _params_without_producer(op):
    return tuple((k, v) for k, v in op.params if k != "producer")


@PBT
@given(name=st.sampled_from(ALL_TAMPER_TYPES), seed=st.integers(0, 10_000),
       doc_id=doc_ids, master=seeds)
def test_property_5_normaliser_removes_only_label_dependent_choices(name, seed, doc_id, master):
    """**Validates: Requirements 2.3, 2.5, 4.5**"""
    result = _tamper_result(name, seed)
    assume(result is not None)
    out = styling.normalise_tamper_result(result, master_seed=master, doc_id=doc_id)

    assert len(out.file_ops) == len(result.file_ops)
    for before, after in zip(result.file_ops, out.file_ops):
        if before.op in ("set_metadata", "resave"):
            assert all(k != "producer" for k, _ in after.params)
            # every other parameter (modified_delta_s, created, ...) equal and in order
            assert after.params == _params_without_producer(before)
            assert dataclasses.replace(after, params=before.params) == before
        elif before.op == "set_font":
            assert after.font in styling.FOREIGN_FONT_POOL
            again = styling.normalise_tamper_result(result, master_seed=master, doc_id=doc_id)
            assert [o.font for o in again.file_ops if o.op == "set_font"] == \
                   [o.font for o in out.file_ops if o.op == "set_font"]
            assert dataclasses.replace(after, font=before.font) == before
        else:
            assert after == before

    # ledger, edits and all label flags are unchanged (notes are rewritten by design)
    assert out.ledger == result.ledger
    assert out.edits == result.edits
    assert dataclasses.replace(out.label, notes="") == dataclasses.replace(result.label, notes="")


@PBT
@given(seed=st.integers(0, 10_000), doc_id=doc_ids, master=seeds)
def test_property_5_set_font_is_the_same_for_the_same_seed_and_doc(seed, doc_id, master):
    """**Validates: Requirements 4.5**"""
    result = _tamper_result("font_mismatch", seed)
    assume(result is not None)
    fonts = {styling.normalise_tamper_result(result, master_seed=master, doc_id=doc_id)
             .file_ops[0].font for _ in range(3)}
    assert len(fonts) == 1 and fonts <= set(styling.FOREIGN_FONT_POOL)


@PBT
@given(name=st.sampled_from(ALL_TAMPER_TYPES), seed=st.integers(0, 10_000),
       doc_id=doc_ids, master=seeds)
def test_property_5_normaliser_is_idempotent(name, seed, doc_id, master):
    """**Validates: Requirements 2.3, 4.5** (design: applying twice equals applying once)"""
    result = _tamper_result(name, seed)
    assume(result is not None)
    once = styling.normalise_tamper_result(result, master_seed=master, doc_id=doc_id)
    twice = styling.normalise_tamper_result(once, master_seed=master, doc_id=doc_id)
    assert twice == once
