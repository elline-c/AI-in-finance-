"""Schema additions for the generator-styling-variety feature (task 2.1)."""
from __future__ import annotations

from pathlib import Path

from fraud_detect.corpus.schema import (
    ALL_TAMPER_TYPES,
    EXTRA_TAMPER_TYPES,
    GENERATOR_VERSION,
    LABELS_COLUMNS,
    STYLE_VERSIONS,
    TAMPER_TYPES,
    BuildConfig,
    StyleKnobs,
)


def test_constants():
    assert STYLE_VERSIONS == (1, 2)
    assert EXTRA_TAMPER_TYPES == ("subtle_amount_edit",)
    assert ALL_TAMPER_TYPES == TAMPER_TYPES + EXTRA_TAMPER_TYPES


def test_style_knobs_defaults():
    k = StyleKnobs()
    assert (k.bank_name, k.address_lines, k.producer) == ("", (), "")
    assert (k.creation_time, k.row_step, k.second_address_line) == ("09:00:00", 0.0, False)
    assert k.creator == ""
    assert k.font_regular == "Helvetica"
    assert k.font_bold == "Helvetica-Bold"
    assert k.print_address is False
    assert k.address_wording == 0
    assert k.producer_family == ""
    assert k.style_version == 1


def test_build_config_default_style_version():
    cfg = BuildConfig(out_dir=Path("x"), master_seed=1,
                      per_layout_genuine=1, per_layout_tampered=1)
    assert cfg.style_version == 1


def test_pinned_values_unchanged():
    assert GENERATOR_VERSION == "1.0.0"
    assert len(TAMPER_TYPES) == 16
    assert len(LABELS_COLUMNS) == 25
