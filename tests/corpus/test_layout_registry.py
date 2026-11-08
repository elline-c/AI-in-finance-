"""Registry tests (task 10.1). Skip cleanly until the layouts are implemented."""
from __future__ import annotations

import pytest

from fraud_detect.corpus.schema import LAYOUTS, SimulatorKnobs
from tests.corpus.helpers import DEFAULT_START, load_renderer, render_ledger

SEED = 4242
# Layouts whose real samples continue a month on a second page (design D3).
REAL_SPILL_LAYOUTS = {"two_col_slash", "two_col_dual_date"}


def _registry():
    layouts = pytest.importorskip("fraud_detect.corpus.layouts")
    return layouts


def _require_all_renderers():
    missing = [n for n in LAYOUTS if load_renderer(n) is None]
    if missing:
        pytest.skip(f"renderers not implemented yet: {', '.join(missing)}")


def test_renderers_has_exactly_the_six_layout_names():
    _require_all_renderers()
    assert set(_registry().RENDERERS) == set(LAYOUTS)
    assert len(_registry().RENDERERS) == 6 == len(LAYOUTS)


def test_each_renderer_name_equals_its_key():
    _require_all_renderers()
    for key, renderer in _registry().RENDERERS.items():
        assert renderer.name == key


def test_get_renderer_returns_registered_renderer_and_rejects_unknown():
    layouts = _registry()
    with pytest.raises(ValueError) as exc:
        layouts.get_renderer("no_such_layout")
    assert "two_col_slash" in str(exc.value)
    _require_all_renderers()
    for name in LAYOUTS:
        assert layouts.get_renderer(name) is layouts.RENDERERS[name]


def test_layout_names_equal_node1_registry_profile_names():
    from fraud_detect.nodes.ingest_parse.layouts import REGISTRY

    assert {p.name for p in REGISTRY} == set(LAYOUTS)


def _long_ledger(seed: int):
    from fraud_detect.corpus.company import make_company
    from fraud_detect.corpus.ledger import simulate_ledger

    knobs = SimulatorKnobs(rows_per_month_range=(110, 120))
    return simulate_ledger(make_company(seed, "C0003"), DEFAULT_START, 2, seed, knobs, None)


@pytest.mark.parametrize("layout", [n for n in LAYOUTS if n != "two_col_dmon"])
def test_spilled_month_flagged_when_layout_does_not_spill_in_real_samples(layout):
    renderer = load_renderer(layout)
    if renderer is None:
        pytest.skip(f"layout {layout!r} not implemented yet")
    result = render_ledger(renderer, _long_ledger(SEED), SEED)
    assert result.spilled_months, "a month with 110+ rows must continue on a second page"
    spill_not_in_real_samples = bool(result.spilled_months) and layout not in REAL_SPILL_LAYOUTS
    assert spill_not_in_real_samples == (layout not in REAL_SPILL_LAYOUTS)


def test_row_caps_follow_the_real_sample_behaviour():
    _require_all_renderers()
    layouts = _registry().RENDERERS
    for name in REAL_SPILL_LAYOUTS:
        assert layouts[name].max_rows_per_month is None
    assert layouts["two_col_dmon"].max_rows_per_month is not None
