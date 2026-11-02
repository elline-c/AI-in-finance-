"""Layout renderer registry: one renderer per name in ``schema.LAYOUTS``."""
from __future__ import annotations

from fraud_detect.corpus.layouts.base import LayoutRenderer
from fraud_detect.corpus.layouts.dataset_family import TABLE_RENDERERS
from fraud_detect.corpus.layouts.generator_v1 import RENDERER as _GENERATOR_V1
from fraud_detect.corpus.schema import LAYOUTS

RENDERERS: dict[str, LayoutRenderer] = {"generator_v1": _GENERATOR_V1, **TABLE_RENDERERS}

__all__ = ["RENDERERS", "get_renderer"]


def get_renderer(name: str) -> LayoutRenderer:
    """Return the renderer for ``name``.

    Unknown names raise ``ValueError`` listing the allowed names; known layouts
    without a renderer raise ``NotImplementedError``.
    """
    if name not in LAYOUTS:
        raise ValueError(f"unknown layout {name!r}; allowed: {', '.join(LAYOUTS)}")
    try:
        return RENDERERS[name]
    except KeyError:
        raise NotImplementedError(f"layout {name!r} has no renderer") from None
