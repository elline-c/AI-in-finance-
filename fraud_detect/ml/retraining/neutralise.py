"""Single-column neutralisation toolbox (design D5).

A "neutralised" feature is transformed inside the saved model by a scikit-learn step
on an existing column, so the live node can load the model without importing this
package. Both forms pickle as references to scikit-learn / NumPy objects only.

Supported recorded forms:

* ``clip(lo, hi)``: ``FunctionTransformer(np.clip, kw_args={a_min, a_max})``
* ``binarise(threshold)``: ``Binarizer(threshold=...)``

Anything else (including forms that need two columns) is refused.
"""
from __future__ import annotations

import math
import re

__all__ = ["NeutraliseError", "make_form", "TOOLBOX_FORMS"]

TOOLBOX_FORMS = ("clip", "binarise")

_FORM_RE = re.compile(r"^\s*([a-z]+)\s*\(([^()]*)\)\s*$")


class NeutraliseError(ValueError):
    """A recorded form is outside the single-column toolbox or is malformed."""


def _numbers(arg_text: str, spec: str, expected: int) -> list[float]:
    parts = [p.strip() for p in arg_text.split(",")] if arg_text.strip() else []
    if len(parts) != expected:
        raise NeutraliseError(
            f"form {spec!r} needs {expected} numeric argument(s), got {len(parts)}")
    values = []
    for p in parts:
        try:
            v = float(p)
        except ValueError:
            raise NeutraliseError(f"form {spec!r} has a non-numeric argument {p!r}") from None
        if not math.isfinite(v):
            raise NeutraliseError(f"form {spec!r} has a non-finite argument {p!r}")
        values.append(v)
    return values


def make_form(spec: str):
    """Return an unfitted single-column scikit-learn transformer for ``spec``."""
    if not isinstance(spec, str):
        raise NeutraliseError(f"form must be a string, got {type(spec).__name__}")
    m = _FORM_RE.match(spec)
    if not m:
        raise NeutraliseError(f"form {spec!r} is not in the toolbox {TOOLBOX_FORMS}")
    name, arg_text = m.group(1), m.group(2)

    if name == "clip":
        lo, hi = _numbers(arg_text, spec, 2)
        if lo > hi:
            raise NeutraliseError(f"form {spec!r} has lower bound above upper bound")
        import numpy as np
        from sklearn.preprocessing import FunctionTransformer
        return FunctionTransformer(np.clip, kw_args={"a_min": lo, "a_max": hi})

    if name == "binarise":
        (threshold,) = _numbers(arg_text, spec, 1)
        from sklearn.preprocessing import Binarizer
        return Binarizer(threshold=threshold)

    raise NeutraliseError(f"form {spec!r} is not in the toolbox {TOOLBOX_FORMS}")
