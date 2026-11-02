"""Stable seeds for the corpus generator (no global random state, no ``hash()``).

Encoding of ``stable_seed(master_seed, *parts)``::

    text   = "|".join(repr(x) for x in (master_seed, *parts))
    digest = SHA-256(text encoded as UTF-8)
    seed   = int.from_bytes(digest[:8], "big")

``repr`` keeps ``1`` and ``'1'`` apart.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any


def stable_seed(master_seed: int, *parts: Any) -> int:
    """Return a 64-bit seed that depends only on the master seed and parts."""
    text = "|".join(repr(x) for x in (master_seed, *parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def rng(master_seed: int, *parts: Any) -> random.Random:
    """Return a fresh, private ``random.Random`` seeded by ``stable_seed``."""
    return random.Random(stable_seed(master_seed, *parts))
