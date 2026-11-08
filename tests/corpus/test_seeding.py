"""Tests for stable seeding (Requirements 10.1, 10.6).

Documented encoding of ``stable_seed(master_seed, *parts)``:

    text  = "|".join(repr(x) for x in (master_seed, *parts))
    digest = SHA-256(text encoded as UTF-8)
    seed  = int.from_bytes(digest[:8], "big")

``repr`` keeps ``1`` and ``'1'`` apart. Python's ``hash()`` is not used.
"""
import hashlib
import os
import random
import subprocess
import sys

from fraud_detect.corpus.seeding import rng, stable_seed

PINNED = 17781915821409977862  # stable_seed(42, "doc", "two_col_dmon", 3)


def _expected(master_seed, *parts):
    text = "|".join(repr(x) for x in (master_seed, *parts))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def test_same_parts_same_seed():
    assert stable_seed(42, "doc", "a", 1) == stable_seed(42, "doc", "a", 1)


def test_different_parts_different_seeds():
    assert stable_seed(42, "doc", "a", 1) != stable_seed(42, "doc", "a", 2)
    assert stable_seed(42, "doc", "a", 1) != stable_seed(42, "company", "a", 1)


def test_different_master_seed_different_seed():
    assert stable_seed(1, "doc") != stable_seed(2, "doc")


def test_order_of_parts_matters():
    assert stable_seed(42, "a", "b") != stable_seed(42, "b", "a")


def test_int_and_str_parts_distinguished():
    assert stable_seed(42, "doc", 1) != stable_seed(42, "doc", "1")


def test_matches_documented_encoding_and_pinned_literal():
    assert stable_seed(42, "doc", "two_col_dmon", 3) == _expected(42, "doc", "two_col_dmon", 3)
    assert stable_seed(42, "doc", "two_col_dmon", 3) == PINNED
    assert isinstance(stable_seed(42), int)


def test_independent_of_global_random_state():
    random.seed(1)
    a = stable_seed(42, "doc", 1)
    random.seed(999)
    random.random()
    assert stable_seed(42, "doc", 1) == a


def test_independent_of_pythonhashseed():
    code = (
        "from fraud_detect.corpus.seeding import stable_seed;"
        "print(stable_seed(42, 'doc', 'two_col_dmon', 3))"
    )
    outputs = set()
    for hashseed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        res = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        outputs.add(res.stdout.strip())
    assert outputs == {str(PINNED)}


def test_rng_is_deterministic_and_independent_objects():
    r1, r2 = rng(42, "doc", 1), rng(42, "doc", 1)
    assert isinstance(r1, random.Random)
    assert r1 is not r2
    first = [r1.random() for _ in range(5)]
    assert first == [r2.random() for _ in range(5)]
    assert first == [random.Random(stable_seed(42, "doc", 1)).random() for _ in range(1)] + first[1:]
    assert rng(42, "doc", 2).random() != first[0]


def test_rng_does_not_touch_global_random_state():
    random.seed(7)
    before = random.getstate()
    r = rng(42, "doc", 1)
    for _ in range(10):
        r.random()
    assert random.getstate() == before
