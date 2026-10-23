"""Tests for the GNN scores-store key helpers (Node 6a, Task 5).

These cover the PURE-Python edge-key construction in
``fraud_detect.ml.gnn.train_gnn`` — the part that must stay byte-for-byte
compatible with the live cross-case memory key (``TxnFingerprint.key()``) so the
offline edge scores and the live lookup collapse to the same string for the same
inputs. ``train_gnn`` imports torch only lazily, so these run offline with NO
torch / torch-geometric installed (the key helpers reuse only ``store``).
"""
from __future__ import annotations

from fraud_detect.ml.gnn.train_gnn import (
    edge_key,
    model_name_with_adaptations,
    normalize_direction,
)
from fraud_detect.store import fingerprint


def test_edge_key_matches_txn_fingerprint_key_credit():
    """For the SAME inputs the edge key must equal ``TxnFingerprint.key()``:
    ``counterparty|amount_band|direction`` with the store's own helpers."""
    cp = "Offshore Nominee A Pte Ltd"
    amt = 149_500.0
    # Credit: positive amount in the memory path == "CR" in the dataset.
    expected = fingerprint(cp, amt).key()
    assert edge_key(cp, amt, "CR") == expected
    assert edge_key(cp, amt, "credit") == expected


def test_edge_key_matches_txn_fingerprint_key_debit():
    """A debit ("DR") must match the memory key built from a negative amount."""
    cp = "Mule Holdings LLC"
    amt = 150_000.0
    expected = fingerprint(cp, -amt).key()  # negative => debit in the memory path
    assert edge_key(cp, amt, "DR") == expected
    assert edge_key(cp, amt, "debit") == expected


def test_edge_key_shape_is_three_pipe_separated_fields():
    """The key always has exactly the three fields of TxnFingerprint.key()."""
    k = edge_key("ACC000456", 140_000, "CR")
    parts = k.split("|")
    assert len(parts) == 3
    counterparty, amount_band, direction = parts
    assert direction == "credit"
    assert amount_band.startswith("~")


def test_edge_key_handles_missing_and_unknown_inputs():
    """Missing amount/direction/counterparty still produce a well-formed key."""
    # Empty counterparty, "na" amount band (None amount), "na" direction.
    assert edge_key(None, None, None) == "|na|na"
    # Unknown direction collapses to "na"; unparseable amount -> "na" band.
    k = edge_key("X", "not-a-number", "SIDEWAYS")
    assert k.endswith("|na|na")
    assert k.split("|")[0] == "x"


def test_normalize_direction_mapping():
    assert normalize_direction("DR") == "debit"
    assert normalize_direction("CR") == "credit"
    assert normalize_direction("credit") == "credit"
    assert normalize_direction("debit") == "debit"
    assert normalize_direction(None) == "na"
    assert normalize_direction("") == "na"
    assert normalize_direction("weird") == "na"


def test_model_name_reflects_enabled_adaptations():
    assert (
        model_name_with_adaptations("gin", reverse_mp=True, ego=True, ports=True)
        == "gin+reverse_mp+ego+ports"
    )
    assert (
        model_name_with_adaptations("GAT", reverse_mp=False, ego=False, ports=False)
        == "gat"
    )
    assert (
        model_name_with_adaptations("pna", reverse_mp=True, ego=False, ports=True)
        == "pna+reverse_mp+ports"
    )
