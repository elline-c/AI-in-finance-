"""Unit tests for verifying an empty or absent audit chain.

Feature: immutable-audit-trail, Task 6.9: empty chain verifies as INTACT_SIGNED.

An empty chain is vacuously intact and vacuously signed: there is nothing to
walk and nothing whose signature could fail, so verification returns
``INTACT_SIGNED`` with ``total_events == 0``. This holds in three shapes of
"empty":

* an ABSENT file -- an ``AuditChain`` pointed at a path that has never been
  written (an absent file is a valid empty chain, not corruption);
* a file that EXISTS but is zero bytes;
* a file that exists but contains only whitespace / blank lines (which carry no
  event and are skipped by the line reader).

It also holds regardless of whether a signer is configured -- an empty chain is
vacuously signed either way -- so each shape is exercised both with no signer
and with a ``LocalFileSigner``.

Finally, verification is read-only: running ``verify()`` on an absent chain must
not create the file (an absent chain stays absent).

Validates: Requirements 7.6
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fraud_detect.audit.chain import AuditChain
from fraud_detect.audit.signer import LocalFileSigner
from fraud_detect.audit.verify import VerificationStatus


@pytest.fixture
def signer(tmp_path: Path) -> LocalFileSigner:
    """A LocalFileSigner whose key PEM lives outside any chain file path."""
    return LocalFileSigner.generate(str(tmp_path / "key.pem"))


def _assert_empty_intact_signed(result) -> None:
    """Assert a VerificationResult is the vacuous empty-chain INTACT_SIGNED."""
    assert result.status == VerificationStatus.INTACT_SIGNED
    assert result.intact is True
    assert result.signed is True
    assert result.total_events == 0
    assert result.unsigned_events == 0
    assert result.first_broken_seq is None
    assert result.first_bad_signature_seq is None


# ---------------------------------------------------------------------------
# Absent chain file
# ---------------------------------------------------------------------------
def test_absent_chain_without_signer_verifies_intact_signed(tmp_path: Path):
    """An absent chain file (no signer) verifies as INTACT_SIGNED, total 0."""
    chain_path = tmp_path / "audit_chain.jsonl"
    assert not chain_path.exists()

    chain = AuditChain(path=str(chain_path))  # no signer
    result = chain.verify()

    _assert_empty_intact_signed(result)


def test_absent_chain_with_signer_verifies_intact_signed(
    tmp_path: Path, signer: LocalFileSigner
):
    """An absent chain file (with a signer) is still vacuously INTACT_SIGNED."""
    chain_path = tmp_path / "audit_chain.jsonl"
    assert not chain_path.exists()

    chain = AuditChain(path=str(chain_path), signer=signer)
    result = chain.verify()

    _assert_empty_intact_signed(result)


def test_verify_on_absent_chain_does_not_create_the_file(tmp_path: Path):
    """verify() must not create the chain file: an absent chain stays absent.

    Verification is strictly read-only (Requirement 9.1); an absent file is a
    valid empty chain and must not be materialized just by inspecting it.
    """
    chain_path = tmp_path / "audit_chain.jsonl"
    # Constructing the chain creates the parent dir but never the file itself.
    chain = AuditChain(path=str(chain_path))
    assert not chain_path.exists()

    chain.verify()

    assert not chain_path.exists()


def test_verify_on_absent_chain_does_not_create_the_file_with_signer(
    tmp_path: Path, signer: LocalFileSigner
):
    """Same no-side-effect guarantee holds when a signer is configured."""
    chain_path = tmp_path / "audit_chain.jsonl"
    chain = AuditChain(path=str(chain_path), signer=signer)
    assert not chain_path.exists()

    chain.verify()

    assert not chain_path.exists()


# ---------------------------------------------------------------------------
# Empty (zero-byte) chain file
# ---------------------------------------------------------------------------
def test_zero_byte_chain_without_signer_verifies_intact_signed(tmp_path: Path):
    """A zero-byte chain file (no signer) verifies as INTACT_SIGNED, total 0."""
    chain_path = tmp_path / "audit_chain.jsonl"
    chain_path.write_text("", encoding="utf-8")
    assert chain_path.exists()

    chain = AuditChain(path=str(chain_path))  # no signer
    result = chain.verify()

    _assert_empty_intact_signed(result)


def test_zero_byte_chain_with_signer_verifies_intact_signed(
    tmp_path: Path, signer: LocalFileSigner
):
    """A zero-byte chain file (with a signer) is still vacuously INTACT_SIGNED."""
    chain_path = tmp_path / "audit_chain.jsonl"
    chain_path.write_text("", encoding="utf-8")

    chain = AuditChain(path=str(chain_path), signer=signer)
    result = chain.verify()

    _assert_empty_intact_signed(result)


# ---------------------------------------------------------------------------
# Whitespace / blank-line-only chain file
# ---------------------------------------------------------------------------
def test_whitespace_only_chain_without_signer_verifies_intact_signed(
    tmp_path: Path,
):
    """A file of only blank/whitespace lines (no signer) verifies as INTACT_SIGNED.

    Blank and whitespace-only lines carry no event and are skipped by the line
    reader, so the chain is still vacuously empty.
    """
    chain_path = tmp_path / "audit_chain.jsonl"
    chain_path.write_text("\n   \n\t\n  \n", encoding="utf-8")

    chain = AuditChain(path=str(chain_path))  # no signer
    result = chain.verify()

    _assert_empty_intact_signed(result)


def test_whitespace_only_chain_with_signer_verifies_intact_signed(
    tmp_path: Path, signer: LocalFileSigner
):
    """A file of only blank/whitespace lines (with a signer) is still INTACT_SIGNED."""
    chain_path = tmp_path / "audit_chain.jsonl"
    chain_path.write_text("\n   \n\t\n  \n", encoding="utf-8")

    chain = AuditChain(path=str(chain_path), signer=signer)
    result = chain.verify()

    _assert_empty_intact_signed(result)


def test_verify_on_whitespace_only_chain_does_not_modify_the_file(tmp_path: Path):
    """verify() on a whitespace-only chain leaves the file bytes unchanged."""
    chain_path = tmp_path / "audit_chain.jsonl"
    original = "\n   \n\t\n  \n"
    chain_path.write_text(original, encoding="utf-8")

    chain = AuditChain(path=str(chain_path))
    chain.verify()

    assert chain_path.read_text(encoding="utf-8") == original
