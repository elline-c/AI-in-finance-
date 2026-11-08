"""Unit tests for the verification CLI exit-code contract (task 7.2).

These exercise :func:`fraud_detect.audit.verify.main` directly -- it accepts an
``argv`` list and returns an ``int`` exit code, so we call it in-process rather
than spawning ``python -m fraud_detect.audit.verify`` as a subprocess. Each test
drives ``main`` to exactly one of the four CLI outcomes and asserts the named
exit-code constant it must return:

* INTACT_SIGNED      -> EXIT_INTACT_SIGNED (0)      -- the only success code
* INTACT_UNSIGNED    -> EXIT_INTACT_UNSIGNED (2)    -- distinct, never "signed"
* BROKEN_CHAIN       -> EXIT_INTEGRITY_FAILED (1)
* SIGNATURE_INVALID  -> EXIT_INTEGRITY_FAILED (1)
* (load error)       -> EXIT_ERROR (3)              -- fail-loud on a bad key

``main`` builds its signer from the environment via ``build_signer_from_env``,
so to get INTACT_SIGNED the chain must be signed with the SAME key that
``AUDIT_SIGNING_KEY_PATH`` points at. We therefore set the env var to a generated
key first, build the signer via ``build_signer_from_env`` (the exact path ``main``
uses), write the chain with THAT signer, then call ``main`` -- so the key that
signs is the key that verifies.

**Validates: Requirements 8.2, 8.3, 8.4**
"""
from __future__ import annotations

import json

from fraud_detect.audit.chain import AuditChain
from fraud_detect.audit.signer import (
    AUDIT_SIGNING_KEY_ENV,
    LocalFileSigner,
    build_signer_from_env,
)
from fraud_detect.audit.verify import (
    EXIT_ERROR,
    EXIT_INTACT_SIGNED,
    EXIT_INTACT_UNSIGNED,
    EXIT_INTEGRITY_FAILED,
    main,
)


def _build_signed_chain(tmp_path, monkeypatch):
    """Generate a key, point the env at it, and write a signed 3-event chain.

    Returns the chain-file path (str). The signer used to write the chain is
    built via ``build_signer_from_env`` -- the same resolution ``main`` performs
    -- so the chain's ``key_id`` matches the key ``main`` will verify with.
    """
    key_path = tmp_path / "audit_key.pem"
    LocalFileSigner.generate(str(key_path))  # write the private key PEM
    monkeypatch.setenv(AUDIT_SIGNING_KEY_ENV, str(key_path))

    signer = build_signer_from_env()
    assert signer is not None  # env is set -> a signer is built

    chain_path = tmp_path / "audit_chain.jsonl"
    chain = AuditChain(str(chain_path), signer=signer)
    chain.append(case_id="C1", event="SCREENED", detail="first")
    chain.append(case_id="C1", event="HUMAN_APPROVED", detail="second")
    chain.append(case_id="C2", event="SCREENED", detail="third")
    return str(chain_path)


# --------------------------------------------------------------------------- #
# INTACT_SIGNED -> exit 0 (Requirement 8.2)
# --------------------------------------------------------------------------- #
def test_intact_signed_exits_zero(tmp_path, monkeypatch):
    """A signed chain verified with the configured key exits 0 (the only OK)."""
    chain_path = _build_signed_chain(tmp_path, monkeypatch)

    assert main(["--path", chain_path]) == EXIT_INTACT_SIGNED
    assert EXIT_INTACT_SIGNED == 0


# --------------------------------------------------------------------------- #
# INTACT_UNSIGNED -> exit 2, distinct from 0 and 1 (Requirement 8.4)
# --------------------------------------------------------------------------- #
def test_intact_unsigned_exits_distinct_nonzero(tmp_path, monkeypatch, capsys):
    """An unsigned chain surfaces distinctly: exit 2, not 0 and not 1."""
    # No signing key configured -> build_signer_from_env() returns None and the
    # chain is written UNSIGNED.
    monkeypatch.delenv(AUDIT_SIGNING_KEY_ENV, raising=False)
    assert build_signer_from_env() is None

    chain_path = tmp_path / "audit_chain.jsonl"
    chain = AuditChain(str(chain_path), signer=None)
    chain.append(case_id="C1", event="SCREENED", detail="first")
    chain.append(case_id="C1", event="HUMAN_REJECTED", detail="second")

    code = main(["--path", str(chain_path)])

    assert code == EXIT_INTACT_UNSIGNED
    # Distinct from the signed-OK and integrity-failed codes.
    assert code != EXIT_INTACT_SIGNED
    assert code != EXIT_INTEGRITY_FAILED

    # The UNSIGNED state is surfaced loudly so it is never read as signed-OK.
    err = capsys.readouterr().err
    assert "UNSIGNED" in err


# --------------------------------------------------------------------------- #
# BROKEN_CHAIN -> exit 1 (Requirement 8.3)
# --------------------------------------------------------------------------- #
def test_broken_chain_exits_integrity_failed(tmp_path, monkeypatch):
    """Corrupting a line on disk yields a BROKEN_CHAIN -> non-zero exit 1."""
    chain_path = _build_signed_chain(tmp_path, monkeypatch)

    # Tamper the first event's content without fixing its hash -> hash mismatch.
    lines = chain_path_lines(chain_path)
    first = json.loads(lines[0])
    first["detail"] = "TAMPERED"
    lines[0] = json.dumps(first)
    _write_lines(chain_path, lines)

    assert main(["--path", chain_path]) == EXIT_INTEGRITY_FAILED
    assert EXIT_INTEGRITY_FAILED == 1


# --------------------------------------------------------------------------- #
# SIGNATURE_INVALID -> exit 1 (Requirement 8.3)
# --------------------------------------------------------------------------- #
def test_signature_invalid_exits_integrity_failed(tmp_path, monkeypatch):
    """Tampering a signature (hash chain intact) yields SIGNATURE_INVALID -> 1."""
    chain_path = _build_signed_chain(tmp_path, monkeypatch)

    # Flip a character in the LAST event's base64 signature, leaving every
    # hash-chain field untouched so pass 1 (hash walk) stays clean and only the
    # signature check fails.
    lines = chain_path_lines(chain_path)
    last = json.loads(lines[-1])
    sig = last["signature"]
    assert sig is not None
    # Swap the first character for a different valid base64 char.
    flipped = ("B" if sig[0] != "B" else "C") + sig[1:]
    last["signature"] = flipped
    lines[-1] = json.dumps(last)
    _write_lines(chain_path, lines)

    assert main(["--path", chain_path]) == EXIT_INTEGRITY_FAILED


# --------------------------------------------------------------------------- #
# ERROR -> exit 3 (fail-loud; Requirements 6.3/6.4 feeding the 8.x contract)
# --------------------------------------------------------------------------- #
def test_bad_signing_key_exits_error(tmp_path, monkeypatch, capsys):
    """A configured-but-unloadable key makes main fail loudly with exit 3."""
    # Point the env at a path that is not a readable Ed25519 PEM key so
    # build_signer_from_env() raises inside main().
    garbage = tmp_path / "not_a_key.pem"
    garbage.write_text("this is not a valid PEM private key\n", encoding="utf-8")
    monkeypatch.setenv(AUDIT_SIGNING_KEY_ENV, str(garbage))

    # A chain path still has to be passed; its contents are irrelevant because
    # main returns before reading it.
    any_path = tmp_path / "audit_chain.jsonl"

    code = main(["--path", str(any_path)])

    assert code == EXIT_ERROR
    assert EXIT_ERROR == 3
    # The failure is surfaced to stderr (fail-loud, never a silent pass).
    assert capsys.readouterr().err.strip() != ""


# --------------------------------------------------------------------------- #
# Small on-disk helpers
# --------------------------------------------------------------------------- #
def chain_path_lines(path):
    """Read the chain file into a list of non-empty JSON lines (strings)."""
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _write_lines(path, lines):
    """Overwrite the chain file with ``lines`` (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
