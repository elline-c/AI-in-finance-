"""Unit tests for the audit signing backend.

Covers fail-loud construction (``build_signer_from_env`` /
``LocalFileSigner.from_file``) and the local-file semantics of
``LocalFileSigner`` (generate -> load round-trip, stable ``key_id``,
sign/verify).

Validates: Requirements 5.2, 5.3, 6.3, 6.4

These are standard unit tests (specific examples and edge cases) and are
expected to pass against the current implementation. The property-based
signing round-trip test lives separately in ``test_audit_signer_props.py``
(task 3.3) and is not touched here.
"""
from __future__ import annotations

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from fraud_detect.audit.signer import (
    AUDIT_SIGNING_KEY_ENV,
    LocalFileSigner,
    build_signer_from_env,
)


# ---------------------------------------------------------------------------
# build_signer_from_env -- UNSIGNED path (unset env -> None, not an error)
# ---------------------------------------------------------------------------

def test_build_signer_from_env_empty_mapping_returns_none():
    """An explicit empty mapping (var absent) is the UNSIGNED path: None, not an error."""
    assert build_signer_from_env({}) is None


def test_build_signer_from_env_empty_string_returns_none():
    """A set-but-empty value is treated as unset -> UNSIGNED (None)."""
    assert build_signer_from_env({AUDIT_SIGNING_KEY_ENV: ""}) is None
    assert build_signer_from_env({AUDIT_SIGNING_KEY_ENV: "   "}) is None


def test_build_signer_from_env_default_env_absent(monkeypatch):
    """With the var absent from the real process environment, default call returns None."""
    monkeypatch.delenv(AUDIT_SIGNING_KEY_ENV, raising=False)
    assert build_signer_from_env() is None


# ---------------------------------------------------------------------------
# build_signer_from_env -- valid key path -> working signer, stable key_id
# ---------------------------------------------------------------------------

def test_build_signer_from_env_valid_key_returns_working_signer(tmp_path):
    """A configured, valid generated key yields a working signer."""
    key_path = tmp_path / "audit_key.pem"
    LocalFileSigner.generate(str(key_path))

    signer = build_signer_from_env({AUDIT_SIGNING_KEY_ENV: str(key_path)})

    assert isinstance(signer, LocalFileSigner)
    assert signer.key_id.startswith("ed25519:")
    # The signer actually works: sign/verify round-trips under its own key_id.
    data = b"canonical signed payload"
    sig = signer.sign(data)
    assert signer.verify(data, sig, key_id=signer.key_id) is True


def test_build_signer_from_env_key_id_stable_across_reloads(tmp_path):
    """key_id is derived from the public key and stable across loads of the same file."""
    key_path = tmp_path / "audit_key.pem"
    LocalFileSigner.generate(str(key_path))

    first = build_signer_from_env({AUDIT_SIGNING_KEY_ENV: str(key_path)})
    second = build_signer_from_env({AUDIT_SIGNING_KEY_ENV: str(key_path)})

    assert first is not None and second is not None
    assert first.key_id == second.key_id


# ---------------------------------------------------------------------------
# build_signer_from_env -- fail-loud (each raises, NOT a silent None)
# ---------------------------------------------------------------------------

def test_build_signer_from_env_nonexistent_path_raises(tmp_path):
    """A configured path that does not exist fails loudly (no silent UNSIGNED)."""
    missing = tmp_path / "does_not_exist.pem"
    with pytest.raises(ValueError):
        build_signer_from_env({AUDIT_SIGNING_KEY_ENV: str(missing)})


def test_build_signer_from_env_garbage_bytes_raises(tmp_path):
    """A file that is not a valid PEM key fails loudly."""
    bad = tmp_path / "garbage.pem"
    bad.write_bytes(b"this is not a PEM key, just some bytes\x00\x01\x02")
    with pytest.raises(ValueError):
        build_signer_from_env({AUDIT_SIGNING_KEY_ENV: str(bad)})


def test_build_signer_from_env_non_ed25519_key_raises(tmp_path):
    """A valid PEM private key that is NOT Ed25519 (RSA here) fails loudly."""
    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = rsa_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    rsa_path = tmp_path / "rsa_key.pem"
    rsa_path.write_bytes(pem)

    with pytest.raises(ValueError):
        build_signer_from_env({AUDIT_SIGNING_KEY_ENV: str(rsa_path)})


def test_build_signer_from_env_encrypted_pem_raises(tmp_path):
    """An encrypted PEM (needs a password) fails loudly -- the prototype expects
    an unencrypted PKCS#8 PEM."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.BestAvailableEncryption(b"secret-pass"),
    )
    enc_path = tmp_path / "encrypted_key.pem"
    enc_path.write_bytes(pem)

    with pytest.raises(ValueError):
        build_signer_from_env({AUDIT_SIGNING_KEY_ENV: str(enc_path)})


# ---------------------------------------------------------------------------
# LocalFileSigner -- local-file semantics (generate -> load -> round-trip)
# ---------------------------------------------------------------------------

def test_generate_then_from_file_same_key_id(tmp_path):
    """generate() writes a PEM that from_file() loads; both yield the same key_id."""
    key_path = tmp_path / "key.pem"
    generated = LocalFileSigner.generate(str(key_path))

    assert key_path.exists()

    loaded = LocalFileSigner.from_file(str(key_path))
    assert loaded.key_id == generated.key_id


def test_sign_verify_round_trip_across_generate_and_load(tmp_path):
    """A signature made by the generated signer verifies under a freshly loaded one."""
    key_path = tmp_path / "key.pem"
    generated = LocalFileSigner.generate(str(key_path))
    loaded = LocalFileSigner.from_file(str(key_path))

    data = b"audit event signed payload bytes"
    sig = generated.sign(data)

    # Round-trips on the same instance and across a reload of the same key file.
    assert generated.verify(data, sig, key_id=generated.key_id) is True
    assert loaded.verify(data, sig, key_id=loaded.key_id) is True


def test_verify_fails_for_unknown_key_id(tmp_path):
    """verify() returns False (never raises) when key_id resolves to no known key."""
    key_path = tmp_path / "key.pem"
    signer = LocalFileSigner.generate(str(key_path))

    data = b"payload"
    sig = signer.sign(data)
    assert signer.verify(data, sig, key_id="ed25519:deadbeef0000") is False
    assert signer.public_key_for("ed25519:deadbeef0000") is None


def test_verify_fails_for_tampered_data(tmp_path):
    """verify() returns False for a valid signature over different data."""
    key_path = tmp_path / "key.pem"
    signer = LocalFileSigner.generate(str(key_path))

    sig = signer.sign(b"original payload")
    assert signer.verify(b"tampered payload", sig, key_id=signer.key_id) is False


def test_from_file_nonexistent_path_raises(tmp_path):
    """from_file() on a missing path fails loudly with a ValueError."""
    with pytest.raises(ValueError):
        LocalFileSigner.from_file(str(tmp_path / "nope.pem"))
