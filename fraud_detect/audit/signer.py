"""Pluggable signing backend for the audit chain.

The audit chain signs each event with Ed25519 over a canonical payload that
commits to the chained ``entry_hash`` (see
:mod:`fraud_detect.audit.canonical`). Signing is encapsulated behind the
:class:`Signer` protocol so the prototype's local-file key
(:class:`LocalFileSigner`) can be swapped for a production KMS/HSM-backed
implementation without touching the chain or the event model -- only the
``Signer`` construction changes.

**Trust boundary.** The private key material is owned entirely by the ``Signer``
and is never exposed through the public API or written back into the data path.
For the prototype the key lives in a local PEM file whose path comes from the
environment (wired up in task 3.2); that path MUST live outside the ``data/``
directory so the key is not co-located with the log it protects. In production
the key MUST move to a KMS/HSM where it never leaves the security boundary --
reading the local key file is enough to forge signatures.

**Key format.** :class:`LocalFileSigner` loads an *unencrypted PEM* Ed25519
private key (PKCS#8, ``-----BEGIN PRIVATE KEY-----``) via
:func:`cryptography.hazmat.primitives.serialization.load_pem_private_key`. PEM
is the on-disk default; other encodings (raw/OpenSSH) are intentionally not
supported to keep the prototype simple.

**key_id.** The identifier recorded alongside every signature is derived
deterministically from the *public* key::

    key_id = "ed25519:" + sha256(public_key_raw_bytes).hexdigest()[:12]

Using the Raw-encoding public-key bytes makes the fingerprint stable across
loads of the same key and independent of PEM formatting. Storing ``key_id`` per
event lets verification resolve the right public key even after a future key
rotation.
"""

from __future__ import annotations

import hashlib
import os
from typing import Mapping, Optional, Protocol, runtime_checkable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

__all__ = ["Signer", "LocalFileSigner", "build_signer_from_env", "AUDIT_SIGNING_KEY_ENV"]

#: Name of the environment variable that points at the Ed25519 private-key PEM
#: file used to sign audit events. Unset/empty => UNSIGNED mode (not an error).
AUDIT_SIGNING_KEY_ENV = "AUDIT_SIGNING_KEY_PATH"


def _fingerprint(public_key: Ed25519PublicKey) -> str:
    """Derive a stable ``key_id`` from an Ed25519 public key.

    Uses the Raw-encoding public-key bytes (the 32-byte Ed25519 public key) so
    the fingerprint is independent of on-disk/PEM formatting and stable across
    loads of the same key.
    """
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return "ed25519:" + hashlib.sha256(raw).hexdigest()[:12]


@runtime_checkable
class Signer(Protocol):
    """Signing backend interface.

    Implementations own the private key material and expose only sign/verify and
    public-key resolution. The prototype backend is :class:`LocalFileSigner`; a
    production backend (KMS/HSM) implements the same protocol so the audit chain
    is unchanged.
    """

    @property
    def key_id(self) -> str:
        """Stable identifier recorded alongside each signature this signer produces."""
        ...

    def sign(self, data: bytes) -> bytes:
        """Produce an Ed25519 signature over ``data``."""
        ...

    def verify(self, data: bytes, signature: bytes, *, key_id: str) -> bool:
        """Return ``True`` iff ``signature`` is a valid Ed25519 signature over
        ``data`` under the public key identified by ``key_id``.

        Selecting the key by ``key_id`` lets a rotated key set still verify
        events signed by an older key. Returns ``False`` (never raises) when the
        signature is invalid or ``key_id`` resolves to no known key.
        """
        ...

    def public_key_for(self, key_id: str) -> Optional[Ed25519PublicKey]:
        """Resolve the public key for ``key_id``; ``None`` if unknown."""
        ...


class LocalFileSigner:
    """Prototype :class:`Signer` backed by an Ed25519 private key in a PEM file.

    Loads an unencrypted PKCS#8 PEM Ed25519 private key, derives a deterministic
    ``key_id`` from the matching public key, and signs/verifies with it. The
    private key is held internally and never exposed through the public API.
    """

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        """Build a signer around an already-loaded Ed25519 private key.

        Most callers use :meth:`from_file` (or :meth:`generate` in tests)
        rather than constructing from a key object directly.
        """
        if not isinstance(private_key, Ed25519PrivateKey):
            raise TypeError("private_key must be an Ed25519PrivateKey")
        self._private_key = private_key
        self._public_key = private_key.public_key()
        self._key_id = _fingerprint(self._public_key)

    @classmethod
    def from_file(cls, key_path: str) -> "LocalFileSigner":
        """Load an Ed25519 private key from an unencrypted PEM file at ``key_path``.

        Raises a clear error if the file cannot be read, is not valid PEM, is not
        an Ed25519 key, or is encrypted (the prototype expects an unencrypted
        PKCS#8 PEM private key).
        """
        try:
            with open(key_path, "rb") as fh:
                pem_bytes = fh.read()
        except OSError as exc:
            raise ValueError(
                f"could not read signing key file {key_path!r}: {exc}"
            ) from exc

        try:
            private_key = serialization.load_pem_private_key(
                pem_bytes, password=None
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"could not parse signing key file {key_path!r} as an "
                f"unencrypted PEM private key: {exc}"
            ) from exc

        if not isinstance(private_key, Ed25519PrivateKey):
            raise ValueError(
                f"signing key file {key_path!r} is not an Ed25519 private key "
                f"(got {type(private_key).__name__})"
            )

        return cls(private_key)

    @classmethod
    def generate(cls, key_path: str) -> "LocalFileSigner":
        """Generate a fresh Ed25519 keypair, write the private key as PEM to
        ``key_path``, and return a signer for it.

        Intended for tests and local bootstrapping. The private key is written
        as an unencrypted PKCS#8 PEM -- the same format :meth:`from_file` reads.
        """
        private_key = Ed25519PrivateKey.generate()
        pem_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        with open(key_path, "wb") as fh:
            fh.write(pem_bytes)
        return cls(private_key)

    @property
    def key_id(self) -> str:
        """Stable identifier for this signer's key (fingerprint of the public key)."""
        return self._key_id

    def sign(self, data: bytes) -> bytes:
        """Return the raw Ed25519 signature bytes over ``data``."""
        return self._private_key.sign(data)

    def verify(self, data: bytes, signature: bytes, *, key_id: str) -> bool:
        """Verify ``signature`` over ``data`` under the key named by ``key_id``.

        Returns ``False`` (never raises) for an invalid signature or an unknown
        ``key_id``.
        """
        public_key = self.public_key_for(key_id)
        if public_key is None:
            return False
        try:
            public_key.verify(signature, data)
        except InvalidSignature:
            return False
        return True

    def public_key_for(self, key_id: str) -> Optional[Ed25519PublicKey]:
        """Return this signer's public key when ``key_id`` matches, else ``None``."""
        if key_id == self._key_id:
            return self._public_key
        return None


def build_signer_from_env(
    env: Optional[Mapping[str, str]] = None,
) -> Optional[Signer]:
    """Build a :class:`Signer` from the environment, fail-loud on a bad key.

    Reads the ``AUDIT_SIGNING_KEY_PATH`` environment variable and returns:

    * ``None`` when the variable is **unset or empty** -- this is the UNSIGNED
      path. The caller (``AuditChain``) still writes a valid hash chain but marks
      events UNSIGNED. This is a normal, supported mode and is **not** an error
      (see design Error Handling Scenario 6).
    * a :class:`LocalFileSigner` loaded from the configured path when the
      variable **is** set.

    **Fail-loud contract.** When the key path is configured but the file is
    missing, unreadable, not a valid Ed25519 PEM key, or encrypted,
    :meth:`LocalFileSigner.from_file` raises :class:`ValueError` with a
    descriptive message. That error is deliberately **not** swallowed: a
    configured-but-unloadable key must NEVER silently fall back to UNSIGNED
    (design Error Handling Scenario 5; Requirements 6.3, 6.4). The
    misconfiguration should be fixed, not masked.

    :param env: Optional mapping to read the variable from (defaults to
        :data:`os.environ`). Injecting a mapping keeps the function testable
        without mutating the real process environment.

    **Key-location constraint.** The configured key path MUST live **outside**
    the ``data/`` directory so the signing key is never co-located with the
    append-only log it protects (Requirement 5.3). This is a documented
    deployment constraint; it is not enforced here (reading the key from an
    unusual location is the operator's responsibility, and the production target
    is a KMS/HSM where the key never touches the filesystem at all).
    """
    if env is None:
        env = os.environ

    key_path = env.get(AUDIT_SIGNING_KEY_ENV)
    if not key_path or not key_path.strip():
        # Unset or empty -> UNSIGNED mode. Not an error.
        return None

    # Configured: build the signer. Any failure to load the key propagates
    # (fail-loud) -- we never silently downgrade to UNSIGNED.
    return LocalFileSigner.from_file(key_path)
