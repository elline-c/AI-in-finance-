"""Property-based test for ``AuditChain.append``'s append-only discipline.

Feature: immutable-audit-trail, Property 1: Append preserves the existing chain
as a prefix.

Appending an event to the chain leaves every prior line byte-identical and adds
exactly one new trailing line. In other words, ``append`` only ever grows the
file by one line and never rewrites, reorders, or truncates earlier bytes -- the
core of the "append-only, never rewritten in place" guarantee.

Validates: Requirements 1.1

Strategy: Hypothesis generates a list of ``(case_id, event, detail)`` tuples.
The test builds a chain on a fresh temp file, appending the tuples one at a
time. Before each append it captures the file's raw bytes; after each append it
asserts (a) the file has exactly N lines after N appends and (b) the first N-1
lines are byte-identical to what they were immediately before the append.

Isolation note: Hypothesis reuses the pytest ``tmp_path`` across examples, which
would let one example's chain file clobber the next. To keep examples
independent, each example creates its OWN fresh temporary directory inside the
test body (``tempfile.TemporaryDirectory``) and tears it down before returning,
so no state leaks between examples. Both UNSIGNED and signed modes are covered
via parametrization: signing adds ``signature``/``key_id`` to each line but must
not change the append-only behavior, and a module-scoped signer keeps key
generation out of the per-example hot path.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fraud_detect.audit.chain import AuditChain
from fraud_detect.audit.signer import LocalFileSigner

# A single event spec: (case_id, event, detail). Non-empty, unicode-capable
# strings so the generators exercise the real payload space (case_id must be
# non-empty to be meaningful; event/detail include arbitrary text).
_event_specs = st.lists(
    st.tuples(
        st.text(min_size=1, max_size=40),   # case_id
        st.text(min_size=0, max_size=40),   # event
        st.text(min_size=0, max_size=200),  # detail
    ),
    min_size=1,
    max_size=25,
)


@pytest.fixture(scope="module")
def module_signer() -> LocalFileSigner:
    """A single Ed25519 signer generated once for the whole module.

    Key generation is comparatively expensive; generating it per Hypothesis
    example would dominate the runtime. The signed-mode property reuses this one
    signer across all examples -- the property under test (append-only) does not
    depend on which key is used.
    """
    with tempfile.TemporaryDirectory() as tmp:
        yield LocalFileSigner.generate(str(Path(tmp) / "key.pem"))


def _run_append_prefix_property(
    specs: List[Tuple[str, str, str]],
    signer: Optional[LocalFileSigner],
) -> None:
    """Append ``specs`` one at a time, asserting the append-only invariant.

    A fresh temp directory is created per call so Hypothesis examples never
    share a chain file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        chain_path = Path(tmp) / "audit_chain.jsonl"
        chain = AuditChain(str(chain_path), signer=signer)

        # Before the first append the file does not exist yet (an absent file is
        # a valid empty chain); treat that as zero bytes.
        for n, (case_id, event, detail) in enumerate(specs, start=1):
            before = chain_path.read_bytes() if chain_path.exists() else b""
            before_lines = before.splitlines(keepends=True)

            chain.append(case_id=case_id, event=event, detail=detail)

            after = chain_path.read_bytes()
            after_lines = after.splitlines(keepends=True)

            # (a) Exactly N lines after N appends -- append added exactly one.
            assert len(after_lines) == n, (
                f"expected {n} lines after {n} appends, got {len(after_lines)}"
            )

            # (b) The first N-1 lines are byte-identical to before this append:
            #     earlier bytes were never rewritten, reordered, or truncated.
            assert after_lines[: n - 1] == before_lines, (
                "append altered previously-written lines"
            )

            # And the prior content is an exact byte prefix of the new content.
            assert after.startswith(before), "prior bytes are not an exact prefix"

            # The newly added bytes are exactly the single new trailing line.
            assert after_lines[n - 1] == after[len(before):]


@settings(max_examples=120)
@given(specs=_event_specs)
def test_append_preserves_prefix_unsigned(specs):
    """Property 1 (UNSIGNED): with no signer, each append adds exactly one line
    and leaves every earlier line byte-identical."""
    _run_append_prefix_property(specs, signer=None)


@settings(max_examples=120, deadline=None)
@given(specs=_event_specs)
def test_append_preserves_prefix_signed(specs, module_signer):
    """Property 1 (signed): with a configured signer, each signed append still
    adds exactly one line and never rewrites earlier lines."""
    _run_append_prefix_property(specs, signer=module_signer)
