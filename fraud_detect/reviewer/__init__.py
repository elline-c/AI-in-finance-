"""Explanation layer for the screening pipeline (runs after Node 4, never changes a score).

* ``breakdown``  deterministic risk / confidence / routing explanation (always on, free)
* ``baseline``   template narrative built from that breakdown (the fallback and the yardstick)
* ``validator``  checks any narrative against the case: citations, numbers, consistency
* ``explainer``  optional local-LLM reviewer with retrieval, behind ``FRAUD_ENABLE_LLM_REVIEWER``
* ``evaluation`` the checks and pass criteria the LLM reviewer is measured against

This package sits outside ``fraud_detect/nodes/`` on purpose: that folder is fingerprinted
by the ML-retraining work and must stay unchanged.
"""
