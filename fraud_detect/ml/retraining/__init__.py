"""ML retraining pipeline (Stage B). Synthetic data only; see .kiro/specs/ml-retraining.

Importing this package imports nothing heavy: no torch, sklearn or pdfplumber.
"""
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_INTEGRITY = 2        # integrity gate refused, or bad arguments
EXIT_WAITING = 3          # waivable gate waiting for a decision
EXIT_IMPLAUSIBLE = 4      # implausible measurement
