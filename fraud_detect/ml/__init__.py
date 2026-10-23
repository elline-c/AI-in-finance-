"""Machine-learning extensions for the fraud-detection pipeline.

Two independent, governed models live here:

    * ``transaction``  — row-level AML/transaction-fraud classifier trained on the
      synthetic transaction dataset (``is_fraud`` target, ~1% positive).
    * ``statement``    — document-level statement-forgery classifier trained on the
      synthetic statement dataset (``is_tampered`` target, 50% positive).

Both models emit explainable ``Finding`` objects through their respective nodes
and NEVER auto-reject. They add evidence that a human reviewer weighs, in line
with the MAS-FEAT governance stance used throughout the pipeline.

Design rules enforced here (see dataset data cards):
    * Train/val/test use the dataset-provided ``split`` column — never re-split
      by row (that leaks account history / document layout across splits).
    * Label-bearing columns are dropped from features (``fraud_pattern``,
      ``ring_id``, ``eval_tags``; ``tamper_type`` and artifact flags for the
      statement model).
    * Scoring uses PR-AUC + precision/recall @ top-1% for the imbalanced
      transaction task, and in-layout vs out-of-distribution layout scores for
      the statement task.
"""
