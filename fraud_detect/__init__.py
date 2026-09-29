"""Bank statement tampering & fraud detection prototype.

A 4-agent pipeline that screens corporate loan bank statements for signs of
tampering and suspicious activity, produces an explainable risk + confidence
score, and routes to a human-in-the-loop reviewer when needed.

Agents (worker nodes):
    Node 1  Ingestion & Preprocessing
    Node 2  Visual Forensics
    Node 3  Financial Logic & Validation
    Node 4  Risk Scoring & Synthesis
"""

__version__ = "0.1.0"
