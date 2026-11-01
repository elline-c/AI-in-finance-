"""New optional ingestion fields must not break old objects or stored data."""
from fraud_detect.models import (
    Case,
    DocumentMetadata,
    IngestionResult,
    SectionSummary,
    Transaction,
)


def test_old_ingestion_result_builds_without_new_fields():
    ing = IngestionResult(
        source_path="x.pdf", page_count=1, raw_text="", metadata=DocumentMetadata()
    )
    assert ing.layout_profile is None
    assert ing.parse_coverage is None
    assert ing.sections == []


def test_old_transaction_defaults():
    t = Transaction(index=0, amount=-1.5)
    assert t.value_date_raw is None
    assert t.direction_method is None


def test_new_fields_round_trip():
    ing = IngestionResult(
        source_path="x.pdf",
        page_count=1,
        raw_text="",
        metadata=DocumentMetadata(),
        layout_profile="two_col_dmon",
        parse_coverage=0.97,
        sections=[SectionSummary(page_start=1, page_end=1, opening=1.0, closing=2.0,
                                 total_deposits=3.0, total_withdrawals=2.0)],
        transactions=[Transaction(index=0, value_date_raw="5 Apr 2026",
                                  direction_method="column")],
    )
    back = IngestionResult.model_validate_json(ing.model_dump_json())
    assert back == ing


def test_stored_case_json_without_new_fields_validates():
    stored = {
        "case_id": "c1",
        "report": {
            "case_id": "c1", "source_path": "a.pdf", "risk_score": 10,
            "risk_band": "LOW", "confidence_score": 80, "confidence_band": "HIGH",
            "routing": "AUTO_CLEAR", "routing_reason": "ok",
            "transactions": [{"index": 0, "date": "2026-01-02", "date_raw": "02/01/2026",
                              "description": "x", "amount": 5.0, "balance": 10.0, "page": 1}],
        },
        "review": {"case_id": "c1"},
    }
    case = Case.model_validate(stored)
    assert case.report.transactions[0].direction_method is None
