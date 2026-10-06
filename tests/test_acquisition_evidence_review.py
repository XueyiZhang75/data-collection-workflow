"""Independent source-assertion review regressions; no external I/O."""
import hashlib
import socket

import pytest

from data_collection_workflow.source_assertions import date_intervals, typed_date_support
from data_collection_workflow.evidence_qualification import assess_record_evidence


@pytest.fixture(autouse=True)
def offline_evidence(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    def forbidden(*args, **kwargs):
        raise AssertionError("Review tests forbid external network calls")
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def qualify(text, **facts):
    digest = hashlib.sha256(text.encode()).hexdigest()
    document = {"source_id": "s", "document_id": "d", "clean_text": text,
                "content_hash": digest, "text_hash": digest}
    chunk = {"source_id": "s", "document_id": "d", "document_hash": digest,
             "chunk_id": "c", "text": text, "char_start": 0, "char_end": len(text)}
    row = {"source_id": "s", "supporting_chunk_id": "c", **facts}
    return assess_record_evidence(row, contract={}, evidence_index={
        "documents": [document], "evidence_chunks": {"c": chunk}})


@pytest.mark.parametrize("text", [
    "En 2025, France n\u2019a pas signal\u00e9 32 456 cas confirm\u00e9s de chikungunya.",
    "En 2025, France pourrait signaler 32 456 cas confirm\u00e9s de chikungunya.",
    "En 2025, si France signalait 32 456 cas confirm\u00e9s de chikungunya, une alerte serait lanc\u00e9e.",
])
def test_french_nonaffirmative_counts_cannot_qualify(text):
    result = qualify(text, disease="chikungunya", country="France",
                     reporting_period="2025", cases_confirmed=32456)
    assert result.status == "candidate"
    assert not next(field for field in result.field_evidence if field.field == "cases_confirmed").supported


def test_french_affirmative_count_stays_qualified():
    text = "En 2025, France a signal\u00e9 32 456 cas confirm\u00e9s de chikungunya."
    result = qualify(text, disease="chikungunya", country="France",
                     reporting_period="2025", cases_confirmed=32456)
    assert result.status == "qualified", result.reasons


@pytest.mark.parametrize("field,value", [
    ("event_start_date", "2025-01-01"), ("event_end_date", "2025-05-23"),
])
def test_case_observation_interval_does_not_establish_event_dates(field, value):
    text = "Canada reported 3,011 confirmed measles cases between 1 January and 23 May 2025."
    result = qualify(text, disease="measles", country="Canada", cases_confirmed=3011,
                     reporting_period="1 January to 23 May 2025", **{field: value})
    assert result.status == "candidate"
    assert not next(evidence for evidence in result.field_evidence if evidence.field == field).supported


def test_same_month_mixed_precision_interval_never_fills_the_missing_day():
    text = "Canada reported 3,011 confirmed measles cases between January and 23 January 2025."
    assert date_intervals(text) == [((2025, 1, None), (2025, 1, 23))]
    assert typed_date_support("metric_period_start", "2025-01-01", text) is False
    assert typed_date_support("metric_period_end", "2025-01-23", text) is True


def test_manuscript_line_sequence_crossing_999_does_not_become_a_case_count():
    from data_collection_workflow.source_assertions import count_mentions, normalized_quote_spans
    text = "999 Canada reported 3,011 confirmed measles\n1000 cases during 2025."
    quote = "Canada reported 3,011 confirmed measles cases during 2025."
    assert [mention["value"] for mention in count_mentions(text)] == [3011]
    spans = normalized_quote_spans(text, quote)
    assert len(spans) == 1
    assert text[spans[0][0]:spans[0][1]].startswith("Canada reported 3,011")
    result = qualify(text, disease="measles", country="Canada", reporting_period="2025", cases_unspecified=1000)
    assert result.status == "candidate"


def test_consecutive_observation_year_lines_are_not_deleted_as_manuscript_labels():
    from data_collection_workflow.source_assertions import analysis_projection
    text = "2024 Canada reported 3,011 confirmed measles cases.\n2025 Canada reported 2,000 confirmed measles cases."
    assert analysis_projection(text).text == text


def test_french_place_name_containing_pas_does_not_negate_its_positive_count():
    text = "En 2025, Pas-de-Calais (France) a signal\u00e9 32 456 cas confirm\u00e9s de chikungunya."
    result = qualify(text, disease="chikungunya", country="France", subnational_location="Pas-de-Calais",
                     reporting_period="2025", cases_confirmed=32456)
    assert result.status == "qualified", result.reasons


@pytest.mark.parametrize("year,expected", [(2025, None), (2024, "2024-02-29")])
def test_inferred_interval_year_is_calendar_valid_before_producing_date_fields(year, expected):
    from data_collection_workflow.source_assertions import observation_dates
    text = f"Canada reported 3,011 confirmed measles cases between 29 February and 23 May {year}."
    dates = observation_dates(text)
    assert dates.get("metric_period_start") == expected
