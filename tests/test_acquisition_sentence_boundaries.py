"""Identifiers are not sentence boundaries; run-on wrong-disease evidence stays isolated."""
import pytest
from data_collection_workflow.disease_relevance import (
    _evidence_sentences, assess_record_disease_compatibility, build_disease_relevance_context,
)
from data_collection_workflow.nodes.extraction import schema_validation_and_repair
from test_acquisition_evidence import assess


@pytest.mark.parametrize("identifier", ["A.2.2", "B.1", "BA.2.86"])
@pytest.mark.parametrize("disease", ["mpox", "measles"])
def test_dotted_identifier_preserves_source_sentence(identifier, disease):
    text = f"Canada reported {disease} lineage {identifier} infections with over 5000 cases during 2025."
    assert _evidence_sentences(text) == [text]
    result = assess_record_disease_compatibility(
        {"disease": disease, "cases_unspecified": 5000, "evidence_quote": text},
        build_disease_relevance_context({"structured_task": {"disease": disease}}),
    )
    assert not result["reject_record"], result


@pytest.mark.parametrize("number", ["5.2", "1,234.5", "0.03"])
def test_decimal_content_is_not_rewritten_or_split(number):
    text = f"The {number} percent rate was reported for measles."
    assert _evidence_sentences(text) == [text]


@pytest.mark.parametrize("separator", [".", "!", "?", ";"])
def test_run_on_case_death_and_wrong_disease_boundaries_stay_separate(separator):
    first = f"Hantavirus was monitored{separator}"
    second = f"Influenza caused 12 cases and 4 deaths{separator}"
    third = "Hantavirus monitoring continued."
    text = first + second + third
    assert _evidence_sentences(text) == [first, second, third]
    result = assess_record_disease_compatibility(
        {"disease": "hantavirus", "cases_unspecified": 12, "deaths": 4, "evidence_quote": text},
        build_disease_relevance_context({"structured_task": {"disease": "hantavirus"}}),
    )
    assert result["reject_record"]
    assert "local_evidence_disease_mismatch" in result["reason"]


def test_run_on_sentence_starting_with_count_still_splits():
    assert _evidence_sentences("Measles was reported.12 cases were confirmed.") == [
        "Measles was reported.", "12 cases were confirmed."]


@pytest.mark.parametrize("disease,identifier", [("mpox", "A.2.2"), ("measles", "B.1.2")])
def test_real_schema_preserves_dotted_identifier_candidate_without_exactifying_lower_bound(monkeypatch, disease, identifier):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    text = f"Canada reported {disease} lineage {identifier} infections with over 5000 cases during 2025."
    record = {"record_id": "candidate", "source_id": "source", "supporting_chunk_id": "chunk",
              "disease": disease, "country": "Canada", "reporting_period": "2025",
              "cases_unspecified": 5000, "evidence_quote": text,
              "source_type": "unknown", "source_url": "https://surveillance.example/report", "extraction_method": "llm_structured_output_extractor"}
    state = {"structured_task": {"disease": disease, "location": "Canada", "start_date": "2025-01-01", "end_date": "2025-12-31"},
             "collection_spec": {"disease": disease, "geography": "Canada", "target_population": "humans"},
             "raw_records": [record], "evidence_chunks": [{"chunk_id": "chunk", "source_id": "source", "text": text}],
             "source_registry": [{"source_id": "source", "url": record["source_url"]}],
             "human_review_queue": [], "collection_trace": []}
    result = schema_validation_and_repair(state)
    assert len(result["validated_records"]) == 1, result["schema_validation_summary"]
    kept = result["validated_records"][0]
    assert kept["record_id"] == "candidate"
    assert kept["cases_unspecified"] == 5000
    qualification = assess(text, disease=disease, country="Canada", reporting_period="2025", cases_unspecified=5000)
    assert qualification.status == "candidate"
    assert not next(item.supported for item in qualification.field_evidence if item.field == "cases_unspecified")
