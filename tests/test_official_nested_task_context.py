"""The official extractor accepts raw task state without changing its caller."""
import copy

import pytest

from data_collection_workflow.config import load_structured_extraction_policy
from data_collection_workflow.models import StructuredExtractionPolicy
from data_collection_workflow.nodes.extraction import extract_official_outbreak_records_from_chunks


@pytest.mark.parametrize("disease", ["mpox", "measles"])
def test_nested_task_context_preserves_source_backed_disease_and_caller(monkeypatch, disease):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    context = {
        "structured_task": {"disease": disease, "location": "Sierra Leone",
                            "start_date": "2025-01-01", "end_date": "2025-12-31"},
        "disease_intelligence": {"disease_standard_name": disease, "aliases": [disease]},
        "source_registry_by_id": {"retained": {"source_id": "retained"}},
        "documents_by_source_id": {"retained": [{"document_id": "d1"}]},
    }
    original = copy.deepcopy(context)
    text = f"As of 2 May 2025, Sierra Leone reported 12 confirmed {disease} cases."
    chunk = {"chunk_id": "c", "source_id": "s", "text": text,
             "title": "Disease Outbreak News", "source_type": "WHO_DON", "publisher": "WHO",
             "source_url": "https://www.who.int/emergencies/disease-outbreak-news/item/test"}
    policy = StructuredExtractionPolicy.model_validate(load_structured_extraction_policy())
    records, diagnostics = extract_official_outbreak_records_from_chunks([chunk], policy=policy, context=context)
    assert len(records) == 1, diagnostics
    assert records[0].disease == disease
    assert records[0].cases_confirmed == 12
    assert disease in records[0].evidence_quote
    assert context == original


def test_nested_task_disease_does_not_make_unlabelled_source_relevant(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    context = {"structured_task": {"disease": "mpox", "location": "Sierra Leone"},
               "disease_intelligence": {"disease_standard_name": "mpox", "aliases": ["mpox"]}}
    original = copy.deepcopy(context)
    chunk = {"chunk_id": "c", "source_id": "s", "text": "As of 2 May 2025, Sierra Leone reported 12 confirmed cases.",
             "title": "Disease Outbreak News", "source_type": "WHO_DON", "publisher": "WHO",
             "source_url": "https://www.who.int/emergencies/disease-outbreak-news/item/test"}
    policy = StructuredExtractionPolicy.model_validate(load_structured_extraction_policy())
    records, diagnostics = extract_official_outbreak_records_from_chunks([chunk], policy=policy, context=context)
    assert records == []
    assert diagnostics[0]["failure_substage"] == "source_found_but_no_relevant_chunks"
    assert context == original
