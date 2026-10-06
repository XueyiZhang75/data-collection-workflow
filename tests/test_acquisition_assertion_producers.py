"""The actual deterministic producer preserves typed local source facts."""
import pytest

from data_collection_workflow.config import load_structured_extraction_policy
from data_collection_workflow.models import StructuredExtractionPolicy
from data_collection_workflow.nodes.extraction import _official_outbreak_record_from_chunk, _official_dates


@pytest.fixture(autouse=True)
def evidence(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")


@pytest.mark.parametrize("text,expected_start,expected_end", [
    ("Between January and 23 May 2025, Canada reported 3,011 confirmed pertussis cases.", None, "2025-05-23"),
    ("France a notifi\u00e9 32 456 cas confirm\u00e9s de chikungunya du 1er janvier au 23 mai 2025.", "2025-01-01", "2025-05-23"),
])
def test_rule_recovers_only_source_precision_observation_dates(text, expected_start, expected_end):
    result = _official_dates(text, "Published in 2026", {})
    assert result.get("metric_period_start") == expected_start
    assert result.get("metric_period_end") == expected_end
    assert result.get("reporting_period") in text
    assert result["event_start_date"] is None and result["event_end_date"] is None


@pytest.mark.parametrize("text", [
    "As of 23 May 2025, Canada reported 3,011 confirmed pertussis cases.",
    "Au 23 mai 2025, France a signal\u00e9 32 456 cas confirm\u00e9s de chikungunya.",
])
def test_rule_converts_explicit_as_of_without_report_date_leakage(text):
    result = _official_dates(text, "", {})
    assert result["as_of_date"] == "2025-05-23"
    assert not result["date_reported"] and not result["publication_date"]


@pytest.mark.parametrize("disease,country,unrelated", [
    ("mpox", "Sierra Leone", "lymphadenopathy"),
    ("measles", "France", "pneumonia"),
])
def test_official_rule_does_not_copy_unrelated_syndrome(disease, country, unrelated):
    chunk = {"source_id": "s", "chunk_id": "c", "source_type": "official_public_health_agency",
             "source_url": "https://offline.invalid", "text":
             f"{country} reported 3,011 confirmed {disease} cases during 2025. "
             f"A separate clinical subgroup had {unrelated}."}
    policy = StructuredExtractionPolicy(**load_structured_extraction_policy())
    context = {"disease_standard_name": disease, "is_hantavirus": False,
               "disease_terms": [disease], "aliases": [disease], "pathogen_terms": [unrelated]}
    record, diagnostics = _official_outbreak_record_from_chunk(chunk, 1, policy, context)
    assert record is not None, diagnostics
    assert record.pathogen_or_syndrome is None
    assert record.virus_or_syndrome is None
