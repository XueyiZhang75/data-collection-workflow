"""Lexical aliases must not add geography or verify publisher identity."""
import pytest

from data_collection_workflow.config import load_record_normalization_policy
from data_collection_workflow.models import RecordNormalizationPolicy
from data_collection_workflow.nodes.normalization import _normalize_record, record_normalization


@pytest.fixture(autouse=True)
def evidence_mode(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")


def normalize(**fields):
    row = dict(record_id="r", disease="Measles", country="Canada",
               source_type="official_public_health_agency", cases_unspecified=12,
               evidence_quote="Canada reported 12 measles cases in Ontario in 2025.")
    row.update(fields)
    return _normalize_record(row, RecordNormalizationPolicy(**load_record_normalization_policy()))[0]


@pytest.mark.parametrize("raw, canonical", [
    ("Canada", "Canada"), ("CANADA", "Canada"), ("canada", "Canada"),
    ("MEXICO", "Mexico"), ("Brazil", "Brazil"), ("South Africa", "South Africa"),
    ("Georgia", "Georgia"), ("Jersey", "Jersey"), ("Romania", "Romania"),
    ("Israel", "Israel"), ("Japan", "Japan"),
])
def test_known_country_names_keep_raw_values_without_false_warnings(raw, canonical):
    row = normalize(country=raw)
    assert row["country"] == canonical
    assert row["country_raw"] == raw
    assert "unrecognized_country_name" not in row["normalization_warnings"]


def test_country_alias_does_not_promote_subnational_observation_to_national_scope():
    row = normalize(country="CANADA", subnational_location="Ontario")
    assert row["country"] == "Canada"
    assert row["subnational_location"] == "Ontario"
    assert row["geographic_scope"] is None
    assert row["geographic_scope_type"] is None


def test_evidence_mode_preserves_established_short_country_name_in_exports():
    row = normalize(country='United States', geographic_scope='United States', geographic_scope_type='country')
    assert row['country'] == row['country_raw'] == row['geographic_scope'] == 'United States'
    assert 'unrecognized_country_name' not in row['normalization_warnings']


@pytest.mark.parametrize("source_type, canonical", [
    ("WHO_DON", "international_organization_report"),
    ("PAHO_outbreak_update", "international_organization_report"),
    ("international_public_health_agency", "international_organization_report"),
    ("official_public_health_outbreak_page", "official_public_health_agency"),
    ("national_public_health_agency", "official_public_health_agency"),
])
def test_source_subtype_aliases_preserve_provenance_without_format_review(source_type, canonical):
    row = normalize(source_type=source_type, requires_human_review=True)
    assert row["source_type"] == canonical
    assert row["source_type_raw"] == source_type
    assert row["requires_human_review"] is True
    assert "unrecognized_source_type" not in row["normalization_warnings"]


@pytest.mark.parametrize("source_type, warning", [
    ("unknown", "unknown_source_type"),
    ("academic_or_peer_reviewed_source", "unverified_source_type_identity"),
    ("unknown_blog", "unrecognized_source_type"),
])
def test_unknown_and_unverified_identity_are_distinct_from_unrecognized_values(source_type, warning):
    row = normalize(source_type=source_type, source_url="https://www.slideshare.net/report")
    assert row["source_type"] == source_type
    assert row["source_type"] != "peer_reviewed_literature"
    assert warning in row["normalization_warnings"]
    assert row["requires_human_review"] is True
    if warning != "unrecognized_source_type":
        assert "unrecognized_source_type" not in row["normalization_warnings"]


def test_normalization_counts_identity_questions_separately_from_unrecognized_types():
    rows = [dict(record_id=str(i), disease="Measles", country="Canada", source_type=kind,
                 evidence_quote="Canada reported 12 measles cases.", cases_unspecified=12)
            for i, kind in enumerate(["unknown", "academic_or_peer_reviewed_source", "unknown_blog", "WHO_DON"])]
    result = record_normalization({"validated_records": rows})
    summary = result["record_normalization_summary"]
    assert summary["source_type_warning_count"] == 3
    assert summary["source_type_identity_warning_count"] == 2
    assert summary["source_type_unrecognized_count"] == 1
