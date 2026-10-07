from __future__ import annotations

from synthetic_workflow_inputs import write_workflow_config

from pathlib import Path

import pytest

from data_collection_workflow.export import export_final_data_package
from data_collection_workflow.graph import build_graph
from data_collection_workflow.nodes.finalization import final_data_package_builder
from data_collection_workflow.nodes.linking_validation import (
    cross_source_consistency_check,
    record_linking,
)


@pytest.fixture(autouse=True)
def _offline_telemetry(monkeypatch):
    # Direct graph fixture tests must not inherit a developer's trace uploads.
    for key in (
        "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING",
        "ENABLE_LANGSMITH_TRACE",
    ):
        monkeypatch.setenv(key, "false")


def _record(record_id: str, **overrides) -> dict:
    record = {
        "record_id": record_id,
        "disease": "COVID-19",
        "disease_standard_name": "COVID-19",
        "virus_or_syndrome": "COVID-19",
        "country": "United States of America",
        "subnational_location": "New York",
        "locality": None,
        "date_reported": "2024-06-01",
        "date_anchor": "2024-06-01",
        "reporting_period": "2024",
        "as_of_date": None,
        "cases_unspecified": 100.0,
        "deaths": 2.0,
        "hospitalizations": 5.0,
        "statistical_count_type": "annual",
        "count_semantics": "annual",
        "source_id": f"src_{record_id}",
        "source_url": f"https://example.org/{record_id}",
        "source_title": "Example public health source",
        "source_type": "official_public_health_agency",
        "publisher": "Example Department of Health",
        "source_role_final": "collection",
        "credibility_score": 0.9,
        "credibility_level": "high",
        "discovery_method": "fixture_search_result",
        "search_provider": "fixture",
        "query_id": "q_fixture_001",
        "query_used": "COVID-19 cases deaths New York 2024",
        "evidence_quote": (
            "New York reported 100 COVID-19 cases, 2 deaths, and "
            "5 hospitalizations on 2024-06-01."
        ),
        "supporting_chunk_id": f"chunk_{record_id}",
        "schema_status": "valid",
        "provenance_status": "verified",
        "normalization_status": "normalized",
        "record_schema": "generic_public_health_record",
        "requires_human_review": False,
    }
    record.update(overrides)
    return record


def _validation_record(record_id: str = "val_001", **overrides) -> dict:
    record = _record(
        record_id,
        source_id="src_validation_official",
        source_url="https://validation.example.org/covid-2024",
        source_role_final="validation",
        evidence_quote="Held-out validation source reports 100 COVID-19 cases.",
        supporting_chunk_id=f"chunk_{record_id}",
    )
    record.update(overrides)
    return record


def _validated_state(
    records: list[dict],
    *,
    validation_records: list[dict] | None = None,
    structured_task: dict | None = None,
    source_registry: list[dict] | None = None,
) -> dict:
    linked = record_linking(
        {
            "normalized_records": records,
            "human_review_queue": [],
            "collection_trace": [],
        }
    )
    state = {
        "normalized_records": linked["normalized_records"],
        "linked_events": linked["linked_events"],
        "event_clusters": linked["event_clusters"],
        "duplicate_clusters": linked["duplicate_clusters"],
        "human_review_queue": linked["human_review_queue"],
        "collection_trace": linked["collection_trace"],
        "conflicts": [],
        "validation_records": validation_records or [],
        "source_registry": source_registry or [],
        "structured_task": structured_task
        or {
            "disease": "COVID-19",
            "location": "New York",
            "start_date": "2024",
            "end_date": "2024",
            "target_fields": ["cases_unspecified", "deaths", "hospitalizations"],
        },
        "collection_spec": {
            "disease": "COVID-19",
            "geography": "New York",
            "start_date": "2024",
            "end_date": "2024",
            "time_window": "2024",
        },
    }
    result = cross_source_consistency_check(state)
    return {**state, **result}


def _results_by_type(result: dict, validation_type: str) -> list[dict]:
    return [
        row
        for row in result.get("validation_results") or []
        if row.get("validation_type") == validation_type
    ]


def _run_fixture_config(config_path: Path) -> dict:
    from data_collection_workflow.runtime_profile import (
        load_workflow_run_config,
        temporary_workflow_env,
        workflow_initial_state_from_config,
        workflow_run_env_from_config,
    )

    config = load_workflow_run_config(config_path)
    with temporary_workflow_env(workflow_run_env_from_config(config)):
        return build_graph().invoke(workflow_initial_state_from_config(config))


def test_validation_models_are_importable():
    from data_collection_workflow.models import (
        ComparabilityAssessment,
        CrossSourceValidationResult,
        TrustedSourceValidationResult,
        ValidationCase,
        ValidationComparison,
        ValidationResult,
        ValidationUnit,
    )

    assert ValidationCase
    assert ValidationComparison
    assert ValidationResult
    assert ValidationUnit
    assert ComparabilityAssessment
    assert TrustedSourceValidationResult
    assert CrossSourceValidationResult


def test_scope_validation_flags_outside_time_window():
    result = _validated_state([
        _record("rec_old", date_reported="2023-12-31", reporting_period="2023")
    ])

    scope_results = _results_by_type(result, "scope_check")
    assert any(row["match_status"] == "outside_requested_scope" for row in scope_results)
    assert any(row["validation_status"] == "outside_scope" for row in scope_results)
    assert any("outside_time_window" in row["reason"] for row in scope_results)
    assert any(item["item_type"] == "validation_scope_issue" for item in result["human_review_queue"])


def test_scope_validation_flags_disease_mismatch():
    result = _validated_state([
        _record("rec_dengue", disease="Dengue", disease_standard_name="Dengue")
    ])

    scope_results = _results_by_type(result, "scope_check")
    assert any(row["match_status"] == "outside_requested_scope" for row in scope_results)
    assert any("disease_mismatch" in row["reason"] for row in scope_results)


def test_scope_validation_flags_outside_geography():
    result = _validated_state([
        _record("rec_fl", subnational_location="Florida")
    ])

    scope_results = _results_by_type(result, "scope_check")
    assert any(row["match_status"] == "outside_requested_scope" for row in scope_results)
    assert any("outside_geography" in row["reason"] for row in scope_results)


def test_scope_validation_flags_insufficient_scope_information():
    result = _validated_state([
        _record("rec_missing", subnational_location=None, date_reported=None, reporting_period=None)
    ])

    scope_results = _results_by_type(result, "scope_check")
    assert any(row["validation_status"] == "needs_human_review" for row in scope_results)
    assert any("insufficient_scope_information" in row["reason"] for row in scope_results)


def test_trusted_source_validation_matches_comparable_records():
    result = _validated_state(
        [_record("rec_collection")],
        validation_records=[_validation_record()],
    )

    trusted = _results_by_type(result, "trusted_source_comparison")
    assert trusted
    assert any(row["validation_status"] == "validated" for row in trusted)
    assert any(row["match_status"] == "matched" for row in trusted)
    example = trusted[0]
    assert example["left_record_ids"]
    assert example["right_record_ids"]
    assert example["left_source_ids"]
    assert example["right_source_ids"]
    assert example["compared_field"] in {"cases_unspecified", "deaths", "hospitalizations"}


def test_trusted_source_validation_detects_conflict():
    result = _validated_state(
        [_record("rec_collection", cases_unspecified=100)],
        validation_records=[_validation_record(cases_unspecified=125)],
    )

    trusted = _results_by_type(result, "trusted_source_comparison")
    assert any(row["match_status"] == "conflict" for row in trusted)
    assert any(row["validation_status"] == "conflict" for row in trusted)
    assert any(item["item_type"] == "validation_conflict" for item in result["human_review_queue"])


def test_missing_validation_counterpart_is_explicit():
    result = _validated_state([_record("rec_collection")], validation_records=[])

    trusted = _results_by_type(result, "held_out_source_comparison")
    assert trusted
    assert any(row["match_status"] == "missing_validation" for row in trusted)
    assert result["trusted_source_validation_summary"]["missing_validation_count"] >= 1


def test_missing_collection_counterpart_is_explicit():
    result = _validated_state([], validation_records=[_validation_record()])

    trusted = _results_by_type(result, "held_out_source_comparison")
    assert trusted
    assert any(row["match_status"] == "missing_collection" for row in trusted)
    assert result["trusted_source_validation_summary"]["missing_collection_count"] >= 1


def test_incompatible_count_semantics_are_not_falsely_matched():
    result = _validated_state(
        [_record("rec_collection", statistical_count_type="cumulative", count_semantics="cumulative")],
        validation_records=[
            _validation_record(statistical_count_type="annual", count_semantics="annual")
        ],
    )

    trusted = _results_by_type(result, "count_semantics_check")
    assert trusted
    assert any(row["comparability_status"] == "not_comparable" for row in trusted)
    assert all(row["match_status"] != "matched" for row in trusted)


def test_cross_source_support_counts_independent_sources_only():
    collection = _record("rec_collection", source_url="https://official.example.org/update")
    support = _record(
        "rec_support",
        source_id="src_news_support",
        source_url="https://news.example.org/update",
        source_type="news_and_situation_report",
        source_role_final="collection_support",
        credibility_score=0.65,
    )

    result = _validated_state([collection, support])

    support_results = _results_by_type(result, "cross_source_support")
    assert support_results
    assert result["cross_source_validation_summary"]["cross_source_supported_cluster_count"] >= 1
    assert any(row["validation_status"] == "validated" for row in support_results)


def test_duplicate_records_do_not_inflate_cross_source_support():
    original = _record(
        "rec_original",
        source_id="src_same",
        source_url="https://official.example.org/same",
        supporting_chunk_id="chunk_same",
    )
    duplicate = _record(
        "rec_duplicate",
        source_id="src_same",
        source_url="https://official.example.org/same",
        supporting_chunk_id="chunk_same",
    )

    result = _validated_state([original, duplicate])

    support_results = _results_by_type(result, "cross_source_support")
    assert support_results
    assert any(row["match_status"] == "missing_validation" for row in support_results)
    assert result["cross_source_validation_summary"]["single_source_only_cluster_count"] >= 1


def test_cross_source_conflict_routes_to_human_review():
    official = _record("rec_official", cases_unspecified=100)
    secondary = _record(
        "rec_secondary",
        source_id="src_secondary",
        source_url="https://secondary.example.org/conflict",
        source_type="news_and_situation_report",
        source_role_final="collection_support",
        cases_unspecified=150,
    )

    result = _validated_state([official, secondary])

    conflicts = _results_by_type(result, "cross_source_conflict")
    assert conflicts
    assert any(row["match_status"] == "conflict" for row in conflicts)
    assert result["conflicts"]
    assert any(item["item_type"] == "validation_conflict" for item in result["human_review_queue"])


def test_aggregate_validation_uses_countable_records_only():
    original = _record(
        "rec_original",
        source_id="src_same",
        source_url="https://official.example.org/same",
        supporting_chunk_id="chunk_same",
        cases_unspecified=100,
    )
    duplicate = _record(
        "rec_duplicate",
        source_id="src_same",
        source_url="https://official.example.org/same",
        supporting_chunk_id="chunk_same",
        cases_unspecified=100,
    )
    validation = _validation_record(cases_unspecified=100)

    result = _validated_state([original, duplicate], validation_records=[validation])

    aggregate = _results_by_type(result, "aggregate_comparison")
    assert aggregate
    cases = [row for row in aggregate if row["compared_field"] == "cases_unspecified"]
    assert cases
    assert cases[0]["left_value"] == 100.0
    record_status = {
        record["record_id"]: record["countable"]
        for record in result["normalized_records"]
    }
    non_countable_ids = [
        record_id for record_id, countable in record_status.items() if countable is False
    ]
    countable_ids = [
        record_id for record_id, countable in record_status.items() if countable is True
    ]
    assert non_countable_ids
    assert countable_ids
    assert all(record_id not in cases[0]["left_record_ids"] for record_id in non_countable_ids)
    assert all(record_id in cases[0]["left_record_ids"] for record_id in countable_ids)


def test_validation_results_include_required_audit_fields():
    result = _validated_state(
        [_record("rec_collection")],
        validation_records=[_validation_record()],
    )

    row = result["validation_results"][0]
    required = {
        "validation_result_id",
        "validation_case_id",
        "validation_type",
        "validation_unit",
        "comparison_id",
        "left_record_ids",
        "right_record_ids",
        "left_source_ids",
        "right_source_ids",
        "compared_field",
        "disease",
        "location",
        "date_or_period",
        "statistical_count_type",
        "count_semantics",
        "left_value",
        "right_value",
        "comparability_status",
        "match_status",
        "validation_status",
        "confidence",
        "reason",
        "evidence_summary",
        "needs_human_review",
        "warnings",
    }
    assert required <= set(row)


def test_final_package_exports_validation_artifacts(tmp_path):
    validated = _validated_state(
        [_record("rec_collection")],
        validation_records=[_validation_record()],
    )
    finalized = final_data_package_builder(
        {
            **validated,
            "source_registry": [],
        }
    )
    package = finalized["final_data_package"]

    assert package["validation_results"]
    assert package["workflow_summaries"]["validation_summary"]
    manifest = export_final_data_package(package, tmp_path / "collection")
    assert Path(manifest["files"]["validation_results_json"]).exists()
    assert Path(manifest["files"]["validation_results_csv"]).exists()


def test_full_graph_covid19_fixture_validation_smoke(tmp_path):
    result = _run_fixture_config(write_workflow_config(tmp_path, disease='COVID-19', location='New York', year='2024', phase='collection'))

    assert result["normalized_records"]
    assert result["event_clusters"]
    assert result["validation_summary"]["validation_result_count"] > 0
    assert result["validation_results"]
    assert result["cross_source_validation_summary"]
    assert result["trusted_source_validation_summary"]
    assert all(record["disease"] == "COVID-19" for record in result["normalized_records"])
    assert result["final_data_package"]["validation_results"]


def test_full_graph_dengue_fixture_validation_smoke(tmp_path):
    result = _run_fixture_config(write_workflow_config(tmp_path, disease='dengue', location='Florida', year='2025', phase='collection'))

    assert result["normalized_records"]
    assert result["validation_summary"]["validation_result_count"] > 0
    assert result["validation_results"]
    assert result["cross_source_validation_summary"]
    assert result["trusted_source_validation_summary"]
    assert all(record["disease"] == "Dengue" for record in result["normalized_records"])


def test_hantavirus_new_mexico_validation_compatibility(monkeypatch):
    from data_collection_workflow.runtime_profile import (
        default_workflow_run_config,
        workflow_run_config_with_overrides,
        temporary_workflow_env,
        workflow_initial_state_from_config,
        workflow_run_env_from_config,
    )

    # A compatibility smoke test must not inherit a user's live search/model
    # configuration. Keep all production provider endpoints forbidden here.
    import requests
    from data_collection_workflow import llm_clients
    external_attempts = []
    def forbidden_external(*args, **kwargs):
        external_attempts.append(True)
        raise AssertionError('offline compatibility test attempted an external provider')
    monkeypatch.setattr(requests.sessions.Session, 'request', forbidden_external)
    monkeypatch.setattr(llm_clients, 'build_chat_model', forbidden_external)
    config = workflow_run_config_with_overrides(default_workflow_run_config(),
                                                live_fetch=False, all_llm=False)
    config['source_search']['enabled'] = False
    with temporary_workflow_env(workflow_run_env_from_config(config)):
        result = build_graph().invoke(workflow_initial_state_from_config(config))
    assert external_attempts == []

    assert "conflicts" in result
    assert "cross_source_consistency_summary" in result
    assert result["validation_summary"]["validation_result_count"] >= 0
    assert result["trusted_source_validation_summary"]
    assert result["cross_source_validation_summary"]
