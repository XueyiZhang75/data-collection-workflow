from __future__ import annotations

from pathlib import Path

import pytest

from data_collection_workflow.graph import build_graph
from data_collection_workflow.nodes.finalization import final_data_package_builder


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
        "event_cluster_id": "event_001",
        "countable": True,
        "requires_human_review": False,
    }
    record.update(overrides)
    return record


def _cluster(cluster_id: str = "event_001", **overrides) -> dict:
    cluster = {
        "event_cluster_id": cluster_id,
        "cluster_status": "single_record_event",
        "disease": "COVID-19",
        "country": "United States of America",
        "subnational_location": "New York",
        "date_reported": "2024-06-01",
        "reporting_period": "2024",
        "statistical_count_type": "annual",
        "count_semantics": "annual",
        "representative_record_id": "rec_001",
        "member_record_ids": ["rec_001"],
        "countable_record_ids": ["rec_001"],
        "source_ids": ["src_rec_001"],
        "source_urls": ["https://example.org/rec_001"],
        "canonical_cases_unspecified": 100.0,
        "canonical_deaths": 2.0,
    }
    cluster.update(overrides)
    return cluster


def _validation_result(result_id: str = "val_result_001", **overrides) -> dict:
    row = {
        "validation_result_id": result_id,
        "validation_case_id": "val_case_001",
        "validation_type": "trusted_source_comparison",
        "validation_unit": "field",
        "comparison_id": "val_cmp_001",
        "left_record_ids": ["rec_001"],
        "right_record_ids": ["val_001"],
        "left_event_cluster_ids": ["event_001"],
        "right_event_cluster_ids": [],
        "left_source_ids": ["src_rec_001"],
        "right_source_ids": ["src_validation"],
        "left_source_urls": ["https://example.org/rec_001"],
        "right_source_urls": ["https://validation.example.org/val"],
        "compared_field": "cases_unspecified",
        "disease": "COVID-19",
        "location": "New York",
        "date_or_period": "2024",
        "statistical_count_type": "annual",
        "count_semantics": "annual",
        "left_value": 100.0,
        "right_value": 100.0,
        "comparability_status": "comparable",
        "match_status": "matched",
        "validation_status": "validated",
        "confidence": 0.9,
        "reason": "collection and held-out validation match",
        "evidence_summary": "left and right evidence are comparable",
        "needs_human_review": False,
        "warnings": [],
    }
    row.update(overrides)
    return row


def _source(source_id: str = "src_rec_001", **overrides) -> dict:
    row = {
        "source_id": source_id,
        "canonical_url": f"https://example.org/{source_id}",
        "title": "Example source",
        "publisher": "Example Department of Health",
        "source_type": "official_public_health_agency",
        "status": "ready_for_content_fetch",
        "source_role_final": "collection",
        "credibility_score": 0.95,
        "credibility_level": "high",
        "final_screening_decision": "include_for_content_fetch",
    }
    row.update(overrides)
    return row


def _state(**overrides) -> dict:
    state = {
        "structured_task": {
            "disease": "COVID-19",
            "location": "New York",
            "start_date": "2024",
            "end_date": "2024",
        },
        "collection_spec": {
            "disease": "COVID-19",
            "geography": "New York",
            "start_date": "2024",
            "end_date": "2024",
            "time_window": "2024",
        },
        "normalized_records": [_record("rec_001")],
        "event_clusters": [_cluster()],
        "validation_results": [],
        "source_registry": [_source()],
        "conflicts": [],
        "human_review_queue": [],
        "human_review_decisions": [],
        "collection_trace": [],
    }
    state.update(overrides)
    return state


def _detect(state: dict) -> dict:
    from data_collection_workflow.anomaly_detection import detect_anomalies

    return detect_anomalies(state)


def _apply(state: dict) -> dict:
    from data_collection_workflow.human_review_application import apply_human_review_decisions

    return apply_human_review_decisions(state)


def _run_fixture_config(config_name: str) -> dict:
    from data_collection_workflow.runtime_profile import (
        load_workflow_run_config,
        temporary_workflow_env,
        workflow_initial_state_from_config,
        workflow_run_env_from_config,
    )

    config = load_workflow_run_config(Path("configs") / "examples" / config_name)
    with temporary_workflow_env(workflow_run_env_from_config(config)):
        return build_graph().invoke(workflow_initial_state_from_config(config))


def test_stage11_models_are_importable():
    from data_collection_workflow.models import (
        AnomalyDetectionDecision,
        AnomalyResult,
        AnomalyRule,
        AnomalySummary,
        AppliedHumanReviewDecision,
        HumanReviewAuditEntry,
        HumanReviewApplicationSummary,
        HumanReviewDecisionInput,
    )

    assert AnomalyResult
    assert AnomalySummary
    assert AnomalyRule
    assert AnomalyDetectionDecision
    assert HumanReviewDecisionInput
    assert AppliedHumanReviewDecision
    assert HumanReviewAuditEntry
    assert HumanReviewApplicationSummary


def test_deaths_greater_than_cases_routes_anomaly_to_review():
    result = _detect(_state(normalized_records=[_record("rec_bad", cases_unspecified=5, deaths=7)]))

    anomalies = result["anomaly_results"]
    assert any(row["anomaly_type"] == "deaths_greater_than_cases" for row in anomalies)
    assert any(row["severity"] in {"high", "critical"} for row in anomalies)
    assert any(item["item_type"] == "anomaly" for item in result["anomaly_review_items"])


def test_negative_count_value_is_critical():
    result = _detect(_state(normalized_records=[_record("rec_negative", cases_unspecified=-1)]))

    anomaly = next(row for row in result["anomaly_results"] if row["anomaly_type"] == "negative_count_value")
    assert anomaly["severity"] == "critical"
    assert anomaly["compared_field"] == "cases_unspecified"
    assert anomaly["needs_human_review"] is True


def test_missing_date_and_location_for_count_bearing_records_are_flagged():
    result = _detect(
        _state(
            normalized_records=[
                _record("rec_missing_date", date_reported=None, reporting_period=None, date_anchor=None),
                _record("rec_missing_location", country=None, subnational_location=None, geographic_scope=None),
            ]
        )
    )

    types = {row["anomaly_type"] for row in result["anomaly_results"]}
    assert "missing_date_for_count_bearing_record" in types
    assert "missing_location_for_count_bearing_record" in types


def test_disease_mismatch_or_unknown_count_bearing_record_is_high_anomaly():
    result = _detect(
        _state(normalized_records=[_record("rec_dengue", disease="Dengue", disease_standard_name="Dengue")])
    )

    anomaly = next(
        row
        for row in result["anomaly_results"]
        if row["anomaly_type"] == "disease_mismatch_or_unknown_for_count_bearing_record"
    )
    assert anomaly["severity"] == "high"
    assert anomaly["record_id"] == "rec_dengue"


def test_out_of_scope_validation_result_creates_anomaly():
    validation = _validation_result(
        validation_type="scope_check",
        match_status="outside_requested_scope",
        validation_status="outside_scope",
        reason="outside_time_window",
    )
    result = _detect(_state(validation_results=[validation]))

    assert any(row["anomaly_type"] == "out_of_scope_count_bearing_record" for row in result["anomaly_results"])


def test_count_semantics_conflict_creates_anomaly():
    validation = _validation_result(
        validation_type="count_semantics_check",
        comparability_status="not_comparable",
        match_status="not_comparable",
        validation_status="not_comparable",
        reason="count_semantics_not_comparable",
        warnings=["count_semantics_not_comparable"],
    )
    result = _detect(_state(validation_results=[validation]))

    assert any(row["anomaly_type"] == "count_semantics_conflict" for row in result["anomaly_results"])


def test_validation_conflict_creates_anomaly_and_review_item():
    validation = _validation_result(
        match_status="conflict",
        validation_status="conflict",
        left_value=100,
        right_value=125,
        reason="held-out validation conflict",
        needs_human_review=True,
    )
    result = _detect(_state(validation_results=[validation]))

    assert any(row["anomaly_type"] == "validation_conflict_anomaly" for row in result["anomaly_results"])
    assert any(item.get("validation_result_id") == "val_result_001" for item in result["anomaly_review_items"])


def test_high_credibility_source_conflict_creates_high_anomaly():
    conflict = {
        "conflict_id": "conflict_001",
        "field": "cases_unspecified",
        "values": [{"record_id": "rec_a", "value": 100}, {"record_id": "rec_b", "value": 150}],
        "conflict_type": "numeric_value_conflict",
        "severity": "high",
        "record_ids": ["rec_a", "rec_b"],
        "source_ids": ["src_high_a", "src_high_b"],
        "source_urls": ["https://a.example.org", "https://b.example.org"],
        "requires_human_review": True,
    }
    result = _detect(
        _state(
            conflicts=[conflict],
            source_registry=[
                _source("src_high_a", credibility_level="high", credibility_score=0.95),
                _source("src_high_b", credibility_level="high", credibility_score=0.93),
            ],
        )
    )

    anomaly = next(row for row in result["anomaly_results"] if row["anomaly_type"] == "high_credibility_source_conflict")
    assert anomaly["severity"] == "high"
    assert anomaly["source_ids"] == ["src_high_a", "src_high_b"]


def test_abrupt_spike_threshold_and_invalid_rate_are_flagged(monkeypatch):
    monkeypatch.setenv("ANOMALY_MAX_CASES_THRESHOLD", "250")
    result = _detect(
        _state(
            normalized_records=[
                _record("rec_prior", date_reported="2024-01-01", cases_unspecified=10, positivity_rate=0.2),
                _record("rec_spike", date_reported="2024-06-01", cases_unspecified=500, positivity_rate=1.5),
            ]
        )
    )

    types = {row["anomaly_type"] for row in result["anomaly_results"]}
    assert "abrupt_spike_simple_threshold" in types
    assert "test_positivity_or_rate_invalid" in types


def test_aggregate_member_mismatch_uses_countable_member_records():
    records = [
        _record("rec_a", event_cluster_id="event_001", cases_unspecified=5),
        _record("rec_b", event_cluster_id="event_001", cases_unspecified=5),
    ]
    cluster = _cluster(
        member_record_ids=["rec_a", "rec_b"],
        countable_record_ids=["rec_a", "rec_b"],
        canonical_cases_unspecified=99,
    )
    result = _detect(_state(normalized_records=records, event_clusters=[cluster]))

    assert any(row["anomaly_type"] == "aggregate_member_mismatch" for row in result["anomaly_results"])


def test_human_review_application_is_disabled_without_explicit_apply_flag():
    state = _state(
        human_review_decisions=[
            {
                "decision_id": "decision_no_apply",
                "review_id": "review_001",
                "decision_type": "correct_fields",
                "reviewer_id": "reviewer_test_public_health_001",
                "decided_at": "2026-06-01T00:00:00Z",
                "target_type": "record",
                "target_ids": ["rec_001"],
                "reason": "test correction",
                "patch": {"cases_unspecified": 111},
                "apply_decision": False,
            }
        ]
    )

    result = _apply(state)
    record = result["final_dataset_post_review"][0]
    assert record["cases_unspecified"] == 100.0
    assert result["human_review_application_summary"]["decisions_applied_count"] == 0
    assert result["human_review_application_summary"]["decisions_rejected_count"] == 1


def test_correct_fields_decision_applies_allowed_patch_and_audit():
    state = _state(
        human_review_decisions=[
            {
                "decision_id": "decision_correct_001",
                "review_id": "review_001",
                "decision_type": "correct_fields",
                "reviewer_id": "reviewer_test_public_health_001",
                "decided_at": "2026-06-01T00:00:00Z",
                "target_type": "record",
                "target_ids": ["rec_001"],
                "reason": "confirmed corrected count",
                "patch": {"cases_unspecified": 101, "notes": "Corrected by reviewer."},
                "confidence": 0.95,
                "apply_decision": True,
            }
        ]
    )

    result = _apply(state)
    record = result["final_dataset_post_review"][0]
    assert record["cases_unspecified"] == 101
    assert record["human_review_applied"] is True
    assert record["review_status"] == "corrected"
    assert result["applied_human_review_decisions"]
    assert any(audit["field_name"] == "cases_unspecified" for audit in result["human_review_audit_trail"])


def test_reject_record_decision_excludes_post_review_dataset_but_preserves_original():
    state = _state(
        human_review_decisions=[
            {
                "decision_id": "decision_reject_001",
                "review_id": "review_001",
                "decision_type": "reject_record",
                "reviewer_id": "reviewer_test_public_health_001",
                "decided_at": "2026-06-01T00:00:00Z",
                "target_type": "record",
                "target_ids": ["rec_001"],
                "reason": "duplicate erroneous record",
                "apply_decision": True,
            }
        ]
    )

    result = _apply(state)
    assert result["normalized_records"][0]["record_id"] == "rec_001"
    assert result["final_dataset_post_review"] == []
    assert result["records_excluded_by_human_review"][0]["record_id"] == "rec_001"
    assert result["human_review_application_summary"]["records_excluded_by_review"] == 1


def test_invalid_unsafe_decision_is_rejected_with_reason():
    state = _state(
        human_review_decisions=[
            {
                "decision_id": "decision_bad_patch",
                "review_id": "review_001",
                "decision_type": "correct_fields",
                "reviewer_id": "reviewer_test_public_health_001",
                "decided_at": "2026-06-01T00:00:00Z",
                "target_type": "record",
                "target_ids": ["rec_001"],
                "reason": "unsafe patch",
                "patch": {"record_id": "rec_other"},
                "apply_decision": True,
            }
        ]
    )

    result = _apply(state)
    assert result["rejected_human_review_decisions"]
    assert "patch_field_not_allowed" in result["rejected_human_review_decisions"][0]["rejection_reason"]
    assert result["final_dataset_post_review"][0]["record_id"] == "rec_001"


def test_countability_validation_anomaly_and_source_decisions_apply():
    validation = _validation_result(validation_status="conflict", match_status="conflict")
    anomaly = _detect(_state(validation_results=[validation]))["anomaly_results"][0]
    state = _state(
        validation_results=[validation],
        anomaly_results=[anomaly],
        source_registry=[_source("src_rec_001")],
        human_review_decisions=[
            {
                "decision_id": "decision_non_countable",
                "review_id": "review_countable",
                "decision_type": "mark_non_countable",
                "reviewer_id": "reviewer_test_public_health_001",
                "decided_at": "2026-06-01T00:00:00Z",
                "target_type": "record",
                "target_ids": ["rec_001"],
                "reason": "not countable after review",
                "apply_decision": True,
            },
            {
                "decision_id": "decision_validation",
                "review_id": "review_validation",
                "decision_type": "override_validation_status",
                "reviewer_id": "reviewer_test_public_health_001",
                "decided_at": "2026-06-01T00:00:00Z",
                "target_type": "validation_result",
                "target_ids": ["val_result_001"],
                "reason": "review resolved validation status",
                "patch": {"validation_status": "validated"},
                "apply_decision": True,
            },
            {
                "decision_id": "decision_anomaly",
                "review_id": "review_anomaly",
                "decision_type": "mark_anomaly_resolved",
                "reviewer_id": "reviewer_test_public_health_001",
                "decided_at": "2026-06-01T00:00:00Z",
                "target_type": "anomaly",
                "target_ids": [anomaly["anomaly_id"]],
                "reason": "review resolved anomaly",
                "apply_decision": True,
            },
            {
                "decision_id": "decision_source",
                "review_id": "review_source",
                "decision_type": "override_source_role",
                "reviewer_id": "reviewer_test_public_health_001",
                "decided_at": "2026-06-01T00:00:00Z",
                "target_type": "source",
                "target_ids": ["src_rec_001"],
                "reason": "source role correction",
                "patch": {"source_role_final": "collection_support"},
                "apply_decision": True,
            },
        ],
    )

    result = _apply(state)
    assert result["final_dataset_post_review"][0]["countable"] is False
    assert result["validation_results"][0]["validation_status"] == "validated"
    assert result["anomaly_results"][0]["anomaly_status"] == "resolved"
    assert result["source_registry"][0]["source_role_final"] == "collection_support"
    assert result["human_review_application_summary"]["decisions_applied_count"] == 4


def test_final_package_exports_anomaly_and_review_application_artifacts(tmp_path):
    detected = _detect(_state(normalized_records=[_record("rec_bad", cases_unspecified=5, deaths=7)]))
    applied = _apply({**_state(), **detected})
    finalized = final_data_package_builder({**_state(), **detected, **applied})
    package = finalized["final_data_package"]

    from data_collection_workflow.export import export_final_data_package

    manifest = export_final_data_package(package, tmp_path / "collection")
    assert package["anomaly_results"]
    assert package["anomaly_summary"]
    assert package["human_review_application_summary"]
    assert Path(manifest["files"]["anomaly_results_json"]).exists()
    assert Path(manifest["files"]["final_dataset_post_review_json"]).exists()
    assert Path(manifest["files"]["human_review_audit_trail_json"]).exists()


def test_full_graph_fixture_review_application_smoke():
    result = _run_fixture_config("covid19/offline_review.jsonc")

    assert result["anomaly_summary"]["anomaly_result_count"] >= 0
    assert result["human_review_application_summary"]["decisions_applied_count"] >= 1
    assert result["human_review_audit_trail"]
    assert result["final_data_package"]["final_dataset_post_review"] is not None
    assert all(record["disease"] == "COVID-19" for record in result["normalized_records"])


def test_full_graph_dengue_fixture_review_application_smoke():
    result = _run_fixture_config("dengue/offline_review.jsonc")

    assert result["anomaly_summary"]["anomaly_result_count"] >= 0
    assert result["human_review_application_summary"]["decisions_applied_count"] >= 1
    assert result["final_data_package"]["final_dataset_post_review"] is not None
    assert all(record["disease"] == "Dengue" for record in result["normalized_records"])


def test_hantavirus_new_mexico_stage11_compatibility(monkeypatch):
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

    assert "anomaly_summary" in result
    assert "human_review_application_summary" in result
    assert result["human_review_application_summary"]["decisions_applied_count"] == 0
    assert result["final_data_package"]["final_dataset_post_review"] is not None
    assert "validation_results" in result["final_data_package"]
