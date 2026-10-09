"""Report-date inventory must stay separate from observation-period coverage."""
import hashlib
import json
from copy import deepcopy

import pytest

from data_collection_workflow.evidence_qualification import build_evidence_index, qualify_records


def _state(text="France reported 12 confirmed measles cases over the full year 2024.", **fields):
    digest = hashlib.sha256(text.encode()).hexdigest()
    record = dict(record_id="r", source_id="s", chunk_id="c", disease="measles",
                  country="France", reporting_period="2024", cases_confirmed=12, **fields)
    state = {
        "structured_task": {"disease": "measles", "location": "France", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "source_registry": [{"source_id": "s", "canonical_url": "https://offline.invalid/report", "status": "included",
                             "source_role_final": "collection", "final_screening_decision": "include_for_content_fetch"}],
        "documents": [{"source_id": "s", "document_id": "d", "clean_text": text, "content_hash": digest}],
        "evidence_chunks": [{"source_id": "s", "document_id": "d", "chunk_id": "c", "text": text,
                             "document_hash": digest, "char_start": 0, "char_end": len(text)}],
        "normalized_records": [record],
        "source_coverage_requirements": [{"requirement_id": "year", "disease": "measles", "country": "France",
                                          "period_start": "2024-01-01", "period_end": "2024-12-31"}],
    }
    state.update(qualify_records([record], contract=state["structured_task"], evidence_index=build_evidence_index(state)))
    assert len(state["qualified_records"]) == 1
    return state


def test_finalization_exports_timeline_without_turning_annual_observation_into_updates(monkeypatch, tmp_path):
    from data_collection_workflow.nodes.finalization import final_data_package_builder
    from data_collection_workflow.export import export_final_data_package
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    result = final_data_package_builder(_state())
    coverage = result["source_coverage_audit"]
    assert coverage["coverage_complete"] is True
    assert "reporting_timeline" in coverage
    timeline = coverage["reporting_timeline"]
    assert timeline["supported_observation_dates"]["distinct_date_count"] == 0
    assert timeline["supported_observation_dates"]["gap_day_count"] == 366
    export_final_data_package(result["final_data_package"], tmp_path)
    exported = json.loads((tmp_path / "source_coverage_audit.json").read_text())
    assert exported == coverage


def _inventory(state, **limits):
    from data_collection_workflow.report_timeline import build_report_timeline_inventory
    return build_report_timeline_inventory(state, **limits)


def test_metadata_publication_observation_cutoff_and_capture_keep_distinct_roles():
    state = _state("France reported 12 confirmed measles cases during 2024 as of 2024-06-30.", as_of_date="2024-06-30")
    state["source_registry"][0].update(published_date="2024-07-02", historical_snapshot={"captured_at": "2024-08-03T00:00:00+00:00", "verification_status": "index_only"})
    before = deepcopy(state)
    result = _inventory(state)
    publication = result["publication_candidates"]
    supported = result["supported_observation_dates"]
    assert publication["distinct_date_count"] == supported["distinct_date_count"] == 1
    assert publication["dates"] == ["2024-07-02"]
    assert supported["dates"] == ["2024-06-30"]
    assert publication["date_basis"] == "candidate_publication_metadata"
    assert supported["date_basis"] == "supported_observation_points"
    entry = result["sources"][0]
    assert entry["publication_date"]["status"] == "candidate_metadata"
    assert entry["archive_capture_at"] == "2024-08-03T00:00:00+00:00"
    assert entry["observation_dates"][0]["field"] == "as_of_date"
    assert entry["observation_dates"][0]["evidence"]["document_hash"] == state["documents"][0]["content_hash"]
    assert state == before


@pytest.mark.parametrize("start,end,days", [
    ("2024-02-29", "2024-02-29", 1),
    ("2024-02-01", "2024-02-29", 29),
    ("2024-01-01", "2024-03-31", 91),
    ("2023-01-01", "2023-12-31", 365),
    ("2023-01-01", "2025-12-31", 1096),
    ("0001-01-01", "9999-12-31", 3652059),
])
def test_empty_timeline_uses_one_exact_window_without_calendar_bins(start, end, days):
    result = _inventory({"collection_spec": {"start_date": start, "end_date": end}})
    assert result["status"] == "evaluated"
    assert result["window"]["inclusive_days"] == days
    for track in (result["publication_candidates"], result["supported_observation_dates"]):
        assert track["distinct_date_count"] == 0
        assert track["gap_day_count"] == days
        assert len(track["gaps"]) == 1
        assert {key: track["gaps"][0][key] for key in ("start_date", "end_date", "day_count")} == {
            "start_date": start, "end_date": end, "day_count": days}


@pytest.mark.parametrize("task", [
    {}, {"start_date": "2024-01-01"}, {"start_date": "2024-03-02", "end_date": "2024-03-01"},
    {"start_date": "2023-02-29", "end_date": "2023-03-01"},
    {"start_date": "2024-02", "end_date": "2024-03"},
    {"start_date": "20240101", "end_date": "20241231"},
])
def test_invalid_or_incomplete_task_dates_are_not_evaluable(task):
    result = _inventory({"structured_task": task})
    assert result["status"] == "not_evaluable"
    assert result["reason"] == "invalid_task_window"
    assert result["publication_candidates"]["gaps"] == []
    assert result["supported_observation_dates"]["gaps"] == []


def test_duplicate_point_dates_leave_exact_boundary_gaps_and_bounded_opportunities():
    sources = [dict(source_id=str(i), source_role_final="collection", published_date=value) for i, value in enumerate([
        "2024-01-02", "2024-01-02", "2024-01-05", "2024-01-07", "2023-12-31", "2024-01-12"])]
    state = {"structured_task": {"start_date": "2024-01-01", "end_date": "2024-01-10"}, "source_registry": sources}
    result = _inventory(state, max_gaps=2)["publication_candidates"]
    assert result["dates"] == ["2024-01-02", "2024-01-05", "2024-01-07"]
    assert result["dated_source_count"] == 4
    assert result["gap_count"] == 4
    assert result["omitted_gap_count"] == 2
    assert result["gap_day_count"] == 7
    assert [(gap["start_date"], gap["end_date"], gap["day_count"]) for gap in result["gaps"]] == [
        ("2024-01-08", "2024-01-10", 3), ("2024-01-03", "2024-01-04", 2)]
    reverse = _inventory({**state, "source_registry": list(reversed(sources))}, max_gaps=2)
    assert reverse["publication_candidates"]["gaps"] == result["gaps"]
    assert all(gap["gap_id"] and gap["date_basis"] == "candidate_publication_metadata" for gap in result["gaps"])


@pytest.mark.parametrize("change", [
    {"source_role_final": "validation_only"}, {"source_role_final": "excluded"},
    {"final_screening_decision": "exclude"}, {"blocked_from_fetch": True},
    {"source_disease_relevance_status": "unrelated_disease"}, {"geography_fit": "mismatch"},
    {"target_fit_status": "best_available_context_candidate"},
])
def test_noncollection_source_cannot_close_either_gap_track(change):
    state = _state("France reported 12 confirmed measles cases during 2024 as of 2024-06-30.", as_of_date="2024-06-30")
    state["source_registry"][0].update(published_date="2024-07-02", **change)
    result = _inventory(state)
    assert result["sources"][0]["collection_eligible"] is False
    assert result["publication_candidates"]["distinct_date_count"] == 0
    assert result["supported_observation_dates"]["distinct_date_count"] == 0


def test_unknown_invalid_coarse_and_archive_dates_never_become_publication_points():
    state = {"structured_task": {"start_date": "2024-01-01", "end_date": "2024-12-31"},
             "source_registry": [dict(source_id=str(i), source_role_final="collection", published_date=value,
                                      historical_snapshot={"captured_at": "2024-06-01T00:00:00Z", "verification_status": "index_only"})
                                 for i, value in enumerate([None, "2024-02", "2024-02-30"])]}
    track = _inventory(state)["publication_candidates"]
    assert track["distinct_date_count"] == 0
    assert track["unknown_date_source_count"] == 3
    assert track["gap_day_count"] == 366


@pytest.mark.parametrize("tamper", ["value", "hash", "locator", "source", "field_role", "status"])
def test_only_current_supported_source_bound_point_evidence_can_anchor_timeline(tamper):
    state = _state("France reported 12 confirmed measles cases during 2024 as of 2024-06-30.", as_of_date="2024-06-30")
    row = state["qualified_records"][0]
    evidence = next(e for e in row["evidence_qualification"]["field_evidence"] if e["field"] == "as_of_date")
    if tamper == "value": row["as_of_date"] = "2024-07-01"
    elif tamper == "hash": evidence["document_hash"] = "wrong"
    elif tamper == "locator": evidence["locator"]["char_end"] = 9999
    elif tamper == "source": row["source_id"] = "other"
    elif tamper == "field_role":
        evidence["field"] = "report_date"
        row.pop("as_of_date")
        row["report_date"] = "2024-06-30"
    else: row["evidence_qualification"]["status"] = "candidate"
    assert _inventory(state)["supported_observation_dates"]["distinct_date_count"] == 0


def test_observation_reporting_date_is_not_published_date():
    state = _state("France reported 12 confirmed measles cases during 2024 on 2024-06-30.", report_date="2024-06-30")
    result = _inventory(state)
    assert result["publication_candidates"]["distinct_date_count"] == 0
    assert result["supported_observation_dates"]["dates"] == ["2024-06-30"]
    assert result["sources"][0]["observation_dates"][0]["field"] == "report_date"


def test_single_day_point_has_no_gap_without_a_completeness_claim():
    state = {"structured_task": {"start_date": "2024-02-29", "end_date": "2024-02-29"},
             "source_registry": [{"source_id": "s", "source_role_final": "collection", "published_date": "2024-02-29T12:00:00Z"}]}
    result = _inventory(state)
    assert result["publication_candidates"]["gap_count"] == 0
    assert result["publication_candidates"]["gap_day_count"] == 0
    assert "coverage_complete" not in result


def test_recovery_diagnostic_preserves_observation_coverage_and_does_not_create_actions(monkeypatch):
    from data_collection_workflow.workflow_recovery import recovery_control
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    state = _state()
    result = recovery_control(state)
    assert result["source_coverage_audit"]["coverage_complete"] is True
    assert result["source_coverage_audit"]["reporting_timeline"]["supported_observation_dates"]["gap_day_count"] == 366
    assert result["recovery_plan"]["actions"] == []


@pytest.mark.parametrize("limit", [-1, True, 1.5, 257])
def test_invalid_gap_limits_fail_closed(limit):
    result = _inventory({"structured_task": {"start_date": "2024-01-01", "end_date": "2024-12-31"}}, max_gaps=limit)
    assert result["status"] == "not_evaluable"
    assert result["reason"] == "invalid_gap_limit"


def test_zero_gap_limit_reports_omitted_opportunity_without_materializing_it():
    track = _inventory({"structured_task": {"start_date": "2024-01-01", "end_date": "2024-12-31"}}, max_gaps=0)["publication_candidates"]
    assert track["gaps"] == []
    assert track["gap_count"] == track["omitted_gap_count"] == 1
    assert track["gap_day_count"] == 366


@pytest.mark.parametrize("change", [{"disease": "rubella"}, {"location": "Germany"}])
def test_stale_qualified_record_cannot_fill_changed_task_timeline(change):
    state = _state("France reported 12 confirmed measles cases during 2024 as of 2024-06-30.", as_of_date="2024-06-30")
    state["structured_task"].update(change)
    assert _inventory(state)["supported_observation_dates"]["distinct_date_count"] == 0


def test_known_outside_window_date_is_not_counted_as_unknown():
    state = {"structured_task": {"start_date": "2024-01-01", "end_date": "2024-12-31"},
             "source_registry": [{"source_id": "s", "source_role_final": "collection", "published_date": "2023-12-31"}]}
    track = _inventory(state)["publication_candidates"]
    assert track["distinct_date_count"] == 0
    assert track["unknown_date_source_count"] == 0
    assert track["outside_window_source_count"] == 1


@pytest.mark.parametrize("point", ["0001-01-01", "9999-12-31"])
def test_nonempty_extreme_day_window_avoids_calendar_overflow(point):
    state = {"structured_task": {"start_date": point, "end_date": point},
             "source_registry": [{"source_id": "s", "source_role_final": "collection", "published_date": point}]}
    track = _inventory(state)["publication_candidates"]
    assert track["dates"] == [point]
    assert track["gaps"] == []


def _version(value, suffix, source_id="s", chunk_id=None):
    state = _state(f"France reported 12 confirmed measles cases during 2024 as of {value}.", as_of_date=value)
    row, doc, chunk = state["normalized_records"][0], state["documents"][0], state["evidence_chunks"][0]
    row.update(record_id="r" + suffix, source_id=source_id, chunk_id=chunk_id or "c" + suffix)
    doc.update(document_id="d" + suffix, source_id=source_id)
    chunk.update(document_id=doc["document_id"], source_id=source_id, chunk_id=row["chunk_id"])
    state["source_registry"][0]["source_id"] = source_id
    state.update(qualify_records([row], contract=state["structured_task"], evidence_index=build_evidence_index(state)))
    assert len(state["qualified_records"]) == 1
    return state


def test_multiple_document_versions_of_one_source_keep_distinct_dated_proofs():
    first, last = _version("2024-03-01", "1"), _version("2024-10-01", "2")
    for key in ("documents", "evidence_chunks", "qualified_records"):
        first[key] += last[key]
    result = _inventory(first)
    assert result["supported_observation_dates"]["dates"] == ["2024-03-01", "2024-10-01"]
    assert result["supported_observation_dates"]["dated_source_count"] == 1
    assert {row["evidence"]["locator"]["document_id"] for row in result["sources"][0]["observation_dates"]} == {"d1", "d2"}


def test_colliding_chunk_identifier_never_attributes_another_sources_date():
    first, last = _version("2024-03-01", "1", "first", "collision"), _version("2024-10-01", "2", "last", "collision")
    for key in ("documents", "evidence_chunks", "qualified_records", "source_registry"):
        first[key] += last[key]
    result = _inventory(first)
    assert result["supported_observation_dates"]["dates"] == ["2024-10-01"]
    assert next(row for row in result["sources"] if row["source_id"] == "first")["observation_dates"] == []


def test_recovery_and_finalization_compute_same_current_timeline(monkeypatch):
    from data_collection_workflow.workflow_recovery import recovery_control
    from data_collection_workflow.nodes.finalization import final_data_package_builder
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    state = _state("France reported 12 confirmed measles cases during 2024 as of 2024-06-30.", as_of_date="2024-06-30")
    state["source_registry"][0]["published_date"] = "2024-07-01"
    recovery = recovery_control(state)["source_coverage_audit"]["reporting_timeline"]
    final = final_data_package_builder(state)["source_coverage_audit"]["reporting_timeline"]
    assert recovery == final


@pytest.mark.parametrize("boundary", [
    {"source_role_final": "validation"}, {"source_role_final": "validation_reserved"},
    {"source_role_final": "validation_source"}, {"source_role_final": "search_endpoint"},
    {"source_role_final": "needs_human_review"}, {"source_role": "context_source"},
    {"final_screening_decision": "reserved_for_validation"}, {"requires_human_review": True},
    {"source_excluded_by_human_review": True}, {"routing_flags": ["blocked_from_structured_extraction"]},
    {"screening_flags": ["user_excluded"]},
    {"routing_flags": ["blocked_from_collection"]}, {"routing_flags": ["validation_reserved"]},
    {"status": "reserved_for_validation"},
])
def test_existing_routing_boundaries_block_timeline_points(boundary):
    state = _state("France reported 12 confirmed measles cases during 2024 as of 2024-06-30.", as_of_date="2024-06-30")
    state["source_registry"][0].update(published_date="2024-07-01", **boundary)
    result = _inventory(state)
    assert result["sources"][0]["collection_eligible"] is False
    assert result["publication_candidates"]["distinct_date_count"] == 0
    assert result["supported_observation_dates"]["distinct_date_count"] == 0


@pytest.mark.parametrize("tamper", ["count", "scope_support", "extra_metric"])
def test_stale_enclosing_observation_proof_cannot_fill_supported_timeline(tamper):
    state = _state("France reported 12 confirmed measles cases during 2024 as of 2024-06-30.", as_of_date="2024-06-30")
    row = state["qualified_records"][0]
    if tamper == "count": row["cases_confirmed"] = 900
    elif tamper == "extra_metric": row["deaths"] = 900
    else:
        next(e for e in row["evidence_qualification"]["field_evidence"] if e["field"] == "disease")["supported"] = False
    assert _inventory(state)["supported_observation_dates"]["distinct_date_count"] == 0
