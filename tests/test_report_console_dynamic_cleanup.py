from __future__ import annotations

import json
import re

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _record(record_id: str, *, disease: str = "hantavirus", location: str = "Shanghai") -> dict:
    return {
        "record_id": record_id,
        "disease": disease,
        "disease_standard_name": disease,
        "country": "China" if location == "Shanghai" else "United States of America",
        "subnational_location": location,
        "date_reported": "2025-01-15",
        "cases_confirmed": 1,
        "deaths": 0,
        "source_id": f"src_{record_id}",
        "source_url": f"https://example.org/{record_id}",
        "source_type": "official_public_health_agency",
        "publisher": "Shanghai Municipal Health Commission"
        if location == "Shanghai"
        else "New Mexico Department of Health",
        "evidence_quote": f"{location} reported one {disease} case.",
        "supporting_chunk_id": f"chunk_{record_id}",
        "extraction_method": "llm_structured_output_extractor",
        "llm_used": True,
        "record_schema": "generic_public_health_record",
        "record_final_inclusion_status": "quarantined_disease_mismatch",
        "quality_gate_reasons": ["disease mismatch"],
    }


def _llm_stage_summary() -> dict:
    return {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "api_key_present": True,
        "source_planning": {
            "enabled": True,
            "status": "success",
            "generation_method": "llm_executable_source_plan",
            "execution_status": "planned_not_executed",
            "planned_query_count": 4,
            "planned_source_category_count": 2,
            "provider_channel_counts": {"official_site_search": 2},
            "agent_query_count": 4,
            "agent_query_added_count": 4,
            "candidate_hint_count": 0,
            "warnings": [],
            "failure_type": None,
            "failure_message": None,
        },
        "source_critic": {
            "enabled": True,
            "attempted_source_count": 6,
            "assessed_source_count": 6,
            "skipped_source_count": 0,
            "blocked_fetch_count": 2,
            "allowed_fetch_count": 1,
            "context_only_count": 0,
            "needs_review_count": 3,
            "max_sources": 6,
            "review_blocks_fetch": False,
            "failure_count": 0,
            "semantic_leakage_count": 0,
            "human_review_recommended_count": 3,
            "decision_counts": {"not_task_relevant": 2, "include": 1},
            "fetch_recommendation_counts": {"block_fetch": 2},
            "risk_flag_counts": {"disease_mismatch": 2},
            "selection": {"selected_source_ids": ["src_search_001"]},
        },
        "source_credibility": {
            "enabled": True,
            "assessed_source_count": 6,
            "role_counts": {"excluded": 5, "collection": 1},
            "risk_flag_counts": {},
            "llm_assessed_count": 6,
            "llm_failure_count": 0,
            "needs_review_count": 1,
        },
        "structured_extraction": {
            "enabled": True,
            "mode": "llm_structured_output",
            "eligible_chunk_count": 0,
            "call_count": 0,
            "success_count": 0,
            "error_count": 0,
            "fallback_count": 0,
            "raw_record_count": 0,
            "max_chunks": 8,
            "error_messages": [],
        },
    }


def _summaries(
    *,
    disease: str,
    location: str,
    start_date: str,
    end_date: str,
    run_quality_status: str,
    final_dataset_count: int,
    pre_quality_count: int,
    quarantined_count: int,
    pending_count: int,
    validation_status: str = "incompatible_validation_source_disabled",
) -> dict:
    collection_spec = {
        "task_type": "public_health_case_and_outbreak_collection",
        "disease": disease,
        "target_population": "human",
        "geography": location,
        "start_date": start_date,
        "end_date": end_date,
        "time_window": f"{start_date}-{end_date}",
        "user_request": f"Collect {disease} data for {location} from {start_date} to {end_date}.",
        "run_label": f"{disease}_{location}_{start_date}_{end_date}",
    }
    return {
        "task_intake_summary": {
            "collection_spec": collection_spec,
            "structured_task": {
                "disease": disease,
                "location": location,
                "start_date": start_date,
                "end_date": end_date,
            },
        },
        "source_search_execution_summary": {
            "search_enabled": True,
            "live_search_enabled": True,
            "search_mode": "live",
            "search_provider": "tavily",
            "planned_query_count": 4,
            "selected_query_count": 3,
            "executed_query_count": 2,
            "candidate_from_search_count": 6,
        },
        "localized_source_planning_summary": {
            "localized_source_planning_enabled": True,
            "localized_query_count": 4,
            "selected_localized_query_count": 3,
        },
        "source_critic_summary": {
            "llm_source_critic_enabled": True,
            "attempted_source_count": 6,
            "assessed_source_count": 6,
            "blocked_fetch_count": 2,
            "allowed_fetch_count": 1,
            "needs_review_count": 3,
        },
        "disease_relevance_summary": {
            "target_disease": disease,
            "chunk_unrelated_disease_count": 3,
            "rejected_incompatible_record_count": quarantined_count,
            "record_compatibility_status_counts": {"incompatible_disease": quarantined_count},
        },
        "validation_source_compatibility_summary": {
            "compatibility_status": validation_status,
            "active_validation_record_count": 0,
            "inactive_validation_record_count": 1,
            "task_disease": disease,
            "task_location": location,
            "warnings": ["no task-compatible held-out validation source is active"],
        },
        "run_quality_summary": {
            "run_quality_status": run_quality_status,
            "final_dataset_mode": "quality_gated_accepted_records",
            "task_disease": disease,
            "task_location": location,
            "task_start_date": start_date,
            "task_end_date": end_date,
            "normalized_record_count": pre_quality_count,
            "accepted_record_count": final_dataset_count,
            "final_dataset_count": final_dataset_count,
            "quarantined_record_count": quarantined_count,
            "pending_review_record_count": pending_count,
            "final_dataset_post_review_count": final_dataset_count,
            "validation_limited": True,
            "no_compatible_validation_source": True,
            "blocking_reason_counts": {"disease_mismatch": quarantined_count}
            if quarantined_count
            else {},
            "warning_counts": {"validation_limited_no_compatible_source": 1},
            "acceptance_reason": "no normalized records were extracted"
            if pre_quality_count == 0
            else "all candidate records were quarantined",
            "recommended_user_message": "No reliable task-relevant records were accepted.",
            "warnings": ["validation_limited_no_compatible_source"],
        },
        "final_dataset_quality_summary": {
            "final_dataset_mode": "quality_gated_accepted_records",
            "normalized_record_count": pre_quality_count,
            "accepted_record_count": final_dataset_count,
            "quarantined_record_count": quarantined_count,
            "pending_review_record_count": pending_count,
            "post_review_record_count": final_dataset_count,
            "record_final_inclusion_status_counts": {
                "quarantined_disease_mismatch": quarantined_count
            }
            if quarantined_count
            else {},
        },
        "anomaly_summary": {"anomaly_result_count": 0, "severity_counts": {}},
        "human_review_application_summary": {
            "decisions_provided_count": 0,
            "decisions_applied_count": 0,
            "decisions_rejected_count": 0,
        },
        "structured_extraction_summary": {
            "llm_enabled": True,
            "extraction_mode": "llm_structured_output",
            "raw_record_count": pre_quality_count,
        },
    }


def _make_session(
    tmp_path: Path,
    *,
    name: str = "synthetic_shanghai",
    disease: str = "hantavirus",
    location: str = "Shanghai",
    start_date: str = "2024",
    end_date: str = "2026",
    final_records: list[dict] | None = None,
    pre_quality_records: list[dict] | None = None,
    quarantined_records: list[dict] | None = None,
    pending_records: list[dict] | None = None,
    run_quality_status: str = "no_task_relevant_records",
    include_collection_spec: bool = True,
) -> Path:
    final_records = list(final_records or [])
    pre_quality_records = list(pre_quality_records or [])
    quarantined_records = list(quarantined_records or [])
    pending_records = list(pending_records or [])
    session = tmp_path / name
    summaries = _summaries(
        disease=disease,
        location=location,
        start_date=start_date,
        end_date=end_date,
        run_quality_status=run_quality_status,
        final_dataset_count=len(final_records),
        pre_quality_count=len(pre_quality_records),
        quarantined_count=len(quarantined_records),
        pending_count=len(pending_records),
    )
    if not include_collection_spec:
        summaries["task_intake_summary"] = {}

    package = {
        "final_dataset": final_records,
        "final_dataset_pre_quality_gate": pre_quality_records,
        "quarantined_records": quarantined_records,
        "pending_review_records": pending_records,
        "final_dataset_post_review": final_records,
        "record_inclusion_decisions": [
            {
                "record_id": row["record_id"],
                "record_final_inclusion_status": row.get("record_final_inclusion_status"),
                "reason": "synthetic quality gate decision",
            }
            for row in pre_quality_records + quarantined_records + pending_records
        ],
        "workflow_summaries": summaries,
        "run_quality_summary": summaries["run_quality_summary"],
        "final_dataset_quality_summary": summaries["final_dataset_quality_summary"],
        "source_registry": [
            {
                "source_id": "src_search_001",
                "title": f"{location} {disease} source",
                "publisher": "Shanghai Municipal Health Commission",
                "canonical_url": "https://wsjkw.sh.gov.cn/example",
                "source_role_final": "collection",
                "discovery_method": "live_search",
                "ready_for_content_fetch": False,
            }
        ],
        "human_review_items": [{"review_id": "review_source_001", "item_type": "source_critic_blocked_source"}],
        "collection_trace": [
            {"node_name": "task_intake_and_scope_planning", "message": "Parsed task."},
            {"node_name": "final_data_package_builder", "message": "Built package."},
        ],
        "package_metadata": {
            "disease": disease,
            "geography": location,
            "time_window": f"{start_date}-{end_date}",
            "llm_used": True,
            "contains_synthetic_fixture_data": False,
        },
        "provenance_manifest": {
            "source_count": 1,
            "normalized_record_count": len(pre_quality_records),
        },
    }
    run_summary = {
        "session_id": name,
        "user_request": f"Collect {disease} data for {location} from {start_date} to {end_date}.",
        "live_fetch_enabled": True,
        "live_search_enabled": True,
        "source_search_mode": "live",
        "source_search_provider": "tavily",
        "document_count": 0,
        "normalized_record_count": len(pre_quality_records),
        "run_quality_status": run_quality_status,
        "final_dataset_count": len(final_records),
        "final_dataset_pre_quality_gate_count": len(pre_quality_records),
        "quarantined_record_count": len(quarantined_records),
        "pending_review_record_count": len(pending_records),
        "final_dataset_post_review_count": len(final_records),
        "llm_stage_summary": _llm_stage_summary(),
        "source_search_execution_summary": summaries["source_search_execution_summary"],
        "source_critic_summary": summaries["source_critic_summary"],
        "disease_relevance_summary": summaries["disease_relevance_summary"],
        "validation_source_compatibility_status": summaries[
            "validation_source_compatibility_summary"
        ]["compatibility_status"],
    }

    _write_json(session / "collection" / "final_package.json", package)
    _write_json(session / "workflow_run_summary.json", run_summary)
    _write_json(session / "diagnostics" / "workflow_summaries.json", summaries)
    _write_json(session / "diagnostics" / "llm_stage_summary.json", _llm_stage_summary())
    _write_json(session / "diagnostics" / "live_fetch_summary.json", {"live_fetch_enabled": True, "documents": []})
    for key, value in summaries.items():
        if key.endswith("_summary"):
            _write_json(session / "diagnostics" / f"{key}.json", value)
    _write_json(session / "diagnostics" / "normalized_records.json", pre_quality_records)
    _write_json(session / "diagnostics" / "raw_records.json", pre_quality_records)
    _write_json(session / "diagnostics" / "validated_records.json", pre_quality_records)
    _write_json(session / "diagnostics" / "validation_results.json", [])
    _write_json(session / "diagnostics" / "anomaly_results.json", [])
    _write_json(session / "diagnostics" / "fetch_manifest.json", [])
    return session


def _build_console_for_session(tmp_path: Path, session: Path) -> tuple[dict, str]:
    from build_console import build_report

    output_dir = tmp_path / "console"
    summary = build_report(output_dir, session)
    html = (output_dir / "data_collection_workflow_console.html").read_text(encoding="utf-8")
    summary_json = json.loads(
        (output_dir / "data_collection_workflow_console_summary.json").read_text(encoding="utf-8")
    )
    assert summary == summary_json
    return summary_json, html


def test_console_does_not_use_static_new_mexico_truth_label_for_shanghai(tmp_path):
    session = _make_session(tmp_path)

    summary, html = _build_console_for_session(tmp_path, session)

    assert "live NMDOH/CDC webpage fetch" not in html
    assert "New Mexico HPS live LLM workflow run" not in html
    assert "current successful product run" not in html
    assert "Shanghai" in html
    assert "hantavirus" in html
    assert "no_task_relevant_records" in html
    assert summary["task_location"] == "Shanghai"
    assert summary["run_quality_summary"]["run_quality_status"] == "no_task_relevant_records"


def test_console_does_not_substitute_new_mexico_collection_spec_for_non_new_mexico(tmp_path):
    session = _make_session(tmp_path, include_collection_spec=False)

    summary, html = _build_console_for_session(tmp_path, session)

    assert summary["task_location"] == "collection_spec unavailable in artifacts"
    assert "collection_spec unavailable in artifacts" in html
    assert summary["task_location"] != "New Mexico"


def test_console_shows_quality_gated_dataset_counts(tmp_path):
    pre_quality = [_record("rec_pre_1"), _record("rec_pre_2"), _record("rec_pre_3")]
    quarantined = list(pre_quality)
    session = _make_session(
        tmp_path,
        pre_quality_records=pre_quality,
        quarantined_records=quarantined,
        run_quality_status="failed_quality_gate",
    )

    summary, html = _build_console_for_session(tmp_path, session)

    assert summary["accepted_record_count"] == 0
    assert summary["pre_quality_record_count"] == 3
    assert summary["quarantined_record_count"] == 3
    assert summary["pending_review_record_count"] == 0
    assert summary["post_review_record_count"] == 0
    assert "quality_gated_accepted_records" in html
    assert "FAILED QUALITY GATE" in html


def test_console_displays_validation_source_compatibility(tmp_path):
    session = _make_session(tmp_path)

    summary, html = _build_console_for_session(tmp_path, session)

    assert (
        summary["validation_source_compatibility_summary"]["compatibility_status"]
        == "incompatible_validation_source_disabled"
    )
    assert "incompatible_validation_source_disabled" in html
    assert "missing_collection" not in html
    assert "missing_validation" not in html


def test_console_displays_source_critic_disease_relevance_and_localized_planning(tmp_path):
    session = _make_session(tmp_path)

    summary, html = _build_console_for_session(tmp_path, session)

    assert summary["source_critic_summary"]["assessed_source_count"] == 6
    assert summary["source_critic_summary"]["blocked_fetch_count"] == 2
    assert summary["disease_relevance_summary"]["chunk_unrelated_disease_count"] == 3
    assert summary["localized_source_planning_summary"]["localized_query_count"] == 4
    for expected in ("Source critic", "Disease relevance", "Localized source planning"):
        assert expected in html


def test_new_mexico_compatibility_console_can_still_mention_new_mexico(tmp_path):
    accepted = [_record("rec_nm", location="New Mexico")]
    accepted[0]["record_final_inclusion_status"] = "accepted"
    session = _make_session(
        tmp_path,
        name="synthetic_new_mexico",
        disease="hantavirus",
        location="New Mexico",
        final_records=accepted,
        pre_quality_records=accepted,
        run_quality_status="passed",
    )

    summary, html = _build_console_for_session(tmp_path, session)

    assert summary["task_location"] == "New Mexico"
    assert "New Mexico" in html
    assert summary["accepted_record_count"] == 1
    assert "PASSED: accepted quality-gated records produced." in html


def test_current_console_source_has_no_hantavirus_record_only_wording():
    text = (PROJECT_ROOT / "scripts" / "build_console.py").read_text(encoding="utf-8")

    assert "抽取 HantavirusRecord" not in text
    assert "HantavirusRecord" not in text
    assert "PublicHealthRecord" in text or "generic public-health records" in text


def test_console_generated_labels_and_artifact_links_are_english(tmp_path):
    session = _make_session(tmp_path)
    summary, html = _build_console_for_session(tmp_path, session)
    assert not re.search(r"[\u3400-\u9fff\uf900-\ufaff]", html)
    assert "_chinese.md" not in html
    assert not any(key.endswith("_zh") for key in summary)
