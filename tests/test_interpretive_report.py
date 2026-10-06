from __future__ import annotations

import json
from pathlib import Path

from scripts.build_console import build_report as build_console_report
from scripts.run_workflow import _write_interpretive_report_outputs

from data_collection_workflow.interpretive_report import (
    build_interpretive_report_summary,
    write_interpretive_reports,
)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _base_record(**overrides) -> dict:
    record = {
        "record_id": "rec_001",
        "disease": "hantavirus",
        "location": "Virginia",
        "date_reported": "2025-03-01",
        "cases_confirmed": 1,
        "deaths": 0,
        "source_title": "Virginia hantavirus report",
        "actual_publisher": "Virginia Department of Health",
        "source_url": "https://example.test/vdh-hantavirus",
        "source_type_final": "official_public_health_agency",
        "corroboration_status": "single_source_unverified",
        "independent_source_count": 1,
        "evidence_quote": "One confirmed hantavirus case was reported.",
    }
    record.update(overrides)
    return record


def _make_session(
    tmp_path: Path,
    *,
    session_id: str = "synthetic_session",
    final_case_records: list[dict] | None = None,
    zero_case_records: list[dict] | None = None,
    exposure_records: list[dict] | None = None,
    context_records: list[dict] | None = None,
    quarantined_records: list[dict] | None = None,
    human_review_items: list[dict] | None = None,
    validation_status: str = "incompatible_validation_source_disabled",
) -> Path:
    session_dir = tmp_path / session_id
    collection_dir = session_dir / "collection"
    diagnostics_dir = session_dir / "diagnostics"
    final_case_records = list(final_case_records or [])
    zero_case_records = list(zero_case_records or [])
    exposure_records = list(exposure_records or [])
    context_records = list(context_records or [])
    quarantined_records = list(quarantined_records or [])
    human_review_items = list(
        human_review_items
        or [
            {
                "review_id": "review_001",
                "item_type": "quality_gate",
                "reason": "No primary case records passed final gates.",
            }
        ]
    )
    final_dataset = list(final_case_records)
    pre_quality = final_case_records + zero_case_records + exposure_records + context_records
    run_quality_status = (
        "partial_with_quarantined_records"
        if final_case_records
        else "failed_quality_gate"
    )
    primary_status = (
        "primary_case_records_present"
        if final_case_records
        else "no_corroborated_primary_case_events"
    )
    summary = {
        "session_id": session_id,
        "user_request": "Collect hantavirus data for Virginia from 2025-01-01 to 2026-06-01.",
        "live_search_enabled": True,
        "live_fetch_enabled": True,
        "source_search_provider": "tavily",
        "source_registry_count": 4,
        "document_count": 2,
        "normalized_record_count": len(pre_quality),
        "claim_count": 3,
        "claim_comparison_count": 3,
        "corroborated_event_count": 1,
        "corroborated_primary_case_event_count": 0,
        "final_dataset_count": len(final_dataset),
        "final_case_dataset_count": len(final_case_records),
        "zero_case_statement_count": len(zero_case_records),
        "exposure_monitoring_record_count": len(exposure_records),
        "surveillance_summary_record_count": 0,
        "outbreak_summary_record_count": 0,
        "context_record_count": len(context_records),
        "unclassified_observation_count": 0,
        "non_primary_observation_count": len(zero_case_records)
        + len(exposure_records)
        + len(context_records),
        "quarantined_record_count": len(quarantined_records),
        "pending_review_record_count": 0,
        "final_dataset_post_review_count": len(final_dataset),
        "run_quality_status": run_quality_status,
        "primary_case_dataset_status": primary_status,
        "validation_source_compatibility_status": validation_status,
        "active_validation_record_count": 0,
        "inactive_validation_record_count": 1,
        "all_three_llm_stages_enabled": True,
        "llm_stage_summary": {
            "source_planning": {"enabled": True},
            "source_critic": {"enabled": True},
            "source_identity": {"enabled": True},
            "structured_extraction": {"enabled": True},
        },
        "source_search_execution_summary": {
            "search_enabled": True,
            "live_search_enabled": True,
            "search_provider": "tavily",
            "candidate_from_search_count": 5,
        },
        "artifact_paths": {},
    }
    run_quality_summary = {
        "run_quality_status": run_quality_status,
        "final_case_dataset_count": len(final_case_records),
        "final_dataset_count": len(final_dataset),
        "final_dataset_pre_quality_gate_count": len(pre_quality),
        "zero_case_statement_count": len(zero_case_records),
        "exposure_monitoring_record_count": len(exposure_records),
        "context_record_count": len(context_records),
        "quarantined_record_count": len(quarantined_records),
        "pending_review_record_count": 0,
        "final_dataset_post_review_count": len(final_dataset),
        "primary_case_dataset_status": primary_status,
        "validation_limited": validation_status != "compatible",
        "no_primary_case_dataset_records": not bool(final_case_records),
        "recommended_primary_dataset_message": "Inspect non-primary views.",
    }
    final_package = {
        "final_dataset": final_dataset,
        "final_case_dataset": final_case_records,
        "final_dataset_pre_quality_gate": pre_quality,
        "final_dataset_post_review": final_dataset,
        "zero_case_statements": zero_case_records,
        "exposure_monitoring_records": exposure_records,
        "surveillance_summary_records": [],
        "outbreak_summary_records": [],
        "context_records": context_records,
        "unclassified_observation_records": [],
        "non_primary_observations": zero_case_records + exposure_records + context_records,
        "quarantined_records": quarantined_records,
        "pending_review_records": [],
        "human_review_items": human_review_items,
        "claims": [{"claim_id": "claim_001", "observation_type": "background_context"}],
        "claim_comparisons": [{"comparison_id": "cmp_001"}],
        "corroborated_events": [
            {
                "corroborated_event_id": "evt_001",
                "corroboration_status": "single_source_unverified",
                "primary_case_dataset_eligible": bool(final_case_records),
            }
        ],
        "source_identity_assessments": [
            {
                "source_id": "src_001",
                "source_type_final": "official_public_health_agency",
                "actual_publisher": "Virginia Department of Health",
            }
        ],
        "run_quality_summary": run_quality_summary,
        "final_dataset_quality_summary": dict(run_quality_summary),
        "observation_type_dataset_summary": {
            "dataset_view_counts": {
                "final_case_dataset": len(final_case_records),
                "zero_case_statements": len(zero_case_records),
                "exposure_monitoring_records": len(exposure_records),
                "context_records": len(context_records),
            }
        },
        "corroboration_summary": {
            "claim_count": 3,
            "claim_comparison_count": 3,
            "corroborated_event_count": 1,
            "corroborated_primary_case_event_count": 0,
            "single_source_unverified_count": 1,
            "conflicting_claim_count": 0,
        },
        "source_identity_summary": {
            "identity_assessed_count": 1,
            "unknown_publisher_count": 0,
            "source_type_counts": {"official_public_health_agency": 1},
        },
        "validation_source_compatibility_summary": {
            "compatibility_status": validation_status,
            "active_validation_record_count": 0,
            "inactive_validation_record_count": 1,
        },
        "anomaly_summary": {"anomaly_count": 0, "severity_counts": {}},
        "human_review_application_summary": {
            "human_review_item_count": len(human_review_items),
            "decisions_applied_count": 0,
        },
        "workflow_summaries": {},
    }
    _write_json(session_dir / "workflow_run_summary.json", summary)
    _write_json(collection_dir / "final_package.json", final_package)
    for name, value in {
        "final_case_dataset": final_case_records,
        "final_dataset": final_dataset,
        "final_dataset_pre_quality_gate": pre_quality,
        "final_dataset_post_review": final_dataset,
        "zero_case_statements": zero_case_records,
        "exposure_monitoring_records": exposure_records,
        "surveillance_summary_records": [],
        "outbreak_summary_records": [],
        "context_records": context_records,
        "unclassified_observation_records": [],
        "non_primary_observations": zero_case_records + exposure_records + context_records,
        "quarantined_records": quarantined_records,
        "pending_review_records": [],
        "claims": final_package["claims"],
        "corroborated_events": final_package["corroborated_events"],
        "source_identity_assessments": final_package["source_identity_assessments"],
    }.items():
        _write_json(collection_dir / f"{name}.json", value)
    _write_json(collection_dir / "record_inclusion_decisions.json", [])
    for name in (
        "run_quality_summary",
        "final_dataset_quality_summary",
        "observation_type_dataset_summary",
        "corroboration_summary",
        "source_identity_summary",
        "validation_source_compatibility_summary",
        "anomaly_summary",
        "human_review_application_summary",
    ):
        _write_json(diagnostics_dir / f"{name}.json", final_package[name])
    _write_json(diagnostics_dir / "workflow_summaries.json", final_package)
    return session_dir


def test_interpretive_writer_emits_only_english_report_and_summary(tmp_path):
    session = _make_session(
        tmp_path,
        context_records=[_base_record(record_id="ctx_001", observation_type="background_context")],
    )

    paths = write_interpretive_reports(session)
    assert set(paths) == {"english_report", "summary_json"}
    assert not list(session.rglob("*_chinese.md"))
    summary = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
    assert "one_sentence_conclusion_zh" not in summary
    assert not any("chinese" in name for name in summary["key_artifacts"])
    text = Path(paths["english_report"]).read_text(encoding="utf-8")
    assert "no accepted primary case dataset records were found" in text
    assert "not suitable as a final epidemiological dataset" in text
    system_output = text + json.dumps(summary, ensure_ascii=False)
    system_output = system_output.replace(str(tmp_path), "<test-dir>").replace(tmp_path.as_posix(), "<test-dir>")
    assert not any("\u4e00" <= char <= "\u9fff" for char in system_output)


def test_interpretive_report_preserves_original_multilingual_evidence(tmp_path):
    quote = "One confirmed hantavirus case was reported. 原文记录保留。"
    session = _make_session(tmp_path, final_case_records=[_base_record(evidence_quote=quote)])
    paths = write_interpretive_reports(session)
    text = Path(paths["english_report"]).read_text(encoding="utf-8")
    assert quote in text
    assert "Primary case dataset findings" in text


def test_english_interpretive_report_generated_from_session_artifacts(tmp_path):
    session = _make_session(
        tmp_path,
        context_records=[_base_record(record_id="ctx_001", observation_type="background_context")],
    )

    paths = write_interpretive_reports(session)
    text = Path(paths["english_report"]).read_text(encoding="utf-8")

    assert "# Data Collection Result Interpretation Report" in text
    assert "## 2. One-sentence conclusion" in text
    assert "no accepted primary case dataset records were found" in text


def test_report_distinguishes_technical_completion_from_data_success(tmp_path):
    session = _make_session(
        tmp_path,
        context_records=[_base_record(record_id="ctx_001", observation_type="background_context")],
    )

    en = build_interpretive_report_summary(session)["one_sentence_conclusion_en"]

    assert "technically completed" in en
    assert "not suitable as a final epidemiological dataset" in en


def test_report_summarizes_primary_case_dataset_when_present(tmp_path):
    session = _make_session(tmp_path, final_case_records=[_base_record()])

    paths = write_interpretive_reports(session)
    en = Path(paths["english_report"]).read_text(encoding="utf-8")

    assert "Virginia Department of Health" in en
    assert "One confirmed hantavirus case was reported." in en
    assert "produced 1 accepted primary case dataset record" in en
    assert "Expert review is still required" in en


def test_report_summarizes_non_case_observations_separately(tmp_path):
    session = _make_session(
        tmp_path,
        zero_case_records=[_base_record(record_id="zero_001", cases_confirmed=0)],
        exposure_records=[
            _base_record(
                record_id="exp_001",
                cases_confirmed=0,
                evidence_quote="People completed monitoring and remained healthy.",
            )
        ],
        context_records=[_base_record(record_id="ctx_001", observation_type="background_context")],
    )

    paths = write_interpretive_reports(session)
    en = Path(paths["english_report"]).read_text(encoding="utf-8")

    assert "Zero-case and exposure-monitoring observations are not confirmed cases" in en
    assert "context/background evidence" in en
    assert "not confirmed cases" in en


def test_report_summarizes_corroboration_without_truth_determination(tmp_path):
    session = _make_session(tmp_path, context_records=[_base_record(record_id="ctx_001")])

    text = build_interpretive_report_summary(session)["one_sentence_conclusion_en"]
    paths = write_interpretive_reports(session)
    report = Path(paths["english_report"]).read_text(encoding="utf-8")

    assert "claim_count" in report
    assert "corroborated_primary_case_event_count" in report
    assert "automatic truth determination" in report
    assert "no accepted primary case dataset records were found" in text


def test_report_summarizes_validation_limitations(tmp_path):
    session = _make_session(
        tmp_path,
        validation_status="no_task_compatible_validation_source",
        context_records=[_base_record(record_id="ctx_001")],
    )

    paths = write_interpretive_reports(session)
    en = Path(paths["english_report"]).read_text(encoding="utf-8")

    assert "validation is limited" in en
    assert "does not prove that no case occurred" in en


def test_report_summarizes_source_quality(tmp_path):
    session = _make_session(tmp_path, context_records=[_base_record(record_id="ctx_001")])

    paths = write_interpretive_reports(session)
    report = Path(paths["english_report"]).read_text(encoding="utf-8")

    assert "Source quality and credibility" in report
    assert "official_public_health_agency" in report
    assert "source_identity_assessed_count" in report


def test_report_explains_parsed_official_sources_with_no_extracted_records(tmp_path):
    session = _make_session(tmp_path, context_records=[])
    _write_json(
        session / "diagnostics" / "source_coverage_audit.json",
        {
            "coverage_status": "target_official_source_parsed",
            "requirement_count": 2,
            "discovered_requirement_count": 2,
            "fetched_requirement_count": 2,
            "parsed_requirement_count": 2,
            "extracted_requirement_count": 0,
            "accepted_requirement_count": 0,
            "extracted_record_count": 0,
            "accepted_record_count": 0,
            "requirements": [
                {
                    "requirement_id": "virginia_influenza_official_week_40_2024",
                    "discovered": True,
                    "fetched": True,
                    "parsed": True,
                    "extracted": False,
                    "accepted": False,
                    "extracted_record_count": 0,
                    "accepted_record_count": 0,
                }
            ],
        },
    )
    _write_json(
        session / "diagnostics" / "structured_extraction_summary.json",
        {
            "official_extraction_failures": [
                {
                    "source_id": "src_vdh_week_40",
                    "reason": "must_fetch_source_produced_no_records",
                }
            ]
        },
    )

    paths = write_interpretive_reports(session)
    en = Path(paths["english_report"]).read_text(encoding="utf-8")

    assert "Coverage and extraction status" in en
    assert "official sources were discovered/fetched/parsed" in en
    assert "produced no extracted records" in en
    assert "must_fetch_source_produced_no_records" in en


def test_report_explains_target_fetch_failures_before_extraction(tmp_path):
    session = _make_session(tmp_path, context_records=[])
    _write_json(
        session / "diagnostics" / "source_coverage_audit.json",
        {
            "coverage_status": "target_official_source_fetch_failed",
            "requirement_count": 1,
            "discovered_requirement_count": 1,
            "fetched_requirement_count": 0,
            "fetch_failed_requirement_count": 1,
            "parsed_requirement_count": 0,
            "unusable_requirement_count": 1,
            "extracted_requirement_count": 0,
            "accepted_requirement_count": 0,
            "requirements": [
                {
                    "requirement_id": "united_states_influenza_official_week_1_2025",
                    "discovered": True,
                    "fetch_attempted": True,
                    "fetch_failed": True,
                    "fetched": False,
                    "parsed": False,
                    "unusable": True,
                }
            ],
        },
    )
    _write_json(
        session / "diagnostics" / "content_fetch_summary.json",
        {
            "fetch_failures_blocking_count": 1,
            "fetch_status_counts": {"fetch_failed": 1},
        },
    )
    _write_json(
        session / "diagnostics" / "evidence_chunking_summary.json",
        {"total_chunk_count": 0},
    )
    _write_json(
        session / "diagnostics" / "structured_extraction_summary.json",
        {"input_chunk_count": 0, "raw_record_count": 0},
    )

    paths = write_interpretive_reports(session)
    en = Path(paths["english_report"]).read_text(encoding="utf-8")

    assert "Coverage and extraction status" in en
    assert "source verification chain" in en
    assert "fetch_failed=1" in en
    assert "failure_stage: `all_target_fetch_failed`" in en


def test_report_includes_human_review_priorities(tmp_path):
    session = _make_session(
        tmp_path,
        human_review_items=[
            {
                "review_id": "review_001",
                "item_type": "validation_limited",
                "reason": "Review whether quarantined sources are in scope.",
            }
        ],
    )

    paths = write_interpretive_reports(session)
    report = Path(paths["english_report"]).read_text(encoding="utf-8")

    assert "Human review priorities" in report
    assert "Review whether quarantined sources are in scope" in report


def test_summary_json_exported(tmp_path):
    session = _make_session(tmp_path, context_records=[_base_record(record_id="ctx_001")])

    paths = write_interpretive_reports(session)
    summary = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))

    assert "one_sentence_conclusion_zh" not in summary
    assert summary["one_sentence_conclusion_en"]
    assert summary["final_case_dataset_count"] == 0
    assert summary["suitable_as_final_epidemiological_dataset"] is False
    assert summary["generated_from_artifacts_only"] is True
    assert summary["llm_called_for_report"] is False
    assert summary["search_called_for_report"] is False
    assert summary["fetch_called_for_report"] is False
    assert "workflow_interpretive_report.md" in summary["key_artifacts"]
    assert "workflow_interpretive_report_chinese.md" not in summary["key_artifacts"]


def test_runner_writes_interpretive_reports(tmp_path):
    session = _make_session(tmp_path, context_records=[_base_record(record_id="ctx_001")])

    paths = _write_interpretive_report_outputs(
        session,
        write_latest_alias=False,
    )

    assert "interpretive_report_chinese" not in paths
    assert Path(paths["interpretive_report_english"]).exists()
    assert Path(paths["interpretive_report_summary"]).exists()


def test_console_links_interpretive_reports(tmp_path):
    session = _make_session(tmp_path, context_records=[_base_record(record_id="ctx_001")])
    write_interpretive_reports(session)

    console_summary = build_console_report(tmp_path / "console", session)
    html = Path(console_summary["html_path"]).read_text(encoding="utf-8")

    assert "workflow_interpretive_report_chinese.md" not in html
    assert "workflow_interpretive_report.md" in html
    assert "workflow_interpretive_report_summary.json" in html


def test_reports_include_disclaimers(tmp_path):
    session = _make_session(tmp_path, context_records=[_base_record(record_id="ctx_001")])

    paths = write_interpretive_reports(session)
    en = Path(paths["english_report"]).read_text(encoding="utf-8")

    assert (
        "Note: This report interprets evidence collected by the workflow. It is not an official surveillance conclusion, medical advice, or automatic truth determination."
        in en
    )


def test_report_avoids_overclaiming_when_no_primary_case_dataset(tmp_path):
    session = _make_session(tmp_path, context_records=[_base_record(record_id="ctx_001")])

    paths = write_interpretive_reports(session)
    report_text = Path(paths["english_report"]).read_text(encoding="utf-8")
    banned = [
        "confirmed " + "true",
        "official " + "truth established",
        "definitely " + "happened",
        "definitely " + "did not happen",
    ]

    for phrase in banned:
        assert phrase not in report_text
