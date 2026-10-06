import csv
import json
from pathlib import Path

from scripts.build_console import build_report as build_console_report

from data_collection_workflow.review_reporting import (
    build_human_review_priority_summary,
    build_review_decision_prefill,
    build_review_decision_template,
    build_review_packet_index,
    build_top_review_items,
    load_human_review_artifacts,
    write_human_review_workflow_artifacts,
)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_session(tmp_path: Path) -> Path:
    session = tmp_path / "session"
    collection = session / "collection"
    diagnostics = session / "diagnostics"
    collection.mkdir(parents=True)
    diagnostics.mkdir(parents=True)

    source_registry = [
        {
            "source_id": "src_vdh",
            "canonical_url": "https://www.vdh.virginia.gov/example",
            "title": "Virginia hantavirus monitoring update",
            "publisher": "Virginia Department of Health",
            "actual_publisher": "Virginia Department of Health",
            "source_type": "official_public_health_agency",
            "source_type_final": "state_or_local_public_health_agency",
            "source_role_final": "collection",
            "actual_publisher_confidence": "high",
            "source_identity_warnings": [],
        },
        {
            "source_id": "src_unknown",
            "canonical_url": "https://example.com/possible-case",
            "title": "Possible hantavirus case discussed",
            "publisher": "Search provider metadata",
            "actual_publisher": None,
            "source_type": "news_and_situation_report",
            "source_type_final": "unknown",
            "source_role_final": "collection_support",
            "actual_publisher_confidence": "low",
            "source_identity_warnings": ["actual_publisher_unknown"],
        },
    ]
    record = {
        "record_id": "rec_possible_primary",
        "disease": "hantavirus",
        "country": "United States of America",
        "subnational_location": "Virginia",
        "date_reported": "2025-05-01",
        "cases_confirmed": 1,
        "source_id": "src_unknown",
        "source_url": "https://example.com/possible-case",
        "source_title": "Possible hantavirus case discussed",
        "publisher": "Search provider metadata",
        "source_type": "news_and_situation_report",
        "evidence_quote": "A possible hantavirus case was discussed in Virginia.",
        "record_final_inclusion_status": "quarantined_needs_review",
        "quality_gate_reasons": ["single_source_unverified"],
        "requires_human_review": True,
        "human_review_reason": "single_source_unverified primary case claim",
        "primary_case_dataset_eligible": True,
    }
    context_record = {
        **record,
        "record_id": "rec_context",
        "cases_confirmed": None,
        "primary_case_dataset_eligible": False,
        "observation_type": "background_context",
        "evidence_quote": "Background prevention text for hantavirus.",
    }
    claims = [
        {
            "claim_id": "claim_primary",
            "record_id": "rec_possible_primary",
            "source_id": "src_unknown",
            "disease": "hantavirus",
            "location": "Virginia",
            "date_or_period": "2025-05-01",
            "observation_type": "confirmed_case_record",
            "is_case_claim": True,
            "primary_case_dataset_eligible": True,
            "claim_status": "pending_review",
            "evidence_quote": record["evidence_quote"],
        }
    ]
    claim_comparisons = [
        {
            "comparison_id": "cmp_001",
            "left_claim_id": "claim_primary",
            "right_claim_id": "claim_other",
            "comparability_status": "comparable",
            "corroboration_match_status": "conflict",
            "needs_human_review": True,
            "human_review_reason": "case count conflict",
        }
    ]
    corroborated_events = [
        {
            "event_id": "event_001",
            "claim_ids": ["claim_primary"],
            "record_ids": ["rec_possible_primary"],
            "source_ids": ["src_unknown"],
            "event_status": "single_source_unverified",
            "primary_case_dataset_eligible": True,
            "needs_human_review": True,
            "human_review_reason": "single source case claim",
        }
    ]
    anomaly_results = [
        {
            "anomaly_id": "anom_001",
            "anomaly_type": "count_semantics_unclear",
            "severity": "high",
            "record_id": "rec_possible_primary",
            "source_ids": ["src_unknown"],
            "evidence_summary": "Count semantics are unclear.",
            "needs_human_review": True,
            "human_review_reason": "count semantics unclear",
        }
    ]
    review_items = [
        {
            "review_id": "review_possible_primary",
            "item_type": "claim_corroboration_review",
            "related_ids": ["event_001", "claim_primary", "rec_possible_primary"],
            "reason": "single_source_unverified primary case claim",
            "status": "pending",
            "record_id": "rec_possible_primary",
            "event_cluster_id": "event_001",
            "source_ids": ["src_unknown"],
            "source_urls": ["https://example.com/possible-case"],
            "suggested_action": "review_claim_corroboration",
            "review_packet": {
                "packet_sections": {
                    "related_records": [record],
                    "source_registry_entry": source_registry[1],
                }
            },
        },
        {
            "review_id": "review_context",
            "item_type": "context_only_review",
            "related_ids": ["rec_context"],
            "reason": "context-only source classification needs confirmation",
            "status": "pending",
            "record_id": "rec_context",
            "review_packet": {"packet_sections": {"related_records": [context_record]}},
        },
        {
            "review_id": "review_source_identity",
            "item_type": "source_credibility",
            "related_ids": ["src_unknown"],
            "reason": "actual_publisher_unknown",
            "status": "pending",
            "source_ids": ["src_unknown"],
            "review_packet": {
                "packet_sections": {"source_registry_entry": source_registry[1]}
            },
        },
    ]
    package = {
        "final_dataset": [],
        "final_case_dataset": [],
        "final_dataset_pre_quality_gate": [record, context_record],
        "quarantined_records": [record],
        "pending_review_records": [],
        "context_records": [context_record],
        "human_review_items": review_items,
        "source_registry": source_registry,
        "source_identity_assessments": [
            {
                "source_id": "src_unknown",
                "actual_publisher": None,
                "actual_publisher_confidence": "low",
                "source_type_final": "unknown",
                "source_identity_warnings": ["actual_publisher_unknown"],
            }
        ],
        "claims": claims,
        "claim_comparisons": claim_comparisons,
        "corroborated_events": corroborated_events,
        "anomaly_results": anomaly_results,
        "record_inclusion_decisions": [
            {
                "record_id": "rec_possible_primary",
                "record_final_inclusion_status": "quarantined_needs_review",
                "reason": "single_source_unverified",
            }
        ],
        "run_quality_summary": {
            "task_disease": "hantavirus",
            "task_location": "Virginia",
            "task_start_date": "2025-01-01",
            "task_end_date": "2026-06-01",
            "run_quality_status": "failed_quality_gate",
            "primary_case_dataset_status": "no_corroborated_primary_case_events",
            "final_case_dataset_count": 0,
            "final_dataset_count": 0,
            "quarantined_record_count": 1,
            "context_record_count": 1,
            "validation_limited": True,
        },
        "final_dataset_quality_summary": {
            "primary_case_dataset_status": "no_corroborated_primary_case_events",
            "final_case_dataset_count": 0,
            "quarantined_record_count": 1,
            "context_record_count": 1,
        },
        "observation_type_dataset_summary": {
            "dataset_view_counts": {"context_records": 1, "final_case_dataset": 0}
        },
        "validation_source_compatibility_summary": {
            "compatibility_status": "incompatible_validation_source_disabled",
            "compatibility_reason": "No task-compatible held-out validation source was configured/found.",
        },
        "corroboration_summary": {
            "claim_count": 1,
            "claim_comparison_count": 1,
            "corroborated_primary_case_event_count": 0,
        },
    }
    run_summary = {
        "session_id": "synthetic_review_session",
        "user_request": "Collect hantavirus data for Virginia from 2025-01-01 to 2026-06-01.",
        "live_search_enabled": True,
        "live_fetch_enabled": True,
        "all_three_llm_stages_enabled": True,
        "human_review_item_count": len(review_items),
        "final_case_dataset_count": 0,
        "quarantined_record_count": 1,
        "context_record_count": 1,
        "run_quality_status": "failed_quality_gate",
        "artifact_paths": {},
    }
    interpretive_summary = {
        "task_disease": "hantavirus",
        "task_location": "Virginia",
        "task_start_date": "2025-01-01",
        "task_end_date": "2026-06-01",
        "final_case_dataset_count": 0,
        "context_record_count": 1,
        "quarantined_record_count": 1,
        "validation_limited": True,
        "run_quality_status": "failed_quality_gate",
        "primary_case_dataset_status": "no_corroborated_primary_case_events",
    }

    _write_json(collection / "final_package.json", package)
    for key, value in package.items():
        if isinstance(value, (dict, list)):
            _write_json(collection / f"{key}.json", value)
            _write_json(diagnostics / f"{key}.json", value)
    _write_json(diagnostics / "workflow_summaries.json", package)
    _write_json(session / "workflow_run_summary.json", run_summary)
    _write_json(session / "workflow_interpretive_report_summary.json", interpretive_summary)
    return session


def test_human_review_priority_summary_is_generated(tmp_path):
    session = _make_session(tmp_path)

    paths = write_human_review_workflow_artifacts(session)

    summary_path = session / "human_review" / "human_review_priority_summary.json"
    markdown_path = session / "human_review" / "human_review_priority_summary.md"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_path.exists()
    assert markdown_path.exists()
    assert summary["review_item_count"] == 3
    assert summary["prioritized_review_item_count"] >= 3
    assert summary["priority_level_counts"]
    assert summary["generated_from_artifacts_only"] is True
    assert summary["llm_called_for_review_productization"] is False
    assert summary["search_called_for_review_productization"] is False
    assert summary["fetch_called_for_review_productization"] is False
    assert Path(paths["human_review_priority_summary_json"]).exists()
    assert (session / "diagnostics" / "human_review_priority_summary.json").exists()
    assert (session / "collection" / "human_review_priority_summary.json").exists()


def test_top_review_items_are_sorted_by_priority(tmp_path):
    artifacts = load_human_review_artifacts(_make_session(tmp_path))

    items = build_top_review_items(artifacts)

    assert items[0]["priority_rank"] == 1
    assert items[0]["priority_level"] in {"P0_critical", "P1_high"}
    assert items[-1]["priority_level"] in {"P2_medium", "P3_low"}
    assert [row["priority_rank"] for row in items] == list(range(1, len(items) + 1))


def test_primary_case_dataset_blocker_receives_high_priority(tmp_path):
    artifacts = load_human_review_artifacts(_make_session(tmp_path))

    items = build_top_review_items(artifacts)
    item = next(row for row in items if row["review_item_id"] == "review_possible_primary")

    assert item["issue_category"] in {
        "primary_case_dataset_blocker",
        "possible_primary_case_evidence",
        "claim_corroboration_review",
    }
    assert item["priority_level"] in {"P0_critical", "P1_high"}
    assert item["blocking_final_case_dataset"] is True


def test_validation_limited_item_is_categorized_without_false_no_case_proof(tmp_path):
    artifacts = load_human_review_artifacts(_make_session(tmp_path))

    summary = build_human_review_priority_summary(artifacts)
    items = build_top_review_items(artifacts)

    assert summary["validation_limited"] is True
    assert summary["validation_limited_review_count"] >= 1
    assert any(row["issue_category"] == "validation_limited_review" for row in items)
    assert all("automatic false" not in row["why_it_matters"].lower() for row in items)


def test_source_identity_uncertainty_item_is_categorized(tmp_path):
    artifacts = load_human_review_artifacts(_make_session(tmp_path))

    items = build_top_review_items(artifacts)
    item = next(row for row in items if row["review_item_id"] == "review_source_identity")

    assert item["issue_category"] == "source_identity_review"
    assert "source_identity_assessments" in item["recommended_artifacts_to_open"]


def test_claim_corroboration_conflict_routes_to_review_category(tmp_path):
    artifacts = load_human_review_artifacts(_make_session(tmp_path))

    items = build_top_review_items(artifacts)
    item = next(row for row in items if row["review_item_id"] == "review_possible_primary")

    assert item["issue_category"] in {
        "claim_corroboration_review",
        "conflicting_claims_review",
        "possible_primary_case_evidence",
        "primary_case_dataset_blocker",
    }
    assert "claims" in item["recommended_artifacts_to_open"]
    assert "corroborated_events" in item["recommended_artifacts_to_open"]


def test_decision_template_is_valid_and_non_applying(tmp_path):
    artifacts = load_human_review_artifacts(_make_session(tmp_path))

    template = build_review_decision_template(artifacts)

    assert template["supported_decision_types"]
    for decision in template["decisions"]:
        assert decision["apply_decision"] is False
        assert decision["decision_type"] in template["supported_decision_types"]


def test_decision_prefill_is_non_applying(tmp_path):
    session = _make_session(tmp_path)

    write_human_review_workflow_artifacts(session)
    prefill = json.loads(
        (session / "human_review" / "review_decision_prefill.json").read_text(
            encoding="utf-8"
        )
    )
    csv_rows = list(
        csv.DictReader(
            (session / "human_review" / "review_decision_prefill.csv").open(
                encoding="utf-8"
            )
        )
    )
    assert prefill["decisions"]
    assert csv_rows
    assert all(row["apply_decision"] is False for row in prefill["decisions"])
    assert all(row["apply_decision"] == "False" for row in csv_rows)
    assert all(row["target_ids"] for row in prefill["decisions"])


def test_review_packet_index_links_artifacts(tmp_path):
    artifacts = load_human_review_artifacts(_make_session(tmp_path))

    index = build_review_packet_index(artifacts)

    assert index["review_packets"]
    packet = index["review_packets"][0]
    assert packet["review_id"]
    assert packet["recommended_artifacts_to_open"]


def test_review_action_guide_includes_apply_instructions(tmp_path):
    session = _make_session(tmp_path)

    write_human_review_workflow_artifacts(session)
    guide = (session / "human_review" / "review_action_guide.md").read_text(
        encoding="utf-8"
    )

    assert "apply_decision=true" in guide
    assert "review_decision_prefill.json" in guide
    assert "final_dataset_post_review" in guide


def test_runner_style_writer_exports_all_human_review_artifacts(tmp_path):
    session = _make_session(tmp_path)

    paths = write_human_review_workflow_artifacts(session)
    summary = json.loads((session / "workflow_run_summary.json").read_text(encoding="utf-8"))

    expected_names = [
        "human_review_priority_summary.json",
        "human_review_priority_summary.md",
        "top_review_items.csv",
        "top_review_items.json",
        "review_decision_template.json",
        "review_decision_prefill.json",
        "review_decision_prefill.csv",
        "review_packet_index.json",
        "review_action_guide.md",
    ]
    for name in expected_names:
        assert (session / "human_review" / name).exists()
    assert "human_review_priority_summary" in paths
    assert "human_review_priority_summary" in summary["artifact_paths"]


def test_console_links_human_review_workflow_artifacts(tmp_path):
    session = _make_session(tmp_path)
    write_human_review_workflow_artifacts(session)

    console_summary = build_console_report(tmp_path / "console", session)
    html = Path(console_summary["html_path"]).read_text(encoding="utf-8")

    assert "human_review_priority_summary.md" in html
    assert "top_review_items.csv" in html
    assert "review_decision_template.json" in html
    assert "review_decision_prefill.json" in html
    assert "review_action_guide.md" in html
