from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.build_console import build_report as build_console_report

from data_collection_workflow.workflow_visualization import (
    build_claim_comparison_cards,
    build_dataset_decision_flow,
    build_evidence_flow_graph,
    build_human_review_visualization,
    build_workflow_graph_topology,
    build_workflow_timeline,
    build_workflow_visualization_summary,
    load_workflow_visualization_artifacts,
    write_workflow_visualization_artifacts,
)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_session(tmp_path: Path, *, final_case: bool = True) -> Path:
    session = tmp_path / ("session_final_case" if final_case else "session_empty_case")
    collection = session / "collection"
    diagnostics = session / "diagnostics"
    human_review = session / "human_review"
    collection.mkdir(parents=True)
    diagnostics.mkdir(parents=True)
    human_review.mkdir(parents=True)

    source = {
        "source_id": "src_vdh",
        "url": "https://www.vdh.virginia.gov/example",
        "canonical_url": "https://www.vdh.virginia.gov/example",
        "title": "Virginia hantavirus update",
        "publisher": "Virginia Department of Health",
        "actual_publisher": "Virginia Department of Health",
        "source_type_final": "state_or_local_public_health_agency",
        "source_role_final": "collection",
        "credibility_level": "high",
        "source_identity_status": "assessed",
    }
    source_support = {
        "source_id": "src_support",
        "url": "https://example.org/support",
        "canonical_url": "https://example.org/support",
        "title": "Support source",
        "publisher": "Example News",
        "actual_publisher": "Example News",
        "source_type_final": "news_or_secondary_report",
        "source_role_final": "collection_support",
        "credibility_level": "medium",
    }
    document = {
        "document_id": "doc_src_vdh_001",
        "source_id": "src_vdh",
        "source_url": source["canonical_url"],
        "title": source["title"],
        "parse_status": "parsed_html",
        "quality_status": "usable",
    }
    chunk = {
        "chunk_id": "chunk_src_vdh_001",
        "document_id": "doc_src_vdh_001",
        "source_id": "src_vdh",
        "text": "Virginia reported one confirmed hantavirus case in 2025.",
        "data_types": ["case_count", "date", "location"],
        "confidence": "medium",
    }
    record = {
        "record_id": "rec_src_vdh_001",
        "source_id": "src_vdh",
        "source_url": source["canonical_url"],
        "source_title": source["title"],
        "publisher": source["publisher"],
        "document_id": "doc_src_vdh_001",
        "supporting_chunk_id": "chunk_src_vdh_001",
        "disease": "hantavirus",
        "subnational_location": "Virginia",
        "date_reported": "2025-05-01",
        "cases_confirmed": 1,
        "observation_type": "confirmed_case_record",
        "primary_case_dataset_eligible": True,
        "evidence_quote": "Virginia reported one confirmed hantavirus case in 2025.",
        "record_final_inclusion_status": "accepted_final_case_dataset"
        if final_case
        else "quarantined_needs_review",
        "quality_gate_reasons": ["accepted"] if final_case else ["single_source_unverified"],
    }
    context_record = {
        **record,
        "record_id": "rec_context_001",
        "cases_confirmed": None,
        "observation_type": "background_context",
        "primary_case_dataset_eligible": False,
        "record_final_inclusion_status": "context_records",
        "quality_gate_reasons": ["background_context"],
        "evidence_quote": "Hantavirus prevention guidance for Virginia residents.",
    }
    claim = {
        "claim_id": "claim_rec_src_vdh_001_cases",
        "source_record_id": "rec_src_vdh_001",
        "record_id": "rec_src_vdh_001",
        "source_id": "src_vdh",
        "document_id": "doc_src_vdh_001",
        "supporting_chunk_id": "chunk_src_vdh_001",
        "disease": "hantavirus",
        "geographic_scope": "Virginia",
        "date_or_period": "2025-05-01",
        "observation_type": "confirmed_case_record",
        "count_field": "cases_confirmed",
        "count_value": 1,
        "primary_case_dataset_eligible": True,
        "evidence_quote": record["evidence_quote"],
        "requires_human_review": not final_case,
    }
    other_claim = {
        **claim,
        "claim_id": "claim_support_001_cases",
        "source_record_id": "rec_support_001",
        "source_id": "src_support",
        "count_value": 2,
        "evidence_quote": "A secondary source mentioned two possible cases.",
    }
    comparison = {
        "comparison_id": "cmp_001",
        "left_claim_id": claim["claim_id"],
        "right_claim_id": other_claim["claim_id"],
        "disease_comparison": "same_disease",
        "geography_comparison": "same_geography",
        "time_comparison": "same_period",
        "count_comparison": "numeric_conflict",
        "observation_type_comparison": "same_observation_type",
        "source_independence_status": "independent_sources",
        "corroboration_match_status": "conflicts",
        "confidence": "medium",
        "reason": "Counts differ across sources.",
        "human_review_needed": True,
    }
    event = {
        "event_id": "event_001",
        "claim_ids": [claim["claim_id"], other_claim["claim_id"]],
        "supporting_claim_ids": [claim["claim_id"]],
        "conflicting_claim_ids": [other_claim["claim_id"]],
        "record_ids": ["rec_src_vdh_001"],
        "source_ids": ["src_vdh", "src_support"],
        "event_status": "conflicting",
        "primary_case_dataset_eligible": True,
    }
    decisions = [
        {
            "record_id": "rec_src_vdh_001",
            "original_status": "pre_quality_gate",
            "final_inclusion_status": "accepted_final_case_dataset"
            if final_case
            else "quarantined_needs_review",
            "final_dataset_included": final_case,
            "final_case_dataset_included": final_case,
            "primary_case_dataset_eligible": True,
            "observation_type": "confirmed_case_record",
            "quality_gate_reasons": ["accepted"] if final_case else ["single_source_unverified"],
            "dataset_view": "final_case_dataset" if final_case else "quarantined_records",
        },
        {
            "record_id": "rec_context_001",
            "original_status": "pre_quality_gate",
            "final_inclusion_status": "context_records",
            "final_dataset_included": False,
            "final_case_dataset_included": False,
            "primary_case_dataset_eligible": False,
            "observation_type": "background_context",
            "quality_gate_reasons": ["background_context"],
            "dataset_view": "context_records",
        },
    ]
    review_items = [
        {
            "review_item_id": "review_001",
            "review_id": "review_001",
            "priority_rank": 1,
            "priority_level": "P0_critical",
            "issue_category": "claim_corroboration_review",
            "short_title": "Conflicting case count",
            "why_it_matters": "May affect primary case dataset inclusion.",
            "suggested_reviewer_question": "Which count is supported by the source text?",
            "suggested_action": "inspect source text and claims",
            "target_ids": ["rec_src_vdh_001", claim["claim_id"], "cmp_001"],
            "source_url": source["canonical_url"],
            "evidence_quote": record["evidence_quote"],
            "allowed_decision_types": ["accept_record", "quarantine_record"],
            "record_id": "rec_src_vdh_001",
            "claim_ids": [claim["claim_id"]],
            "source_ids": ["src_vdh"],
        }
    ]

    workflow_summaries = {
        "task_intake_summary": {
            "collection_spec": {
                "disease": "hantavirus",
                "geography": "Virginia",
                "start_date": "2025-01-01",
                "end_date": "2026-06-01",
                "user_request": "Collect hantavirus data for Virginia.",
            }
        },
        "source_search_execution_summary": {
            "search_enabled": True,
            "live_search_enabled": True,
            "search_mode": "live",
            "search_provider": "tavily",
            "planned_query_count": 3,
            "executed_query_count": 2,
            "raw_result_count": 8,
            "candidate_from_search_count": 4,
        },
        "iterative_source_discovery_summary": {
            "iterative_source_discovery_enabled": True,
            "search_iteration_count": 1,
            "stop_decision": "stop_sufficient",
            "stop_reason": "Enough candidate sources found.",
        },
        "run_quality_summary": {
            "run_quality_status": "passed" if final_case else "no_primary_case_dataset_records",
            "final_case_dataset_count": 1 if final_case else 0,
            "final_dataset_count": 1 if final_case else 0,
            "context_record_count": 1,
            "quarantined_record_count": 0 if final_case else 1,
            "pending_review_record_count": 0,
            "primary_case_dataset_status": "primary_case_records_present"
            if final_case
            else "no_primary_case_dataset_records",
        },
        "final_dataset_quality_summary": {
            "final_case_dataset_count": 1 if final_case else 0,
            "accepted_record_count": 1 if final_case else 0,
            "quarantined_record_count": 0 if final_case else 1,
            "context_record_count": 1,
        },
        "observation_type_dataset_summary": {
            "dataset_view_counts": {
                "final_case_dataset": 1 if final_case else 0,
                "context_records": 1,
                "quarantined_records": 0 if final_case else 1,
            }
        },
        "corroboration_summary": {
            "claim_count": 2,
            "claim_comparison_count": 1,
            "corroborated_event_count": 1,
        },
        "human_review_application_summary": {
            "decisions_provided_count": 0,
            "decisions_applied_count": 0,
            "decisions_rejected_count": 0,
        },
    }
    run_summary = {
        "session_id": session.name,
        "user_request": "Collect hantavirus data for Virginia.",
        "live_search_enabled": True,
        "live_fetch_enabled": True,
        "all_three_llm_stages_enabled": True,
        "llm_stage_summary": {
            "source_planning": {"enabled": True},
            "source_critic": {"enabled": True},
            "structured_extraction": {"enabled": True},
        },
        "run_quality_status": workflow_summaries["run_quality_summary"]["run_quality_status"],
        "final_case_dataset_count": 1 if final_case else 0,
        "final_dataset_count": 1 if final_case else 0,
        "context_record_count": 1,
        "quarantined_record_count": 0 if final_case else 1,
        "human_review_item_count": len(review_items),
        "artifact_paths": {},
    }
    trace = [
        {"node_name": "task_intake_and_scope_planning", "message": "Task parsed.", "metadata": {"status": "completed"}},
        {"node_name": "source_discovery", "message": "Sources discovered.", "metadata": {"candidate_count": 4}},
        {"node_name": "content_fetch_and_parse", "message": "Documents fetched.", "metadata": {"document_count": 1}},
        {"node_name": "structured_extraction", "message": "Records extracted.", "metadata": {"raw_record_count": 2}},
        {"node_name": "quality_gate_routing", "message": "Quality route selected.", "metadata": {"route_after_node": "human_review"}},
        {"node_name": "human_review", "message": "Review queue prepared.", "metadata": {"review_item_count": 1}},
        {"node_name": "final_data_package_builder", "message": "Package exported.", "metadata": {"final_case_dataset_count": 1 if final_case else 0}},
    ]

    final_case_rows = [record] if final_case else []
    quarantined_rows = [] if final_case else [record]
    final_dataset_rows = [record] if final_case else []
    package = {
        "source_registry": [source, source_support],
        "documents": [document],
        "evidence_chunks": [chunk],
        "raw_records": [record, context_record],
        "validated_records": [record, context_record],
        "normalized_records": [record, context_record],
        "claims": [claim, other_claim],
        "claim_comparisons": [comparison],
        "corroborated_events": [event],
        "record_inclusion_decisions": decisions,
        "final_case_dataset": final_case_rows,
        "final_dataset": final_dataset_rows,
        "final_dataset_pre_quality_gate": [record, context_record],
        "context_records": [context_record],
        "quarantined_records": quarantined_rows,
        "pending_review_records": [],
        "non_primary_observations": [context_record],
        "human_review_items": review_items,
        "collection_trace": trace,
        "workflow_summaries": workflow_summaries,
        **workflow_summaries,
    }

    json_files = {
        session / "workflow_run_summary.json": run_summary,
        collection / "final_package.json": package,
        collection / "source_registry.json": [source, source_support],
        collection / "collection_trace.json": trace,
        collection / "final_case_dataset.json": final_case_rows,
        collection / "final_dataset.json": final_dataset_rows,
        collection / "final_dataset_pre_quality_gate.json": [record, context_record],
        collection / "context_records.json": [context_record],
        collection / "quarantined_records.json": quarantined_rows,
        collection / "pending_review_records.json": [],
        collection / "non_primary_observations.json": [context_record],
        collection / "claims.json": [claim, other_claim],
        collection / "claim_comparisons.json": [comparison],
        collection / "corroborated_events.json": [event],
        collection / "record_inclusion_decisions.json": decisions,
        collection / "source_identity_assessments.json": [source],
        collection / "human_review_items.json": review_items,
        collection / "workflow_summaries.json": workflow_summaries,
        diagnostics / "workflow_summaries.json": workflow_summaries,
        diagnostics / "source_registry.json": [source, source_support],
        diagnostics / "fetch_manifest.json": [document],
        diagnostics / "live_fetch_summary.json": {
            "live_fetch_enabled": True,
            "document_count": 1,
            "documents": [document],
        },
        diagnostics / "document_parse_summary.json": {"document_count": 1},
        diagnostics / "raw_records.json": [record, context_record],
        diagnostics / "validated_records.json": [record, context_record],
        diagnostics / "normalized_records.json": [record, context_record],
        diagnostics / "claims.json": [claim, other_claim],
        diagnostics / "claim_comparisons.json": [comparison],
        diagnostics / "corroborated_events.json": [event],
        diagnostics / "record_inclusion_decisions.json": decisions,
        diagnostics / "final_case_dataset.json": final_case_rows,
        diagnostics / "final_dataset_pre_quality_gate.json": [record, context_record],
        diagnostics / "context_records.json": [context_record],
        diagnostics / "quarantined_records.json": quarantined_rows,
        diagnostics / "pending_review_records.json": [],
        diagnostics / "non_primary_observations.json": [context_record],
        diagnostics / "run_quality_summary.json": workflow_summaries["run_quality_summary"],
        diagnostics / "final_dataset_quality_summary.json": workflow_summaries["final_dataset_quality_summary"],
        diagnostics / "observation_type_dataset_summary.json": workflow_summaries["observation_type_dataset_summary"],
        diagnostics / "source_search_execution_summary.json": workflow_summaries["source_search_execution_summary"],
        diagnostics / "iterative_source_discovery_summary.json": workflow_summaries["iterative_source_discovery_summary"],
        diagnostics / "search_iteration_plans.json": [{"iteration": 1, "queries": ["Virginia hantavirus 2025"]}],
        diagnostics / "search_iteration_observations.json": [{"iteration": 1, "accepted_candidate_count": 2}],
        diagnostics / "search_refinement_decisions.json": [{"iteration": 1, "decision": "stop", "reason": "sufficient"}],
        diagnostics / "iterative_search_queries.json": [{"iteration": 1, "query": "Virginia hantavirus 2025"}],
        diagnostics / "human_review_application_summary.json": workflow_summaries["human_review_application_summary"],
        diagnostics / "human_review_priority_summary.json": {
            "review_item_count": len(review_items),
            "prioritized_review_item_count": len(review_items),
            "priority_level_counts": {"P0_critical": 1},
            "issue_category_counts": {"claim_corroboration_review": 1},
            "top_review_item_ids": ["review_001"],
            "generated_from_artifacts_only": True,
        },
        human_review / "human_review_priority_summary.json": {
            "review_item_count": len(review_items),
            "prioritized_review_item_count": len(review_items),
            "priority_level_counts": {"P0_critical": 1},
            "issue_category_counts": {"claim_corroboration_review": 1},
            "top_review_item_ids": ["review_001"],
            "generated_from_artifacts_only": True,
        },
        human_review / "top_review_items.json": review_items,
        human_review / "review_packet_index.json": {"review_packets": review_items},
        human_review / "review_decision_template.json": {
            "decisions": [{"review_id": "review_001", "apply_decision": False}]
        },
        human_review / "review_decision_prefill.json": {
            "decisions": [{"review_id": "review_001", "apply_decision": False}]
        },
    }
    for path, data in json_files.items():
        _write_json(path, data)
    _write_csv(human_review / "top_review_items.csv", review_items)
    (human_review / "review_action_guide.md").write_text(
        "# Review action guide\n\nGenerated decisions are not auto-applied.\n",
        encoding="utf-8",
    )
    return session


def test_visualization_artifacts_are_generated(tmp_path):
    session = _make_session(tmp_path)

    paths = write_workflow_visualization_artifacts(session)

    assert (session / "workflow_visualization" / "index.html").exists()
    assert (session / "workflow_visualization" / "workflow_visualization_summary.json").exists()
    assert (session / "workflow_visualization" / "workflow_timeline.json").exists()
    assert (session / "workflow_visualization" / "workflow_timeline.html").exists()
    assert (session / "workflow_visualization" / "evidence_flow_graph.json").exists()
    assert (session / "workflow_visualization" / "evidence_flow_graph.html").exists()
    assert (session / "workflow_visualization" / "dataset_decision_flow.json").exists()
    assert (session / "workflow_visualization" / "dataset_decision_flow.html").exists()
    assert "workflow_visualization_index" in paths


def test_workflow_timeline_uses_collection_trace(tmp_path):
    artifacts = load_workflow_visualization_artifacts(_make_session(tmp_path))

    timeline = build_workflow_timeline(artifacts)

    assert len(timeline["items"]) >= len(artifacts["collection_trace"])
    assert timeline["items"][0]["node_name"] == "task_intake_and_scope_planning"
    assert timeline["run_quality_status"] == "passed"


def test_graph_topology_visualization_is_generated(tmp_path):
    artifacts = load_workflow_visualization_artifacts(_make_session(tmp_path))

    topology = build_workflow_graph_topology(artifacts)

    node_ids = {node["id"] for node in topology["nodes"]}
    assert "task_intake_and_scope_planning" in node_ids
    assert "final_data_package_builder" in node_ids
    assert topology["edges"]
    assert topology["graph_name"] == "data collection workflow"


def test_evidence_flow_connects_source_to_record_to_claim_to_dataset_view(tmp_path):
    artifacts = load_workflow_visualization_artifacts(_make_session(tmp_path))

    graph = build_evidence_flow_graph(artifacts)

    node_ids = {node["id"] for node in graph["nodes"]}
    edge_types = {edge["edge_type"] for edge in graph["edges"]}
    assert "source:src_vdh" in node_ids
    assert "document:doc_src_vdh_001" in node_ids
    assert "evidence_chunk:chunk_src_vdh_001" in node_ids
    assert "normalized_record:rec_src_vdh_001" in node_ids
    assert "claim:claim_rec_src_vdh_001_cases" in node_ids
    assert "dataset_view:final_case_dataset" in node_ids
    assert {"fetched_as", "chunked_into", "extracted_into", "produced_claim", "included_in_dataset_view"}.issubset(edge_types)


def test_evidence_flow_handles_empty_final_case_dataset(tmp_path):
    artifacts = load_workflow_visualization_artifacts(_make_session(tmp_path, final_case=False))

    graph = build_evidence_flow_graph(artifacts)
    summary = build_workflow_visualization_summary(artifacts)

    dataset_nodes = [node for node in graph["nodes"] if node["node_type"] == "dataset_view"]
    final_case_node = next(node for node in dataset_nodes if node["id"] == "dataset_view:final_case_dataset")
    assert final_case_node["count_summary"]["record_count"] == 0
    assert "dataset_view:context_records" in {node["id"] for node in dataset_nodes}
    assert "dataset_view:quarantined_records" in {node["id"] for node in dataset_nodes}
    assert summary["final_case_dataset_count"] == 0


def test_claim_comparison_cards_generated(tmp_path):
    artifacts = load_workflow_visualization_artifacts(_make_session(tmp_path))

    cards = build_claim_comparison_cards(artifacts)

    assert cards
    assert cards[0]["corroboration_match_status"] == "conflicts"
    assert cards[0]["group"] == "conflicts"
    assert "truth" not in json.dumps(cards, ensure_ascii=False).lower()


def test_dataset_decision_flow_uses_record_inclusion_decisions(tmp_path):
    artifacts = load_workflow_visualization_artifacts(_make_session(tmp_path, final_case=False))

    flow = build_dataset_decision_flow(artifacts)

    assert flow["dataset_view_counts"]["final_case_dataset"] == 0
    paths = {row["record_id"]: row["dataset_view"] for row in flow["record_decisions"]}
    assert paths["rec_src_vdh_001"] == "quarantined_records"
    assert paths["rec_context_001"] == "context_records"


def test_human_review_workflow_visualization_uses_optimization6_artifacts(tmp_path):
    artifacts = load_workflow_visualization_artifacts(_make_session(tmp_path))

    review = build_human_review_visualization(artifacts)

    assert review["review_item_count"] == 1
    assert review["top_review_items"]
    assert review["decision_template_path"].endswith("human_review/review_decision_template.json")
    assert review["decision_prefill_path"].endswith("human_review/review_decision_prefill.json")
    assert review["decisions_auto_applied"] is False


def test_visualization_summary_flags_artifact_only_behavior(tmp_path):
    artifacts = load_workflow_visualization_artifacts(_make_session(tmp_path))

    summary = build_workflow_visualization_summary(artifacts)

    assert summary["generated_from_artifacts_only"] is True
    assert summary["llm_called_for_visualization"] is False
    assert summary["search_called_for_visualization"] is False
    assert summary["fetch_called_for_visualization"] is False
    assert summary["evidence_flow_node_count"] > 0


def test_html_files_are_self_contained_without_external_cdn(tmp_path):
    session = _make_session(tmp_path)
    write_workflow_visualization_artifacts(session)

    html_files = list((session / "workflow_visualization").glob("*.html"))
    assert html_files
    forbidden = ["<script src=", "<link href=", "https://cdn", "http://cdn", "unpkg", "jsdelivr", "cdnjs", "fonts.googleapis"]
    for path in html_files:
        text = path.read_text(encoding="utf-8").lower()
        assert not any(token in text for token in forbidden), path


def test_runner_style_writer_updates_summary_artifact_paths(tmp_path):
    session = _make_session(tmp_path)

    write_workflow_visualization_artifacts(session)
    summary = json.loads((session / "workflow_run_summary.json").read_text(encoding="utf-8"))

    assert "workflow_visualization_index" in summary["artifact_paths"]
    assert "workflow_visualization_summary" in summary["artifact_paths"]
    assert (session / "diagnostics" / "workflow_visualization_summary.json").exists()


def test_console_links_visualization_artifacts(tmp_path):
    session = _make_session(tmp_path)
    write_workflow_visualization_artifacts(session)

    console_summary = build_console_report(tmp_path / "console", session)
    html = Path(console_summary["html_path"]).read_text(encoding="utf-8")

    assert "workflow_visualization/index.html" in html
    assert "workflow_timeline.html" in html
    assert "evidence_flow_graph.html" in html
    assert "claim_comparison_cards.html" in html
    assert "dataset_decision_flow.html" in html
    assert "human_review_workflow.html" in html
