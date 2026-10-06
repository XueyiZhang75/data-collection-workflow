"""Static workflow visualization artifacts for completed workflow runs.

This module is intentionally artifact-only. It reads a completed session folder
and writes HTML/JSON/Markdown/CSV visualization files without calling LLMs,
search providers, webpage fetchers, or mutating workflow records.
"""

from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .export import write_csv_rows, write_json


VIS_DIR = "workflow_visualization"

GRAPH_NODE_ORDER = [
    "task_intake_and_scope_planning",
    "disease_intelligence_builder",
    "profile_and_schema_setup",
    "executable_source_planning",
    "query_strategy_builder",
    "source_discovery",
    "source_dedup_and_registry",
    "source_screening",
    "source_critic_and_uncertainty_routing",
    "content_fetch_and_parse",
    "document_quality_check",
    "evidence_chunking_and_data_presence_flagging",
    "structured_extraction",
    "schema_validation_and_repair",
    "record_normalization",
    "record_linking",
    "cross_source_consistency_check",
    "quality_gate_routing",
    "human_review",
    "final_data_package_builder",
]

NODE_LABELS = {
    "task_intake_and_scope_planning": "Task intake and scope planning",
    "disease_intelligence_builder": "Disease intelligence builder",
    "profile_and_schema_setup": "Profile and schema setup",
    "executable_source_planning": "Executable source planning",
    "query_strategy_builder": "Query strategy builder",
    "source_discovery": "Source discovery",
    "source_dedup_and_registry": "Source dedup and registry",
    "source_screening": "Source screening",
    "source_critic_and_uncertainty_routing": "Source critic and uncertainty routing",
    "content_fetch_and_parse": "Content fetch and parse",
    "document_quality_check": "Document quality check",
    "evidence_chunking_and_data_presence_flagging": "Evidence chunking and data presence flagging",
    "structured_extraction": "Structured extraction",
    "schema_validation_and_repair": "Schema validation and repair",
    "record_normalization": "Record normalization",
    "record_linking": "Record linking",
    "cross_source_consistency_check": "Cross-source consistency check",
    "quality_gate_routing": "Quality gate routing",
    "human_review": "Human review",
    "final_data_package_builder": "Final data package builder",
}

SUMMARY_KEYS_BY_NODE = {
    "task_intake_and_scope_planning": ("task_intake_summary",),
    "disease_intelligence_builder": ("disease_intelligence_summary",),
    "profile_and_schema_setup": ("profile_schema_summary",),
    "executable_source_planning": (
        "executable_source_plan_summary",
        "source_planning_agent_summary",
        "localized_source_planning_summary",
    ),
    "source_discovery": (
        "source_search_execution_summary",
        "iterative_source_discovery_summary",
        "source_discovery_summary",
    ),
    "source_dedup_and_registry": ("source_registry_summary",),
    "source_screening": (
        "source_screening_summary",
        "source_credibility_summary",
        "source_identity_summary",
        "disease_relevance_summary",
    ),
    "source_critic_and_uncertainty_routing": (
        "source_critic_summary",
        "source_routing_summary",
    ),
    "content_fetch_and_parse": ("content_fetch_summary", "document_parse_summary"),
    "document_quality_check": ("document_quality_summary",),
    "evidence_chunking_and_data_presence_flagging": (
        "evidence_chunking_summary",
        "data_presence_summary",
    ),
    "structured_extraction": ("structured_extraction_summary", "llm_extraction_summary"),
    "schema_validation_and_repair": ("schema_validation_summary",),
    "record_normalization": ("record_normalization_summary",),
    "record_linking": (
        "record_linking_summary",
        "event_clustering_summary",
        "duplicate_detection_summary",
    ),
    "cross_source_consistency_check": (
        "validation_summary",
        "trusted_source_validation_summary",
        "cross_source_validation_summary",
        "corroboration_summary",
        "anomaly_summary",
    ),
    "quality_gate_routing": ("run_quality_summary", "final_dataset_quality_summary"),
    "human_review": ("human_review_summary", "human_review_application_summary"),
    "final_data_package_builder": ("finalization_summary",),
}

DATASET_VIEW_NAMES = [
    "final_case_dataset",
    "final_dataset",
    "final_dataset_pre_quality_gate",
    "final_dataset_post_review",
    "zero_case_statements",
    "exposure_monitoring_records",
    "surveillance_summary_records",
    "outbreak_summary_records",
    "context_records",
    "unclassified_observation_records",
    "non_primary_observations",
    "quarantined_records",
    "pending_review_records",
]


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _first_present(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return default


def _id(row: dict, *fields: str) -> str:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    return ""


def _read_artifact_list(
    session_dir: Path,
    package: dict,
    key: str,
    warnings: list[str],
    *,
    csv_fallback: bool = True,
) -> list[dict]:
    candidates = [
        session_dir / "diagnostics" / f"{key}.json",
        session_dir / "collection" / f"{key}.json",
    ]
    for path in candidates:
        data = _read_json(path, None)
        if isinstance(data, list):
            return data
    package_value = package.get(key)
    if isinstance(package_value, list):
        return package_value
    if csv_fallback:
        for path in (
            session_dir / "collection" / f"{key}.csv",
            session_dir / "human_review" / f"{key}.csv",
        ):
            rows = _read_csv(path)
            if rows:
                return rows
    warnings.append(f"missing optional list artifact: {key}")
    return []


def _read_artifact_dict(
    session_dir: Path,
    package: dict,
    key: str,
    warnings: list[str],
) -> dict:
    for path in (
        session_dir / "diagnostics" / f"{key}.json",
        session_dir / "collection" / f"{key}.json",
    ):
        data = _read_json(path, None)
        if isinstance(data, dict):
            return data
    package_value = package.get(key)
    if isinstance(package_value, dict):
        return package_value
    warnings.append(f"missing optional dict artifact: {key}")
    return {}


def _task_metadata(artifacts: dict) -> dict:
    run_summary = artifacts["workflow_run_summary"]
    workflow_summaries = artifacts["workflow_summaries"]
    task_summary = _as_dict(workflow_summaries.get("task_intake_summary"))
    spec = _as_dict(task_summary.get("collection_spec"))
    structured_task = _as_dict(task_summary.get("structured_task"))
    run_quality = _as_dict(artifacts.get("run_quality_summary"))
    return {
        "session_id": run_summary.get("session_id") or artifacts["session_dir"].name,
        "disease": _first_present(
            run_summary.get("task_disease"),
            run_quality.get("task_disease"),
            spec.get("disease"),
            structured_task.get("disease"),
        ),
        "location": _first_present(
            run_summary.get("task_location"),
            run_quality.get("task_location"),
            spec.get("geography"),
            spec.get("location"),
            structured_task.get("location"),
        ),
        "start_date": _first_present(
            run_summary.get("task_start_date"),
            run_quality.get("task_start_date"),
            spec.get("start_date"),
            structured_task.get("start_date"),
        ),
        "end_date": _first_present(
            run_summary.get("task_end_date"),
            run_quality.get("task_end_date"),
            spec.get("end_date"),
            structured_task.get("end_date"),
        ),
        "user_request": _first_present(
            run_summary.get("user_request"),
            spec.get("user_request"),
            structured_task.get("user_request"),
        ),
    }


def load_workflow_visualization_artifacts(session_dir: Path | str) -> dict:
    """Load existing artifacts from a completed session directory."""

    session_dir = Path(session_dir)
    warnings: list[str] = []
    run_summary = _as_dict(_read_json(session_dir / "workflow_run_summary.json", {}))
    if not run_summary:
        warnings.append("workflow_run_summary.json missing or unreadable")
    package = _as_dict(_read_json(session_dir / "collection" / "final_package.json", {}))
    workflow_summaries = _read_artifact_dict(session_dir, package, "workflow_summaries", warnings)
    if not workflow_summaries:
        workflow_summaries = _as_dict(package.get("workflow_summaries"))

    artifacts: dict[str, Any] = {
        "session_dir": session_dir,
        "warnings": warnings,
        "workflow_run_summary": run_summary,
        "final_package": package,
        "workflow_summaries": workflow_summaries,
    }
    list_keys = [
        "source_registry",
        "source_identity_assessments",
        "fetch_manifest",
        "documents",
        "evidence_chunks",
        "raw_records",
        "validated_records",
        "normalized_records",
        "claims",
        "claim_comparisons",
        "corroborated_events",
        "record_inclusion_decisions",
        *DATASET_VIEW_NAMES,
        "human_review_items",
        "anomaly_results",
        "validation_results",
        "applied_human_review_decisions",
        "rejected_human_review_decisions",
        "human_review_audit_trail",
        "search_iteration_plans",
        "search_iteration_observations",
        "search_refinement_decisions",
        "iterative_search_queries",
        "source_search_results",
    ]
    for key in list_keys:
        artifacts[key] = _read_artifact_list(session_dir, package, key, warnings)

    dict_keys = [
        "run_quality_summary",
        "final_dataset_quality_summary",
        "observation_type_dataset_summary",
        "source_search_execution_summary",
        "executable_source_plan_summary",
        "source_planning_agent_summary",
        "localized_source_planning_summary",
        "iterative_source_discovery_summary",
        "source_discovery_summary",
        "source_critic_summary",
        "corroboration_summary",
        "source_identity_summary",
        "human_review_application_summary",
        "content_fetch_summary",
        "document_parse_summary",
        "human_review_priority_summary",
    ]
    for key in dict_keys:
        artifacts[key] = _read_artifact_dict(session_dir, package, key, warnings)

    live_fetch = _as_dict(_read_json(session_dir / "diagnostics" / "live_fetch_summary.json", {}))
    if live_fetch:
        artifacts["live_fetch_summary"] = live_fetch
        docs = live_fetch.get("documents")
        if docs and not artifacts.get("documents"):
            artifacts["documents"] = list(docs)
    else:
        artifacts["live_fetch_summary"] = {}

    human_dir = session_dir / "human_review"
    artifacts["human_review_priority_summary"] = _as_dict(
        _read_json(human_dir / "human_review_priority_summary.json", {})
        or artifacts.get("human_review_priority_summary")
    )
    top_items = _read_json(human_dir / "top_review_items.json", None)
    if isinstance(top_items, list):
        artifacts["top_review_items"] = top_items
    else:
        artifacts["top_review_items"] = _read_csv(human_dir / "top_review_items.csv")
    artifacts["review_packet_index"] = _as_dict(
        _read_json(human_dir / "review_packet_index.json", {})
    )
    artifacts["review_decision_template"] = _as_dict(
        _read_json(human_dir / "review_decision_template.json", {})
    )
    artifacts["review_decision_prefill"] = _as_dict(
        _read_json(human_dir / "review_decision_prefill.json", {})
    )
    artifacts["review_action_guide_path"] = (
        "human_review/review_action_guide.md"
        if (human_dir / "review_action_guide.md").exists()
        else None
    )

    trace = _read_artifact_list(session_dir, package, "collection_trace", warnings)
    artifacts["collection_trace"] = trace
    artifacts["task"] = _task_metadata(artifacts)
    return artifacts


def _node_summary_counts(node_name: str, artifacts: dict) -> dict:
    counts = {}
    for key in SUMMARY_KEYS_BY_NODE.get(node_name, ()):
        value = _as_dict(artifacts.get(key)) or _as_dict(
            artifacts.get("workflow_summaries", {}).get(key)
        )
        for sub_key, sub_value in value.items():
            if isinstance(sub_value, (int, float, bool)) and len(counts) < 12:
                counts[f"{key}.{sub_key}"] = sub_value
    return counts


def build_workflow_timeline(artifacts: dict) -> dict:
    trace = [row for row in _as_list(artifacts.get("collection_trace")) if isinstance(row, dict)]
    trace_by_node: dict[str, list[dict]] = defaultdict(list)
    for row in trace:
        trace_by_node[_clean(row.get("node_name"))].append(row)

    items = []
    nodes = list(dict.fromkeys([*GRAPH_NODE_ORDER, *[row.get("node_name") for row in trace if row.get("node_name")]]))
    for idx, node_name in enumerate(nodes, start=1):
        trace_events = trace_by_node.get(node_name, [])
        last_trace = trace_events[-1] if trace_events else {}
        metadata = _as_dict(last_trace.get("metadata"))
        executed = bool(trace_events)
        items.append(
            {
                "step_index": idx,
                "node_name": node_name,
                "node_label": NODE_LABELS.get(node_name, node_name.replace("_", " ").title()),
                "status": "executed" if executed else "not_observed_in_trace",
                "route_after_node": metadata.get("route_after_node")
                or metadata.get("current_route")
                or artifacts["workflow_run_summary"].get("current_route")
                if node_name == "quality_gate_routing"
                else metadata.get("route_after_node"),
                "input_count_summary": metadata.get("input_count_summary") or {},
                "output_count_summary": metadata.get("output_count_summary") or {},
                "important_counts": _node_summary_counts(node_name, artifacts),
                "key_decisions": metadata.get("key_decisions") or metadata.get("decision") or [],
                "warnings": _as_list(metadata.get("warnings")),
                "artifact_links": _artifact_links_for_node(node_name),
                "user_facing_explanation": _node_explanation(node_name, artifacts),
                "trace_message": last_trace.get("message"),
            }
        )
    run_quality = _as_dict(artifacts.get("run_quality_summary"))
    run_summary = _as_dict(artifacts.get("workflow_run_summary"))
    return {
        "timeline_type": "workflow_timeline",
        "session_id": artifacts["task"].get("session_id"),
        "items": items,
        "technical_workflow_completed": bool(trace),
        "ended_in_human_review": run_summary.get("current_route") == "human_review"
        or bool(artifacts.get("human_review_items")),
        "final_case_dataset_count": _count_dataset(artifacts, "final_case_dataset"),
        "final_dataset_count": _count_dataset(artifacts, "final_dataset"),
        "quarantined_record_count": _count_dataset(artifacts, "quarantined_records"),
        "human_review_item_count": len(_as_list(artifacts.get("human_review_items"))),
        "run_quality_status": run_quality.get("run_quality_status")
        or run_summary.get("run_quality_status"),
        "warnings": list(artifacts.get("warnings") or []),
        "generated_from_artifacts_only": True,
    }


def _artifact_links_for_node(node_name: str) -> list[str]:
    links = {
        "source_discovery": ["diagnostics/source_search_execution_summary.json"],
        "source_screening": ["collection/source_registry.json", "diagnostics/source_identity_summary.json"],
        "content_fetch_and_parse": ["diagnostics/live_fetch_summary.json", "diagnostics/fetch_manifest.json"],
        "structured_extraction": ["diagnostics/structured_extraction_summary.json", "diagnostics/raw_records.json"],
        "record_normalization": ["diagnostics/normalized_records.json"],
        "cross_source_consistency_check": ["diagnostics/claim_comparisons.json", "diagnostics/corroborated_events.json"],
        "quality_gate_routing": ["diagnostics/run_quality_summary.json", "diagnostics/final_dataset_quality_summary.json"],
        "human_review": ["human_review/human_review_priority_summary.json", "human_review/top_review_items.csv"],
        "final_data_package_builder": ["collection/final_package.json"],
    }
    return links.get(node_name, [])


def _node_explanation(node_name: str, artifacts: dict) -> str:
    if node_name == "quality_gate_routing" and _count_dataset(artifacts, "final_case_dataset") == 0:
        return "The run completed technically, but no accepted primary case records reached final_case_dataset."
    return {
        "task_intake_and_scope_planning": "Turns the user request into structured task scope.",
        "source_discovery": "Executes configured source discovery and records search-derived candidates.",
        "content_fetch_and_parse": "Fetches and parses allowed source documents.",
        "structured_extraction": "Extracts structured public-health records from evidence chunks.",
        "cross_source_consistency_check": "Compares claims across sources without deciding official truth.",
        "human_review": "Packages uncertain or high-impact evidence for manual review.",
        "final_data_package_builder": "Exports auditable datasets, diagnostics, reports, and review artifacts.",
    }.get(node_name, "Workflow node state and summaries are reconstructed from completed run artifacts.")


def build_workflow_graph_topology(artifacts: dict) -> dict:
    trace_nodes = {row.get("node_name") for row in _as_list(artifacts.get("collection_trace"))}
    edges = [
        ("START", "task_intake_and_scope_planning", None, None),
        ("task_intake_and_scope_planning", "disease_intelligence_builder", None, None),
        ("disease_intelligence_builder", "profile_and_schema_setup", None, None),
        ("profile_and_schema_setup", "executable_source_planning", None, None),
        ("executable_source_planning", "query_strategy_builder", None, None),
        ("query_strategy_builder", "source_discovery", None, None),
        ("source_discovery", "source_dedup_and_registry", None, None),
        ("source_dedup_and_registry", "source_screening", None, None),
        ("source_screening", "source_critic_and_uncertainty_routing", None, None),
        ("source_critic_and_uncertainty_routing", "content_fetch_and_parse", None, None),
        ("content_fetch_and_parse", "document_quality_check", None, None),
        ("document_quality_check", "evidence_chunking_and_data_presence_flagging", None, None),
        ("evidence_chunking_and_data_presence_flagging", "structured_extraction", None, None),
        ("structured_extraction", "schema_validation_and_repair", None, None),
        ("schema_validation_and_repair", "record_normalization", None, None),
        ("record_normalization", "record_linking", None, None),
        ("record_linking", "cross_source_consistency_check", None, None),
        ("cross_source_consistency_check", "quality_gate_routing", None, None),
        ("quality_gate_routing", "human_review", "current_route == human_review", "needs human review"),
        ("quality_gate_routing", "final_data_package_builder", "current_route != human_review", "finalize"),
        ("human_review", "final_data_package_builder", None, None),
        ("final_data_package_builder", "END", None, None),
    ]
    node_ids = ["START", *GRAPH_NODE_ORDER, "END"]
    nodes = [
        {
            "id": node_id,
            "label": NODE_LABELS.get(node_id, node_id),
            "description": _node_explanation(node_id, artifacts),
            "node_type": "terminal" if node_id in {"START", "END"} else "workflow_node",
            "executed": node_id in trace_nodes or node_id in {"START", "END"},
            "status": "executed" if node_id in trace_nodes else "structural",
            "summary_counts": _node_summary_counts(node_id, artifacts),
            "artifact_links": _artifact_links_for_node(node_id),
        }
        for node_id in node_ids
    ]
    return {
        "graph_name": "data collection workflow",
        "source": "static graph order mirrored from src/data_collection_workflow/graph.py",
        "nodes": nodes,
        "edges": [
            {
                "source": source,
                "target": target,
                "condition": condition,
                "route_label": route_label,
            }
            for source, target, condition, route_label in edges
        ],
        "warnings": [
            "topology is exported from documented graph.py node sequence; no graph topology is mutated."
        ],
    }


def build_agentic_search_timeline(artifacts: dict) -> dict:
    source_search = _as_dict(artifacts.get("source_search_execution_summary"))
    iterative = _as_dict(artifacts.get("iterative_source_discovery_summary"))
    plans = _as_list(artifacts.get("search_iteration_plans"))
    observations = _as_list(artifacts.get("search_iteration_observations"))
    decisions = _as_list(artifacts.get("search_refinement_decisions"))
    queries = _as_list(artifacts.get("iterative_search_queries"))
    cards = [
        {
            "card_type": "PLAN",
            "title": "Search plan",
            "summary": {
                "planned_query_count": source_search.get("planned_query_count"),
                "iterative_search_enabled": iterative.get("iterative_source_discovery_enabled"),
            },
        }
    ]
    if iterative.get("iterative_source_discovery_enabled"):
        max_len = max(len(plans), len(observations), len(decisions), 1)
        for idx in range(max_len):
            iteration = idx + 1
            cards.append(
                {
                    "card_type": f"SEARCH iteration {iteration}",
                    "queries": [
                        row
                        for row in queries
                        if str(row.get("iteration") or row.get("iteration_index") or iteration)
                        == str(iteration)
                    ],
                    "plan": plans[idx] if idx < len(plans) else {},
                }
            )
            cards.append(
                {
                    "card_type": f"OBSERVE iteration {iteration}",
                    "observation": observations[idx] if idx < len(observations) else {},
                }
            )
            cards.append(
                {
                    "card_type": f"REFINE decision {iteration}",
                    "decision": decisions[idx] if idx < len(decisions) else {},
                }
            )
    else:
        cards.append(
            {
                "card_type": "SEARCH one-shot",
                "summary": source_search,
            }
        )
    cards.append(
        {
            "card_type": "STOP",
            "stop_decision": iterative.get("stop_decision"),
            "stop_reason": iterative.get("stop_reason"),
        }
    )
    return {
        "timeline_type": "agentic_search_timeline",
        "planned_query_count": source_search.get("planned_query_count"),
        "iterative_search_enabled": bool(iterative.get("iterative_source_discovery_enabled")),
        "iteration_count": iterative.get("search_iteration_count", len(plans)),
        "executed_query_count": source_search.get("executed_query_count"),
        "raw_result_count": source_search.get("raw_result_count"),
        "search_derived_candidate_count": source_search.get("candidate_from_search_count")
        or source_search.get("search_derived_candidate_count"),
        "stop_decision": iterative.get("stop_decision"),
        "stop_reason": iterative.get("stop_reason"),
        "cards": cards,
        "final_gap_assessment": iterative.get("final_gap_assessment")
        or iterative.get("gap_assessment"),
        "generated_from_artifacts_only": True,
    }


def _source_map(artifacts: dict) -> dict[str, dict]:
    return {
        _id(row, "source_id"): row
        for row in _as_list(artifacts.get("source_registry"))
        if isinstance(row, dict) and _id(row, "source_id")
    }


def _records_by_id(artifacts: dict) -> dict[str, dict]:
    rows = []
    for key in ("raw_records", "validated_records", "normalized_records", *DATASET_VIEW_NAMES):
        rows.extend([row for row in _as_list(artifacts.get(key)) if isinstance(row, dict)])
    return {_id(row, "record_id", "source_record_id"): row for row in rows if _id(row, "record_id", "source_record_id")}


def _append_node(nodes: dict[str, dict], node: dict) -> None:
    nodes.setdefault(node["id"], node)


def _append_edge(edges: list[dict], edge: dict) -> None:
    key = (edge["source"], edge["target"], edge["edge_type"])
    if not any((row["source"], row["target"], row["edge_type"]) == key for row in edges):
        edges.append(edge)


def _node_label(row: dict, *fields: str, fallback: str) -> str:
    return _clean(_first_present(*[row.get(field) for field in fields], default=fallback))


def build_evidence_flow_graph(artifacts: dict) -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    warnings: list[str] = []
    sources = _source_map(artifacts)

    for source_id, source in sources.items():
        _append_node(
            nodes,
            {
                "id": f"source:{source_id}",
                "node_type": "source",
                "label": source_id,
                "title": _node_label(source, "title", "actual_publisher", "publisher", fallback=source_id),
                "status": source.get("source_role_final") or source.get("status"),
                "source_id": source_id,
                "source_url": source.get("canonical_url") or source.get("url"),
                "actual_publisher": source.get("actual_publisher") or source.get("publisher"),
                "source_type_final": source.get("source_type_final") or source.get("source_type"),
                "credibility_level": source.get("credibility_level") or source.get("credibility_level_llm"),
                "count_summary": {},
                "warnings": _as_list(source.get("source_identity_warnings")),
            },
        )

    documents = []
    documents.extend(_as_list(artifacts.get("documents")))
    documents.extend(_as_list(artifacts.get("fetch_manifest")))
    seen_doc_ids = set()
    for idx, doc in enumerate([row for row in documents if isinstance(row, dict)], start=1):
        doc_id = _id(doc, "document_id", "doc_id") or f"doc_{idx:03d}"
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        source_id = _id(doc, "source_id")
        _append_node(
            nodes,
            {
                "id": f"document:{doc_id}",
                "node_type": "document",
                "label": doc_id,
                "title": _node_label(doc, "title", "source_title", "url", fallback=doc_id),
                "status": doc.get("parse_status") or doc.get("fetch_status"),
                "source_id": source_id or None,
                "document_id": doc_id,
                "source_url": doc.get("source_url") or doc.get("url") or doc.get("final_url"),
                "count_summary": {},
                "warnings": [],
            },
        )
        if source_id and f"source:{source_id}" in nodes:
            _append_edge(
                edges,
                {
                    "source": f"source:{source_id}",
                    "target": f"document:{doc_id}",
                    "edge_type": "fetched_as",
                    "label": "fetched as document",
                    "reason": "document.source_id links to source",
                    "warnings": [],
                },
            )
        elif source_id:
            warnings.append(f"document {doc_id} references missing source_id {source_id}")

    for idx, chunk in enumerate(_as_list(artifacts.get("evidence_chunks")), start=1):
        if not isinstance(chunk, dict):
            continue
        chunk_id = _id(chunk, "chunk_id", "evidence_chunk_id") or f"chunk_{idx:03d}"
        doc_id = _id(chunk, "document_id")
        source_id = _id(chunk, "source_id")
        _append_node(
            nodes,
            {
                "id": f"evidence_chunk:{chunk_id}",
                "node_type": "evidence_chunk",
                "label": chunk_id,
                "title": _clean(chunk.get("title") or chunk.get("chunk_id") or chunk_id),
                "status": chunk.get("confidence"),
                "source_id": source_id or None,
                "document_id": doc_id or None,
                "chunk_id": chunk_id,
                "evidence_quote_excerpt": _clean(chunk.get("text"))[:220],
                "count_summary": {"data_types": chunk.get("data_types")},
                "warnings": [],
            },
        )
        if doc_id and f"document:{doc_id}" in nodes:
            _append_edge(
                edges,
                {
                    "source": f"document:{doc_id}",
                    "target": f"evidence_chunk:{chunk_id}",
                    "edge_type": "chunked_into",
                    "label": "chunked into evidence",
                    "reason": "chunk.document_id links to document",
                    "warnings": [],
                },
            )
        elif doc_id:
            warnings.append(f"chunk {chunk_id} references missing document_id {doc_id}")

    record_nodes_by_kind: dict[tuple[str, str], str] = {}
    for record_key, node_type, input_key in (
        ("raw_record", "raw_record", "raw_records"),
        ("validated_record", "validated_record", "validated_records"),
        ("normalized_record", "normalized_record", "normalized_records"),
    ):
        for idx, record in enumerate(_as_list(artifacts.get(input_key)), start=1):
            if not isinstance(record, dict):
                continue
            record_id = _id(record, "record_id", "source_record_id") or f"{record_key}_{idx:03d}"
            node_id = f"{node_type}:{record_id}"
            record_nodes_by_kind[(node_type, record_id)] = node_id
            source_id = _id(record, "source_id")
            chunk_id = _id(record, "supporting_chunk_id", "evidence_chunk_id")
            _append_node(nodes, _record_node(node_id, node_type, record_id, record))
            if chunk_id and f"evidence_chunk:{chunk_id}" in nodes:
                _append_edge(
                    edges,
                    {
                        "source": f"evidence_chunk:{chunk_id}",
                        "target": node_id,
                        "edge_type": "extracted_into",
                        "label": "extracted into record",
                        "reason": "record.supporting_chunk_id links to evidence chunk",
                        "warnings": [],
                    },
                )
            elif source_id and f"source:{source_id}" in nodes:
                _append_edge(
                    edges,
                    {
                        "source": f"source:{source_id}",
                        "target": node_id,
                        "edge_type": "extracted_into",
                        "label": "record provenance source",
                        "reason": "chunk/document id unavailable; record.source_id links to source",
                        "warnings": ["record lacked linkable supporting_chunk_id"],
                    },
                )
                warnings.append(f"record {record_id} lacked linkable supporting_chunk_id")

    for record_id in {rid for _, rid in record_nodes_by_kind}:
        raw = record_nodes_by_kind.get(("raw_record", record_id))
        validated = record_nodes_by_kind.get(("validated_record", record_id))
        normalized = record_nodes_by_kind.get(("normalized_record", record_id))
        if raw and validated:
            _append_edge(edges, _simple_edge(raw, validated, "validated_into", "validated into"))
        if validated and normalized:
            _append_edge(edges, _simple_edge(validated, normalized, "normalized_into", "normalized into"))

    claims = [row for row in _as_list(artifacts.get("claims")) if isinstance(row, dict)]
    claims_by_id = {}
    for idx, claim in enumerate(claims, start=1):
        claim_id = _id(claim, "claim_id") or f"claim_{idx:03d}"
        claims_by_id[claim_id] = claim
        record_id = _id(claim, "source_record_id", "record_id")
        _append_node(
            nodes,
            {
                "id": f"claim:{claim_id}",
                "node_type": "claim",
                "label": claim_id,
                "title": _clean(
                    f"{claim.get('observation_type') or claim.get('claim_type') or 'claim'} / {claim.get('count_value', '')}"
                ),
                "status": claim.get("claim_status"),
                "claim_id": claim_id,
                "record_id": record_id or None,
                "source_id": claim.get("source_id"),
                "disease": claim.get("disease"),
                "location": claim.get("geographic_scope") or claim.get("subnational_location"),
                "date_or_period": claim.get("date_or_period"),
                "observation_type": claim.get("observation_type"),
                "evidence_quote_excerpt": _clean(claim.get("evidence_quote"))[:220],
                "count_summary": {
                    "count_field": claim.get("count_field"),
                    "count_value": claim.get("count_value"),
                },
                "warnings": [],
            },
        )
        record_node = f"normalized_record:{record_id}"
        if record_id and record_node in nodes:
            _append_edge(edges, _simple_edge(record_node, f"claim:{claim_id}", "produced_claim", "produced claim"))
        elif record_id:
            warnings.append(f"claim {claim_id} references missing normalized record {record_id}")

    comparisons = [row for row in _as_list(artifacts.get("claim_comparisons")) if isinstance(row, dict)]
    for idx, comparison in enumerate(comparisons, start=1):
        comparison_id = _id(comparison, "comparison_id") or f"claim_comparison_{idx:03d}"
        _append_node(
            nodes,
            {
                "id": f"claim_comparison:{comparison_id}",
                "node_type": "claim_comparison",
                "label": comparison_id,
                "title": comparison.get("corroboration_match_status")
                or comparison.get("comparability_status"),
                "status": comparison.get("corroboration_match_status"),
                "count_summary": {},
                "warnings": [],
            },
        )
        for claim_id in (_id(comparison, "left_claim_id"), _id(comparison, "right_claim_id")):
            if claim_id and f"claim:{claim_id}" in nodes:
                _append_edge(edges, _simple_edge(f"claim:{claim_id}", f"claim_comparison:{comparison_id}", "compared_with", "compared with"))
            elif claim_id:
                warnings.append(f"comparison {comparison_id} references missing claim {claim_id}")

    for idx, event in enumerate(_as_list(artifacts.get("corroborated_events")), start=1):
        if not isinstance(event, dict):
            continue
        event_id = _id(event, "event_id", "linked_event_id") or f"event_{idx:03d}"
        _append_node(
            nodes,
            {
                "id": f"corroborated_event:{event_id}",
                "node_type": "corroborated_event",
                "label": event_id,
                "title": event.get("event_status") or event_id,
                "status": event.get("event_status"),
                "event_id": event_id,
                "count_summary": {
                    "claim_count": len(_as_list(event.get("claim_ids")) or _as_list(event.get("supporting_claim_ids"))),
                    "record_count": len(_as_list(event.get("record_ids"))),
                },
                "warnings": [],
            },
        )
        for claim_id in set(
            _as_list(event.get("claim_ids"))
            + _as_list(event.get("supporting_claim_ids"))
            + _as_list(event.get("conflicting_claim_ids"))
            + _as_list(event.get("unverified_claim_ids"))
        ):
            if claim_id and f"claim:{claim_id}" in nodes:
                _append_edge(edges, _simple_edge(f"claim:{claim_id}", f"corroborated_event:{event_id}", "grouped_into_event", "grouped into event"))

    for view_name in DATASET_VIEW_NAMES:
        rows = [row for row in _as_list(artifacts.get(view_name)) if isinstance(row, dict)]
        _append_node(
            nodes,
            {
                "id": f"dataset_view:{view_name}",
                "node_type": "dataset_view",
                "label": view_name,
                "title": view_name,
                "status": "empty" if not rows else "populated",
                "dataset_view": view_name,
                "count_summary": {"record_count": len(rows)},
                "warnings": [],
            },
        )
        for row in rows:
            record_id = _id(row, "record_id", "source_record_id")
            record_node = f"normalized_record:{record_id}"
            if record_id and record_node in nodes:
                edge_type = {
                    "quarantined_records": "quarantined_as",
                    "pending_review_records": "pending_review_as",
                    "context_records": "context_for",
                }.get(view_name, "included_in_dataset_view")
                _append_edge(
                    edges,
                    {
                        "source": record_node,
                        "target": f"dataset_view:{view_name}",
                        "edge_type": edge_type,
                        "label": f"included in {view_name}",
                        "reason": f"record appears in {view_name}",
                        "warnings": [],
                    },
                )

    review_items = _as_list(artifacts.get("top_review_items")) or _as_list(
        artifacts.get("human_review_items")
    )
    for idx, item in enumerate([row for row in review_items if isinstance(row, dict)], start=1):
        review_id = _id(item, "review_item_id", "review_id") or f"review_item_{idx:03d}"
        _append_node(
            nodes,
            {
                "id": f"human_review_item:{review_id}",
                "node_type": "human_review_item",
                "label": review_id,
                "title": item.get("short_title") or item.get("reason") or review_id,
                "status": item.get("priority_level") or item.get("status"),
                "source_url": item.get("source_url"),
                "evidence_quote_excerpt": _clean(item.get("evidence_quote"))[:220],
                "count_summary": {},
                "warnings": [],
            },
        )
        for target in set(
            _as_list(item.get("target_ids"))
            + _as_list(item.get("related_ids"))
            + _as_list(item.get("claim_ids"))
            + _as_list(item.get("record_ids"))
        ):
            target_node = _review_target_node(str(target), nodes)
            if target_node:
                _append_edge(edges, _simple_edge(target_node, f"human_review_item:{review_id}", "requires_review", "requires review"))

    return {
        "graph_type": "evidence_flow_graph",
        "nodes": list(nodes.values()),
        "edges": edges,
        "warnings": [*warnings, *list(artifacts.get("warnings") or [])],
        "generated_from_artifacts_only": True,
    }


def _record_node(node_id: str, node_type: str, record_id: str, record: dict) -> dict:
    return {
        "id": node_id,
        "node_type": node_type,
        "label": record_id,
        "title": record.get("source_title") or record.get("disease") or record_id,
        "status": record.get("record_final_inclusion_status") or record.get("validation_status"),
        "record_id": record_id,
        "source_id": record.get("source_id"),
        "source_url": record.get("source_url"),
        "document_id": record.get("document_id"),
        "chunk_id": record.get("supporting_chunk_id"),
        "disease": record.get("disease"),
        "location": record.get("subnational_location") or record.get("country"),
        "date_or_period": record.get("date_reported") or record.get("date_or_period"),
        "observation_type": record.get("observation_type"),
        "dataset_view": record.get("dataset_view"),
        "evidence_quote_excerpt": _clean(record.get("evidence_quote"))[:220],
        "count_summary": {
            "cases_confirmed": record.get("cases_confirmed"),
            "cases_unspecified": record.get("cases_unspecified"),
            "deaths": record.get("deaths"),
            "hospitalizations": record.get("hospitalizations"),
        },
        "warnings": _as_list(record.get("warnings")),
    }


def _simple_edge(source: str, target: str, edge_type: str, label: str) -> dict:
    return {
        "source": source,
        "target": target,
        "edge_type": edge_type,
        "label": label,
        "reason": "matched provenance identifier",
        "warnings": [],
    }


def _review_target_node(target: str, nodes: dict[str, dict]) -> str | None:
    candidates = [
        f"normalized_record:{target}",
        f"raw_record:{target}",
        f"validated_record:{target}",
        f"claim:{target}",
        f"claim_comparison:{target}",
        f"corroborated_event:{target}",
        f"source:{target}",
    ]
    return next((candidate for candidate in candidates if candidate in nodes), None)


def build_claim_comparison_cards(artifacts: dict) -> list[dict]:
    claims = {
        _id(row, "claim_id"): row
        for row in _as_list(artifacts.get("claims"))
        if isinstance(row, dict) and _id(row, "claim_id")
    }
    sources = _source_map(artifacts)
    cards = []
    for idx, comparison in enumerate(_as_list(artifacts.get("claim_comparisons")), start=1):
        if not isinstance(comparison, dict):
            continue
        left_id = _id(comparison, "left_claim_id")
        right_id = _id(comparison, "right_claim_id")
        left = claims.get(left_id, {})
        right = claims.get(right_id, {})
        status = comparison.get("corroboration_match_status") or comparison.get("match_status") or "insufficient_information"
        group = _comparison_group(str(status))
        left_source = sources.get(_id(left, "source_id"), {})
        right_source = sources.get(_id(right, "source_id"), {})
        cards.append(
            {
                "comparison_id": _id(comparison, "comparison_id") or f"comparison_{idx:03d}",
                "left_claim_id": left_id,
                "right_claim_id": right_id,
                "left_source": left.get("source_id"),
                "right_source": right.get("source_id"),
                "left_publisher": left_source.get("actual_publisher") or left_source.get("publisher"),
                "right_publisher": right_source.get("actual_publisher") or right_source.get("publisher"),
                "disease_comparison": comparison.get("disease_comparison"),
                "geography_comparison": comparison.get("geography_comparison"),
                "time_comparison": comparison.get("time_comparison"),
                "count_comparison": comparison.get("count_comparison"),
                "observation_type_comparison": comparison.get("observation_type_comparison"),
                "source_independence_status": comparison.get("source_independence_status"),
                "corroboration_match_status": status,
                "group": group,
                "confidence": comparison.get("confidence"),
                "reason": comparison.get("reason"),
                "human_review_needed": bool(
                    comparison.get("human_review_needed")
                    or comparison.get("requires_human_review")
                ),
                "evidence_quotes": [
                    left.get("evidence_quote"),
                    right.get("evidence_quote"),
                ],
            }
        )
    return cards


def _comparison_group(status: str) -> str:
    text = status.lower()
    if "conflict" in text:
        return "conflicts"
    if "partial" in text:
        return "partially_supports"
    if "corroborat" in text or "support" in text or text == "match":
        return "corroborates"
    if "duplicate" in text:
        return "duplicate_same_source"
    if "single" in text:
        return "single_source_unverified"
    if "not_comparable" in text or "not comparable" in text:
        return "not_comparable"
    if "review" in text:
        return "needs_human_review"
    return "insufficient_information"


def build_dataset_decision_flow(artifacts: dict) -> dict:
    decisions = [
        row
        for row in _as_list(artifacts.get("record_inclusion_decisions"))
        if isinstance(row, dict)
    ]
    records = _records_by_id(artifacts)
    if not decisions:
        for view_name in DATASET_VIEW_NAMES:
            for row in _as_list(artifacts.get(view_name)):
                if not isinstance(row, dict):
                    continue
                record_id = _id(row, "record_id", "source_record_id")
                decisions.append({"record_id": record_id, "dataset_view": view_name})
    rows = []
    for decision in decisions:
        record_id = _id(decision, "record_id", "source_record_id")
        record = records.get(record_id, {})
        dataset_view = decision.get("dataset_view") or _infer_dataset_view(record_id, artifacts)
        rows.append(
            {
                "record_id": record_id,
                "original_status": decision.get("original_status"),
                "final_inclusion_status": decision.get("final_inclusion_status")
                or decision.get("record_final_inclusion_status")
                or record.get("record_final_inclusion_status"),
                "final_dataset_included": _boolish(
                    decision.get("final_dataset_included"),
                    record_id in _record_ids(artifacts.get("final_dataset")),
                ),
                "final_case_dataset_included": _boolish(
                    decision.get("final_case_dataset_included"),
                    record_id in _record_ids(artifacts.get("final_case_dataset")),
                ),
                "primary_case_dataset_eligible": _boolish(
                    decision.get("primary_case_dataset_eligible"),
                    record.get("primary_case_dataset_eligible"),
                ),
                "observation_type": decision.get("observation_type") or record.get("observation_type"),
                "quality_gate_reasons": decision.get("quality_gate_reasons")
                or record.get("quality_gate_reasons")
                or decision.get("reason"),
                "blocking_flags": decision.get("blocking_flags") or record.get("blocking_flags"),
                "quarantine_reason": decision.get("quarantine_reason") or record.get("quarantine_reason"),
                "anomaly_flags": decision.get("anomaly_flags"),
                "validation_flags": decision.get("validation_flags"),
                "human_review_reason": decision.get("human_review_reason")
                or record.get("human_review_reason"),
                "dataset_view": dataset_view,
                "recommended_reviewer_action": decision.get("recommended_reviewer_action"),
            }
        )
    dataset_view_counts = {view: _count_dataset(artifacts, view) for view in DATASET_VIEW_NAMES}
    return {
        "flow_type": "dataset_decision_flow",
        "record_decisions": rows,
        "dataset_view_counts": dataset_view_counts,
        "run_quality_summary": artifacts.get("run_quality_summary") or {},
        "final_dataset_quality_summary": artifacts.get("final_dataset_quality_summary") or {},
        "observation_type_dataset_summary": artifacts.get("observation_type_dataset_summary") or {},
        "final_case_dataset_empty": dataset_view_counts["final_case_dataset"] == 0,
        "warnings": list(artifacts.get("warnings") or []),
        "generated_from_artifacts_only": True,
    }


def _record_ids(rows: Any) -> set[str]:
    return {
        _id(row, "record_id", "source_record_id")
        for row in _as_list(rows)
        if isinstance(row, dict) and _id(row, "record_id", "source_record_id")
    }


def _infer_dataset_view(record_id: str, artifacts: dict) -> str | None:
    for view_name in DATASET_VIEW_NAMES:
        if record_id in _record_ids(artifacts.get(view_name)):
            return view_name
    return None


def _boolish(value: Any, default: Any = False) -> bool:
    if value in (None, ""):
        return bool(default)
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def build_human_review_visualization(artifacts: dict) -> dict:
    summary = _as_dict(artifacts.get("human_review_priority_summary"))
    top = _as_list(artifacts.get("top_review_items")) or _as_list(
        artifacts.get("human_review_items")
    )
    prefill = _as_dict(artifacts.get("review_decision_prefill"))
    decisions = _as_list(prefill.get("decisions"))
    return {
        "visualization_type": "human_review_workflow",
        "review_item_count": int(summary.get("review_item_count") or len(top)),
        "prioritized_review_item_count": int(
            summary.get("prioritized_review_item_count") or len(top)
        ),
        "priority_level_counts": summary.get("priority_level_counts") or {},
        "issue_category_counts": summary.get("issue_category_counts") or {},
        "top_review_items": top[:10],
        "decision_template_path": "human_review/review_decision_template.json",
        "decision_prefill_path": "human_review/review_decision_prefill.json",
        "action_guide_path": artifacts.get("review_action_guide_path")
        or "human_review/review_action_guide.md",
        "decisions_auto_applied": any(_boolish(row.get("apply_decision")) for row in decisions),
        "decisions_applied_count": _as_dict(artifacts.get("human_review_application_summary")).get(
            "decisions_applied_count", 0
        ),
        "warning": "Generated decision templates are not auto-applied.",
        "generated_from_artifacts_only": True,
    }


def build_workflow_visualization_summary(artifacts: dict) -> dict:
    timeline = build_workflow_timeline(artifacts)
    topology = build_workflow_graph_topology(artifacts)
    evidence = build_evidence_flow_graph(artifacts)
    cards = build_claim_comparison_cards(artifacts)
    dataset_flow = build_dataset_decision_flow(artifacts)
    review = build_human_review_visualization(artifacts)
    run_summary = _as_dict(artifacts.get("workflow_run_summary"))
    run_quality = _as_dict(artifacts.get("run_quality_summary"))
    task = _as_dict(artifacts.get("task"))
    node_counts = Counter(row["node_type"] for row in evidence["nodes"])
    missing_link_warnings = [
        warning
        for warning in evidence.get("warnings", [])
        if "missing" in str(warning).lower() or "lacked linkable" in str(warning).lower()
    ]
    files = _visualization_file_list()
    return {
        "session_id": task.get("session_id"),
        "task_disease": task.get("disease"),
        "task_location": task.get("location"),
        "task_start_date": task.get("start_date"),
        "task_end_date": task.get("end_date"),
        "live_search_enabled": bool(run_summary.get("live_search_enabled")),
        "live_fetch_enabled": bool(run_summary.get("live_fetch_enabled")),
        "llm_stages_enabled": bool(
            run_summary.get("all_three_llm_stages_enabled")
            or any(
                _as_dict(run_summary.get("llm_stage_summary", {})).get(stage, {}).get("enabled")
                for stage in ("source_planning", "source_critic", "structured_extraction")
            )
        ),
        "run_quality_status": run_quality.get("run_quality_status")
        or run_summary.get("run_quality_status"),
        "final_case_dataset_count": _count_dataset(artifacts, "final_case_dataset"),
        "final_dataset_count": _count_dataset(artifacts, "final_dataset"),
        "final_dataset_pre_quality_gate_count": _count_dataset(
            artifacts, "final_dataset_pre_quality_gate"
        ),
        "context_record_count": _count_dataset(artifacts, "context_records"),
        "quarantined_record_count": _count_dataset(artifacts, "quarantined_records"),
        "pending_review_record_count": _count_dataset(artifacts, "pending_review_records"),
        "human_review_item_count": len(_as_list(artifacts.get("human_review_items"))),
        "prioritized_review_item_count": review.get("prioritized_review_item_count"),
        "workflow_timeline_step_count": len(timeline["items"]),
        "graph_node_count": len(topology["nodes"]),
        "graph_edge_count": len(topology["edges"]),
        "evidence_flow_node_count": len(evidence["nodes"]),
        "evidence_flow_edge_count": len(evidence["edges"]),
        "source_node_count": node_counts.get("source", 0),
        "document_node_count": node_counts.get("document", 0),
        "evidence_chunk_node_count": node_counts.get("evidence_chunk", 0),
        "record_node_count": sum(
            node_counts.get(kind, 0)
            for kind in ("raw_record", "validated_record", "normalized_record")
        ),
        "claim_node_count": node_counts.get("claim", 0),
        "claim_comparison_card_count": len(cards),
        "corroborated_event_node_count": node_counts.get("corroborated_event", 0),
        "dataset_view_node_count": node_counts.get("dataset_view", 0),
        "human_review_visualization_item_count": len(review.get("top_review_items") or []),
        "missing_link_warning_count": len(missing_link_warnings),
        "generated_from_artifacts_only": True,
        "llm_called_for_visualization": False,
        "search_called_for_visualization": False,
        "fetch_called_for_visualization": False,
        "visualization_files": files,
        "warnings": sorted(set(map(str, [*artifacts.get("warnings", []), *evidence.get("warnings", [])]))),
    }


def _count_dataset(artifacts: dict, name: str) -> int:
    rows = artifacts.get(name)
    if isinstance(rows, list):
        return len(rows)
    return 0


def _visualization_file_list() -> list[str]:
    return [
        "workflow_visualization/index.html",
        "workflow_visualization/workflow_visualization_summary.json",
        "workflow_visualization/workflow_visualization_manifest.json",
        "workflow_visualization/workflow_timeline.json",
        "workflow_visualization/workflow_timeline.md",
        "workflow_visualization/workflow_timeline.html",
        "workflow_visualization/workflow_graph_topology.json",
        "workflow_visualization/workflow_graph_topology.html",
        "workflow_visualization/agentic_search_timeline.json",
        "workflow_visualization/agentic_search_timeline.md",
        "workflow_visualization/agentic_search_timeline.html",
        "workflow_visualization/evidence_flow_graph.json",
        "workflow_visualization/evidence_flow_graph.html",
        "workflow_visualization/evidence_flow_table.csv",
        "workflow_visualization/claim_comparison_cards.json",
        "workflow_visualization/claim_comparison_cards.md",
        "workflow_visualization/claim_comparison_cards.html",
        "workflow_visualization/dataset_decision_flow.json",
        "workflow_visualization/dataset_decision_flow.html",
        "workflow_visualization/dataset_decision_table.csv",
        "workflow_visualization/human_review_workflow.json",
        "workflow_visualization/human_review_workflow.md",
        "workflow_visualization/human_review_workflow.html",
        "workflow_visualization/visualization_styles.css",
    ]


def write_workflow_visualization_artifacts(session_dir: Path | str) -> dict[str, str]:
    session_dir = Path(session_dir)
    vis_dir = session_dir / VIS_DIR
    vis_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_workflow_visualization_artifacts(session_dir)

    timeline = build_workflow_timeline(artifacts)
    topology = build_workflow_graph_topology(artifacts)
    search = build_agentic_search_timeline(artifacts)
    evidence = build_evidence_flow_graph(artifacts)
    cards = build_claim_comparison_cards(artifacts)
    dataset_flow = build_dataset_decision_flow(artifacts)
    review = build_human_review_visualization(artifacts)
    summary = build_workflow_visualization_summary(artifacts)

    paths: dict[str, str] = {}
    payloads = {
        "workflow_timeline": timeline,
        "workflow_graph_topology": topology,
        "agentic_search_timeline": search,
        "evidence_flow_graph": evidence,
        "claim_comparison_cards": cards,
        "dataset_decision_flow": dataset_flow,
        "human_review_workflow": review,
        "workflow_visualization_summary": summary,
    }
    for stem, payload in payloads.items():
        path = write_json(payload, vis_dir / f"{stem}.json")
        paths[_path_key(stem)] = str(path)

    (vis_dir / "visualization_styles.css").write_text(_css(), encoding="utf-8")
    (vis_dir / "workflow_timeline.md").write_text(_timeline_md(timeline), encoding="utf-8")
    (vis_dir / "agentic_search_timeline.md").write_text(_agentic_md(search), encoding="utf-8")
    (vis_dir / "claim_comparison_cards.md").write_text(_claim_cards_md(cards), encoding="utf-8")
    (vis_dir / "human_review_workflow.md").write_text(_human_review_md(review), encoding="utf-8")

    write_csv_rows(_evidence_table_rows(evidence), vis_dir / "evidence_flow_table.csv")
    write_csv_rows(dataset_flow["record_decisions"], vis_dir / "dataset_decision_table.csv")

    html_payloads = {
        "workflow_timeline": _timeline_html(timeline),
        "workflow_graph_topology": _topology_html(topology),
        "agentic_search_timeline": _agentic_html(search),
        "evidence_flow_graph": _evidence_html(evidence),
        "claim_comparison_cards": _claim_cards_html(cards),
        "dataset_decision_flow": _dataset_flow_html(dataset_flow),
        "human_review_workflow": _human_review_html(review),
    }
    for stem, content in html_payloads.items():
        path = vis_dir / f"{stem}.html"
        path.write_text(content, encoding="utf-8")
        paths[f"{stem}_html"] = str(path)

    manifest = {
        "session_id": summary.get("session_id"),
        "manifest_type": "workflow_visualization_manifest",
        "files": {
            file_path: str(session_dir / file_path).replace("\\", "/")
            for file_path in _visualization_file_list()
        },
        "generated_from_artifacts_only": True,
        "llm_called_for_visualization": False,
        "search_called_for_visualization": False,
        "fetch_called_for_visualization": False,
    }
    write_json(manifest, vis_dir / "workflow_visualization_manifest.json")
    paths["workflow_visualization_manifest"] = str(
        vis_dir / "workflow_visualization_manifest.json"
    )
    index = _index_html(summary, manifest)
    (vis_dir / "index.html").write_text(index, encoding="utf-8")
    paths["workflow_visualization_index"] = str(vis_dir / "index.html")
    paths["workflow_visualization_summary"] = str(
        vis_dir / "workflow_visualization_summary.json"
    )

    diagnostics = session_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    write_json(summary, diagnostics / "workflow_visualization_summary.json")
    write_json(evidence, diagnostics / "evidence_flow_graph.json")
    write_json(dataset_flow, diagnostics / "dataset_decision_flow.json")

    run_summary_path = session_dir / "workflow_run_summary.json"
    run_summary = _as_dict(_read_json(run_summary_path, {}))
    artifact_paths = _as_dict(run_summary.get("artifact_paths"))
    artifact_paths.update(paths)
    run_summary["artifact_paths"] = artifact_paths
    run_summary["workflow_visualization"] = {
        "workflow_visualization_summary": paths["workflow_visualization_summary"],
        "workflow_visualization_index": paths["workflow_visualization_index"],
        "generated_from_artifacts_only": True,
        "llm_called_for_visualization": False,
        "search_called_for_visualization": False,
        "fetch_called_for_visualization": False,
        "evidence_flow_node_count": summary["evidence_flow_node_count"],
        "evidence_flow_edge_count": summary["evidence_flow_edge_count"],
        "missing_link_warning_count": summary["missing_link_warning_count"],
    }
    write_json(run_summary, run_summary_path)
    return paths


def _path_key(stem: str) -> str:
    return {
        "workflow_visualization_summary": "workflow_visualization_summary",
    }.get(stem, f"{stem}_json")


def _evidence_table_rows(evidence: dict) -> list[dict]:
    nodes = {node["id"]: node for node in evidence.get("nodes", [])}
    return [
        {
            "source": edge["source"],
            "source_type": nodes.get(edge["source"], {}).get("node_type"),
            "target": edge["target"],
            "target_type": nodes.get(edge["target"], {}).get("node_type"),
            "edge_type": edge["edge_type"],
            "reason": edge.get("reason"),
            "warnings": "; ".join(map(str, edge.get("warnings") or [])),
        }
        for edge in evidence.get("edges", [])
    ]


def _html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_esc(title)}</title>
  <style>{_css()}</style>
</head>
<body>
<header><h1>{_esc(title)}</h1><p>Static artifact-only visualization for the data collection workflow.</p></header>
<main>{body}</main>
</body>
</html>
"""


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _json_block(value: Any) -> str:
    return _esc(json.dumps(value, ensure_ascii=False, indent=2))


def _css() -> str:
    return """
body{margin:0;background:#f5f7fb;color:#142033;font-family:Segoe UI,Arial,sans-serif;line-height:1.5}
header{padding:28px 36px;background:#0f2747;color:#fff}
header h1{margin:0 0 8px;font-size:30px;letter-spacing:0}
header p{margin:0;color:#dbeafe}
main{padding:24px 30px 44px}
section,.card{border:1px solid #d8e1ed;border-radius:8px;background:#fff;padding:18px;margin:0 0 16px}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.metric{border:1px solid #d8e1ed;border-radius:8px;padding:14px;background:#fbfdff}
.metric span{display:block;color:#637083;font-size:13px}.metric strong{font-size:28px}
table{width:100%;border-collapse:collapse;background:#fff}td,th{border:1px solid #d8e1ed;padding:8px;text-align:left;vertical-align:top}
code{background:#edf3f9;border-radius:4px;padding:2px 5px}pre{white-space:pre-wrap;background:#f7f9fc;border:1px solid #d8e1ed;border-radius:6px;padding:12px;overflow:auto}
.warn{border-left:5px solid #9a6700;background:#fff9eb}.ok{border-left:5px solid #13795b}.bad{border-left:5px solid #b42318}
a{color:#155eaa}.pill{display:inline-block;border:1px solid #d8e1ed;border-radius:999px;padding:3px 8px;margin:2px;background:#fbfdff}
@media(max-width:900px){.grid{grid-template-columns:1fr}main{padding:16px}}
"""


def _metrics_html(summary: dict) -> str:
    metrics = [
        ("Final case dataset", summary.get("final_case_dataset_count")),
        ("Final dataset", summary.get("final_dataset_count")),
        ("Evidence nodes", summary.get("evidence_flow_node_count")),
        ("Evidence edges", summary.get("evidence_flow_edge_count")),
        ("Claim cards", summary.get("claim_comparison_card_count")),
        ("Human review items", summary.get("human_review_visualization_item_count")),
    ]
    return '<div class="grid">' + "".join(
        f'<div class="metric"><span>{_esc(label)}</span><strong>{_esc(value)}</strong></div>'
        for label, value in metrics
    ) + "</div>"


def _index_html(summary: dict, manifest: dict) -> str:
    links = [
        ("Workflow timeline", "workflow_timeline.html"),
        ("Graph topology", "workflow_graph_topology.html"),
        ("Agentic search timeline", "agentic_search_timeline.html"),
        ("Evidence flow graph", "evidence_flow_graph.html"),
        ("Claim comparison cards", "claim_comparison_cards.html"),
        ("Dataset decision flow", "dataset_decision_flow.html"),
        ("Human review workflow", "human_review_workflow.html"),
        ("Visualization summary JSON", "workflow_visualization_summary.json"),
        ("Visualization manifest JSON", "workflow_visualization_manifest.json"),
        ("Workflow console", "../workflow_console/data_collection_workflow_console.html"),
        ("Interpretive report", "../workflow_interpretive_report.md"),
        ("Human review action guide", "../human_review/review_action_guide.md"),
    ]
    body = f"""
<section><h2>Run Status</h2>{_metrics_html(summary)}
<p><strong>Task:</strong> {_esc(summary.get('task_disease'))} / {_esc(summary.get('task_location'))} / {_esc(summary.get('task_start_date'))} to {_esc(summary.get('task_end_date'))}</p>
<p><strong>run_quality_status:</strong> <code>{_esc(summary.get('run_quality_status'))}</code></p>
<p><strong>Artifact-only:</strong> <code>true</code>; visualization did not call LLM, search, or fetch.</p>
</section>
<section><h2>Visualization Panels</h2><ul>{''.join(f'<li><a href="{_esc(href)}">{_esc(label)}</a></li>' for label, href in links)}</ul></section>
<section><h2>Warnings</h2><pre>{_json_block(summary.get('warnings') or [])}</pre></section>
<section><h2>Manifest</h2><pre>{_json_block(manifest)}</pre></section>
"""
    return _html_page("Workflow Visualization Index", body)


def _timeline_md(timeline: dict) -> str:
    lines = ["# Workflow Timeline", ""]
    for item in timeline.get("items", []):
        lines.append(
            f"{item['step_index']}. `{item['node_name']}` - {item['status']} - {item['user_facing_explanation']}"
        )
    lines.append("")
    return "\n".join(lines)


def _timeline_html(timeline: dict) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{_esc(item['step_index'])}</td><td><code>{_esc(item['node_name'])}</code></td>"
        f"<td>{_esc(item['status'])}</td><td>{_esc(item.get('route_after_node'))}</td>"
        f"<td>{_esc(item['user_facing_explanation'])}</td>"
        "</tr>"
        for item in timeline.get("items", [])
    )
    return _html_page(
        "Workflow Timeline",
        f"<section><h2>Timeline</h2><table><tr><th>#</th><th>Node</th><th>Status</th><th>Route</th><th>Explanation</th></tr>{rows}</table></section>",
    )


def _topology_html(topology: dict) -> str:
    rows = "".join(
        f"<tr><td><code>{_esc(edge['source'])}</code></td><td><code>{_esc(edge['target'])}</code></td><td>{_esc(edge.get('condition'))}</td><td>{_esc(edge.get('route_label'))}</td></tr>"
        for edge in topology.get("edges", [])
    )
    return _html_page(
        "Workflow Graph Topology",
        f"<section><h2>Topology</h2><p>Source: {_esc(topology.get('source'))}</p><table><tr><th>Source</th><th>Target</th><th>Condition</th><th>Route</th></tr>{rows}</table><pre>{_json_block(topology.get('warnings'))}</pre></section>",
    )


def _agentic_md(search: dict) -> str:
    lines = ["# Agentic Search Timeline", ""]
    for card in search.get("cards", []):
        lines.append(f"- **{card.get('card_type')}**: `{card.get('stop_decision') or card.get('title') or ''}`")
    return "\n".join(lines) + "\n"


def _agentic_html(search: dict) -> str:
    cards = "".join(
        f"<article class='card'><h2>{_esc(card.get('card_type'))}</h2><pre>{_json_block(card)}</pre></article>"
        for card in search.get("cards", [])
    )
    return _html_page("Agentic Search Timeline", cards)


def _evidence_html(evidence: dict) -> str:
    node_rows = "".join(
        f"<tr><td><code>{_esc(node['id'])}</code></td><td>{_esc(node['node_type'])}</td><td>{_esc(node.get('title'))}</td><td>{_esc(node.get('status'))}</td></tr>"
        for node in evidence.get("nodes", [])
    )
    edge_rows = "".join(
        f"<tr><td><code>{_esc(edge['source'])}</code></td><td><code>{_esc(edge['target'])}</code></td><td>{_esc(edge['edge_type'])}</td><td>{_esc(edge.get('reason'))}</td></tr>"
        for edge in evidence.get("edges", [])
    )
    return _html_page(
        "Evidence Flow Graph",
        f"<section><h2>Nodes</h2><table><tr><th>ID</th><th>Type</th><th>Title</th><th>Status</th></tr>{node_rows}</table></section><section><h2>Edges</h2><table><tr><th>Source</th><th>Target</th><th>Type</th><th>Reason</th></tr>{edge_rows}</table></section><section class='warn'><h2>Warnings</h2><pre>{_json_block(evidence.get('warnings'))}</pre></section>",
    )


def _claim_cards_md(cards: list[dict]) -> str:
    lines = ["# Claim Comparison Cards", ""]
    if not cards:
        return "# Claim Comparison Cards\n\nNo claim comparisons were available.\n"
    for card in cards:
        lines.append(
            f"- `{card['comparison_id']}`: {card['group']} ({card.get('reason') or 'no reason supplied'})"
        )
    return "\n".join(lines) + "\n"


def _claim_cards_html(cards: list[dict]) -> str:
    if not cards:
        body = "<section><p>No claim comparisons were available. The workflow did not generate automatic truth conclusions.</p></section>"
    else:
        body = "".join(
            f"<article class='card'><h2>{_esc(card['comparison_id'])}</h2><p><strong>Group:</strong> {_esc(card['group'])}</p><p>{_esc(card.get('reason'))}</p><pre>{_json_block(card)}</pre></article>"
            for card in cards
        )
    return _html_page("Claim Comparison Cards", body)


def _dataset_flow_html(flow: dict) -> str:
    rows = "".join(
        f"<tr><td><code>{_esc(row['record_id'])}</code></td><td>{_esc(row.get('dataset_view'))}</td><td>{_esc(row.get('final_inclusion_status'))}</td><td>{_esc(row.get('quality_gate_reasons'))}</td></tr>"
        for row in flow.get("record_decisions", [])
    )
    status = "No accepted primary case records" if flow.get("final_case_dataset_empty") else "Accepted primary case records present"
    return _html_page(
        "Dataset Decision Flow",
        f"<section class='{'warn' if flow.get('final_case_dataset_empty') else 'ok'}'><h2>{_esc(status)}</h2><pre>{_json_block(flow.get('dataset_view_counts'))}</pre></section><section><table><tr><th>Record</th><th>Dataset View</th><th>Status</th><th>Reasons</th></tr>{rows}</table></section>",
    )


def _human_review_md(review: dict) -> str:
    lines = [
        "# Human Review Workflow",
        "",
        f"- Review items: `{review.get('review_item_count')}`",
        f"- Prioritized items: `{review.get('prioritized_review_item_count')}`",
        "- Generated decision templates are not auto-applied.",
        "",
    ]
    for item in review.get("top_review_items", []):
        lines.append(
            f"- `{item.get('review_item_id') or item.get('review_id')}`: {item.get('priority_level')} / {item.get('short_title') or item.get('reason')}"
        )
    return "\n".join(lines) + "\n"


def _human_review_html(review: dict) -> str:
    cards = "".join(
        f"<article class='card'><h2>{_esc(item.get('review_item_id') or item.get('review_id'))}</h2><p><strong>{_esc(item.get('priority_level'))}</strong> {_esc(item.get('issue_category'))}</p><p>{_esc(item.get('short_title') or item.get('reason'))}</p><pre>{_json_block(item)}</pre></article>"
        for item in review.get("top_review_items", [])
    )
    body = f"<section class='warn'><h2>Manual Review Required</h2><p>Generated decision templates are not auto-applied.</p><pre>{_json_block({k: v for k, v in review.items() if k != 'top_review_items'})}</pre></section>{cards}"
    return _html_page("Human Review Workflow", body)
