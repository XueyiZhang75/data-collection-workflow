"""LangGraph shared state definition and helpers."""

from __future__ import annotations

from typing import TypedDict

from .models import TraceEvent


class DataCollectionState(TypedDict, total=False):
    qualified_records: list[dict]
    candidate_records: list[dict]
    rejected_evidence_records: list[dict]
    record_qualifications: list[dict]
    qualified_case_records: list[dict]
    qualified_aggregate_records: list[dict]
    qualified_context_records: list[dict]
    unresolved_records: list[dict]
    pipeline_mode: str
    session_artifact_root: str
    recovery_round: int
    recovery_previous_gain: list[str]
    recovery_search_streak_start: int
    recovery_gaps: list[dict]
    recovery_plan: dict
    recovery_action_history: list[dict]
    recovery_stop_reason: str | None
    acquisition_frontier: dict
    run_budget_ledger: dict
    extraction_attempted_chunk_ids: list[str]
    supported_event_terms: list[dict]
    result_manifest: dict
    evidence_products: dict
    user_request: str
    structured_task: dict | None
    collection_spec: dict | None
    task_intake_summary: dict | None
    disease_intelligence: dict | None
    disease_intelligence_summary: dict | None
    disease_profile: dict | None
    collection_schema: dict | None
    source_strategy: dict | None
    screening_criteria: dict | None
    task_acceptance_contract: dict | None
    task_evidence_contract: dict | None
    profile_schema_summary: dict | None
    search_queries: dict | None
    search_query_inventory: list[dict]
    evidence_strategy_plan: dict | None
    agentic_source_plan: dict | None
    executable_source_plan_summary: dict | None
    localized_source_planning_summary: dict | None
    source_planning_agent_summary: dict | None
    source_candidates: list[dict]
    official_coverage_candidates: list[dict]
    source_search_results: list[dict]
    source_search_execution_summary: dict | None
    source_discovery_summary: dict | None
    iterative_source_discovery_summary: dict | None
    search_iteration_plans: list[dict]
    search_iteration_observations: list[dict]
    search_refinement_decisions: list[dict]
    iterative_search_queries: list[dict]
    source_registry: list[dict]
    source_registry_summary: dict | None
    source_screening_summary: dict | None
    source_triage_results: list[dict]
    source_critic_summary: dict | None
    source_critic_results: list[dict]
    source_routing_summary: dict | None
    source_credibility_assessments: list[dict]
    source_credibility_summary: dict | None
    source_identity_assessments: list[dict]
    source_identity_summary: dict | None
    source_coverage_requirements: list[dict]
    source_coverage_audit: dict | None
    target_official_fetch_plan: list[dict]
    must_fetch_sources: list[dict]
    fetch_failures_blocking: list[dict]
    disease_relevance_summary: dict | None
    content_fetch_requests: list[dict]
    content_fetch_summary: dict | None
    document_parse_summary: dict | None
    fetch_manifest: list[dict]
    fixture_document_summary: dict | None
    documents: list[dict]
    document_quality_summary: dict | None
    evidence_chunks: list[dict]
    evidence_chunking_summary: dict | None
    data_presence_summary: dict | None
    chunk_relevance_assessments: list[dict]
    raw_records: list[dict]
    structured_extraction_summary: dict | None
    llm_extraction_summary: dict | None
    record_task_fit_assessments: list[dict]
    metric_extraction_plan: dict | None
    metric_row_extraction_audit: list[dict]
    official_extraction_queue: list[dict]
    official_extraction_failures: list[dict]
    extraction_budget_by_source: dict | None
    validated_records: list[dict]
    rejected_records: list[dict]
    schema_validation_summary: dict | None
    normalized_records: list[dict]
    record_normalization_summary: dict | None
    linked_events: list[dict]
    record_linking_summary: dict | None
    event_clusters: list[dict]
    duplicate_clusters: list[dict]
    event_clustering_summary: dict | None
    duplicate_detection_summary: dict | None
    validation_records: list[dict]
    active_validation_records: list[dict]
    inactive_validation_records: list[dict]
    validation_source_compatibility_summary: dict | None
    validation_cases: list[dict]
    validation_comparisons: list[dict]
    validation_results: list[dict]
    validation_summary: dict | None
    trusted_source_validation_summary: dict | None
    cross_source_validation_summary: dict | None
    claims: list[dict]
    claim_comparisons: list[dict]
    corroborated_events: list[dict]
    corroboration_summary: dict | None
    anomaly_results: list[dict]
    anomaly_summary: dict | None
    anomaly_review_items: list[dict]
    conflicts: list[dict]
    cross_source_consistency_summary: dict | None
    human_review_queue: list[dict]
    human_review_enabled: bool | None
    human_review_decisions: list[dict]
    human_review_decisions_path: str | None
    applied_human_review_decisions: list[dict]
    rejected_human_review_decisions: list[dict]
    human_review_audit_trail: list[dict]
    human_review_application_summary: dict | None
    final_dataset_pre_quality_gate: list[dict]
    quarantined_records: list[dict]
    pending_review_records: list[dict]
    non_primary_observations: list[dict]
    final_case_dataset: list[dict]
    probable_case_dataset: list[dict]
    suspected_case_dataset: list[dict]
    unspecified_case_dataset: list[dict]
    death_dataset: list[dict]
    hospitalization_dataset: list[dict]
    zero_case_statements: list[dict]
    exposure_monitoring_records: list[dict]
    surveillance_summary_records: list[dict]
    outbreak_summary_records: list[dict]
    context_records: list[dict]
    unclassified_observation_records: list[dict]
    observation_type_dataset_summary: dict | None
    record_inclusion_decisions: list[dict]
    run_quality_summary: dict | None
    final_dataset_quality_summary: dict | None
    direct_collection_summary: dict | None
    collection_decision_summary: dict | None
    final_dataset_post_review: list[dict]
    records_excluded_by_human_review: list[dict]
    human_review_summary: dict | None
    final_data_package: dict | None
    finalization_summary: dict | None
    collection_trace: list[dict]
    current_route: str | None


def append_trace(
    state: DataCollectionState,
    node_name: str,
    message: str,
    metadata: dict | None = None,
) -> list[dict]:
    """Return an updated collection_trace list with a new TraceEvent appended.

    The original state is not mutated; callers should assign the returned list
    back into the state update returned from their node function.
    """

    existing = list(state.get("collection_trace") or [])
    event = TraceEvent(
        node_name=node_name,
        message=message,
        metadata=metadata or {},
    )
    existing.append(event.model_dump())
    return existing
