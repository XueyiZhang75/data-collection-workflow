"""LangGraph routing for disease data collection and bounded evidence recovery."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    content_fetch_and_parse,
    cross_source_consistency_check,
    document_quality_check,
    evidence_chunking_and_data_presence_flagging,
    final_data_package_builder,
    human_review,
    disease_intelligence_builder,
    executable_source_planning,
    profile_and_schema_setup,
    quality_gate_routing,
    query_strategy_builder,
    record_linking,
    record_normalization,
    schema_validation_and_repair,
    source_critic_and_uncertainty_routing,
    source_dedup_and_registry,
    source_discovery,
    source_screening,
    structured_extraction,
    task_intake_and_scope_planning,
)
from .state import DataCollectionState


def _route_after_quality_gate(state: DataCollectionState) -> str:
    if state.get("current_route") == "human_review":
        return "human_review"
    return "final_data_package_builder"


def build_graph(*, checkpointer=None):
    """Compile the configured workflow, preserving the legacy route when evidence is disabled."""

    from .evidence_qualification import evidence_qualification_enabled
    universal = evidence_qualification_enabled()
    builder = StateGraph(DataCollectionState)

    builder.add_node("task_intake_and_scope_planning", task_intake_and_scope_planning)
    builder.add_node("disease_intelligence_builder", disease_intelligence_builder)
    builder.add_node("profile_and_schema_setup", profile_and_schema_setup)
    builder.add_node("executable_source_planning", executable_source_planning)
    builder.add_node("query_strategy_builder", query_strategy_builder)
    builder.add_node("source_discovery", source_discovery)
    builder.add_node("source_dedup_and_registry", source_dedup_and_registry)
    builder.add_node("source_screening", source_screening)
    builder.add_node("source_critic_and_uncertainty_routing", source_critic_and_uncertainty_routing)
    builder.add_node("content_fetch_and_parse", content_fetch_and_parse)
    builder.add_node("document_quality_check", document_quality_check)
    builder.add_node("evidence_chunking_and_data_presence_flagging", evidence_chunking_and_data_presence_flagging)
    builder.add_node("structured_extraction", structured_extraction)
    builder.add_node("schema_validation_and_repair", schema_validation_and_repair)
    builder.add_node("record_normalization", record_normalization)
    builder.add_node("record_linking", record_linking)
    builder.add_node("cross_source_consistency_check", cross_source_consistency_check)
    builder.add_node("quality_gate_routing", quality_gate_routing)
    builder.add_node("human_review", human_review)
    builder.add_node("final_data_package_builder", final_data_package_builder)

    if universal:
        from .session_runtime import require_ready_runtime
        from .workflow_recovery import recovery_control, recovery_execute
        builder.add_node('universal_preflight', require_ready_runtime)
        builder.add_node('recovery_control', recovery_control)
        builder.add_node('recovery_execute', recovery_execute)
        builder.add_edge(START, 'universal_preflight')
        builder.add_edge('universal_preflight', 'task_intake_and_scope_planning')
    else:
        builder.add_edge(START, "task_intake_and_scope_planning")
    builder.add_edge("task_intake_and_scope_planning", "disease_intelligence_builder")
    builder.add_edge("disease_intelligence_builder", "profile_and_schema_setup")
    builder.add_edge("profile_and_schema_setup", "executable_source_planning")
    builder.add_edge("executable_source_planning", "query_strategy_builder")
    builder.add_edge("query_strategy_builder", "source_discovery")
    builder.add_edge("source_discovery", "source_dedup_and_registry")
    builder.add_edge("source_dedup_and_registry", "source_screening")
    builder.add_edge("source_screening", "source_critic_and_uncertainty_routing")
    builder.add_edge("source_critic_and_uncertainty_routing", "content_fetch_and_parse")
    builder.add_edge("content_fetch_and_parse", "document_quality_check")
    builder.add_edge("document_quality_check", "evidence_chunking_and_data_presence_flagging")
    builder.add_edge("evidence_chunking_and_data_presence_flagging", "structured_extraction")
    builder.add_edge("structured_extraction", "schema_validation_and_repair")
    builder.add_edge("schema_validation_and_repair", "record_normalization")
    builder.add_edge("record_normalization", "record_linking")
    builder.add_edge("record_linking", "cross_source_consistency_check")
    builder.add_edge("cross_source_consistency_check", "quality_gate_routing")

    if universal:
        builder.add_edge('quality_gate_routing', 'recovery_control')
        builder.add_conditional_edges('recovery_control', lambda state: 'recovery_execute' if (state.get('recovery_plan') or {}).get('actions') else _route_after_quality_gate(state), {'recovery_execute':'recovery_execute','human_review':'human_review','final_data_package_builder':'final_data_package_builder'})
        builder.add_edge('recovery_execute','schema_validation_and_repair')
    else:
        builder.add_conditional_edges('quality_gate_routing', _route_after_quality_gate, {'human_review':'human_review','final_data_package_builder':'final_data_package_builder'})

    builder.add_edge("human_review", "final_data_package_builder")
    builder.add_edge("final_data_package_builder", END)

    return builder.compile(checkpointer=checkpointer)
