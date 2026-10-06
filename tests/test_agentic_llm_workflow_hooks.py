"""Offline-safe tests for optional LLM agent hooks in the workflow."""

from __future__ import annotations

import sys
import importlib
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_collection_workflow import llm_clients  # noqa: E402
from data_collection_workflow.agents import source_planning_agent  # noqa: E402
from data_collection_workflow.graph import build_graph  # noqa: E402

source_screening_module = importlib.import_module(  # noqa: E402
    "data_collection_workflow.nodes.source_screening"
)

from synthetic_source_roles import (
    COLLECTION_SOURCE_ID, CONTEXT_SOURCE_ID, VALIDATION_SOURCE_ID,
    write_source_role_overlays,
)

_CONTEXT = CONTEXT_SOURCE_ID
_VALIDATION = VALIDATION_SOURCE_ID


@pytest.fixture(autouse=True)
def _safe_offline_env(monkeypatch):
    for name in (
        "ENABLE_LLM_SOURCE_PLANNING",
        "ENABLE_LLM_SOURCE_CRITIC",
        "ENABLE_LLM_EXTRACTION",
        "ENABLE_LIVE_FETCH",
        "USE_FIXTURE_DOCUMENTS",
        "COLLECTION_MODE",
        "SEED_SOURCE_OVERLAY_PATH",
        "SOURCE_ROLE_POLICY_OVERLAY_PATH",
        "SOURCE_ID_ALLOWLIST",
    ):
        monkeypatch.delenv(name, raising=False)


def _initial_state() -> dict:
    return {
        "user_request": (
            "Collect global human hantavirus case, outbreak, and surveillance data "
            "from 2020 to 2026."
        ),
        "source_candidates": [],
        "source_discovery_summary": None,
        "source_registry": [],
        "source_registry_summary": None,
        "documents": [],
        "evidence_chunks": [],
        "raw_records": [],
        "validated_records": [],
        "normalized_records": [],
        "linked_events": [],
        "conflicts": [],
        "human_review_queue": [],
        "human_review_decisions": [],
        "collection_trace": [],
        "collection_spec": None,
        "disease_profile": None,
        "collection_schema": None,
        "source_strategy": None,
        "screening_criteria": None,
        "search_queries": None,
        "search_query_inventory": [],
        "agentic_source_plan": None,
        "source_planning_agent_summary": None,
        "content_fetch_requests": [],
        "content_fetch_summary": None,
        "fixture_document_summary": None,
        "document_quality_summary": None,
        "final_data_package": None,
        "current_route": None,
    }


def _registry_by_id(result: dict) -> dict[str, dict]:
    return {
        entry.get("source_id"): entry
        for entry in (result.get("source_registry") or [])
    }


def test_llm_agent_feature_flags_default_off_and_no_real_helper_call(
    monkeypatch,
):
    def fail_if_called(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("LLM helper should not be called when flags are off")

    monkeypatch.setattr(llm_clients, "run_structured_llm_json", fail_if_called)
    monkeypatch.setattr(llm_clients, "run_pydantic_structured_llm", fail_if_called)

    result = build_graph().invoke(_initial_state())

    planning_summary = result.get("source_planning_agent_summary") or {}
    routing_summary = result.get("source_routing_summary") or {}
    assert planning_summary.get("llm_source_planning_enabled") is False
    assert routing_summary.get("llm_source_critic_enabled") is False
    plan = result.get("agentic_source_plan") or {}
    assert plan.get("generation_method") == "deterministic_executable_source_plan"
    assert plan.get("execution_status") == "planned_not_executed"
    assert result.get("executable_source_plan_summary")
    assert not [
        item for item in (result.get("search_query_inventory") or [])
        if item.get("query_source") == "llm_source_planning_agent"
    ]


def test_source_planning_agent_mocked_output_appends_queries(monkeypatch):
    def mock_llm_json(*args, **kwargs):  # noqa: ARG001
        return {
            "plan_id": "plan_agentic_hook_test",
            "_structured_output_mode": "provider_native",
            "disease": "Hantavirus disease",
            "location": "global",
            "time_window": "2020-2026",
            "target_fields": ["cases", "deaths", "date", "location"],
            "generation_method": "llm_executable_source_plan",
            "llm_enabled": True,
            "execution_status": "planned_not_executed",
            "warnings": ["candidate_urls_not_allowed_in_executable_plan"],
            "source_discovery_objectives": [
                {
                    "objective_id": "obj_agentic_hook_001",
                    "objective": "Plan official collection queries.",
                    "source_role_hint": "collection",
                    "rationale": "Official sources support extractable case records.",
                    "priority": 1,
                }
            ],
            "planned_source_categories": [
                {
                    "source_category_id": "cat_agentic_hook_001",
                    "source_type": "official_public_health_agency",
                    "role_hint": "collection",
                    "priority": 1,
                    "expected_fields": ["cases", "deaths", "date", "location"],
                    "why_relevant": "Likely to contain primary case reporting.",
                    "risk_notes": ["requires_screening_before_fetch"],
                }
            ],
            "planned_queries": [
                {
                    "query_id": "q_agentic_hook_001",
                    "query": '"Andes virus" "MV Hondius" local health agency',
                    "query_type": "general_web",
                    "provider_channel": "web_search",
                    "source_type": "official_public_health_agency",
                    "role_hint": "collection",
                    "priority": 1,
                    "expected_fields": ["cases", "deaths", "date", "location"],
                    "disease_terms_used": ["Andes virus"],
                    "location_terms_used": ["MV Hondius"],
                    "time_terms_used": ["2020-2026"],
                    "rationale": "Look for non-held-out local authority coverage.",
                    "execution_status": "planned_not_executed",
                }
            ],
            "source_planning_risks": [
                {
                    "risk_id": "risk_agentic_hook_001",
                    "risk": "LLM query needs screening before execution.",
                    "severity": "medium",
                    "applies_to": ["collection"],
                    "mitigation": "Keep planned query status as planned_not_executed.",
                    "human_review_trigger": True,
                }
            ],
        }

    monkeypatch.setenv("ENABLE_LLM_SOURCE_PLANNING", "true")
    monkeypatch.setattr(llm_clients, "run_pydantic_structured_llm", mock_llm_json)

    result = build_graph().invoke(_initial_state())
    inventory = result.get("search_query_inventory") or []
    agent_queries = [
        item for item in inventory
        if item.get("query_source") == "executable_source_plan"
        and item.get("query_id") == "q_agentic_hook_001"
    ]

    assert result.get("agentic_source_plan")
    assert result["source_planning_agent_summary"]["status"] == "success"
    assert result["source_planning_agent_summary"]["agent_query_added_count"] == 1
    assert result["agentic_source_plan"]["structured_output_mode"] == "provider_native"
    assert agent_queries
    assert agent_queries[0]["query"] == (
        '"Andes virus" "MV Hondius" local health agency'
    )
    assert any(item.get("query_id", "").startswith("q_official_") for item in inventory)


def test_source_planning_pydantic_schema_defaults_and_validation():
    output = source_planning_agent.SourcePlanningOutput()

    assert output.agent_name == "source_planning_agent"
    assert output.agent_version == "0.3"
    assert output.proposed_search_queries == []
    assert output.candidate_source_hints == []

    parsed = source_planning_agent.SourcePlanningOutput(
        proposed_search_queries=[
            {
                "query": '"hantavirus" "New Mexico"',
                "source_type": "official_public_health_agency",
                "priority": "2",
            }
        ]
    )
    assert parsed.proposed_search_queries[0].priority == 2

    with pytest.raises(Exception):
        source_planning_agent.SourcePlanningOutput(
            proposed_search_queries=[{"priority": "not-an-int"}]
        )


def test_run_pydantic_structured_llm_uses_provider_native(monkeypatch):
    class FakeStructuredModel:
        def invoke(self, messages):
            assert messages
            return source_planning_agent.SourcePlanningOutput(
                proposed_search_queries=[
                    {"query": '"hantavirus" "New Mexico" official data'}
                ]
            )

    class FakeChatModel:
        def with_structured_output(self, schema_model):
            assert schema_model is source_planning_agent.SourcePlanningOutput
            return FakeStructuredModel()

    monkeypatch.setattr(
        llm_clients, "build_chat_model", lambda settings=None: FakeChatModel()
    )

    result = llm_clients.run_pydantic_structured_llm(
        system_prompt="system",
        user_prompt="user",
        schema_model=source_planning_agent.SourcePlanningOutput,
        model="fake-model",
    )

    assert result["_structured_output_mode"] == "provider_native"
    assert result["proposed_search_queries"][0]["query"] == (
        '"hantavirus" "New Mexico" official data'
    )


def test_run_pydantic_structured_llm_passes_langsmith_run_metadata(monkeypatch):
    captured = {}

    class FakeStructuredModel:
        def invoke(self, messages, config=None):
            captured["config"] = config
            return source_planning_agent.SourcePlanningOutput(
                proposed_search_queries=[
                    {"query": '"hantavirus" "Virginia" official data'}
                ]
            )

    class FakeChatModel:
        def with_structured_output(self, schema_model):
            assert schema_model is source_planning_agent.SourcePlanningOutput
            return FakeStructuredModel()

    monkeypatch.setattr(
        llm_clients, "build_chat_model", lambda settings=None: FakeChatModel()
    )
    monkeypatch.setenv("TRACE_SESSION_ID", "trace_session")
    monkeypatch.setenv("TRACE_ID", "trace-123")
    monkeypatch.setenv("LANGSMITH_PROJECT", "hdc-workflow-demo")

    result = llm_clients.run_pydantic_structured_llm(
        system_prompt="system",
        user_prompt="user",
        schema_model=source_planning_agent.SourcePlanningOutput,
        model="fake-model",
    )

    assert result["_structured_output_mode"] == "provider_native"
    assert captured["config"]["run_name"] == "llm.SourcePlanningOutput"
    assert "data-collection-llm" in captured["config"]["tags"]
    assert captured["config"]["metadata"]["workflow_stage"] == "SourcePlanningOutput"
    assert captured["config"]["metadata"]["llm_settings"]["model"] == "fake-model"
    assert not any(key.startswith("hdc_") for key in captured["config"]["metadata"])
    assert captured["config"]["metadata"]["session_id"] == "trace_session"
    assert captured["config"]["metadata"]["trace_id"] == "trace-123"
    assert captured["config"]["metadata"]["langsmith_project"] == "hdc-workflow-demo"


def test_source_planning_prompt_requires_json_only_with_skeleton():
    prompt_text = source_planning_agent._load_prompt_text()

    assert "Return exactly one JSON object" in prompt_text
    assert "Do not wrap the JSON in Markdown" in prompt_text
    assert "If uncertain" in prompt_text
    assert '"proposed_search_queries"' in prompt_text
    assert '"candidate_source_hints"' in prompt_text
    assert '"warnings"' in prompt_text


def test_source_planning_agent_empty_output_retries_once(monkeypatch):
    calls = []

    def mock_llm_json(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ValueError("LLM returned empty output.")
        return {
            "agent_name": "source_planning_agent",
            "_structured_output_mode": "provider_native",
            "proposed_search_queries": [
                {
                    "query": '"hantavirus" "New Mexico" health department cases',
                    "source_type": "official_public_health_agency",
                }
            ],
            "proposed_collection_source_types": [
                "official_public_health_agency"
            ],
            "proposed_validation_source_types": [
                "international_organization_report"
            ],
            "proposed_context_source_types": ["official_background_page"],
            "reasoning_summary": "Retry returned a minimal source plan.",
            "warnings": [],
        }

    monkeypatch.setattr(llm_clients, "run_pydantic_structured_llm", mock_llm_json)

    plan = source_planning_agent.plan_sources_with_llm(
        user_request="Plan New Mexico HPS sources without fetching webpages.",
        collection_spec={"disease": "Hantavirus disease"},
        disease_profile={
            "disease_standard_name": "Hantavirus disease",
            "include_terms": ["hantavirus"],
        },
        source_strategy={"source_categories": []},
        collection_schema={"core_fields": [{"name": "cases_confirmed"}]},
    )

    assert len(calls) == 2
    assert plan["retry_attempted"] is True
    assert plan["retry_succeeded"] is True
    assert plan["structured_output_attempted"] is True
    assert plan["structured_output_mode"] == "provider_native"
    assert plan["proposed_search_queries"][0]["query"] == (
        '"hantavirus" "New Mexico" health department cases'
    )
    assert "Return a minimal source planning JSON object" in calls[1]["user_prompt"]


def test_source_planning_agent_retry_failure_preserves_workflow_fallback(monkeypatch):
    call_count = {"count": 0}

    def empty_response(*args, **kwargs):  # noqa: ARG001
        call_count["count"] += 1
        raise ValueError("LLM returned empty output.")

    monkeypatch.setenv("ENABLE_LLM_SOURCE_PLANNING", "true")
    monkeypatch.setattr(llm_clients, "run_pydantic_structured_llm", empty_response)

    result = build_graph().invoke(_initial_state())
    inventory = result.get("search_query_inventory") or []
    summary = result.get("source_planning_agent_summary") or {}

    assert call_count["count"] == 1
    assert inventory
    assert summary.get("status") == "failed_deterministic_fallback"
    assert summary.get("generation_method") == "llm_failed_deterministic_fallback"
    assert "LLM returned empty output" in summary.get("failure_message")
    assert not [
        item for item in inventory
        if item.get("query_source") == "llm_source_planning_agent"
    ]


def test_source_planning_agent_uses_compact_payload(monkeypatch):
    captured = {}

    def mock_llm_json(*args, **kwargs):
        captured["user_prompt"] = kwargs["user_prompt"]
        return {
            "agent_name": "source_planning_agent",
            "_structured_output_mode": "provider_native",
            "proposed_search_queries": [],
            "proposed_collection_source_types": [],
            "proposed_validation_source_types": [],
            "proposed_context_source_types": [],
            "reasoning_summary": "No query needed in test.",
            "warnings": [],
        }

    monkeypatch.setattr(llm_clients, "run_pydantic_structured_llm", mock_llm_json)

    source_planning_agent.plan_sources_with_llm(
        user_request="Plan sources.",
        collection_spec={"required_fields": ["cases_confirmed"]},
        disease_profile={
            "disease_standard_name": "Hantavirus disease",
            "include_terms": ["hantavirus"],
            "raw_large_notes": "SHOULD_NOT_BE_SENT",
        },
        source_strategy={
            "source_categories": [
                {
                    "source_type": "official_public_health_agency",
                    "priority": 1,
                    "description": "SHOULD_NOT_BE_SENT",
                }
            ],
            "screening_criteria": {"include_if_all_apply": ["SHOULD_NOT_BE_SENT"]},
        },
        collection_schema={
            "core_fields": [
                {
                    "name": "cases_confirmed",
                    "description": "SHOULD_NOT_BE_SENT",
                }
            ]
        },
    )

    user_prompt = captured["user_prompt"]
    assert "cases_confirmed" in user_prompt
    assert "official_public_health_agency" in user_prompt
    assert "SHOULD_NOT_BE_SENT" not in user_prompt


def test_source_planning_agent_malformed_output_falls_back(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_SOURCE_PLANNING", "true")
    monkeypatch.setattr(
        llm_clients,
        "run_pydantic_structured_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("structured output validation failed")
        ),
    )

    result = build_graph().invoke(_initial_state())
    inventory = result.get("search_query_inventory") or []
    summary = result.get("source_planning_agent_summary") or {}

    assert inventory
    assert summary.get("status") == "failed_deterministic_fallback"
    assert summary.get("generation_method") == "llm_failed_deterministic_fallback"
    assert "llm_source_planning_failed_deterministic_fallback_used" in (
        summary.get("warnings") or []
    )
    assert not [
        item for item in inventory
        if item.get("query_source") == "llm_source_planning_agent"
    ]


def test_source_critic_agent_mocked_assessment_annotates_registry(monkeypatch):
    def mock_assess(source_entry, collection_spec, screening_policy, source_role_policy):  # noqa: ARG001
        source_id = source_entry.get("source_id")
        risky = source_id == "src_cdc_reported_cases"
        return {
            "source_id": source_id,
            "proposed_source_role": "data_source",
            "proposed_screening_decision": "include",
            "credibility_level": "high",
            "credibility_reason": "Official public health source.",
            "expected_extractable_fields": ["cases", "deaths"],
            "semantic_leakage_risk": risky,
            "semantic_leakage_reason": "May cite reserved authority." if risky else "",
            "context_only_risk": False,
            "validation_candidate_risk": risky,
            "needs_human_review": risky,
            "human_review_reason": "Review semantic leakage." if risky else "",
            "confidence": 0.84,
            "reasoning_summary": "Mocked source critic assessment.",
        }

    monkeypatch.setenv("ENABLE_LLM_SOURCE_CRITIC", "true")
    monkeypatch.setattr(
        source_screening_module, "assess_source_with_llm", mock_assess
    )

    result = build_graph().invoke(_initial_state())
    registry = _registry_by_id(result)
    entry = registry["src_cdc_reported_cases"]

    assert entry["llm_source_critic_enabled"] is True
    assert entry["llm_semantic_leakage_risk"] is True
    assert entry["llm_needs_human_review"] is True
    assert entry["llm_reasoning_summary"] == "Mocked source critic assessment."
    assert entry.get("screening_decision")
    assert entry.get("source_role")
    assert result["source_routing_summary"]["llm_semantic_leakage_count"] == 1


def test_direct_collection_source_critic_keeps_ambiguous_high_value_candidates():
    registry = [
        {
            "source_id": "src_ny_target_week",
            "canonical_url": (
                "https://www.health.ny.gov/diseases/communicable/influenza/"
                "surveillance/2024-2025/archive/2024-11-02_flu_report.pdf"
            ),
            "title": "New York State Influenza Surveillance Report",
            "publisher": "New York State Department of Health",
            "discovery_method": "live_search_result",
            "source_type": "official_public_health_agency",
            "must_fetch": True,
        },
        {
            "source_id": "src_cdc_ambiguous_high_value",
            "canonical_url": "https://www.cdc.gov/fluview/surveillance/2025-week-04.html",
            "title": "CDC FluView 2025 Week 04",
            "publisher": "CDC",
            "discovery_method": "live_search_result",
            "source_type": "official_public_health_agency",
        },
        {
            "source_id": "src_cdc_background",
            "canonical_url": "https://www.cdc.gov/flu/about/index.html",
            "title": "About Influenza",
            "publisher": "CDC",
            "discovery_method": "live_search_result",
            "source_type": "official_public_health_agency",
            "source_role_final": "context",
            "data_product_type": "background_fact_sheet",
            "task_specificity": "generic",
        },
        {
            "source_id": "src_instagram",
            "canonical_url": "https://www.instagram.com/reel/DTeGGvXgj32/",
            "title": "Instagram reel",
            "publisher": "Instagram",
            "discovery_method": "live_search_result",
            "source_type": "social_media",
        },
    ]

    selected, skipped, summary = source_screening_module._select_llm_source_critic_candidates(
        registry,
        allowlist=None,
        max_sources=10,
        collection_mode="direct_collection",
    )

    assert selected == {"src_cdc_ambiguous_high_value"}
    assert skipped["src_ny_target_week"] == "target_official_must_fetch_skips_llm_source_critic"
    assert skipped["src_cdc_background"] == (
        "direct_low_value_context_skips_source_critic"
    )
    assert skipped["src_instagram"] == (
        "direct_low_value_context_skips_source_critic"
    )
    assert summary["direct_target_official_fast_path"] is True


def test_source_critic_node_uses_state_collection_mode_and_coverage_before_llm(monkeypatch):
    calls: list[str] = []

    def mock_assess(source_entry, collection_spec, screening_policy, source_role_policy):  # noqa: ARG001
        calls.append(str(source_entry.get("source_id")))
        return {
            "source_id": source_entry.get("source_id"),
            "proposed_source_role": "data_source",
            "proposed_screening_decision": "include",
            "credibility_level": "high",
            "credibility_reason": "Mocked source critic assessment.",
            "expected_extractable_fields": ["tests_positive"],
            "semantic_leakage_risk": False,
            "context_only_risk": False,
            "validation_candidate_risk": False,
            "needs_human_review": False,
            "confidence": 0.8,
            "reasoning_summary": "Mocked.",
        }

    monkeypatch.setenv("ENABLE_LLM_SOURCE_CRITIC", "true")
    monkeypatch.setenv("ENABLE_LLM_SOURCE_IDENTITY", "false")
    monkeypatch.setattr(
        source_screening_module, "assess_source_with_llm", mock_assess
    )
    state = {
        "structured_task": {
            "disease": "FLU",
            "location": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        "collection_spec": {
            "disease": "FLU",
            "geography": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        "source_registry": [
            {
                "source_id": "src_cdc_week_40_2024",
                "canonical_url": "https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
                "title": "CDC FluView Week 40, 2024",
                "publisher": "CDC",
                "source_type": "official_public_health_agency",
                "status": "screened",
                "screening_decision": "include",
                "source_role": "data_source",
                "final_screening_decision": "include_for_content_fetch",
                "ready_for_content_fetch": True,
            },
            {
                "source_id": "src_cdc_week_40_2025",
                "canonical_url": "https://www.cdc.gov/fluview/surveillance/2025-week-40.html",
                "title": "CDC FluView Week 40, 2025",
                "publisher": "CDC",
                "source_type": "official_public_health_agency",
                "status": "screened",
                "screening_decision": "include",
                "source_role": "data_source",
                "final_screening_decision": "include_for_content_fetch",
                "ready_for_content_fetch": True,
            },
        ],
        "human_review_queue": [],
        "collection_trace": [],
    }

    result = source_screening_module.source_critic_and_uncertainty_routing(state)

    assert calls == ["src_cdc_week_40_2025"]
    assert result["source_coverage_audit"]["coverage_status"] in {
        "target_official_source_discovered_not_fetched",
        "target_official_source_parsed",
    }
    assert result["source_critic_selection_summary"]["collection_mode"] == "direct_collection"
    assert result["source_critic_selection_summary"]["direct_target_official_fast_path"] is True
    assert result["direct_fast_path_summary"]["target_source_count"] == 1
    assert result["direct_fast_path_summary"]["critic_skipped_source_count"] >= 1


def test_direct_collection_verified_target_skips_identity_and_credibility_llm(monkeypatch):
    identity_calls: list[str] = []
    credibility_calls: list[str] = []

    def fake_identity(**kwargs):
        identity_calls.append(str(kwargs["source_entry"].get("source_id")))
        return {}

    def fake_credibility(**kwargs):
        credibility_calls.append(str(kwargs.get("user_prompt") or ""))
        raise AssertionError("verified target sources should not call credibility LLM")

    monkeypatch.setenv("ENABLE_LLM_SOURCE_IDENTITY", "true")
    monkeypatch.setenv("ENABLE_LLM_SOURCE_CREDIBILITY", "true")
    monkeypatch.setenv("ENABLE_LLM_SOURCE_CRITIC", "true")
    monkeypatch.setattr(
        "data_collection_workflow.source_identity.assess_source_identity_with_llm",
        fake_identity,
    )
    monkeypatch.setattr(
        "data_collection_workflow.source_credibility.llm_clients.run_pydantic_structured_llm",
        fake_credibility,
    )
    state = {
        "structured_task": {
            "disease": "FLU",
            "location": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        "collection_spec": {
            "disease": "FLU",
            "geography": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        "source_registry": [
            {
                "source_id": "src_cdc_week_40_2024",
                "canonical_url": "https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
                "title": "CDC FluView Week 40, 2024",
                "publisher": "CDC",
                "source_type": "official_public_health_agency",
                "status": "screened",
                "screening_decision": "include",
                "source_role": "data_source",
                "final_screening_decision": "include_for_content_fetch",
                "ready_for_content_fetch": True,
                "must_fetch": True,
                "coverage_requirement_ids": ["req_cdc_week_40"],
            },
            {
                "source_id": "src_context_news",
                "canonical_url": "https://example-news.test/flu-week-40-context",
                "title": "News context about flu season",
                "publisher": "Example News",
                "source_type": "news_media",
                "status": "screened",
                "screening_decision": "include",
                "source_role": "context_source",
                "source_role_final": "context_source",
                "triage_role": "context_only",
                "target_fit_status": "context_only",
                "final_screening_decision": "include_for_context_only",
                "ready_for_content_fetch": False,
            }
        ],
        "human_review_queue": [],
        "collection_trace": [],
    }

    result = source_screening_module.source_critic_and_uncertainty_routing(state)

    assert identity_calls == []
    assert credibility_calls == []
    fast_path = result["direct_fast_path_summary"]
    assert fast_path["identity_llm_assessed_source_count"] == 0
    assert fast_path["source_credibility_llm_assessed_count"] == 0
    assert fast_path["identity_llm_skipped_reason_counts"] == {
        "direct_target_official_fast_path_skips_source_identity": 2
    }
    assert fast_path["credibility_llm_skipped_reason_counts"] == {
        "direct_target_official_fast_path_skips_source_credibility": 2
    }


def test_llm_source_critic_cannot_override_validation_reserved(monkeypatch):
    def mock_assess(source_entry, collection_spec, screening_policy, source_role_policy):  # noqa: ARG001
        return {
            "source_id": source_entry.get("source_id"),
            "proposed_source_role": "data_source",
            "proposed_screening_decision": "include",
            "credibility_level": "high",
            "credibility_reason": "Mock says collect it.",
            "expected_extractable_fields": ["cases", "deaths"],
            "semantic_leakage_risk": False,
            "semantic_leakage_reason": "",
            "context_only_risk": False,
            "validation_candidate_risk": False,
            "needs_human_review": False,
            "human_review_reason": "",
            "confidence": 0.9,
            "reasoning_summary": "Mock tries to keep source collectable.",
        }

    monkeypatch.setenv("COLLECTION_MODE", "masked_validation")
    monkeypatch.setenv("ENABLE_LLM_SOURCE_CRITIC", "true")
    monkeypatch.setattr(
        source_screening_module, "assess_source_with_llm", mock_assess
    )

    result = build_graph().invoke(_initial_state())
    entry = _registry_by_id(result)["src_cdc_reported_cases"]

    assert entry["llm_proposed_source_role"] == "data_source"
    assert entry["source_role"] == "validation_reserved"
    assert entry["final_screening_decision"] == "reserved_for_validation"
    assert entry["ready_for_content_fetch"] is False
    assert {"validation_reserved", "blocked_from_collection"} <= set(
        entry.get("routing_flags") or []
    )


def test_llm_source_critic_cannot_override_context_only(monkeypatch, tmp_path):
    seed_overlay, policy_overlay = write_source_role_overlays(tmp_path)
    def mock_assess(source_entry, collection_spec, screening_policy, source_role_policy):  # noqa: ARG001
        return {
            "source_id": source_entry.get("source_id"),
            "proposed_source_role": "data_source",
            "proposed_screening_decision": "include",
            "credibility_level": "medium",
            "credibility_reason": "Mock says source has data.",
            "expected_extractable_fields": ["cases", "date", "location"],
            "semantic_leakage_risk": False,
            "semantic_leakage_reason": "",
            "context_only_risk": False,
            "validation_candidate_risk": False,
            "needs_human_review": False,
            "human_review_reason": "",
            "confidence": 0.88,
            "reasoning_summary": "Mock tries to override context-only.",
        }

    monkeypatch.setenv("COLLECTION_MODE", "masked_validation")
    monkeypatch.setenv("SEED_SOURCE_OVERLAY_PATH", str(seed_overlay))
    monkeypatch.setenv("SOURCE_ROLE_POLICY_OVERLAY_PATH", str(policy_overlay))
    monkeypatch.setenv("ENABLE_LLM_SOURCE_CRITIC", "true")
    monkeypatch.setattr(
        source_screening_module, "assess_source_with_llm", mock_assess
    )

    result = build_graph().invoke(_initial_state())
    registry = _registry_by_id(result)
    vdh = registry[_CONTEXT]
    who = registry[_VALIDATION]

    assert vdh["llm_proposed_source_role"] == "data_source"
    assert vdh["source_role"] == "context_source"
    assert vdh["final_screening_decision"] == "include_for_context_fetch"
    assert vdh["ready_for_content_fetch"] is True
    assert {"context_only", "blocked_from_structured_extraction"} <= set(
        vdh.get("routing_flags") or []
    )
    assert who["source_role"] == "validation_reserved"
