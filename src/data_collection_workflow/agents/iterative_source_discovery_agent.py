"""LLM advisory agent for bounded iterative source discovery.

This module does not search or fetch webpages. It asks the configured LLM to
plan query batches and refine them from compact search-result metadata. The
source_discovery node remains responsible for executing searches, validating
URLs, deduplicating results, and enforcing bounds.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from .. import llm_clients
from ..models import SearchIterationPlan, SearchRefinementDecision
from ..query_policy import universal_queries_enabled


_SYSTEM_PROMPT = """
You are planning source discovery for a public-health data collection workflow.

The user task includes disease, location, time window, and target fields. You
must decide what to search next based on the task and, for refinement, based on
current search-result metadata.

Rules:
- Do not browse.
- Do not fetch webpages.
- Do not invent source content.
- Do not treat query text as proof that a result is relevant.
- Do not directly insert source URLs into source candidates.
- You may propose search queries only.
- You may suggest expected source types, trust signals, and evidence patterns.
- Consider local language, local agency naming, national/regional reporting
  structures, and disease-specific terminology when relevant.
- Explain why the current result set is sufficient or insufficient.
- Stop when additional search is unlikely to improve evidence coverage or when
  limits are reached.
- Return structured JSON only.

Do not use fixed source quotas or hard-coded jurisdiction-specific domain
rules. Prefer sources likely to contain authoritative, task-relevant evidence
and justify the task-specific search direction.
""".strip()


def _model_dump(value) -> dict:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return dict(value)
    raise ValueError(f"Expected structured LLM JSON object, got {type(value)!r}.")


def validate_refinement_advice(value: dict) -> dict:
    """Reject incomplete control advice instead of trusting parser defaults."""
    if not value.get('decision') or not str(value.get('decision_reason') or '').strip():
        raise ValueError('refinement advice requires an explicit decision and reason')
    if not isinstance(value.get('next_query_batch'), list):
        raise ValueError('refinement advice requires an explicit next_query_batch list')
    if value['decision'] == 'continue_search' and not value['next_query_batch']:
        raise ValueError('continue_search advice requires a nonempty query batch')
    return value


def _compact_task(
    user_request: str,
    structured_task: dict | None,
    collection_spec: dict | None,
) -> dict:
    structured = structured_task or {}
    spec = collection_spec or {}
    return {
        "user_request": user_request,
        "disease": structured.get("disease") or spec.get("disease"),
        "location": (
            structured.get("location")
            or spec.get("geography")
            or spec.get("location")
        ),
        "start_date": structured.get("start_date") or spec.get("start_date"),
        "end_date": structured.get("end_date") or spec.get("end_date"),
        "time_window": spec.get("time_window"),
        "target_fields": structured.get("target_fields")
        or spec.get("required_fields")
        or [],
    }


def plan_initial_search_iteration(
    *,
    user_request: str,
    structured_task: dict | None,
    collection_spec: dict | None,
    planned_queries: list[dict],
    search_query_inventory: list[dict],
    limits: dict,
) -> dict:
    """Ask the LLM for the first bounded search iteration plan."""

    payload = {
        "instruction": (
            "Plan the first search iteration. You may reuse/refine upstream "
            "planned queries or propose new query text. Do not provide source "
            "candidates or page content."
        ),
        "task": _compact_task(user_request, structured_task, collection_spec),
        "upstream_planned_queries": planned_queries,
        "search_query_inventory": search_query_inventory,
        "bounds": limits,
    }
    raw = llm_clients.run_pydantic_structured_llm(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=json.dumps(payload, ensure_ascii=True, indent=2),
        schema_model=SearchIterationPlan,
        temperature=0.0,
    )
    value = _model_dump(raw)
    if universal_queries_enabled() and not value.get('query_batch'):
        raise ValueError('initial search advice requires a nonempty query batch')
    return SearchIterationPlan.model_validate(value).model_dump()


def refine_search_iteration(
    *,
    user_request: str,
    structured_task: dict | None,
    collection_spec: dict | None,
    previous_plans: list[dict],
    observations: list[dict],
    observation: dict,
    limits: dict,
) -> dict:
    """Ask the LLM to continue or stop after observing search metadata."""

    payload = {
        "instruction": (
            "Review the compact search metadata. Decide whether to continue "
            "searching or stop. If continuing, return the next bounded query "
            "batch. Do not invent facts from snippets."
        ),
        "task": _compact_task(user_request, structured_task, collection_spec),
        "previous_iteration_plans": previous_plans,
        "all_observations": observations,
        "current_observation": observation,
        "bounds": limits,
    }
    raw = llm_clients.run_pydantic_structured_llm(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=json.dumps(payload, ensure_ascii=True, indent=2),
        schema_model=SearchRefinementDecision,
        temperature=0.0,
    )
    value = _model_dump(raw)
    if universal_queries_enabled():
        validate_refinement_advice(value)
    return SearchRefinementDecision.model_validate(value).model_dump()
