"""Optional LLM advisory agent for source planning.

This module performs no search and no fetching. It formats workflow context for
the central LLM helper and normalizes the returned JSON so the deterministic
query strategy node can safely consume it behind a feature flag.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from .. import llm_clients

_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "source_planning_agent_prompt.json"
)


class SourcePlanningQuery(BaseModel):
    query: str
    source_type: str = ""
    priority: int = 5
    rationale: str = ""
    expected_fields: list[str] = Field(default_factory=list)


class SourcePlanningCandidateHint(BaseModel):
    hint: str | None = None
    url: str | None = None
    title: str | None = None
    publisher: str | None = None
    source_type: str | None = None
    proposed_role: str | None = None
    rationale: str | None = None


class SourcePlanningOutput(BaseModel):
    agent_name: str = "source_planning_agent"
    agent_version: str = "0.3"
    disease: str = ""
    geography: str | None = None
    time_window: str | None = None
    target_fields: list[str] = Field(default_factory=list)
    source_categories: list[str] = Field(default_factory=list)
    proposed_search_queries: list[SourcePlanningQuery] = Field(default_factory=list)
    proposed_collection_source_types: list[str] = Field(default_factory=list)
    proposed_validation_source_types: list[str] = Field(default_factory=list)
    proposed_context_source_types: list[str] = Field(default_factory=list)
    candidate_source_hints: list[SourcePlanningCandidateHint] = Field(
        default_factory=list
    )
    reasoning_summary: str = ""
    human_review_recommended: bool = False
    warnings: list[str] = Field(default_factory=list)


def _load_prompt_text() -> str:
    with _PROMPT_PATH.open("r", encoding="utf-8") as f:
        policy = json.load(f)
    skeleton = policy.get("response_skeleton") or {}
    return "\n".join(
        [
            str(policy.get("system_prompt") or ""),
            "",
            "Rules:",
            *[f"- {rule}" for rule in policy.get("rules") or []],
            "",
            "Required JSON fields:",
            ", ".join(policy.get("required_fields") or []),
            "",
            "Compact JSON skeleton example:",
            json.dumps(skeleton, ensure_ascii=True, indent=2),
        ]
    ).strip()


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_str_list(value) -> list[str]:
    return [str(item).strip() for item in _as_list(value) if str(item).strip()]


def _target_fields(collection_spec: dict | None, collection_schema: dict | None) -> list[str]:
    spec_fields = _as_str_list((collection_spec or {}).get("required_fields"))
    if spec_fields:
        return spec_fields
    schema_fields = []
    for field in (collection_schema or {}).get("core_fields") or []:
        if isinstance(field, dict) and field.get("name"):
            schema_fields.append(str(field["name"]))
    return schema_fields


def _source_category_names(source_strategy: dict | None) -> list[str]:
    names = []
    for category in (source_strategy or {}).get("source_categories") or []:
        if isinstance(category, dict) and category.get("source_type"):
            names.append(str(category["source_type"]))
    return names


def _compact_collection_spec(collection_spec: dict | None) -> dict:
    spec = collection_spec or {}
    return {
        "disease": spec.get("disease"),
        "geography": spec.get("geography"),
        "time_window": spec.get("time_window"),
        "target_population": spec.get("target_population"),
        "data_focus": spec.get("data_focus"),
        "required_fields": _as_str_list(spec.get("required_fields")),
        "source_priority": _as_str_list(spec.get("source_priority")),
    }


def _compact_disease_profile(disease_profile: dict | None) -> dict:
    profile = disease_profile or {}
    return {
        "disease_standard_name": profile.get("disease_standard_name"),
        "include_terms": _as_str_list(profile.get("include_terms")),
        "syndrome_terms": _as_str_list(profile.get("syndrome_terms")),
        "virus_terms": _as_str_list(profile.get("virus_terms")),
        "target_population": profile.get("target_population"),
    }


def _compact_collection_schema(collection_schema: dict | None) -> dict:
    return {
        "field_names": _target_fields(None, collection_schema),
    }


def _compact_source_strategy(source_strategy: dict | None) -> dict:
    categories = []
    for category in (source_strategy or {}).get("source_categories") or []:
        if not isinstance(category, dict):
            continue
        source_type = str(category.get("source_type") or "").strip()
        if not source_type:
            continue
        categories.append(
            {
                "source_type": source_type,
                "priority": category.get("priority"),
            }
        )
    return {"source_categories": categories}


def _seed_source_id(source: dict) -> str:
    return str(
        source.get("source_id")
        or source.get("seed_source_id")
        or ""
    ).strip()


def _current_run_sources(source_strategy: dict | None) -> list[dict]:
    """Only caller-supplied current-run sources may inform advisory planning."""
    sources = (source_strategy or {}).get("current_run_sources") or []
    return [source for source in sources if isinstance(source, dict)] if isinstance(sources, list) else []


def _known_seed_source_summary(source_strategy: dict | None = None) -> list[dict]:
    # Keep the payload field compatible while removing implicit case catalogs.
    return [
        {
            "source_id": _seed_source_id(source),
            **{key: source.get(key) for key in (
                "title", "publisher", "source_type", "source_purpose", "notes"
            )},
        }
        for source in _current_run_sources(source_strategy)
    ]


def _known_seed_urls(source_strategy: dict | None = None) -> set[str]:
    return {str(source["url"]).strip().rstrip("/")
            for source in _current_run_sources(source_strategy) if source.get("url")}


def _looks_like_url(value: str) -> bool:
    parts = urlsplit(value)
    return parts.scheme in {"http", "https"} and bool(parts.netloc)


def _normalize_query(item, index: int, default_fields: list[str]) -> dict | None:
    if isinstance(item, str):
        query = item.strip()
        source_type = "news_and_situation_report"
        rationale = "LLM source planning agent proposed this search query."
        expected_fields = default_fields
        priority = 5
    elif isinstance(item, dict):
        query = str(item.get("query") or item.get("search_query") or "").strip()
        source_type = str(
            item.get("source_type")
            or item.get("proposed_source_type")
            or "news_and_situation_report"
        ).strip()
        rationale = str(
            item.get("rationale")
            or item.get("reason")
            or "LLM source planning agent proposed this search query."
        ).strip()
        expected_fields = _as_str_list(item.get("expected_fields")) or default_fields
        try:
            priority = int(item.get("priority") or 5)
        except (TypeError, ValueError):
            priority = 5
    else:
        return None
    if not query:
        return None
    return {
        "query_id": f"q_agent_{index:03d}",
        "query": query,
        "source_type": source_type or "news_and_situation_report",
        "priority": priority,
        "rationale": rationale,
        "expected_fields": expected_fields,
        "query_source": "llm_source_planning_agent",
        "discovery_method": "llm_source_planning_agent",
    }


def _normalize_candidate_hint(item, known_urls: set[str]) -> dict | None:
    if isinstance(item, str):
        value = item.strip()
        if not value:
            return None
        hint = {"url": value} if _looks_like_url(value) else {"hint": value}
    elif isinstance(item, dict):
        hint = dict(item)
    else:
        return None
    url = str(hint.get("url") or "").strip().rstrip("/")
    hint["agent_proposed_unverified"] = bool(url and url not in known_urls)
    if url:
        hint["url"] = url
    if "requires_human_review" not in hint:
        hint["requires_human_review"] = True
    return hint


def _normalize_plan(
    raw: dict,
    collection_spec: dict | None,
    disease_profile: dict | None,
    source_strategy: dict | None,
    collection_schema: dict | None,
    retry_attempted: bool = False,
    retry_succeeded: bool = False,
    failure_type: str | None = None,
    failure_message: str | None = None,
    structured_output_attempted: bool = True,
) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected source planning JSON object, got {type(raw)!r}.")

    target_fields = _target_fields(collection_spec, collection_schema)
    categories = _source_category_names(source_strategy)

    proposed_queries: list[dict] = []
    for item in _as_list(raw.get("proposed_search_queries")):
        normalized = _normalize_query(item, len(proposed_queries) + 1, target_fields)
        if normalized:
            proposed_queries.append(normalized)

    known_urls = _known_seed_urls(source_strategy)
    candidate_hints = [
        hint for hint in (
            _normalize_candidate_hint(item, known_urls)
            for item in _as_list(raw.get("candidate_source_hints"))
        )
        if hint
    ]

    spec = collection_spec or {}
    profile = disease_profile or {}
    return {
        "agent_name": str(raw.get("agent_name") or "source_planning_agent"),
        "agent_version": str(raw.get("agent_version") or "0.1"),
        "disease": str(
            raw.get("disease")
            or spec.get("disease")
            or profile.get("disease_standard_name")
            or ""
        ),
        "geography": raw.get("geography") or spec.get("geography"),
        "time_window": raw.get("time_window") or spec.get("time_window"),
        "target_fields": _as_str_list(raw.get("target_fields")) or target_fields,
        "source_categories": _as_str_list(raw.get("source_categories")) or categories,
        "proposed_search_queries": proposed_queries,
        "proposed_collection_source_types": _as_str_list(
            raw.get("proposed_collection_source_types")
        ),
        "proposed_validation_source_types": _as_str_list(
            raw.get("proposed_validation_source_types")
        ),
        "proposed_context_source_types": _as_str_list(
            raw.get("proposed_context_source_types")
        ),
        "candidate_source_hints": candidate_hints,
        "reasoning_summary": str(raw.get("reasoning_summary") or "").strip(),
        "human_review_recommended": bool(raw.get("human_review_recommended", False)),
        "warnings": _as_str_list(raw.get("warnings")),
        "structured_output_mode": str(
            raw.get("_structured_output_mode")
            or raw.get("structured_output_mode")
            or "unknown"
        ),
        "structured_output_attempted": structured_output_attempted,
        "retry_attempted": retry_attempted,
        "retry_succeeded": retry_succeeded,
        "failure_type": failure_type,
        "failure_message": failure_message,
    }


def _build_user_payload(
    user_request: str,
    collection_spec: dict | None,
    disease_profile: dict | None,
    source_strategy: dict | None,
    collection_schema: dict | None,
) -> dict:
    return {
        "user_request": user_request,
        "collection_spec_summary": _compact_collection_spec(collection_spec),
        "disease_profile_summary": _compact_disease_profile(disease_profile),
        "collection_schema_summary": _compact_collection_schema(collection_schema),
        "source_strategy_summary": _compact_source_strategy(source_strategy),
        "known_seed_source_summary": _known_seed_source_summary(source_strategy),
        "network_policy": "Do not perform broad web search or fetch URLs.",
        "instruction": (
            "Return a source planning JSON object for this task. If exact URLs "
            "are uncertain, provide search queries and source categories."
        ),
    }


def _build_retry_user_prompt(
    user_request: str,
    collection_spec: dict | None,
    disease_profile: dict | None,
    source_strategy: dict | None,
) -> str:
    payload = {
        "instruction": (
            "Return a minimal source planning JSON object. Include only these "
            "fields if needed: agent_name, proposed_search_queries, "
            "proposed_collection_source_types, proposed_validation_source_types, "
            "proposed_context_source_types, reasoning_summary, warnings. Return "
            "empty arrays when uncertain; do not return an empty response."
        ),
        "user_request": user_request,
        "collection_spec_summary": _compact_collection_spec(collection_spec),
        "disease_profile_summary": _compact_disease_profile(disease_profile),
        "source_strategy_summary": _compact_source_strategy(source_strategy),
        "network_policy": "Do not perform broad web search or fetch URLs.",
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _should_retry_structured_error(exc: Exception) -> bool:
    message = str(exc).strip().lower()
    retry_markers = (
        "llm returned empty output",
        "jsondecodeerror",
        "expecting",
        "validation",
        "structured",
        "fallback_json_error",
    )
    return any(marker in message for marker in retry_markers)


def plan_sources_with_llm(
    user_request: str,
    collection_spec: dict | None,
    disease_profile: dict | None,
    source_strategy: dict | None,
    collection_schema: dict | None,
) -> dict:
    """Return a normalized advisory source plan from the configured LLM."""

    system_prompt = _load_prompt_text()
    user_payload = _build_user_payload(
        user_request=user_request,
        collection_spec=collection_spec,
        disease_profile=disease_profile,
        source_strategy=source_strategy,
        collection_schema=collection_schema,
    )
    first_structured_exc: Exception | None = None
    try:
        raw = llm_clients.run_pydantic_structured_llm(
            system_prompt=system_prompt,
            user_prompt=json.dumps(user_payload, ensure_ascii=True, indent=2),
            schema_model=SourcePlanningOutput,
            temperature=0.0,
        )
        return _normalize_plan(
            raw,
            collection_spec=collection_spec,
            disease_profile=disease_profile,
            source_strategy=source_strategy,
            collection_schema=collection_schema,
        )
    except Exception as first_exc:
        if not _should_retry_structured_error(first_exc):
            raise
        first_structured_exc = first_exc

    try:
        raw = llm_clients.run_pydantic_structured_llm(
            system_prompt=system_prompt,
            user_prompt=_build_retry_user_prompt(
                user_request=user_request,
                collection_spec=collection_spec,
                disease_profile=disease_profile,
                source_strategy=source_strategy,
            ),
            schema_model=SourcePlanningOutput,
            temperature=0.0,
        )
        return _normalize_plan(
            raw,
            collection_spec=collection_spec,
            disease_profile=disease_profile,
            source_strategy=source_strategy,
            collection_schema=collection_schema,
            retry_attempted=True,
            retry_succeeded=True,
            failure_type=type(first_structured_exc).__name__,
            failure_message=f"Initial attempt failed: {first_structured_exc}",
        )
    except Exception as retry_exc:
        raise ValueError(
            f"{first_structured_exc} Retry failed: {retry_exc}"
        ) from retry_exc
