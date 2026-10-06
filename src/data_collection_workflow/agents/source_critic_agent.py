"""Optional LLM advisory agent for source credibility and role critique."""

from __future__ import annotations

import json
from pathlib import Path

from .. import llm_clients

_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "source_critic_agent_prompt.json"
)


def _load_prompt_text() -> str:
    with _PROMPT_PATH.open("r", encoding="utf-8") as f:
        policy = json.load(f)
    return "\n".join(
        [
            str(policy.get("system_prompt") or ""),
            "",
            "Rules:",
            *[f"- {rule}" for rule in policy.get("rules") or []],
            "",
            "Required JSON fields:",
            ", ".join(policy.get("required_fields") or []),
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


def _confidence(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


_CRITIC_DECISION_ALIASES = {
    "exclude": "not_task_relevant",
    "excluded": "not_task_relevant",
    "not_relevant": "not_task_relevant",
    "context_only": "no_extractable_data",
}


def _normalize_assessment(raw: dict, source_entry: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected source critic JSON object, got {type(raw)!r}.")

    critic_decision = str(raw.get("critic_decision") or "").strip().lower()
    if not critic_decision:
        if bool(raw.get("validation_candidate_risk", False)):
            critic_decision = "suitable_for_validation"
        elif bool(raw.get("context_only_risk", False)):
            critic_decision = "suitable_for_context"
        elif str(raw.get("proposed_screening_decision") or "").lower() == "exclude":
            critic_decision = "not_task_relevant"
        else:
            critic_decision = "suitable_for_collection"
    critic_decision = _CRITIC_DECISION_ALIASES.get(
        critic_decision, critic_decision
    )
    recommended_role = str(
        raw.get("recommended_role") or raw.get("proposed_source_role") or ""
    ).strip()
    fetch_recommendation = str(
        raw.get("fetch_recommendation") or "allow_fetch"
    ).strip()
    review_required = bool(
        raw.get("review_required", False)
        or raw.get("needs_human_review", False)
    )

    return {
        "source_id": str(raw.get("source_id") or source_entry.get("source_id") or ""),
        "proposed_source_role": str(raw.get("proposed_source_role") or "").strip(),
        "proposed_screening_decision": str(
            raw.get("proposed_screening_decision") or ""
        ).strip(),
        "credibility_level": str(raw.get("credibility_level") or "unknown").strip(),
        "credibility_reason": str(raw.get("credibility_reason") or "").strip(),
        "expected_extractable_fields": _as_str_list(
            raw.get("expected_extractable_fields")
        ),
        "semantic_leakage_risk": bool(raw.get("semantic_leakage_risk", False)),
        "semantic_leakage_reason": str(
            raw.get("semantic_leakage_reason") or ""
        ).strip(),
        "context_only_risk": bool(raw.get("context_only_risk", False)),
        "validation_candidate_risk": bool(raw.get("validation_candidate_risk", False)),
        "needs_human_review": bool(raw.get("needs_human_review", False)),
        "human_review_reason": str(raw.get("human_review_reason") or "").strip(),
        "confidence": _confidence(raw.get("confidence")),
        "reasoning_summary": str(raw.get("reasoning_summary") or "").strip(),
        "critic_decision": critic_decision,
        "risk_flags": _as_str_list(raw.get("risk_flags")),
        "recommended_role": recommended_role,
        "fetch_recommendation": fetch_recommendation,
        "review_required": review_required,
        "warnings": _as_str_list(raw.get("warnings")),
    }


def assess_source_with_llm(
    source_entry: dict,
    collection_spec: dict | None,
    screening_policy: dict | None,
    source_role_policy: dict | None,
) -> dict:
    """Return a normalized advisory source assessment from the configured LLM."""

    system_prompt = _load_prompt_text()
    # Direct collection has no globally held-out publisher domains. Explicit
    # per-source user boundaries remain present in source_entry.
    role_policy = source_role_policy or {}
    from ..query_policy import universal_queries_enabled
    if universal_queries_enabled() and (collection_spec or {}).get("collection_mode") == "direct_collection":
        role_policy = {**role_policy, "validation_reserved_domains": [],
                       "validation_reserved_source_ids": []}
    user_payload = {
        "source_entry": source_entry,
        "collection_spec": collection_spec or {},
        "screening_policy": screening_policy or {},
        "source_role_policy_summary": {
            "collection_mode_env_var": role_policy.get(
                "enabled_env_var"
            ),
            "validation_reserved_source_ids": role_policy.get(
                "validation_reserved_source_ids"
            )
            or [],
            "validation_reserved_domains": role_policy.get(
                "validation_reserved_domains"
            )
            or [],
            "context_only_source_ids": role_policy.get(
                "context_only_source_ids"
            )
            or [],
        },
        "policy_boundary": (
            "Do not decide final policy. Deterministic guardrails outside the "
            "model enforce validation_reserved and context_only behavior."
        ),
    }
    raw = llm_clients.run_structured_llm_json(
        system_prompt=system_prompt,
        user_prompt=json.dumps(user_payload, ensure_ascii=True, indent=2),
        expected_schema_name="SourceCriticAgentOutput",
        temperature=0.0,
    )
    return _normalize_assessment(raw, source_entry)
