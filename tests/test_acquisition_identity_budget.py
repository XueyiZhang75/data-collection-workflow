"""Offline identity-budget recovery must preserve discovered sources and boundaries."""
import importlib
import socket

import pytest

from data_collection_workflow.models import SourceCandidate
from data_collection_workflow.source_identity import apply_source_identity_to_registry
from data_collection_workflow.workflow_recovery import RecoveryAction, RecoveryPlan, execute_recovery, merge_recovery_delta
from data_collection_workflow.session_runtime import BudgetExceeded, RunContext


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    for key in ("ENABLE_LLM_SOURCE_CRITIC", "ENABLE_LLM_SOURCE_CREDIBILITY", "ENABLE_LLM_EXTRACTION", "ENABLE_LIVE_SEARCH", "ENABLE_LIVE_FETCH"):
        monkeypatch.setenv(key, "false")
    monkeypatch.setenv("ENABLE_LLM_SOURCE_IDENTITY", "true")
    monkeypatch.setenv("LLM_SOURCE_IDENTITY_REQUIRE_LLM", "true")
    monkeypatch.setenv("LLM_SOURCE_IDENTITY_ALLOW_DETERMINISTIC_FALLBACK", "false")
    monkeypatch.delenv("LLM_SOURCE_IDENTITY_MAX_SOURCES", raising=False)
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: pytest.fail("network forbidden"))


def entry(source_id="new", **changes):
    return {"source_id": source_id, "canonical_url": f"https://unknown.example/{source_id}",
            "url": f"https://unknown.example/{source_id}", "title": "Measles Canada 2025 surveillance",
            "snippet": "Canada reported 12 confirmed measles cases during 2025.",
            "source_type": "official_site_search", "search_provider": "fixture",
            "discovery_method": "live_search_result", **changes}


def state(rows=None):
    return {"structured_task": {"disease": "measles", "location": "Canada", "start_date": "2025-01-01", "end_date": "2025-12-31"},
            "collection_spec": {"disease": "measles", "geography": "Canada", "time_window": "2025", "collection_mode": "direct_collection"},
            "source_registry": list(rows or []), "collection_trace": [], "human_review_queue": []}


def fail_identity(monkeypatch, error):
    module = importlib.import_module("data_collection_workflow.source_identity")
    def unavailable(**kwargs):
        raise error
    monkeypatch.setattr(module, "assess_source_identity_with_llm", unavailable)


def test_evidence_identity_budget_is_unverified_not_required_llm_block(monkeypatch):
    fail_identity(monkeypatch, BudgetExceeded("identity stage exhausted", kind="model:SourceIdentityAgentOutput"))
    rows, assessments, summary = apply_source_identity_to_registry([entry()], state()["collection_spec"], llm_enabled=True,
        require_llm=True, allow_deterministic_fallback=False)
    assert assessments[0]["source_identity_status"] != "blocked_llm_required"
    assert rows[0]["source_identity_unverified"] is True
    assert rows[0]["source_identity_llm_used"] is False
    assert "official" not in rows[0]["source_type_final"]
    assert any("budget" in warning for warning in assessments[0]["warnings"])


@pytest.mark.parametrize("revision,error", [("legacy", BudgetExceeded("full", kind="model:SourceIdentityAgentOutput")),
                                            ("evidence", TimeoutError("provider unavailable"))])
def test_required_identity_non_budget_or_legacy_semantics_remain(monkeypatch, revision, error):
    monkeypatch.setenv("PIPELINE_MODE", revision)
    fail_identity(monkeypatch, error)
    _, assessments, _ = apply_source_identity_to_registry([entry()], state()["collection_spec"], llm_enabled=True,
        require_llm=True, allow_deterministic_fallback=False)
    assert assessments[0]["source_identity_status"] == "blocked_llm_required"


def setup_search(monkeypatch, tmp_path, *, failure, old=None):
    discovery = importlib.import_module("data_collection_workflow.nodes.source_discovery")
    identity = importlib.import_module("data_collection_workflow.source_identity")
    content = importlib.import_module("data_collection_workflow.nodes.content_processing")
    ctx = RunContext(tmp_path, {"pipeline_mode": "evidence", "universal": {
        "budget_policy": {"version": 2, "mode": "adaptive", "soft_source_target": 50},
        "budget_limits": {"search": 1, "search_results": 10, "source_targets": 2, "http_requests": 3,
                          "model:SourceIdentityAgentOutput": 0}}})
    searches, fetched = [], []
    candidates = [SourceCandidate(**entry())]
    if old:
        candidates.append(SourceCandidate(**entry("alias", url=old["canonical_url"], canonical_url=old["canonical_url"])))
    def search(**kwargs):
        def provider():
            searches.append("search")
            return [candidate.model_dump() for candidate in candidates]
        rows = ctx.call("search", {"query": "measles Canada 2025 report"}, provider)
        kwargs["totals"]["executed_query_count"] += 1
        return [SourceCandidate(**row) for row in rows], [], []
    monkeypatch.setattr(discovery, "_execute_iterative_query_batch", search)
    monkeypatch.setattr(discovery, "_provider_for_settings", lambda _: None)
    def identity_call(**kwargs):
        if failure == "provider":
            raise TimeoutError("identity provider unavailable")
        return ctx.call("model:SourceIdentityAgentOutput", {"source_id": kwargs["source_entry"]["source_id"]},
                        lambda: pytest.fail("exhausted model must never dispatch"))
    monkeypatch.setattr(identity, "assess_source_identity_with_llm", identity_call)
    def fetch(local):
        fetched.extend(local.get("source_registry") or [])
        return {"documents": []}
    monkeypatch.setattr(content, "content_fetch_and_parse", fetch)
    monkeypatch.setattr(content, "document_quality_check", lambda _: {})
    monkeypatch.setattr(content, "evidence_chunking_and_data_presence_flagging", lambda _: {"evidence_chunks": []})
    action = RecoveryAction("search", "official_surveillance", "search-action", "independent evidence",
                            query={"query": "measles Canada 2025 report", "source_type": "official_site_search"})
    return ctx, RecoveryPlan([action]), state([old] if old else []), searches, fetched


@pytest.mark.parametrize("failure", ["budget", "provider"])
def test_successful_search_survives_identity_failure_and_cache_retry(monkeypatch, tmp_path, failure):
    ctx, plan, initial, searches, fetched = setup_search(monkeypatch, tmp_path, failure=failure)
    with ctx.activate():
        first = execute_recovery(plan, context=ctx, artifacts=initial, budget=ctx.ledger)
        merged = {**initial, **merge_recovery_delta(initial, first)}
        second = execute_recovery(plan, context=ctx, artifacts=merged, budget=ctx.ledger)
        merged = {**merged, **merge_recovery_delta(merged, second)}
    assert len(merged["source_registry"]) == 1, first.actions
    row = merged["source_registry"][0]
    assert row["source_id"] == "new" and row["snippet"] == entry()["snippet"]
    assert row.get("source_identity_unverified") is True
    assert not row.get("source_identity_llm_used")
    assert searches == ["search"]
    snap = ctx.ledger.snapshot()
    assert snap["used"]["search"] == 1
    assert snap["limits"]["model:SourceIdentityAgentOutput"] == 0
    if failure == "budget":
        assert first.actions[0]["status"] == "completed", first.actions
        assert fetched and fetched[0]["ready_for_content_fetch"]
        assert row.get("source_identity_status") != "blocked_llm_required"
    else:
        assert first.actions[0]["status"] == "failed"
        assert first.actions[0]["error"] in row.get("source_identity_errors", [])
        assert row.get("source_identity_warnings")
        assert not fetched


def test_discovery_alias_does_not_replace_fetched_explicit_exclusion(monkeypatch, tmp_path):
    old = entry("old", target_fit_status="wrong_disease", task_fit_evidence_origin="fetched_content",
                source_identity_unverified=False, final_screening_decision="exclude", blocked_from_fetch=True,
                blocked_from_fetch_reason="user_excluded", content_hash="saved-raw-hash")
    ctx, plan, initial, _, _ = setup_search(monkeypatch, tmp_path, failure="provider", old=old)
    with ctx.activate():
        delta = execute_recovery(plan, context=ctx, artifacts=initial, budget=ctx.ledger)
    merged = {**initial, **merge_recovery_delta(initial, delta)}
    assert any(row["source_id"] == "new" for row in merged["source_registry"])
    retained = next(row for row in merged["source_registry"] if row["source_id"] == "old")
    assert retained == old
    assert not any(row["source_id"] == "alias" for row in merged["source_registry"])


def test_later_identity_assessment_replaces_provisional_discovery_only(monkeypatch, tmp_path):
    ctx, plan, initial, _, _ = setup_search(monkeypatch, tmp_path, failure="provider")
    identity = importlib.import_module("data_collection_workflow.source_identity")
    with ctx.activate():
        first = execute_recovery(plan, context=ctx, artifacts=initial, budget=ctx.ledger)
        merged = {**initial, **merge_recovery_delta(initial, first)}
        assert len(merged["source_registry"]) == 1
        assert merged["source_registry"][0]["source_identity_status"] == "pending_assessment"
        monkeypatch.setattr(identity, "assess_source_identity_with_llm", lambda **_: {"llm_used": False})
        second = execute_recovery(plan, context=ctx, artifacts=merged, budget=ctx.ledger)
        final = {**merged, **merge_recovery_delta(merged, second)}
    assert len(final["source_registry"]) == 1
    assert final["source_registry"][0].get("source_identity_status") == "assessed"
    assert final["source_registry"][0].get("ready_for_content_fetch") is True


@pytest.mark.parametrize("reason", ["wrong_disease", "wrong_country", "wrong_period"])
def test_retained_discovery_cannot_replace_body_contradiction_without_user_flag(monkeypatch, tmp_path, reason):
    old = entry("old", target_fit_status=reason, task_fit_evidence_origin="fetched_content",
                source_identity_unverified=False, source_identity_status="assessed",
                final_screening_decision="include_for_context_fetch", content_hash="body-hash")
    ctx, plan, initial, _, _ = setup_search(monkeypatch, tmp_path, failure="provider", old=old)
    with ctx.activate():
        delta = execute_recovery(plan, context=ctx, artifacts=initial, budget=ctx.ledger)
    merged = {**initial, **merge_recovery_delta(initial, delta)}
    assert next(row for row in merged["source_registry"] if row["source_id"] == "old") == old
    assert initial["source_registry"][0] == old


def test_upgrading_pending_discovery_does_not_mutate_previous_checkpoint_state():
    from data_collection_workflow.workflow_recovery import RecoveryDelta
    pending = entry(source_identity_status="pending_assessment", source_identity_unverified=True)
    assessed = {**pending, "source_identity_status": "assessed", "ready_for_content_fetch": True}
    original = state([pending])
    result = merge_recovery_delta(original, RecoveryDelta(sources=[assessed]))
    assert original["source_registry"][0]["source_identity_status"] == "pending_assessment"
    assert result["source_registry"][0]["source_identity_status"] == "assessed"


@pytest.mark.parametrize("reason", ["wrong_disease", "wrong_country", "user_excluded"])
def test_cached_discovery_alias_never_reopens_existing_fetch_boundary(monkeypatch, tmp_path, reason):
    old = entry("old", target_fit_status=reason, task_fit_evidence_origin="fetched_content",
                source_identity_unverified=False, source_identity_status="assessed",
                final_screening_decision="exclude", blocked_from_fetch=True, blocked_from_fetch_reason=reason)
    ctx, plan, initial, _, fetched = setup_search(monkeypatch, tmp_path, failure="budget", old=old)
    with ctx.activate():
        delta = execute_recovery(plan, context=ctx, artifacts=initial, budget=ctx.ledger)
    assert delta.actions[0]["status"] == "completed", delta.actions
    duplicate_rows = [row for row in fetched if row.get("canonical_url") == old["canonical_url"]]
    assert duplicate_rows == [old]


def test_failed_recovery_search_still_has_at_most_two_planned_attempts(monkeypatch):
    from data_collection_workflow import workflow_recovery as recovery
    monkeypatch.setattr(recovery, "_recovery_query", lambda *args: {
        "query": "measles Canada 2025 report", "search_direction": "official_surveillance"})
    initial = state()
    gaps = [recovery.RecoveryGap("source_missing", "annual", "missing independent evidence")]
    budget = {"remaining": {"search": 10, "search_results": 10, "fetch": 10, "fetch_ordinary": 10, "extraction": 10}}
    plan = recovery.plan_recovery(gaps, state=initial, budget=budget)
    assert len(plan.actions) == 1
    failed = {"action_id": plan.actions[0].action_id, "kind": "search", "status": "failed"}
    initial["recovery_action_history"] = [failed]
    assert len(recovery.plan_recovery(gaps, state=initial, budget=budget).actions) == 1
    initial["recovery_action_history"].append(dict(failed))
    assert not recovery.plan_recovery(gaps, state=initial, budget=budget).actions
