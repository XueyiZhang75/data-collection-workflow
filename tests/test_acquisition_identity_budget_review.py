"""Independent guards for the identity-budget fallback boundary."""
import pytest
from data_collection_workflow.source_identity import apply_source_identity_to_registry
from data_collection_workflow.session_runtime import BudgetExceeded
from test_acquisition_identity_budget import offline, entry, state, fail_identity


@pytest.mark.parametrize("kind", ["source_targets", "http_requests", "model:SourceCredibilityAgentOutput", "action_attempts"])
def test_unrelated_budget_error_is_not_identity_budget_exhaustion(monkeypatch, kind):
    fail_identity(monkeypatch, BudgetExceeded("other operation exhausted", kind=kind))
    _, assessments, _ = apply_source_identity_to_registry(
        [entry()], state()["collection_spec"], llm_enabled=True,
        require_llm=True, allow_deterministic_fallback=False)
    assert assessments[0]["source_identity_status"] == "blocked_llm_required"


@pytest.mark.parametrize("reason", ["wrong_disease", "wrong_country", "user_excluded"])
def test_identity_budget_fallback_never_removes_an_existing_fetch_block(monkeypatch, reason):
    fail_identity(monkeypatch, BudgetExceeded("identity exhausted", kind="model:SourceIdentityAgentOutput"))
    source = entry(blocked_from_fetch=True, blocked_from_fetch_reason=reason,
                   final_screening_decision="exclude", task_fit_evidence_origin="fetched_content")
    rows, _, _ = apply_source_identity_to_registry(
        [source], state()["collection_spec"], llm_enabled=True,
        require_llm=True, allow_deterministic_fallback=False)
    assert rows[0]["blocked_from_fetch"] is True
    assert rows[0]["blocked_from_fetch_reason"] == reason
    assert rows[0]["final_screening_decision"] == "exclude"


def test_pending_same_url_alias_accepts_later_assessment_under_original_identity(monkeypatch, tmp_path):
    import importlib
    from data_collection_workflow.workflow_recovery import execute_recovery, merge_recovery_delta
    from test_acquisition_identity_budget import setup_search
    prior = entry("prior", canonical_url="https://unknown.example/new", url="https://unknown.example/new",
                  source_identity_status="pending_assessment", source_identity_unverified=True)
    ctx, plan, initial, _, _ = setup_search(monkeypatch, tmp_path, failure="provider", old=prior)
    identity = importlib.import_module("data_collection_workflow.source_identity")
    monkeypatch.setattr(identity, "assess_source_identity_with_llm", lambda **_: {"llm_used": False})
    with ctx.activate():
        delta = execute_recovery(plan, context=ctx, artifacts=initial, budget=ctx.ledger)
        result = merge_recovery_delta(initial, delta)
    assert len(result["source_registry"]) == 1
    row = result["source_registry"][0]
    assert row["source_id"] == "prior"
    assert row["source_identity_status"] == "assessed"
    assert row.get("ready_for_content_fetch") is True
    assert initial["source_registry"][0] == prior
