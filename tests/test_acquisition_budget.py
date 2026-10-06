"""Behavioral regressions for the approved v2 acquisition budget contract.

All work is local: callbacks represent transport boundaries and the real SQLite
ledger/frontier enforce reservations, caching, concurrency, and continuation.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
from typing import TypedDict

import pytest

from data_collection_workflow import session_runtime as runtime


def adaptive_config(*, source_targets=200, http_requests=1000, soft=50):
    return {
        "pipeline_mode": "evidence",
        "structured_task": {"disease": "chikungunya", "location": "Reunion"},
        "content_fetch": {"max_total_sources": 50, "max_search_derived_sources": 50},
        "source_search": {
            "max_total_results": 400,
            "iterative": {"enabled": True, "max_total_queries": 20, "max_total_results": 260},
            "authority_gap_retry": {"enabled": True, "max_queries": 24,
                "known_domain_max_queries": 14, "jurisdiction_max_queries": 16,
                "official_page_family_max_queries": 14, "result_budget": 120},
        },
        "universal": {
            "budget_policy": {"version": 2, "mode": "adaptive", "soft_source_target": soft},
            "budget_limits": {"source_targets": source_targets, "http_requests": http_requests},
        },
    }


def _call(ctx, url, *, target=None, fn=None, kind="http_request"):
    assert "source_target" in inspect.signature(ctx.call).parameters, (
        "The acquisition target and first HTTP request need one atomic reservation"
    )
    return ctx.call(kind, {"url": url}, fn or (lambda: {"url": url}), source_target=target or url)


def _amend(ctx, amendment_id, increases):
    amend = getattr(ctx, "amend_budget", None)
    assert callable(amend), "Same-code continuation needs a persistent budget amendment API"
    return amend(amendment_id=amendment_id, increases=increases,
                 reason="Explicit test continuation allowance")


def _frontier(path):
    spec = importlib.util.find_spec("data_collection_workflow.acquisition_frontier")
    assert spec is not None, "Pending acquisition targets need a persistent shared frontier"
    cls = importlib.import_module("data_collection_workflow.acquisition_frontier").AcquisitionFrontier
    return cls(path)


def _enqueue(frontier, target, *, priority=0, payload=None):
    return frontier.enqueue(target_id=target, url=target, source_id="source-" + target.rsplit("/", 1)[-1],
                            priority=priority, payload=payload or {})


def test_adaptive_policy_does_not_reuse_legacy_source_limit_as_http_cap():
    limits = runtime.derive_budget_limits(adaptive_config())
    assert limits.get("source_targets") == 200
    assert limits.get("http_requests") == 1000
    assert "fetch_ordinary" not in limits
    assert "fetch" not in limits
    assert {key: limits[key] for key in ("search", "search_results", "extraction", "browser", "ocr")} == {
        "search": 88, "search_results": 380, "extraction": 2400, "browser": 20, "ocr": 100,
    }


@pytest.mark.parametrize("limit", [0, 2, 50])
def test_unversioned_configs_preserve_explicit_strict_source_caps(limit):
    config = {"content_fetch": {"max_total_sources": limit, "max_search_derived_sources": limit}}
    limits = runtime.derive_budget_limits(config)
    assert limits["fetch_ordinary"] == limit
    assert limits["fetch"] == 200
    assert "source_targets" not in limits


@pytest.mark.parametrize("policy", [
    {"version": 3, "mode": "adaptive", "soft_source_target": 50},
    {"version": 2, "mode": "adaptve", "soft_source_target": 50},
])
def test_unknown_policy_fails_explicitly_instead_of_silently_changing_budget_semantics(policy):
    config = adaptive_config()
    config["universal"]["budget_policy"] = policy
    with pytest.raises(ValueError):
        runtime.derive_budget_limits(config)


def test_first_fifty_targets_are_a_checkpoint_not_a_hard_stop(tmp_path):
    ctx = runtime.RunContext(tmp_path, adaptive_config())
    for number in range(51):
        _call(ctx, f"https://reports.example/report/{number}")
    used = ctx.ledger.snapshot()["used"]
    assert used["source_targets"] == 51
    assert used["http_requests"] == 51


def test_redirect_and_alternate_transport_reuse_one_acquisition_target(tmp_path):
    ctx = runtime.RunContext(tmp_path, adaptive_config(source_targets=1, http_requests=3))
    target = "https://reports.example/report"
    _call(ctx, target, target=target)
    _call(ctx, "https://cdn.example/report", target=target)
    _call(ctx, "https://cdn.example/report?format=html", target=target)
    assert ctx.ledger.snapshot()["used"] == {"source_targets": 1, "http_requests": 3}


def test_two_attachments_are_two_targets_even_when_publication_family_is_shared(tmp_path):
    ctx = runtime.RunContext(tmp_path, adaptive_config(source_targets=2))
    _call(ctx, "https://reports.example/study/report.pdf")
    _call(ctx, "https://reports.example/study/data.csv")
    assert ctx.ledger.snapshot()["used"]["source_targets"] == 2


def test_exhausted_http_cannot_charge_a_target_before_any_request_starts(tmp_path):
    ctx = runtime.RunContext(tmp_path, adaptive_config(source_targets=5, http_requests=0))
    started = []
    with pytest.raises(runtime.BudgetExceeded) as error:
        _call(ctx, "https://reports.example/new", fn=lambda: started.append(True))
    assert error.value.kind == "http_requests"
    assert started == []
    assert ctx.ledger.snapshot()["used"].get("source_targets", 0) == 0
    assert ctx.ledger.operation_audit() == []


def test_concurrent_first_requests_cannot_overspend_target_cap(tmp_path):
    ctx = runtime.RunContext(tmp_path, adaptive_config(source_targets=3, http_requests=30))
    def fetch(number):
        try:
            return _call(ctx, f"https://reports.example/{number}")
        except runtime.BudgetExceeded:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fetch, range(20)))
    assert sum(result is not None for result in results) == 3
    assert ctx.ledger.snapshot()["used"] == {"source_targets": 3, "http_requests": 3}


def test_concurrent_requests_to_one_target_cannot_overspend_http_cap(tmp_path):
    ctx = runtime.RunContext(tmp_path, adaptive_config(source_targets=1, http_requests=3))
    def fetch(number):
        try:
            return _call(ctx, f"https://reports.example/redirect/{number}",
                         target="https://reports.example/report")
        except runtime.BudgetExceeded:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fetch, range(20)))
    assert sum(result is not None for result in results) == 3
    assert ctx.ledger.snapshot()["used"] == {"source_targets": 1, "http_requests": 3}


def test_failed_attempts_cost_http_but_never_recharge_the_same_target(tmp_path):
    ctx = runtime.RunContext(tmp_path, adaptive_config(source_targets=1, http_requests=10))
    def unavailable():
        raise OSError("transport failed")
    for _ in range(2):
        with pytest.raises(OSError):
            _call(ctx, "https://reports.example/report", fn=unavailable)
    with pytest.raises(runtime.BudgetExceeded) as error:
        _call(ctx, "https://reports.example/report", fn=unavailable)
    assert error.value.kind == "action_attempts"
    assert ctx.ledger.snapshot()["used"] == {"source_targets": 1, "http_requests": 2}


def test_cached_response_is_usable_at_both_caps_without_charging_an_alias_target(tmp_path):
    ctx = runtime.RunContext(tmp_path, adaptive_config(source_targets=1, http_requests=1))
    url = "https://reports.example/report"
    original = _call(ctx, url)
    def unexpected_transport():
        pytest.fail("A completed response must be served from the session cache")
    assert _call(ctx, url, target="https://reports.example/alias", fn=unexpected_transport) == original
    assert ctx.ledger.snapshot()["used"] == {"source_targets": 1, "http_requests": 1}


def test_browser_navigation_does_not_hide_its_individual_http_requests(tmp_path):
    ctx = runtime.RunContext(tmp_path, adaptive_config(source_targets=1, http_requests=2))
    ctx.call("browser_navigation", {"url": "https://reports.example/dashboard"}, lambda: {"navigation": 1})
    _call(ctx, "https://reports.example/dashboard", target="https://reports.example/dashboard")
    _call(ctx, "https://reports.example/data.json", target="https://reports.example/dashboard")
    with pytest.raises(runtime.BudgetExceeded) as error:
        _call(ctx, "https://reports.example/chart.js", target="https://reports.example/dashboard")
    assert error.value.kind == "http_requests"
    assert ctx.ledger.snapshot()["used"] == {"browser": 1, "source_targets": 1, "http_requests": 2}


def test_budget_amendment_survives_resume_without_changing_cache_identity(tmp_path):
    config = adaptive_config(source_targets=1, http_requests=1)
    ctx = runtime.RunContext(tmp_path, config)
    url = "https://reports.example/first"
    _call(ctx, url)
    original_fingerprint = ctx.fingerprint
    _amend(ctx, "continue-1", {"source_targets": 2, "http_requests": 2})
    resumed = runtime.RunContext(tmp_path, config, resume=True)
    assert resumed.fingerprint == original_fingerprint
    def unexpected_transport():
        pytest.fail("A budget amendment must preserve completed operation cache keys")
    _call(resumed, url, fn=unexpected_transport)
    _call(resumed, "https://reports.example/second")
    snapshot = resumed.ledger.snapshot()
    assert snapshot["used"] == {"source_targets": 2, "http_requests": 2}
    assert snapshot["limits"]["source_targets"] == 2
    assert snapshot["limits"]["search"] == 88
    assert snapshot["limits"]["extraction"] == 2400
    assert snapshot["budget_revision"] == 1
    assert len(snapshot["budget_amendments"]) == 1


def test_identical_amendment_is_idempotent_but_conflicting_id_is_rejected(tmp_path):
    ctx = runtime.RunContext(tmp_path, adaptive_config(source_targets=1, http_requests=1))
    _amend(ctx, "continue-1", {"source_targets": 2, "http_requests": 2})
    _amend(ctx, "continue-1", {"source_targets": 2, "http_requests": 2})
    with pytest.raises(ValueError):
        _amend(ctx, "continue-1", {"source_targets": 3})
    snapshot = ctx.ledger.snapshot()
    assert snapshot["budget_revision"] == 1
    assert snapshot["limits"]["source_targets"] == 2


@pytest.mark.parametrize("increases", [
    {"source_targets": 0}, {"max_bytes": 20_000_000}, {"unexpected": 9},
    {"model:ArbitraryUnconfiguredStage": 99},
])
def test_amendment_rejects_decreases_and_non_budget_or_unknown_dimensions(tmp_path, increases):
    ctx = runtime.RunContext(tmp_path, adaptive_config(source_targets=1, http_requests=1))
    with pytest.raises(ValueError):
        _amend(ctx, "invalid", increases)
    assert ctx.ledger.snapshot().get("budget_revision", 0) == 0


def test_amendment_does_not_reset_failed_action_attempts(tmp_path):
    ctx = runtime.RunContext(tmp_path, adaptive_config(source_targets=1, http_requests=2))
    def unavailable():
        raise OSError("still unavailable")
    for _ in range(2):
        with pytest.raises(OSError):
            _call(ctx, "https://reports.example/report", fn=unavailable)
    _amend(ctx, "continue-1", {"http_requests": 10})
    with pytest.raises(runtime.BudgetExceeded) as error:
        _call(ctx, "https://reports.example/report", fn=unavailable)
    assert error.value.kind == "action_attempts"
    assert ctx.ledger.snapshot()["used"]["http_requests"] == 2


def test_budget_amendment_does_not_allow_task_config_mutation(tmp_path):
    config = adaptive_config()
    runtime.RunContext(tmp_path, config)
    changed = deepcopy(config)
    changed["structured_task"]["disease"] = "measles"
    with pytest.raises(runtime.ResumeMismatch):
        runtime.RunContext(tmp_path, changed, resume=True)


def test_frontier_reorders_persisted_pending_work_when_better_evidence_is_discovered(tmp_path):
    path = tmp_path / "operations.sqlite"
    frontier = _frontier(path)
    _enqueue(frontier, "https://reports.example/context", priority=1)
    _enqueue(frontier, "https://reports.example/data.csv", priority=20,
             payload={"publication_family": "study-1", "assessment": "unknown"})
    reopened = _frontier(path)
    first = reopened.claim_next()
    assert first["target_id"] == "https://reports.example/data.csv"
    assert first["payload"]["assessment"] == "unknown"
    assert reopened.claim_next()["target_id"] == "https://reports.example/context"
    assert _frontier(path).claim_next() is None


def test_frontier_duplicate_enqueue_cannot_restart_completed_target(tmp_path):
    frontier = _frontier(tmp_path / "operations.sqlite")
    target = "https://reports.example/report"
    _enqueue(frontier, target, priority=1)
    assert frontier.claim_next()["target_id"] == target
    frontier.finish(target, status="completed", reason="readable", operation_started=True)
    _enqueue(frontier, target, priority=99)
    assert frontier.claim_next() is None
    rows = frontier.snapshot()["items"]
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["attempts"] == 1


def test_frontier_budget_deferral_remains_resumable_without_counting_a_false_attempt(tmp_path):
    path = tmp_path / "operations.sqlite"
    frontier = _frontier(path)
    target = "https://reports.example/report"
    _enqueue(frontier, target)
    frontier.claim_next()
    frontier.finish(target, status="budget_deferred", reason="http_requests", operation_started=False)
    assert frontier.claim_next() is None
    reopened = _frontier(path)
    assert reopened.snapshot()["items"][0]["attempts"] == 0
    reopened.resume_budget_deferred(budget_revision=1)
    resumed = reopened.claim_next()
    assert resumed["target_id"] == target
    assert resumed["attempts"] == 0
    assert resumed["budget_revision"] == 1


def test_frontier_amendment_cannot_retry_unrelated_failures_or_reset_attempts(tmp_path):
    frontier = _frontier(tmp_path / "operations.sqlite")
    target = "https://reports.example/report"
    _enqueue(frontier, target)
    frontier.claim_next()
    frontier.finish(target, status="failed", reason="permanent_http_error", operation_started=True)
    frontier.resume_budget_deferred(budget_revision=1)
    assert frontier.claim_next() is None
    assert frontier.snapshot()["items"][0]["attempts"] == 1


def test_frontier_claim_is_atomic_across_workers(tmp_path):
    path = tmp_path / "operations.sqlite"
    frontier = _frontier(path)
    _enqueue(frontier, "https://reports.example/report")
    def claim(_):
        return _frontier(path).claim_next()
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(claim, range(16)))
    assert sum(claim is not None for claim in claims) == 1


def test_new_interactive_evidence_preset_expands_acquisition_without_expanding_model_allowance():
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run([
        sys.executable, "-B", "scripts/collect.py", "--print-config-only",
        "--pipeline-mode", "evidence", "--disease", "chikungunya",
        "--location", "Reunion", "--start-date", "2025-01-01", "--end-date", "2025-12-31",
        "--session-id", "budget-preset-preview", "--provider", "anthropic", "--model", "test-model",
    ], cwd=project, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    preview = json.loads(result.stdout.split("sanitized_config_json:", 1)[1])
    config = preview["config"]
    assert config["content_fetch"]["max_bytes"] == 20_000_000
    limits = runtime.derive_budget_limits(config)
    assert limits["source_targets"] == 200
    assert limits["http_requests"] == 1000
    assert limits["extraction"] == 2400
    assert limits["search"] == 88
    assert limits["browser"] == 20
    assert limits["ocr"] == 100


@pytest.mark.parametrize("adaptive", [False, True])
def test_runner_allows_budget_bounded_adaptive_continuation_beyond_legacy_100_steps(
    tmp_path, monkeypatch, adaptive,
):
    from langgraph.errors import GraphRecursionError
    from langgraph.graph import END, START, StateGraph
    import scripts.run_workflow as runner

    class CounterState(TypedDict):
        steps: int

    builder = StateGraph(CounterState)
    builder.add_node("recovery_control", lambda state: {"steps": state["steps"] + 1})
    builder.add_edge(START, "recovery_control")
    builder.add_conditional_edges("recovery_control", lambda state: END if state["steps"] >= 120 else "recovery_control")
    graph = builder.compile()

    @contextmanager
    def checkpoint_graph(_context):
        yield graph

    monkeypatch.setattr(runtime, "checkpoint_graph", checkpoint_graph)
    config = adaptive_config() if adaptive else {"pipeline_mode": "evidence"}
    ctx = runtime.RunContext(tmp_path, config)
    with ctx.activate():
        if adaptive:
            result = runner._run_graph_with_events({"steps": 0}, output_dir=tmp_path,
                                                   session_id="adaptive-budget", live_status=False)
            assert result["steps"] == 120
        else:
            with pytest.raises(GraphRecursionError):
                runner._run_graph_with_events({"steps": 0}, output_dir=tmp_path,
                                             session_id="strict-budget", live_status=False)


def test_frontier_soft_checkpoint_can_lower_pending_priority_without_reopening_other_states(tmp_path):
    frontier = _frontier(tmp_path / "operations.sqlite")
    first = "https://reports.example/another-attachment"
    second = "https://independent.example/report"
    running = "https://reports.example/running"
    completed = "https://reports.example/completed"
    _enqueue(frontier, completed, priority=200)
    frontier.claim_next()
    frontier.finish(completed, status="completed", reason="readable", operation_started=True)
    _enqueue(frontier, running, priority=100)
    frontier.claim_next()
    _enqueue(frontier, first, priority=10)
    _enqueue(frontier, second, priority=5)

    frontier.reprioritize({first: 1, second: 20, running: 0, completed: 999,
                          "https://reports.example/not-enqueued": 1000})

    rows = {row["target_id"]: row for row in frontier.snapshot()["items"]}
    assert len(rows) == 4
    assert rows[first]["priority"] == 1
    assert rows[second]["priority"] == 20
    assert rows[running]["priority"] == 100
    assert rows[running]["status"] == "running"
    assert rows[completed]["priority"] == 200
    assert rows[completed]["status"] == "completed"
    assert rows[completed]["attempts"] == 1
    selected = frontier.claim_next()
    assert selected["target_id"] == second
    assert selected["attempts"] == 0


def test_frontier_retains_result_reference_before_graph_checkpoint(tmp_path):
    frontier = _frontier(tmp_path / "operations.sqlite")
    target = "https://reports.example/report"
    _enqueue(frontier, target)
    frontier.claim_next()
    frontier.finish(target, status="completed", reason="readable", operation_started=True,
                    result_ref=".universal/frontier-results/document.json")
    reopened = _frontier(tmp_path / "operations.sqlite")
    assert reopened.snapshot()["items"][0]["result_ref"] == ".universal/frontier-results/document.json"
    assert reopened.claim_next() is None


@pytest.mark.parametrize("reference", ["../outside.json", "E:/outside.json", "/outside.json"])
def test_frontier_rejects_result_references_outside_session(tmp_path, reference):
    frontier = _frontier(tmp_path / "operations.sqlite")
    target = "https://reports.example/report"
    _enqueue(frontier, target)
    frontier.claim_next()
    with pytest.raises(ValueError):
        frontier.finish(target, status="completed", result_ref=reference)
    assert frontier.snapshot()["items"][0]["status"] == "running"


def test_frontier_explicit_retry_preserves_attempt_count_and_stops_after_two(tmp_path):
    frontier = _frontier(tmp_path / "operations.sqlite")
    target = "https://reports.example/report"
    _enqueue(frontier, target)
    frontier.claim_next()
    frontier.finish(target, status="failed", operation_started=True)
    assert frontier.retry(target) is True
    assert frontier.claim_next()["attempts"] == 1
    frontier.finish(target, status="failed", operation_started=True)
    assert frontier.retry(target) is False
    assert frontier.claim_next() is None


def test_interactive_quick_preset_keeps_small_explicit_budgets_and_zero_values():
    from data_collection_workflow import acquisition_budget as policy
    import scripts.collect as interactive

    config = interactive._real_run_config(disease="chikungunya", location="Reunion",
        start_date="2025-01-01", end_date="2025-12-31", target_fields=[],
        session_id="quick-preview", provider="anthropic", model="test-model",
        output_dir=None, no_llm=False, quick_test_mode=True)
    config["pipeline_mode"] = "evidence"
    config = policy.apply_interactive_acquisition_preset(config, quick_test=True)
    limits = runtime.derive_budget_limits(config)
    assert limits["source_targets"] == 5
    assert limits["http_requests"] == 25
    assert limits["search"] == 3
    assert limits["search_results"] == 6
    assert limits["extraction"] == 5
    assert config["universal"]["extraction_reserve"] < limits["extraction"]
    config["content_fetch"]["max_total_sources"] = 0
    config["source_search"]["iterative"]["max_total_queries"] = 0
    config["llm"]["max_chunks"] = 0
    zero = policy.apply_interactive_acquisition_preset(config, quick_test=True)
    zero_limits = runtime.derive_budget_limits(zero)
    assert zero_limits["source_targets"] == zero_limits["http_requests"] == 0
    assert zero_limits["search"] == zero_limits["extraction"] == 0


def test_configured_and_interactive_cli_accept_a_separate_budget_amendment_file():
    import scripts.collect as interactive
    import scripts.run_workflow as runner
    arguments = ["--resume-session", "same-session", "--budget-amendment", "allowance.json"]
    assert interactive.build_parser().parse_args(arguments).budget_amendment == "allowance.json"
    assert runner._build_parser().parse_args(arguments).budget_amendment == "allowance.json"


def test_budget_amendment_file_cannot_smuggle_non_budget_configuration(tmp_path):
    from data_collection_workflow import acquisition_budget as policy
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps({"amendment_id": "continue-1", "reason": "more coverage",
                               "increases": {"http_requests": 2000}, "max_bytes": 99}), encoding="utf-8")
    with pytest.raises(ValueError):
        policy.load_budget_amendment(path)


def test_resume_requeues_interrupted_frontier_claim_when_transport_completed(tmp_path):
    config = adaptive_config()
    ctx = runtime.RunContext(tmp_path, config)
    target = "https://reports.example/report"
    _enqueue(ctx.frontier, target)
    ctx.frontier.claim_next()
    _call(ctx, target)
    resumed = runtime.RunContext(tmp_path, config, resume=True)
    assert resumed.frontier.claim_next()["target_id"] == target
    def no_repeat():
        pytest.fail("Completed transport must survive a missing node checkpoint")
    _call(resumed, target, fn=no_repeat)
    assert resumed.ledger.snapshot()["used"]["http_requests"] == 1


def test_adaptive_resume_retries_unknown_transport_once_without_refunding_first_charge(tmp_path):
    config = adaptive_config()
    ctx = runtime.RunContext(tmp_path, config)
    target = "https://reports.example/report"
    _enqueue(ctx.frontier, target)
    ctx.frontier.claim_next()
    ctx.ledger.begin("http_request", {"fingerprint": ctx.fingerprint, "input": {"url": target}}, source_target=target)
    resumed = runtime.RunContext(tmp_path, config, resume=True)
    assert resumed.frontier.claim_next()["attempts"] == 1
    _call(resumed, target)
    resumed.frontier.finish(target, status="completed", operation_started=True)
    assert resumed.ledger.snapshot()["used"]["http_requests"] == 2
    assert resumed.frontier.snapshot()["items"][0]["attempts"] == 2



def test_explicit_known_budget_amendment_does_not_implicitly_raise_other_dimensions(tmp_path):
    ctx = runtime.RunContext(tmp_path, adaptive_config())
    _amend(ctx, "search-only", {"search": 90})
    limits = ctx.ledger.snapshot()["limits"]
    assert limits["search"] == 90
    assert limits["source_targets"] == 200
    assert limits["http_requests"] == 1000
    assert limits["extraction"] == 2400


def test_replayed_amendment_cannot_reopen_a_deferral_from_the_same_budget_revision(tmp_path):
    ctx = runtime.RunContext(tmp_path, adaptive_config(source_targets=1, http_requests=0))
    _amend(ctx, "more-http", {"http_requests": 1})
    target = "https://reports.example/report"
    _enqueue(ctx.frontier, target)
    ctx.frontier.claim_next()
    ctx.frontier.finish(target, status="budget_deferred", operation_started=False)
    assert ctx.frontier.snapshot()["items"][0]["budget_revision"] == 1
    _amend(ctx, "more-http", {"http_requests": 1})
    assert ctx.frontier.claim_next() is None


def test_adaptive_resume_uses_durable_complete_response_from_interrupted_operation(tmp_path):
    config = adaptive_config()
    ctx = runtime.RunContext(tmp_path, config)
    target = "https://reports.example/report"
    _enqueue(ctx.frontier, target)
    ctx.frontier.claim_next()
    ticket = ctx.ledger.begin("http_request", {"fingerprint": ctx.fingerprint, "input": {"url": target}}, source_target=target)
    ctx.ledger.stage_response(ticket, {"body": "complete saved response"})
    resumed = runtime.RunContext(tmp_path, config, resume=True)
    assert resumed.frontier.claim_next()["attempts"] == 1
    def forbidden():
        pytest.fail("A complete saved response must not trigger another HTTP request")
    assert _call(resumed, target, fn=forbidden) == {"body": "complete saved response"}
    assert resumed.ledger.snapshot()["used"]["http_requests"] == 1


def test_second_unknown_transport_outcome_is_terminal_without_third_attempt(tmp_path):
    config = adaptive_config()
    ctx = runtime.RunContext(tmp_path, config)
    target = "https://reports.example/report"
    _enqueue(ctx.frontier, target)
    for _ in range(2):
        assert ctx.frontier.claim_next() is not None
        ctx.ledger.begin("http_request", {"fingerprint": ctx.fingerprint, "input": {"url": target}}, source_target=target)
        ctx = runtime.RunContext(tmp_path, config, resume=True)
    assert ctx.frontier.claim_next() is None
    row = ctx.frontier.snapshot()["items"][0]
    assert row["status"] == "failed"
    assert row["attempts"] == 2
    with pytest.raises(runtime.BudgetExceeded) as error:
        _call(ctx, target)
    assert error.value.kind == "action_attempts"
    assert ctx.ledger.snapshot()["used"]["http_requests"] == 2


def test_budget_continuation_reenters_recovery_from_end_without_replaying_acquisition(tmp_path, monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    import scripts.run_workflow as runner

    class SavedState(TypedDict, total=False):
        documents: list
        recovery_stop_reason: str | None
        run_budget_ledger: dict

    visits = []
    def acquisition(state):
        visits.append("acquisition")
        return {"documents": [{"source_id": "already-saved"}]}
    def recovery(state):
        visits.append("recovery")
        ctx = runtime.get_runtime()
        item = ctx.frontier.claim_next()
        documents = list(state.get("documents") or [])
        if item:
            _call(ctx, item["url"])
            documents.append({"source_id": item["source_id"]})
            ctx.frontier.finish(item["target_id"], status="completed", operation_started=True)
        return {"documents": documents, "run_budget_ledger": ctx.ledger.snapshot(),
                "recovery_stop_reason": None if item else "budget_exhausted"}
    builder = StateGraph(SavedState)
    builder.add_node("content_fetch_and_parse", acquisition)
    builder.add_node("quality_gate_routing", lambda state: {})
    builder.add_node("recovery_control", recovery)
    builder.add_edge(START, "content_fetch_and_parse")
    builder.add_edge("content_fetch_and_parse", "quality_gate_routing")
    builder.add_edge("quality_gate_routing", "recovery_control")
    builder.add_edge("recovery_control", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    @contextmanager
    def checkpoint_graph(_context):
        yield graph
    monkeypatch.setattr(runtime, "checkpoint_graph", checkpoint_graph)
    config = adaptive_config(source_targets=1, http_requests=0)
    ctx = runtime.RunContext(tmp_path, config)
    target = "https://reports.example/deferred"
    _enqueue(ctx.frontier, target)
    ctx.frontier.claim_next()
    ctx.frontier.finish(target, status="budget_deferred", operation_started=False)
    with ctx.activate():
        first = runner._run_graph_with_events({}, output_dir=tmp_path, session_id="resume-from-end", live_status=False)
    assert first["documents"] == [{"source_id": "already-saved"}]
    resumed = runtime.RunContext(tmp_path, config, resume=True)
    resumed.resume = True
    _amend(resumed, "continue-1", {"http_requests": 1})
    with resumed.activate():
        second = runner._run_graph_with_events({}, output_dir=tmp_path, session_id="resume-from-end", live_status=False)
    assert second["documents"] == [{"source_id": "already-saved"}, {"source_id": "source-deferred"}]
    assert visits == ["acquisition", "recovery", "recovery"]
    assert resumed.ledger.snapshot()["used"]["http_requests"] == 1


@pytest.mark.parametrize("adaptive", [False, True])
def test_interactive_resume_loads_only_named_saved_config_without_prompt_or_rewrite(tmp_path, adaptive):
    config = adaptive_config()
    if not adaptive:
        config["universal"].pop("budget_policy")
        config["universal"].pop("budget_limits")
    config["content_fetch"]["max_bytes"] = 12345
    config["llm"] = {"provider": "anthropic", "model": "saved-session-model",
                     "structured_extraction_enabled": True}
    config["output"] = {"run_output_root": str(tmp_path / "outputs"), "sessionized": True,
                        "session_id": "resume-budget"}
    config = runtime.prepare_universal_config(config)
    runtime.RunContext(tmp_path / "outputs/sessions/resume-budget", config)
    generated = tmp_path / "outputs/generated_configs/resume-budget.json"
    generated.parent.mkdir(parents=True)
    original = json.dumps(config, ensure_ascii=False, indent=2)
    generated.write_text(original, encoding="utf-8")
    amendment = tmp_path / "allowance.json"
    amendment.write_text(json.dumps({"amendment_id": "more-http", "reason": "continue",
                                     "increases": {"http_requests": 2000}}), encoding="utf-8")
    code = r"""
import builtins, json, os, sys
from pathlib import Path
import scripts.collect as interactive
interactive.PROJECT_ROOT = Path(sys.argv[1])
os.environ['LLM_MODEL'] = 'different-environment-model'
def forbidden(*args, **kwargs):
    raise AssertionError('resume must not prompt or rewrite the saved config')
builtins.input = forbidden
interactive._write_generated_config = forbidden
interactive._require_keys = lambda **kwargs: []
def run(args):
    saved = json.loads(Path(args.config).read_text(encoding='utf-8'))
    assert saved['llm']['model'] == 'saved-session-model'
    assert saved['content_fetch']['max_bytes'] == 12345
    assert args.resume_session == 'resume-budget'
    print('SAVED_CONFIG_DISPATCHED')
    return {'ok': True}
interactive.run_workflow = run
raise SystemExit(interactive.main(sys.argv[2:]))
"""
    arguments = ["--resume-session", "resume-budget", "--no-dashboard", "--no-run-notebook", "--no-live-status"]
    if adaptive:
        arguments += ["--budget-amendment", str(amendment)]
    result = subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path), *arguments],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True,
                            text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SAVED_CONFIG_DISPATCHED" in result.stdout
    assert "llm_model: saved-session-model" in result.stdout
    assert "different-environment-model" not in result.stdout
    assert generated.read_text(encoding="utf-8") == original


def test_interactive_resume_rejects_conflicting_explicit_model_before_dispatch(tmp_path, monkeypatch):
    import scripts.collect as interactive
    config = adaptive_config()
    config["llm"] = {"provider": "anthropic", "model": "saved-session-model"}
    config["output"] = {"run_output_root": str(tmp_path / "outputs"), "sessionized": True,
                        "session_id": "resume-budget"}
    config = runtime.prepare_universal_config(config)
    runtime.RunContext(tmp_path / "outputs/sessions/resume-budget", config)
    generated = tmp_path / "outputs/generated_configs/resume-budget.json"
    generated.parent.mkdir(parents=True)
    generated.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(interactive, "PROJECT_ROOT", tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("conflicting resume settings must fail before prompts or dispatch")
    monkeypatch.setattr(interactive, "_collect_inputs", forbidden)
    monkeypatch.setattr(interactive, "run_workflow", forbidden)
    assert interactive.main(["--resume-session", "resume-budget", "--model", "another-model"]) == 2


@pytest.mark.parametrize("generic_payload", [False, True])
def test_failed_partial_response_is_audited_without_becoming_a_completed_cache(tmp_path, generic_payload):
    ctx = runtime.RunContext(tmp_path, adaptive_config())
    target = "https://reports.example/partial"
    failure = RuntimeError("incomplete response")
    failure.llm_raw_response = {"body": "legacy audit payload"}
    expected = failure.llm_raw_response
    if generic_payload:
        failure.response_payload = {"body": b"partial received bytes", "status_code": 200}
        expected = failure.response_payload
    def incomplete():
        raise failure
    with pytest.raises(RuntimeError, match="incomplete response"):
        _call(ctx, target, fn=incomplete)
    with ctx.ledger._db() as db:
        row = db.execute("SELECT status,response,response_ready FROM operations").fetchone()
    assert row["status"] == "failed"
    assert row["response_ready"] == 0
    assert runtime._decode(json.loads(row["response"])) == expected
    assert not ctx.has_cached("http_request", {"url": target})
    assert _call(ctx, target, fn=lambda: {"body": "complete retry"}) == {"body": "complete retry"}
    assert ctx.ledger.snapshot()["used"] == {"source_targets": 1, "http_requests": 2}
