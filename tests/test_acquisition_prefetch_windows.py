"""Offline regression contracts for prefetch after rolling quality checkpoints.

These tests exercise the real scheduler with a local fake model. They do not
change live configuration or dispatch provider requests.
"""

from collections import Counter
from types import SimpleNamespace

import pytest

from data_collection_workflow import llm_clients
from data_collection_workflow.models import LLMExtractionOutput, LLMExtractedRecord
from data_collection_workflow.session_runtime import RunBudgetLedger
from data_collection_workflow.nodes import extraction
from test_extraction_scheduler_efficiency import (
    _chunk,
    _disable_deterministic_and_recovery_paths,
    _scheduler_policies,
)


def _run_window_probe(monkeypatch, *, concurrency, count=140, safety=400,
                      duplicate=False, strong=True, positive=False, fail_ids=(), runtime=None,
                      strong_indices=None):
    monkeypatch.setenv("LLM_EXTRACTION_SCHEDULER_MODE", "quality_adaptive")
    monkeypatch.setenv("LLM_EXTRACTION_MAX_CONCURRENCY", str(concurrency))
    monkeypatch.setenv("LLM_EXTRACTION_ROLLING_YIELD_WINDOW", "40")
    monkeypatch.setenv("LLM_EXTRACTION_SOFT_CHECKPOINT_CALLS", "400")
    monkeypatch.setenv("LLM_EXTRACTION_SAFETY_MAX_CALLS", str(safety))
    monkeypatch.setattr(llm_clients, "get_llm_settings",
                        lambda: {"provider": "anthropic", "model": "local-fake"})
    _disable_deterministic_and_recovery_paths(monkeypatch)
    captured = {}
    real_factory = extraction.ExtractionBudgetLedger.from_environment

    def make_ledger(cls):
        ledger = real_factory()
        captured["ledger"] = ledger
        return ledger

    monkeypatch.setattr(extraction.ExtractionBudgetLedger, "from_environment",
                        classmethod(make_ledger))
    submitted = []
    batches = []
    real_bounded = extraction._run_bounded_llm_calls

    def observed_bounded(chunks, *, policy, max_concurrency, result_cache=None):
        cache = result_cache if result_cache is not None else {}
        new_keys = {
            extraction._llm_semantic_span_hash(chunk, policy)
            for chunk in chunks
        } - set(cache)
        batches.append({
            "processed_window": sum(captured["ledger"].recent_model_call_counts),
            "new_calls": len(new_keys),
            "worker_cap": max_concurrency,
        })
        return real_bounded(chunks, policy=policy, max_concurrency=max_concurrency,
                            result_cache=result_cache)

    monkeypatch.setattr(extraction, "_run_bounded_llm_calls", observed_bounded)

    if positive:
        monkeypatch.setattr(
            extraction, "_build_record_from_llm_output",
            lambda row, chunk, *_args: SimpleNamespace(
                record_id=chunk["chunk_id"], model_dump=lambda: {"cases_confirmed": 1},
            ),
        )

    def fake_extract(chunk, _policy):
        ticket = None
        if runtime is not None:
            ticket = runtime.begin("extraction", {"input": {"chunk_id": chunk["chunk_id"]}})
            assert not ticket["cached"]
        submitted.append(chunk["chunk_id"])
        if chunk["chunk_id"] in fail_ids:
            raise ValueError("Local prefetched response failed validation")
        if ticket is not None:
            runtime.finish(ticket, value={"ok": True})
        productive = positive(chunk) if callable(positive) else positive
        return LLMExtractionOutput(
            chunk_is_relevant=True,
            records=[LLMExtractedRecord(cases_confirmed=1)] if productive else [],
        )

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    chunks = []
    for index in range(count):
        is_strong = index in strong_indices if strong_indices is not None else strong
        text = (f"Patient {index + 1} was confirmed by PCR." if is_strong
                else f"General surveillance update number {index + 1}.")
        extra = ({"case_span_id": f"case_{index:03d}", "case_span_quote": text}
                 if is_strong else {"extraction_priority": 0})
        chunks.append(_chunk(f"source_{index:03d}", text,
                             contains_target_data=True, **extra))
    if duplicate:
        chunks.append(dict(chunks[0]))
    llm_policy, deterministic_policy = _scheduler_policies()
    _, stats = extraction._llm_extract_records_from_chunks(
        chunks, llm_policy, deterministic_policy, fallback_to_rule_based=False,
        context={"collection_mode": "standard"},
    )
    return captured["ledger"], submitted, batches, stats


@pytest.mark.parametrize("concurrency", [2, 4])
def test_prefetch_resumes_after_full_quality_window(monkeypatch, concurrency):
    ledger, calls, batches, stats = _run_window_probe(
        monkeypatch, concurrency=concurrency,
    )

    assert len(calls) == len(set(calls)) == 140
    assert stats["scheduler_model_call_count"] == 140
    # Passing the checkpoint must retain the full rolling quality history.
    assert sum(ledger.recent_model_call_counts) == 40
    assert ledger.recent_valid_record_counts == [0] * 40
    later_dispatches = [b["new_calls"] for b in batches
                        if b["processed_window"] == 40 and b["new_calls"]]
    assert later_dispatches, "The fixture must reach uncached work after the window."
    assert max(later_dispatches) > 1, (
        "Eligible strong spans became permanently serial after the first "
        f"40-result quality window: {later_dispatches}"
    )
    assert max(b["new_calls"] for b in batches) <= concurrency


def test_explicit_single_worker_remains_single_after_window(monkeypatch):
    ledger, calls, batches, _ = _run_window_probe(
        monkeypatch, concurrency=1, count=45,
    )
    assert len(calls) == 45
    assert all(b["new_calls"] <= 1 and b["worker_cap"] == 1 for b in batches)
    assert sum(ledger.recent_model_call_counts) == 40


def test_hard_call_cap_is_preserved_across_quality_window(monkeypatch):
    ledger, calls, batches, stats = _run_window_probe(
        monkeypatch, concurrency=4, safety=43,
    )
    assert len(calls) == len(set(calls)) == 43
    assert ledger.actual_model_call_count == 43
    assert stats["extraction_budget_ledger"]["incomplete_due_to_safety_cap"] is True
    assert max(b["new_calls"] for b in batches) <= 4


def test_replayed_semantic_span_does_not_add_call_or_quality_history(monkeypatch):
    ledger, calls, _, stats = _run_window_probe(
        monkeypatch, concurrency=4, count=45, duplicate=True,
    )
    assert len(calls) == 45
    assert all(n == 1 for n in Counter(calls).values())
    assert ledger.actual_model_call_count == 45
    assert stats["scheduler_model_call_count"] == 45
    assert sum(ledger.recent_model_call_counts) == 40


def test_zero_yield_checkpoint_runs_before_more_weak_spans_are_dispatched(monkeypatch):
    ledger, calls, _, stats = _run_window_probe(
        monkeypatch, concurrency=4, count=48, strong=False,
    )
    assert len(calls) == 40, (
        "Reservations must count outstanding calls toward the next quality "
        "checkpoint, so zero-yield weak spans beyond it remain uncalled."
    )
    assert stats["rolling_low_yield_stop_triggered"] is True
    assert sum(ledger.recent_model_call_counts) == 40


def test_productive_quality_history_allows_later_parallel_weak_spans(monkeypatch):
    ledger, calls, batches, stats = _run_window_probe(
        monkeypatch, concurrency=4, strong=False, positive=True,
    )
    assert len(calls) == 140
    assert stats["rolling_low_yield_stop_triggered"] is False
    assert ledger.recent_valid_record_counts == [1] * 40
    assert sum(ledger.recent_model_call_counts) == 40
    assert max(b["new_calls"] for b in batches if b["processed_window"] == 40) > 1


def test_prefetched_failure_is_applied_once_and_does_not_disable_later_windows(monkeypatch):
    ledger, calls, batches, stats = _run_window_probe(
        monkeypatch, concurrency=4, fail_ids={"chunk_source_002"},
    )
    assert len(calls) == len(set(calls)) == 140
    assert stats["llm_error_count"] == 1
    assert ledger.actual_model_call_count == 140
    assert sum(ledger.recent_model_call_counts) == 40
    assert max(b["new_calls"] for b in batches if b["processed_window"] == 40) > 1


@pytest.mark.parametrize("limit", [0, 5])
def test_real_atomic_extraction_ledger_retains_explicit_small_caps(monkeypatch, tmp_path, limit):
    runtime = RunBudgetLedger(tmp_path / "operations.sqlite", {"extraction": limit},
                              extraction_reserve=0, adaptive=True)
    _, calls, batches, _ = _run_window_probe(
        monkeypatch, concurrency=4, count=12, runtime=runtime,
    )
    snapshot = runtime.snapshot()
    assert len(calls) == limit
    assert snapshot["used"].get("extraction", 0) == limit
    assert snapshot["operations"].get("completed", 0) == limit
    assert snapshot["operations"].get("running", 0) == 0
    assert max(b["new_calls"] for b in batches) <= 4


def test_strong_current_span_cannot_prefetch_zero_yield_weak_neighbors(monkeypatch):
    # Hold admission order fixed: this tests checkpoint eligibility, not ranking.
    monkeypatch.setattr(extraction, "_ordered_llm_chunks", lambda chunks, _context: list(chunks))
    ledger, calls, _, stats = _run_window_probe(
        monkeypatch, concurrency=4, count=48, strong_indices={40},
    )
    assert len(calls) == 41
    assert "chunk_source_040" in calls
    assert all(f"chunk_source_{index:03d}" not in calls for index in range(41, 48))
    assert stats["rolling_low_yield_stop_triggered"] is True
    # Candidate screening is a dry run; only actual main-loop skips count.
    assert ledger.low_yield_stop_count == 7
    assert ledger.recent_valid_record_counts == [0] * 40


def test_quality_stop_still_works_after_a_productive_first_window(monkeypatch):
    ledger, calls, _, stats = _run_window_probe(
        monkeypatch, concurrency=4, strong=False,
        positive=lambda chunk: int(chunk["source_id"].rsplit("_", 1)[1]) < 40,
    )
    assert 40 < len(calls) <= 80
    assert stats["rolling_low_yield_stop_triggered"] is True
    assert sum(ledger.recent_model_call_counts) == 40
    assert sum(ledger.recent_valid_record_counts) <= 1


@pytest.mark.parametrize("revision", ["evidence", "legacy"])
@pytest.mark.parametrize("fallback_enabled", [False, True])
@pytest.mark.parametrize("skip_kind", ["covered_source", "non_target"])
def test_prefetch_matches_metric_row_consumer_eligibility(
    monkeypatch, revision, fallback_enabled, skip_kind,
):
    monkeypatch.setenv("PIPELINE_MODE", revision)
    monkeypatch.setenv("LLM_EXTRACTION_MAX_CONCURRENCY", "4")
    monkeypatch.setenv("LLM_EXTRACTION_ROLLING_YIELD_WINDOW", "40")
    monkeypatch.setenv("LLM_EXTRACTION_SOFT_CHECKPOINT_CALLS", "400")
    monkeypatch.setenv("LLM_EXTRACTION_SAFETY_MAX_CALLS", "400")
    monkeypatch.setenv("DIRECT_MIN_TARGET_METRIC_RECORDS", "6")
    monkeypatch.setenv("DIRECT_ENABLE_TEXT_FALLBACK_AFTER_ROW_EXTRACTION",
                       "true" if fallback_enabled else "false")
    monkeypatch.setattr(llm_clients, "get_llm_settings",
                        lambda: {"provider": "anthropic", "model": "local-fake"})
    _disable_deterministic_and_recovery_paths(monkeypatch)
    monkeypatch.setattr(extraction, "_ordered_llm_chunks", lambda chunks, _context: list(chunks))
    # Keep the real metric-row and main-loop consumer branches; mock only the
    # deterministic row record producer, which has its own evidence contracts.
    metric_records = [SimpleNamespace(record_id=f"row_record_{i}",
                                      model_dump=lambda: {"cases_confirmed": 1})
                      for i in range(6)]
    monkeypatch.setattr(extraction, "_deterministic_metric_row_records",
                        lambda *_args, **_kwargs: metric_records)
    captured = {}
    factory = extraction.ExtractionBudgetLedger.from_environment

    def make_ledger(cls):
        captured["ledger"] = factory()
        return captured["ledger"]

    monkeypatch.setattr(extraction.ExtractionBudgetLedger, "from_environment", classmethod(make_ledger))
    calls = []

    def fake_extract(chunk, _policy):
        calls.append(chunk["chunk_id"])
        return LLMExtractionOutput(chunk_is_relevant=True)

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    covered = _chunk("covered", "A confirmed case was reported.", chunk_id="covered_row",
                     chunk_kind="metric_row", row_id="row1", source_role_final="collection",
                     contains_target_data=True)
    candidates = [_chunk(f"allowed_{i}", f"Patient {i + 1} was confirmed by PCR.",
                         source_role_final="collection", contains_target_data=True)
                  for i in range(45)]
    skipped = _chunk("covered" if skip_kind == "covered_source" else "non_target",
                     "Canada reported 7 confirmed measles cases in March 2025.",
                     chunk_id="consumer_candidate",
                     contains_target_data=True,
                     **({"source_role_final": "collection"} if skip_kind == "covered_source" else {}))
    chunks = [covered, candidates[0], skipped, *candidates[1:]]
    llm_policy, deterministic_policy = _scheduler_policies()
    _, stats = extraction._llm_extract_records_from_chunks(
        chunks, llm_policy, deterministic_policy, fallback_to_rule_based=False,
        context={"collection_mode": "direct_collection"},
    )
    expected_candidate = (revision == "evidence"
                          or (fallback_enabled and skip_kind == "covered_source"))
    assert ("consumer_candidate" in calls) is expected_candidate
    assert stats["deterministic_metric_row_record_count"] == 6
    assert len(calls) == 45 + int(expected_candidate)
    if revision == "evidence":
        assert stats["skipped_text_fallback_after_row_record_count"] == 0
        assert stats["skipped_context_extraction_count"] == 0
    elif skip_kind == "covered_source" and not fallback_enabled:
        assert stats["skipped_text_fallback_after_row_record_count"] == 1
    elif skip_kind == "non_target":
        assert stats["skipped_context_extraction_count"] == 1
    ledger = captured["ledger"]
    assert ledger.primary_model_call_count == ledger.primary_result_call_count == len(calls)
