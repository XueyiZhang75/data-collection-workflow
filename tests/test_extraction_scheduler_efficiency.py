"""Focused contracts for the extraction scheduler runtime."""

from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from data_collection_workflow import llm_clients
from data_collection_workflow.config import (
    load_llm_structured_extraction_policy,
    load_structured_extraction_policy,
)
from data_collection_workflow.models import (
    LLMExtractionOutput,
    LLMStructuredExtractionPolicy,
    StructuredExtractionPolicy,
)
from data_collection_workflow.nodes import extraction
from data_collection_workflow.runtime_profile import (
    default_workflow_run_config,
    workflow_run_env,
)


def _chunk(source_id: str, text: str, **overrides) -> dict:
    chunk = {
        "chunk_id": f"chunk_{source_id}",
        "source_id": source_id,
        "source_url": f"https://{source_id}.example.org/report",
        "text": text,
        "chunk_kind": "text",
        "fetch_purpose": "data_extraction",
    }
    chunk.update(overrides)
    return chunk


def _ledger(**overrides) -> extraction.ExtractionBudgetLedger:
    values = {
        "scheduler_mode": "quality_adaptive",
        "max_concurrency": 4,
        "soft_checkpoint_calls": 20,
        "safety_max_calls": 20,
        "rolling_yield_window": 2,
        "focused_recovery_reserved_calls": 2,
        "max_domain_call_share": 1.0,
        "high_value_source_min_spans": 1,
        "event_source_min_spans": 1,
    }
    values.update(overrides)
    return extraction.ExtractionBudgetLedger(**values)


def test_legacy_max_chunks_becomes_only_the_soft_checkpoint(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CHUNKS", "7")
    monkeypatch.delenv("LLM_EXTRACTION_SOFT_CHECKPOINT_CALLS", raising=False)
    monkeypatch.delenv("LLM_EXTRACTION_SAFETY_MAX_CALLS", raising=False)

    ledger = extraction.ExtractionBudgetLedger.from_environment()

    assert ledger.soft_checkpoint_calls == 7
    assert ledger.safety_max_calls == 2400
    assert ledger.as_dict()["legacy_llm_max_chunks_deprecated"] is True


def test_safety_cap_counts_primary_and_recovery_calls_together():
    ledger = _ledger(soft_checkpoint_calls=1, safety_max_calls=2)
    chunk = _chunk("official", "One confirmed case.")
    budget = {"budget_bucket": "verified_target_collection"}

    ledger.register_primary_call(chunk, None)
    ledger.register_recovery_call(chunk, None)

    allowed, reason = ledger.can_attempt_primary(chunk, budget, None)
    assert allowed is False
    assert reason == "safety_max_calls_reached"
    assert ledger.as_dict()["incomplete_due_to_safety_cap"] is True


def test_low_yield_stop_waits_for_coverage_floor_or_strong_span():
    ledger = _ledger(
        soft_checkpoint_calls=1,
        recent_valid_record_counts=[0, 0],
        recent_model_call_counts=[1, 1],
    )
    uncovered = _chunk("official", "background text")
    strong = _chunk(
        "official",
        "background text",
        case_span_quote="One confirmed case was reported.",
    )

    assert ledger.should_stop_low_yield(
        uncovered,
        {"budget_bucket": "verified_target_collection"},
        None,
    ) is False
    assert ledger.should_stop_low_yield(
        strong,
        {"budget_bucket": "other"},
        None,
    ) is False
    assert ledger.should_stop_low_yield(
        _chunk("background", "background text"),
        {"budget_bucket": "other"},
        None,
    ) is True


def test_cache_hits_do_not_consume_safety_domain_or_rolling_model_call_budget():
    ledger = _ledger(safety_max_calls=2, rolling_yield_window=2)
    chunk = _chunk("shared", "One confirmed case.")
    budget = {"budget_bucket": "other"}

    for _ in range(5):
        allowed, reason = ledger.can_attempt_primary(
            chunk, budget, None, actual_model_call=False
        )
        assert (allowed, reason) == (True, None)
        ledger.register_primary_call(chunk, None, actual_model_call=False)
        ledger.register_primary_result(0, actual_model_calls=0)

    assert ledger.actual_model_call_count == 0
    assert ledger.calls_by_domain == {}
    assert ledger.recent_model_call_counts == []
    assert ledger.recent_valid_record_counts == []

    for _ in range(2):
        allowed, reason = ledger.can_attempt_primary(
            chunk, budget, None, actual_model_call=True
        )
        assert (allowed, reason) == (True, None)
        ledger.register_primary_call(chunk, None, actual_model_call=True)

    allowed, reason = ledger.can_attempt_primary(
        chunk, budget, None, actual_model_call=True
    )
    assert allowed is False
    assert reason == "safety_max_calls_reached"


def test_domain_cap_counts_only_actual_model_calls():
    ledger = _ledger(
        soft_checkpoint_calls=100,
        safety_max_calls=20,
        max_domain_call_share=0.1,
    )
    chunk = _chunk("same-domain", "One confirmed case.")
    budget = {"budget_bucket": "other"}

    for _ in range(8):
        ledger.register_primary_call(chunk, None, actual_model_call=True)
    for _ in range(20):
        ledger.register_primary_call(chunk, None, actual_model_call=False)

    allowed_cache, cache_reason = ledger.can_attempt_primary(
        chunk, budget, None, actual_model_call=False
    )
    allowed_model, model_reason = ledger.can_attempt_primary(
        chunk, budget, None, actual_model_call=True
    )

    assert (allowed_cache, cache_reason) == (True, None)
    assert allowed_model is False
    assert model_reason == "primary_domain_cap_reached"
    assert ledger.calls_by_domain == {"same-domain.example.org": 8}


def test_bounded_llm_calls_are_concurrent_cacheable_and_result_ordered(monkeypatch):
    active = 0
    peak_active = 0
    call_count = 0
    lock = threading.Lock()

    def fake_extract(chunk, _policy):
        nonlocal active, peak_active, call_count
        with lock:
            active += 1
            peak_active = max(peak_active, active)
            call_count += 1
        time.sleep(0.03 if chunk["source_id"] == "source_a" else 0.01)
        with lock:
            active -= 1
        return LLMExtractionOutput(chunk_is_relevant=True)

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    source_a = _chunk(
        "source_a",
        "Same explicit evidence span: one confirmed case.",
        title="Authority A report",
        publisher="Authority A",
        source_type="official_public_health_agency",
    )
    chunks = [
        source_a,
        dict(source_a),
        _chunk(
            "source_b",
            "Same explicit evidence span: one confirmed case.",
            title="Authority B report",
            publisher="Authority B",
            source_type="official_public_health_agency",
        ),
        _chunk("source_c", "A separate explicit evidence span: two confirmed cases."),
    ]

    results, diagnostics = extraction._run_bounded_llm_calls(
        chunks,
        policy=object(),
        max_concurrency=2,
    )

    assert peak_active == 2
    assert call_count == 3
    assert [result["chunk"]["source_id"] for result in results] == [
        "source_a",
        "source_a",
        "source_b",
        "source_c",
    ]
    assert [result["cache_hit"] for result in results] == [
        False,
        True,
        False,
        False,
    ]
    assert results[0]["semantic_span_hash"] == results[1]["semantic_span_hash"]
    assert results[0]["semantic_span_hash"] != results[2]["semantic_span_hash"]
    assert diagnostics == {"model_call_count": 3, "cache_hit_count": 1}


def test_semantic_cache_key_covers_every_source_field_in_the_llm_prompt():
    base = _chunk(
        "source",
        "One confirmed case.",
        title="Report title",
        publisher="Agency",
        source_type="official_public_health_agency",
        data_types=["case_count"],
        context_types=["surveillance"],
    )
    base_hash = extraction._llm_semantic_span_hash(base, object())

    for field, value in {
        "source_id": "different-source",
        "source_url": "https://different.example.org/report",
        "source_type": "peer_reviewed_literature",
        "title": "Different report",
        "publisher": "Different agency",
        "chunk_id": "different-chunk",
        "data_types": ["case_line_list"],
        "context_types": ["clinical"],
    }.items():
        changed = dict(base)
        changed[field] = value
        assert extraction._llm_semantic_span_hash(changed, object()) != base_hash


def test_provider_exceptions_are_not_persisted_in_result_cache(monkeypatch):
    attempts = 0

    def flaky_extract(_chunk, _policy):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient provider failure")
        return LLMExtractionOutput(chunk_is_relevant=True)

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", flaky_extract)
    cache = {}
    chunk = _chunk("source", "One confirmed case.")

    first, first_diagnostics = extraction._run_bounded_llm_calls(
        [chunk], policy=object(), max_concurrency=1, result_cache=cache
    )
    second, second_diagnostics = extraction._run_bounded_llm_calls(
        [chunk], policy=object(), max_concurrency=1, result_cache=cache
    )

    assert isinstance(first[0]["error"], RuntimeError)
    assert first_diagnostics == {"model_call_count": 1, "cache_hit_count": 0}
    assert second[0]["error"] is None
    assert second[0]["cache_hit"] is False
    assert second_diagnostics == {"model_call_count": 1, "cache_hit_count": 0}
    assert attempts == 2


def _scheduler_policies():
    return (
        LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy()),
        StructuredExtractionPolicy(**load_structured_extraction_policy()),
    )


def _disable_deterministic_and_recovery_paths(monkeypatch):
    monkeypatch.setattr(
        extraction,
        "extract_official_outbreak_records_from_chunks",
        lambda *_args, **_kwargs: ([], []),
    )
    monkeypatch.setattr(
        extraction,
        "_rule_based_extract_records_from_chunks",
        lambda *_args, **_kwargs: ([], {}),
    )
    monkeypatch.setattr(extraction, "_focused_recovery_queue", lambda *_args: [])


def test_prefetch_does_not_cross_rolling_stop_and_explicit_span_is_preserved(
    monkeypatch,
):
    monkeypatch.setenv("LLM_EXTRACTION_MAX_CONCURRENCY", "3")
    monkeypatch.setenv("LLM_EXTRACTION_ROLLING_YIELD_WINDOW", "1")
    monkeypatch.setenv("LLM_EXTRACTION_SOFT_CHECKPOINT_CALLS", "10")
    monkeypatch.setenv("LLM_EXTRACTION_SAFETY_MAX_CALLS", "10")
    monkeypatch.setattr(
        llm_clients,
        "get_llm_settings",
        lambda: {"provider": "anthropic", "model": "test-model"},
    )
    _disable_deterministic_and_recovery_paths(monkeypatch)
    calls = []

    def empty_extract(chunk, _policy):
        calls.append(chunk["source_id"])
        return LLMExtractionOutput(chunk_is_relevant=True)

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", empty_extract)
    llm_policy, deterministic_policy = _scheduler_policies()
    chunks = [
        _chunk(
            "first",
            "General surveillance update without an explicit case span.",
            contains_target_data=True,
            extraction_priority=0,
        ),
        _chunk(
            "low_priority",
            "Another general surveillance update.",
            contains_target_data=True,
            extraction_priority=0,
        ),
        _chunk(
            "explicit",
            "Patient 1 was confirmed by PCR.",
            case_span_id="case_001",
            case_span_quote="Patient 1 was confirmed by PCR.",
            contains_target_data=True,
        ),
    ]

    _, stats = extraction._llm_extract_records_from_chunks(
        chunks,
        llm_policy,
        deterministic_policy,
        fallback_to_rule_based=False,
        context={"collection_mode": "standard"},
    )

    assert calls == ["first", "explicit"]
    assert stats["rolling_low_yield_stop_triggered"] is True
    assert stats["scheduler_model_call_count"] == 2


def test_prefetch_only_submits_spans_that_pass_soft_budget_reservation(monkeypatch):
    monkeypatch.setenv("LLM_EXTRACTION_MAX_CONCURRENCY", "3")
    monkeypatch.setenv("LLM_EXTRACTION_ROLLING_YIELD_WINDOW", "10")
    monkeypatch.setenv("LLM_EXTRACTION_SOFT_CHECKPOINT_CALLS", "1")
    monkeypatch.setenv("LLM_EXTRACTION_SAFETY_MAX_CALLS", "10")
    monkeypatch.setattr(
        llm_clients,
        "get_llm_settings",
        lambda: {"provider": "anthropic", "model": "test-model"},
    )
    _disable_deterministic_and_recovery_paths(monkeypatch)
    calls = []

    def empty_extract(chunk, _policy):
        calls.append(chunk["source_id"])
        return LLMExtractionOutput(chunk_is_relevant=True)

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", empty_extract)
    llm_policy, deterministic_policy = _scheduler_policies()
    chunks = [
        _chunk(
            "first",
            "General surveillance update.",
            contains_target_data=True,
            extraction_priority=0,
        ),
        _chunk(
            "over_soft_cap",
            "Another general surveillance update.",
            contains_target_data=True,
            extraction_priority=0,
        ),
    ]

    extraction._llm_extract_records_from_chunks(
        chunks,
        llm_policy,
        deterministic_policy,
        fallback_to_rule_based=False,
        context={"collection_mode": "standard"},
    )

    assert calls == ["first"]


def test_concurrent_prefetch_never_exceeds_actual_model_call_safety_cap(monkeypatch):
    monkeypatch.setenv("LLM_EXTRACTION_MAX_CONCURRENCY", "4")
    monkeypatch.setenv("LLM_EXTRACTION_ROLLING_YIELD_WINDOW", "40")
    monkeypatch.setenv("LLM_EXTRACTION_SOFT_CHECKPOINT_CALLS", "10")
    monkeypatch.setenv("LLM_EXTRACTION_SAFETY_MAX_CALLS", "2")
    monkeypatch.setattr(
        llm_clients,
        "get_llm_settings",
        lambda: {"provider": "anthropic", "model": "test-model"},
    )
    _disable_deterministic_and_recovery_paths(monkeypatch)
    calls = []

    def empty_extract(chunk, _policy):
        calls.append(chunk["source_id"])
        return LLMExtractionOutput(chunk_is_relevant=True)

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", empty_extract)
    llm_policy, deterministic_policy = _scheduler_policies()
    chunks = [
        _chunk(
            f"source_{index}",
            f"Patient {index} was confirmed by PCR.",
            case_span_id=f"case_{index:03d}",
            case_span_quote=f"Patient {index} was confirmed by PCR.",
            contains_target_data=True,
        )
        for index in range(1, 5)
    ]

    _, stats = extraction._llm_extract_records_from_chunks(
        chunks,
        llm_policy,
        deterministic_policy,
        fallback_to_rule_based=False,
        context={"collection_mode": "standard"},
    )

    assert len(calls) == 2
    assert stats["scheduler_model_call_count"] == 2
    assert stats["extraction_budget_ledger"]["actual_model_call_count"] == 2
    assert stats["extraction_budget_ledger"]["incomplete_due_to_safety_cap"] is True


def test_focused_recovery_keeps_explicit_span_even_at_low_source_priority():
    broad = _chunk("broad", "One confirmed case was reported.")
    explicit = _chunk(
        "explicit",
        "Patient 1 was confirmed by PCR.",
        case_span_id="case_001",
        case_span_quote="Patient 1 was confirmed by PCR.",
    )

    queued = extraction._focused_recovery_queue([broad, explicit], {}, None)

    assert [chunk["source_id"] for chunk in queued] == ["explicit"]


def test_runtime_profiles_and_interactive_defaults_emit_scheduler_settings():
    config = default_workflow_run_config()
    scheduler = config["llm"]["extraction"]["scheduler"]
    assert scheduler == {
        "mode": "quality_adaptive",
        "max_concurrency": 4,
        "soft_checkpoint_calls": 400,
        "safety_max_calls": 2400,
        "rolling_yield_window": 40,
    }

    env = workflow_run_env()
    assert env["LLM_EXTRACTION_SCHEDULER_MODE"] == "quality_adaptive"
    assert env["LLM_EXTRACTION_MAX_CONCURRENCY"] == "4"
    assert env["LLM_EXTRACTION_SOFT_CHECKPOINT_CALLS"] == "400"
    assert env["LLM_EXTRACTION_SAFETY_MAX_CALLS"] == "2400"
    assert env["LLM_EXTRACTION_ROLLING_YIELD_WINDOW"] == "40"

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "collect.py"
    spec = importlib.util.spec_from_file_location("interactive_scheduler_defaults", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    interactive = module._real_run_config(
        disease="influenza",
        location="Virginia",
        start_date="2024-10-01",
        end_date="2024-10-10",
        target_fields=["cases_confirmed"],
        session_id="scheduler_test",
        provider="anthropic",
        model="mock-model",
        output_dir=None,
        no_llm=False,
    )
    assert interactive["llm"]["extraction"]["scheduler"] == scheduler


def test_must_fetch_does_not_promote_context_span_and_not_ready_span_is_blocked():
    context_span = _chunk(
        "context_source",
        "background context",
        source_role="context_source",
        must_fetch=True,
        extraction_readiness="not_ready",
    )

    assert extraction._chunk_priority(context_span, {"must_fetch_source_ids": ["context_source"]}) == 4
    policy = SimpleNamespace(
        allowed_fetch_purposes=["data_extraction"], allowed_chunk_kinds=["text"]
    )
    assert extraction._direct_llm_chunk_allowed(
        context_span, policy, {"collection_mode": "direct_collection", "must_fetch_source_ids": ["context_source"]}
    ) is False

    context_span.update(
        contains_target_data=True,
        case_span_quote="A locally revalidated target case was confirmed.",
    )
    assert extraction._direct_llm_chunk_allowed(
        context_span, policy, {"collection_mode": "direct_collection", "must_fetch_source_ids": ["context_source"]}
    ) is True
