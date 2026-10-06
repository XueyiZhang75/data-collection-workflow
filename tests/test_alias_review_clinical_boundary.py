"""Independent clinical signal review: numeric role must precede budget boost."""
import pytest
from data_collection_workflow.nodes.extraction import _chunk_has_strong_record_signal, ExtractionBudgetLedger

def ledger():
    value=ExtractionBudgetLedger(scheduler_mode="quality_adaptive",max_concurrency=4,
        soft_checkpoint_calls=2,safety_max_calls=12,rolling_yield_window=4,
        focused_recovery_reserved_calls=2,max_domain_call_share=0.2,
        high_value_source_min_spans=2,event_source_min_spans=1)
    value.primary_call_count=2
    value.recent_valid_record_counts=[0,0,0,0]
    value.recent_model_call_counts=[1,1,1,1]
    return value

@pytest.mark.parametrize("disease",["dengue","measles"])
@pytest.mark.parametrize("template",[
    "In 2025 {disease} cases were monitored.",
    "After 2 weeks {disease} cases require reassessment.",
    "Table 12 describes {disease} cases.",
    "3 years since {disease} cases were first detected.",
])
def test_noncount_numbers_do_not_prolong_low_yield_budget(monkeypatch,disease,template):
    monkeypatch.setenv("PIPELINE_MODE","evidence")
    chunk={"source_id":"s","text":template.format(disease=disease)}
    assert _chunk_has_strong_record_signal(chunk) is False
    row={"budget_bucket":"ordinary","official_or_high_trust":False}
    value=ledger()
    assert value.can_attempt_primary(chunk,row,{}) == (False,"primary_soft_cap_no_uncovered_strong_signal")
    assert value.should_stop_low_yield(chunk,row,{}) is True

@pytest.mark.parametrize("text",[
    "Canada reported 12 confirmed measles cases.",
    "Brazil reported 18 000 suspected dengue cases.",
    "Brazil reported 2025 dengue cases.",
    "La Reunion rapporte 12 cas confirmes.",
])
def test_true_counts_and_grouped_counts_keep_signal(monkeypatch,text):
    monkeypatch.setenv("PIPELINE_MODE","evidence")
    assert _chunk_has_strong_record_signal({"source_id":"s","text":text}) is True

@pytest.mark.parametrize("revision",["legacy","standard"])
def test_legacy_count_signal_is_not_extended(monkeypatch,revision):
    monkeypatch.setenv("PIPELINE_MODE",revision)
    assert _chunk_has_strong_record_signal({"text":"Canada reported 12 confirmed measles cases."}) is False
    assert _chunk_has_strong_record_signal({"text":"PCR guidance for hantavirus."}) is True
