"""A legacy outbreak keyword must not prolong a general evidence extraction budget."""
import pytest
from data_collection_workflow.nodes.extraction import ExtractionBudgetLedger, _chunk_has_strong_record_signal


def ledger():
    value = ExtractionBudgetLedger(scheduler_mode='quality_adaptive', max_concurrency=4,
        soft_checkpoint_calls=2, safety_max_calls=12, rolling_yield_window=4,
        focused_recovery_reserved_calls=2, max_domain_call_share=0.2,
        high_value_source_min_spans=2, event_source_min_spans=1)
    value.primary_call_count = 2
    value.recent_valid_record_counts = [0, 0, 0, 0]
    value.recent_model_call_counts = [1, 1, 1, 1]
    return value


@pytest.mark.parametrize('marker', ['hantavirus', 'Andes virus', 'ANDV', 'HPS', 'Hondius'])
@pytest.mark.parametrize('disease,country', [('mpox', 'Sierra Leone'), ('chikungunya', 'Reunion'), ('hantavirus', 'Canada')])
def test_evidence_clinical_keyword_without_case_evidence_has_no_special_extension(monkeypatch, marker, disease, country):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    # Background diagnostic guidance has no patient, observation or count.
    chunk = {'source_id': 's', 'text': f'{country}: PCR guidance for {marker} laboratory surveillance.'}
    context = {'structured_task': {'disease': disease, 'country': country}}
    budget_row = {'budget_bucket': 'ordinary', 'official_or_high_trust': False}
    value = ledger()
    assert _chunk_has_strong_record_signal(chunk) is False
    assert value.can_attempt_primary(chunk, budget_row, context) == (False, 'primary_soft_cap_no_uncovered_strong_signal')
    assert value.should_stop_low_yield(chunk, budget_row, context) is True


@pytest.mark.parametrize('marker', ['hantavirus', 'Andes virus', 'ANDV', 'HPS', 'Hondius'])
def test_legacy_clinical_priority_is_preserved(monkeypatch, marker):
    monkeypatch.setenv('PIPELINE_MODE', 'standard')
    chunk = {'source_id': 's', 'text': f'PCR guidance for {marker} laboratory surveillance.'}
    budget_row = {'budget_bucket': 'ordinary', 'official_or_high_trust': False}
    assert _chunk_has_strong_record_signal(chunk) is True
    assert ledger().can_attempt_primary(chunk, budget_row, {}) == (True, None)
    assert ledger().should_stop_low_yield(chunk, budget_row, {}) is False


@pytest.mark.parametrize('text', [
    'Patient 7 was confirmed with mpox in Sierra Leone.',
    'An adult female was hospitalized with chikungunya in Reunion.',
    'Canada reported 12 confirmed measles cases.',
    'Peru reported 3 confirmed cholera cases.',
    'Patient 2 was confirmed with hantavirus in Canada.',
    'La Reunion rapporte 12 cas confirmes.',
    'Canada reported 7 confirmed new disease cases.',
    'A hospital treated 2 patients.',
])
def test_evidence_generic_patient_or_numeric_evidence_retains_priority(monkeypatch, text):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    chunk = {'source_id': 's', 'text': text}
    assert _chunk_has_strong_record_signal(chunk) is True
    assert ledger().can_attempt_primary(chunk, {'budget_bucket':'ordinary'}, {}) == (True, None)


def test_evidence_source_bound_patient_span_retains_clinical_signal(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    chunk = {'source_id':'s','text':'PCR confirmed the infection.',
             'case_span_quote':'The woman was admitted with fever. PCR confirmed the infection.'}
    assert _chunk_has_strong_record_signal(chunk) is True


@pytest.mark.parametrize('disease', ['mpox', 'measles', 'chikungunya'])
def test_pcr_alone_does_not_replace_the_legacy_special_case_with_a_broad_boost(monkeypatch, disease):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    assert _chunk_has_strong_record_signal({'text':f'PCR guidance for {disease} laboratory surveillance.'}) is False
