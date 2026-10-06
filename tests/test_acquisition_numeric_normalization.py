"""EVIDENCE normalization preserves claims for the shared source-bound qualifier."""
import socket
import pytest
from data_collection_workflow.config import load_llm_structured_extraction_policy
from data_collection_workflow.models import LLMExtractedRecord, LLMStructuredExtractionPolicy
from data_collection_workflow.nodes.extraction import _build_record_from_llm_output, schema_validation_and_repair
from data_collection_workflow.nodes.normalization import record_normalization
from data_collection_workflow.evidence_qualification import assess_record_evidence, build_evidence_index
from test_acquisition_evidence import evidence

@pytest.fixture(autouse=True)
def offline_evidence(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')
    def denied(*args, **kwargs):
        raise AssertionError('external network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', denied)

def normalize(text, disease, country, field, value):
    _, document, chunk = evidence(text)
    chunk.update(source_url='https://surveillance.example/report', source_type='unknown', confidence=0.9)
    output = LLMExtractedRecord(disease=disease, country=country, reporting_period='2025',
                               evidence_quote=text, **{field:value})
    built = _build_record_from_llm_output(output, chunk, 1,
        LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy()), {},
        {'disease_standard_name':disease, 'is_hantavirus':False})
    assert built is not None
    state = {'structured_task':{'disease':disease, 'location':country},
        'collection_spec':{'disease':disease, 'geography':country},
        'raw_records':[built.model_dump(mode='json')], 'documents':[document], 'evidence_chunks':[chunk],
        'source_registry':[{'source_id':'source', 'url':chunk['source_url'], 'source_type':'unknown'}],
        'collection_trace':[], 'human_review_queue':[]}
    state.update(schema_validation_and_repair(state))
    state.update(record_normalization(state))
    assert len(state['normalized_records']) == 1
    row = state['normalized_records'][0]
    qualification = assess_record_evidence(row, contract={}, evidence_index=build_evidence_index(state))
    return row, qualification

@pytest.mark.parametrize('disease,country', [('chikungunya','France'),('measles','Canada')])
@pytest.mark.parametrize('field,value,claim', [
    ('cases_unspecified',4156,'4 156 nouveaux cas'),
    ('cases_confirmed',4156,'4\u202f156 cas confirmés'),
    ('deaths',29,'29 décès'),
])
def test_french_count_survives_normalization_beside_percentage(disease,country,field,value,claim):
    text = f'En 2025, {country} a déclaré {claim} de {disease}, soit une hausse de 16%.'
    row, qualification = normalize(text,disease,country,field,value)
    assert row[field] == value
    count = next(item for item in qualification.field_evidence if item.field == field)
    assert count.supported, qualification.reasons
    assert qualification.status == 'qualified', qualification.reasons
    assert f'numeric_semantics_rejected:{field}' not in row.get('semantic_warnings',[])

@pytest.mark.parametrize('field,value,claim', [
    ('cases_unspecified',16,'4 156 nouveaux cas de chikungunya, une hausse de 16%'),
    ('cases_unspecified',18,'18 000 nouveaux cas de chikungunya, une hausse de 16%'),
    ('cases_confirmed',124,'124 Case investigation of chikungunya during 2025'),
    ('cases_confirmed',40,'40 countries monitored chikungunya in 2025, 4 confirmed cases'),
    ('deaths',29,'29 cas de chikungunya et 2 décès, une hausse de 16%'),
])
def test_wrong_quantity_is_retained_for_audit_but_never_supported(field,value,claim):
    text = f'France, en 2025: {claim}.'
    row, qualification = normalize(text,'chikungunya','France',field,value)
    assert row[field] == value
    count = next(item for item in qualification.field_evidence if item.field == field)
    assert not count.supported
    assert qualification.status != 'qualified'

def test_legacy_numeric_sanitizer_keeps_original_behavior(monkeypatch):
    from data_collection_workflow.config import load_record_normalization_policy
    from data_collection_workflow.models import RecordNormalizationPolicy
    from data_collection_workflow.nodes.normalization import _normalize_record
    monkeypatch.setenv('PIPELINE_MODE','standard')
    row = {'record_id':'legacy', 'source_id':'source', 'disease':'chikungunya',
           'cases_unspecified':4156, 'evidence_quote':'4 156 nouveaux cas, une hausse de 16%.'}
    normalized,_ = _normalize_record(row,RecordNormalizationPolicy(**load_record_normalization_policy()))
    assert normalized['cases_unspecified'] is None
    assert 'numeric_semantics_rejected:cases_unspecified' in normalized['semantic_warnings']
