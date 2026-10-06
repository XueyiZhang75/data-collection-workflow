"""EVIDENCE task metadata must not manufacture observation fields during extraction."""
import pytest
from data_collection_workflow.models import LLMExtractedRecord, LLMStructuredExtractionPolicy
from data_collection_workflow.config import load_llm_structured_extraction_policy
from data_collection_workflow.nodes import extraction


def _context(location):
    return {'collection_mode':'direct_collection','disease_standard_name':'Dengue','is_hantavirus':False,
            'structured_task':{'disease':'Dengue','location':location,'start_date':'2024-01-01','end_date':'2024-12-31'}}


def _chunk():
    return {'source_id':'s','chunk_id':'c','text':'Dengue surveillance identified 12 confirmed cases.',
            'source_url':'https://example.invalid/report','source_type':'official_public_health_agency',
            'source_role_final':'collection','reporting_period_start':'2024-01-01',
            'reporting_period_end':'2024-12-31','reporting_period_label':'2024'}


@pytest.mark.parametrize('location',['United States','Brazil','Unknown District'])
def test_evidence_record_builder_does_not_inherit_task_geography_or_year_end_date(monkeypatch,location):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    monkeypatch.setenv('COLLECTION_MODE','direct_collection')
    row=extraction._build_record_from_llm_output(
        LLMExtractedRecord(disease='Dengue',cases_confirmed=12,metric_name='cases',metric_value=12),
        _chunk(),1,LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy()),{},_context(location))
    assert row is not None
    result=row.model_dump()
    for field in ('country','geographic_scope','geographic_scope_type','date_reported','date_anchor','reporting_period'):
        assert not result.get(field), (field,result.get(field))
    assert result['cases_confirmed']==12


def test_evidence_verified_source_priority_does_not_prove_observation_period(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    monkeypatch.setenv('COLLECTION_MODE','direct_collection')
    monkeypatch.setattr(extraction,'_chunk_priority',lambda *args:1)
    record={'disease':'Dengue','metric_name':'cases','metric_value':12}
    result=extraction._fill_metric_period_from_verified_source(record,_chunk(),_context('Brazil'))
    assert 'metric_period_start' not in result and 'metric_period_end' not in result


def test_evidence_preserves_explicit_extracted_geography_and_period(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    record={'metric_name':'cases','metric_value':12,'country':'Brazil','geographic_scope':'Brazil',
            'reporting_period':'2023','date_reported':'2024-02-15','metric_period_start':'2023-01-01','metric_period_end':'2023-12-31'}
    assert extraction._inherit_task_source_context_for_metric_record(record,_chunk(),{},_context('United States'))==record
    assert extraction._fill_metric_period_from_verified_source(record,_chunk(),_context('United States'))==record


def test_legacy_task_context_fallback_remains_available(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','standard')
    result=extraction._inherit_task_source_context_for_metric_record({'metric_name':'cases','metric_value':12},_chunk(),{},_context('United States'))
    assert result['country']=='United States' and result['date_reported']=='2024-12-31'
