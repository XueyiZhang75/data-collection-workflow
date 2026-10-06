"""Positive non-case metrics and the actual model-failure deterministic route."""
import pytest
from test_deterministic_source_binding import _run,_diagnostics
from test_deterministic_disease_aliases import make_state
from data_collection_workflow.nodes import extraction
from data_collection_workflow.nodes.normalization import record_normalization
from data_collection_workflow.evidence_qualification import assess_record_evidence,build_evidence_index


@pytest.fixture(autouse=True)
def evidence(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    monkeypatch.setenv('ENABLE_LLM_EXTRACTION','false')
    monkeypatch.setenv('ENABLE_LANGSMITH_TRACE','false')


@pytest.mark.parametrize('text,metric,value,unit,status',[
    ('During 2025, the Pertussis incidence rate was 23 per 100,000 population in France.','incidence rate',23,'per 100,000 population','qualified'),
    ('During 2025, 12 hospitalizations for Pertussis were reported in France.','hospitalizations',12,'count','qualified'),
])
def test_explicit_non_case_metrics_survive_public_deterministic_path(tmp_path,text,metric,value,unit,status):
    state=_run(tmp_path,text)
    rows=[r for r in state.get('raw_records') or [] if r.get('metric_name')==metric]
    assert rows,_diagnostics(state)
    assert any(r.get('metric_value')==value and r.get('metric_unit')==unit and r.get('reporting_period')=='2025' for r in rows)
    assert all(not r.get('date_reported') and not r.get('metric_period_start') and not r.get('metric_period_end') for r in rows)
    assert any(r['record_id'] in {v['record_id'] for v in rows} and r['evidence_qualification']['status']==status
               for r in state['assessed_records']),_diagnostics(state)


def test_model_failure_falls_back_without_task_supplied_dates(tmp_path,monkeypatch):
    text='In 2025, 23 Pertussis cases were reported in France.'
    state=make_state(tmp_path,'Pertussis',text)
    state['structured_task']['location']='France'
    monkeypatch.setattr(extraction.llm_clients,'llm_extraction_enabled',lambda:True)
    monkeypatch.setattr(extraction.llm_clients,'llm_fallback_to_rule_based',lambda:True)
    monkeypatch.setattr(extraction.llm_clients,'get_llm_settings',lambda:{'provider':'anthropic','model':'offline-test'})
    monkeypatch.setenv('LLM_FOCUSED_RECOVERY_ENABLED','false')
    calls=[]
    def failure(chunk,policy):
        calls.append(chunk['chunk_id'])
        raise RuntimeError('offline mocked model failure')
    monkeypatch.setattr(extraction.llm_clients,'extract_chunk_with_llm',failure)
    state.update(extraction.structured_extraction(state))
    assert calls and state['llm_extraction_summary']['llm_fallback_count']>0
    state.update(extraction.schema_validation_and_repair(state));state.update(record_normalization(state))
    rows=[row for row in state['normalized_records'] if row.get('cases_unspecified')==23]
    assert rows
    assert all(not row.get('date_reported') and not row.get('metric_period_start') and row.get('reporting_period')=='2025' for row in rows)
    assert any(assess_record_evidence(row,contract={},evidence_index=build_evidence_index(state)).status=='qualified' for row in rows)


@pytest.mark.parametrize('source,facts',[
    ('During 2025, the Pertussis incidence rate was 23 per 100,000 population in France. The mortality rate was 4 per 1,000 population.',
     {'incidence_rate':23,'metric_name':'incidence rate','metric_value':23,'metric_unit':'per 1,000 population','metric_denominator':'1,000 population'}),
    ('During 2025, the Pertussis incidence rate was 23 per 100,000 population in France.',
     {'deaths':23,'metric_name':'deaths','metric_value':23,'metric_unit':'count'}),
    ('During 2025, the Pertussis incidence rate was 23 per 100,000 population in France.',
     {'incidence_rate':23,'metric_name':'incidence rate','metric_value':23,'metric_unit':'count','metric_denominator':'100,000 population'}),
    ('During 2025, 23% of Pertussis cases were reported in France.',
     {'cases_unspecified':23,'metric_name':'cases','metric_value':23,'metric_unit':'count'}),
])
def test_rate_value_units_and_count_types_cannot_be_borrowed(tmp_path,source,facts):
    from test_evidence_positive_extraction import source_state,qualified_outputs
    state=source_state(tmp_path,source.encode(),'text/plain')
    rows,_=qualified_outputs(state,{'disease':'Pertussis','country':'France','reporting_period':'2025',**facts})
    assert not rows
