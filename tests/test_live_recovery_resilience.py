"""Regression cases reproduced from the live evidence failure chain."""
import hashlib
import json
import pytest
from pydantic import ValidationError
from data_collection_workflow.models import SearchRefinementDecision
from data_collection_workflow.workflow_recovery import assess_collection_gaps, plan_recovery, recovery_control, RecoveryGap
from data_collection_workflow.session_runtime import RunContext


def test_missing_search_decision_cannot_become_stop_sufficient():
    with pytest.raises(ValidationError):
        SearchRefinementDecision(iteration_index=2, coverage_assessment='Unusable response')


def test_recovery_does_not_schedule_explicitly_ineligible_context():
    chunks = [
        {'chunk_id':'background', 'text':'general advice', 'contains_target_data':False},
        {'chunk_id':'wrong', 'text':'cases', 'extraction_eligible_for_task_disease':False},
        {'chunk_id':'other', 'text':'cases', 'disease_relevance_status':'unrelated_disease'},
        {'chunk_id':'eligible', 'text':'12 cases', 'contains_target_data':True},
    ]
    gaps=assess_collection_gaps({'evidence_chunks':chunks, 'source_registry':[{'source_id':'s'}]})
    assert [g.target_id for g in gaps if g.kind=='unprocessed_span']==['eligible']


def test_backlog_cannot_starve_affordable_discovery(tmp_path):
    gaps=[RecoveryGap('unprocessed_span',f'c{i:04}', 'pending') for i in range(900)]
    gaps.insert(0, RecoveryGap('parse_missing','doc','parse'))
    gaps.append(RecoveryGap('source_missing','annual','missing coverage'))
    text='France reported 12 confirmed measles cases during 2025.'
    (tmp_path/'saved.raw').write_bytes(text.encode())
    state={'structured_task':{'disease':'measles','location':'France',
                              'start_date':'2025-01-01','end_date':'2025-12-31'},
           'documents':[{'document_id':'doc','source_id':'source','clean_text':text,
                         'content_hash':hashlib.sha256(text.encode()).hexdigest(),
                         'content_readable':True,'parse_eligible':True,'raw_content_complete':True,
                         'request_success':True,'http_status_code':200,'raw_artifact_path':'saved.raw'}]}
    plan=plan_recovery(gaps,state=state,budget={'remaining':{'search':10,'fetch':20,'fetch_ordinary':10,'extraction':100}})
    assert len(plan.actions)==8
    assert [a.kind for a in plan.actions][0]=='reparse'
    assert plan.actions[-1].kind=='search'
    assert any(a.kind=='extract' for a in plan.actions)


def evidence_state():
    text='Measles in France: 12 confirmed cases during 2024.'
    digest=hashlib.sha256(text.encode()).hexdigest()
    return {'structured_task':{'disease':'measles','location':'France'},
            'source_coverage_requirements':[{'requirement_id':'annual','disease':'measles','country':'France','reporting_period':'2024'}],
            'normalized_records':[{'record_id':'r','chunk_id':'c','disease':'measles','country':'France','reporting_period':'2024','cases_confirmed':12}],
            'documents':[{'source_id':'s','content_hash':digest,'clean_text':text}],
            'evidence_chunks':[{'source_id':'s','chunk_id':'c','text':text,'document_hash':digest}],
            'source_registry':[{'source_id':'s'}], 'extraction_attempted_chunk_ids':['c']}


def test_loop_boundary_assesses_current_records_before_progress_and_coverage():
    state=evidence_state()
    state.update(recovery_round=1,recovery_previous_gain=[])
    result=recovery_control(state)
    assert result['recovery_previous_gain']
    assert result['source_coverage_audit']['coverage_status']=='complete'
    assert result['recovery_stop_reason']=='coverage_satisfied'
    assert len(result['qualified_records'])==1


def test_loop_boundary_does_not_trust_stale_qualification():
    state=evidence_state()
    state['normalized_records'][0]['cases_confirmed']=999
    state['qualified_records']=[dict(state['normalized_records'][0], evidence_qualification={'status':'qualified','product_kind':'aggregate'})]
    result=recovery_control(state)
    assert result['qualified_records']==[]
    assert result['source_coverage_audit']['coverage_status']=='incomplete'


def test_failed_raw_response_is_auditable_but_not_valid_cache(tmp_path):
    context=RunContext(tmp_path,{})
    error=ValueError('invalid response schema')
    error.llm_raw_response={'content':'{"records":"invalid"}', 'model':'test-model'}
    calls=[]
    def fail():
        calls.append(True)
        raise error
    with pytest.raises(ValueError): context.call('model:Test',{'input':1},fail)
    assert not context.has_cached('model:Test',{'input':1})
    with context.ledger._db() as db:
        row=db.execute('SELECT status,response FROM operations').fetchone()
    assert row['status']=='failed'
    assert json.loads(row['response'])==error.llm_raw_response
    with pytest.raises(ValueError): context.call('model:Test',{'input':1},fail)
    assert len(calls)==2


@pytest.mark.parametrize('status', ['http_error', 'blocked', 'error_page'])
def test_failed_acquisition_body_is_not_reparse_work(status):
    state = {'source_registry': [{'source_id': 'failed', 'url': 'https://example.org/report', 'final_screening_decision': 'include_for_content_fetch'}],
             'documents': [{'source_id': 'failed', 'document_id': 'doc',
                            'raw_artifact_path': 'saved-error.html', 'clean_text': 'Access denied',
                            'content_hash': 'digest', 'content_readable': False,
                            'acquisition_status': status, 'parse_status': 'failed'}]}
    gaps = assess_collection_gaps(state)
    assert [(gap.kind, gap.target_id) for gap in gaps] == [('fetch_failed', 'failed')]


def test_saved_parser_failure_remains_recoverable():
    state = {'source_registry': [{'source_id': 's'}],
             'documents': [{'source_id': 's', 'document_id': 'doc',
                            'raw_artifact_path': 'saved-report.pdf', 'content_hash': 'digest',
                            'content_readable': False, 'acquisition_status': 'parse_error',
                            'parse_status': 'failed'}]}
    assert any(g.kind == 'parse_failed' for g in assess_collection_gaps(state))


def test_failed_extraction_retries_precede_unprocessed_backlog():
    gaps = [RecoveryGap('unprocessed_span', f'a{i:04}', 'pending') for i in range(900)]
    gaps.append(RecoveryGap('extraction_failed', 'z_failed', 'one retry remains'))
    plan = plan_recovery(gaps, state={}, budget={'remaining': {'extraction': 100}})
    assert len(plan.actions) == 8
    assert plan.actions[0].target_id == 'z_failed'
    assert all(action.kind == 'extract' for action in plan.actions)


@pytest.mark.parametrize('domain', ['who.int', 'ecdc.europa.eu'])
def test_recovery_preserves_explicit_verified_requirement_domain(domain):
    from data_collection_workflow.workflow_recovery import _recovery_query
    state = {'structured_task': {'disease': 'Pertussis', 'location': 'United States',
             'start_date': '2025-01-01', 'end_date': '2025-12-31'},
             'source_coverage_audit': {'requirements': [
                 {'requirement_id': 'external', 'official_domains': [domain]}]}}
    query = _recovery_query(state, 'external')
    assert query and query['official_domain_hint'] == domain
    assert 'site:' + domain in query['query']


def test_unverified_requirement_domain_does_not_fall_back_to_another_authority():
    from data_collection_workflow.workflow_recovery import _recovery_query
    state = {'structured_task': {'disease': 'Pertussis', 'location': 'United States'},
             'source_coverage_audit': {'requirements': [
                 {'requirement_id': 'external', 'official_domains': ['unverified.invalid']}]}}
    assert _recovery_query(state, 'external') is None
