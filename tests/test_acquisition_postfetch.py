"""Metadata hypotheses can change; explicit constraints and source binding cannot."""
import pytest

from data_collection_workflow.nodes.source_screening import assess_source_task_fit

STATE={'structured_task':{'disease':'dengue','location':'Canada','start_date':'2025-01-01','end_date':'2025-12-31'}}


def source():
    return {'source_id':'s','title':'Dengue Canada report','disease_fit':'match','geography_fit':'match',
        'date_fit':'mismatch','task_fit_evidence_origin':'discovery_metadata','task_fit_assessment_version':1}


def document():
    return {'source_id':'s','content_readable':True,'request_success':True,'acquisition_status':'readable',
        'content_hash':'version2','clean_text':'Canada recorded 12 dengue cases during 2025.'}


def test_same_source_body_replaces_provisional_metadata_mismatch():
    result=assess_source_task_fit(source(),STATE,document=document())
    assert result['date_fit']=='match'
    assert result['task_fit_evidence_origin']=='fetched_content'
    assert result['task_fit_content_hash']=='version2'


def test_explicit_constraint_survives_positive_body():
    entry=source();entry['task_fit_evidence_origin']='explicit_constraint'
    assert assess_source_task_fit(entry,STATE,document=document())['date_fit']=='mismatch'


def test_search_title_is_not_promoted_to_received_page_evidence():
    entry=source();entry.pop('date_fit');entry['title']='Dengue Canada during 2025: 12 cases'
    doc=document();doc.update(title=entry['title'],clean_text='Zika in Brazil during 2024: 4 cases.')
    result=assess_source_task_fit(entry,STATE,document=doc)
    assert result['target_verification_status']!='verified_target'
    assert result['disease_fit']!='match'


@pytest.mark.parametrize("origin", [None, "explicit_constraint", "fetched_content"])
def test_only_discovery_mismatch_can_be_replaced(origin):
    entry=source()
    entry['task_fit_evidence_origin']=origin
    first=assess_source_task_fit(entry,STATE)
    second=assess_source_task_fit({**entry,**first},STATE,document=document())
    assert second['date_fit']=='mismatch'
    assert second['task_fit_evidence_origin']!='discovery_metadata'


@pytest.mark.parametrize('changes', [
    {'source_id':'another'}, {'content_readable':False}, {'request_success':False},
    {'parse_eligible':False}, {'raw_content_complete':False}, {'acquisition_incomplete':True},
    {'http_status_code':302}, {'http_status_code':500},
])
def test_unbound_or_incomplete_response_cannot_replace_discovery_mismatch(changes):
    result=assess_source_task_fit(source(),STATE,document={**document(),**changes})
    assert result['date_fit']=='mismatch'
    assert result['task_fit_evidence_origin']=='discovery_metadata'
    assert result['task_fit_content_hash'] is None


def test_provisional_disease_classifier_can_be_replaced_by_same_source_body():
    entry={**source(),'source_disease_relevance_status':'unrelated_disease','disease_fit':'mismatch'}
    result=assess_source_task_fit(entry,STATE,document=document())
    assert result['disease_fit']=='match'
    assert result['target_verification_status']=='verified_target'


def test_discovery_assessment_provenance_survives_registry_model(monkeypatch):
    from data_collection_workflow.nodes.source_screening import source_screening
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    entry={'source_id':'s','canonical_url':'https://data.example/report','status':'registered',
           'title':'Dengue in Canada during 2024'}
    result=source_screening({**STATE,'source_registry':[entry]})
    row=result['source_registry'][0]
    assert row['date_fit']=='mismatch'
    assert row['task_fit_evidence_origin']=='discovery_metadata'
    assert row['task_fit_assessment_version']==2
    assert row['task_fit_content_hash'] is None
    assert assess_source_task_fit(row,STATE,document=document())['date_fit']=='match'


@pytest.mark.parametrize('disease_mismatch', [False,True])
def test_content_node_replaces_provisional_routing_with_same_source_body(tmp_path, monkeypatch, disease_mismatch):
    import requests
    from test_evidence_resource_discovery import _env,_source
    from test_acquisition_transport import _adaptive_runtime
    from data_collection_workflow.nodes.content_processing import content_fetch_and_parse
    _env(monkeypatch)
    url='https://data.example/report'
    entry=_source(url)
    entry.update(title='Dengue Canada report',date_fit='mismatch',target_fit_status='context_only',
        target_verification_status='temporal_mismatch',source_role='context_source',source_role_final='context',
        final_screening_decision='include_for_context_fetch',task_fit_evidence_origin='discovery_metadata',
        task_fit_assessment_version=1)
    if disease_mismatch:
        entry.update(disease_fit='mismatch',source_disease_relevance_status='unrelated_disease')
    excluded=_source('https://data.example/excluded',2)
    excluded.update(blocked_from_fetch=True,blocked_from_fetch_reason='user_excluded',
        source_role_final='excluded',final_screening_decision='exclude')
    visited=[]
    class Response:
        status_code=200;headers={'content-type':'text/plain'}
        def __init__(self,url):self.url=url
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,size):yield b'Canada recorded 12 dengue cases during 2025.'
    def get(url,**kwargs):visited.append(url);return Response(url)
    monkeypatch.setattr(requests,'get',get)
    runtime=_adaptive_runtime(tmp_path)
    with runtime.activate():
        result=content_fetch_and_parse({**STATE,'source_registry':[entry,excluded],'collection_trace':[]})
    assert visited==[url]
    row=next(item for item in result['source_registry'] if item['source_id']==entry['source_id'])
    assert row['target_verification_status']=='verified_target'
    assert row['target_fit_status']=='verified_target'
    assert row['source_role_final']=='collection'
    assert row['final_screening_decision']=='include_for_content_fetch'
    assert row.get('source_disease_relevance_status')!='unrelated_disease'
    doc=result['documents'][0]
    assert doc['source_role_final']=='collection'
    assert row['task_fit_content_hash']==doc['content_hash']
    saved=runtime.frontier.snapshot()['items'][0]['result_ref']
    import json
    artifact=json.loads((tmp_path/saved).read_text(encoding='utf-8'))
    assert artifact['entry']['target_fit_status']=='verified_target'


def test_content_node_does_not_keep_search_target_proof_for_unrelated_received_body(tmp_path, monkeypatch):
    import requests
    from test_evidence_resource_discovery import _env,_source
    from test_acquisition_transport import _adaptive_runtime
    from data_collection_workflow.nodes.content_processing import content_fetch_and_parse
    _env(monkeypatch)
    entry=_source('https://data.example/report')
    entry.update(title='Dengue Canada during 2025: 12 cases',disease_fit='match',geography_fit='match',date_fit='match',
        target_fit_status='verified_target',target_verification_status='verified_target',
        task_fit_evidence_origin='discovery_metadata',task_fit_assessment_version=1)
    class Response:
        status_code=200;headers={'content-type':'text/plain'}
        url=entry['url']
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,size):yield b'Zika in Brazil during 2024: 4 cases.'
    monkeypatch.setattr(requests,'get',lambda *args,**kwargs:Response())
    runtime=_adaptive_runtime(tmp_path)
    with runtime.activate():result=content_fetch_and_parse({**STATE,'source_registry':[entry],'collection_trace':[]})
    row=result['source_registry'][0]
    assert row['target_fit_status']!='verified_target'
    assert row['target_verification_status']!='verified_target'
    assert row['disease_fit']!='match'


@pytest.mark.parametrize('boundary', [
    {'blocked_from_fetch':True,'blocked_from_fetch_reason':'user_excluded'},
    {'source_role_final':'excluded','final_screening_decision':'exclude'},
    {'source_role_final':'validation'}, {'routing_flags':['context_only']},
    {'requires_human_review':True},
])
def test_same_source_positive_body_cannot_replace_explicit_routing_boundary(boundary):
    entry={**source(),'source_role_final':'context','final_screening_decision':'include_for_context_fetch',**boundary}
    result=assess_source_task_fit(entry,STATE,document=document())
    assert 'source_role_final' not in result
    assert 'final_screening_decision' not in result


def test_discovery_origin_cannot_override_human_exclusion_without_legacy_block_flag(tmp_path, monkeypatch):
    from test_acquisition_queue import setup_node
    from data_collection_workflow.nodes.content_processing import content_fetch_and_parse
    runtime,state,visits=setup_node(monkeypatch,tmp_path)
    entry=state['source_registry'][0]
    entry.update(task_fit_evidence_origin='discovery_metadata',source_excluded_by_human_review=True,
        source_role_final='excluded',final_screening_decision='exclude')
    with runtime.activate():content_fetch_and_parse(state)
    assert visits==[state['source_registry'][1]['url']]


def test_partial_parser_output_does_not_replace_provisional_metadata_mismatch():
    partial={**document(),'acquisition_status':'parse_error','parse_status':'parsed_partial',
             'parse_error':'failed reading a later PDF page'}
    result=assess_source_task_fit(source(),STATE,document=partial)
    assert result['date_fit']=='mismatch'
    assert result['task_fit_evidence_origin']=='discovery_metadata'
    assert result['task_fit_content_hash'] is None


@pytest.mark.parametrize('origin', [None,'explicit_constraint','discovery_metadata'])
def test_aggregate_wrong_period_boundary_needs_explicit_discovery_origin_to_be_replaced(origin):
    entry={'source_id':'s','target_fit_status':'wrong_period','task_fit_evidence_origin':origin}
    result=assess_source_task_fit(entry,STATE,document=document())
    if origin=='discovery_metadata':
        assert result['target_fit_status']=='verified_target'
    else:
        assert result['target_fit_status']=='wrong_period'
        assert result['date_fit']=='mismatch'
        assert result['task_fit_evidence_origin']=='explicit_constraint'
