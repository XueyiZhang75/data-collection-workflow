"""Cross-stage regressions from comprehensive review passes 07-12.
Run with PYTHONPATH=src;tests; no external services are used.
"""
import base64
import io
import json
import pytest
from test_evidence_positive_extraction import source_state,qualified_outputs
from data_collection_workflow.evidence_qualification import assess_record_evidence,build_evidence_index
from data_collection_workflow.evidence_products import build_evidence_products
from data_collection_workflow import document_acquisition as acq
from data_collection_workflow.nodes.content_processing import evidence_chunking_and_data_presence_flagging

@pytest.fixture(autouse=True)
def evidence(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')

FACTS=dict(disease='Pertussis',country='France',reporting_period='2025',cases_confirmed=12)

def test_07_native_pdf_page_survives_unrelated_low_confidence_ocr(tmp_path,monkeypatch):
    from reportlab.pdfgen.canvas import Canvas
    b=io.BytesIO();canvas=Canvas(b);canvas.showPage()
    canvas.drawString(30,740,'Pertussis in France during 2025: 12 confirmed cases.');canvas.save()
    monkeypatch.setattr(acq,'_ocr',lambda *a,**k:{'text':'Scan image unreadable','words':[{'text':'Scan','confidence':20}]})
    state=source_state(tmp_path,b.getvalue(),'application/pdf')
    assert any(s.get('page')==2 for s in state['documents'][0]['locator_spans'])
    rows,attempts=qualified_outputs(state,{**FACTS,'evidence_quote':'Pertussis in France during 2025: 12 confirmed cases.'})
    assert rows,[(c['text'],q.reasons) for c,q in attempts]


def test_08_two_browser_csv_responses_keep_distinct_table_ownership(tmp_path,monkeypatch):
    from test_live_acquisition_resilience import fake_transport,CHALLENGE
    from data_collection_workflow.session_runtime import RunContext
    fake_transport(monkeypatch,CHALLENGE)
    responses=[('fr','disease,country,reporting_period,cases_confirmed\nPertussis,France,2025,12'),
               ('ca','disease,country,reporting_period,cases_confirmed\nPertussis,Canada,2025,7')]
    monkeypatch.setattr(acq,'_browser',lambda url,config:{'body':base64.b64encode(b'<h1>Pertussis statistical data</h1>').decode(),
       'content_type':'text/html','status_code':200,'final_url':url,'browser_responses':[
        {'url':'https://data.invalid/'+name+'.csv','content_type':'text/csv','status_code':200,'body':base64.b64encode(body.encode()).decode()} for name,body in responses]})
    runtime=RunContext(tmp_path/'run',{'universal':{'budget_limits':{'fetch':2,'fetch_ordinary':1,'browser':1}}})
    with runtime.activate():doc=acq.acquire_document('https://authority.invalid/report',source_id='source',session_dir=runtime.session_dir)
    doc.update(document_id='doc',quality_status='usable',extraction_readiness='ready')
    state={'documents':[doc],'structured_task':{'disease':'Pertussis'}}
    state.update(evidence_chunking_and_data_presence_flagging(state))
    ids=[t['table_id'] for t in doc['tables']]
    rows=[c for c in state['evidence_chunks'] if c.get('row_id')]
    assert len(ids)==len(set(ids)),{'table_ids':ids,'rows':[(c['table_id'],c['row_id'],c['text'],c['char_start']) for c in rows]}
    assert len(rows)==2
    assert qualified_outputs(state,FACTS)[0]
    assert qualified_outputs(state,{**FACTS,'country':'Canada','cases_confirmed':7})[0]


def test_09_json_envelope_scope_reaches_leaf_observation(tmp_path):
    body=json.dumps({'disease':'Pertussis','country':'France','reporting_period':'2025','observations':[{'cases_confirmed':12}]}).encode()
    state=source_state(tmp_path,body,'application/json')
    rows,attempts=qualified_outputs(state,FACTS)
    assert rows,{'chunks':[(c['text'],c['contains_target_data'],c['bound_context_spans']) for c in state['evidence_chunks']],
                 'attempts':[(c['text'],q.reasons) for c,q in attempts]}


@pytest.mark.parametrize('country',['Canada','France'])
def test_10_explicit_annual_rule_record_keeps_source_period(tmp_path,country):
    from data_collection_workflow.nodes.extraction import _official_outbreak_record_from_chunk
    from data_collection_workflow.config import load_structured_extraction_policy
    from data_collection_workflow.models import StructuredExtractionPolicy
    text=f'Pertussis in {country}: 12 cases were reported during 2025.'
    state=source_state(tmp_path,text.encode(),'text/plain')
    chunk=next(c for c in state['evidence_chunks'] if text in c['text'])
    policy=StructuredExtractionPolicy(**load_structured_extraction_policy())
    record,_=_official_outbreak_record_from_chunk(chunk,1,policy,{'disease_standard_name':'Pertussis','disease_terms':['pertussis'],
        'structured_task':{'disease':'Pertussis','start_date':'2025-01-01','end_date':'2025-12-31'}})
    assert record is not None
    from data_collection_workflow.nodes.normalization import record_normalization
    normalized = record_normalization({**state,'validated_records':[record.model_dump()]})['normalized_records']
    assert normalized
    q=assess_record_evidence(normalized[0],contract={},evidence_index=build_evidence_index(state))
    assert q.status=='qualified',{'period':record.reporting_period,'reasons':q.reasons}


def test_11_restarted_local_patient_labels_do_not_merge_across_sections(tmp_path):
    body=b'<h1>Pertussis surveillance</h1><h2>France</h2><p>Patient 1 with Pertussis in France was aged 45.</p><h2>Canada</h2><p>Patient 1 with Pertussis in Canada was aged 30.</p>'
    state=source_state(tmp_path,body,'text/html')
    observations=[]
    for country,age in [('France','45'),('Canada','30')]:
        rows,attempts=qualified_outputs(state,{'disease':'Pertussis','country':country,'workflow_case_label':'Patient 1','age':age})
        assert rows,[(c['text'],q.reasons) for c,q in attempts]
        observations.extend(rows)
    products=build_evidence_products(observations,evidence_index=build_evidence_index(state))
    assert len(products['case_entities'])==2,products['case_entities']


def test_12_candidate_field_gap_is_not_declared_coverage_satisfied(tmp_path):
    from data_collection_workflow.workflow_recovery import recovery_control
    from data_collection_workflow.session_runtime import RunContext
    state=source_state(tmp_path,b'Pertussis in France during 2025: 12 confirmed cases.','text/plain')
    chunk=state['evidence_chunks'][0]
    state.update(source_registry=[{'source_id':'source','url':'https://authority.invalid/report','source_role_final':'collection'}],
        extraction_attempted_chunk_ids=[chunk['chunk_id']],raw_records=[{'record_id':'r','source_id':'source','supporting_chunk_id':chunk['chunk_id'],
          'disease':'Pertussis','country':'France','cases_confirmed':12}],normalized_records=[{'record_id':'r','source_id':'source','supporting_chunk_id':chunk['chunk_id'],
          'disease':'Pertussis','country':'France','cases_confirmed':12}])
    runtime=RunContext(tmp_path/'run',{'universal':{'budget_limits':{'fetch':5,'fetch_ordinary':5,'extraction':5,'search':5}}})
    with runtime.activate():result=recovery_control(state)
    assert result['candidate_records']
    assert result['recovery_stop_reason']!='coverage_satisfied',result['recovery_plan']



def test_07_low_confidence_ocr_fact_stays_candidate(tmp_path,monkeypatch):
    from reportlab.pdfgen.canvas import Canvas
    b=io.BytesIO();canvas=Canvas(b);canvas.showPage();canvas.save()
    monkeypatch.setattr(acq,'_ocr',lambda *a,**k:{'text':'Pertussis in France during 2025: 12 confirmed cases.',
        'words':[{'text':'12','confidence':20}]+[{'text':'other','confidence':99} for _ in range(30)]})
    state=source_state(tmp_path,b.getvalue(),'application/pdf')
    assert not qualified_outputs(state,FACTS)[0]


def test_09_child_json_scope_overrides_ancestor_and_never_inherits_parent_count(tmp_path):
    body=json.dumps({'disease':'Pertussis','country':'France','reporting_period':'2025','cases_confirmed':99,
                     'observations':[{'country':'Canada','cases_confirmed':7},{'population_scope':'children'}]}).encode()
    state=source_state(tmp_path,body,'application/json')
    assert qualified_outputs(state,{**FACTS,'country':'Canada','cases_confirmed':7})[0]
    assert not qualified_outputs(state,{**FACTS,'cases_confirmed':7})[0]
    assert qualified_outputs(state,{**FACTS,'cases_confirmed':99})[0]
    assert not qualified_outputs(state,{**FACTS,'cases_confirmed':99,'population_scope':'children'})[0]


@pytest.mark.parametrize('sentence',[
    'Pertussis in Canada: 12 cases were reported. Published in 2025.',
    'Pertussis in Canada: 12 cases were reported on 2025-03-01.',
])
def test_10_no_annual_period_from_publication_or_day_precision(tmp_path,sentence):
    from data_collection_workflow.nodes.extraction import _official_outbreak_record_from_chunk
    from data_collection_workflow.config import load_structured_extraction_policy
    from data_collection_workflow.models import StructuredExtractionPolicy
    state=source_state(tmp_path,sentence.encode(),'text/plain')
    chunk=state['evidence_chunks'][0]
    row,_=_official_outbreak_record_from_chunk(chunk,1,StructuredExtractionPolicy(**load_structured_extraction_policy()),
        {'disease_standard_name':'Pertussis','disease_terms':['pertussis'],'structured_task':{'disease':'Pertussis','start_date':'2025-01-01','end_date':'2025-12-31'}})
    assert row is not None and row.reporting_period is None


def test_11_exact_repeated_observation_does_not_duplicate_patient(tmp_path):
    state=source_state(tmp_path,b'Patient 1 with Pertussis in Canada was aged 30.','text/plain')
    rows,_=qualified_outputs(state,{'disease':'Pertussis','country':'Canada','workflow_case_label':'Patient 1','age':'30'})
    assert rows
    products=build_evidence_products(rows+rows,evidence_index=build_evidence_index(state))
    assert len(products['case_entities'])==1



def test_08_browser_json_scope_reaches_qualified_evidence(tmp_path,monkeypatch):
    from test_live_acquisition_resilience import fake_transport,CHALLENGE
    from data_collection_workflow.session_runtime import RunContext
    fake_transport(monkeypatch,CHALLENGE)
    body=json.dumps({'disease':'Pertussis','country':'France','reporting_period':'2025','observations':[{'cases_confirmed':12}]}).encode()
    monkeypatch.setattr(acq,'_browser',lambda url,config:{'body':base64.b64encode(b'<h1>Pertussis data portal</h1>').decode(),
        'content_type':'text/html','status_code':200,'final_url':url,'browser_responses':[
          {'url':'https://data.invalid/report','content_type':'application/json','status_code':200,'body':base64.b64encode(body).decode()}]})
    runtime=RunContext(tmp_path/'run',{'universal':{'budget_limits':{'fetch':2,'fetch_ordinary':1,'browser':1}}})
    with runtime.activate():doc=acq.acquire_document('https://authority.invalid/report',source_id='source',session_dir=runtime.session_dir)
    doc.update(document_id='doc',quality_status='usable',extraction_readiness='ready')
    state={'documents':[doc],'structured_task':{'disease':'Pertussis'}}
    state.update(evidence_chunking_and_data_presence_flagging(state))
    rows,attempts=qualified_outputs(state,FACTS)
    assert rows,[(c['text'],q.reasons) for c,q in attempts]
    assert build_evidence_products(rows,evidence_index=build_evidence_index(state))['aggregate_groups']



def test_09_json_parent_metric_is_preserved_as_its_own_observation(tmp_path):
    body=json.dumps({'disease':'Pertussis','country':'France','year':2025,'cases_confirmed':12,
                     'regions':[{'country':'Canada','cases_confirmed':3}]}).encode()
    state=source_state(tmp_path,body,'application/json')
    assert qualified_outputs(state,FACTS)[0]
    assert qualified_outputs(state,{**FACTS,'country':'Canada','cases_confirmed':3})[0]
    assert not qualified_outputs(state,{**FACTS,'country':'Canada','cases_confirmed':12})[0]



@pytest.mark.parametrize('bounds',[
    {'metric_period_start':'2025-01-01','metric_period_end':'2025-12-31'},
    {'reporting_period':'2025'},
])
def test_12_all_supported_annual_bound_names_require_full_interval(bounds):
    from data_collection_workflow.evidence_qualification import qualified_coverage
    row={'record_id':'month','disease':'Pertussis','country':'France','reporting_period':'2025',
         'metric_period_start':'2025-01-01','metric_period_end':'2025-01-31','cases_confirmed':12,
         'evidence_qualification':{'status':'qualified','product_kind':'aggregate'}}
    requirement={'requirement_id':'annual','disease':'Pertussis',**bounds}
    assert not qualified_coverage([requirement],[row])['coverage_complete']


@pytest.mark.parametrize('nested',[False,True])
def test_09_generic_json_metric_keeps_its_own_identity(tmp_path,nested):
    payload={'disease':'Pertussis','country':'France','year':2025,
             'metric_name':'cases_confirmed','metric_value':12,'metric_unit':'cases','metric_category':'case_count'}
    if nested:
        payload['regions']=[{'country':'Canada','cases_confirmed':3},{'country':'Germany','metric_value':5}]
    state=source_state(tmp_path,json.dumps(payload).encode(),'application/json')
    facts={**FACTS,'metric_name':'cases_confirmed','metric_value':12,'metric_unit':'cases','metric_category':'case_count'}
    rows,attempts=qualified_outputs(state,facts)
    assert rows,[(c['text'],q.reasons) for c,q in attempts]
    assert not qualified_outputs(state,{**facts,'cases_confirmed':13,'metric_value':13})[0]
    assert not qualified_outputs(state,{'disease':'Pertussis','country':'France','reporting_period':'2025',
                                      'deaths':12,'metric_name':'deaths','metric_value':12})[0]
    if nested:
        assert qualified_outputs(state,{**FACTS,'country':'Canada','cases_confirmed':3})[0]
        assert not qualified_outputs(state,{**facts,'country':'Germany','cases_confirmed':5,'metric_value':5})[0]
        child = next(c for c in state['evidence_chunks'] if 'Germany' in c['text'])
        assert not any(span.get('json_field') == 'metric_name' for span in child['bound_context_spans'])


@pytest.mark.parametrize('cutoff,complete',[('2025-03-01',False),('2025-12-31',True),('2026-01-02',True)])
def test_12_source_as_of_limits_observed_annual_coverage(tmp_path,cutoff,complete):
    from data_collection_workflow.evidence_qualification import qualified_coverage
    text=f'As of {cutoff}, 12 confirmed Pertussis cases were reported in France during 2025.'
    state=source_state(tmp_path,text.encode(),'text/plain')
    rows,attempts=qualified_outputs(state,{**FACTS,'as_of_date':cutoff})
    assert rows,[(c['text'],q.reasons) for c,q in attempts]
    assert qualified_coverage([{'requirement_id':'annual','country':'France','reporting_period':'2025'}],rows)['coverage_complete'] is complete


def test_12_source_id_alone_is_not_dispatch_permission():
    from data_collection_workflow.workflow_recovery import _recovery_fetch_skip_reason
    assert _recovery_fetch_skip_reason({'source_id':'source'}, {}) is not None
