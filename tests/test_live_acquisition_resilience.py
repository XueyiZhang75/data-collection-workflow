"""Local acquisition failures, source numbers and linked-resource contracts."""
import base64
import io
import json

import pytest

from data_collection_workflow import document_acquisition as acquisition
from data_collection_workflow.nodes import content_processing as content
from data_collection_workflow.source_coverage import _looks_like_error_page
from data_collection_workflow.session_runtime import RunContext


@pytest.fixture(autouse=True)
def evidence(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')


def parsed(tmp_path,body,kind='text/html',status=200):
    return acquisition.parse_response(body,url='https://authority.invalid/report',source_id='source',
        session_dir=tmp_path,content_type=kind,status_code=status)


@pytest.mark.parametrize('kind,body',[
    ('text/html',b'<title>Example fever report</title><h1>Example fever in Canada during 2025</h1><table><tr><th>Confirmed cases</th></tr><tr><td>404</td></tr></table>'),
    ('application/json',b'{"disease":"Example fever","country":"Canada","reporting_period":"2025","cases_confirmed":404}'),
    ('text/plain',b'404 confirmed cases of Example fever were reported in Canada during 2025.'),
    ('text/plain',b'404\nConfirmed cases of Example fever in Canada during 2025.'),
])
def test_real_observation_number_404_is_never_an_error_page(tmp_path,kind,body):
    doc=parsed(tmp_path,body,kind)
    assert doc['content_readable']
    assert not content._document_looks_like_error_page(doc)
    assert not _looks_like_error_page(doc)
    result=content.document_quality_check({'documents':[doc],'structured_task':{'disease':'Example fever'}})
    assert result['documents'][0]['quality_status']!='unusable'
    assert content.evidence_chunking_and_data_presence_flagging({**result,'structured_task':{'disease':'Example fever'}})['evidence_chunks']


def test_pdf_observation_number_404_remains_readable(tmp_path):
    canvas_module=pytest.importorskip('reportlab.pdfgen.canvas')
    stream=io.BytesIO(); canvas=canvas_module.Canvas(stream)
    canvas.drawString(30,740,'Example fever in Canada during 2025: 404 confirmed cases.')
    canvas.save()
    doc=parsed(tmp_path,stream.getvalue(),'application/pdf')
    assert doc['content_readable']
    assert not content._document_looks_like_error_page(doc)
    assert not _looks_like_error_page(doc)


@pytest.mark.parametrize('status,body',[
    (404,b'<h1>Not found</h1>'),
    (200,b'<title>404 - Not Found</title><h1>Page not found</h1>'),
    (200,b'<title>Publication server</title><h1>Access denied</h1>'),
    (200,b'<h1>Not found | authority</h1>'),
])
def test_actual_error_pages_are_consistently_rejected(tmp_path,status,body):
    doc=parsed(tmp_path,body,status=status)
    assert not doc['content_readable']
    assert content._document_looks_like_error_page(doc)
    assert _looks_like_error_page(doc)


CHALLENGE=b'<!doctype html><html><head><title>Checking your browser - reCAPTCHA</title></head><body>Checking your browser before accessing journal.invalid ... Click here if you are not automatically redirected after 5 seconds.</body></html>'


def test_browser_check_is_blocked_not_evidence(tmp_path):
    doc=parsed(tmp_path,CHALLENGE)
    assert doc['acquisition_status']=='blocked'
    assert not doc['content_readable']
    assert content._document_looks_like_error_page(doc) and _looks_like_error_page(doc)
    result=content.document_quality_check({'documents':[doc],'structured_task':{'disease':'Example fever'}})
    assert result['documents'][0]['quality_status']=='unusable'


def test_article_about_captcha_is_not_a_challenge(tmp_path):
    doc=parsed(tmp_path,b'<title>CAPTCHA accessibility study</title><h1>Review of CAPTCHA use in health reporting</h1><p>This article discusses browser checks and reCAPTCHA.</p>')
    assert doc['content_readable']
    assert not content._document_looks_like_error_page(doc)


def fake_transport(monkeypatch,body):
    import requests
    calls=[]
    class Response:
        status_code=200
        headers={'content-type':'text/html'}
        url='https://authority.invalid/report'
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,size):yield body
    def get(url,**kwargs):
        calls.append(url)
        return Response()
    monkeypatch.setattr(requests,'get',get)
    return calls


@pytest.mark.parametrize('rendered_body,status,expected',[(CHALLENGE,200,'blocked'),(CHALLENGE,403,'http_error'),
    (b'<h1>Access denied</h1>',200,'error_page'),
    (b'<h1>Example fever in Canada during 2025</h1><p>12 confirmed cases.</p>',200,'readable')])
def test_browser_check_gets_one_budgeted_render_without_challenge_bypass(tmp_path,monkeypatch,rendered_body,status,expected):
    readable=expected=='readable'
    calls=fake_transport(monkeypatch,CHALLENGE)
    renders=[]
    def browser(url,config):
        renders.append(url)
        return {'body':base64.b64encode(rendered_body).decode(),'content_type':'text/html','status_code':status,'final_url':url,
                'browser_responses':[{'url':url+'/verification','content_type':'application/json','status_code':200,'body':base64.b64encode(b'{"success":false}').decode()}] if not readable else []}
    monkeypatch.setattr(acquisition,'_browser',browser)
    runtime=RunContext(tmp_path/'run',{'universal':{'budget_limits':{'fetch':2,'fetch_ordinary':1,'browser':1}}})
    with runtime.activate():
        doc=acquisition.acquire_document('https://authority.invalid/report',source_id='source',session_dir=runtime.session_dir)
        cached=acquisition.acquire_document('https://authority.invalid/report',source_id='source',session_dir=runtime.session_dir)
    assert len(calls)==len(renders)==1
    assert doc['content_readable'] is readable and cached['content_readable'] is readable
    assert doc['response_content_hash'] and doc['response_artifact_path']
    assert runtime.ledger.snapshot()['used']['browser']==1
    if not readable:
        assert doc['acquisition_status']==expected
        assert 'success' not in doc['clean_text']


def test_blocked_browser_budget_keeps_original_response(tmp_path,monkeypatch):
    fake_transport(monkeypatch,CHALLENGE)
    monkeypatch.setattr(acquisition,'_browser',lambda *args:pytest.fail('budget must prevent browser dispatch'))
    runtime=RunContext(tmp_path/'run',{'universal':{'budget_limits':{'fetch':2,'browser':0}}})
    with runtime.activate():
        doc=acquisition.acquire_document('https://authority.invalid/report',source_id='source',session_dir=runtime.session_dir)
    assert doc['acquisition_status']=='budget_exhausted'
    assert doc['budget_exhausted_kind']=='browser'
    assert not doc['content_readable']
    assert (runtime.session_dir/doc['raw_artifact_path']).read_bytes()==CHALLENGE


def resource_node(tmp_path,monkeypatch,pages,*,total=20,limit=None,sources=None):
    import os
    import requests
    from test_evidence_resource_discovery import _source
    from data_collection_workflow.environment import WORKFLOW_ENV_NAMES
    for key in list(os.environ):
        if key in WORKFLOW_ENV_NAMES or key.startswith('HDC_'):
            monkeypatch.delenv(key,raising=False)
    for key,value in {'PIPELINE_MODE':'evidence','ENABLE_LIVE_FETCH':'true',
        'FETCH_SEARCH_DERIVED_SOURCES':'true','FETCH_MAX_SEARCH_DERIVED_SOURCES':str(total),
        'FETCH_MAX_TOTAL_SOURCES':str(total),'USE_FIXTURE_DOCUMENTS':'false',
        'ENABLE_LLM_SOURCE_IDENTITY':'false','LLM_SOURCE_IDENTITY_POST_FETCH':'false'}.items():
        monkeypatch.setenv(key,value)
    if limit is not None:
        monkeypatch.setenv('RESOURCE_LINK_EXPANSION_LIMIT',str(limit))
    visits=[]
    class Response:
        status_code=200
        def __init__(self,url):
            self.url=url
            kind,self.body=pages[url]
            self.headers={'content-type':kind}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,size):yield self.body
    def get(url,**kwargs):
        visits.append(url)
        return Response(url)
    monkeypatch.setattr(requests,'get',get)
    state={'structured_task':{'disease':'Example fever','location':'Canada','start_date':'2025-01-01','end_date':'2025-12-31'},
           'source_registry':sources or [_source('https://authority.invalid/index')],'collection_trace':[]}
    runtime=RunContext(tmp_path/'run',{'pipeline_mode':'evidence',
            'universal':{'budget_limits':{'fetch':total,'fetch_ordinary':total}}})
    with runtime.activate():
        result=content.content_fetch_and_parse(state)
    return result,visits,runtime.ledger.snapshot()


def fixture_pdf():
    canvas_module=pytest.importorskip('reportlab.pdfgen.canvas')
    stream=io.BytesIO(); canvas=canvas_module.Canvas(stream)
    canvas.drawString(30,740,'Example fever in Canada during 2025: 12 confirmed cases.')
    canvas.save()
    return stream.getvalue()


def test_resource_default_uses_existing_total_budget_without_hidden_eight_cap(tmp_path,monkeypatch):
    links=''.join(f'<a href="/files/report{i}.pdf">Download report {i}</a>' for i in range(10))
    pages={'https://authority.invalid/index':('text/html',('<h1>Example fever Canada 2025 reports</h1>'+links).encode())}
    pdf=fixture_pdf()
    pages.update({f'https://authority.invalid/files/report{i}.pdf':('application/pdf',pdf) for i in range(10)})
    result,visits,budget=resource_node(tmp_path,monkeypatch,pages,total=12)
    assert len(visits)==11
    assert sum(d['document_type']=='pdf' for d in result['documents'])==10
    assert budget['used']['fetch']==11


@pytest.mark.parametrize('limit',[0,1])
def test_explicit_resource_limit_still_bounds_relative_downloads(tmp_path,monkeypatch,limit):
    pages={'https://authority.invalid/index':('text/html',b'<h1>Example fever Canada reports</h1><a href="../data/counts.json">Download data</a>'),
           'https://authority.invalid/data/counts.json':('application/json',b'{"cases":12}')}
    result,visits,budget=resource_node(tmp_path,monkeypatch,pages,total=3,limit=limit)
    assert len(visits)==1+limit and budget['used']['fetch']==1+limit


def test_archive_then_landing_can_finish_at_bounded_pdf_download(tmp_path,monkeypatch):
    pages={
        'https://authority.invalid/index':('text/html',b'<h1>Example fever report series</h1><a href="/archive">Report archive</a>'),
        'https://authority.invalid/archive':('text/html',b'<h1>Example fever report archive</h1><a href="/report/2025">2025 report</a>'),
        'https://authority.invalid/report/2025':('text/html',b'<h1>Example fever Canada 2025 report</h1><a href="../../files/report.pdf">Download</a><a href="/archive/deeper">More report archive</a>'),
        'https://authority.invalid/files/report.pdf':('application/pdf',fixture_pdf()),
    }
    result,visits,_=resource_node(tmp_path,monkeypatch,pages)
    assert visits==list(pages)
    assert any(d['document_type']=='pdf' and d['content_readable'] for d in result['documents'])


def test_downloads_precede_new_archive_navigation_without_starving_other_seed_pages(tmp_path,monkeypatch):
    from test_evidence_resource_discovery import _source
    pages={
        'https://authority.invalid/index':('text/html',b'<h1>Example fever report series</h1><a href="/archive">Report archive</a>'),
        'https://authority.invalid/latest':('text/html',b'<h1>Example fever Canada 2025 report</h1><a href="/files/report.pdf">Download</a>'),
        'https://authority.invalid/files/report.pdf':('application/pdf',fixture_pdf()),
        'https://authority.invalid/archive':('text/html',b'<h1>Example fever historical report archive</h1>'),
    }
    sources=[_source('https://authority.invalid/index'),_source('https://authority.invalid/latest',2)]
    sources[0]['target_fit_status']='verified_target_collection'
    result,visits,_=resource_node(tmp_path,monkeypatch,pages,sources=sources,total=4)
    assert visits==list(pages)


def test_mime_download_link_uses_source_heading_ancestry_on_data_host(tmp_path):
    from data_collection_workflow.resource_discovery import task_resource_candidates
    body=b'<h1>Example fever Canada 2025 report</h1><h2>Documents</h2><a href="https://files.invalid/get/opaque" type="application/pdf" download>Download</a>'
    doc=parsed(tmp_path,body)
    candidates=task_resource_candidates(doc,{}, {'structured_task':{'disease':'Example fever'}})
    assert candidates and candidates[0]['resource_type']=='pdf'
    assert candidates[0]['url']=='https://files.invalid/get/opaque'


@pytest.mark.parametrize('banner,expected',[
    ('Page not found','error_page'),
    ('Verify you are human','blocked'),
    ('404 confirmed cases of Example fever in Canada during 2025', 'readable'),
])
def test_primary_page_banner_survives_long_site_navigation(tmp_path,banner,expected):
    navigation=''.join(f'<a href="/section/{i}">Public health department navigation item {i}</a>' for i in range(12))
    body=('<head><title>Public Health Department</title></head><body><nav>'+navigation+
          '</nav><main><h1>'+banner+'</h1><p>Public health information.</p></main></body>').encode()
    doc=parsed(tmp_path,body)
    assert doc['acquisition_status']==expected
    assert doc['content_readable']==(expected=='readable')
    assert content._document_looks_like_error_page(doc)==(expected!='readable')
    assert _looks_like_error_page(doc)==(expected!='readable')


def test_navigation_gets_one_slot_when_initial_seed_queue_fills_total_budget(tmp_path,monkeypatch):
    from test_evidence_resource_discovery import _source
    pages={
        'https://authority.invalid/index':('text/html',b'<h1>Example fever report series</h1><a href="/archive">Report archive</a>'),
        'https://authority.invalid/other':('text/html',b'<h1>Example fever Canada 2025 report</h1>'),
        'https://authority.invalid/archive':('text/html',b'<h1>Example fever Canada 2025 report archive</h1><p>12 confirmed cases.</p>'),
    }
    sources=[_source('https://authority.invalid/index'),_source('https://authority.invalid/other',2)]
    sources[0]['target_fit_status']='verified_target_collection'
    result,visits,budget=resource_node(tmp_path,monkeypatch,pages,sources=sources,total=2)
    assert visits==['https://authority.invalid/index','https://authority.invalid/archive']
    assert budget['used']['fetch']==2


def test_full_queue_promotes_one_navigation_after_downloads(tmp_path,monkeypatch):
    from test_evidence_resource_discovery import _source
    pages={
        'https://authority.invalid/index':('text/html',b'<h1>Example fever report series</h1><a href="/archive-a">Report archive A</a><a href="/archive-b">Report archive B</a><a href="/report.pdf">Download report</a>'),
        'https://authority.invalid/report.pdf':('application/pdf',fixture_pdf()),
        'https://authority.invalid/archive-a':('text/html',b'<h1>Example fever historical report archive</h1>'),
        'https://authority.invalid/archive-b':('text/html',b'<h1>Example fever historical report archive</h1>'),
    }
    sources=[_source('https://authority.invalid/index')]
    sources[0]['target_fit_status']='verified_target_collection'
    for i in range(2,5):
        url=f'https://authority.invalid/seed-{i}'
        sources.append(_source(url,i))
        pages[url]=('text/html',b'<h1>Example fever report</h1>')
    result,visits,budget=resource_node(tmp_path,monkeypatch,pages,sources=sources,total=4)
    assert visits[:3]==['https://authority.invalid/index','https://authority.invalid/report.pdf','https://authority.invalid/archive-a']
    assert visits[3].startswith('https://authority.invalid/seed-')
    assert budget['used']['fetch']==4


@pytest.mark.parametrize('body',[
    b"<title>We couldn't find that Web page (Error 404) - Public Health</title><main><h1>We couldn't find that Web page (Error 404)</h1><p>Please return to the home page.</p></main>",
    b"<title>Technical Difficulties</title><p>We are sorry, this site is currently experiencing technical difficulties.</p><p>Please try again in a few moments.</p><p>Exception: forbidden</p>",
])
def test_explicit_missing_page_heading_with_success_status_is_an_error(tmp_path,body):
    doc=acquisition.parse_response(body,url='https://authority.invalid/mpox',source_id='missing-page',
        session_dir=tmp_path,content_type='text/html',status_code=200,final_url='https://authority.invalid/404.html')
    assert doc['acquisition_status']=='error_page'
    assert not doc['content_readable']
    assert content._document_looks_like_error_page(doc) and _looks_like_error_page(doc)


@pytest.mark.parametrize('payload',[
    {'InterceptDefinition':{'InterceptName':'Feedback Button','ActionSets':{'AS_1':{'CreativeType':'FeedbackButton','Target':{'Type':'Survey'}}}}},
    {'CreativeDefinition':{'Title':'Feedback control','Options':{'Desktop':{'LookAndFeel':{'ButtonText':'Feedback','ButtonColor':'#ffffff'}}}}},
    {'Intercepts':[{'InterceptID':'SI_1','Decision':{'Target':{'Type':'Survey','URL':'https://feedback.invalid/form'},'Creative':None},'SurveyID':'SV_1'}]},
])
def test_browser_interface_configuration_is_retained_for_audit_but_not_evidence(tmp_path,monkeypatch,payload):
    fake_transport(monkeypatch,CHALLENGE)
    config_body=json.dumps(payload).encode()
    data_body=b'{"records":[{"disease":"Example fever","country":"Canada","reporting_period":"2025","cases_confirmed":404}]}'
    csv_body=b'country,year,cases_confirmed\nCanada,2025,12'
    resources=[('https://feedback.invalid/settings','application/json',config_body),
               ('https://data.invalid/stats','application/json',data_body),
               ('https://data.invalid/counts.csv','text/csv',csv_body)]
    def browser(url,config):
        return {'body':base64.b64encode(b'<h1>Example fever report</h1><p>Canada in 2025.</p>').decode(),
                'content_type':'text/html','status_code':200,'final_url':url,
                'browser_responses':[{'url':u,'content_type':t,'status_code':200,'body':base64.b64encode(b).decode()} for u,t,b in resources]}
    monkeypatch.setattr(acquisition,'_browser',browser)
    runtime=RunContext(tmp_path/'run',{'universal':{'budget_limits':{'fetch':2,'fetch_ordinary':1,'browser':1}}})
    with runtime.activate():
        doc=acquisition.acquire_document('https://authority.invalid/report',source_id='source',session_dir=runtime.session_dir)
    assert len(doc['browser_responses'])==3
    assert (runtime.session_dir/doc['browser_responses'][0]['raw_artifact_path']).read_bytes()==config_body
    assert next(iter(payload)) not in doc['clean_text']
    assert 'cases_confirmed' in doc['clean_text'] and '404' in doc['clean_text']
    assert 'Canada | 2025 | 12' in doc['clean_text']
    response_urls={x.get('response_url') for x in doc['locator_spans']}
    assert 'https://feedback.invalid/settings' not in response_urls
    assert {'https://data.invalid/stats','https://data.invalid/counts.csv'}<=response_urls


@pytest.mark.parametrize('years,countries',[
    (('2027','2028'),('France','Germany')),
    (('2031','2029'),('Kenya','Peru')),
])
def test_multilevel_table_headers_preserve_grid_and_qualified_cell_scope(tmp_path,years,countries):
    from test_evidence_positive_extraction import source_state,qualified_outputs
    html=(f'<h1>Pertussis surveillance</h1><table><caption>Pertussis reported observations</caption>'
          '<tr><th rowspan="2">Country</th><th colspan="2">Confirmed cases</th><th rowspan="2">Deaths</th></tr>'
          f'<tr><th>{years[0]}</th><th>{years[1]}</th></tr>'
          f'<tr><td>{countries[0]}</td><td>12</td><td>7</td><td>1</td></tr>'
          f'<tr><td>{countries[1]}</td><td>3</td><td>5</td><td>0</td></tr></table>')
    state=source_state(tmp_path,html.encode(),'text/html')
    doc=state['documents'][0];table=doc['tables'][0]
    assert table.get('header_row_ids')==[0,1]
    assert table.get('row_ids')==[2,3]
    assert len(table['rows'])==2
    assert table.get('cell_spans')
    for cell in table['cell_spans']:
        assert doc['clean_text'][cell['char_start']:cell['char_end']]==cell['quote']
    rows=[c for c in state['evidence_chunks'] if c.get('table_id')=='table_1' and c.get('row_id')]
    assert {c['row_id'] for c in rows}=={'2','3'}
    for chunk in rows:
        assert doc['clean_text'][chunk['char_start']:chunk['char_end']]==chunk['text']
        headers=[x['quote'] for x in chunk['bound_context_spans'] if x['role']=='table_header']
        assert 'Country | Confirmed cases | Deaths' in headers
        assert ' | '.join(years) in headers
        assert chunk['source_column_labels'][1]=='Confirmed cases / '+years[0]
        assert chunk['source_column_labels'][2]=='Confirmed cases / '+years[1]
    facts={'disease':'Pertussis','country':countries[0],'reporting_period':years[0],
           'cases_confirmed':12,'metric_name':'confirmed_cases','metric_value':12,'metric_unit':'count','metric_category':'case_count',
           'evidence_quote':countries[0]+' | 12 | 7 | 1',
           'field_provenance_json':{'cases_confirmed':{'quote':'12'}}}
    outputs,attempts=qualified_outputs(state,facts)
    assert outputs,[(c['text'],q.reasons) for c,q in attempts]
    from data_collection_workflow.evidence_products import build_evidence_products
    from data_collection_workflow.evidence_qualification import build_evidence_index
    products=build_evidence_products(outputs,evidence_index=build_evidence_index(state))
    assert products['aggregate_groups'] and not products['excluded_observations']
    assert qualified_outputs(state,{**facts,'field_provenance_json':{'cases_confirmed':{'quote':'7'}}})[0]==[]
    assert qualified_outputs(state,{**facts,'reporting_period':years[1]})[0]==[]
    assert qualified_outputs(state,{**facts,'country':countries[1]})[0]==[]
    assert qualified_outputs(state,{**facts,'cases_confirmed':3})[0]==[]


def test_rowspan_country_binds_only_its_explicit_row_range(tmp_path):
    from test_evidence_positive_extraction import source_state,qualified_outputs
    html=b'<h1>Pertussis reported observations</h1><table><tr><th>Country</th><th>Reporting period</th><th>Confirmed cases</th></tr><tr><td rowspan="2">France</td><td>2027</td><td>12</td></tr><tr><td>2028</td><td>7</td></tr><tr><td>Germany</td><td>2028</td><td>3</td></tr></table>'
    state=source_state(tmp_path,html,'text/html')
    row=next(c for c in state['evidence_chunks'] if c.get('row_id')=='2')
    assert row['text']=='2028 | 7'
    assert any(s['quote']=='France' for s in row['bound_context_spans'])
    facts={'disease':'Pertussis','country':'France','reporting_period':'2028','cases_confirmed':7,
           'evidence_quote':'2028 | 7','field_provenance_json':{'cases_confirmed':{'quote':'7'},'country':{'quote':'France'}}}
    outputs,attempts=qualified_outputs(state,facts)
    assert outputs,[(c['text'],q.reasons) for c,q in attempts]
    assert qualified_outputs(state,{**facts,'country':'Germany'})[0]==[]
    assert qualified_outputs(state,{**facts,'cases_confirmed':3})[0]==[]


def test_multilevel_country_columns_bind_only_the_selected_country(tmp_path):
    from test_evidence_positive_extraction import source_state,qualified_outputs
    html=b'<h1>Pertussis surveillance</h1><table><tr><th rowspan="2">Year</th><th colspan="2">France</th><th colspan="2">Germany</th></tr><tr><th>Confirmed cases</th><th>Deaths</th><th>Confirmed cases</th><th>Deaths</th></tr><tr><td>2027</td><td>12</td><td>1</td><td>7</td><td>0</td></tr></table>'
    state=source_state(tmp_path,html,'text/html')
    facts={'disease':'Pertussis','country':'France','reporting_period':'2027','cases_confirmed':12}
    assert qualified_outputs(state,facts)[0]
    assert qualified_outputs(state,{**facts,'country':'Germany'})[0]==[]
    assert qualified_outputs(state,{**facts,'cases_confirmed':7})[0]==[]


def test_multilevel_table_disease_row_cannot_borrow_page_disease(tmp_path):
    from test_evidence_positive_extraction import source_state,qualified_outputs
    html=b'<h1>Pertussis and Measles surveillance in France</h1><table><tr><th rowspan="2">Disease</th><th colspan="2">Confirmed cases</th></tr><tr><th>2027</th><th>2028</th></tr><tr><td>Measles</td><td>12</td><td>7</td></tr></table>'
    state=source_state(tmp_path,html,'text/html')
    from data_collection_workflow.evidence_qualification import assess_record_evidence,build_evidence_index
    row=next(c for c in state['evidence_chunks'] if c.get('row_id')=='2')
    facts={'record_id':'wrong-disease','source_id':'source','supporting_chunk_id':row['chunk_id'],
           'disease':'Pertussis','country':'France','reporting_period':'2027','cases_confirmed':12}
    assert assess_record_evidence(facts,contract={},evidence_index=build_evidence_index(state)).status=='candidate'


@pytest.mark.parametrize('numeral,field,value,expected',[
    ('2,519','cases_confirmed',2519,True),
    ('2,519','cases_confirmed',251,False),
    ('2,519','cases_confirmed',519,False),
    ('2,519','deaths',2519,False),
    ('2,519%','cases_confirmed',2519,False),
    ('25,19','cases_confirmed',2519,False),
])
def test_exact_grouped_prose_counts_preserve_value_and_metric(tmp_path,numeral,field,value,expected):
    from test_evidence_positive_extraction import source_state,qualified_outputs
    body=f'<h1>Pertussis in France during 2025</h1><p>{numeral} confirmed cases were reported.</p>'
    state=source_state(tmp_path,body.encode(),'text/html')
    facts={'disease':'Pertussis','country':'France','reporting_period':'2025',field:value}
    outputs,attempts=qualified_outputs(state,facts)
    assert bool(outputs)==expected,[(c['text'],q.reasons) for c,q in attempts]


@pytest.mark.parametrize('sentence,year,expected',[
    ('In 2025, 12 cases were reported.','2025',True),
    ('In 2025, 12 cases were reported.','2024',False),
    ('12 cases were reported during 2025.','2024',False),
    ('12 cases were reported.','2024',False),
    ('12 cases were reported.','2025',False),
    ('12 cases were reported.','2024-2025',True),
])
def test_explicit_count_year_overrides_broader_heading_period(tmp_path,sentence,year,expected):
    from test_evidence_positive_extraction import source_state,qualified_outputs
    body=f'<h1>Pertussis in France during 2024-2025</h1><p>{sentence}</p>'
    state=source_state(tmp_path,body.encode(),'text/html')
    facts={'disease':'Pertussis','country':'France','reporting_period':year,'cases_unspecified':12,
           'evidence_quote':sentence}
    outputs,attempts=qualified_outputs(state,facts)
    assert bool(outputs)==expected,[(c['text'],q.reasons) for c,q in attempts]


@pytest.mark.parametrize('decision',[True,1,'accepted',[],None])
def test_arbitrary_json_intercept_decisions_are_not_interface_configuration(decision):
    payload={'Intercepts':[{'InterceptID':'x','Decision':decision}]}
    assert not acquisition._browser_interface_configuration({'document_type':'json','clean_text':json.dumps(payload)})


@pytest.mark.parametrize('date_field',['as_of_date','date_reported'])
def test_observation_year_does_not_overwrite_separate_report_date(tmp_path,date_field):
    from test_evidence_positive_extraction import source_state,qualified_outputs
    sentence=('In 2025, 12 cases were reported, as of 2026-01-02.' if date_field == 'as_of_date'
              else 'In 2025, 12 cases were reported (report date: 2026-01-02).')
    state=source_state(tmp_path,('<h1>Pertussis in France</h1><p>'+sentence+'</p>').encode(),'text/html')
    facts={'disease':'Pertussis','country':'France','reporting_period':'2025','cases_unspecified':12,
           date_field:'2026-01-02','evidence_quote':sentence}
    outputs,attempts=qualified_outputs(state,facts)
    assert outputs,[(c['text'],q.reasons) for c,q in attempts]
    assert qualified_outputs(state,{**facts,'reporting_period':'2024'})[0]==[]


@pytest.mark.parametrize('heading_country,body_country', [('France','Brazil'),('Canada','Germany'),('France','Nepal'),('Germany','Japan')])
def test_explicit_count_country_overrides_broader_heading(tmp_path,heading_country,body_country):
    from test_evidence_positive_extraction import source_state,qualified_outputs
    sentence=f'In {body_country}, 12 confirmed pertussis cases were reported during 2025.'
    body=f'<h1>Pertussis surveillance in {heading_country} during 2025</h1><p>{sentence}</p>'
    state=source_state(tmp_path,body.encode(),'text/html')
    facts={'disease':'Pertussis','country':body_country,'reporting_period':'2025','cases_confirmed':12,'evidence_quote':sentence}
    assert qualified_outputs(state,facts)[0]
    assert qualified_outputs(state,{**facts,'country':heading_country})[0]==[]


def test_single_source_heading_country_remains_bound_when_body_omits_it(tmp_path):
    from test_evidence_positive_extraction import source_state,qualified_outputs
    state=source_state(tmp_path,b'<h1>Pertussis in Canada during 2025</h1><p>12 confirmed cases were reported.</p>','text/html')
    assert qualified_outputs(state,{'disease':'Pertussis','country':'Canada','reporting_period':'2025','cases_confirmed':12})[0]


@pytest.mark.parametrize('sentence',[
    'In France and Brazil, 12 confirmed pertussis cases were reported during 2025.',
    'France and Brazil reported 12 confirmed pertussis cases during 2025.',
])
def test_pooled_country_count_cannot_be_assigned_to_one_country(tmp_path,sentence):
    from test_evidence_positive_extraction import source_state,qualified_outputs
    state=source_state(tmp_path,('<h1>Pertussis in France during 2025</h1><p>'+sentence+'</p>').encode(),'text/html')
    assert qualified_outputs(state,{'disease':'Pertussis','country':'France','reporting_period':'2025','cases_confirmed':12,'evidence_quote':sentence})[0]==[]


def test_origin_country_does_not_override_explicit_event_country(tmp_path):
    from test_evidence_positive_extraction import source_state,qualified_outputs
    sentence='In Canada, 12 confirmed pertussis cases from Japan were reported during 2025.'
    state=source_state(tmp_path,('<h1>Pertussis surveillance</h1><p>'+sentence+'</p>').encode(),'text/html')
    assert qualified_outputs(state,{'disease':'Pertussis','country':'Canada','reporting_period':'2025','cases_confirmed':12,'evidence_quote':sentence})[0]



def test_explicit_state_label_does_not_become_a_second_country(tmp_path):
    from test_evidence_positive_extraction import source_state,qualified_outputs
    sentence='In the US state of Georgia, 12 confirmed pertussis cases were reported during 2025.'
    state=source_state(tmp_path,('<h1>Pertussis in United States during 2025</h1><p>'+sentence+'</p>').encode(),'text/html')
    assert qualified_outputs(state,{'disease':'Pertussis','country':'United States','reporting_period':'2025','cases_confirmed':12,'evidence_quote':sentence})[0]
