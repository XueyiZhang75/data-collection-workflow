"""Generic transport/content regressions from the acquisition audit (offline)."""
import io
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pytest
from PIL import Image
from data_collection_workflow import document_acquisition as acquisition
from data_collection_workflow.resource_discovery import task_resource_candidates

@pytest.mark.parametrize('kind', ['application/octet-stream', 'text/plain'])
def test_binary_is_never_readable_text(tmp_path, kind):
    doc = acquisition.parse_response(bytes(range(256)) * 3, url='https://data.example/file',
        source_id='binary', session_dir=tmp_path, content_type=kind)
    assert not doc['content_readable']
    assert doc['acquisition_status'] == 'unsupported_format'
    assert not doc['clean_text']

@pytest.mark.parametrize('fmt,kind', [('PNG','image/png'),('JPEG','application/octet-stream')])
def test_image_magic_routes_to_ocr_with_locations(tmp_path, monkeypatch, fmt, kind):
    data=io.BytesIO(); Image.new('RGB',(100,50),'white').save(data,format=fmt)
    calls=[]
    def recognize(image, config):
        calls.append(image.size)
        return {'text':'Dengue 12 cases', 'words':[{'text':'12','confidence':97,
            'bbox':[10,10,20,10], 'line':[1,1,1]}]}
    monkeypatch.setattr(acquisition,'_ocr',recognize)
    doc=acquisition.parse_response(data.getvalue(),url='https://data.example/figure',
        source_id='image',session_dir=tmp_path,content_type=kind)
    assert calls == [(100,50)]
    assert doc['document_type']=='image' and doc['content_readable']
    assert doc['ocr_words'][0]['page']==1
    assert doc['locator_spans'][0]['extraction_method']=='ocr'
    assert (tmp_path/doc['raw_artifact_path']).read_bytes()==data.getvalue()

@pytest.mark.parametrize('status,body', [(200,b''),(302,b'Loading target'),(500,b'12 cases')])
def test_incomplete_or_error_response_is_not_reparse_input(tmp_path,status,body):
    doc=acquisition.parse_response(body,url='https://data.example/target',source_id='s',
        session_dir=tmp_path,status_code=status)
    assert doc.get('parse_eligible') is False
    assert not doc['content_readable']

def _resource_document(disease,location):
    return {'content_readable':True, 'url':'https://journal.example/article',
        'clean_text':f'{disease} outbreak in {location}. Data availability.',
        'metadata':{'outbound_links':[
            {'href':'https://journal.example/peer-review.pdf','text':'Transparent Peer Review file',
             'context':'Supplementary information','heading':'Supplementary information'},
            {'href':'https://data.example/counts','text':'Underlying data',
             'context':f'{disease} case counts in {location} are available here.',
             'heading':'Data availability'},
            {'href':'https://journal.example/unrelated.pdf','text':'SARS-CoV-2 capacity building',
             'context':'Recommended reading','heading':'Related articles'},
        ]}}

@pytest.mark.parametrize('disease,location',[('mpox','Sierra Leone'),('measles','Canada'),('dengue','Peru')])
def test_primary_data_link_beats_peer_review_and_recommendations(disease,location):
    doc=_resource_document(disease,location)
    rows=task_resource_candidates(doc,{}, {'structured_task':{'disease':disease,'location':location,
        'start_date':'2025-01-01','end_date':'2025-12-31'}})
    assert rows and rows[0]['url']=='https://data.example/counts'
    assert not any('unrelated.pdf' in row['url'] for row in rows)
    assert not any('peer-review.pdf' in row['url'] for row in rows)

def test_cross_host_data_availability_link_inherits_scoped_relationship_only():
    doc=_resource_document('dengue','Peru')
    doc['metadata']['outbound_links']=[{'href':'https://repository.example/files/1',
        'text':'Download data','context':'The data underlying this study are available here.',
        'heading':'Data availability'}]
    rows=task_resource_candidates(doc,{}, {'structured_task':{'disease':'dengue','location':'Peru'}})
    assert len(rows)==1
    assert rows[0]['provenance']['context']=='The data underlying this study are available here.'


def test_adaptive_queue_does_not_truncate_at_legacy_soft_selection_limit(tmp_path,monkeypatch):
    import requests
    from test_evidence_resource_discovery import _env, _source
    from data_collection_workflow.session_runtime import RunContext
    from data_collection_workflow.nodes.content_processing import content_fetch_and_parse
    _env(monkeypatch,limit=8)
    monkeypatch.setenv('FETCH_MAX_TOTAL_SOURCES','2')
    monkeypatch.setenv('FETCH_MAX_SEARCH_DERIVED_SOURCES','2')
    visited=[]
    class Response:
        status_code=200
        headers={'content-type':'text/plain'}
        def __init__(self,url):self.url=url
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,size):yield b'Example fever Canada 2025: 12 confirmed cases.'
    def get(url,**kwargs):visited.append(url);return Response(url)
    monkeypatch.setattr(requests,'get',get)
    runtime=RunContext(tmp_path/'session',{'pipeline_mode':'evidence',
        'universal':{'budget_policy':{'version':2,'mode':'adaptive','soft_source_target':2},
                     'budget_limits':{'source_targets':6,'http_requests':12}}})
    state={'structured_task':{'disease':'Example fever','location':'Canada',
           'start_date':'2025-01-01','end_date':'2025-12-31'},
           'source_registry':[_source(f'https://source{i}.example/report',i) for i in range(6)],
           'collection_trace':[]}
    with runtime.activate():result=content_fetch_and_parse(state)
    assert len(visited)==6
    assert len([d for d in result['documents'] if d.get('content_readable')])==6
    assert runtime.ledger.snapshot()['used']['source_targets']==6


def _adaptive_runtime(tmp_path, targets=200, requests=1000):
    from data_collection_workflow.session_runtime import RunContext
    return RunContext(tmp_path, {'pipeline_mode':'evidence',
        'universal':{'budget_policy':{'version':2,'mode':'adaptive','soft_source_target':50},
        'budget_limits':{'source_targets':targets,'http_requests':requests}}})


def test_fifty_redirected_targets_use_fifty_targets_not_fifty_requests(tmp_path, monkeypatch):
    import requests
    from urllib.parse import urlsplit
    class Response:
        def __init__(self,url):
            self.url=url
            step=int(url.rsplit('/',1)[-1])
            self.status_code=302 if step<3 else 200
            self.headers={'content-type':'text/plain'}
            if step<3:self.headers['location']=str(step+1)
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,size):yield b'Dengue Canada 2025: 12 confirmed cases.' if self.status_code==200 else b''
    monkeypatch.setattr(requests,'get',lambda url,**kwargs:Response(url))
    runtime=_adaptive_runtime(tmp_path)
    with runtime.activate():
        docs=[acquisition.acquire_document(f'https://data.example/target-{i}/0',source_id=str(i),
            session_dir=tmp_path) for i in range(50)]
    assert all(doc['content_readable'] for doc in docs)
    used=runtime.ledger.snapshot()['used']
    assert used['source_targets']==50
    assert used['http_requests']==200


def test_redirect_chain_stops_after_five_hops(tmp_path,monkeypatch):
    import requests
    visited=[]
    class Response:
        status_code=302
        def __init__(self,url):self.url=url;self.headers={'location':str(int(url.rsplit('/',1)[-1])+1)}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,size):yield b''
    def get(url,**kwargs):visited.append(url);return Response(url)
    monkeypatch.setattr(requests,'get',get)
    runtime=_adaptive_runtime(tmp_path)
    with runtime.activate():
        doc=acquisition.acquire_document('https://data.example/0',source_id='s',session_dir=tmp_path)
    assert len(visited)==6
    assert not doc['content_readable'] and doc['parse_eligible'] is False
    assert 'redirect limit' in doc['fetch_error']


def test_default_download_supports_file_above_one_megabyte(tmp_path,monkeypatch):
    import requests
    body=b'Dengue data Canada 2025. '+b'x'*1_100_000
    class Response:
        status_code=200;headers={'content-type':'text/plain'};url='https://data.example/large'
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,size):
            for start in range(0,len(body),size):yield body[start:start+size]
    monkeypatch.setattr(requests,'get',lambda *args,**kwargs:Response())
    runtime=_adaptive_runtime(tmp_path)
    with runtime.activate():doc=acquisition.acquire_document(Response.url,source_id='s',session_dir=tmp_path)
    assert doc['content_readable'] and doc['raw_content_complete'] is True
    assert (tmp_path/doc['raw_artifact_path']).read_bytes()==body


def test_explicit_size_limit_retains_partial_bytes_as_ineligible(tmp_path,monkeypatch):
    import requests
    class Response:
        status_code=200;headers={'content-type':'text/plain'};url='https://data.example/large'
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,size):yield b'first chunk';yield b'oversize second chunk'
    monkeypatch.setattr(requests,'get',lambda *args,**kwargs:Response())
    runtime=_adaptive_runtime(tmp_path)
    with runtime.activate():doc=acquisition.acquire_document(Response.url,source_id='s',session_dir=tmp_path,config={'max_bytes':12})
    assert doc['parse_eligible'] is False and doc['raw_content_complete'] is False
    assert doc['raw_artifact_path'] and not doc['content_readable']
    assert (tmp_path/doc['raw_artifact_path']).read_bytes()==b'first chunk'


def test_transient_request_failure_retries_once_and_counts_both(tmp_path,monkeypatch):
    import requests
    calls=[]
    class Response:
        status_code=200;headers={'content-type':'text/plain'};url='https://data.example/target'
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,size):yield b'Dengue Canada 2025: 12 confirmed cases.'
    def get(url,**kwargs):
        calls.append(url)
        if len(calls)==1:raise requests.Timeout('temporary')
        return Response()
    monkeypatch.setattr(requests,'get',get)
    runtime=_adaptive_runtime(tmp_path)
    with runtime.activate():doc=acquisition.acquire_document(Response.url,source_id='s',session_dir=tmp_path)
    assert doc['content_readable'] and len(calls)==2
    assert runtime.ledger.snapshot()['used']['http_requests']==2
    assert runtime.ledger.snapshot()['used']['source_targets']==1


@pytest.mark.parametrize('credibility',[None,0.1])
def test_adaptive_acquires_unknown_publisher_without_granting_trust(tmp_path,monkeypatch,credibility):
    from test_evidence_resource_discovery import _env,_source
    from data_collection_workflow.nodes.content_processing import _build_fetch_requests,_fetch_config_from_env
    from data_collection_workflow.models import ContentFetchPolicy
    from data_collection_workflow.config import load_content_fetch_policy
    _env(monkeypatch)
    entry=_source('https://unknown.example/data')
    entry.update(source_type='unknown',source_identity_unverified=True,credibility_level='low',
        credibility_score=credibility, recommended_fetch_use='verify_identity_before_use')
    task={'disease':'Example fever','location':'Canada','start_date':'2025-01-01','end_date':'2025-12-31'}
    policy=ContentFetchPolicy(**load_content_fetch_policy())
    runtime=_adaptive_runtime(tmp_path)
    with runtime.activate():
        requests,skipped,manifest=_build_fetch_requests({'structured_task':task,'source_registry':[entry]},
            policy,True,_fetch_config_from_env(policy))
    assert [request.source_id for request in requests]==[entry['source_id']],skipped
    assert entry['source_identity_unverified'] is True


def test_enabled_alternative_fetch_shares_target_and_http_ledger(tmp_path,monkeypatch):
    import requests
    from test_evidence_resource_discovery import _env,_source
    from data_collection_workflow.nodes import content_processing as cp
    from data_collection_workflow.models import ContentFetchRequest
    _env(monkeypatch)
    url='https://data.example/blocked'
    class Response:
        status_code=403;headers={'content-type':'text/html'}
        def __init__(self):self.url=url
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,size):yield b'<h1>Access denied</h1>'
    monkeypatch.setattr(requests,'get',lambda *args,**kwargs:Response())
    def blocked(*args):raise RuntimeError('browser remained blocked')
    monkeypatch.setattr(acquisition,'_browser',blocked)
    calls=[]
    def alternate(*args):
        calls.append(True)
        return {'success':True,'body':b'Example fever Canada during 2025: 12 cases.',
            'content_type':'text/plain','http_status_code':200,'provider':'tavily_extract',
            'metadata':{'source_url':url}}
    monkeypatch.setattr(cp,'_tavily_extract_fetch',alternate)
    runtime=_adaptive_runtime(tmp_path,targets=2,requests=5)
    entry=_source(url)
    request=ContentFetchRequest(source_id=entry['source_id'],url=url,canonical_url=url,
        final_screening_decision='include_for_content_fetch',fetch_purpose='data_extraction')
    with runtime.activate():
        doc=cp._fetch_live_document_with_providers(request,entry,None,{
            'external_fetch_enabled':True,'external_fetch_provider_order':['native_requests','tavily_extract']})
    assert calls==[True]
    assert doc.content_readable and doc.fetch_provider=='tavily_extract'
    assert runtime.ledger.snapshot()['used']['source_targets']==1
    assert runtime.ledger.snapshot()['used']['http_requests']==2
    assert doc.response_artifact_path and doc.raw_artifact_path
