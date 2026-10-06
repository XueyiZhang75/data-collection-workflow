"""Real completion timestamps, and no invented times on offline/cache conversion."""
from datetime import datetime as RealDatetime, timezone
import json
import pytest
from data_collection_workflow import search_providers as providers, source_coverage as coverage
import importlib
discovery=importlib.import_module("data_collection_workflow.nodes.source_discovery")
from data_collection_workflow.models import SearchResult, SearchProviderResponse, SeedSource
from data_collection_workflow.session_runtime import RunContext

QUERY={'query_id':'q','query':'dengue Peru 2025','source_type':'public_health_report'}
NOW='2026-09-24T10:11:12.345678Z'
OLD='2024-02-03T04:05:06Z'

@pytest.fixture(autouse=True)
def no_external_tracing(monkeypatch):
    monkeypatch.setenv('LANGCHAIN_TRACING_V2','false')
    monkeypatch.setenv('LANGSMITH_TRACING','false')


def fake_live(monkeypatch):
    events=[]
    class Clock:
        @classmethod
        def now(cls,tz):
            assert tz is timezone.utc
            events.append('clock')
            return RealDatetime(2026,9,24,10,11,12,345678,tzinfo=timezone.utc)
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):events.append('response_closed')
        def read(self):
            events.append('read_complete')
            return json.dumps({'results':[{'title':'A','url':'https://example.test/a','content':'12 cases','retrieved_at':OLD},{'title':'B','url':'https://example.test/b','content':'24 cases'}]}).encode()
    def urlopen(*args,**kwargs):events.append('request');return Response()
    monkeypatch.setattr(providers,'datetime',Clock,raising=False)
    monkeypatch.setattr(providers.urllib.request,'urlopen',urlopen)
    monkeypatch.setenv('TAVILY_API_KEY','synthetic-test-key')
    return events


@pytest.mark.parametrize('revision',['evidence','legacy_revision'])
def test_live_timestamp_once_after_response_complete_not_provider_date(monkeypatch,revision):
    monkeypatch.setenv('PIPELINE_MODE',revision)
    events=fake_live(monkeypatch)
    response=providers.TavilySearchProvider().search(QUERY,max_results=2,timeout_seconds=1)
    assert [r.retrieved_at for r in response.results]==[NOW,NOW]
    assert events.count('clock')==1
    assert events.index('clock')>events.index('read_complete')
    assert [r.published_date for r in response.results]==[None,None]


@pytest.mark.parametrize('saved',[OLD,'2026-05-25T00:00:00Z',None])
def test_shared_normalizer_preserves_saved_or_unknown_time(saved):
    result=providers._result_from_dict({'url':'https://example.test/a','retrieved_at':saved},provider='tavily',planned_query=QUERY,rank=1)
    assert result.retrieved_at==saved


@pytest.mark.parametrize('saved',[OLD,None])
def test_fixture_missing_time_remains_unknown_and_known_time_deterministic(tmp_path,saved):
    p=tmp_path/'fixture.json';p.write_text(json.dumps({'results':[{'url':'https://example.test/a','retrieved_at':saved}]}))
    fixture=providers.FixtureSearchProvider(p)
    first=fixture.search(QUERY,max_results=1,timeout_seconds=1)
    second=fixture.search(QUERY,max_results=1,timeout_seconds=1)
    assert first.model_dump()==second.model_dump()
    assert first.results[0].retrieved_at==saved


@pytest.mark.parametrize('saved',[OLD,'2026-05-25T00:00:00Z',None])
@pytest.mark.parametrize('form',['dict','model'])
def test_cached_result_decode_and_candidate_never_invents_or_replaces_time(saved,form):
    raw={'provider':'tavily','results':[{'url':'https://example.test/a','retrieved_at':saved}]}
    if form=='model':raw=SearchProviderResponse.model_validate(raw)
    response=discovery._response_from_provider_output(raw,provider='tavily',planned_query=QUERY)
    assert response.results[0].retrieved_at==saved
    candidate=discovery._candidate_from_search_result(response.results[0],QUERY,settings=discovery.SourceSearchSettings(mode='live',provider='tavily'),canonical_url='https://example.test/a')
    assert candidate.retrieved_at==saved
    assert discovery._safe_search_row(response.results[0],candidate)['retrieved_at']==saved


def test_offline_seed_catalog_has_no_claimed_http_retrieval():
    seed=SeedSource(seed_source_id='seed_a',title='A',url='https://example.test/a',publisher='Agency',source_type='report',priority=1,source_purpose='discovery',expected_fields=[],match_terms=[])
    assert discovery._make_source_candidate(seed,QUERY).retrieved_at is None


@pytest.mark.parametrize('saved',[OLD,None])
def test_official_candidate_conversion_preserves_known_or_unknown(saved):
    result=discovery._candidate_from_official_coverage({'url':'https://example.test/a','retrieved_at':saved})
    assert result.retrieved_at==saved


def test_generated_coverage_urls_are_not_retrieved_responses(monkeypatch):
    monkeypatch.setattr(coverage,'build_source_coverage_requirements',lambda state:[{'official_candidate_urls':['https://example.test/a'],'requirement_id':'r','agency':'Agency','year':2025,'week':1}])
    rows=coverage.build_official_coverage_candidates({})
    assert rows and rows[0]['retrieved_at'] is None


def config():return {'pipeline_mode':'evidence','structured_task':{'disease':'dengue'},'universal':{'budget_limits':{'search':2,'extraction':2},'extraction_reserve':0}}


def test_same_session_cache_and_resume_keep_original_live_response_time(tmp_path,monkeypatch):
    events=fake_live(monkeypatch)
    ctx=RunContext(tmp_path,config())
    payload={'query':QUERY,'provider':'tavily','max_results':2}
    first=ctx.call('search',payload,lambda:providers.TavilySearchProvider().search(QUERY,max_results=2,timeout_seconds=1))
    assert first.results[0].retrieved_at==NOW
    def forbidden():raise AssertionError('Cached response cannot request or restamp')
    for active in [ctx,RunContext(tmp_path,config(),resume=True)]:
        raw=active.call('search',payload,forbidden)
        response=discovery._response_from_provider_output(raw,provider='tavily',planned_query=QUERY)
        assert [r.retrieved_at for r in response.results]==[NOW,NOW]
        assert active.ledger.snapshot()['used']['search']==1
    assert events.count('request')==events.count('clock')==1


def test_old_unknown_cached_response_stays_unknown_after_resume(tmp_path):
    ctx=RunContext(tmp_path,config())
    ctx.call('search',{'q':'old'},lambda:{'results':[{'url':'https://example.test/a'}]})
    def forbidden():raise AssertionError('No external call for old cached response')
    resumed=RunContext(tmp_path,config(),resume=True)
    response=discovery._response_from_provider_output(resumed.call('search',{'q':'old'},forbidden),provider='tavily',planned_query=QUERY)
    assert response.results[0].retrieved_at is None
