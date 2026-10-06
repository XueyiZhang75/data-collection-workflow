"""Independent recovery admission checks against real current-session ledger."""
import pytest

from data_collection_workflow.session_runtime import RunContext
from data_collection_workflow.document_acquisition import PARSER_VERSION
from data_collection_workflow.workflow_recovery import RecoveryGap, plan_recovery, _fetch_affordable
from test_acquisition_budget import adaptive_config


def test_charged_target_can_use_http_after_source_allowance_is_spent(tmp_path):
    ctx=RunContext(tmp_path,adaptive_config(source_targets=1,http_requests=2))
    source={'source_id':'s','url':'https://data.example/report'}
    with ctx.activate():
        with pytest.raises(ConnectionError):
            ctx.call('fetch_ordinary',{'parser_version':PARSER_VERSION,'url':source['url'],'max_bytes':20_000_000},
                lambda:(_ for _ in ()).throw(ConnectionError()),source_target=source['url'])
        remaining=ctx.ledger.snapshot()['remaining']
        assert remaining['source_targets']==0
        assert _fetch_affordable(source,{},remaining,ctx.ledger)
        assert not _fetch_affordable({**source,'url':'https://data.example/new'},{},remaining,ctx.ledger)


def test_complete_browser_cache_can_replay_when_all_dispatch_budgets_are_zero(tmp_path):
    config=adaptive_config(source_targets=1,http_requests=1)
    config['universal']['budget_limits']['browser']=1
    ctx=RunContext(tmp_path,config)
    source={'source_id':'s','url':'https://data.example/report'}
    with ctx.activate():
        ctx.call('http_request',{'url':source['url']},lambda:{'body':'ok'},source_target=source['url'])
        ctx.call('browser_navigation',{'parser_version':PARSER_VERSION,'url':source['url'],'render_wait_ms':300},
            lambda:{'body':'rendered'},source_target=source['url'])
        remaining=ctx.ledger.snapshot()['remaining']
        assert remaining['source_targets']==remaining['http_requests']==remaining['browser']==0
        assert _fetch_affordable(source,{},remaining,ctx.ledger,needs_browser=True)


def test_adaptive_ocr_can_process_remaining_affordable_pages_without_full_document_budget(tmp_path):
    config=adaptive_config()
    config['universal']['budget_limits']['ocr']=1
    ctx=RunContext(tmp_path,config)
    (tmp_path/'raw.pdf').write_bytes(b'%PDF-current-session')
    doc={'source_id':'s','document_id':'d','content_hash':'hash','raw_artifact_path':'raw.pdf',
         'request_success':True,'raw_content_complete':True,'parse_eligible':True,
         'content_readable':True,'clean_text':'Already read page 1.',
         'acquisition_incomplete':True,'budget_exhausted_kind':'ocr','unprocessed_pages':[2,3]}
    state={'documents':[doc],'source_registry':[{'source_id':'s'}]}
    gap=RecoveryGap('acquisition_incomplete','d','two pending pages',source_id='s',budget_kind='ocr',document_hash='hash')
    with ctx.activate():
        plan=plan_recovery([gap],state=state,budget=ctx.ledger)
    assert [(a.kind,a.target_id) for a in plan.actions]==[('reparse','d')]
