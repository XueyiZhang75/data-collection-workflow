"""Per-strategy retries retain cumulative target history and actual ledger costs."""
import sqlite3

import pytest

from data_collection_workflow.session_runtime import RunContext, BudgetExceeded
from test_acquisition_budget import adaptive_config


def setup_target(tmp_path):
    context=RunContext(tmp_path/'session',adaptive_config(source_targets=1,http_requests=10))
    url='https://data.example/report'
    context.frontier.enqueue(target_id=url,url=url,source_id='s',payload={'entry':{'source_id':'s'}})
    return context,url


def fail():
    raise ConnectionError('transport failure')


def test_native_two_http_failures_then_partial_browser_can_retry_browser_once(tmp_path):
    ctx,url=setup_target(tmp_path)
    frontier=ctx.frontier
    frontier.claim_next(default_strategy='native')
    for _ in range(2):
        with pytest.raises(ConnectionError):
            ctx.call('fetch_ordinary',{'url':url},fail,source_target=url)
    frontier.finish(url,status='failed',operation_started=True)
    assert frontier.retry(url,strategy='browser')
    frontier.claim_next()
    def partial_browser():
        ctx.call('http_request',{'url':url,'browser_attempt':1},lambda:{'body':'partial'},source_target=url)
        raise ConnectionError('incomplete browser resources')
    with pytest.raises(ConnectionError):
        ctx.call('browser_navigation',{'url':url},partial_browser,source_target=url)
    frontier.finish(url,status='failed',operation_started=True)
    row=frontier.snapshot()['items'][0]
    assert row['attempts']==2
    assert row['strategy_attempts']=={'native':1,'browser':1}
    assert frontier.can_retry(url,strategy='browser')
    assert frontier.retry(url,strategy='browser')
    frontier.claim_next()
    ctx.call('browser_navigation',{'url':url},lambda:ctx.call('http_request',
        {'url':url,'browser_attempt':2},lambda:{'body':'complete'},source_target=url),source_target=url)
    frontier.finish(url,status='failed',operation_started=True,reason='test second completed transport lacked target data')
    row=frontier.snapshot()['items'][0]
    assert row['attempts']==3
    assert row['strategy_attempts']=={'native':1,'browser':2}
    assert not frontier.retry(url,strategy='browser')
    assert ctx.ledger.snapshot()['used']=={'source_targets':1,'http_requests':4,'browser':2}
    with pytest.raises(BudgetExceeded,match='attempt limit'):
        ctx.call('fetch_ordinary',{'url':url},fail,source_target=url)


def test_legacy_retry_keeps_total_attempt_cap_while_explicit_new_strategy_is_independent(tmp_path):
    ctx,url=setup_target(tmp_path)
    for attempt in range(2):
        ctx.frontier.claim_next()
        ctx.frontier.finish(url,status='failed',operation_started=True)
        if attempt==0:
            assert ctx.frontier.retry(url)
    assert not ctx.frontier.retry(url)
    assert ctx.frontier.can_retry(url,strategy='browser')
    assert ctx.frontier.retry(url,strategy='browser')
    row=ctx.frontier.claim_next()
    assert row['active_strategy']=='browser'
    assert row['payload']['entry']['acquisition_strategy']=='browser'
    assert row['attempts']==2
    ctx.frontier.finish(url,status='budget_deferred',operation_started=False)
    assert ctx.frontier.snapshot()['items'][0]['strategy_attempts']=={'native':2}


def test_source_override_and_default_strategy_are_persisted_before_dispatch(tmp_path):
    ctx,url=setup_target(tmp_path)
    row=ctx.frontier.claim_next(default_strategy='browser')
    assert row['active_strategy']=='browser'
    ctx.frontier.finish(url,status='failed',operation_started=True)
    assert ctx.frontier.retry(url,strategy='native')
    # Explicit retry survives a configured browser default.
    row=ctx.frontier.claim_next(default_strategy='browser')
    assert row['active_strategy']=='native'
    assert row['strategy_attempts']=={'browser':1}


def test_interrupted_first_browser_attempt_after_two_native_claims_remains_retryable(tmp_path):
    ctx,url=setup_target(tmp_path)
    for attempt in range(2):
        ctx.frontier.claim_next()
        with pytest.raises(ConnectionError):
            ctx.call('fetch_ordinary',{'url':url},fail,source_target=url)
        ctx.frontier.finish(url,status='failed',operation_started=True)
        if attempt==0:
            assert ctx.frontier.retry(url)
    assert ctx.frontier.retry(url,strategy='browser')
    ctx.frontier.claim_next()
    def interrupted():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        ctx.call('browser_navigation',{'url':url},interrupted,source_target=url)
    resumed=RunContext(ctx.session_dir,ctx.config,resume=True)
    row=resumed.frontier.snapshot()['items'][0]
    assert row['status']=='pending'
    assert row['attempts']==3
    assert row['strategy_attempts']=={'native':2,'browser':1}
    assert resumed.ledger.snapshot()['used']=={'source_targets':1,'http_requests':2,'browser':1}


def test_old_schema_preserves_unattributed_attempts_without_opening_extra_retries(tmp_path):
    from data_collection_workflow.acquisition_frontier import AcquisitionFrontier
    path=tmp_path/'old.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE acquisition_frontier (position INTEGER PRIMARY KEY,target_id TEXT UNIQUE,url TEXT,source_id TEXT,priority REAL,status TEXT,attempts INTEGER,payload TEXT,reason TEXT,budget_revision INTEGER,updated REAL,result_ref TEXT)')
        db.execute("INSERT INTO acquisition_frontier VALUES(1,'https://data.example/r','https://data.example/r','s',0,'failed',2,'{}',NULL,0,0,NULL)")
    frontier=AcquisitionFrontier(path)
    row=frontier.snapshot()['items'][0]
    assert row['attempts']==2
    assert row['strategy_attempts']=={'legacy_unknown':2}
    assert not frontier.retry(row['target_id'],strategy='browser')


def test_unknown_strategy_cannot_create_unbounded_new_attempt_buckets(tmp_path):
    ctx,url=setup_target(tmp_path)
    with pytest.raises(ValueError,match='strategy'):
        ctx.frontier.claim_next(default_strategy='browser-v99')
    row=ctx.frontier.claim_next()
    ctx.frontier.finish(url,status='failed',operation_started=True)
    with pytest.raises(ValueError,match='strategy'):
        ctx.frontier.retry(url,strategy='browser-v99')
