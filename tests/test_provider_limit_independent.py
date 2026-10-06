import pytest
from data_collection_workflow import session_runtime as rt
from test_provider_account_guard import context, payload, ProviderError


def test_provider_stop_is_per_session_and_real_denial_remains_charged(tmp_path):
    (tmp_path/'a').mkdir()
    (tmp_path/'b').mkdir()
    first=context(tmp_path/'a')
    second=context(tmp_path/'b')
    with pytest.raises(rt.ProviderAccountLimit):
        first.call('extraction',payload('first'),lambda:(_ for _ in ()).throw(ProviderError()))
    assert second.call('extraction',payload('first'),lambda:'independent')=='independent'
    assert second.ledger.provider_status('anthropic') is None
    assert first.ledger.snapshot()['operations']=={'failed':1}
    assert first.ledger.snapshot()['used']=={'extraction':1}
    assert first.ledger.attempted_chunk_ids()==['first']


def test_cached_success_and_uncached_blocked_calls_have_distinct_receipts(tmp_path):
    ctx=context(tmp_path)
    with rt.capture_model_dispatch() as first:
        assert ctx.call('extraction',payload('cached'),lambda:[])==[]
    with pytest.raises(rt.ProviderAccountLimit):
        ctx.call('extraction',payload('denied'),lambda:(_ for _ in ()).throw(ProviderError()))
    before=ctx.ledger.snapshot()['used']
    with rt.capture_model_dispatch() as cached:
        assert ctx.call('extraction',payload('cached'),lambda:pytest.fail('external call'))==[]
    with rt.capture_model_dispatch() as blocked:
        with pytest.raises(rt.ProviderAccountLimit) as error:
            ctx.call('extraction',payload('new'),lambda:pytest.fail('external call'))
    assert first['dispatched'] is True and cached['dispatched'] is False
    assert blocked['dispatched'] is not True and not error.value.dispatched
    assert ctx.ledger.snapshot()['used']==before
    assert ctx.ledger.attempted_chunk_ids()==['cached','denied']


def test_resume_new_event_does_not_change_limits_or_failure_attempt_cap(tmp_path):
    ctx=context(tmp_path)
    for attempt in range(2):
        with pytest.raises(rt.ProviderAccountLimit):
            ctx.call('extraction',payload('same'),lambda:(_ for _ in ()).throw(ProviderError()))
        ctx.ledger.resume_provider(provider='anthropic',event_id=f'restored-{attempt}',reason='account adjusted externally')
    assert ctx.ledger.snapshot()['used']=={'extraction':2}
    with pytest.raises(rt.BudgetExceeded):
        ctx.call('extraction',payload('same'),lambda:pytest.fail('third actual request'))
    assert ctx.ledger.snapshot()['used']=={'extraction':2}


def test_temporary_error_does_not_prevent_other_work(tmp_path):
    ctx=context(tmp_path)
    with pytest.raises(ProviderError):
        ctx.call('model:plan',payload('invalid'),lambda:(_ for _ in ()).throw(ProviderError('Rate limit exceeded',429,'rate_limit_error')))
    assert ctx.ledger.provider_status('anthropic') is None
    assert ctx.call('extraction',payload('fresh'),lambda:'ok')=='ok'
