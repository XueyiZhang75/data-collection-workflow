"""Failed extraction recovery is explicit, bounded, and distinct from valid empty output."""
import pytest

from data_collection_workflow.workflow_recovery import assess_collection_gaps, execute_recovery, plan_recovery, RecoveryAction, RecoveryPlan
from data_collection_workflow.session_runtime import BudgetExceeded, RunContext


def context(tmp_path, limit=10):
    return RunContext(tmp_path, {'universal': {'budget_limits': {'extraction': limit}, 'extraction_reserve': 0}})


def fail():
    raise ValueError('invalid extraction response')


def state(*ids):
    return {'source_registry': [{'source_id': 's', 'source_role_final': 'excluded'}],
            'evidence_chunks': [{'chunk_id': key, 'source_id': 's', 'text': 'Source-bound cases.', 'contains_target_data': True} for key in ids],
            'extraction_attempted_chunk_ids': list(ids)}


def test_failed_and_valid_empty_operations_have_different_recovery_eligibility(tmp_path):
    ctx = context(tmp_path)
    with pytest.raises(ValueError):
        ctx.call('extraction', {'chunk_id': 'failed'}, fail)
    ctx.call('extraction', {'chunk_id': 'empty'}, lambda: {'records': []})
    with ctx.activate():
        gaps = assess_collection_gaps(state('failed', 'empty'))
        plan = plan_recovery(gaps, state=state('failed', 'empty'), budget=ctx.ledger)
    assert ctx.ledger.failed_chunk_ids() == ['failed']
    assert [(g.kind, g.target_id) for g in gaps] == [('extraction_failed', 'failed')]
    assert [(a.kind, a.target_id) for a in plan.actions] == [('extract', 'failed')]


def test_second_failure_is_retained_but_no_third_attempt_or_recovery_gap(tmp_path):
    ctx = context(tmp_path)
    for version in [1, 2]:
        with pytest.raises(ValueError):
            ctx.call('extraction', {'chunk_id': 'c', 'prompt_version': version}, fail, recovery=version == 2)
    assert ctx.ledger.failed_chunk_ids() == ['c']
    assert ctx.ledger.failed_chunk_ids(retryable_only=True) == []
    with pytest.raises(BudgetExceeded, match='attempt limit'):
        ctx.call('extraction', {'chunk_id': 'c', 'prompt_version': 3}, lambda: {'records': []}, recovery=True)
    with ctx.activate():
        assert assess_collection_gaps(state('c')) == []
    assert ctx.ledger.snapshot()['used']['extraction'] == 2


def test_batch_maps_actual_rows_and_changed_batch_cannot_bypass_attempt_cap(tmp_path):
    ctx = context(tmp_path)
    with pytest.raises(ValueError):
        ctx.call('extraction', {'chunk_id': 'batch-a', 'chunk_ids': ['row-a', 'row-b']}, fail)
    assert {'row-a', 'row-b'} <= set(ctx.ledger.attempted_chunk_ids())
    with pytest.raises(ValueError):
        ctx.call('extraction', {'chunk_id': 'batch-b', 'chunk_ids': ['row-a']}, fail, recovery=True)
    assert 'row-a' not in ctx.ledger.failed_chunk_ids(retryable_only=True)
    assert 'row-b' in ctx.ledger.failed_chunk_ids(retryable_only=True)
    with pytest.raises(BudgetExceeded):
        ctx.call('extraction', {'chunk_id': 'batch-c', 'chunk_ids': ['row-a']}, lambda: {'records': []}, recovery=True)
    assert ctx.ledger.snapshot()['used']['extraction'] == 2


@pytest.mark.parametrize('retry_fails', [False, True])
def test_recovery_action_status_tracks_ledger_result_not_swallowed_error(monkeypatch, tmp_path, retry_fails):
    import data_collection_workflow.nodes.extraction as extraction
    ctx = context(tmp_path)
    with pytest.raises(ValueError):
        ctx.call('extraction', {'chunk_id': 'c'}, fail)

    def extract(local):
        assert local['extraction_attempted_chunk_ids'] == []
        try:
            ctx.call('extraction', {'chunk_id': 'c'}, fail if retry_fails else lambda: {'records': []})
        except ValueError:
            pass  # Production extraction catches a chunk failure and continues.
        return {'raw_records': [], 'extraction_attempted_chunk_ids': ctx.ledger.attempted_chunk_ids()}

    monkeypatch.setattr(extraction, 'structured_extraction', extract)
    with ctx.activate():
        delta = execute_recovery(RecoveryPlan([RecoveryAction('extract', 'c', 'retry-c', 'recover failed response')]), context=ctx, artifacts=state('c'), budget=ctx.ledger)
    assert delta.actions[0]['status'] == ('failed' if retry_fails else 'completed')
    assert bool(ctx.ledger.failed_chunk_ids()) is retry_fails
    assert ctx.ledger.snapshot()['used']['extraction'] == 2


def test_retry_respects_exhausted_extraction_budget(tmp_path):
    ctx = context(tmp_path, limit=1)
    with pytest.raises(ValueError):
        ctx.call('extraction', {'chunk_id': 'c'}, fail)
    with ctx.activate():
        gaps = assess_collection_gaps(state('c'))
        plan = plan_recovery(gaps, state=state('c'), budget=ctx.ledger)
    assert gaps[0].kind == 'extraction_failed'
    assert plan.actions == []
    assert plan.stop_reason == 'budget_exhausted'


def test_successful_retry_clears_failed_status_and_uses_valid_empty_cache(tmp_path):
    ctx = context(tmp_path)
    with pytest.raises(ValueError):
        ctx.call('extraction', {'chunk_id': 'c'}, fail)
    ctx.call('extraction', {'chunk_id': 'c'}, lambda: {'records': []}, recovery=True)
    assert ctx.ledger.failed_chunk_ids() == []
    assert ctx.call('extraction', {'chunk_id': 'c'}, fail, recovery=True) == {'records': []}
    assert ctx.ledger.snapshot()['used']['extraction'] == 2

@pytest.mark.parametrize('acquisition_status', ['blocked', 'error_page', 'http_error'])
def test_recovery_fetch_of_unreadable_error_content_remains_failed(monkeypatch, tmp_path, acquisition_status):
    from data_collection_workflow.nodes import content_processing as content
    ctx = context(tmp_path)
    monkeypatch.setattr(content, 'content_fetch_and_parse', lambda local: {'documents': [
        {'source_id': 's', 'acquisition_status': acquisition_status, 'content_readable': False, 'parse_status': 'failed'}]})
    monkeypatch.setattr(content, 'document_quality_check', lambda local: {})
    monkeypatch.setattr(content, 'evidence_chunking_and_data_presence_flagging', lambda local: {'evidence_chunks': []})
    delta = execute_recovery(RecoveryPlan([RecoveryAction('fetch', 's', 'fetch-s', 'recover content')]), context=ctx,
                             artifacts={'source_registry': [{'source_id': 's', 'url': 'https://offline.invalid/report'}]}, budget=ctx.ledger)
    assert delta.actions[0]['status'] == 'failed'
    assert acquisition_status in delta.actions[0]['error']
