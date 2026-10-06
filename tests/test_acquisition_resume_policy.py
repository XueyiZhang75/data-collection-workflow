"""Resume identity covers executable entry points, not generated artifacts."""
from pathlib import Path

import pytest

from data_collection_workflow import session_runtime as runtime
from test_acquisition_budget import adaptive_config


@pytest.fixture
def isolated_policy_tree(tmp_path,monkeypatch):
    repo=tmp_path/'repository'
    package=repo/'src/data_collection_workflow'
    package.mkdir(parents=True)
    module=package/'session_runtime.py'
    module.write_text('# stable source policy\n',encoding='utf-8')
    runners=repo/'scripts'
    runners.mkdir()
    for name in ('run_workflow.py','collect.py'):
        (runners/name).write_text('# entry point version one\n',encoding='utf-8')
    monkeypatch.setattr(runtime,'__file__',str(module))
    return repo


@pytest.mark.parametrize('runner',['run_workflow.py','collect.py'])
def test_runner_code_change_invalidates_existing_session_before_budget_amendment(tmp_path,isolated_policy_tree,runner):
    config=adaptive_config()
    session=tmp_path/'session'
    original=runtime.RunContext(session,config)
    before=original.fingerprint
    (isolated_policy_tree/'scripts'/runner).write_text('# changed resume execution policy\n',encoding='utf-8')
    with pytest.raises(runtime.ResumeMismatch,match='fingerprint'):
        runtime.RunContext(session,config,resume=True)
    assert original.ledger.budget_revision==0
    assert original.fingerprint==before


def test_generated_artifacts_and_unrelated_reporting_script_do_not_change_execution_identity(isolated_policy_tree):
    before=runtime.policy_fingerprint()
    for relative in ('outputs/session/raw.txt','docs/notes.md','scripts/render_report.py'):
        path=isolated_policy_tree/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('generated output, not execution policy',encoding='utf-8')
    assert runtime.policy_fingerprint()==before


def test_source_policy_resource_change_still_changes_execution_identity(isolated_policy_tree):
    before=runtime.policy_fingerprint()
    (isolated_policy_tree/'src/data_collection_workflow/policy.json').write_text('{"limit": 3}',encoding='utf-8')
    assert runtime.policy_fingerprint()!=before


def test_same_code_budget_amendment_keeps_cache_and_fingerprint(tmp_path,isolated_policy_tree):
    config=adaptive_config(source_targets=1,http_requests=1)
    session=tmp_path/'session'
    first=runtime.RunContext(session,config)
    url='https://data.example/report'
    expected={'body':'same-code complete response'}
    first.call('http_request',{'url':url},lambda:expected,source_target=url)
    resumed=runtime.RunContext(session,config,resume=True)
    resumed.amend_budget(amendment_id='continue',increases={'http_requests':2},reason='Explicit allowance')
    def forbidden():
        pytest.fail('same-code allowance change must preserve response cache')
    assert resumed.call('http_request',{'url':url},forbidden,source_target=url)==expected
    assert first.fingerprint==resumed.fingerprint
    assert resumed.ledger.snapshot()['used']['http_requests']==1


def test_source_target_membership_is_read_only_canonical_and_persistent(tmp_path):
    config=adaptive_config(source_targets=1,http_requests=2)
    context=runtime.RunContext(tmp_path/'session',config)
    target='https://DATA.example/report#search-fragment'
    before=context.ledger.path.read_bytes()
    assert context.ledger.has_source_target(target) is False
    assert context.ledger.path.read_bytes()==before
    def interrupted_transport():
        raise ConnectionError('outcome unknown')
    with pytest.raises(ConnectionError):
        context.call('http_request',{'url':target},interrupted_transport,source_target=target)
    used=context.ledger.snapshot()['used']
    before=context.ledger.path.read_bytes()
    assert context.ledger.has_source_target('https://data.example/report') is True
    assert context.ledger.has_source_target('https://data.example/new') is False
    assert context.ledger.path.read_bytes()==before
    assert context.ledger.snapshot()['used']==used
    resumed=runtime.RunContext(tmp_path/'session',config,resume=True)
    assert resumed.ledger.has_source_target(target) is True
    assert resumed.ledger.adaptive is True
