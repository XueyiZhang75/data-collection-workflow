"""Explicit provider resume is separate from task/config mutation and report completion."""
import argparse
import copy
import importlib.util
import json
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest
from data_collection_workflow import result_manifest as manifest
from data_collection_workflow import session_runtime as runtime
from test_provider_account_guard import context,payload,ProviderError

COPY=Path(__file__).resolve().parents[1]
def load(name,file):
    spec=importlib.util.spec_from_file_location(name,COPY/'scripts'/file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
runner=load('provider_cli_review','run_workflow.py')
interactive=load('provider_interactive_review','collect.py')
RUN_GRAPH=runner._run_graph_with_events

@pytest.fixture(autouse=True)
def offline(monkeypatch,tmp_path):
    def forbidden(*args,**kwargs):raise AssertionError('No workflow, provider SDK, or network execution in CLI tests')
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    import requests,httpx
    monkeypatch.setattr(requests.Session,'request',forbidden)
    monkeypatch.setattr(httpx.Client,'send',forbidden)
    monkeypatch.setattr(httpx.AsyncClient,'send',forbidden)
    monkeypatch.setattr(runner,'_run_graph_with_events',forbidden)
    monkeypatch.setattr(runner,'_preflight_llm_with_trace_policy',forbidden)
    monkeypatch.setattr(runner,'workflow_output_dir_from_config',lambda c:tmp_path)
    monkeypatch.setattr(runner,'workflow_run_env_from_config',lambda c:{})
    from data_collection_workflow import llm_clients
    monkeypatch.setattr(llm_clients,'build_chat_model',forbidden)


def request(tmp_path,**changes):
    data={'provider':'anthropic','event_id':'restore-1','reason':'Account allowance was restored by the user.'};data.update(changes)
    p=tmp_path/'provider.json';p.write_text(json.dumps(data),encoding='utf-8');return p


def test_both_cli_parsers_accept_explicit_provider_resume():
    argv=['--resume-session','saved','--provider-resume','provider.json']
    assert runner._build_parser().parse_args(argv).provider_resume=='provider.json'
    args=interactive.build_parser().parse_args(argv)
    assert args.provider_resume=='provider.json'
    assert interactive._runner_args(Path('saved.json'),args).provider_resume=='provider.json'


@pytest.mark.parametrize('change',[{'model':'other'},{'budget':999},{'task':{}},{'reason':''},{'event_id':''},{'provider':'openai'},{'provider':True},{'event_id':4}])
def test_resume_request_cannot_smuggle_config_or_other_provider(tmp_path,change):
    with pytest.raises(ValueError):runner._load_provider_resume(request(tmp_path,**change),provider='anthropic')


def test_valid_resume_request_is_unchanged_and_fingerprint_excluded(tmp_path):
    p=request(tmp_path);value=runner._load_provider_resume(p,provider='anthropic')
    assert value==json.loads(p.read_text())
    assert set(value)=={'provider','event_id','reason'}


@pytest.mark.parametrize('revision,resume',[('evidence',None),('legacy','saved')])
def test_provider_resume_rejects_fresh_or_legacy_before_initialization(monkeypatch,tmp_path,revision,resume):
    cfg={'pipeline_mode':revision,'structured_task':{"disease": "measles", "location": "Canada", "start_date": "2024-01-01", "end_date": "2024-12-31"}}
    monkeypatch.setattr(runner,'_config_with_cli_overrides',lambda args:(None,cfg))
    monkeypatch.setattr(runtime,'prepare_universal_config',lambda c:c)
    monkeypatch.setattr(runtime,'initialize_universal_run',lambda *a,**k:pytest.fail('must reject before opening a new run'))
    args=SimpleNamespace(provider_resume=str(request(tmp_path)),resume_session=resume,budget_amendment=None)
    with pytest.raises(ValueError,match='provider-resume'):runner.run_workflow(args)


def test_same_version_validation_happens_before_provider_unblock(monkeypatch,tmp_path):
    cfg={'pipeline_mode':'evidence','structured_task':{"disease": "measles", "location": "Canada", "start_date": "2024-01-01", "end_date": "2024-12-31"},'output':{'session_id':'saved'},'llm':{'provider':'anthropic','model':'original'}}
    monkeypatch.setattr(runner,'_config_with_cli_overrides',lambda args:(None,cfg))
    monkeypatch.setattr(runtime,'prepare_universal_config',lambda c:c)
    monkeypatch.setattr(runner,'workflow_output_dir_from_config',lambda c:tmp_path)
    monkeypatch.setattr(runner,'workflow_run_env_from_config',lambda c:{})
    seen=[]
    def mismatch(*a,**k):seen.append('fingerprint');raise runtime.ResumeMismatch('task/config/policy mismatch')
    monkeypatch.setattr(runtime,'initialize_universal_run',mismatch)
    with pytest.raises(runtime.ResumeMismatch):runner.run_workflow(SimpleNamespace(provider_resume=str(request(tmp_path)),resume_session='saved',budget_amendment=None))
    assert seen==['fingerprint']


def test_resume_event_does_not_reset_usage_or_unlock_a_later_halt(tmp_path):
    c=context(tmp_path);c.provider_continuation=False
    with pytest.raises(runtime.ProviderAccountLimit):c.call('extraction',payload('first'),lambda:(_ for _ in ()).throw(ProviderError()))
    data=runner._load_provider_resume(request(tmp_path),provider='anthropic')
    old=c.ledger.snapshot()['used']
    runner._apply_provider_resume(c,data)
    assert c.provider_continuation is True and c.ledger.snapshot()['used']==old
    with pytest.raises(runtime.ProviderAccountLimit):c.call('extraction',payload('second'),lambda:(_ for _ in ()).throw(ProviderError()))
    c.provider_continuation=False
    runner._apply_provider_resume(c,data)
    assert c.provider_continuation is False
    assert c.ledger.provider_status('anthropic')['status']=='halted'


def test_absent_explicit_resume_never_unlocks(tmp_path):
    c=context(tmp_path);c.provider_continuation=False
    with pytest.raises(runtime.ProviderAccountLimit):c.call('extraction',payload('first'),lambda:(_ for _ in ()).throw(ProviderError()))
    before=c.ledger.snapshot();runner._apply_provider_resume(c,None)
    assert c.ledger.snapshot()==before and c.provider_continuation is False


def dataset():
    row={'record_id':'a','cases_confirmed':4,'product_kind':'aggregate','evidence_qualification':{'status':'qualified'}}
    return {'pipeline_mode':'evidence','final_dataset':[row],'final_case_dataset':[],'aggregate_dataset':[row],'candidate_records':[{'record_id':'c'}],'source_registry':[],'context_records':[]}


def test_provider_halt_report_preserves_data_and_distinguishes_partial_collection(tmp_path):
    package=dataset();state={'run_budget_ledger':{'providers':{'anthropic':{'status':'halted','reason':'account usage limit'}}}}
    m=manifest.build_result_manifest(package,state)
    assert m['technical_completion']=='completed'
    assert m['collection_status']=='partial'
    assert m['recovery_stop_reason']=='provider_account_limit'
    assert m['data_availability']=='qualified_aggregates_without_individual_cases'
    assert m['counts']['qualified_observations']==1 and m['counts']['candidate_records']==1
    assert manifest.manifest_summary(m)['collection_status']=='partial'
    package['result_manifest']=m;manifest.write_universal_outputs(package,tmp_path)
    for name in ['workflow_console_summary.json','workflow_console.html']:
        assert 'provider_account_limit' in (tmp_path/name).read_text(encoding='utf-8')


def test_previously_resumed_provider_does_not_force_partial_result():
    m=manifest.build_result_manifest(dataset(),{'run_budget_ledger':{'providers':{'anthropic':{'status':'resumed'}}},'recovery_stop_reason':'no_action'})
    assert m['collection_status']=='not_assessed' and m['recovery_stop_reason']=='no_action'


def test_offline_fixture_blocks_model_network_and_child_processes():
    import requests,httpx
    from data_collection_workflow import llm_clients
    attempts=[lambda:llm_clients.build_chat_model(),lambda:requests.get('https://invalid.example/'),lambda:httpx.Client().send(httpx.Request('GET','https://invalid.example/')),lambda:subprocess.Popen(['python','-c','pass'])]
    for attempt in attempts:
        with pytest.raises(AssertionError,match='No workflow'):attempt()


def test_halted_preflight_is_not_called_or_unlocked(monkeypatch,tmp_path):
    c=context(tmp_path)
    with pytest.raises(runtime.ProviderAccountLimit):c.call('extraction',payload('first'),lambda:(_ for _ in ()).throw(ProviderError()))
    before=c.ledger.snapshot()
    runner._preflight_universal_provider(c,'anthropic')
    assert c.ledger.snapshot()==before


def test_fresh_preflight_account_halt_permits_only_local_continuation(monkeypatch,tmp_path):
    c=context(tmp_path)
    def refuse():c.call('model:preflight',payload('preflight'),lambda:(_ for _ in ()).throw(ProviderError()))
    monkeypatch.setattr(runner,'_preflight_llm_with_trace_policy',refuse)
    runner._preflight_universal_provider(c,'anthropic')
    assert c.ledger.provider_status('anthropic')['status']=='halted'
    assert c.ledger.snapshot()['used']=={'model:preflight':1}


def test_preflight_other_errors_remain_errors(monkeypatch,tmp_path):
    c=context(tmp_path)
    def fail():raise ValueError('wrong model identifier')
    monkeypatch.setattr(runner,'_preflight_llm_with_trace_policy',fail)
    with pytest.raises(ValueError,match='wrong model'):runner._preflight_universal_provider(c,'anthropic')


def test_explicit_resume_reopens_saved_partial_once_without_replaying_acquisition(monkeypatch,tmp_path):
    from contextlib import contextmanager
    from typing import TypedDict
    from langgraph.graph import StateGraph,START,END
    from langgraph.checkpoint.memory import InMemorySaver
    class State(TypedDict,total=False):
        run_budget_ledger:dict
        recovery_stop_reason:str|None
        documents:list
    visits=[]
    c=context(tmp_path);c.session_dir=tmp_path;c.frontier=None;c.resume=False;c.provider_continuation=False;c.budget_continuation=False
    with pytest.raises(runtime.ProviderAccountLimit):c.call('extraction',payload('first'),lambda:(_ for _ in ()).throw(ProviderError()))
    def acquire(state):visits.append('acquire');return {'documents':[{'source_id':'saved'}]}
    def recovery(state):
        visits.append('recover')
        halted=(c.ledger.provider_status('anthropic') or {}).get('status')=='halted'
        return {'run_budget_ledger':c.ledger.snapshot(),'recovery_stop_reason':'provider_account_limit' if halted else 'no_action'}
    builder=StateGraph(State);builder.add_node('content_fetch_and_parse',acquire);builder.add_node('quality_gate_routing',lambda s:{});builder.add_node('recovery_control',recovery)
    builder.add_edge(START,'content_fetch_and_parse');builder.add_edge('content_fetch_and_parse','quality_gate_routing');builder.add_edge('quality_gate_routing','recovery_control');builder.add_edge('recovery_control',END)
    graph=builder.compile(checkpointer=InMemorySaver())
    @contextmanager
    def checkpoint(_):yield graph
    monkeypatch.setattr(runtime,'checkpoint_graph',checkpoint)
    monkeypatch.setattr(runner,'_flush_langsmith_tracers',lambda:None)
    with c.activate():first=RUN_GRAPH({},output_dir=tmp_path,session_id='test-only',live_status=False)
    assert first['recovery_stop_reason']=='provider_account_limit'
    c.resume=True
    with c.activate():still=RUN_GRAPH({},output_dir=tmp_path,session_id='test-only',live_status=False)
    assert visits==['acquire','recover'] and still['documents']==first['documents']
    data=runner._load_provider_resume(request(tmp_path),provider='anthropic');runner._apply_provider_resume(c,data)
    with c.activate():second=RUN_GRAPH({},output_dir=tmp_path,session_id='test-only',live_status=False)
    runner._apply_provider_resume(c,data)
    with c.activate():third=RUN_GRAPH({},output_dir=tmp_path,session_id='test-only',live_status=False)
    assert visits==['acquire','recover','recover']
    assert second['documents']==third['documents']==first['documents']
    assert third['recovery_stop_reason']=='no_action'


@pytest.mark.parametrize('state',[
    {'recovery_stop_reason':'budget_exhausted'},
    {'recovery_gaps':[{'kind':'missing_field','target_id':'candidate'}]},
    {'source_registry':[{'source_id':'pending','url':'https://invalid.example/report'}]},
])
def test_non_provider_unfinished_work_is_still_partial(state):
    assert manifest.build_result_manifest(dataset(),state)['collection_status']=='partial'


def test_collection_complete_requires_existing_coverage_decision():
    p=dataset();p['source_coverage_audit']={'coverage_complete':True,'coverage_status':'complete'}
    assert manifest.build_result_manifest(p,{})['collection_status']=='completed'
