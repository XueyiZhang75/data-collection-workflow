"""Resume and provider attempt accounting regressions, without paid calls."""
import pytest

from data_collection_workflow.session_runtime import RunContext, ResumeMismatch, initialize_universal_run
from data_collection_workflow.llm_clients import _invoke_with_config, _invoke_without_ledger


def test_mismatched_resume_rejects_before_preflight_writes(tmp_path, monkeypatch):
    config = {'pipeline_mode': 'evidence', 'structured_task': {'disease': 'measles'}}
    RunContext(tmp_path, config)
    report = tmp_path / 'acquisition-preflight.json'
    report.write_text('original preflight')
    def preflight(config, directory):
        report.write_text('overwritten preflight')
        return {'ready': True}
    monkeypatch.setattr('data_collection_workflow.document_acquisition.preflight_acquisition', preflight)
    with pytest.raises(ResumeMismatch, match='fingerprint mismatch'):
        initialize_universal_run({**config, 'structured_task': {'disease': 'dengue'}}, tmp_path, resume_session=tmp_path.name)
    assert report.read_text() == 'original preflight'


def test_missing_resume_rejects_before_creating_preflight_artifacts(tmp_path, monkeypatch):
    session = tmp_path / 'missing'
    def preflight(config, directory):
        directory.mkdir(parents=True)
        (directory / 'preflight.txt').write_text('side effect')
        return {'ready': True}
    monkeypatch.setattr('data_collection_workflow.document_acquisition.preflight_acquisition', preflight)
    with pytest.raises(ResumeMismatch, match='missing session'):
        initialize_universal_run({}, session, resume_session='missing')
    assert not session.exists()


def test_valid_resume_still_requires_successful_preflight(tmp_path, monkeypatch):
    config = {'pipeline_mode': 'evidence'}
    original = RunContext(tmp_path, config)
    original.ledger.begin('search', {'unfinished': True})
    def unavailable(*args):
        raise RuntimeError('browser unavailable')
    monkeypatch.setattr('data_collection_workflow.document_acquisition.preflight_acquisition', unavailable)
    with pytest.raises(RuntimeError, match='browser unavailable'):
        initialize_universal_run(config, tmp_path, resume_session=tmp_path.name)
    assert original.ledger.snapshot()['operations'] == {'running': 1}


def test_provider_typeerror_cannot_dispatch_twice_inside_one_charge(tmp_path, monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    ctx = RunContext(tmp_path, {'universal': {'extraction_reserve': 0}})
    class Model:
        calls = 0
        def invoke(self, messages, config=None):
            self.calls += 1
            if self.calls == 1:
                raise TypeError('provider response config could not be decoded')
            return {'records': []}
    model = Model()
    with ctx.activate(), pytest.raises(TypeError, match='could not be decoded'):
        _invoke_with_config(model, [], {'metadata': {'hdc_stage': 'structured_extraction'}})
    assert model.calls == 1
    assert ctx.ledger.snapshot()['used']['extraction'] == 1
    assert ctx.ledger.snapshot()['operations'] == {'failed': 1}


def test_legacy_no_config_callable_is_selected_before_invocation():
    class Model:
        def invoke(self, messages):
            return messages
    assert _invoke_without_ledger(Model(), ['payload'], {}) == ['payload']


def test_interrupted_structured_response_cannot_resume_via_json_fallback(tmp_path, monkeypatch):
    from pydantic import BaseModel
    from data_collection_workflow.llm_clients import run_pydantic_structured_llm
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    class Output(BaseModel):
        value: int
    class Model:
        calls = 0
        def with_structured_output(self, schema): return self
        def invoke(self, messages, config=None):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt('interrupted after provider dispatch')
            return {'value': 1}
    model = Model()
    monkeypatch.setattr('data_collection_workflow.llm_clients.build_chat_model', lambda *args, **kwargs: model)
    original = RunContext(tmp_path, {})
    with original.activate(), pytest.raises(KeyboardInterrupt):
        run_pydantic_structured_llm('system', 'user', Output)
    resumed = RunContext(tmp_path, {}, resume=True)
    with resumed.activate(), pytest.raises(ResumeMismatch, match='in_doubt'):
        run_pydantic_structured_llm('system', 'user', Output)
    assert model.calls == 1
    assert resumed.ledger.snapshot()['used'] == {'model:Output': 1}
    assert resumed.ledger.snapshot()['operations'] == {'in_doubt': 1}


def test_structured_budget_stop_preserves_budget_exception(tmp_path, monkeypatch):
    from pydantic import BaseModel
    from data_collection_workflow.llm_clients import run_pydantic_structured_llm
    from data_collection_workflow.session_runtime import BudgetExceeded
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    class Output(BaseModel):
        value: int
    class Model:
        def with_structured_output(self, schema): return self
        def invoke(self, messages, config=None):
            raise AssertionError('budget denied provider dispatch')
    monkeypatch.setattr('data_collection_workflow.llm_clients.build_chat_model', lambda *args, **kwargs: Model())
    ctx = RunContext(tmp_path, {'universal': {'model_limits': {'Output': 0}}})
    with ctx.activate(), pytest.raises(BudgetExceeded):
        run_pydantic_structured_llm('system', 'user', Output)
    assert ctx.ledger.snapshot()['used'] == {}


@pytest.mark.parametrize('section,legacy,stage', [
    ('source_identity', 'SourceIdentityAssessment', 'SourceIdentityAgentOutput'),
    ('source_credibility', 'SourceCredibilityAssessment', 'LLMSourceCredibilitySuggestion'),
])
@pytest.mark.parametrize('source_cap,legacy_cap,actual_cap,budget_cap', [
    (1, None, None, None), (5, 1, 4, None), (5, 4, 1, None), (1, 4, 5, None), (5, 4, 6, 1),
])
def test_actual_model_stage_uses_smallest_configured_limit(tmp_path, monkeypatch, section, legacy, stage, source_cap, legacy_cap, actual_cap, budget_cap):
    from data_collection_workflow.session_runtime import BudgetExceeded
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    model_limits = {name: value for name, value in [(legacy, legacy_cap), (stage, actual_cap)] if value is not None}
    config = {'llm': {section: {'max_sources': source_cap}}, 'universal': {'model_limits': model_limits}}
    if budget_cap is not None:
        config['universal']['budget_limits'] = {'model:' + legacy: budget_cap}
    ctx = RunContext(tmp_path, config)
    class Model:
        calls = 0
        def invoke(self, messages, config=None):
            self.calls += 1
            return {}
    model = Model()
    with ctx.activate():
        _invoke_with_config(model, ['first source'], {'metadata': {'hdc_stage': stage}})
        with pytest.raises(BudgetExceeded):
            _invoke_with_config(model, ['second source'], {'metadata': {'hdc_stage': stage}})
    assert model.calls == 1
    assert ctx.ledger.snapshot()['used'] == {'model:' + stage: 1}


def test_inactive_session_context_can_be_released_without_global_history(tmp_path):
    import gc
    import weakref
    from data_collection_workflow.session_runtime import RunContext, get_runtime

    context = RunContext(tmp_path / 'completed', {'pipeline_mode': 'evidence'})
    ref = weakref.ref(context)
    with context.activate():
        assert get_runtime() is context
    assert get_runtime() is None
    del context
    gc.collect()
    assert ref() is None, 'Completed session contexts must not accumulate in a process-wide history registry'
