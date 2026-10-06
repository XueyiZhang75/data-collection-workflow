"""Evaluation-only final fact admission remains strict after page routing."""
import pytest
from test_deterministic_source_binding import _run, _diagnostics

@pytest.fixture(autouse=True)
def evidence(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    monkeypatch.setenv('ENABLE_LLM_EXTRACTION','false')
    monkeypatch.setenv('ENABLE_LANGSMITH_TRACE','false')

@pytest.mark.parametrize('other',['dengue','malaria','an unfamiliar syndrome'])
def test_heading_disease_does_not_qualify_local_other_disease_count(tmp_path,other):
    state=_run(tmp_path, '<h1>Cholera surveillance in Ghana, 2025</h1>'
        +f'<p>In 2025, 81 {other} cases were reported in Ghana.</p>',
        disease='cholera',country='Ghana',official=True)
    assert not state['qualified_records'], _diagnostics(state)
