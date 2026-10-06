"""Independent metric-type boundaries through the public deterministic route."""
import socket
import pytest
from test_deterministic_source_binding import _run, _diagnostics


@pytest.fixture(autouse=True)
def offline_evidence(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    monkeypatch.setenv("ENABLE_LLM_EXTRACTION", "false")
    monkeypatch.setenv("ENABLE_LANGSMITH_TRACE", "false")
    def forbidden(*args, **kwargs):
        raise AssertionError("External network is forbidden in metric review")
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def test_mixed_case_hospitalization_and_death_counts_keep_their_types(tmp_path):
    state = _run(tmp_path, "During 2025, France reported 23 confirmed Pertussis cases, 12 hospitalizations and 4 deaths.")
    rows = state["qualified_records"]
    assert any(row.get("cases_confirmed") == 23 for row in rows), _diagnostics(state)
    assert any(row.get("metric_name") == "hospitalizations" and row.get("metric_value") == 12 for row in rows), _diagnostics(state)
    assert any(row.get("deaths") == 4 for row in rows), _diagnostics(state)
    assert not any(row.get(key) == 12 for row in state.get("raw_records", [])
                   for key in ("cases_confirmed", "cases_unspecified", "deaths")), _diagnostics(state)


@pytest.mark.parametrize("amount", ["12%", "at least 12", "12-15"])
def test_nonexact_hospitalization_expression_does_not_qualify_as_exact_count(tmp_path, amount):
    state = _run(tmp_path, f"During 2025, {amount} hospitalizations for Pertussis were reported in France.")
    assert not any(row.get("metric_name") == "hospitalizations" and row.get("metric_value") in {12, 15}
                   for row in state["qualified_records"]), _diagnostics(state)


def test_case_count_does_not_become_hospitalization_metric(tmp_path):
    state = _run(tmp_path, "During 2025, France reported 12 confirmed Pertussis cases.")
    assert state["qualified_records"], _diagnostics(state)
    assert not any(row.get("hospitalizations") is not None or row.get("metric_name") == "hospitalizations"
                   for row in state.get("raw_records", [])), _diagnostics(state)
