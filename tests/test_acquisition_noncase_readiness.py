"""Typed non-case counts reach extraction without becoming case counts."""
import pytest

from test_deterministic_source_binding import _run, _diagnostics
from data_collection_workflow.nodes.extraction import _official_best_count_mentions
from data_collection_workflow.source_assertions import count_mentions


@pytest.fixture(autouse=True)
def offline_evidence(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    monkeypatch.setenv("ENABLE_LLM_EXTRACTION", "false")
    monkeypatch.setenv("ENABLE_LANGSMITH_TRACE", "false")


@pytest.mark.parametrize(("disease", "country", "label"), [
    ("Measles", "Canada", "hospitalizations"),
    ("Dengue", "Brazil", "hospitalisations"),
])
def test_hospitalization_counts_have_their_own_type_and_reach_real_producer(tmp_path, disease, country, label):
    text = f"During 2025, 12 {label} for {disease} were reported in {country}."
    mentions = count_mentions(text)
    assert len(mentions) == 1
    assert mentions[0]["field"] == "hospitalizations"
    assert mentions[0]["value"] == 12
    assert text[mentions[0]["char_start"]:mentions[0]["char_end"]] == mentions[0]["span"]
    case, death, _ = _official_best_count_mentions(text)
    assert case is None and death is None
    state = _run(tmp_path, text, disease=disease, country=country)
    assert state["extraction_attempted_chunk_ids"]
    qualified = state["qualified_records"]
    assert any(row.get("metric_name") == "hospitalizations" and row.get("metric_value") == 12
               for row in qualified), _diagnostics(state)
    assert not any(row.get(field) for row in qualified
                   for field in ("cases_confirmed", "cases_suspected", "cases_probable", "cases_unspecified"))


@pytest.mark.parametrize("text", [
    "During 2025, 12 hospitalizations for Measles were not reported in Canada.",
    "During 2025, at least 12 hospitalizations for Measles were reported in Canada.",
    "During 2025, 12% hospitalizations for Measles were reported in Canada.",
])
def test_nonaffirmative_bounded_and_percentage_hospitalizations_cannot_qualify_as_exact_counts(tmp_path, text):
    state = _run(tmp_path, text, disease="Measles", country="Canada")
    assert not any(row.get("metric_value") == 12 or row.get("hospitalizations") == 12
                   for row in state["qualified_records"]), _diagnostics(state)
