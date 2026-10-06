"""Tests for the LangGraph Studio setup (config + import + invocation sanity)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _sanity_initial_state() -> dict:
    return {
        "user_request": (
            "Collect global human hantavirus case, outbreak, and surveillance data "
            "from 2020 to 2026, including cases, deaths, dates, locations, source URLs, "
            "source types, and evidence quotes."
        ),
        "source_candidates": [],
        "source_discovery_summary": None,
        "source_registry": [],
        "source_registry_summary": None,
        "documents": [],
        "evidence_chunks": [],
        "raw_records": [],
        "validated_records": [],
        "normalized_records": [],
        "linked_events": [],
        "conflicts": [],
        "human_review_queue": [],
        "collection_trace": [],
        "collection_spec": None,
        "disease_profile": None,
        "collection_schema": None,
        "source_strategy": None,
        "screening_criteria": None,
        "search_queries": None,
        "search_query_inventory": [],
        "final_data_package": None,
        "current_route": None,
    }


def test_langgraph_json_exists_and_has_graph_entry():
    config_path = _PROJECT_ROOT / "langgraph.json"
    assert config_path.exists(), f"missing {config_path}"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert "." in config.get("dependencies", []), "dependencies should include '.'"
    graphs = config.get("graphs", {})
    assert "data_collection_workflow" in graphs
    assert (
        graphs["data_collection_workflow"]
        == "./src/data_collection_workflow/studio_app.py:graph"
    )
    assert config.get("env") == ".env"


def test_studio_app_exposes_compiled_graph():
    from data_collection_workflow import studio_app

    assert studio_app.graph is not None


def test_studio_graph_invocation_returns_final_package():
    from data_collection_workflow.studio_app import graph

    result = graph.invoke(_sanity_initial_state())
    assert result.get("final_data_package") is not None
    assert len(result.get("source_candidates") or []) >= 10
    assert len(result.get("source_registry") or []) >= 10


def test_print_studio_initial_state_script_outputs_valid_json():
    script = _PROJECT_ROOT / "scripts" / "print_studio_initial_state.py"
    request = "Collect measles case counts in Canada during 2024."
    completed = subprocess.run(
        [sys.executable, str(script), "--user-request", request],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["user_request"] == request
    assert payload.get("source_candidates") == []
    assert payload.get("search_query_inventory") == []
    assert payload.get("current_route") is None
