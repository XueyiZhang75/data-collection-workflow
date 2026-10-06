"""Sanity check for the LangGraph Studio entry point.

Imports the compiled `graph` exposed via `data_collection_workflow.studio_app`, invokes it
once with a local sanity initial state, and prints a short summary. Does not
launch any long-running Studio server.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_collection_workflow.studio_app import graph  # noqa: E402


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


def main() -> None:
    print(f"graph object type: {type(graph).__name__}")

    result = graph.invoke(_sanity_initial_state())

    final_data_package = result.get("final_data_package")
    candidates = result.get("source_candidates") or []
    registry = result.get("source_registry") or []
    trace = result.get("collection_trace") or []

    print(f"final_data_package exists: {final_data_package is not None}")
    print(f"source_candidates count: {len(candidates)}")
    print(f"source_registry count: {len(registry)}")
    print(f"collection_trace length: {len(trace)}")
    print("ordered node names:")
    for i, event in enumerate(trace, start=1):
        print(f"  {i:>2}. {event.get('node_name')}")


if __name__ == "__main__":
    main()
