"""Generate a Jupyter notebook replay from workflow run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .live_dashboard_data import load_dashboard_snapshot


def _markdown_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def _code_cell(source: str, output: Any | None = None) -> dict[str, Any]:
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": source,
        "outputs": [],
    }
    if output is not None:
        cell["outputs"].append(
            {
                "output_type": "execute_result",
                "metadata": {},
                "execution_count": None,
                "data": {"application/json": output},
            }
        )
    return cell


def _event_table(events: list[dict]) -> str:
    lines = [
        "| # | Type | Node | Status | Message | Duration ms |",
        "|---|---|---|---|---|---|",
    ]
    for event in events:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(event.get("sequence") or ""),
                    str(event.get("event_type") or ""),
                    str(event.get("node_name") or ""),
                    str(event.get("status") or ""),
                    str(event.get("message") or "").replace("|", "/")[:180],
                    str(event.get("duration_ms") or ""),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _artifact_table(artifact_paths: dict[str, str]) -> str:
    if not artifact_paths:
        return "No artifact paths were recorded."
    lines = ["| Artifact | Path |", "|---|---|"]
    for key, value in sorted(artifact_paths.items()):
        lines.append(f"| {key} | `{value}` |")
    return "\n".join(lines)


def build_workflow_replay_notebook(session_dir: str | Path) -> dict[str, Any]:
    """Build a minimal nbformat v4 notebook as a plain JSON structure."""

    root = Path(session_dir)
    snapshot = load_dashboard_snapshot(root)
    events = snapshot["events"]
    summary = snapshot["summary"]
    run_status = snapshot["run_status"]
    artifact_paths = snapshot["artifact_paths"]
    session_id = (
        summary.get("session_id")
        or run_status.get("session_id")
        or root.name
    )
    cells = [
        _markdown_cell(
            f"# Workflow Replay: {session_id}\n\n"
            "This notebook is generated from local run events and artifacts. "
            "It is intended for review, teaching, and supplementary material."
        ),
        _markdown_cell(
            "## Run Status\n\n"
            f"- Status: `{run_status.get('status') or summary.get('run_status') or 'unknown'}`\n"
            f"- Current node: `{run_status.get('current_node') or 'none'}`\n"
            f"- Event count: `{len(events)}`\n"
            f"- Normalized record count: `{summary.get('normalized_record_count', 0)}`\n"
            f"- Human review item count: `{summary.get('human_review_item_count', 0)}`"
        ),
        _markdown_cell("## Runtime Event Timeline\n\n" + _event_table(events)),
        _code_cell(
            "run_status = ...\nrun_summary = ...\nrun_events = ...",
            {
                "run_status": run_status,
                "run_summary": summary,
                "run_events": events[:200],
            },
        ),
        _markdown_cell("## Artifact Links\n\n" + _artifact_table(artifact_paths)),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "data_collection_workflow": {"session_dir": str(root), "session_id": session_id},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_workflow_replay_notebook(session_dir: str | Path) -> Path:
    root = Path(session_dir)
    notebook = build_workflow_replay_notebook(root)
    path = root / "workflow_replay_notebook.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
