"""Streamlit dashboard for a workflow session directory.

Run with:
    streamlit run scripts/dashboard.py -- --session-dir outputs/sessions/<session_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_collection_workflow.live_dashboard_data import load_dashboard_snapshot  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View live Data Collection Workflow run status.")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--refresh-seconds", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        import streamlit as st
    except ImportError:
        print(
            "Streamlit is not installed. Install the visualization extra with "
            "`pip install -e .[visualization]`.",
            file=sys.stderr,
        )
        return 2

    session_dir = Path(args.session_dir)
    snapshot = load_dashboard_snapshot(session_dir)
    run_status = snapshot["run_status"]
    summary = snapshot["summary"]
    events = snapshot["events"]
    node_status = snapshot["node_status"]

    st.set_page_config(page_title="Data Collection Workflow Live Dashboard", layout="wide")
    st.title("Data Collection Workflow Live Dashboard")
    st.caption(str(session_dir))

    status = run_status.get("status") or "unknown"
    current_node = run_status.get("current_node") or "none"
    cols = st.columns(4)
    cols[0].metric("Status", status)
    cols[1].metric("Current node", current_node)
    cols[2].metric("Events", len(events))
    cols[3].metric("Normalized records", summary.get("normalized_record_count", 0))

    node_rows = []
    for node_name, row in node_status.items():
        node_rows.append(
            {
                "node": node_name,
                "status": row.get("status"),
                "duration_ms": row.get("duration_ms"),
                "last_message": row.get("last_message"),
            }
        )
    event_rows = [
        {
            "sequence": event.get("sequence"),
            "type": event.get("event_type"),
            "node": event.get("node_name"),
            "status": event.get("status"),
            "message": event.get("message"),
            "duration_ms": event.get("duration_ms"),
        }
        for event in events[-200:]
    ]

    tab_nodes, tab_events, tab_artifacts, tab_raw = st.tabs(
        ["Nodes", "Events", "Artifacts", "Raw status"]
    )
    with tab_nodes:
        st.dataframe(node_rows, use_container_width=True)
    with tab_events:
        st.dataframe(event_rows, use_container_width=True)
    with tab_artifacts:
        st.dataframe(
            [
                {"artifact": key, "path": value}
                for key, value in sorted(snapshot["artifact_paths"].items())
            ],
            use_container_width=True,
        )
    with tab_raw:
        st.json({"run_status": run_status, "summary": summary})

    if status == "running":
        st.info(f"Refresh the page or enable Streamlit autorefresh every {args.refresh_seconds}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
