"""run_visual_workflow: interactive launcher for the Data Collection Langflow visual workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_collection_workflow.langflow_demo import (  # noqa: E402
    generated_session_id,
    normalize_date_range,
    normalize_session_id,
)
from data_collection_workflow.runtime_profile import load_project_env, resolve_llm_selection  # noqa: E402

import start_langflow_demo  # noqa: E402


def _prompt(label: str, current: str | None = None) -> str:
    suffix = f" [{current}]" if current else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or (current or "")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Data Collection Workflow through the Langflow visual interface."
    )
    parser.add_argument("--disease")
    parser.add_argument("--location")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--user-request")
    parser.add_argument("--session-id")
    parser.add_argument("--provider", choices=("anthropic", "openai"), default=None)
    parser.add_argument("--model", default=None, help="Your provider's exact model ID; otherwise use LLM_MODEL.")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--quick-test-mode", action="store_true")
    parser.add_argument("--api-port", type=int, default=8010)
    parser.add_argument("--langflow-port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def _collect_inputs(args: argparse.Namespace) -> dict[str, object]:
    disease = args.disease or _prompt("Disease / virus")
    location = args.location or _prompt("Location")
    start_raw = args.start_date or _prompt("Start date (YYYY, YYYY-M-D, or YYYY-MM-DD)")
    end_raw = args.end_date or _prompt("End date (YYYY, YYYY-M-D, or YYYY-MM-DD)")
    start_date, end_date = normalize_date_range(start_raw, end_raw)
    default_session = generated_session_id(disease, location, start_date, end_date)
    raw_session_id = args.session_id or _prompt(
        "Session id; press Enter to use the generated safe name",
        default_session,
    )
    session_id = normalize_session_id(raw_session_id or default_session)
    user_request = args.user_request
    if user_request is None:
        user_request = _prompt(
            "User request; press Enter to auto-generate",
            (
                f"Collect {disease} cases, deaths, dates, locations, source URLs, "
                f"source types, and evidence quotes for {location} from {start_date} to {end_date}."
            ),
        )
    return {
        "disease": disease,
        "location": location,
        "start_date": start_date,
        "end_date": end_date,
        "session_id": session_id,
        "user_request": user_request,
        "provider": args.provider,
        "model": args.model,
        "no_llm": bool(args.no_llm),
        "quick_test_mode": bool(args.quick_test_mode),
    }


def _start_args(collected: dict[str, object], args: argparse.Namespace) -> list[str]:
    start_args = [
        "--api-port",
        str(args.api_port),
        "--langflow-port",
        str(args.langflow_port),
        "--runner-disease",
        str(collected["disease"]),
        "--runner-location",
        str(collected["location"]),
        "--runner-start-date",
        str(collected["start_date"]),
        "--runner-end-date",
        str(collected["end_date"]),
        "--runner-session-id",
        str(collected["session_id"]),
        "--runner-user-request",
        str(collected["user_request"]),
        "--runner-provider",
        str(collected["provider"]),
        "--runner-model",
        str(collected["model"]),
    ]
    if collected["no_llm"]:
        start_args.append("--runner-no-llm")
    if collected["quick_test_mode"]:
        start_args.append("--runner-quick-test-mode")
    if args.no_browser:
        start_args.append("--no-browser")
    return start_args


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_project_env()
    args.provider, args.model = resolve_llm_selection(args.provider, args.model)
    try:
        collected = _collect_inputs(args)
    except ValueError as exc:
        print(f"Invalid visual workflow input: {exc}", file=sys.stderr)
        return 2

    print("Data Collection visual workflow launcher")
    print(f"session_id: {collected['session_id']}")
    print(f"date_range: {collected['start_date']} to {collected['end_date']}")
    print(f"quick_test_mode: {str(bool(collected['quick_test_mode'])).lower()}")
    print("Starting Langflow visual services. A prefilled session-specific flow will open when ready.")
    print(
        "Workflow state: NOT_STARTED. Click Play on "
        "`Data Collection Final Results - Run Full Workflow` once to start the full visual run."
    )
    return start_langflow_demo.main(_start_args(collected, args))


if __name__ == "__main__":
    raise SystemExit(main())
