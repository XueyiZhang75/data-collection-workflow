"""Start LangGraph Studio from an Data Collection Workflow runtime profile."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_collection_workflow.workflow_run_config import (  # noqa: E402
    DEFAULT_WORKFLOW_RUN_CONFIG_PATH,
    api_key_present,
    safe_env_for_display,
    load_workflow_run_config,
    resolve_workflow_run_config_path,
    workflow_initial_state_from_config,
    workflow_run_config_with_overrides,
    workflow_run_env_from_config,
)
from data_collection_workflow.runtime_profile import load_project_env  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch LangGraph Studio with settings from a workflow runtime profile."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_WORKFLOW_RUN_CONFIG_PATH),
        help="Path to the workflow runtime JSON config file.",
    )
    parser.add_argument(
        "--enable-live-fetch",
        action="store_true",
        help="Override config and enable live HTTP fetch for this launch.",
    )
    parser.add_argument(
        "--disable-live-fetch",
        action="store_true",
        help="Override config and disable live HTTP fetch for this launch.",
    )
    parser.add_argument(
        "--enable-all-llm",
        action="store_true",
        help=(
            "Override config and enable source planning, source critic, and "
            "structured extraction LLM stages together."
        ),
    )
    parser.add_argument(
        "--disable-all-llm",
        action="store_true",
        help="Override config and disable all three LLM stages for this launch.",
    )
    parser.add_argument("--provider", choices=("anthropic", "openai"), default=None)
    parser.add_argument("--model", default=None, help="Your provider's exact model ID; otherwise use LLM_MODEL.")
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--llm-max-chunks", type=int, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--print-config-only",
        action="store_true",
        help="Print sanitized launch settings and Studio input without starting the server.",
    )
    return parser


def _studio_command(config: dict) -> list[str]:
    studio = config.get("studio") or {}
    command = ["langgraph", "dev"]
    if studio.get("no_reload", True):
        command.append("--no-reload")
    if studio.get("port"):
        command.extend(["--port", str(studio["port"])])
    return command


def _llm_enabled(env_updates: dict[str, str]) -> bool:
    return any(
        env_updates.get(key) == "true"
        for key in (
            "ENABLE_LLM_DISEASE_INTELLIGENCE",
            "ENABLE_LLM_SOURCE_PLANNING",
            "ENABLE_LLM_SOURCE_CRITIC",
            "ENABLE_LLM_SOURCE_CREDIBILITY",
            "ENABLE_LLM_SOURCE_IDENTITY",
            "ENABLE_LLM_EXTRACTION",
        )
    )


def _console_text(value) -> str:
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def _override_value(enable: bool, disable: bool) -> bool | None:
    if enable and disable:
        raise ValueError("Enable and disable flags cannot both be set.")
    if enable:
        return True
    if disable:
        return False
    return None


def _config_with_cli_overrides(args: argparse.Namespace) -> tuple[Path, dict]:
    config_path = resolve_workflow_run_config_path(args.config)
    config = load_workflow_run_config(config_path)
    return config_path, workflow_run_config_with_overrides(
        config,
        live_fetch=_override_value(args.enable_live_fetch, args.disable_live_fetch),
        all_llm=_override_value(args.enable_all_llm, args.disable_all_llm),
        provider=args.provider,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        llm_max_chunks=args.llm_max_chunks,
        port=args.port,
    )


def _print_launch_context(
    config_path: Path,
    config: dict,
    env_updates: dict[str, str],
) -> None:
    llm = config.get("llm") or {}
    workflow = config.get("workflow") or {}
    print("=" * 72)
    print("LangGraph Studio workflow launch")
    print(f"Studio graph: {workflow.get('graph_name', 'data_collection_workflow')}")
    print(f"config_path: {_console_text(config_path)}")
    print(f"profile_name: {config.get('profile_name')}")
    print(f"live_fetch_enabled: {env_updates.get('ENABLE_LIVE_FETCH')}")
    print(
        "llm_stages_enabled:",
        {
            "source_planning": env_updates.get("ENABLE_LLM_SOURCE_PLANNING"),
            "source_critic": env_updates.get("ENABLE_LLM_SOURCE_CRITIC"),
            "source_credibility": env_updates.get(
                "ENABLE_LLM_SOURCE_CREDIBILITY"
            ),
            "structured_extraction": env_updates.get("ENABLE_LLM_EXTRACTION"),
        },
    )
    print(f"provider: {llm.get('provider')}")
    print(f"model: {llm.get('model')}")
    print(f"api_key_present: {api_key_present(str(llm.get('provider') or ''))}")
    print("sanitized_environment:")
    print(json.dumps(safe_env_for_display(env_updates), indent=2, sort_keys=True))
    print("studio_minimal_input:")
    print(
        json.dumps(
            workflow_initial_state_from_config(config, include_empty_fields=False),
            indent=2,
        )
    )
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    load_project_env()
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        print("--timeout-seconds must be positive.", file=sys.stderr)
        return 2
    if args.llm_max_chunks is not None and args.llm_max_chunks <= 0:
        print("--llm-max-chunks must be positive.", file=sys.stderr)
        return 2

    try:
        config_path, config = _config_with_cli_overrides(args)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    env_updates = workflow_run_env_from_config(config)
    _print_launch_context(config_path, config, env_updates)
    if args.print_config_only:
        return 0

    provider = str((config.get("llm") or {}).get("provider") or "")
    if _llm_enabled(env_updates):
        if provider not in {"anthropic", "openai"}:
            print("Unsupported LLM provider. Choose anthropic or openai.", file=sys.stderr)
            return 2
        if not str((config.get("llm") or {}).get("model") or "").strip():
            print(
                "An LLM model ID is required. Set --model, llm.model in the config, "
                "or LLM_MODEL in the project .env file before starting Studio.",
                file=sys.stderr,
            )
            return 2
    if _llm_enabled(env_updates) and not api_key_present(provider):
        print(
            f"Missing API key for provider '{provider}'. Set the key in the "
            "environment before starting Studio with LLM stages enabled.",
            file=sys.stderr,
        )
        return 3

    launch_env = os.environ.copy()
    launch_env.update(env_updates)
    command = _studio_command(config)
    print("starting_command:", " ".join(command))
    completed = subprocess.run(command, cwd=_PROJECT_ROOT, env=launch_env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
