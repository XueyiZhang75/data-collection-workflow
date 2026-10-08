"""Interactive real-run entrypoint for the data collection workflow."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data_collection_workflow.environment import get_env
from data_collection_workflow.runtime_profile import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    PIPELINE_MODE,
    default_workflow_run_config,
    temporary_workflow_env,
    workflow_run_env_from_config,
)
from data_collection_workflow import llm_clients  # noqa: E402
from data_collection_workflow.langflow_demo import (  # noqa: E402
    apply_quick_test_mode,
    generated_session_id,
    normalize_date_range,
    normalize_session_id as normalize_workflow_session_id,
)
from run_workflow import (  # noqa: E402
    _preflight_llm_with_trace_policy,
    run_workflow,
)


def _configure_utf8_stdio(*, stdout=None, stderr=None) -> None:
    """Prefer UTF-8 console output so Windows paths with non-ASCII chars print safely."""

    for stream in (stdout or sys.stdout, stderr or sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower()
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure) and encoding not in {"utf-8", "utf8"}:
            reconfigure(encoding="utf-8", errors="replace")


DEFAULT_TARGET_FIELDS = [
    "disease",
    "country",
    "subnational_location",
    "locality",
    "date_reported",
    "cases_confirmed",
    "cases_probable",
    "cases_suspected",
    "cases_unspecified",
    "deaths",
    "hospitalizations",
    "source_url",
    "source_type",
    "evidence_quote",
]

LLM_KEY_BY_PROVIDER = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "workflow_run"


def _normalize_session_id(value: str, default: str) -> str:
    raw = value.strip()
    if not raw or raw in {"?", "？"}:
        return default
    cleaned = normalize_workflow_session_id(raw)
    if cleaned == "workflow_run" and raw != "workflow_run":
        return default
    return cleaned or default


def _prompt(label: str, current: str | None = None) -> str:
    suffix = f" [{current}]" if current else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or (current or "")


def _env_present(name: str) -> bool:
    return bool(get_env(name))


def _provider_key_name(provider: str) -> str:
    return LLM_KEY_BY_PROVIDER.get(provider.strip().lower(), f"{provider.upper()}_API_KEY")


def _default_user_request(disease: str, location: str, start_date: str, end_date: str) -> str:
    return (
        f"Collect {disease} cases, deaths, dates, locations, source URLs, "
        f"source types, and evidence quotes for {location} from {start_date} to {end_date}."
    )


def _require_keys(*, provider: str, llm_enabled: bool) -> list[str]:
    missing: list[str] = []
    if not _env_present("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    if llm_enabled:
        llm_key = _provider_key_name(provider)
        if not _env_present(llm_key):
            missing.append(llm_key)
    return missing


def _real_run_config(
    *,
    disease: str,
    location: str,
    start_date: str,
    end_date: str,
    target_fields: list[str],
    session_id: str,
    provider: str,
    model: str,
    output_dir: str | None,
    no_llm: bool,
    user_request: str | None = None,
    quick_test_mode: bool = False,
    audit_mode: bool = False,
    llm_thinking: str = "auto",
    llm_effort: str | None = None,
    llm_output_mode: str = "auto",
) -> dict:
    config = deepcopy(default_workflow_run_config())
    config["pipeline_mode"] = PIPELINE_MODE
    request_text = user_request or _default_user_request(disease, location, start_date, end_date)
    collection_mode = "standard" if audit_mode else "direct_collection"
    validation_mode = "live_cross_source" if audit_mode else "diagnostic_only"

    config["profile_name"] = f"{_slug(disease)}_{_slug(location)}_{start_date}_{end_date}_real_workflow"
    config["description"] = (
        "Interactive real data collection workflow run. Live search, live fetch, "
        "and LLM stages are enabled by default; API keys are read from environment variables."
    )
    config["workflow"] = {
        "collection_mode": collection_mode,
        "use_fixture_documents": False,
    }
    config["user_request"] = request_text
    config["structured_task"] = {
        "disease": disease,
        "location": location,
        "start_date": start_date,
        "end_date": end_date,
        "target_fields": target_fields,
        "source_preferences": [
            "official_public_health_agency",
            "international_organization_report",
            "structured_database",
            "peer_reviewed_literature",
            "news_and_situation_report",
        ],
        "collection_mode": collection_mode,
        "user_request": request_text,
        "run_label": session_id,
    }
    config["live_web"] = {
        "enabled": True,
        "timeout_seconds": 30,
    }
    config["source_search"] = {
        "enabled": True,
        "mode": "live",
        "provider": "tavily",
        "fixture_path": "src/data_collection_workflow/resources/search_fixtures/example_search_results.json",
        "max_queries": 16,
        "max_results_per_query": 16,
        "max_total_results": 400,
        "timeout_seconds": 60,
        "combine_with_seed_catalog": True,
        "cache_enabled": True,
        "provider_channel_allowlist": [
            "web_search",
            "official_site_search",
            "news_search",
            "literature_api",
            "database_search",
        ],
        "iterative": {
            "enabled": True,
            "max_iterations": 3 if audit_mode else 3,
            "max_queries_per_iteration": 4,
            "max_total_queries": 20 if audit_mode else 20,
            "max_total_results": 200 if audit_mode else 260,
            "require_llm": True,
            "allow_deterministic_fallback": False,
            "stop_when_llm_says_sufficient": True,
            "require_observation_after_each_iteration": True,
        },
        "authority_gap_retry": {
            "enabled": True,
            "max_queries": 24,
            "known_domain_max_queries": 14,
            "jurisdiction_max_queries": 16,
            "official_page_family_max_queries": 14,
            "result_budget": 120,
            "max_iterations": 1,
        },
    }
    config["content_fetch"] = {
        "fetch_search_derived_sources": True,
        "max_search_derived_sources": 50 if audit_mode else 50,
        "max_total_sources": 50 if audit_mode else 50,
        "min_credibility_score": 0.55,
        "allowed_final_roles": [
            "collection",
            "validation",
            "collection_support",
            "context",
        ],
        "allow_needs_review": True,
        "domain_allowlist": [],
        "domain_blocklist": [],
        "max_bytes": 1_000_000,
        "parse_pdf_text": True,
        "parse_tables": True,
        "store_raw_text": False,
        "user_agent": "data-collection-workflow/0.1",
        "content_fixture_map_path": None,
        "external_fetch": {
            "enabled": True,
            "provider_order": ["tavily_extract", "native_requests"],
            "tavily_extract": {
                "format": "markdown",
                "extract_depth": "advanced",
                "timeout_seconds": 45,
                "chunks_per_source": 5,
            },
            "adaptive_budget": {
                "max_candidate_urls": 120 if audit_mode else 50,
                "max_fetch_urls": 50 if audit_mode else 50,
                "min_usable_documents": 20,
                "min_collection_sources": 6,
                "min_validation_sources": 3 if audit_mode else 0,
                "max_iterations": 3 if audit_mode else 2,
                "stop_when_llm_says_sufficient": True,
            },
        },
    }
    llm_enabled = not no_llm
    config["llm"] = {
        "provider": provider,
        "model": model,
        "thinking": llm_thinking,
        "effort": llm_effort,
        "structured_output_method": llm_output_mode,
        "source_planning_enabled": llm_enabled,
        "source_critic_enabled": llm_enabled,
        "structured_extraction_enabled": llm_enabled,
        "max_chunks": 400,
        "max_tokens": 8192,
        "extraction": {
            "focused_recovery_reserved_calls": 120,
            "max_domain_call_share": 0.20,
            "high_value_source_min_spans": 2,
            "event_source_min_spans": 1,
            "scheduler": {
                "mode": "quality_adaptive",
                "max_concurrency": 4,
                "soft_checkpoint_calls": 400,
                "safety_max_calls": 2400,
                "rolling_yield_window": 40,
            },
        },
        "focused_recovery": {
            "max_calls": 120,
            "max_sources": 80,
        },
        "fallback_to_rule_based": False,
        "source_critic": {
            "max_sources": 6 if audit_mode else 4,
            "review_blocks_fetch": False,
        },
        "source_credibility": {
            "enabled": llm_enabled,
            "max_sources": 6 if audit_mode else 4,
            "source_id_allowlist": [],
        },
        "source_identity": {
            "enabled": llm_enabled,
            "max_sources": 30 if audit_mode else 16,
            "post_fetch": True,
            "require_llm": llm_enabled,
            "allow_deterministic_fallback": not llm_enabled,
        },
        "must_fetch_min_chunks_per_source": 10,
        "official_extraction_max_chunks": 120,
    }
    config["disease_intelligence"] = {
        "llm_enabled": llm_enabled,
        "force_llm": llm_enabled,
        "fallback_to_curated": True,
    }
    config["human_review"] = {
        "enabled": False,
        "decisions_path": None,
        "apply_decisions": False,
        "require_reviewer_id": True,
    }
    config["validation"] = {
        "mode": validation_mode,
        "held_out_records_path": None,
        "allow_incompatible_validation_records": False,
    }
    config["source_sets"] = {
        "source_id_allowlist_enabled": False,
        "collection_source_ids": [],
        "context_source_ids": [],
        "validation_reserved_source_ids": [],
        "workflow_source_ids": [],
        "llm_source_critic_source_ids": [],
    }
    config["output"] = {
        "run_output_root": output_dir or "outputs",
        "sessionized": True,
        "session_id": session_id,
        "auto_build_console": True,
        "console_output_root": "outputs/workflow_console",
        "write_latest_alias": True,
    }
    if quick_test_mode:
        apply_quick_test_mode(config)
        config.setdefault("structured_task", {})["quick_test_mode"] = True
    return config


def _collect_inputs(args: argparse.Namespace) -> dict:
    prompt_mode = not (args.disease and args.location and args.start_date and args.end_date)
    disease = args.disease or _prompt("Disease / virus")
    location = args.location or _prompt("Location")
    start_raw = args.start_date or _prompt("Start date (YYYY, YYYY-M-D, or YYYY-MM-DD)")
    end_raw = args.end_date or _prompt("End date (YYYY, YYYY-M-D, or YYYY-MM-DD)")
    start_date, end_date = normalize_date_range(start_raw, end_raw)
    target_fields = args.target_field or list(DEFAULT_TARGET_FIELDS)
    default_session_id = generated_session_id(disease, location, start_date, end_date)
    raw_session_id = args.session_id or (
        _prompt("Session id; press Enter to use the generated safe name", default_session_id)
        if (sys.stdin.isatty() or prompt_mode)
        else default_session_id
    )
    session_id = _normalize_session_id(raw_session_id, default_session_id)
    if raw_session_id != session_id:
        print(f"Using safe session id: {session_id}")
    default_request = _default_user_request(disease, location, start_date, end_date)
    if args.user_request is not None:
        user_request = args.user_request
    elif prompt_mode:
        user_request = _prompt("User request; press Enter to auto-generate", default_request)
    else:
        user_request = default_request
    return {
        "disease": disease,
        "location": location,
        "start_date": start_date,
        "end_date": end_date,
        "target_fields": target_fields,
        "session_id": session_id,
        "user_request": user_request,
    }


def _sanitized_preview(config: dict, provider: str, no_llm: bool) -> dict:
    return {
        "project_name": "data collection workflow",
        "mode": "interactive_real_run",
        "api_keys": {
            "tavily_api_key_present": _env_present("TAVILY_API_KEY"),
            "llm_provider": provider,
            "llm_api_key_name": None if no_llm else _provider_key_name(provider),
            "llm_api_key_present": True if no_llm else _env_present(_provider_key_name(provider)),
        },
        "config": config,
    }


def _write_generated_config(config: dict, session_id: str) -> Path:
    output = PROJECT_ROOT / "outputs" / "generated_configs" / f"{session_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(config, indent=2, ensure_ascii=False)
    if config.get("pipeline_mode") == "evidence":
        try:
            with output.open("x", encoding="utf-8") as handle:
                handle.write(text)
        except FileExistsError:
            if json.loads(output.read_text(encoding="utf-8-sig")) != config:
                raise FileExistsError(f"Saved config for session {session_id!r} differs; use a new session id or the unchanged original config.") from None
    else:
        output.write_text(text, encoding="utf-8")
    return output


def _session_dir_for_config(config: dict) -> Path:
    output = config.get("output") or {}
    root = Path(output.get("run_output_root") or "outputs")
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    session_id = str(output.get("session_id") or "workflow_run")
    if bool(output.get("sessionized", True)):
        return root / "sessions" / session_id
    return root


def _load_named_resume_config(args, explicit_options):
    """Read only the explicitly named session; never infer a task or regenerate it."""
    session_id = str(args.resume_session or "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", session_id):
        raise ValueError("--resume-session must be a single saved session ID")
    config_path = PROJECT_ROOT / "outputs" / "generated_configs" / (session_id + ".json")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if config.get("pipeline_mode") != "evidence":
        raise ValueError("saved session is not evidence")
    output = config.get("output") or {}
    if output.get("session_id") != session_id:
        raise ValueError("saved config session ID does not match --resume-session")
    session_dir = _session_dir_for_config(config).resolve()
    if session_dir.name != session_id:
        raise ValueError("saved config does not point to the named session directory")
    task = config.get("structured_task") or {}
    llm = config.get("llm") or {}
    expected = {"disease": task.get("disease"), "location": task.get("location"),
                "start_date": task.get("start_date"), "end_date": task.get("end_date"),
                "user_request": config.get("user_request") or task.get("user_request"),
                "session_id": session_id, "target_field": task.get("target_fields") or [],
                "provider": llm.get("provider"), "model": llm.get("model"),
                "llm_thinking": llm.get("thinking", "auto"), "llm_effort": llm.get("effort"),
                "llm_output_mode": llm.get("structured_output_method", "auto"),
                "pipeline_mode": config["pipeline_mode"]}
    for field, saved in expected.items():
        option = "--" + field.replace("_", "-")
        if option in explicit_options and getattr(args, field) != saved:
            raise ValueError(option + " conflicts with the saved session config")
    llm_enabled = any(bool(llm.get(name)) for name in
                      ("source_planning_enabled", "source_critic_enabled", "structured_extraction_enabled"))
    quick = bool(config.get("quick_test_mode") or task.get("quick_test_mode"))
    audit = (config.get("workflow") or {}).get("collection_mode") == "standard"
    for option, conflicts in (("--no-llm", llm_enabled), ("--quick-test-mode", not quick), ("--audit-mode", not audit)):
        if option in explicit_options and conflicts:
            raise ValueError(option + " conflicts with the saved session config")
    if "--output-dir" in explicit_options:
        requested = Path(args.output_dir)
        saved_root = Path(output.get("run_output_root") or "outputs")
        requested = requested if requested.is_absolute() else PROJECT_ROOT / requested
        saved_root = saved_root if saved_root.is_absolute() else PROJECT_ROOT / saved_root
        if requested.resolve() != saved_root.resolve():
            raise ValueError("--output-dir conflicts with the saved session config")
    from data_collection_workflow.session_runtime import _validate_resume_manifest
    _validate_resume_manifest(session_dir, config)
    args.provider = str(llm.get("provider") or DEFAULT_PROVIDER)
    args.model = str(llm.get("model") or DEFAULT_MODEL)
    args.no_llm = not llm_enabled
    args.pipeline_mode = config["pipeline_mode"]
    args.output_dir = None  # The configured runner must receive the original config unchanged.
    args.quick_test_mode = quick
    args.audit_mode = audit
    return config, {**task, "session_id": session_id}, config_path


def _find_available_port(preferred_port: int) -> int:
    for port in range(int(preferred_port), int(preferred_port) + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return int(preferred_port)


def _launch_live_dashboard(
    session_dir: Path,
    *,
    preferred_port: int = 8501,
) -> dict:
    """Start Streamlit dashboard for this session without blocking the workflow."""

    if importlib.util.find_spec("streamlit") is None:
        return {
            "started": False,
            "reason": "streamlit_not_installed",
            "install_command": "python -m pip install streamlit plotly nbformat nbconvert ipywidgets pypdf",
        }
    port = _find_available_port(preferred_port)
    log_dir = PROJECT_ROOT / "outputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "streamlit_live_dashboard.log"
    stderr_path = log_dir / "streamlit_live_dashboard.err.log"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "scripts" / "dashboard.py"),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--",
        "--session-dir",
        str(session_dir),
    ]
    stdout_handle = stdout_path.open("a", encoding="utf-8")
    stderr_handle = stderr_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            stdin=subprocess.DEVNULL,
        )
    except OSError as exc:
        stdout_handle.close()
        stderr_handle.close()
        return {
            "started": False,
            "reason": f"{exc.__class__.__name__}: {exc}",
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }
    stdout_handle.close()
    stderr_handle.close()
    return {
        "started": True,
        "pid": process.pid,
        "port": port,
        "url": f"http://localhost:{port}",
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def _runner_args(config_path: Path, args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(config_path),
        enable_live_fetch=False,
        disable_live_fetch=False,
        enable_all_llm=False,
        disable_all_llm=False,
        provider=None,
        model=None,
        timeout_seconds=None,
        llm_max_chunks=None,
        output_dir=args.output_dir,
        session_id=None,
        user_request=None,
        print_config_only=False,
        live_status=bool(getattr(args, "live_status", True)),
        write_run_notebook=bool(getattr(args, "write_run_notebook", True)),
        budget_amendment=getattr(args, "budget_amendment", None),
        provider_resume=getattr(args, "provider_resume", None),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively run the data collection workflow in real mode. "
            "Live search, live fetch, and LLM stages are enabled by default."
        )
    )
    parser.add_argument('--pipeline-mode', choices=[PIPELINE_MODE, 'evidence'], default=PIPELINE_MODE)
    parser.add_argument('--provider-resume', help='JSON provider/event_id/reason acknowledgement for same-version evidence --resume-session.')
    parser.add_argument('--resume-session', help='Resume this existing session with identical task/config/policy.')
    parser.add_argument('--budget-amendment', help='JSON file of explicit budget increases for same-code --resume-session.')
    parser.add_argument("--disease")
    parser.add_argument("--location")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--user-request")
    parser.add_argument("--target-field", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--session-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--provider", type=str.lower, choices=sorted(LLM_KEY_BY_PROVIDER),
                        help="LLM provider; inherits LLM_PROVIDER when omitted.")
    parser.add_argument("--model", help="Exact model ID from your account; inherits a matching LLM_MODEL when omitted.")
    parser.add_argument("--llm-thinking", choices=["auto", "adaptive", "disabled"],
                        help="Thinking mode for models that support it (default: auto).")
    parser.add_argument("--llm-effort", choices=["low", "medium", "high", "xhigh", "max"],
                        help="Reasoning effort; supported levels depend on the selected model.")
    parser.add_argument("--llm-output-mode", choices=["auto", "function_calling", "json_schema", "json_prompt"],
                        help="Structured response method (default: auto).")
    parser.add_argument("--no-llm", action="store_true", help="Internal/debug option; real user runs enable LLM by default.")
    parser.add_argument("--quick-test-mode", action="store_true", help="Use the same reduced budget controls as the Langflow visual demo.")
    parser.add_argument(
        "--audit-mode",
        action="store_true",
        help="Use the legacy full audit collection mode instead of the default direct collection mode.",
    )
    parser.add_argument("--print-config-only", action="store_true", help="Preview sanitized generated config without running.")
    parser.add_argument("--live-status", dest="live_status", action="store_true", default=True)
    parser.add_argument("--no-live-status", dest="live_status", action="store_false")
    parser.add_argument("--write-run-notebook", dest="write_run_notebook", action="store_true", default=True)
    parser.add_argument("--no-run-notebook", dest="write_run_notebook", action="store_false")
    parser.add_argument("--dashboard", dest="dashboard", action="store_true", default=True)
    parser.add_argument("--no-dashboard", dest="dashboard", action="store_false")
    parser.add_argument("--dashboard-port", type=int, default=8501)
    return parser


def _print_task_result_summary(summary: dict) -> None:
    artifacts = summary.get('artifact_paths') or {}
    report = artifacts.get('final_report_english') or artifacts.get('task_result_english')
    if report:
        label = 'final_report' if artifacts.get('final_report_english') else 'task_result_report'
        print(f"{label}: {report}")
    if artifacts.get('report_bundle'):
        print(f"report_bundle: {artifacts['report_bundle']}")
    result = summary.get('task_result_summary') or {}
    if result.get('headline'):
        print(f"task_result: {result['headline']}")


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_stdio()
    parser = build_parser()
    argument_tokens = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argument_tokens)
    from dotenv import load_dotenv

    # Load only this checkout's explicit .env; use neutral setting names.
    inherited_settings = set(os.environ)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    saved_config_path = None
    if args.resume_session:
        try:
            config, collected, saved_config_path = _load_named_resume_config(
                args, {token.split("=", 1)[0] for token in argument_tokens if token.startswith("--")})
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Cannot resume saved session: {exc}", file=sys.stderr)
            return 2
        setting_sources = {"provider": "saved session config", "model": "saved session config"}
    else:
        setting_sources = {}
        explicit_model = (args.model or "").strip()
        for option, env_name, default in (
            ("provider", "LLM_PROVIDER", DEFAULT_PROVIDER),
            ("model", "LLM_MODEL", DEFAULT_MODEL),
            ("llm_thinking", "LLM_THINKING", "auto"),
            ("llm_effort", "LLM_EFFORT", None),
            ("llm_output_mode", "LLM_STRUCTURED_OUTPUT_METHOD", "auto"),
        ):
            explicit_value = getattr(args, option)
            environment_value = "" if explicit_value else (get_env(env_name) or "").strip()
            value = explicit_value or environment_value or default
            if option in {"provider", "model"}:
                if explicit_value:
                    setting_sources[option] = f"--{option}"
                elif environment_value:
                    source = (
                        f"process environment {env_name}" if env_name in inherited_settings
                        else f"project .env ({PROJECT_ROOT / '.env'}): {env_name}"
                    )
                    setting_sources[option] = source
                else:
                    setting_sources[option] = "project default"
            setattr(args, option, value)
        from data_collection_workflow.runtime_profile import resolve_llm_selection
        args.provider, args.model = resolve_llm_selection(args.provider, explicit_model or None)
        if not args.model:
            setting_sources['model'] = 'not selected'
        if not args.no_llm and not args.print_config_only and sys.stdin.isatty():
            if setting_sources['provider'] == 'project default':
                args.provider = _prompt('LLM provider (anthropic/openai)', args.provider).strip().lower()
                setting_sources['provider'] = 'interactive selection'
                args.provider, args.model = resolve_llm_selection(args.provider, explicit_model or None)
            if not args.model:
                args.model = _prompt('Model ID available in your account').strip()
                setting_sources['model'] = 'interactive selection'
        if args.provider not in LLM_KEY_BY_PROVIDER:
            print('Choose a supported LLM provider: anthropic or openai.', file=sys.stderr)
            return 2
        try:
            collected = _collect_inputs(args)
        except ValueError as exc:
            print(f"Invalid interactive workflow input: {exc}", file=sys.stderr)
            return 2
        config = _real_run_config(
            disease=collected["disease"],
            location=collected["location"],
            start_date=collected["start_date"],
            end_date=collected["end_date"],
            target_fields=collected["target_fields"],
            session_id=collected["session_id"],
            provider=args.provider,
            model=args.model,
            output_dir=args.output_dir,
            no_llm=args.no_llm,
            user_request=collected["user_request"],
            quick_test_mode=bool(args.quick_test_mode),
            audit_mode=bool(args.audit_mode),
            llm_thinking=args.llm_thinking,
            llm_effort=args.llm_effort,
            llm_output_mode=args.llm_output_mode,
        )

        config['pipeline_mode'] = args.pipeline_mode
        if args.pipeline_mode == 'evidence':
            from data_collection_workflow.session_runtime import prepare_universal_config
            config = prepare_universal_config(config)
            from data_collection_workflow.acquisition_budget import apply_interactive_acquisition_preset
            config = apply_interactive_acquisition_preset(config, quick_test=bool(args.quick_test_mode))
    from data_collection_workflow.acquisition_budget import is_adaptive_budget
    if args.provider_resume and (not args.resume_session or config.get('pipeline_mode') != 'evidence'):
        parser.error('--provider-resume requires a same-version evidence --resume-session')
    if args.budget_amendment and (not args.resume_session or not is_adaptive_budget(config)):
        parser.error('--budget-amendment requires an adaptive evidence --resume-session')
    if args.resume_session and args.resume_session != collected['session_id']:
        parser.error('--resume-session must match --session-id')

    if args.print_config_only:
        print("data collection workflow interactive real-run preview")
        print("sanitized_config_json:")
        print(json.dumps(_sanitized_preview(config, args.provider, args.no_llm), indent=2, ensure_ascii=False))
        return 0

    if not args.no_llm and not args.model:
        print('Choose your model with --model <model-id> or LLM_MODEL in your environment or .env file.', file=sys.stderr)
        return 2

    print(f"llm_provider: {args.provider} (source: {setting_sources['provider']})")
    print(f"llm_model: {args.model} (source: {setting_sources['model']})")
    missing = _require_keys(provider=args.provider, llm_enabled=not args.no_llm)
    if missing:
        for name in missing:
            print(f"Missing required API key: {name}", file=sys.stderr)
        print(
            "Set the missing key in the environment before running a real workflow. "
            "The workflow will not fall back to fixture data for this interactive entrypoint.",
            file=sys.stderr,
        )
        return 2

    if not args.no_llm and config['pipeline_mode'] != 'evidence':
        try:
            with temporary_workflow_env(workflow_run_env_from_config(config)):
                preflight = _preflight_llm_with_trace_policy()
            print(
                "llm_preflight: "
                f"{preflight.get('status')} "
                f"{preflight.get('provider')}/{preflight.get('model')}"
            )
        except Exception as exc:  # noqa: BLE001 - fail fast before a long live run
            print(f"LLM model preflight failed: {exc}", file=sys.stderr)
            return 2

    try:
        generated_config = saved_config_path or _write_generated_config(config, collected["session_id"])
    except (OSError, ValueError) as exc:
        print(f"Cannot preserve generated run config: {exc}", file=sys.stderr)
        return 2
    session_dir = _session_dir_for_config(config)
    print("data collection workflow interactive real run")
    print(f"pipeline_mode: {config['pipeline_mode']}")
    print(f"generated_config: {generated_config}")
    print(f"session_id: {collected['session_id']}")
    print("live_search: true")
    print("live_fetch: true")
    print(f"llm_enabled: {str(not args.no_llm).lower()}")
    print(f"collection_mode: {config.get('workflow', {}).get('collection_mode')}")
    print(f"quick_test_mode: {str(bool(args.quick_test_mode)).lower()}")
    print(f"write_run_notebook: {str(args.write_run_notebook).lower()}")
    if args.dashboard:
        dashboard = _launch_live_dashboard(
            session_dir,
            preferred_port=args.dashboard_port,
        )
        if dashboard.get("started"):
            print(f"live_dashboard: {dashboard['url']}")
            print(f"dashboard_pid: {dashboard['pid']}")
        else:
            print(f"live_dashboard: not started ({dashboard.get('reason')})")
            if dashboard.get("install_command"):
                print(f"dashboard_install_command: {dashboard['install_command']}")
    else:
        print("live_dashboard: disabled")
    print("running workflow now...")
    runner = _runner_args(generated_config, args)
    runner.resume_session = args.resume_session
    runner.acquisition_settings_origin = (
        "saved session config" if args.resume_session else
        "interactive quick preset" if args.quick_test_mode else "interactive preset"
    )
    try:
        summary = run_workflow(runner)
    except llm_clients.LLMModelPreflightError as exc:
        if config['pipeline_mode'] != 'evidence':
            raise
        print(str(exc), file=sys.stderr)
        print(
            f"Selected model: {args.provider}/{args.model} "
            f"(source: {setting_sources['model']}). "
            "Rerun with --model <available-model-id> to explicitly select an available model.",
            file=sys.stderr,
        )
        return 2
    if summary is not None:
        _print_task_result_summary(summary)
    return int(summary is None)


if __name__ == "__main__":
    raise SystemExit(main())
