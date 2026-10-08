"""Run an Data Collection Workflow runtime profile and export readable artifacts."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import json
import os
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from data_collection_workflow.environment import get_env
from data_collection_workflow.evaluation_report_builder import (  # noqa: E402
    build_evaluation_report,
    build_evaluation_review_items,
    read_csv_records,
    write_csv_records,
    write_evaluation_outputs,
)
from data_collection_workflow.collection_readiness import (  # noqa: E402
    write_collection_readiness_outputs,
)
from data_collection_workflow.export import export_final_data_package, write_json  # noqa: E402
from data_collection_workflow.graph import build_graph  # noqa: E402
from data_collection_workflow.interpretive_report import (  # noqa: E402
    ENGLISH_REPORT,
    SUMMARY_JSON,
    write_interpretive_reports,
)
from data_collection_workflow.reporting.final_report_renderer import write_final_reports  # noqa: E402
from data_collection_workflow.run_events import (  # noqa: E402
    RunEventWriter,
    summarize_state_update,
)
from data_collection_workflow.workflow_notebook import write_workflow_replay_notebook  # noqa: E402
from data_collection_workflow.review_reporting import (  # noqa: E402
    write_human_review_workflow_artifacts,
)
from data_collection_workflow.workflow_visualization import (  # noqa: E402
    write_workflow_visualization_artifacts,
)
from data_collection_workflow.search_providers import search_api_key_present  # noqa: E402
from data_collection_workflow.validation_source_compatibility import (  # noqa: E402
    resolve_task_compatible_validation_records,
)
from data_collection_workflow.workflow_run_config import (  # noqa: E402
    COLLECTION_SOURCE_IDS,
    CONTEXT_SOURCE_IDS,
    DEFAULT_WORKFLOW_RUN_CONFIG_PATH,
    VALIDATION_SOURCE_IDS,
    api_key_present,
    safe_env_for_display,
    load_workflow_run_config,
    resolve_workflow_run_config_path,
    temporary_workflow_env,
    validation_records_path_from_config,
    validate_workflow_task,
    workflow_console_output_dir_from_config,
    workflow_initial_state_from_config,
    workflow_output_dir_from_config,
    workflow_run_config_with_overrides,
    workflow_run_env_from_config,
)

from build_console import (  # noqa: E402
    build_report as build_workflow_console,
    user_facing_run_status,
)

_TOP_LEVEL_REPORT_PATH = (
    _PROJECT_ROOT / "outputs" / "workflow_runs" / "latest_workflow_run_report.md"
)
_TOP_LEVEL_SUMMARY_PATH = (
    _PROJECT_ROOT / "outputs" / "workflow_runs" / "latest_workflow_run_summary.json"
)
_TOP_LEVEL_INTERPRETIVE_ENGLISH_PATH = (
    _PROJECT_ROOT
    / "outputs"
    / "workflow_runs"
    / "latest_workflow_interpretive_report.md"
)
_TOP_LEVEL_INTERPRETIVE_SUMMARY_PATH = (
    _PROJECT_ROOT
    / "outputs"
    / "workflow_runs"
    / "latest_workflow_interpretive_report_summary.json"
)
_TOP_LEVEL_FINAL_REPORT_ENGLISH_PATH = (
    _PROJECT_ROOT / "outputs" / "workflow_runs" / "latest_final_report.md"
)
_TOP_LEVEL_FINAL_REPORT_FACTS_PATH = (
    _PROJECT_ROOT / "outputs" / "workflow_runs" / "latest_final_report_facts.json"
)
_TOP_LEVEL_UNIFIED_REPORT_PATH = (
    _PROJECT_ROOT / "outputs" / "workflow_runs" / "latest_final_report.html"
)
_TOP_LEVEL_WORKFLOW_VISUALIZATION_DIR = (
    _PROJECT_ROOT / "outputs" / "workflow_visualization"
)
WORKFLOW_NODE_ORDER = [
    "task_intake_and_scope_planning",
    "disease_intelligence_builder",
    "profile_and_schema_setup",
    "executable_source_planning",
    "query_strategy_builder",
    "source_discovery",
    "source_dedup_and_registry",
    "source_screening",
    "source_critic_and_uncertainty_routing",
    "content_fetch_and_parse",
    "document_quality_check",
    "evidence_chunking_and_data_presence_flagging",
    "structured_extraction",
    "schema_validation_and_repair",
    "record_normalization",
    "record_linking",
    "cross_source_consistency_check",
    "quality_gate_routing",
    "human_review",
    "final_data_package_builder",
]


def _langsmith_project_name() -> str:
    return (get_env("LANGSMITH_PROJECT") or "data-collection-workflow-demo").strip()


def _langsmith_trace_config(session_id: str) -> dict:
    trace_uuid = uuid.uuid4()
    trace_id = str(trace_uuid)
    return {
        "run_name": f"Data Collection Workflow run {session_id}",
        "run_id": trace_uuid,
        "tags": ["data-collection-workflow", f"session:{session_id}"],
        "metadata": {
            "session_id": session_id,
            "trace_id": trace_id,
            "langsmith_project": _langsmith_project_name(),
            "langgraph_graph": "data_collection_workflow",
        },
    }


def _install_trace_env(trace_config: dict) -> dict[str, str | None]:
    metadata = trace_config.get("metadata") or {}
    updates = {
        "TRACE_SESSION_ID": str(metadata.get("session_id") or ""),
        "TRACE_ID": str(metadata.get("trace_id") or ""),
        "TRACE_RUN_NAME": str(trace_config.get("run_name") or ""),
    }
    previous = {key: os.environ.get(key) for key in updates}
    for key, value in updates.items():
        if value:
            os.environ[key] = value
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _langsmith_external_trace_enabled() -> bool:
    return (get_env("ENABLE_LANGSMITH_TRACE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _suppress_external_langsmith_env() -> dict[str, str | None]:
    keys = [
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_TRACING",
    ]
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.pop("LANGSMITH_API_KEY", None)
    os.environ.pop("LANGCHAIN_API_KEY", None)
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING"] = "false"
    return previous


def _flush_langsmith_tracers() -> None:
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
    except Exception:
        return
    try:
        wait_for_all_tracers()
    except Exception:
        return


def _preflight_llm_with_trace_policy() -> dict:
    """Apply the same explicit external-tracing policy as graph execution."""
    from data_collection_workflow.llm_clients import preflight_llm_model

    previous = (
        None if _langsmith_external_trace_enabled() else _suppress_external_langsmith_env()
    )
    try:
        return preflight_llm_model()
    finally:
        _flush_langsmith_tracers()
        if previous is not None:
            _restore_env(previous)


def _copy_text_file(source: Path | str, target: Path) -> None:
    source = Path(source)
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _write_workflow_visualization_latest_aliases(paths: dict[str, str]) -> dict[str, str]:
    aliases = {
        "latest_workflow_visualization_index": (
            paths.get("workflow_visualization_index"),
            _TOP_LEVEL_WORKFLOW_VISUALIZATION_DIR / "index.html",
        ),
        "latest_workflow_visualization_summary": (
            paths.get("workflow_visualization_summary"),
            _TOP_LEVEL_WORKFLOW_VISUALIZATION_DIR
            / "workflow_visualization_summary.json",
        ),
        "latest_evidence_flow_graph_html": (
            paths.get("evidence_flow_graph_html"),
            _TOP_LEVEL_WORKFLOW_VISUALIZATION_DIR / "evidence_flow_graph.html",
        ),
        "latest_dataset_decision_flow_html": (
            paths.get("dataset_decision_flow_html"),
            _TOP_LEVEL_WORKFLOW_VISUALIZATION_DIR / "dataset_decision_flow.html",
        ),
        "latest_human_review_workflow_html": (
            paths.get("human_review_workflow_html"),
            _TOP_LEVEL_WORKFLOW_VISUALIZATION_DIR / "human_review_workflow.html",
        ),
    }
    written: dict[str, str] = {}
    for key, (source, target) in aliases.items():
        if source:
            _copy_text_file(source, target)
            if target.exists():
                written[key] = str(target)
    return written


def _append_visualization_report_section(
    report_path: Path,
    visualization_paths: dict[str, str],
) -> None:
    if not report_path.exists():
        return
    lines = [
        "",
        "## 12. Workflow visualization artifacts",
        "",
        "- Workflow visualization index: "
        f"`{visualization_paths.get('workflow_visualization_index')}`",
        "- Workflow timeline: "
        f"`{visualization_paths.get('workflow_timeline_html')}`",
        "- Evidence flow graph: "
        f"`{visualization_paths.get('evidence_flow_graph_html')}`",
        "- Claim comparison cards: "
        f"`{visualization_paths.get('claim_comparison_cards_html')}`",
        "- Dataset decision flow: "
        f"`{visualization_paths.get('dataset_decision_flow_html')}`",
        "- Human review workflow visualization: "
        f"`{visualization_paths.get('human_review_workflow_html')}`",
        "- Visualization summary: "
        f"`{visualization_paths.get('workflow_visualization_summary')}`",
        "",
    ]
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _write_interpretive_report_outputs(
    output_dir: Path,
    *,
    write_latest_alias: bool,
) -> dict:
    """Write deterministic interpretive reports from current session artifacts."""

    paths = write_interpretive_reports(output_dir)
    manifest = {
        "interpretive_report_english": paths["english_report"],
        "interpretive_report_summary": paths["summary_json"],
    }
    if write_latest_alias:
        _TOP_LEVEL_INTERPRETIVE_ENGLISH_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        _TOP_LEVEL_INTERPRETIVE_ENGLISH_PATH.write_text(
            Path(paths["english_report"]).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        _TOP_LEVEL_INTERPRETIVE_SUMMARY_PATH.write_text(
            Path(paths["summary_json"]).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        manifest.update(
            {
                "stable_interpretive_report_english": str(
                    _TOP_LEVEL_INTERPRETIVE_ENGLISH_PATH
                ),
                "stable_interpretive_report_summary": str(
                    _TOP_LEVEL_INTERPRETIVE_SUMMARY_PATH
                ),
            }
        )
    return manifest


def _write_final_report_outputs(
    output_dir: Path,
    *,
    write_latest_alias: bool,
) -> dict:
    """Write deterministic final reports from current session artifacts."""

    paths = write_final_reports(output_dir)
    manifest = {
        "final_report_english": paths["english_report"],
        "final_report_facts": paths["facts_json"],
        "final_report_diagnostics": paths["diagnostics_json"],
    }
    if write_latest_alias:
        _TOP_LEVEL_FINAL_REPORT_ENGLISH_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        _TOP_LEVEL_FINAL_REPORT_ENGLISH_PATH.write_text(
            Path(paths["english_report"]).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        _TOP_LEVEL_FINAL_REPORT_FACTS_PATH.write_text(
            Path(paths["facts_json"]).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        manifest.update(
            {
                "stable_final_report_english": str(
                    _TOP_LEVEL_FINAL_REPORT_ENGLISH_PATH
                ),
                "stable_final_report_facts": str(_TOP_LEVEL_FINAL_REPORT_FACTS_PATH),
            }
        )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a configured Data Collection Workflow profile with live source fetch, LLM "
            "source planning, source identity, source critic, and structured extraction."
        )
    )
    parser.add_argument('--provider-resume', help='JSON provider/event_id/reason acknowledgement; requires same-version evidence --resume-session.')
    parser.add_argument('--resume-session', help='Explicit current session ID to resume; task/config/policy fingerprints must match.')
    parser.add_argument('--budget-amendment', help='JSON file with amendment_id, increases and reason; requires adaptive same-code --resume-session.')
    parser.add_argument(
        "--config",
        default=str(DEFAULT_WORKFLOW_RUN_CONFIG_PATH),
        help="Path to the workflow runtime JSON config file.",
    )
    parser.add_argument(
        "--enable-live-fetch",
        action="store_true",
        help="Override config and enable live HTTP fetch for this run.",
    )
    parser.add_argument(
        "--disable-live-fetch",
        action="store_true",
        help="Override config and disable live HTTP fetch for this run.",
    )
    parser.add_argument(
        "--enable-all-llm",
        action="store_true",
        help="Override config and enable all configured LLM stages for this run.",
    )
    parser.add_argument(
        "--disable-all-llm",
        action="store_true",
        help="Override config and disable all configured LLM stages for this run.",
    )
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--llm-max-chunks", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional fixed run session id. Defaults to a UTC timestamp.",
    )
    parser.add_argument("--user-request", default=None)
    parser.add_argument(
        "--print-config-only",
        action="store_true",
        help="Print sanitized configuration without running live fetch or LLM calls.",
    )
    parser.add_argument(
        "--live-status",
        dest="live_status",
        action="store_true",
        default=True,
        help="Show an interactive Rich terminal status panel while the graph runs.",
    )
    parser.add_argument(
        "--no-live-status",
        dest="live_status",
        action="store_false",
        help="Disable terminal live status; runtime event files are still written.",
    )
    parser.add_argument(
        "--write-run-notebook",
        action="store_true",
        help="Write workflow_replay_notebook.ipynb after the run completes.",
    )
    parser.add_argument(
        "--case-study-real-mode",
        action="store_true",
        help=(
            "Require live search, live fetch, non-fixture content, and a real "
            "collection funnel in case-study readiness diagnostics."
        ),
    )
    return parser


def _override_value(enable: bool, disable: bool) -> bool | None:
    if enable and disable:
        raise ValueError("Enable/allow and disable flags cannot both be set.")
    if enable:
        return True
    if disable:
        return False
    return None


def _config_with_cli_overrides(args: argparse.Namespace) -> tuple[Path, dict]:
    config_path = resolve_workflow_run_config_path(args.config)
    config = load_workflow_run_config(config_path)
    updated = workflow_run_config_with_overrides(
        config,
        live_fetch=_override_value(args.enable_live_fetch, args.disable_live_fetch),
        all_llm=_override_value(args.enable_all_llm, args.disable_all_llm),
        provider=args.provider,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        llm_max_chunks=args.llm_max_chunks,
        output_dir=args.output_dir,
        session_id=args.session_id,
        user_request=args.user_request,
    )
    if _case_study_real_mode_enabled(args):
        updated = _apply_case_study_real_mode_config(updated)
    return config_path, updated


def _case_study_real_mode_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "case_study_real_mode", False)) or get_env("RUN_MODE") == "case_study_real"


def _apply_case_study_real_mode_config(config: dict) -> dict:
    updated = dict(config)
    updated["run_mode"] = "case_study_real"
    workflow = dict(updated.get("workflow") or {})
    workflow["use_fixture_documents"] = False
    updated["workflow"] = workflow
    live_web = dict(updated.get("live_web") or {})
    live_web["enabled"] = True
    updated["live_web"] = live_web
    source_search = dict(updated.get("source_search") or {})
    source_search["enabled"] = True
    source_search["mode"] = "live"
    source_search["live_search_enabled"] = True
    updated["source_search"] = source_search
    content_fetch = dict(updated.get("content_fetch") or {})
    content_fetch["fetch_search_derived_sources"] = True
    content_fetch["content_fixture_map_path"] = None
    external_fetch = dict(content_fetch.get("external_fetch") or {})
    external_fetch["enabled"] = True
    content_fetch["external_fetch"] = external_fetch
    updated["content_fetch"] = content_fetch
    return updated


def _config_provider_model(config: dict) -> tuple[str, str]:
    llm = config.get("llm") or {}
    return str(llm.get("provider") or ""), str(llm.get("model") or "")


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


def _format_counts(counts: dict | None) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _preview(value, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "none"
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _cell(value) -> str:
    text = str(value if value is not None else "").replace("|", "/")
    return _preview(text, 140)


def _case_value(record: dict) -> str:
    for field in (
        "cases_confirmed",
        "cases_unspecified",
        "cases_probable",
        "cases_suspected",
    ):
        value = record.get(field)
        if value not in (None, "", []):
            return str(value)
    return ""


def _count_by(records: list[dict], field: str) -> dict:
    return dict(Counter(str(row.get(field) or "unknown") for row in records))


def _source_role(entry: dict) -> str:
    source_id = entry.get("source_id")
    flags = set(entry.get("routing_flags") or [])
    if (
        source_id in VALIDATION_SOURCE_IDS
        or entry.get("source_role") == "validation_reserved"
        or "validation_reserved" in flags
    ):
        return "validation_reserved"
    if (
        source_id in CONTEXT_SOURCE_IDS
        or entry.get("source_role") == "context_source"
        or "context_only" in flags
    ):
        return "context_only"
    if source_id in COLLECTION_SOURCE_IDS:
        return "collection"
    return "other"


def _source_split_summary(registry: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {
        "collection": [],
        "context_only": [],
        "validation_reserved": [],
        "other": [],
    }
    for entry in registry:
        grouped.setdefault(_source_role(entry), []).append(entry)
    return {
        role: [
            {
                "source_id": entry.get("source_id"),
                "title": entry.get("title"),
                "publisher": entry.get("publisher"),
                "canonical_url": entry.get("canonical_url"),
                "final_screening_decision": entry.get("final_screening_decision"),
                "ready_for_content_fetch": entry.get("ready_for_content_fetch"),
                "credibility_score": entry.get("credibility_score"),
                "credibility_level": entry.get("credibility_level"),
            }
            for entry in entries
        ]
        for role, entries in grouped.items()
    }


def _live_fetch_summary(result: dict) -> dict:
    documents = list(result.get("documents") or [])
    fetch_summary = result.get("content_fetch_summary") or {}
    quality_summary = result.get("document_quality_summary") or {}
    rows = []
    for doc in documents:
        rows.append(
            {
                "source_id": doc.get("source_id"),
                "url": doc.get("url"),
                "canonical_url": doc.get("canonical_url"),
                "title": doc.get("title"),
                "discovery_method": doc.get("discovery_method"),
                "source_role_final": doc.get("source_role_final"),
                "credibility_score": doc.get("credibility_score"),
                "credibility_level": doc.get("credibility_level"),
                "fetch_status": doc.get("fetch_status"),
                "fetch_provider": doc.get("fetch_provider"),
                "provider_attempts": doc.get("provider_attempts") or [],
                "http_status_code": doc.get("http_status_code"),
                "quality_status": doc.get("quality_status"),
                "parse_status": doc.get("parse_status"),
                "parser_used": doc.get("parser_used"),
                "content_type": doc.get("content_type"),
                "clean_text_char_count": len(doc.get("clean_text") or ""),
                "table_count": doc.get("table_count") or len(doc.get("tables") or []),
                "is_live_fetched": bool(doc.get("is_live_fetched")),
            }
        )
    return {
        "live_fetch_enabled": bool(fetch_summary.get("live_fetch_enabled")),
        "document_count": len(documents),
        "document_source_ids": sorted({row["source_id"] for row in rows if row["source_id"]}),
        "fetch_status_counts": fetch_summary.get("fetch_status_counts") or {},
        "external_fetch_enabled": fetch_summary.get("external_fetch_enabled"),
        "external_fetch_provider_order": fetch_summary.get(
            "external_fetch_provider_order"
        )
        or [],
        "fetch_provider_counts": fetch_summary.get("fetch_provider_counts") or {},
        "external_fetch_failure_counts": fetch_summary.get(
            "external_fetch_failure_counts"
        )
        or {},
        "selected_fetch_bucket_counts": fetch_summary.get(
            "selected_fetch_bucket_counts"
        )
        or {},
        "parser_status_counts": fetch_summary.get("parser_status_counts") or {},
        "parser_used_counts": fetch_summary.get("parser_used_counts") or {},
        "selected_search_derived_fetch_count": fetch_summary.get(
            "selected_search_derived_fetch_count", 0
        ),
        "skipped_search_derived_fetch_disabled_count": fetch_summary.get(
            "skipped_search_derived_fetch_disabled_count", 0
        ),
        "skipped_search_derived_by_reason_counts": fetch_summary.get(
            "skipped_search_derived_by_reason_counts"
        )
        or {},
        "quality_status_counts": quality_summary.get("quality_status_counts") or {},
        "skipped_validation_reserved_source_ids": fetch_summary.get(
            "skipped_validation_reserved_source_ids"
        )
        or [],
        "skipped_not_in_allowlist_count": fetch_summary.get(
            "skipped_not_in_allowlist_count", 0
        ),
        "documents": rows,
    }


def _llm_stage_summary(result: dict, provider: str, model: str) -> dict:
    disease_intelligence = result.get("disease_intelligence_summary") or {}
    executable_plan = result.get("executable_source_plan_summary") or {}
    planning = result.get("source_planning_agent_summary") or {}
    critic = result.get("source_critic_summary") or {}
    credibility = result.get("source_credibility_summary") or {}
    identity = result.get("source_identity_summary") or {}
    extraction = result.get("structured_extraction_summary") or {}
    llm_extraction = result.get("llm_extraction_summary") or {}
    return {
        "provider": provider,
        "model": model,
        "api_key_present": api_key_present(provider),
        "disease_intelligence": {
            "generation_method": disease_intelligence.get("generation_method"),
            "disease_standard_name": disease_intelligence.get(
                "disease_standard_name"
            ),
            "query_term_count": disease_intelligence.get("query_term_count", 0),
            "warnings": disease_intelligence.get("warnings") or [],
            "llm_call_succeeded": (
                disease_intelligence.get("generation_method") == "llm_generated"
            ),
        },
        "source_planning": {
            "enabled": bool(planning.get("llm_source_planning_enabled")),
            "status": planning.get("status"),
            "generation_method": executable_plan.get("generation_method")
            or planning.get("generation_method"),
            "execution_status": executable_plan.get("execution_status")
            or planning.get("execution_status"),
            "planned_query_count": executable_plan.get("planned_query_count", 0),
            "planned_source_category_count": executable_plan.get(
                "planned_source_category_count", 0
            ),
            "provider_channel_counts": executable_plan.get(
                "provider_channel_counts"
            )
            or {},
            "agent_query_count": planning.get("agent_query_count", 0),
            "agent_query_added_count": planning.get("agent_query_added_count", 0),
            "candidate_hint_count": planning.get("agent_candidate_hint_count", 0),
            "warnings": executable_plan.get("warnings")
            or planning.get("warnings")
            or [],
            "failure_type": planning.get("failure_type"),
            "failure_message": planning.get("failure_message"),
        },
        "source_critic": {
            "enabled": bool(critic.get("llm_source_critic_enabled")),
            "attempted_source_count": critic.get(
                "attempted_source_count",
                critic.get("llm_source_critic_attempted_source_count", 0),
            ),
            "assessed_source_count": critic.get(
                "assessed_source_count", critic.get("llm_assessed_source_count", 0)
            ),
            "skipped_source_count": critic.get(
                "skipped_source_count", critic.get("llm_skipped_source_count", 0)
            ),
            "blocked_fetch_count": critic.get("blocked_fetch_count", 0),
            "needs_review_count": critic.get("needs_review_count", 0),
            "allowed_fetch_count": critic.get("allowed_fetch_count", 0),
            "context_only_count": critic.get("context_only_count", 0),
            "decision_counts": critic.get("decision_counts") or {},
            "fetch_recommendation_counts": (
                critic.get("fetch_recommendation_counts") or {}
            ),
            "risk_flag_counts": critic.get("risk_flag_counts") or {},
            "selection": critic.get("source_critic_selection_summary") or {},
            "allowlist_enabled": critic.get(
                "llm_source_critic_allowlist_enabled", False
            ),
            "max_sources": critic.get("llm_source_critic_max_sources"),
            "review_blocks_fetch": critic.get(
                "llm_source_critic_review_blocks_fetch"
            ),
            "failure_count": critic.get("llm_source_critic_failure_count", 0),
            "semantic_leakage_count": critic.get("llm_semantic_leakage_count", 0),
            "human_review_recommended_count": critic.get(
                "llm_human_review_recommended_count", 0
            ),
        },
        "source_credibility": {
            "enabled": bool(credibility.get("llm_enabled")),
            "assessed_source_count": credibility.get("assessed_source_count", 0),
            "role_counts": credibility.get("role_counts") or {},
            "risk_flag_counts": credibility.get("risk_flag_counts") or {},
            "llm_assessed_count": credibility.get("llm_assessed_count", 0),
            "llm_failure_count": credibility.get("llm_failure_count", 0),
            "needs_review_count": credibility.get("needs_review_count", 0),
        },
        "source_identity": {
            "enabled": (
                identity.get("llm_identity_assessed_count", 0) > 0
                or bool(identity.get("identity_assessed_count", 0))
            ),
            "identity_assessed_count": identity.get("identity_assessed_count", 0),
            "llm_identity_assessed_count": identity.get(
                "llm_identity_assessed_count", 0
            ),
            "post_fetch_identity_assessed_count": identity.get(
                "post_fetch_identity_assessed_count", 0
            ),
            "unknown_publisher_count": identity.get("unknown_publisher_count", 0),
            "source_type_counts": identity.get("source_type_counts") or {},
            "claim_support_role_counts": identity.get(
                "claim_support_role_counts"
            )
            or {},
            "recommended_fetch_use_counts": identity.get(
                "recommended_fetch_use_counts"
            )
            or {},
            "warning_counts": identity.get("warning_counts") or {},
        },
        "structured_extraction": {
            "enabled": bool(extraction.get("llm_enabled")),
            "mode": extraction.get("extraction_mode"),
            "eligible_chunk_count": extraction.get("llm_eligible_chunk_count", 0),
            "call_count": extraction.get("llm_call_count", 0),
            "success_count": extraction.get("llm_success_count", 0),
            "error_count": extraction.get("llm_error_count", 0),
            "fallback_count": extraction.get("llm_fallback_count", 0),
            "raw_record_count": extraction.get("raw_record_count", 0),
            "max_chunks": extraction.get("llm_max_chunks"),
            "error_messages": llm_extraction.get("llm_error_messages") or [],
        },
    }


def _write_validation_outputs(
    output_dir: Path,
    validation_records: list[dict],
    registry: list[dict],
    *,
    inactive_validation_records: list[dict] | None = None,
    validation_source_compatibility_summary: dict | None = None,
    raw_validation_records: list[dict] | None = None,
) -> dict:
    validation_dir = output_dir / "validation"
    all_validation_records = (
        list(validation_records)
        + list(inactive_validation_records or [])
        + list(raw_validation_records or [])
    )
    validation_fieldnames = sorted(
        {key for record in all_validation_records for key in record.keys()}
    )
    write_csv_records(
        validation_dir / "ground_truth_records.csv",
        validation_records,
        fieldnames=validation_fieldnames or None,
    )
    write_csv_records(
        validation_dir / "inactive_validation_records.csv",
        list(inactive_validation_records or []),
        fieldnames=validation_fieldnames or None,
    )
    write_json(
        list(inactive_validation_records or []),
        validation_dir / "inactive_validation_records.json",
    )
    write_json(
        validation_source_compatibility_summary or {},
        validation_dir / "validation_source_compatibility_summary.json",
    )
    validation_ids = {
        row.get("source_id") for row in validation_records if row.get("source_id")
    }
    validation_registry = [
        entry for entry in registry if entry.get("source_id") in validation_ids
    ]
    write_json(validation_registry, validation_dir / "validation_source_registry.json")
    return {
        "ground_truth_records_csv": str(validation_dir / "ground_truth_records.csv"),
        "validation_source_registry_json": str(
            validation_dir / "validation_source_registry.json"
        ),
        "validation_source_registry_count": len(validation_registry),
        "active_validation_record_count": len(validation_records),
        "inactive_validation_record_count": len(inactive_validation_records or []),
        "raw_validation_record_count": len(raw_validation_records or []),
        "validation_mode": (validation_source_compatibility_summary or {}).get(
            "validation_mode"
        ),
        "validation_records_source": (
            validation_source_compatibility_summary or {}
        ).get("validation_records_source"),
        "validation_source_compatibility_status": (
            validation_source_compatibility_summary or {}
        ).get("compatibility_status"),
        "inactive_validation_records_csv": str(
            validation_dir / "inactive_validation_records.csv"
        ),
        "inactive_validation_records_json": str(
            validation_dir / "inactive_validation_records.json"
        ),
        "validation_source_compatibility_summary_json": str(
            validation_dir / "validation_source_compatibility_summary.json"
        ),
    }


def _append_stage11_report_section(
    lines: list[str],
    *,
    anomaly_results: list[dict],
    anomaly_summary: dict,
    human_review_application_summary: dict,
    applied_decisions: list[dict],
    rejected_decisions: list[dict],
    audit_trail: list[dict],
    final_dataset_post_review: list[dict],
    records_excluded_by_review: list[dict],
) -> None:
    lines.extend(
        [
            "",
            "## 9.5 Anomaly detection and review application",
            "",
            f"- Anomaly result count: `{len(anomaly_results)}`",
            f"- Anomaly severity counts: `{_format_counts(anomaly_summary.get('severity_counts'))}`",
            f"- Anomaly needs-human-review count: `{anomaly_summary.get('needs_human_review_count', 0)}`",
            f"- Decisions provided: `{human_review_application_summary.get('decisions_provided_count', 0)}`",
            f"- Decisions applied: `{human_review_application_summary.get('decisions_applied_count', 0)}`",
            f"- Decisions rejected: `{human_review_application_summary.get('decisions_rejected_count', 0)}`",
            f"- Audit entries: `{len(audit_trail)}`",
            f"- Final dataset post-review count: `{len(final_dataset_post_review)}`",
            f"- Records excluded by review: `{len(records_excluded_by_review)}`",
            "",
            "| Anomaly ID | Type | Severity | Target | Reason |",
            "|---|---|---|---|---|",
        ]
    )
    for anomaly in anomaly_results[:12]:
        target = (
            anomaly.get("record_id")
            or anomaly.get("event_cluster_id")
            or anomaly.get("validation_result_id")
            or anomaly.get("source_id")
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(anomaly.get("anomaly_id")),
                    _cell(anomaly.get("anomaly_type")),
                    _cell(anomaly.get("severity")),
                    _cell(target),
                    _cell(anomaly.get("reason")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "| Applied decision | Type | Target | Audit IDs | Reason |",
            "|---|---|---|---|---|",
        ]
    )
    for decision in applied_decisions[:12]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(decision.get("decision_id")),
                    _cell(decision.get("decision_type")),
                    _cell(", ".join(decision.get("target_ids") or [])),
                    _cell(", ".join(decision.get("audit_ids") or [])),
                    _cell(decision.get("reason")),
                ]
            )
            + " |"
        )
    if rejected_decisions:
        lines.extend(
            [
                "",
                "| Rejected decision | Type | Target | Rejection reason |",
                "|---|---|---|---|",
            ]
        )
        for decision in rejected_decisions[:12]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(decision.get("decision_id")),
                        _cell(decision.get("decision_type")),
                        _cell(", ".join(decision.get("target_ids") or [])),
                        _cell(decision.get("rejection_reason")),
                    ]
                )
                + " |"
            )


def _write_report(
    path: Path,
    *,
    user_request: str,
    provider: str,
    model: str,
    output_dir: Path,
    result: dict,
    collection_manifest: dict,
    validation_manifest: dict,
    evaluation_outputs: dict,
    evaluation_rows: list[dict],
    evaluation_summary: dict,
    source_split: dict,
    live_summary: dict,
    llm_summary: dict,
) -> str:
    trace = list(result.get("collection_trace") or [])
    registry = list(result.get("source_registry") or [])
    records = list(result.get("normalized_records") or [])
    review_items = list(result.get("human_review_queue") or [])
    final_package = result.get("final_data_package") or {}
    metadata = final_package.get("package_metadata") or {}
    anomaly_results = list(
        result.get("anomaly_results") or final_package.get("anomaly_results") or []
    )
    anomaly_summary = (
        result.get("anomaly_summary") or final_package.get("anomaly_summary") or {}
    )
    human_review_application_summary = (
        result.get("human_review_application_summary")
        or final_package.get("human_review_application_summary")
        or {}
    )
    applied_decisions = list(
        result.get("applied_human_review_decisions")
        or final_package.get("applied_human_review_decisions")
        or []
    )
    rejected_decisions = list(
        result.get("rejected_human_review_decisions")
        or final_package.get("rejected_human_review_decisions")
        or []
    )
    audit_trail = list(
        result.get("human_review_audit_trail")
        or final_package.get("human_review_audit_trail")
        or []
    )
    final_dataset_post_review = list(
        result.get("final_dataset_post_review")
        or final_package.get("final_dataset_post_review")
        or []
    )
    records_excluded_by_review = list(
        result.get("records_excluded_by_human_review")
        or final_package.get("records_excluded_by_human_review")
        or []
    )
    route = result.get("current_route")
    source_search = result.get("source_search_execution_summary") or {}
    iterative_source_discovery = (
        result.get("iterative_source_discovery_summary") or {}
    )
    source_discovery = result.get("source_discovery_summary") or {}
    source_credibility = result.get("source_credibility_summary") or {}
    source_identity = result.get("source_identity_summary") or {}
    disease_relevance = result.get("disease_relevance_summary") or {}
    run_quality = result.get("run_quality_summary") or final_package.get(
        "run_quality_summary"
    ) or {}
    final_dataset_quality = result.get(
        "final_dataset_quality_summary"
    ) or final_package.get("final_dataset_quality_summary") or {}
    claims = list(result.get("claims") or final_package.get("claims") or [])
    claim_comparisons = list(
        result.get("claim_comparisons")
        or final_package.get("claim_comparisons")
        or []
    )
    corroborated_events = list(
        result.get("corroborated_events")
        or final_package.get("corroborated_events")
        or []
    )
    corroboration_summary = (
        result.get("corroboration_summary")
        or final_package.get("corroboration_summary")
        or {}
    )
    final_dataset_pre_quality_gate = list(
        result.get("final_dataset_pre_quality_gate")
        or final_package.get("final_dataset_pre_quality_gate")
        or []
    )
    quarantined_records = list(
        result.get("quarantined_records")
        or final_package.get("quarantined_records")
        or []
    )
    pending_review_records = list(
        result.get("pending_review_records")
        or final_package.get("pending_review_records")
        or []
    )
    non_primary_observations = list(
        result.get("non_primary_observations")
        or final_package.get("non_primary_observations")
        or []
    )
    final_case_dataset = list(final_package.get("final_case_dataset") or [])
    zero_case_statements = list(final_package.get("zero_case_statements") or [])
    exposure_monitoring_records = list(
        final_package.get("exposure_monitoring_records") or []
    )
    surveillance_summary_records = list(
        final_package.get("surveillance_summary_records") or []
    )
    outbreak_summary_records = list(final_package.get("outbreak_summary_records") or [])
    context_records = list(final_package.get("context_records") or [])
    unclassified_observation_records = list(
        final_package.get("unclassified_observation_records") or []
    )
    observation_type_dataset_summary = (
        final_package.get("observation_type_dataset_summary") or {}
    )
    accepted_records = list(final_package.get("final_dataset") or [])
    accepted_record_count = len(accepted_records)
    pre_quality_record_count = len(final_dataset_pre_quality_gate)
    quarantined_record_count = len(quarantined_records)
    pending_review_record_count = len(pending_review_records)
    non_primary_observation_count = len(non_primary_observations)
    post_review_record_count = len(final_dataset_post_review)
    run_status_text = user_facing_run_status(run_quality)
    recommended_user_message = run_quality.get("recommended_user_message")
    validation_compatibility_status = validation_manifest.get(
        "validation_source_compatibility_status"
    )
    validation_mode = (
        validation_manifest.get("validation_mode")
        or (result.get("validation_source_compatibility_summary") or {}).get(
            "validation_mode"
        )
        or "live_cross_source"
    )
    validation_limited = bool(
        run_quality.get("validation_limited")
        or run_quality.get("no_compatible_validation_source")
        or validation_compatibility_status
        in {
            "incompatible_validation_source_disabled",
            "no_task_compatible_validation_source",
            "missing_validation_source",
        }
    )

    node_lines = [
        f"{idx}. `{event.get('node_name')}` - {_preview(event.get('message'), 120)}"
        for idx, event in enumerate(trace, 1)
    ]
    if not node_lines:
        node_lines = ["- no trace events"]

    lines = [
        "# Data Collection Workflow Run Report",
        "",
        "## 1. Input task",
        "",
        user_request,
        "",
        "## 2. Run configuration",
        "",
        f"- Live webpage fetch: `{live_summary.get('live_fetch_enabled', False)}`",
        f"- Fixture documents: `{bool((result.get('content_fetch_summary') or {}).get('fixture_documents_enabled'))}`",
        f"- Provider: `{provider}`",
        f"- Model: `{model}`",
        f"- API key present: `{api_key_present(provider)}`",
        f"- LLM source planning: `{llm_summary['source_planning']['enabled']}`",
        f"- LLM source critic: `{llm_summary['source_critic']['enabled']}`",
        f"- LLM source identity assessed sources: `{llm_summary['source_identity'].get('llm_identity_assessed_count')}`",
        f"- LLM structured extraction: `{llm_summary['structured_extraction']['enabled']}`",
        f"- Source search mode: `{source_search.get('search_mode') or 'disabled'}`",
        f"- Source search provider: `{source_search.get('search_provider') or 'n/a'}`",
        f"- Source search executed queries: `{source_search.get('executed_query_count', 0)}`",
        f"- Search-derived source candidates: `{source_search.get('candidate_from_search_count', 0)}`",
        f"- Iterative source discovery: `{iterative_source_discovery.get('iterative_source_discovery_enabled', False)}`",
        f"- Iterative search iterations: `{iterative_source_discovery.get('search_iteration_count', 0)}`",
        f"- Iterative stop decision: `{iterative_source_discovery.get('stop_decision') or 'n/a'}`",
        f"- Iterative stop reason: `{iterative_source_discovery.get('stop_reason') or 'n/a'}`",
        f"- Source credibility assessed sources: `{source_credibility.get('assessed_source_count', 0)}`",
        f"- Source credibility role counts: `{source_credibility.get('role_counts') or {}}`",
        f"- Source identity assessed sources: `{source_identity.get('identity_assessed_count', 0)}`",
        f"- Source identity type counts: `{source_identity.get('source_type_counts') or {}}`",
        f"- Source identity warning counts: `{source_identity.get('warning_counts') or {}}`",
        f"- Source discovery method: `{source_discovery.get('discovery_method')}`",
        f"- Disease relevance target: `{disease_relevance.get('target_disease') or 'n/a'}`",
        f"- Disease relevance source status counts: `{disease_relevance.get('source_status_counts') or {}}`",
        f"- Disease relevance chunk status counts: `{disease_relevance.get('chunk_status_counts') or {}}`",
        f"- Disease relevance record status counts: `{disease_relevance.get('record_compatibility_status_counts') or {}}`",
        f"- Rejected incompatible record count: `{disease_relevance.get('rejected_incompatible_record_count', 0)}`",
        f"- Final route: `{route}`",
        "",
        "## 2.5 Run quality status",
        "",
        f"- User-facing run status: `{run_status_text}`",
        "- Technical execution status: `completed`",
        f"- Run quality status: `{run_quality.get('run_quality_status') or 'n/a'}`",
        f"- Final dataset mode: `{run_quality.get('final_dataset_mode') or 'n/a'}`",
        f"- Accepted final dataset count: `{accepted_record_count}`",
        f"- Final case dataset count: `{len(final_case_dataset)}`",
        f"- Zero-case statement count: `{len(zero_case_statements)}`",
        f"- Exposure-monitoring record count: `{len(exposure_monitoring_records)}`",
        f"- Surveillance summary record count: `{len(surveillance_summary_records)}`",
        f"- Outbreak summary record count: `{len(outbreak_summary_records)}`",
        f"- Context record count: `{len(context_records)}`",
        f"- Unclassified observation count: `{len(unclassified_observation_records)}`",
        f"- Observation dataset view counts: `{observation_type_dataset_summary.get('dataset_view_counts') or {}}`",
        f"- Pre-quality-gate record count: `{pre_quality_record_count}`",
        f"- Quarantined record count: `{quarantined_record_count}`",
        f"- Pending review record count: `{pending_review_record_count}`",
        f"- Non-primary observation count: `{non_primary_observation_count}`",
        f"- Final dataset post-review count: `{post_review_record_count}`",
        f"- Primary-case dataset status: `{run_quality.get('primary_case_dataset_status') or 'n/a'}`",
        f"- Recommended primary dataset message: `{run_quality.get('recommended_primary_dataset_message') or 'n/a'}`",
        f"- Primary-case eligible accepted count: `{run_quality.get('primary_case_dataset_eligible_count', 0)}`",
        f"- Corroborated primary case event count: `{run_quality.get('corroborated_primary_case_event_count', 0)}`",
        f"- Recommended user message: `{recommended_user_message or 'n/a'}`",
        "",
        *(
            [
                (
                    "workflow technically completed, but no quality-gated accepted "
                    "records were produced."
                ),
                "",
            ]
            if accepted_record_count == 0
            else [
                "Workflow technically completed and produced quality-gated accepted records.",
                "",
            ]
        ),
        *(
            [
                (
                    "Workflow technically completed, but no primary case dataset "
                    "records were accepted. Non-primary observations were preserved "
                    "separately and should not be read as final epidemiological case data."
                ),
                "",
            ]
            if run_quality.get("no_primary_case_dataset_records")
            else []
        ),
        *(
            [
                (
                    "Live cross-source validation was limited because this run "
                    "did not find a task-compatible validation source."
                ),
                (
                    "This does not prove absence of cases; it means the live "
                    "search/fetch set did not include enough independent "
                    "validation evidence."
                ),
                "",
            ]
            if validation_limited and validation_mode == "live_cross_source"
            else []
        ),
        *(
            [
                (
                    "Held-out validation was limited because no task-compatible "
                    "validation source was available."
                ),
                "",
            ]
            if validation_limited and validation_mode != "live_cross_source"
            else []
        ),
        "## 3. Workflow execution",
        "",
        (
            "Graph is executed as a mostly serial workflow. The only conditional "
            "branch is after `quality_gate_routing`: the run enters "
            "`human_review` when quality or validation checks require review, "
            "then still builds the final data package. In this run, the graph "
            "route is shown above; validation comparison review flags are "
            "reported separately in Section 8."
        ),
        "",
        *node_lines,
        "",
        "## 4. Source roles",
        "",
        "| Role | Source ID | Publisher / title | Credibility | Final decision | Fetch ready |",
        "|---|---|---|---|---|---|",
    ]
    for role in ("collection", "context_only", "validation_reserved", "other"):
        for entry in source_split.get(role) or []:
            title = f"{entry.get('publisher') or ''} / {entry.get('title') or ''}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(role),
                        _cell(entry.get("source_id")),
                        _cell(title),
                        _cell(
                            f"{entry.get('credibility_level') or 'n/a'}"
                            f" ({entry.get('credibility_score') or 'n/a'})"
                        ),
                        _cell(entry.get("final_screening_decision")),
                        _cell(entry.get("ready_for_content_fetch")),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## 5. Document acquisition results",
            "",
            f"- Documents fetched/parsed: `{live_summary.get('document_count', 0)}`",
            f"- Search-derived sources selected for fetch: `{live_summary.get('selected_search_derived_fetch_count', 0)}`",
            f"- Search-derived skipped by reason: `{live_summary.get('skipped_search_derived_by_reason_counts') or {}}`",
            f"- Fetch status counts: `{_format_counts(live_summary.get('fetch_status_counts'))}`",
            f"- External fetch enabled: `{live_summary.get('external_fetch_enabled')}`",
            f"- Fetch provider counts: `{_format_counts(live_summary.get('fetch_provider_counts'))}`",
            f"- External fetch failure counts: `{_format_counts(live_summary.get('external_fetch_failure_counts'))}`",
            f"- Selected fetch bucket counts: `{_format_counts(live_summary.get('selected_fetch_bucket_counts'))}`",
            f"- Parser status counts: `{_format_counts(live_summary.get('parser_status_counts'))}`",
            f"- Parser used counts: `{_format_counts(live_summary.get('parser_used_counts'))}`",
            f"- Quality status counts: `{_format_counts(live_summary.get('quality_status_counts'))}`",
            f"- Validation-reserved skipped from fetch: `{live_summary.get('skipped_validation_reserved_source_ids')}`",
            "",
            "| Source ID | Provider | Fetch status | HTTP | Parse | Parser | Quality | Text chars | Tables |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in live_summary.get("documents") or []:
        lines.append(
            "| "
            + " | ".join(
                    [
                        _cell(row.get("source_id")),
                        _cell(row.get("fetch_provider")),
                        _cell(row.get("fetch_status")),
                        _cell(row.get("http_status_code")),
                        _cell(row.get("parse_status")),
                        _cell(row.get("parser_used")),
                        _cell(row.get("quality_status")),
                        _cell(row.get("clean_text_char_count")),
                        _cell(row.get("table_count")),
                    ]
                )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 6. Model call results",
            "",
            "### 6.1 LLM Source Planning",
            "",
            f"- Status: `{llm_summary['source_planning'].get('status')}`",
            f"- Plan generation method: `{llm_summary['source_planning'].get('generation_method')}`",
            f"- Plan execution status: `{llm_summary['source_planning'].get('execution_status')}`",
            f"- Planned query count: `{llm_summary['source_planning'].get('planned_query_count')}`",
            f"- Planned source category count: `{llm_summary['source_planning'].get('planned_source_category_count')}`",
            f"- Provider channel counts: `{_format_counts(llm_summary['source_planning'].get('provider_channel_counts'))}`",
            f"- Agent query count: `{llm_summary['source_planning'].get('agent_query_count')}`",
            f"- Agent query added count: `{llm_summary['source_planning'].get('agent_query_added_count')}`",
            f"- Candidate hint count: `{llm_summary['source_planning'].get('candidate_hint_count')}`",
            "",
            "### 6.1.5 LLM Iterative Source Discovery",
            "",
            f"- Enabled: `{iterative_source_discovery.get('iterative_source_discovery_enabled', False)}`",
            f"- LLM iterative planning enabled: `{iterative_source_discovery.get('llm_iterative_planning_enabled', False)}`",
            f"- Search iteration count: `{iterative_source_discovery.get('search_iteration_count', 0)}`",
            f"- LLM refinement call count: `{iterative_source_discovery.get('llm_refinement_call_count', 0)}`",
            f"- Total queries planned: `{iterative_source_discovery.get('total_queries_planned', 0)}`",
            f"- Total queries executed: `{iterative_source_discovery.get('total_queries_executed', 0)}`",
            f"- Stop decision: `{iterative_source_discovery.get('stop_decision') or 'n/a'}`",
            f"- Stop reason: `{iterative_source_discovery.get('stop_reason') or 'n/a'}`",
            f"- Query source counts: `{source_search.get('query_source_counts') or {}}`",
            f"- Iteration query counts: `{source_search.get('iteration_query_counts') or {}}`",
            "",
            "### 6.2 LLM Source Critic",
            "",
            f"- Attempted source count: `{llm_summary['source_critic'].get('attempted_source_count')}`",
            f"- Assessed source count: `{llm_summary['source_critic'].get('assessed_source_count')}`",
            f"- Skipped source count: `{llm_summary['source_critic'].get('skipped_source_count')}`",
            f"- Blocked fetch count: `{llm_summary['source_critic'].get('blocked_fetch_count')}`",
            f"- Allowed fetch count: `{llm_summary['source_critic'].get('allowed_fetch_count')}`",
            f"- Context-only count: `{llm_summary['source_critic'].get('context_only_count')}`",
            f"- Needs review count: `{llm_summary['source_critic'].get('needs_review_count')}`",
            f"- Max sources: `{llm_summary['source_critic'].get('max_sources')}`",
            f"- Review blocks fetch: `{llm_summary['source_critic'].get('review_blocks_fetch')}`",
            f"- Failure count: `{llm_summary['source_critic'].get('failure_count')}`",
            f"- Semantic leakage count: `{llm_summary['source_critic'].get('semantic_leakage_count')}`",
            f"- Human review recommended count: `{llm_summary['source_critic'].get('human_review_recommended_count')}`",
            f"- Critic decision counts: `{_format_counts(llm_summary['source_critic'].get('decision_counts'))}`",
            f"- Fetch recommendation counts: `{_format_counts(llm_summary['source_critic'].get('fetch_recommendation_counts'))}`",
            f"- Risk flag counts: `{_format_counts(llm_summary['source_critic'].get('risk_flag_counts'))}`",
            f"- Selected source IDs: `{', '.join(llm_summary['source_critic'].get('selection', {}).get('selected_source_ids') or [])}`",
            "",
            "### 6.3 Optional LLM Source Credibility Advisory",
            "",
            f"- Enabled: `{llm_summary['source_credibility'].get('enabled')}`",
            f"- Assessed source count: `{llm_summary['source_credibility'].get('assessed_source_count')}`",
            f"- Final role counts: `{_format_counts(llm_summary['source_credibility'].get('role_counts'))}`",
            f"- Risk flag counts: `{_format_counts(llm_summary['source_credibility'].get('risk_flag_counts'))}`",
            f"- LLM assessed count: `{llm_summary['source_credibility'].get('llm_assessed_count')}`",
            f"- LLM failure count: `{llm_summary['source_credibility'].get('llm_failure_count')}`",
            f"- Needs review count: `{llm_summary['source_credibility'].get('needs_review_count')}`",
            "",
            "### 6.3.5 Source Identity / Publisher Assessment",
            "",
            f"- Identity assessed source count: `{llm_summary['source_identity'].get('identity_assessed_count')}`",
            f"- LLM identity assessed source count: `{llm_summary['source_identity'].get('llm_identity_assessed_count')}`",
            f"- Post-fetch identity assessed source count: `{llm_summary['source_identity'].get('post_fetch_identity_assessed_count')}`",
            f"- Unknown publisher count: `{llm_summary['source_identity'].get('unknown_publisher_count')}`",
            f"- Source type counts: `{_format_counts(llm_summary['source_identity'].get('source_type_counts'))}`",
            f"- Claim support role counts: `{_format_counts(llm_summary['source_identity'].get('claim_support_role_counts'))}`",
            f"- Fetch use counts: `{_format_counts(llm_summary['source_identity'].get('recommended_fetch_use_counts'))}`",
            f"- Warning counts: `{_format_counts(llm_summary['source_identity'].get('warning_counts'))}`",
            "",
            "### 6.4 LLM Structured Extraction",
            "",
            f"- Extraction mode: `{llm_summary['structured_extraction'].get('mode')}`",
            f"- Eligible chunk count: `{llm_summary['structured_extraction'].get('eligible_chunk_count')}`",
            f"- LLM call count: `{llm_summary['structured_extraction'].get('call_count')}`",
            f"- LLM success count: `{llm_summary['structured_extraction'].get('success_count')}`",
            f"- LLM error count: `{llm_summary['structured_extraction'].get('error_count')}`",
            f"- Raw record count: `{llm_summary['structured_extraction'].get('raw_record_count')}`",
            "",
            "## 7. Extracted records",
            "",
            f"- Normalized record count: `{len(records)}`",
            f"- Run quality status: `{run_quality.get('run_quality_status') or 'n/a'}`",
            f"- Final dataset mode: `{run_quality.get('final_dataset_mode') or 'n/a'}`",
            f"- Quality-gated accepted final dataset count: `{accepted_record_count}`",
            f"- Final case dataset count: `{len(final_case_dataset)}`",
            f"- Zero-case statement count: `{len(zero_case_statements)}`",
            f"- Exposure-monitoring record count: `{len(exposure_monitoring_records)}`",
            f"- Surveillance summary record count: `{len(surveillance_summary_records)}`",
            f"- Outbreak summary record count: `{len(outbreak_summary_records)}`",
            f"- Context record count: `{len(context_records)}`",
            f"- Unclassified observation count: `{len(unclassified_observation_records)}`",
            f"- Observation dataset view counts: `{observation_type_dataset_summary.get('dataset_view_counts') or {}}`",
            f"- Pre-quality-gate record count: `{pre_quality_record_count}`",
            f"- Quarantined record count: `{quarantined_record_count}`",
            f"- Pending review record count: `{pending_review_record_count}`",
            f"- Non-primary observation count: `{non_primary_observation_count}`",
            f"- Final dataset post-review count: `{post_review_record_count}`",
            f"- Primary-case dataset status: `{run_quality.get('primary_case_dataset_status') or 'n/a'}`",
            f"- Recommended primary dataset message: `{run_quality.get('recommended_primary_dataset_message') or 'n/a'}`",
            f"- Primary-case eligible accepted count: `{run_quality.get('primary_case_dataset_eligible_count', 0)}`",
            f"- Corroborated primary case event count: `{run_quality.get('corroborated_primary_case_event_count', 0)}`",
            f"- Record inclusion status counts: `{final_dataset_quality.get('record_final_inclusion_status_counts') or {}}`",
            f"- Run quality warnings: `{run_quality.get('warnings') or []}`",
            f"- Accepted source counts: `{_format_counts(_count_by(accepted_records, 'source_id'))}`",
            "",
            *(
                [
                    "No quality-gated records are available to list.",
                    "Candidate records, if any, are recorded in the pre-quality, quarantined, and pending-review artifacts.",
                    "",
                ]
                if accepted_record_count == 0
                else []
            ),
            "| Record ID | Date / period | Location | Cases | Deaths | Source | LLM used |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for record in accepted_records[:12]:
        date_value = record.get("reporting_period") or record.get("date_reported") or record.get("date_anchor")
        location = record.get("subnational_location") or record.get("geographic_scope") or record.get("country")
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(record.get("record_id")),
                    _cell(date_value),
                    _cell(location),
                    _cell(_case_value(record)),
                    _cell(record.get("deaths")),
                    _cell(record.get("source_id")),
                    _cell(record.get("llm_used")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 8. Validation comparison",
            "",
            "### Claim-level corroboration",
            "",
            f"- Claim count: `{len(claims)}`",
            f"- Claim comparison count: `{len(claim_comparisons)}`",
            f"- Corroborated event count: `{len(corroborated_events)}`",
            (
                "- Corroborated primary case event count: "
                f"`{corroboration_summary.get('corroborated_primary_case_event_count', 0)}`"
            ),
            (
                "- Observation type counts: "
                f"`{_format_counts(corroboration_summary.get('observation_type_counts'))}`"
            ),
            (
                "- Corroboration status counts: "
                f"`{_format_counts(corroboration_summary.get('status_counts'))}`"
            ),
            "",
            (
                "- Validation source compatibility status: "
                f"`{validation_manifest.get('validation_source_compatibility_status')}`"
            ),
            (
                "- Active / inactive / raw validation records: "
                f"`{validation_manifest.get('active_validation_record_count', 0)}` / "
                f"`{validation_manifest.get('inactive_validation_record_count', 0)}` / "
                f"`{validation_manifest.get('raw_validation_record_count', 0)}`"
            ),
            f"- Validation record count: `{evaluation_summary.get('validation_record_count', 0)}`",
            f"- Evaluation row count: `{evaluation_summary.get('evaluation_row_count', 0)}`",
            f"- Evaluation rows flagged for human review: `{evaluation_summary.get('human_review_flagged_row_count', 0)}`",
            f"- Overall match status counts: `{_format_counts(evaluation_summary.get('overall_match_status_counts'))}`",
            f"- Masking compliance status counts: `{_format_counts(evaluation_summary.get('masking_compliance_status_counts'))}`",
            f"- Reserved source leakage count: `{evaluation_summary.get('reserved_source_leakage_count', 0)}`",
            "",
            "| Eval row | Status | Collection cases | Validation cases | Human review | Reason |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in evaluation_rows[:12]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(row.get("evaluation_row_id")),
                    _cell(row.get("overall_match_status")),
                    _cell(row.get("collection_case_count")),
                    _cell(row.get("validation_case_count")),
                    _cell(row.get("human_review_flag")),
                    _cell(row.get("review_reason")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 9. Human review queue",
            "",
            f"- Human review item count: `{len(review_items)}`",
            f"- Evaluation review flag count: `{evaluation_summary.get('human_review_flagged_row_count', 0)}`",
            f"- Anomaly review item count: `{len([item for item in review_items if item.get('item_type') == 'anomaly'])}`",
            "",
            "| Review ID | Type | Related IDs | Reason |",
            "|---|---|---|---|",
        ]
    )
    for item in review_items[:12]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(item.get("review_id")),
                    _cell(item.get("item_type")),
                    _cell(", ".join(item.get("related_ids") or [])),
                    _cell(item.get("reason")),
                ]
            )
            + " |"
        )

    _append_stage11_report_section(
        lines,
        anomaly_results=anomaly_results,
        anomaly_summary=anomaly_summary,
        human_review_application_summary=human_review_application_summary,
        applied_decisions=applied_decisions,
        rejected_decisions=rejected_decisions,
        audit_trail=audit_trail,
        final_dataset_post_review=final_dataset_post_review,
        records_excluded_by_review=records_excluded_by_review,
    )

    lines.extend(
        [
            "",
            "## 10. Output files",
            "",
            f"- Run output directory: `{output_dir}`",
            f"- Collection final dataset: `{collection_manifest['files'].get('final_dataset_csv')}`",
            f"- Collection final case dataset: `{collection_manifest['files'].get('final_case_dataset_csv')}`",
            f"- Collection zero-case statements: `{collection_manifest['files'].get('zero_case_statements_csv')}`",
            f"- Collection exposure-monitoring records: `{collection_manifest['files'].get('exposure_monitoring_records_csv')}`",
            f"- Collection surveillance summaries: `{collection_manifest['files'].get('surveillance_summary_records_csv')}`",
            f"- Collection outbreak summaries: `{collection_manifest['files'].get('outbreak_summary_records_csv')}`",
            f"- Collection context records: `{collection_manifest['files'].get('context_records_csv')}`",
            f"- Collection unclassified observations: `{collection_manifest['files'].get('unclassified_observation_records_csv')}`",
            f"- Collection observation dataset summary: `{collection_manifest['files'].get('observation_type_dataset_summary_json')}`",
            f"- Collection pre-quality-gate records: `{collection_manifest['files'].get('final_dataset_pre_quality_gate_csv')}`",
            f"- Collection quarantined records: `{collection_manifest['files'].get('quarantined_records_csv')}`",
            f"- Collection pending review records: `{collection_manifest['files'].get('pending_review_records_csv')}`",
            f"- Collection non-primary observations: `{collection_manifest['files'].get('non_primary_observations_csv')}`",
            f"- Collection record inclusion decisions: `{collection_manifest['files'].get('record_inclusion_decisions_json')}`",
            f"- Collection final dataset post-review: `{collection_manifest['files'].get('final_dataset_post_review_csv')}`",
            f"- Collection anomaly results: `{collection_manifest['files'].get('anomaly_results_json')}`",
            f"- Collection human review audit trail: `{collection_manifest['files'].get('human_review_audit_trail_json')}`",
            f"- Collection source registry: `{collection_manifest['files'].get('source_registry_json')}`",
            f"- Validation mode: `{validation_manifest.get('validation_mode')}`",
            f"- Validation records source: `{validation_manifest.get('validation_records_source')}`",
            f"- Validation records output: `{validation_manifest.get('ground_truth_records_csv')}`",
            (
                "- Inactive validation records: "
                f"`{validation_manifest.get('inactive_validation_records_csv')}`"
            ),
            (
                "- Validation compatibility summary: "
                f"`{validation_manifest.get('validation_source_compatibility_summary_json')}`"
            ),
            f"- Evaluation report CSV: `{evaluation_outputs.get('evaluation_report_csv')}`",
            f"- Human-readable report: `{path}`",
            "",
            "## 11. Run conclusion",
            "",
            (
                "This workflow run executed the configured data collection workflow "
                "through the exported graph and artifact pipeline. The user-facing "
                f"run status is: {run_status_text}"
            ),
            *(
                [
                    (
                        "The run completed technically, but it should not be read as "
                        "a successful data collection result because no quality-gated "
                        "accepted records were produced."
                    )
                ]
                if accepted_record_count == 0
                else [
                    (
                        "The run completed technically and produced quality-gated "
                        "accepted records in the final dataset."
                    )
                ]
            ),
            "",
            f"- Package metadata says LLM used: `{metadata.get('llm_used')}`",
            f"- Contains synthetic fixture data: `{metadata.get('contains_synthetic_fixture_data')}`",
            f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
            "",
        ]
    )
    report = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return report


def _stream_chunk_type_and_data(chunk) -> tuple[str | None, object]:
    if isinstance(chunk, tuple) and len(chunk) == 2:
        mode, data = chunk
        return str(mode), data
    if isinstance(chunk, dict) and "type" in chunk and "data" in chunk:
        return str(chunk.get("type")), chunk.get("data")
    if isinstance(chunk, dict) and len(chunk) == 1:
        mode, data = next(iter(chunk.items()))
        if mode in {"tasks", "updates", "custom", "values", "debug"}:
            return str(mode), data
    return None, chunk


def _task_node_name(data: object) -> str | None:
    if not isinstance(data, dict):
        return None
    name = data.get("name") or data.get("node") or data.get("node_name")
    return str(name) if name else None


def _is_task_start(data: object) -> bool:
    return (
        isinstance(data, dict)
        and "input" in data
        and "result" not in data
        and "error" not in data
    )


def _is_task_finish(data: object) -> bool:
    return isinstance(data, dict) and ("result" in data or "error" in data)


def _run_completion_payload(result: dict) -> dict:
    return {
        "source_candidate_count": len(result.get("source_candidates") or []),
        "source_registry_count": len(result.get("source_registry") or []),
        "document_count": len(result.get("documents") or []),
        "raw_record_count": len(result.get("raw_records") or []),
        "validated_record_count": len(result.get("validated_records") or []),
        "normalized_record_count": len(result.get("normalized_records") or []),
        "human_review_item_count": len(result.get("human_review_queue") or []),
        "current_route": result.get("current_route"),
    }


def _run_graph_with_events(
    initial_state: dict,
    *,
    output_dir: Path,
    session_id: str,
    live_status: bool = True,
    return_writer: bool = False,
):
    """Run the LangGraph graph through stream events and return final state."""

    final_state = dict(initial_state)
    latest_node_payloads: dict[str, dict] = {}
    current_node: str | None = None
    failed_event_written = False
    trace_config = _langsmith_trace_config(session_id)
    previous_trace_env = _install_trace_env(trace_config)
    previous_external_trace_env = (
        None
        if _langsmith_external_trace_enabled()
        else _suppress_external_langsmith_env()
    )
    from data_collection_workflow.session_runtime import get_runtime
    active_runtime = get_runtime()
    writer = RunEventWriter(
        session_dir=output_dir,
        session_id=session_id,
        node_order=WORKFLOW_NODE_ORDER,
        live_status=live_status,
        resume=bool(active_runtime and getattr(active_runtime, "resume", False)),
    )
    try:
        with writer, ExitStack() as graph_stack:
            writer.append_run_started(
                {
                    "user_request": initial_state.get("user_request"),
                    "structured_task": initial_state.get("structured_task"),
                    **(trace_config.get("metadata") or {}),
                }
            )
            try:
                from data_collection_workflow.session_runtime import get_runtime, checkpoint_graph
                runtime = get_runtime()
                graph = graph_stack.enter_context(checkpoint_graph(runtime)) if runtime else build_graph()
                graph_input = initial_state
                if runtime:
                    trace_config['configurable'] = {**(trace_config.get('configurable') or {}), 'thread_id': session_id}
                    from data_collection_workflow.acquisition_budget import graph_recursion_limit
                    trace_config['recursion_limit'] = graph_recursion_limit(runtime.config, runtime.ledger.snapshot()['limits'])
                    if getattr(runtime, 'resume', False):
                        saved = graph.get_state(trace_config)
                        if not saved.values:
                            raise ValueError('session has no graph checkpoint to resume')
                        final_state.update(saved.values)
                        saved_revision=int((saved.values.get('run_budget_ledger') or {}).get('budget_revision',0))
                        current_budget = runtime.ledger.snapshot()
                        saved_providers = (saved.values.get('run_budget_ledger') or {}).get('providers') or {}
                        provider_reopened = bool(getattr(runtime, 'provider_continuation', False)) and any(
                            row.get('status') == 'resumed'
                            and row.get('event_id') != (saved_providers.get(name) or {}).get('event_id')
                            for name, row in (current_budget.get('providers') or {}).items()
                        )
                        budget_reopened = runtime.frontier is not None and runtime.ledger.budget_revision > saved_revision
                        if not saved.next and (budget_reopened or provider_reopened):
                            runtime.budget_continuation = bool(budget_reopened)
                            graph.update_state(trace_config, {'recovery_stop_reason':None, 'run_budget_ledger':current_budget}, as_node='quality_gate_routing')
                        graph_input = None
                for chunk in graph.stream(
                    graph_input,
                    config=trace_config,
                    stream_mode=["tasks", "updates", "custom"],
                    version="v2",
                ):
                    chunk_type, data = _stream_chunk_type_and_data(chunk)
                    if chunk_type == "tasks":
                        node_name = _task_node_name(data)
                        if _is_task_start(data) and node_name:
                            current_node = node_name
                            payload = {"input": data.get("input")} if isinstance(data, dict) else {}
                            writer.append_node_started(node_name, payload)
                        elif _is_task_finish(data) and node_name:
                            current_node = node_name
                            if isinstance(data, dict) and data.get("error") is not None:
                                error_value = data.get("error")
                                failed_event_written = True
                                writer.append_node_failed(
                                    node_name,
                                    error_value,
                                    latest_node_payloads.get(node_name) or data,
                                )
                                writer.mark_run_failed(error_value, node_name=node_name)
                                if isinstance(error_value, BaseException):
                                    raise error_value
                                if isinstance(error_value, dict):
                                    raise RuntimeError(
                                        str(error_value.get("message") or error_value)
                                    )
                                raise RuntimeError(str(error_value))
                            else:
                                payload = latest_node_payloads.get(node_name)
                                if payload is None and isinstance(data, dict):
                                    payload = summarize_state_update(data.get("result") or {})
                                writer.append_node_completed(node_name, payload or {})
                        continue
                    if chunk_type == "updates" and isinstance(data, dict):
                        for node_name, update in data.items():
                            if isinstance(update, dict):
                                final_state.update(update)
                            latest_node_payloads[str(node_name)] = summarize_state_update(update)
                        continue
                    if chunk_type == "custom":
                        payload = data if isinstance(data, dict) else {"value": data}
                        writer.append_custom_progress(
                            node_name=payload.get("node_name"),
                            message=str(payload.get("message") or "workflow progress"),
                            payload=payload.get("payload", payload),
                            status=str(payload.get("status") or "running"),
                        )
                writer.append_run_completed(_run_completion_payload(final_state))
            except Exception as exc:
                if not failed_event_written:
                    writer.append_node_failed(current_node, exc)
                writer.mark_run_failed(exc, node_name=current_node)
                raise
    finally:
        _flush_langsmith_tracers()
        if previous_external_trace_env is not None:
            _restore_env(previous_external_trace_env)
        _restore_env(previous_trace_env)
    if return_writer:
        return final_state, writer
    return final_state


def _print_effective_acquisition_settings(
    config: dict, *, env_updates: dict, budget_snapshot: dict, origin: str
) -> None:
    """Render effective settings without changing config, budget or provenance."""
    from data_collection_workflow.acquisition_budget import is_adaptive_budget

    universal = config.get("universal") or {}
    policy = universal.get("budget_policy") or {}
    adaptive = is_adaptive_budget(config)
    byte_source = (
        "content_fetch.max_bytes"
        if "max_bytes" in (config.get("content_fetch") or {})
        else "runtime profile default"
    )
    print(f"max_bytes: {int(env_updates['FETCH_MAX_BYTES'])} bytes per response "
          f"(source: {origin}; {byte_source})")
    print(f"acquisition_budget_policy: {'adaptive' if adaptive else 'strict'} "
          f"(source: {origin}; universal.budget_policy "
          f"{'version 2' if adaptive else 'absent or strict'})")
    if adaptive:
        print(f"soft_source_target: {int(policy.get('soft_source_target', 50))} "
              f"acquisition targets; soft checkpoint "
              f"(source: {origin}; universal.budget_policy.soft_source_target)")

    base_sources = {
        "source_targets": "adaptive budget default",
        "http_requests": "adaptive budget default",
        "fetch": "strict total fetch default",
        "fetch_ordinary": "min(content_fetch.max_total_sources, content_fetch.max_search_derived_sources)",
        "search": "source_search.iterative/max_queries + source_search.authority_gap_retry allowances",
        "search_results": "source_search/iterative result caps + source_search.authority_gap_retry.result_budget",
        "extraction": "min(evidence default, llm.extraction.scheduler.safety_max_calls)",
        "browser": "evidence browser default",
        "ocr": "evidence OCR default",
    }
    units = {
        "source_targets": "canonical acquisition targets", "http_requests": "actual HTTP requests",
        "fetch": "fetch operations", "fetch_ordinary": "ordinary fetch operations",
        "search": "search queries", "search_results": "search results", "extraction": "extraction calls",
        "browser": "browser navigations", "ocr": "OCR pages",
    }
    amendments = {}
    for event in budget_snapshot.get("budget_amendments") or []:
        for kind in event.get("increases") or {}:
            amendments[kind] = int(event["budget_revision"])
    limits = budget_snapshot.get("limits") or {}
    configured_limits = universal.get("budget_limits") or {}
    used = budget_snapshot.get("used") or {}
    for kind in [*base_sources, *sorted(key for key in limits if key.startswith("model:"))]:
        if kind not in limits:
            continue
        if kind.startswith("model:"):
            source = "model stage default restricted by universal.budget_limits/model_limits"
            if kind == "model:SourceIdentityAgentOutput":
                source += " and llm.source_identity.max_sources"
            elif kind == "model:LLMSourceCredibilitySuggestion":
                source += " and llm.source_credibility.max_sources"
        else:
            source = base_sources[kind]
            if kind in configured_limits:
                restriction = "" if adaptive and kind in {"source_targets", "http_requests"} else source + "; restricted by "
                source = restriction + "universal.budget_limits." + kind
            if kind in {"search", "search_results"} and not (config.get("source_search") or {}).get("enabled", True):
                source += "; source_search.enabled=false"
        if kind in amendments:
            source = f"accepted budget amendment revision {amendments[kind]}"
        cap, spent = int(limits[kind]), int(used.get(kind, 0))
        print(f"{kind}: {cap} {units.get(kind, 'model calls')}; "
              f"used={spent}, remaining={max(0, cap - spent)} (source: {origin}; {source})")
    print(f"extraction_reserve: {int(budget_snapshot.get('extraction_reserve', 0))} calls "
          "included in extraction total (source: effective ledger reserve)")
    print(f"model_default_limit: {int(budget_snapshot.get('model_default_limit', 30))} calls per other stage "
          "(source: runtime model-stage default)")
    print(f"budget_revision: {int(budget_snapshot.get('budget_revision', 0))}", flush=True)


def _load_provider_resume(path, *, provider):
    """An explicit acknowledgement cannot change task, model or budget settings."""
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if (not isinstance(payload, dict) or set(payload) != {"provider", "event_id", "reason"}
            or not all(isinstance(value, str) and value.strip() for value in payload.values())):
        raise ValueError('--provider-resume requires only nonempty provider, event_id and reason strings')
    if payload['provider'] != str(provider).strip().lower():
        raise ValueError('--provider-resume must name the configured provider')
    return payload


def _apply_provider_resume(runtime, acknowledgement):
    if acknowledgement is None:
        return
    # initialize_universal_run must already have verified the saved fingerprints.
    event = runtime.ledger.resume_provider(**acknowledgement)
    current = runtime.ledger.provider_status(acknowledgement['provider']) or {}
    runtime.provider_continuation = (
        current.get('status') == 'resumed' and current.get('event_id') == event.get('event_id')
    )


def _preflight_universal_provider(runtime, provider):
    from data_collection_workflow.provider_failures import ProviderAccountLimit
    if (runtime.ledger.provider_status(provider) or {}).get('status') == 'halted':
        print('llm_preflight: deferred (provider_account_limit); saved evidence and local export remain available', flush=True)
        return
    try:
        _preflight_llm_with_trace_policy()
    except ProviderAccountLimit:
        print('llm_preflight: provider_account_limit; continuing local processing and partial export', flush=True)


def run_workflow(args: argparse.Namespace) -> dict:
    _, config = _config_with_cli_overrides(args)
    validate_workflow_task(config)
    from data_collection_workflow.session_runtime import prepare_universal_config, initialize_universal_run, REVISION
    config = prepare_universal_config(config)
    universal = config.get('pipeline_mode') == REVISION
    provider_resume_path = getattr(args, 'provider_resume', None)
    if provider_resume_path and (not universal or not getattr(args, 'resume_session', None)):
        raise ValueError('--provider-resume requires a same-version evidence --resume-session')
    from data_collection_workflow.acquisition_budget import is_adaptive_budget, load_budget_amendment
    amendment_path = getattr(args, 'budget_amendment', None)
    if amendment_path and (not universal or not getattr(args, 'resume_session', None) or not is_adaptive_budget(config)):
        raise ValueError('--budget-amendment requires an adaptive same-code --resume-session')
    amendment = load_budget_amendment(amendment_path) if amendment_path else None
    provider, model = _config_provider_model(config)
    acknowledgement = _load_provider_resume(provider_resume_path, provider=provider) if provider_resume_path else None
    output_dir = workflow_output_dir_from_config(config)
    env_updates = workflow_run_env_from_config(config)
    if _llm_enabled(env_updates) and not model:
        raise ValueError("An LLM model is required. Set llm.model, LLM_MODEL, or --model before enabling LLM stages.")
    universal_context = initialize_universal_run(config, output_dir, resume_session=getattr(args, 'resume_session', None)) if universal else None
    if acknowledgement is not None:
        _apply_provider_resume(universal_context, acknowledgement)
    if amendment is not None:
        universal_context.amend_budget(**amendment)
    if universal_context is not None:
        _print_effective_acquisition_settings(
            config, env_updates=env_updates, budget_snapshot=universal_context.ledger.snapshot(),
            origin=getattr(args, "acquisition_settings_origin", None)
            or ("saved session config" if getattr(args, "resume_session", None) else "config file"),
        )
    output_config = config.get("output") or {}
    claim_comparison_storage = str(
        output_config.get("claim_comparison_storage")
        or get_env("CLAIM_COMPARISON_STORAGE")
        or "meaningful_edges"
    ).strip().lower()
    if claim_comparison_storage not in {"meaningful_edges", "full"}:
        claim_comparison_storage = "meaningful_edges"
    env_updates["CLAIM_COMPARISON_STORAGE"] = claim_comparison_storage
    final_package_mode = str(
        output_config.get("package_mode")
        or get_env("FINAL_PACKAGE_MODE")
        or "compact"
    ).strip().lower()
    if final_package_mode not in {"compact", "full_audit"}:
        final_package_mode = "compact"
    workflow_config = config.get("workflow") or {}
    validation_config = config.get("validation") or {}
    validation_records_requested_path = (
        validation_config.get("held_out_records_path")
        or validation_config.get("validation_records_path")
        or workflow_config.get("validation_ground_truth_records_path")
    )
    validation_records_explicit = validation_records_requested_path not in {
        None,
        "",
    }
    validation_records_path = None if universal else validation_records_path_from_config(config)
    validation_records = (
        read_csv_records(validation_records_path)
        if validation_records_path and validation_records_path.exists()
        else []
    )
    allow_incompatible_validation_records = validation_config.get(
        "allow_incompatible_validation_records"
    )
    initial_state = workflow_initial_state_from_config(config)
    resolved_validation = resolve_task_compatible_validation_records(
        validation_records=validation_records,
        state_or_task_context=initial_state,
        validation_records_path=validation_records_path,
        validation_records_path_requested=validation_records_requested_path,
        validation_records_explicit=validation_records_explicit,
        validation_records_defaulted=not validation_records_explicit,
        allow_incompatible_validation_records=allow_incompatible_validation_records,
        validation_mode=validation_config.get("mode") or "live_cross_source",
    )

    with temporary_workflow_env(env_updates), (universal_context.activate() if universal_context else nullcontext()):
        from data_collection_workflow.reporting.run_settings import REPORT_ENVIRONMENT_VARIABLES
        report_environment = {key: get_env(key) for key in REPORT_ENVIRONMENT_VARIABLES if get_env(key) is not None}
        if universal and _llm_enabled(env_updates):
            _preflight_universal_provider(universal_context, provider)
        initial_state["validation_records"] = validation_records
        initial_state["active_validation_records"] = resolved_validation[
            "active_validation_records"
        ]
        initial_state["inactive_validation_records"] = resolved_validation[
            "inactive_validation_records"
        ]
        initial_state["validation_source_compatibility_summary"] = (
            resolved_validation["validation_source_compatibility_summary"]
        )
        result, event_writer = _run_graph_with_events(
            initial_state,
            output_dir=output_dir,
            session_id=str((config.get("output") or {}).get("session_id") or output_dir.name),
            live_status=bool(getattr(args, "live_status", True)),
            return_writer=True,
        )

    package = dict(result.get("final_data_package") or {})
    registry = list(result.get("source_registry") or package.get("source_registry") or [])
    active_validation_records = list(
        result.get("active_validation_records")
        if "active_validation_records" in result
        else resolved_validation["active_validation_records"]
    )
    inactive_validation_records = list(
        result.get("inactive_validation_records")
        if "inactive_validation_records" in result
        else resolved_validation["inactive_validation_records"]
    )
    validation_source_compatibility_summary = dict(
        result.get("validation_source_compatibility_summary")
        or resolved_validation["validation_source_compatibility_summary"]
    )
    validation_manifest = _write_validation_outputs(
        output_dir,
        active_validation_records,
        registry,
        inactive_validation_records=inactive_validation_records,
        validation_source_compatibility_summary=validation_source_compatibility_summary,
        raw_validation_records=validation_records,
    )
    evaluation_rows, evaluation_summary = build_evaluation_report(
        collection_records=package.get("final_dataset") or [],
        validation_records=active_validation_records,
        collection_source_registry=package.get("source_registry") or [],
        reserved_source_ids=set(VALIDATION_SOURCE_IDS),
        conflicts=package.get("conflicts") or [],
        human_review_items=package.get("human_review_items") or [],
        validation_source_compatibility_summary=validation_source_compatibility_summary,
    )
    evaluation_summary.update(
        {
            "live_fetch_enabled": env_updates.get("ENABLE_LIVE_FETCH") == "true",
            "fixture_documents_enabled": False,
            "llm_source_planning_enabled": env_updates.get(
                "ENABLE_LLM_SOURCE_PLANNING"
            )
            == "true",
            "llm_source_critic_enabled": env_updates.get(
                "ENABLE_LLM_SOURCE_CRITIC"
            )
            == "true",
            "llm_source_identity_enabled": env_updates.get(
                "ENABLE_LLM_SOURCE_IDENTITY"
            )
            == "true",
            "llm_extraction_enabled": env_updates.get("ENABLE_LLM_EXTRACTION")
            == "true",
            "provider": provider,
            "model": model,
        }
    )
    existing_review_ids = {
        str(item.get("review_id"))
        for item in (package.get("human_review_items") or [])
        if item.get("review_id")
    }
    evaluation_review_items = build_evaluation_review_items(
        evaluation_rows,
        existing_review_ids=existing_review_ids,
    )
    if evaluation_review_items:
        package["human_review_items"] = (
            evaluation_review_items + list(package.get("human_review_items") or [])
        )
        result["human_review_queue"] = (
            evaluation_review_items + list(result.get("human_review_queue") or [])
        )
        result["final_data_package"] = package
        evaluation_summary["evaluation_review_item_count"] = len(evaluation_review_items)
    else:
        evaluation_summary["evaluation_review_item_count"] = 0
    evaluation_summary["human_review_item_count"] = len(
        package.get("human_review_items") or []
    )
    collection_manifest = export_final_data_package(
        package,
        output_dir / "collection",
        package_mode=final_package_mode,
    )
    evaluation_outputs = write_evaluation_outputs(
        evaluation_rows,
        evaluation_summary,
        output_dir / "evaluation",
    )
    source_split = _source_split_summary(registry)
    live_summary = _live_fetch_summary(result)
    llm_summary = _llm_stage_summary(result, provider, model)

    diagnostics_dir = output_dir / "diagnostics"
    write_json(source_split, diagnostics_dir / "source_split_summary.json")
    write_json(live_summary, diagnostics_dir / "live_fetch_summary.json")
    write_json(
        result.get("content_fetch_summary") or {},
        diagnostics_dir / "content_fetch_summary.json",
    )
    write_json(
        result.get("document_quality_summary") or {},
        diagnostics_dir / "document_quality_summary.json",
    )
    write_json(
        result.get("document_parse_summary") or {},
        diagnostics_dir / "document_parse_summary.json",
    )
    write_json(
        result.get("fetch_manifest") or [],
        diagnostics_dir / "fetch_manifest.json",
    )
    write_json(llm_summary, diagnostics_dir / "llm_stage_summary.json")
    write_json(
        result.get("source_search_execution_summary") or {},
        diagnostics_dir / "source_search_execution_summary.json",
    )
    write_json(
        result.get("source_search_results") or [],
        diagnostics_dir / "search_results_manifest.json",
    )
    write_json(
        result.get("source_discovery_summary") or {},
        diagnostics_dir / "source_discovery_summary.json",
    )
    write_json(
        result.get("iterative_source_discovery_summary") or {},
        diagnostics_dir / "iterative_source_discovery_summary.json",
    )
    write_json(
        result.get("search_iteration_plans") or [],
        diagnostics_dir / "search_iteration_plans.json",
    )
    write_json(
        result.get("search_iteration_observations") or [],
        diagnostics_dir / "search_iteration_observations.json",
    )
    write_json(
        result.get("search_refinement_decisions") or [],
        diagnostics_dir / "search_refinement_decisions.json",
    )
    write_json(
        result.get("iterative_search_queries") or [],
        diagnostics_dir / "iterative_search_queries.json",
    )
    write_json(
        result.get("source_credibility_summary") or {},
        diagnostics_dir / "source_credibility_summary.json",
    )
    write_json(
        result.get("source_credibility_assessments") or [],
        diagnostics_dir / "source_credibility_assessments.json",
    )
    write_json(
        result.get("source_identity_summary") or {},
        diagnostics_dir / "source_identity_summary.json",
    )
    write_json(
        result.get("source_identity_assessments") or [],
        diagnostics_dir / "source_identity_assessments.json",
    )
    write_json(
        result.get("source_triage_results") or [],
        diagnostics_dir / "source_triage_results.json",
    )
    write_json(
        result.get("chunk_relevance_assessments")
        or package.get("chunk_relevance_assessments")
        or [],
        diagnostics_dir / "chunk_relevance_assessments.json",
    )
    write_json(
        result.get("record_task_fit_assessments")
        or package.get("record_task_fit_assessments")
        or [],
        diagnostics_dir / "record_task_fit_assessments.json",
    )
    write_json(
        result.get("metric_extraction_plan")
        or package.get("metric_extraction_plan")
        or {},
        diagnostics_dir / "metric_extraction_plan.json",
    )
    write_json(
        result.get("source_coverage_requirements") or [],
        diagnostics_dir / "source_coverage_requirements.json",
    )
    write_json(
        result.get("source_coverage_audit") or {},
        diagnostics_dir / "source_coverage_audit.json",
    )
    write_json(
        result.get("must_fetch_sources") or [],
        diagnostics_dir / "must_fetch_sources.json",
    )
    write_json(
        result.get("fetch_failures_blocking") or [],
        diagnostics_dir / "fetch_failures_blocking.json",
    )
    write_json(
        result.get("official_extraction_queue") or [],
        diagnostics_dir / "official_extraction_queue.json",
    )
    write_json(
        result.get("official_extraction_failures") or [],
        diagnostics_dir / "official_extraction_failures.json",
    )
    write_json(
        result.get("extraction_budget_by_source") or {},
        diagnostics_dir / "extraction_budget_by_source.json",
    )
    write_json(
        result.get("evidence_chunks") or package.get("evidence_chunks") or [],
        diagnostics_dir / "evidence_chunks.json",
    )
    write_json(result.get("raw_records") or [], diagnostics_dir / "raw_records.json")
    write_json(
        result.get("validated_records") or [],
        diagnostics_dir / "validated_records.json",
    )
    write_json(
        result.get("rejected_records") or [],
        diagnostics_dir / "rejected_records.json",
    )
    write_json(
        result.get("normalized_records") or [],
        diagnostics_dir / "normalized_records.json",
    )
    write_json(
        result.get("event_clusters") or [],
        diagnostics_dir / "event_clusters.json",
    )
    write_json(
        result.get("duplicate_clusters") or [],
        diagnostics_dir / "duplicate_clusters.json",
    )
    write_json(
        result.get("validation_cases") or [],
        diagnostics_dir / "validation_cases.json",
    )
    write_json(
        result.get("validation_comparisons") or [],
        diagnostics_dir / "validation_comparisons.json",
    )
    write_json(
        result.get("validation_results") or [],
        diagnostics_dir / "validation_results.json",
    )
    write_json(
        active_validation_records,
        diagnostics_dir / "active_validation_records.json",
    )
    write_json(
        inactive_validation_records,
        diagnostics_dir / "inactive_validation_records.json",
    )
    write_json(
        validation_source_compatibility_summary,
        diagnostics_dir / "validation_source_compatibility_summary.json",
    )
    write_json(
        result.get("validation_summary") or {},
        diagnostics_dir / "validation_summary.json",
    )
    write_json(
        result.get("trusted_source_validation_summary") or {},
        diagnostics_dir / "trusted_source_validation_summary.json",
    )
    write_json(
        result.get("cross_source_validation_summary") or {},
        diagnostics_dir / "cross_source_validation_summary.json",
    )
    write_json(result.get("claims") or [], diagnostics_dir / "claims.json")
    write_json(
        {
            "artifact_type": "claim_comparisons",
            "artifact_path": collection_manifest["files"]["claim_comparisons_json"],
            "collection_manifest_path": collection_manifest["files"]["final_package_json"],
            "claim_comparison_storage": claim_comparison_storage,
            "claim_comparison_count": collection_manifest["section_counts"][
                "claim_comparisons"
            ],
        },
        diagnostics_dir / "claim_comparisons_reference.json",
    )
    write_json(
        result.get("corroborated_events") or [],
        diagnostics_dir / "corroborated_events.json",
    )
    write_json(
        result.get("corroboration_summary") or {},
        diagnostics_dir / "corroboration_summary.json",
    )
    write_json(result.get("conflicts") or [], diagnostics_dir / "conflicts.json")
    write_json(
        result.get("anomaly_results") or [],
        diagnostics_dir / "anomaly_results.json",
    )
    write_json(
        result.get("anomaly_summary") or {},
        diagnostics_dir / "anomaly_summary.json",
    )
    write_json(
        result.get("human_review_decisions") or [],
        diagnostics_dir / "human_review_decisions.json",
    )
    write_json(
        result.get("applied_human_review_decisions") or [],
        diagnostics_dir / "applied_human_review_decisions.json",
    )
    write_json(
        result.get("rejected_human_review_decisions") or [],
        diagnostics_dir / "rejected_human_review_decisions.json",
    )
    write_json(
        result.get("human_review_audit_trail") or [],
        diagnostics_dir / "human_review_audit_trail.json",
    )
    write_json(
        result.get("human_review_application_summary") or {},
        diagnostics_dir / "human_review_application_summary.json",
    )
    write_json(
        result.get("run_quality_summary") or package.get("run_quality_summary") or {},
        diagnostics_dir / "run_quality_summary.json",
    )
    write_json(
        result.get("direct_collection_summary")
        or package.get("direct_collection_summary")
        or {},
        diagnostics_dir / "direct_collection_summary.json",
    )
    source_critic_summary_for_fast_path = result.get("source_critic_summary") or {}
    write_json(
        result.get("direct_fast_path_summary")
        or package.get("direct_fast_path_summary")
        or source_critic_summary_for_fast_path.get("direct_fast_path_summary")
        or {},
        diagnostics_dir / "direct_fast_path_summary.json",
    )
    write_json(
        result.get("final_dataset_quality_summary")
        or package.get("final_dataset_quality_summary")
        or {},
        diagnostics_dir / "final_dataset_quality_summary.json",
    )
    write_json(
        result.get("record_inclusion_decisions")
        or package.get("record_inclusion_decisions")
        or [],
        diagnostics_dir / "record_inclusion_decisions.json",
    )
    write_json(
        result.get("final_dataset_pre_quality_gate")
        or package.get("final_dataset_pre_quality_gate")
        or [],
        diagnostics_dir / "final_dataset_pre_quality_gate.json",
    )
    write_json(
        result.get("quarantined_records") or package.get("quarantined_records") or [],
        diagnostics_dir / "quarantined_records.json",
    )
    write_json(
        result.get("pending_review_records")
        or package.get("pending_review_records")
        or [],
        diagnostics_dir / "pending_review_records.json",
    )
    write_json(
        result.get("non_primary_observations")
        or package.get("non_primary_observations")
        or [],
        diagnostics_dir / "non_primary_observations.json",
    )
    observation_dataset_sections = [
        "final_case_dataset",
        "zero_case_statements",
        "exposure_monitoring_records",
        "surveillance_summary_records",
        "outbreak_summary_records",
        "context_records",
        "unclassified_observation_records",
    ]
    for section in observation_dataset_sections:
        write_json(
            result.get(section) or package.get(section) or [],
            diagnostics_dir / f"{section}.json",
        )
    write_json(
        result.get("observation_type_dataset_summary")
        or package.get("observation_type_dataset_summary")
        or {},
        diagnostics_dir / "observation_type_dataset_summary.json",
    )
    write_json(
        result.get("final_dataset_post_review") or [],
        diagnostics_dir / "final_dataset_post_review.json",
    )
    write_json(
        result.get("records_excluded_by_human_review") or [],
        diagnostics_dir / "records_excluded_by_human_review.json",
    )
    write_json(
        result.get("structured_extraction_summary") or {},
        diagnostics_dir / "structured_extraction_summary.json",
    )
    write_json(
        result.get("schema_validation_summary") or {},
        diagnostics_dir / "schema_validation_summary.json",
    )
    write_json(
        result.get("record_normalization_summary") or {},
        diagnostics_dir / "record_normalization_summary.json",
    )
    write_json(
        result.get("event_clustering_summary") or {},
        diagnostics_dir / "event_clustering_summary.json",
    )
    write_json(
        result.get("duplicate_detection_summary") or {},
        diagnostics_dir / "duplicate_detection_summary.json",
    )
    write_json(
        result.get("disease_relevance_summary") or {},
        diagnostics_dir / "disease_relevance_summary.json",
    )
    write_json(
        result.get("localized_source_planning_summary") or {},
        diagnostics_dir / "localized_source_planning_summary.json",
    )
    write_json(
        result.get("source_critic_summary") or {},
        diagnostics_dir / "source_critic_summary.json",
    )
    write_json(
        result.get("source_critic_results") or [],
        diagnostics_dir / "source_critic_results.json",
    )
    write_json(result.get("collection_trace") or [], diagnostics_dir / "collection_trace.json")
    write_json(
        {
            "task_intake_summary": result.get("task_intake_summary"),
            "disease_intelligence_summary": result.get(
                "disease_intelligence_summary"
            ),
            "profile_schema_summary": result.get("profile_schema_summary"),
            "executable_source_plan_summary": result.get(
                "executable_source_plan_summary"
            ),
            "localized_source_planning_summary": result.get(
                "localized_source_planning_summary"
            ),
            "source_planning_agent_summary": result.get("source_planning_agent_summary"),
            "source_search_execution_summary": result.get(
                "source_search_execution_summary"
            ),
            "iterative_source_discovery_summary": result.get(
                "iterative_source_discovery_summary"
            ),
            "source_discovery_summary": result.get("source_discovery_summary"),
            "source_registry_summary": result.get("source_registry_summary"),
            "source_screening_summary": result.get("source_screening_summary"),
            "source_critic_summary": result.get("source_critic_summary"),
            "source_routing_summary": result.get("source_routing_summary"),
            "source_credibility_summary": result.get("source_credibility_summary"),
            "source_identity_summary": result.get("source_identity_summary"),
            "content_fetch_summary": result.get("content_fetch_summary"),
            "document_parse_summary": result.get("document_parse_summary"),
            "fixture_document_summary": result.get("fixture_document_summary"),
            "document_quality_summary": result.get("document_quality_summary"),
            "evidence_chunking_summary": result.get("evidence_chunking_summary"),
            "data_presence_summary": result.get("data_presence_summary"),
            "structured_extraction_summary": result.get("structured_extraction_summary"),
            "llm_extraction_summary": result.get("llm_extraction_summary"),
            "schema_validation_summary": result.get("schema_validation_summary"),
            "record_normalization_summary": result.get("record_normalization_summary"),
            "record_linking_summary": result.get("record_linking_summary"),
            "event_clustering_summary": result.get("event_clustering_summary"),
            "duplicate_detection_summary": result.get("duplicate_detection_summary"),
            "validation_summary": result.get("validation_summary"),
            "validation_source_compatibility_summary": (
                validation_source_compatibility_summary
            ),
            "trusted_source_validation_summary": result.get(
                "trusted_source_validation_summary"
            ),
            "cross_source_validation_summary": result.get(
                "cross_source_validation_summary"
            ),
            "corroboration_summary": result.get("corroboration_summary"),
            "cross_source_consistency_summary": result.get(
                "cross_source_consistency_summary"
            ),
            "anomaly_summary": result.get("anomaly_summary"),
            "human_review_summary": result.get("human_review_summary"),
            "human_review_application_summary": result.get(
                "human_review_application_summary"
            ),
            "run_quality_summary": result.get("run_quality_summary")
            or package.get("run_quality_summary"),
            "final_dataset_quality_summary": result.get(
                "final_dataset_quality_summary"
            )
            or package.get("final_dataset_quality_summary"),
            "observation_type_dataset_summary": result.get(
                "observation_type_dataset_summary"
            )
            or package.get("observation_type_dataset_summary"),
            "disease_relevance_summary": result.get("disease_relevance_summary"),
            "finalization_summary": result.get("finalization_summary"),
        },
        diagnostics_dir / "workflow_summaries.json",
    )

    run_quality_summary = result.get("run_quality_summary") or package.get(
        "run_quality_summary"
    ) or {}
    final_dataset_quality_summary = result.get(
        "final_dataset_quality_summary"
    ) or package.get("final_dataset_quality_summary") or {}
    final_dataset = list(package.get("final_dataset") or [])
    final_dataset_pre_quality_gate = list(
        package.get("final_dataset_pre_quality_gate") or []
    )
    quarantined_records = list(package.get("quarantined_records") or [])
    pending_review_records = list(package.get("pending_review_records") or [])
    non_primary_observations = list(package.get("non_primary_observations") or [])
    final_case_dataset = list(package.get("final_case_dataset") or [])
    zero_case_statements = list(package.get("zero_case_statements") or [])
    exposure_monitoring_records = list(
        package.get("exposure_monitoring_records") or []
    )
    surveillance_summary_records = list(
        package.get("surveillance_summary_records") or []
    )
    outbreak_summary_records = list(package.get("outbreak_summary_records") or [])
    context_records = list(package.get("context_records") or [])
    unclassified_observation_records = list(
        package.get("unclassified_observation_records") or []
    )
    observation_type_dataset_summary = (
        package.get("observation_type_dataset_summary") or {}
    )
    final_dataset_post_review = list(result.get("final_dataset_post_review") or [])

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "user_request": config.get("user_request"),
        "run_mode": config.get("run_mode") or get_env("RUN_MODE") or "standard",
        "case_study_real_mode": _case_study_real_mode_enabled(args),
        "output_dir": str(output_dir),
        "provider": provider,
        "model": model,
        "api_key_present": api_key_present(provider),
        "config_path": str(resolve_workflow_run_config_path(args.config)),
        "live_fetch_enabled": env_updates.get("ENABLE_LIVE_FETCH") == "true",
        "live_search_enabled": env_updates.get("ENABLE_LIVE_SEARCH") == "true",
        "external_fetch_enabled": env_updates.get("EXTERNAL_FETCH_ENABLED")
        == "true",
        "external_fetch_provider_order": env_updates.get(
            "EXTERNAL_FETCH_PROVIDER_ORDER"
        ),
        "human_review_enabled": env_updates.get("HUMAN_REVIEW_ENABLED")
        == "true",
        "source_search_mode": env_updates.get("SEARCH_MODE"),
        "source_search_provider": env_updates.get("SEARCH_PROVIDER"),
        "source_search_api_key_present": search_api_key_present(
            env_updates.get("SEARCH_PROVIDER") or ""
        ),
        "fixture_documents_enabled": False,
        "all_three_llm_stages_enabled": all(
            env_updates.get(key) == "true"
            for key in (
                "ENABLE_LLM_SOURCE_PLANNING",
                "ENABLE_LLM_SOURCE_CRITIC",
                "ENABLE_LLM_EXTRACTION",
            )
        ),
        "trace_node_count": len(result.get("collection_trace") or []),
        "current_route": result.get("current_route"),
        "source_registry_count": len(registry),
        "source_identity_assessed_count": (
            result.get("source_identity_summary") or {}
        ).get("identity_assessed_count", 0),
        "llm_source_identity_assessed_count": (
            result.get("source_identity_summary") or {}
        ).get("llm_identity_assessed_count", 0),
        "document_count": live_summary.get("document_count", 0),
        "fetch_provider_counts": live_summary.get("fetch_provider_counts") or {},
        "external_fetch_failure_counts": live_summary.get(
            "external_fetch_failure_counts"
        )
        or {},
        "selected_fetch_bucket_counts": live_summary.get(
            "selected_fetch_bucket_counts"
        )
        or {},
        "normalized_record_count": len(result.get("normalized_records") or []),
        "evaluation_row_count": evaluation_summary.get("evaluation_row_count", 0),
        "validation_source_compatibility_status": (
            validation_source_compatibility_summary.get("compatibility_status")
        ),
        "validation_mode": validation_source_compatibility_summary.get(
            "validation_mode"
        )
        or env_updates.get("VALIDATION_MODE"),
        "validation_records_source": validation_source_compatibility_summary.get(
            "validation_records_source"
        ),
        "active_validation_record_count": len(active_validation_records),
        "inactive_validation_record_count": len(inactive_validation_records),
        "raw_validation_record_count": len(validation_records),
        "validation_result_count": (result.get("validation_summary") or {}).get(
            "validation_result_count", 0
        ),
        "claim_count": (result.get("corroboration_summary") or {}).get(
            "claim_count", 0
        ),
        "claim_comparison_count": (
            result.get("corroboration_summary") or {}
        ).get("claim_comparison_count", 0),
        "corroborated_event_count": (
            result.get("corroboration_summary") or {}
        ).get("corroborated_event_count", 0),
        "corroborated_primary_case_event_count": (
            result.get("corroboration_summary") or {}
        ).get("corroborated_primary_case_event_count", 0),
        "corroboration_summary": result.get("corroboration_summary") or {},
        "anomaly_result_count": len(result.get("anomaly_results") or []),
        "anomaly_severity_counts": (
            result.get("anomaly_summary") or {}
        ).get("severity_counts")
        or {},
        "human_review_item_count": len(result.get("human_review_queue") or []),
        "human_review_decisions_provided_count": (
            result.get("human_review_application_summary") or {}
        ).get("decisions_provided_count", 0),
        "human_review_decisions_applied_count": (
            result.get("human_review_application_summary") or {}
        ).get("decisions_applied_count", 0),
        "human_review_decisions_rejected_count": (
            result.get("human_review_application_summary") or {}
        ).get("decisions_rejected_count", 0),
        "human_review_audit_entry_count": len(
            result.get("human_review_audit_trail") or []
        ),
        "run_quality_status": run_quality_summary.get("run_quality_status"),
        "run_quality_summary": run_quality_summary,
        "final_dataset_quality_summary": final_dataset_quality_summary,
        "user_facing_run_status": user_facing_run_status(run_quality_summary),
        "recommended_user_message": run_quality_summary.get(
            "recommended_user_message"
        ),
        "accepted_record_count": len(final_dataset),
        "final_dataset_count": len(final_dataset),
        "final_case_dataset_count": len(final_case_dataset),
        "zero_case_statement_count": len(zero_case_statements),
        "exposure_monitoring_record_count": len(exposure_monitoring_records),
        "surveillance_summary_record_count": len(surveillance_summary_records),
        "outbreak_summary_record_count": len(outbreak_summary_records),
        "context_record_count": len(context_records),
        "unclassified_observation_count": len(unclassified_observation_records),
        "observation_type_dataset_summary": observation_type_dataset_summary,
        "dataset_view_counts": observation_type_dataset_summary.get(
            "dataset_view_counts"
        )
        or {},
        "primary_case_dataset_eligible_count": run_quality_summary.get(
            "primary_case_dataset_eligible_count",
            final_dataset_quality_summary.get("primary_case_dataset_eligible_count", 0),
        ),
        "accepted_primary_case_record_count": run_quality_summary.get(
            "accepted_primary_case_record_count",
            final_dataset_quality_summary.get("accepted_primary_case_record_count", 0),
        ),
        "accepted_non_primary_observation_count": run_quality_summary.get(
            "accepted_non_primary_observation_count",
            final_dataset_quality_summary.get(
                "accepted_non_primary_observation_count", 0
            ),
        ),
        "non_primary_observation_count": len(non_primary_observations),
        "primary_case_dataset_status": run_quality_summary.get(
            "primary_case_dataset_status"
        )
        or final_dataset_quality_summary.get("primary_case_dataset_status"),
        "no_primary_case_dataset_records": bool(
            run_quality_summary.get("no_primary_case_dataset_records")
            or final_dataset_quality_summary.get("no_primary_case_dataset_records")
        ),
        "no_corroborated_primary_case_events": bool(
            run_quality_summary.get("no_corroborated_primary_case_events")
            or final_dataset_quality_summary.get("no_corroborated_primary_case_events")
        ),
        "accepted_records_are_not_primary_case_records": bool(
            run_quality_summary.get("accepted_records_are_not_primary_case_records")
            or final_dataset_quality_summary.get(
                "accepted_records_are_not_primary_case_records"
            )
        ),
        "pre_quality_record_count": len(final_dataset_pre_quality_gate),
        "final_dataset_pre_quality_gate_count": len(final_dataset_pre_quality_gate),
        "quarantined_record_count": len(quarantined_records),
        "pending_review_record_count": len(pending_review_records),
        "post_review_record_count": len(final_dataset_post_review),
        "final_dataset_post_review_count": len(final_dataset_post_review),
        "llm_stage_summary": llm_summary,
        "source_search_execution_summary": result.get(
            "source_search_execution_summary"
        )
        or {},
        "iterative_source_discovery_summary": result.get(
            "iterative_source_discovery_summary"
        )
        or {},
        "localized_source_planning_summary": result.get(
            "localized_source_planning_summary"
        )
        or {},
        "source_critic_summary": result.get("source_critic_summary") or {},
        "source_credibility_summary": result.get("source_credibility_summary") or {},
        "source_identity_summary": result.get("source_identity_summary") or {},
        "disease_relevance_summary": result.get("disease_relevance_summary") or {},
        "validation_source_compatibility_summary": (
            validation_source_compatibility_summary
        ),
        "artifact_paths": {
            "collection_manifest": collection_manifest,
            "validation_manifest": validation_manifest,
            "evaluation_outputs": evaluation_outputs,
            "source_split_summary": str(diagnostics_dir / "source_split_summary.json"),
            "llm_stage_summary": str(diagnostics_dir / "llm_stage_summary.json"),
            "source_search_execution_summary": str(
                diagnostics_dir / "source_search_execution_summary.json"
            ),
            "search_results_manifest": str(
                diagnostics_dir / "search_results_manifest.json"
            ),
            "source_discovery_summary": str(
                diagnostics_dir / "source_discovery_summary.json"
            ),
            "iterative_source_discovery_summary": str(
                diagnostics_dir / "iterative_source_discovery_summary.json"
            ),
            "search_iteration_plans": str(
                diagnostics_dir / "search_iteration_plans.json"
            ),
            "search_iteration_observations": str(
                diagnostics_dir / "search_iteration_observations.json"
            ),
            "search_refinement_decisions": str(
                diagnostics_dir / "search_refinement_decisions.json"
            ),
            "iterative_search_queries": str(
                diagnostics_dir / "iterative_search_queries.json"
            ),
            "localized_source_planning_summary": str(
                diagnostics_dir / "localized_source_planning_summary.json"
            ),
            "source_critic_summary": str(
                diagnostics_dir / "source_critic_summary.json"
            ),
            "source_critic_results": str(
                diagnostics_dir / "source_critic_results.json"
            ),
            "source_credibility_summary": str(
                diagnostics_dir / "source_credibility_summary.json"
            ),
            "source_credibility_assessments": str(
                diagnostics_dir / "source_credibility_assessments.json"
            ),
            "source_identity_summary": str(
                diagnostics_dir / "source_identity_summary.json"
            ),
            "source_identity_assessments": str(
                diagnostics_dir / "source_identity_assessments.json"
            ),
            "source_triage_results": str(
                diagnostics_dir / "source_triage_results.json"
            ),
            "chunk_relevance_assessments": str(
                diagnostics_dir / "chunk_relevance_assessments.json"
            ),
            "record_task_fit_assessments": str(
                diagnostics_dir / "record_task_fit_assessments.json"
            ),
            "metric_extraction_plan": str(
                diagnostics_dir / "metric_extraction_plan.json"
            ),
            "source_coverage_requirements": str(
                diagnostics_dir / "source_coverage_requirements.json"
            ),
            "source_coverage_audit": str(
                diagnostics_dir / "source_coverage_audit.json"
            ),
            "must_fetch_sources": str(diagnostics_dir / "must_fetch_sources.json"),
            "fetch_failures_blocking": str(
                diagnostics_dir / "fetch_failures_blocking.json"
            ),
            "evidence_chunks": str(diagnostics_dir / "evidence_chunks.json"),
            "content_fetch_summary": str(diagnostics_dir / "content_fetch_summary.json"),
            "document_quality_summary": str(
                diagnostics_dir / "document_quality_summary.json"
            ),
            "document_parse_summary": str(
                diagnostics_dir / "document_parse_summary.json"
            ),
            "fetch_manifest": str(diagnostics_dir / "fetch_manifest.json"),
            "validation_cases": str(diagnostics_dir / "validation_cases.json"),
            "validation_comparisons": str(
                diagnostics_dir / "validation_comparisons.json"
            ),
            "validation_results": str(diagnostics_dir / "validation_results.json"),
            "active_validation_records": str(
                diagnostics_dir / "active_validation_records.json"
            ),
            "inactive_validation_records": str(
                diagnostics_dir / "inactive_validation_records.json"
            ),
            "validation_source_compatibility_summary": str(
                diagnostics_dir / "validation_source_compatibility_summary.json"
            ),
            "validation_summary": str(diagnostics_dir / "validation_summary.json"),
            "trusted_source_validation_summary": str(
                diagnostics_dir / "trusted_source_validation_summary.json"
            ),
            "cross_source_validation_summary": str(
                diagnostics_dir / "cross_source_validation_summary.json"
            ),
            "claims": str(diagnostics_dir / "claims.json"),
            "claim_comparisons": collection_manifest["files"][
                "claim_comparisons_json"
            ],
            "claim_comparisons_reference": str(
                diagnostics_dir / "claim_comparisons_reference.json"
            ),
            "corroborated_events": str(diagnostics_dir / "corroborated_events.json"),
            "corroboration_summary": str(diagnostics_dir / "corroboration_summary.json"),
            "conflicts": str(diagnostics_dir / "conflicts.json"),
            "anomaly_results": str(diagnostics_dir / "anomaly_results.json"),
            "anomaly_summary": str(diagnostics_dir / "anomaly_summary.json"),
            "human_review_decisions": str(
                diagnostics_dir / "human_review_decisions.json"
            ),
            "applied_human_review_decisions": str(
                diagnostics_dir / "applied_human_review_decisions.json"
            ),
            "rejected_human_review_decisions": str(
                diagnostics_dir / "rejected_human_review_decisions.json"
            ),
            "human_review_audit_trail": str(
                diagnostics_dir / "human_review_audit_trail.json"
            ),
            "human_review_application_summary": str(
                diagnostics_dir / "human_review_application_summary.json"
            ),
            "run_quality_summary": str(diagnostics_dir / "run_quality_summary.json"),
            "final_dataset_quality_summary": str(
                diagnostics_dir / "final_dataset_quality_summary.json"
            ),
            "record_inclusion_decisions": str(
                diagnostics_dir / "record_inclusion_decisions.json"
            ),
            "observation_type_dataset_summary": str(
                diagnostics_dir / "observation_type_dataset_summary.json"
            ),
            "final_case_dataset": str(diagnostics_dir / "final_case_dataset.json"),
            "zero_case_statements": str(diagnostics_dir / "zero_case_statements.json"),
            "exposure_monitoring_records": str(
                diagnostics_dir / "exposure_monitoring_records.json"
            ),
            "surveillance_summary_records": str(
                diagnostics_dir / "surveillance_summary_records.json"
            ),
            "outbreak_summary_records": str(
                diagnostics_dir / "outbreak_summary_records.json"
            ),
            "context_records": str(diagnostics_dir / "context_records.json"),
            "unclassified_observation_records": str(
                diagnostics_dir / "unclassified_observation_records.json"
            ),
            "final_dataset_pre_quality_gate": str(
                diagnostics_dir / "final_dataset_pre_quality_gate.json"
            ),
            "quarantined_records": str(diagnostics_dir / "quarantined_records.json"),
            "pending_review_records": str(
                diagnostics_dir / "pending_review_records.json"
            ),
            "non_primary_observations": str(
                diagnostics_dir / "non_primary_observations.json"
            ),
            "final_dataset_post_review": str(
                diagnostics_dir / "final_dataset_post_review.json"
            ),
            "records_excluded_by_human_review": str(
                diagnostics_dir / "records_excluded_by_human_review.json"
            ),
            "disease_relevance_summary": str(
                diagnostics_dir / "disease_relevance_summary.json"
            ),
        },
    }
    summary["artifact_paths"].update(event_writer.artifact_paths)
    summary.update(event_writer.artifact_paths)
    write_json(summary, output_dir / "workflow_run_summary.json")
    human_review_workflow_paths = write_human_review_workflow_artifacts(output_dir)
    summary["artifact_paths"].update(human_review_workflow_paths)
    priority_summary_path = Path(
        human_review_workflow_paths["human_review_priority_summary_json"]
    )
    if priority_summary_path.exists():
        priority_summary = json.loads(priority_summary_path.read_text(encoding="utf-8"))
        summary["review_reporting"] = {
            "human_review_priority_summary": str(priority_summary_path),
            "prioritized_review_item_count": priority_summary.get(
                "prioritized_review_item_count", 0
            ),
            "top_n_recommended": priority_summary.get(
                "top_n_recommended", 10
            ),
            "priority_level_counts": priority_summary.get(
                "priority_level_counts", {}
            ),
            "issue_category_counts": priority_summary.get(
                "issue_category_counts", {}
            ),
            "generated_from_artifacts_only": True,
            "llm_called_for_review_productization": False,
            "search_called_for_review_productization": False,
            "fetch_called_for_review_productization": False,
        }
    write_json(summary, output_dir / "workflow_run_summary.json")
    collection_readiness_paths = write_collection_readiness_outputs(
        output_dir,
        case_study_real_mode=_case_study_real_mode_enabled(args),
    )
    summary["artifact_paths"].update(collection_readiness_paths)
    readiness_summary_path = Path(
        collection_readiness_paths["collection_readiness_summary_json"]
    )
    if readiness_summary_path.exists():
        summary["collection_readiness_summary"] = json.loads(
            readiness_summary_path.read_text(encoding="utf-8")
        )
    write_json(summary, output_dir / "workflow_run_summary.json")
    workflow_visualization_paths = write_workflow_visualization_artifacts(output_dir)
    summary["artifact_paths"].update(workflow_visualization_paths)
    visualization_summary_path = Path(
        workflow_visualization_paths["workflow_visualization_summary"]
    )
    if visualization_summary_path.exists():
        visualization_summary = json.loads(
            visualization_summary_path.read_text(encoding="utf-8")
        )
        summary["workflow_visualization"] = {
            "workflow_visualization_index": workflow_visualization_paths.get(
                "workflow_visualization_index"
            ),
            "workflow_visualization_summary": str(visualization_summary_path),
            "generated_from_artifacts_only": True,
            "llm_called_for_visualization": False,
            "search_called_for_visualization": False,
            "fetch_called_for_visualization": False,
            "workflow_timeline_step_count": visualization_summary.get(
                "workflow_timeline_step_count", 0
            ),
            "evidence_flow_node_count": visualization_summary.get(
                "evidence_flow_node_count", 0
            ),
            "evidence_flow_edge_count": visualization_summary.get(
                "evidence_flow_edge_count", 0
            ),
            "claim_comparison_card_count": visualization_summary.get(
                "claim_comparison_card_count", 0
            ),
            "dataset_view_node_count": visualization_summary.get(
                "dataset_view_node_count", 0
            ),
            "human_review_visualization_item_count": visualization_summary.get(
                "human_review_visualization_item_count", 0
            ),
            "missing_link_warning_count": visualization_summary.get(
                "missing_link_warning_count", 0
            ),
        }
    if bool(output_config.get("write_latest_alias", True)):
        latest_visualization_paths = _write_workflow_visualization_latest_aliases(
            workflow_visualization_paths
        )
        summary["artifact_paths"].update(latest_visualization_paths)
    write_json(summary, output_dir / "workflow_run_summary.json")
    if bool(output_config.get("auto_build_console", True)):
        console_dir = workflow_console_output_dir_from_config(
            config,
            run_output_dir=output_dir,
        )
        console_summary = build_workflow_console(console_dir, output_dir)
        summary["artifact_paths"]["workflow_console_html"] = console_summary.get(
            "html_path"
        )
        summary["artifact_paths"]["workflow_console_summary_json"] = (
            console_summary.get("summary_path")
        )
        if bool(output_config.get("write_latest_alias", True)):
            latest_console_summary = build_workflow_console(
                _PROJECT_ROOT / "outputs" / "workflow_console",
                output_dir,
            )
            summary["artifact_paths"]["latest_workflow_console_html"] = (
                latest_console_summary.get("html_path")
            )
            summary["artifact_paths"]["latest_workflow_console_summary_json"] = (
                latest_console_summary.get("summary_path")
        )
        write_json(summary, output_dir / "workflow_run_summary.json")
    from data_collection_workflow.reporting.run_settings import sanitize_configuration
    from data_collection_workflow.reporting.output_contract import retire_legacy_reading_files, without_legacy_reading_links, write_report_shortcut
    from data_collection_workflow.reporting.unified_report import write_unified_report
    write_json(sanitize_configuration(config), output_dir / "run_config.json")
    summary["run_config_path"] = str(output_dir / "run_config.json")
    summary["run_status"] = dict(event_writer.run_status)
    report_state = {**result, "runtime_profile": {"env": report_environment},
                    "run_status": dict(event_writer.run_status),
                    "execution_options": {key: getattr(args, key, None) for key in (
                        "live_status", "write_run_notebook", "resume_session", "provider_resume", "budget_amendment")}}
    if universal:
        from data_collection_workflow.result_manifest import write_universal_run_outputs
        write_universal_run_outputs(package, summary, output_dir, config=config, state=report_state)
    else:
        retire_legacy_reading_files(output_dir)
        summary["artifact_paths"] = without_legacy_reading_links(summary["artifact_paths"])
        summary["artifact_paths"].update(write_unified_report(output_dir, package, summary, config=config, state=report_state))
    if bool(output_config.get("write_latest_alias", True)):
        summary["artifact_paths"]["stable_final_report_english"] = write_report_shortcut(
            _TOP_LEVEL_UNIFIED_REPORT_PATH, summary["artifact_paths"]["final_report_english"])
    write_json(summary, output_dir / "workflow_run_summary.json")
    for artifact_key in (
        "final_report_english",
        "report_bundle",
        "workflow_console_html",
        "workflow_console_summary_json",
        "workflow_visualization_index",
        "final_report_facts",
    ):
        artifact_path = summary["artifact_paths"].get(artifact_key)
        if artifact_path:
            event_writer.append_artifact_written(artifact_key, artifact_path)
    event_writer.append_artifact_written(
        "workflow_run_summary",
        output_dir / "workflow_run_summary.json",
    )
    if bool(getattr(args, "write_run_notebook", False)):
        notebook_path = write_workflow_replay_notebook(output_dir)
        summary["artifact_paths"]["workflow_replay_notebook"] = str(notebook_path)
        summary["workflow_replay_notebook"] = str(notebook_path)
        write_json(summary, output_dir / "workflow_run_summary.json")
        event_writer.append_artifact_written("workflow_replay_notebook", notebook_path)
    if bool(output_config.get("write_latest_alias", True)):
        write_json(summary, _TOP_LEVEL_SUMMARY_PATH)
    return summary


def _print_config(args: argparse.Namespace) -> None:
    config_path, config = _config_with_cli_overrides(args)
    provider, model = _config_provider_model(config)
    env_updates = workflow_run_env_from_config(config)
    print(f"config_path: {_console_text(config_path)}")
    print(f"provider: {provider}")
    print(f"model: {model}")
    print(f"api_key_present: {api_key_present(provider)}")
    print("sanitized_environment:")
    print(json.dumps(safe_env_for_display(env_updates), indent=2, sort_keys=True))
    print("studio_minimal_input:")
    print(
        json.dumps(
            workflow_initial_state_from_config(config, include_empty_fields=False),
            indent=2,
        )
    )


def main() -> int:
    args = _build_parser().parse_args()
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        print("--timeout-seconds must be positive.", file=sys.stderr)
        return 2
    if args.llm_max_chunks is not None and args.llm_max_chunks <= 0:
        print("--llm-max-chunks must be positive.", file=sys.stderr)
        return 2
    try:
        _, config = _config_with_cli_overrides(args)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    env_updates = workflow_run_env_from_config(config)
    provider, model = _config_provider_model(config)
    if args.print_config_only:
        _print_config(args)
        return 0

    if _llm_enabled(env_updates) and not model:
        print("Config llm.model or --model is required.", file=sys.stderr)
        return 3
    if _llm_enabled(env_updates) and not api_key_present(provider):
        print(
            f"Missing API key for provider '{provider}'.",
            file=sys.stderr,
        )
        return 4

    summary = run_workflow(args)
    llm = summary.get("llm_stage_summary") or {}
    print("=" * 72)
    print("Data Collection Workflow run completed.")
    print(f"output_dir: {_console_text(summary.get('output_dir'))}")
    artifact_paths = summary.get("artifact_paths") or {}
    if artifact_paths.get("report_bundle"):
        print(f"report_bundle: {_console_text(artifact_paths['report_bundle'])}")
    if artifact_paths.get("final_report_english"):
        print(
            "final_report_english:",
            _console_text(artifact_paths.get("final_report_english")),
        )
    if artifact_paths.get("task_result_english"):
        print("task_result_english:", _console_text(artifact_paths.get("task_result_english")))
    task_summary = summary.get("task_result_summary") or {}
    if task_summary.get("headline"):
        print("task_result:", _console_text(task_summary["headline"]))
    if artifact_paths.get("workflow_console_html"):
        print(
            "workflow_console:",
            _console_text(artifact_paths.get("workflow_console_html")),
        )
    if artifact_paths.get("interpretive_report_english"):
        print(
            "interpretive_report_english:",
            _console_text(artifact_paths.get("interpretive_report_english")),
        )
    print(f"provider: {summary.get('provider')}")
    print(f"model: {summary.get('model')}")
    print(f"trace_node_count: {summary.get('trace_node_count')}")
    print(f"current_route: {summary.get('current_route')}")
    print(f"document_count: {summary.get('document_count')}")
    source_search = summary.get("source_search_execution_summary") or {}
    print(f"source_search_mode: {summary.get('source_search_mode')}")
    print(f"source_search_provider: {summary.get('source_search_provider')}")
    print(
        "source_search_executed_query_count:",
        source_search.get("executed_query_count"),
    )
    print(
        "search_derived_candidate_count:",
        source_search.get("candidate_from_search_count"),
    )
    iterative_search = summary.get("iterative_source_discovery_summary") or {}
    print(
        "iterative_source_discovery_enabled:",
        iterative_search.get("iterative_source_discovery_enabled"),
    )
    print(
        "iterative_search_iteration_count:",
        iterative_search.get("search_iteration_count"),
    )
    print("iterative_search_stop_decision:", iterative_search.get("stop_decision"))
    print(f"normalized_record_count: {summary.get('normalized_record_count')}")
    print(f"run_quality_status: {summary.get('run_quality_status')}")
    print(f"final_dataset_count: {summary.get('final_dataset_count')}")
    print(f"final_case_dataset_count: {summary.get('final_case_dataset_count')}")
    print(f"zero_case_statement_count: {summary.get('zero_case_statement_count')}")
    print(
        "exposure_monitoring_record_count:",
        summary.get("exposure_monitoring_record_count"),
    )
    print(f"context_record_count: {summary.get('context_record_count')}")
    print(
        "surveillance_summary_record_count:",
        summary.get("surveillance_summary_record_count"),
    )
    print(
        "outbreak_summary_record_count:",
        summary.get("outbreak_summary_record_count"),
    )
    print(
        "unclassified_observation_count:",
        summary.get("unclassified_observation_count"),
    )
    print(f"primary_case_dataset_status: {summary.get('primary_case_dataset_status')}")
    print(
        "primary_case_dataset_eligible_count:",
        summary.get("primary_case_dataset_eligible_count"),
    )
    print(
        "corroborated_primary_case_event_count:",
        summary.get("corroborated_primary_case_event_count"),
    )
    print(
        "non_primary_observation_count:",
        summary.get("non_primary_observation_count"),
    )
    print(f"quarantined_record_count: {summary.get('quarantined_record_count')}")
    print(f"pending_review_record_count: {summary.get('pending_review_record_count')}")
    print(f"evaluation_row_count: {summary.get('evaluation_row_count')}")
    print(f"human_review_item_count: {summary.get('human_review_item_count')}")
    print(
        "llm_source_planning_status:",
        (llm.get("source_planning") or {}).get("status"),
    )
    print(
        "llm_source_critic_assessed_source_count:",
        (llm.get("source_critic") or {}).get("assessed_source_count"),
    )
    print(
        "source_credibility_assessed_source_count:",
        (llm.get("source_credibility") or {}).get("assessed_source_count"),
    )
    print(
        "source_identity_assessed_source_count:",
        (llm.get("source_identity") or {}).get("identity_assessed_count"),
    )
    print(
        "llm_source_identity_assessed_source_count:",
        (llm.get("source_identity") or {}).get("llm_identity_assessed_count"),
    )
    print(
        "llm_structured_extraction_call_count:",
        (llm.get("structured_extraction") or {}).get("call_count"),
    )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
