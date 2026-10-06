"""User-facing command line interface for the data collection workflow."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from .export import export_final_data_package, write_csv_rows, write_json
from .search_providers import search_api_key_present
from .workflow_run_config import (
    DEFAULT_WORKFLOW_RUN_CONFIG_PATH,
    DEFAULT_MODEL,
    PROJECT_ROOT,
    api_key_present,
    load_workflow_run_config,
    resolve_workflow_run_config_path,
    safe_env_for_display,
    validate_workflow_task,
    workflow_initial_state_from_config,
    workflow_output_dir_from_config,
    workflow_run_config_with_overrides,
    workflow_run_env_from_config,
)


SECRET_VALUE_PREFIXES = ("sk-", "sk_ant", "sk-ant", "tvly-", "tavily-")
DEFAULT_EXPORT_SECTIONS = {
    "final_dataset": "final_dataset",
    "final_dataset_post_review": "final_dataset_post_review",
    "source_registry": "source_registry",
    "validation_results": "validation_results",
    "event_clusters": "event_clusters",
    "duplicate_clusters": "duplicate_clusters",
    "anomaly_results": "anomaly_results",
    "applied_human_review_decisions": "applied_human_review_decisions",
    "rejected_human_review_decisions": "rejected_human_review_decisions",
    "human_review_audit_trail": "human_review_audit_trail",
    "human_review_application_summary": "human_review_application_summary",
    "collection_trace": "collection_trace",
}


def _json_dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True)


def _console_text(value: Any) -> str:
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def _resolve_path(path: str | Path | None, *, default: Path | None = None) -> Path:
    resolved = Path(path) if path else Path(default or ".")
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def _coerce_bool(value: bool | None) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "***REDACTED***" if _looks_secret_key(key) else _redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        if any(lowered.startswith(prefix) for prefix in SECRET_VALUE_PREFIXES):
            return "***REDACTED***"
    return value


def _looks_secret_key(key: str) -> bool:
    lowered = key.lower()
    return (
        "api_key" in lowered
        or "apikey" in lowered
        or "secret" in lowered
        or lowered in {"token", "access_token", "refresh_token", "bearer_token"}
        or lowered.endswith("_token")
    )


def _contains_embedded_secret(value: Any, path: str = "config") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if _looks_secret_key(str(key)) and item not in (None, "", [], {}):
                findings.append(item_path)
            findings.extend(_contains_embedded_secret(item, item_path))
        return findings
    if isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(_contains_embedded_secret(item, f"{path}[{index}]"))
        return findings
    if isinstance(value, str):
        lowered = value.strip().lower()
        if any(lowered.startswith(prefix) for prefix in SECRET_VALUE_PREFIXES):
            findings.append(path)
    return findings


def _provider_model(config: dict) -> tuple[str, str]:
    llm = config.get("llm") or {}
    return str(llm.get("provider") or ""), str(llm.get("model") or "")


def _override_from_flags(enable: bool, disable: bool, name: str) -> bool | None:
    if enable and disable:
        raise ValueError(f"{name}: enable and disable flags cannot both be set.")
    if enable:
        return True
    if disable:
        return False
    return None


def _apply_cli_config_overrides(config: dict, args: argparse.Namespace) -> dict:
    updated = workflow_run_config_with_overrides(
        config,
        live_fetch=_override_from_flags(
            getattr(args, "enable_live_fetch", False),
            getattr(args, "disable_live_fetch", False),
            "live fetch",
        ),
        all_llm=_override_from_flags(
            getattr(args, "enable_all_llm", False),
            getattr(args, "disable_all_llm", False),
            "LLM",
        ),
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        timeout_seconds=getattr(args, "timeout_seconds", None),
        llm_max_chunks=getattr(args, "llm_max_chunks", None),
        output_dir=getattr(args, "output_dir", None),
        session_id=getattr(args, "session_id", None),
        user_request=getattr(args, "user_request", None),
    )

    structured = updated.setdefault("structured_task", {})
    structured_override_requested = False
    for field in ("disease", "location", "start_date", "end_date"):
        value = getattr(args, field, None)
        if value:
            structured[field] = value
            structured_override_requested = True
    if getattr(args, "target_field", None):
        structured["target_fields"] = list(args.target_field)
        structured_override_requested = True

    if structured:
        disease = structured.get("disease")
        location = structured.get("location")
        start_date = structured.get("start_date")
        end_date = structured.get("end_date")
        existing_request = updated.get("user_request") or structured.get("user_request")
        should_generate_request = (
            not getattr(args, "user_request", None)
            and disease
            and location
            and start_date
            and end_date
            and (structured_override_requested or not existing_request)
        )
        if should_generate_request:
            request = (
                f"Collect {disease} data for {location} from {start_date} to "
                f"{end_date}, extract the requested fields, preserve source "
                "provenance, validate against reserved sources, and route "
                "uncertain results to human review."
            )
            updated["user_request"] = request
            structured["user_request"] = request

    search = updated.setdefault("source_search", {})
    search_mode = getattr(args, "search_mode", None)
    if search_mode:
        if search_mode == "disabled":
            search["enabled"] = False
            search["mode"] = "disabled"
        else:
            search["enabled"] = True
            search["mode"] = search_mode
            if search_mode == "fixture":
                search.setdefault("provider", "fixture")
            if search_mode == "live":
                search.setdefault("provider", "tavily")
                search["live_search_enabled"] = True
    live_search = _override_from_flags(
        getattr(args, "enable_live_search", False),
        getattr(args, "disable_live_search", False),
        "live search",
    )
    if live_search is not None:
        search["enabled"] = bool(live_search)
        search["mode"] = "live" if live_search else "disabled"
        search["live_search_enabled"] = bool(live_search)
        if live_search:
            search.setdefault("provider", "tavily")
    if getattr(args, "search_provider", None):
        search["provider"] = args.search_provider

    content_fetch = updated.setdefault("content_fetch", {})
    fetch_search_derived = _coerce_bool(getattr(args, "fetch_search_derived_sources", None))
    if fetch_search_derived is not None:
        content_fetch["fetch_search_derived_sources"] = fetch_search_derived

    human_review = updated.setdefault("human_review", {})
    if getattr(args, "human_review_decisions_path", None):
        human_review["decisions_path"] = str(args.human_review_decisions_path)
    if getattr(args, "apply_review_decisions", False):
        human_review["apply_decisions"] = True

    return updated


def _print_safe_runtime_preview(config_path: Path, config: dict) -> None:
    provider, model = _provider_model(config)
    env_updates = workflow_run_env_from_config(config)
    source_search = config.get("source_search") or {}
    print("data collection workflow configuration preview")
    print(f"config_path: {_console_text(config_path)}")
    print(f"provider: {provider}")
    print(f"model: {model}")
    print(f"api_key_present: {api_key_present(provider)}")
    print(f"source_search_api_key_present: {search_api_key_present(str(source_search.get('provider') or ''))}")
    print("sanitized_environment:")
    print(_json_dump(safe_env_for_display(env_updates)))
    print("structured_task:")
    print(_json_dump(_redact_value(config.get("structured_task") or {})))
    print("studio_minimal_input:")
    print(
        _json_dump(
            _redact_value(
                workflow_initial_state_from_config(
                    config,
                    include_empty_fields=False,
                )
            )
        )
    )


def _load_and_override_config(args: argparse.Namespace) -> tuple[Path, dict]:
    config_path = resolve_workflow_run_config_path(getattr(args, "config", None))
    config = load_workflow_run_config(config_path)
    config = _apply_cli_config_overrides(config, args)
    return config_path, config


def _write_applied_config(output_dir: Path, config_path: Path, config: dict) -> None:
    payload = {
        "project_name": "data collection workflow",
        "config_path": str(config_path),
        "environment": safe_env_for_display(workflow_run_env_from_config(config)),
        "config": _redact_value(config),
    }
    write_json(payload, output_dir / "applied_workflow_config.json")


def _call_configured_runner(
    config_path: Path,
    *,
    live_status: bool = True,
    write_run_notebook: bool = False,
) -> dict:
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from run_workflow import run_workflow  # noqa: PLC0415

    runner_args = argparse.Namespace(
        config=str(config_path),
        enable_live_fetch=False,
        disable_live_fetch=False,
        enable_all_llm=False,
        disable_all_llm=False,
        provider=None,
        model=None,
        timeout_seconds=None,
        llm_max_chunks=None,
        output_dir=None,
        session_id=None,
        user_request=None,
        print_config_only=False,
        live_status=live_status,
        write_run_notebook=write_run_notebook,
    )
    return run_workflow(runner_args)


def cmd_collect(args: argparse.Namespace) -> int:
    config_path, config = _load_and_override_config(args)
    if args.dry_run or args.print_config_only:
        _print_safe_runtime_preview(config_path, config)
        if args.dry_run:
            print("dry_run: true")
            print("graph_invoked: false")
        return 0

    try:
        validate_workflow_task(config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    env_updates = workflow_run_env_from_config(config)
    provider, model = _provider_model(config)
    llm_enabled = any(
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
    if llm_enabled and not model:
        print("Config llm.model or --model is required when LLM stages are enabled.", file=sys.stderr)
        return 3
    if llm_enabled and not api_key_present(provider):
        print(f"Missing API key for provider '{provider}'.", file=sys.stderr)
        return 4

    search = config.get("source_search") or {}
    if bool(search.get("enabled")) and str(search.get("mode") or "").lower() == "live":
        search_provider = str(search.get("provider") or "tavily")
        if not search_api_key_present(search_provider):
            print(f"Missing search API key for provider '{search_provider}'.", file=sys.stderr)
            return 5

    with tempfile.TemporaryDirectory(prefix="data_collection_workflow_cli_") as temp_dir:
        applied_path = Path(temp_dir) / "applied_config.json"
        applied_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = _call_configured_runner(
            applied_path,
            live_status=bool(getattr(args, "live_status", True)),
            write_run_notebook=bool(getattr(args, "write_run_notebook", False)),
        )

    output_dir = Path(str(summary.get("output_dir")))
    _write_applied_config(output_dir, config_path, config)
    print("data collection workflow run completed.")
    print(f"output_dir: {_console_text(summary.get('output_dir'))}")
    artifacts = summary.get("artifact_paths") or {}
    for label in (
        "run_report",
        "workflow_console_html",
        "workflow_console_summary_json",
    ):
        if artifacts.get(label):
            print(f"{label}: {_console_text(artifacts[label])}")
    print(f"source_search_mode: {summary.get('source_search_mode')}")
    print(f"source_search_provider: {summary.get('source_search_provider')}")
    print(f"normalized_record_count: {summary.get('normalized_record_count')}")
    print(f"anomaly_result_count: {summary.get('anomaly_result_count')}")
    print(f"human_review_item_count: {summary.get('human_review_item_count')}")
    return 0


def _config_validation_issues(config: dict) -> list[str]:
    issues: list[str] = []
    structured = config.get("structured_task") or {}
    if not isinstance(structured, dict):
        issues.append("structured_task must be a JSON object.")
    else:
        for field in ("disease", "location", "start_date", "end_date"):
            if not structured.get(field):
                issues.append(f"structured_task.{field} is recommended.")
        if not structured.get("target_fields"):
            issues.append("structured_task.target_fields is recommended.")

    search = config.get("source_search") or {}
    if search.get("enabled") and not search.get("provider"):
        issues.append("source_search.provider is required when source search is enabled.")
    if str(search.get("mode") or "").lower() == "fixture" and not search.get("fixture_path"):
        issues.append("source_search.fixture_path is required for fixture mode.")

    human_review = config.get("human_review") or {}
    decisions_path = human_review.get("decisions_path")
    if human_review.get("apply_decisions") and decisions_path:
        if not _resolve_path(decisions_path).exists():
            issues.append(f"human_review.decisions_path does not exist: {decisions_path}")

    embedded = _contains_embedded_secret(config)
    if embedded:
        issues.append(
            "config appears to contain embedded secret values at: "
            + ", ".join(sorted(set(embedded)))
        )
    return issues


def cmd_validate_config(args: argparse.Namespace) -> int:
    try:
        config_path, config = _load_and_override_config(args)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print("valid: false")
        print(f"error: {exc}")
        return 2

    issues = _config_validation_issues(config)
    source_search = config.get("source_search") or {}
    provider, model = _provider_model(config)
    print(f"valid: {'false' if issues else 'true'}")
    print(f"config_path: {_console_text(config_path)}")
    print(f"project_name: data collection workflow")
    print(f"disease: {(config.get('structured_task') or {}).get('disease')}")
    print(f"location: {(config.get('structured_task') or {}).get('location')}")
    print(f"start_date: {(config.get('structured_task') or {}).get('start_date')}")
    print(f"end_date: {(config.get('structured_task') or {}).get('end_date')}")
    print(f"llm_provider: {provider}")
    print(f"llm_model: {model}")
    print(f"api_key_present: {api_key_present(provider)}")
    print(f"live_search_enabled: {bool(source_search.get('enabled')) and str(source_search.get('mode') or '').lower() == 'live'}")
    print(f"source_search_provider: {source_search.get('provider')}")
    print(f"source_search_api_key_present: {search_api_key_present(str(source_search.get('provider') or ''))}")
    if issues:
        print("issues:")
        for issue in issues:
            print(f"- {issue}")
        return 2
    return 0


def _load_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_session_package(session_dir: Path) -> tuple[dict, dict]:
    summary = _load_json_if_exists(session_dir / "workflow_run_summary.json", {})
    package = _load_json_if_exists(session_dir / "collection" / "final_package.json", {})
    return summary, package


def cmd_inspect_run(args: argparse.Namespace) -> int:
    session_dir = _resolve_path(args.session_dir)
    if not session_dir.exists():
        print(f"session_dir not found: {_console_text(session_dir)}", file=sys.stderr)
        return 2
    summary, package = _load_session_package(session_dir)
    source_search = summary.get("source_search_execution_summary") or {}
    rows = {
        "session_dir": _console_text(session_dir),
        "output_dir": _console_text(summary.get("output_dir") or str(session_dir)),
        "source_count": summary.get("source_registry_count", len(package.get("source_registry") or [])),
        "search_derived_source_count": source_search.get("candidate_from_search_count", 0),
        "document_count": summary.get("document_count", 0),
        "final_dataset_count": len(package.get("final_dataset") or []),
        "final_dataset_post_review_count": len(package.get("final_dataset_post_review") or []),
        "normalized_record_count": summary.get("normalized_record_count", 0),
        "event_cluster_count": len(package.get("event_clusters") or []),
        "validation_result_count": summary.get("validation_result_count", len(package.get("validation_results") or [])),
        "anomaly_count": summary.get("anomaly_result_count", len(package.get("anomaly_results") or [])),
        "human_review_item_count": summary.get("human_review_item_count", len(package.get("human_review_items") or [])),
        "decisions_applied_count": summary.get("human_review_decisions_applied_count", 0),
        "decisions_rejected_count": summary.get("human_review_decisions_rejected_count", 0),
        "current_route": summary.get("current_route"),
    }
    for key, value in rows.items():
        print(f"{key}: {value}")
    artifacts = summary.get("artifact_paths") or {}
    for key, value in sorted(artifacts.items()):
        if isinstance(value, str) and value:
            print(f"artifact.{key}: {_console_text(value)}")
    return 0


def cmd_review_summary(args: argparse.Namespace) -> int:
    session_dir = _resolve_path(args.session_dir)
    if not session_dir.exists():
        print(f"session_dir not found: {_console_text(session_dir)}", file=sys.stderr)
        return 2
    summary, package = _load_session_package(session_dir)
    items = list(package.get("human_review_items") or [])
    applied = list(package.get("applied_human_review_decisions") or [])
    rejected = list(package.get("rejected_human_review_decisions") or [])
    audit = list(package.get("human_review_audit_trail") or [])
    print(f"session_dir: {_console_text(session_dir)}")
    print(f"human_review_item_count: {len(items)}")
    print(f"decisions_applied_count: {len(applied) or summary.get('human_review_decisions_applied_count', 0)}")
    print(f"decisions_rejected_count: {len(rejected) or summary.get('human_review_decisions_rejected_count', 0)}")
    print(f"audit_trail_count: {len(audit) or summary.get('human_review_audit_entry_count', 0)}")
    type_counts: dict[str, int] = {}
    for item in items:
        item_type = str(item.get("item_type") or "unknown")
        type_counts[item_type] = type_counts.get(item_type, 0) + 1
    print("item_type_counts:")
    print(_json_dump(type_counts))
    return 0


def _write_export_section(name: str, data: Any, output_dir: Path, export_format: str) -> None:
    if export_format in {"json", "both"}:
        write_json(data, output_dir / f"{name}.json")
    if export_format in {"csv", "both"}:
        if isinstance(data, list):
            rows = [row for row in data if isinstance(row, dict)]
        elif isinstance(data, dict):
            rows = [data]
        else:
            rows = []
        write_csv_rows(rows, output_dir / f"{name}.csv")


def cmd_export(args: argparse.Namespace) -> int:
    session_dir = _resolve_path(args.session_dir)
    output_dir = _resolve_path(args.output_dir)
    if not session_dir.exists():
        print(f"session_dir not found: {_console_text(session_dir)}", file=sys.stderr)
        return 2
    _, package = _load_session_package(session_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.format in {"json", "both"}:
        export_final_data_package(package, output_dir)
    else:
        for section, key in DEFAULT_EXPORT_SECTIONS.items():
            _write_export_section(section, package.get(key) or [], output_dir, "csv")

    for section, key in DEFAULT_EXPORT_SECTIONS.items():
        data = package.get(key)
        if data not in (None, [], {}):
            _write_export_section(section, data, output_dir, args.format)

    source_console = session_dir / "workflow_console" / "data_collection_workflow_console.html"
    if source_console.exists():
        shutil.copy2(source_console, output_dir / "data_collection_workflow_console.html")
    print(f"export_output_dir: {_console_text(output_dir)}")
    print(f"format: {args.format}")
    return 0


def _fixture_for_disease(disease: str) -> tuple[str, str, str]:
    key = disease.strip().lower()
    if "covid" in key:
        return (
            "src/data_collection_workflow/resources/search_fixtures/covid19_new_york_search_results.json",
            "src/data_collection_workflow/resources/content_fixtures/content_map.json",
            "src/data_collection_workflow/resources/human_review_decision_fixtures/covid19_review_decisions.json",
        )
    if "dengue" in key:
        return (
            "src/data_collection_workflow/resources/search_fixtures/dengue_florida_search_results.json",
            "src/data_collection_workflow/resources/content_fixtures/content_map.json",
            "src/data_collection_workflow/resources/human_review_decision_fixtures/dengue_review_decisions.json",
        )
    return (
        "src/data_collection_workflow/resources/search_fixtures/example_search_results.json",
        "src/data_collection_workflow/resources/content_fixtures/content_map.json",
        "",
    )


def _template_text(args: argparse.Namespace) -> str:
    disease = args.disease
    location = args.location
    start_date = args.start_date
    end_date = args.end_date
    target_fields = args.target_field or ["cases_unspecified", "deaths", "date_reported", "source_url", "evidence_quote"]
    fixture_path, fixture_map, decision_path = _fixture_for_disease(disease)
    live_search = args.mode == "live-search"
    fixture_search = args.mode == "fixture-search"
    disabled_search = args.mode == "offline"
    source_provider = "tavily" if live_search else "fixture"
    search_mode = "live" if live_search else "fixture" if fixture_search else "disabled"
    config = f"""{{
  // data collection workflow runtime profile.
  // Put real API keys in environment variables only:
  //   TAVILY_API_KEY for live source search metadata.
  //   ANTHROPIC_API_KEY or OPENAI_API_KEY for optional LLM stages.
  "profile_name": "{disease.lower().replace(' ', '_')}_{location.lower().replace(' ', '_')}_{start_date}_{end_date}",
  "description": "Generated config template for the data collection workflow.",
  "workflow": {{
    "collection_mode": "standard",
    "use_fixture_documents": false
  }},
  "user_request": "Collect {disease} data for {location} from {start_date} to {end_date}.",
  "structured_task": {{
    "disease": "{disease}",
    "location": "{location}",
    "start_date": "{start_date}",
    "end_date": "{end_date}",
    "target_fields": {_json_dump(target_fields)},
    "source_preferences": [
      "official_public_health_agency",
      "international_organization_report",
      "structured_database"
    ],
    "collection_mode": "standard",
    "user_request": "Collect {disease} data for {location} from {start_date} to {end_date}.",
    "run_label": "{disease.lower().replace(' ', '_')}_{location.lower().replace(' ', '_')}_{start_date}_{end_date}"
  }},
  "live_web": {{
    "enabled": {str(live_search).lower()},
    "timeout_seconds": 30
  }},
  "source_search": {{
    "enabled": {str(not disabled_search).lower()},
    "mode": "{search_mode}",
    "provider": "{source_provider}",
    "fixture_path": "{fixture_path}",
    "max_queries": 3,
    "max_results_per_query": 5,
    "max_total_results": 15,
    "timeout_seconds": 15,
    "combine_with_seed_catalog": {str(not fixture_search).lower()},
    "cache_enabled": true
  }},
  "content_fetch": {{
    "fetch_search_derived_sources": {str(live_search or fixture_search).lower()},
    "max_search_derived_sources": 1,
    "max_total_sources": 2,
    "min_credibility_score": 0.55,
    "allowed_final_roles": [
      "collection",
      "collection_support",
      "context"
    ],
    "allow_needs_review": false,
    "domain_allowlist": [],
    "content_fixture_map_path": "{fixture_map}"
  }},
  "llm": {{
    "provider": "",
    "model": "{DEFAULT_MODEL}",
    "source_planning_enabled": false,
    "source_critic_enabled": false,
    "structured_extraction_enabled": false,
    "source_credibility": {{
      "enabled": false
    }},
    "source_identity": {{
      "enabled": false,
      "require_llm": false,
      "allow_deterministic_fallback": true
    }}
  }},
  "disease_intelligence": {{
    "llm_enabled": false,
    "force_llm": false,
    "fallback_to_curated": true
  }},
  "anomaly_detection": {{
    "enabled": true,
    "max_cases_threshold": 1000000,
    "max_deaths_threshold": 100000,
    "spike_multiplier": 10,
    "min_prior_records": 1
  }},
  "human_review": {{
    "decisions_path": {"null" if not decision_path else _json_dump(decision_path)},
    "apply_decisions": {str(bool(decision_path and fixture_search)).lower()},
    "require_reviewer_id": true
  }},
  "source_sets": {{
    "source_id_allowlist_enabled": false,
    "workflow_source_ids": []
  }},
  "output": {{
    "run_output_root": "outputs",
    "sessionized": true,
    "session_id": null,
    "auto_build_console": true,
    "write_latest_alias": true
  }}
}}
"""
    return config


def cmd_init_config(args: argparse.Namespace) -> int:
    output = _resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_template_text(args), encoding="utf-8")
    print(f"config_written: {_console_text(output)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data-collection-workflow",
        description="Run and inspect the data collection workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Run the configured workflow.")
    collect.add_argument("--config", default=str(DEFAULT_WORKFLOW_RUN_CONFIG_PATH))
    collect.add_argument("--disease")
    collect.add_argument("--location")
    collect.add_argument("--start-date")
    collect.add_argument("--end-date")
    collect.add_argument("--target-field", action="append", default=[])
    collect.add_argument("--user-request")
    collect.add_argument("--output-dir")
    collect.add_argument("--session-id")
    collect.add_argument("--provider")
    collect.add_argument("--model")
    collect.add_argument("--timeout-seconds", type=float)
    collect.add_argument("--llm-max-chunks", type=int)
    collect.add_argument("--enable-live-fetch", action="store_true")
    collect.add_argument("--disable-live-fetch", action="store_true")
    collect.add_argument("--enable-live-search", action="store_true")
    collect.add_argument("--disable-live-search", action="store_true")
    collect.add_argument("--search-mode", choices=["disabled", "fixture", "live"])
    collect.add_argument("--search-provider")
    collect.add_argument("--fetch-search-derived-sources", dest="fetch_search_derived_sources", action="store_true")
    collect.add_argument("--no-fetch-search-derived-sources", dest="fetch_search_derived_sources", action="store_false")
    collect.set_defaults(fetch_search_derived_sources=None)
    collect.add_argument("--enable-all-llm", action="store_true")
    collect.add_argument("--disable-all-llm", action="store_true")
    collect.add_argument("--human-review-decisions-path")
    collect.add_argument("--apply-review-decisions", action="store_true")
    collect.add_argument("--dry-run", action="store_true")
    collect.add_argument("--print-config-only", action="store_true")
    collect.add_argument("--live-status", dest="live_status", action="store_true", default=True)
    collect.add_argument("--no-live-status", dest="live_status", action="store_false")
    collect.add_argument("--write-run-notebook", action="store_true")
    collect.set_defaults(func=cmd_collect)

    validate = subparsers.add_parser("validate-config", help="Validate a workflow config.")
    validate.add_argument("--config", required=True)
    validate.set_defaults(
        disease=None,
        location=None,
        start_date=None,
        end_date=None,
        target_field=[],
        user_request=None,
        output_dir=None,
        session_id=None,
        provider=None,
        model=None,
        timeout_seconds=None,
        llm_max_chunks=None,
        enable_live_fetch=False,
        disable_live_fetch=False,
        enable_all_llm=False,
        disable_all_llm=False,
        enable_live_search=False,
        disable_live_search=False,
        search_mode=None,
        search_provider=None,
        fetch_search_derived_sources=None,
        human_review_decisions_path=None,
        apply_review_decisions=False,
        func=cmd_validate_config,
    )

    inspect_run = subparsers.add_parser("inspect-run", help="Summarize an existing run session.")
    inspect_run.add_argument("--session-dir", required=True)
    inspect_run.set_defaults(func=cmd_inspect_run)

    review = subparsers.add_parser("review-summary", help="Summarize human review queue and decisions.")
    review.add_argument("--session-dir", required=True)
    review.set_defaults(func=cmd_review_summary)

    export = subparsers.add_parser("export", help="Export selected run artifacts.")
    export.add_argument("--session-dir", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--format", choices=["json", "csv", "both"], default="both")
    export.set_defaults(func=cmd_export)

    init = subparsers.add_parser("init-config", help="Create a safe config template.")
    init.add_argument("--disease", required=True)
    init.add_argument("--location", required=True)
    init.add_argument("--start-date", required=True)
    init.add_argument("--end-date", required=True)
    init.add_argument("--target-field", action="append", default=[])
    init.add_argument("--mode", choices=["offline", "fixture-search", "live-search"], default="offline")
    init.add_argument("--output", required=True)
    init.set_defaults(func=cmd_init_config)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
