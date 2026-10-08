"""Build a portable settings inventory from supplied session evidence only.

Current reference defaults are packaged metadata, never inferred run values.
This module does not load a project configuration, .env, or process environment.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from importlib.resources import files
import json
from pathlib import PurePosixPath, PureWindowsPath
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_MISSING = object()
_REDACTED = "[redacted]"
_DESCRIPTOR = json.loads(
    files("data_collection_workflow").joinpath("resources/report_run_settings.json").read_text(encoding="utf-8")
)
_SCHEMA = {row["key"]: row for row in _DESCRIPTOR["settings"]}
_ENV_KEYS = {
    alias: row["key"]
    for row in _DESCRIPTOR["settings"]
    if not row["key"].startswith("credentials.")
    for alias in row["environment_variables"]
}
# The live runner may capture these named operational settings while its scoped
# environment is active. Credential names and unrelated system keys are excluded.
REPORT_ENVIRONMENT_VARIABLES = tuple(sorted(_ENV_KEYS))
_CLI_KEYS = {
    alias.split(" ", 1)[0].removeprefix("--").replace("-", "_"): row["key"]
    for row in _DESCRIPTOR["settings"]
    for alias in row["cli_aliases"]
}
_SUMMARY_KEYS = {
    "pipeline_mode": "pipeline_mode",
    "run_mode": "run_mode",
    "provider": "llm.provider",
    "model": "llm.model",
    "live_fetch_enabled": "live_web.enabled",
    "live_search_enabled": "source_search.enabled",
    "external_fetch_enabled": "content_fetch.external_fetch.enabled",
    "external_fetch_provider_order": "content_fetch.external_fetch.provider_order",
    "human_review_enabled": "human_review.enabled",
    "source_search_mode": "source_search.mode",
    "source_search_provider": "source_search.provider",
    "fixture_documents_enabled": "workflow.use_fixture_documents",
    "config_path": "cli.config",
    "case_study_real_mode": "cli.case_study_real_mode",
    "quick_test_mode": "cli.quick_test_mode",
}


def _secret_key(key: str) -> bool:
    """Recognize mixed-case credential names without hiding max_tokens."""
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key)).lower()
    parts = re.split(r"[^a-z0-9]+", words)
    compact = "".join(parts)
    # Exempt only the token-limit suffix, never a credential-bearing ancestor.
    if parts[-1:] == ["tokens"] and len(parts) > 1 and parts[-2] in {"max", "min", "output", "input", "total"}:
        parts = parts[:-1]
    if any(word in parts for word in ("password", "passwd", "secret", "secrets", "auth", "authorization", "credential", "credentials", "token", "tokens", "bearer", "cookie", "cookies")):
        return True
    return any(word in compact for word in ("apikey", "accesskey", "privatekey", "clientsecret", "accesstoken", "refreshtoken", "idtoken", "sessiontoken"))


def _safe_url(match: re.Match) -> str:
    raw = match.group(0)
    try:
        url = urlsplit(raw)
        authority = url.netloc.rsplit("@", 1)[-1]

        def clean_query(query):
            return urlencode([
                (name, _REDACTED if _secret_key(name) or name.lower() in {"key", "sig", "signature", "auth", "jwt"} else _safe(value))
                for name, value in parse_qsl(query, keep_blank_values=True)
            ])

        fragment = clean_query(url.fragment) if "=" in url.fragment else url.fragment
        return urlunsplit((url.scheme, authority, url.path, clean_query(url.query), fragment))
    except ValueError:
        # A malformed URL cannot safely retain potentially embedded credentials.
        return "[redacted URL]"


def _safe(value, *, short_paths=False):
    if isinstance(value, Mapping):
        return {
            str(key): _REDACTED if _secret_key(str(key)) else _safe(item, short_paths=short_paths)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe(item, short_paths=short_paths) for item in value]
    if isinstance(value, str):
        value = re.sub(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s<>\"']+", _safe_url, value)
        if short_paths:
            if PureWindowsPath(value).is_absolute():
                return PureWindowsPath(value).name
            if PurePosixPath(value).is_absolute():
                return PurePosixPath(value).name
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _safe(str(value), short_paths=short_paths)


def _display(value):
    if value is _MISSING:
        return "Not recorded"
    if value is None:
        return "Unset (null)"
    if value is True:
        return "Enabled"
    if value is False:
        return "Disabled"
    if value == "":
        return "Empty string"
    safe = _safe(value, short_paths=True)
    return json.dumps(safe, ensure_ascii=False, sort_keys=True) if isinstance(safe, (dict, list)) else str(safe)


def sanitize_configuration(config: dict | None) -> dict:
    """Copy a configuration for provenance storage, without credential values.

    Absolute configuration paths are preserved here; readable report displays
    shorten them. The sanitized copy is evidence, not a resumable configuration.
    Explicit environment mappings retain only supported operational controls.
    """
    if not isinstance(config, Mapping):
        return {}
    safe = _safe(config)
    for field in ("env", "environment"):
        if isinstance(safe.get(field), dict):
            safe[field] = {key: value for key, value in safe[field].items() if key in _ENV_KEYS}
    return safe


def _flatten(value, prefix=""):
    if not isinstance(value, Mapping):
        return {}
    result = {}
    for name, item in value.items():
        key = f"{prefix}.{name}" if prefix else str(name)
        if _secret_key(key):
            result[key] = _REDACTED
        elif isinstance(item, Mapping) and item:
            result.update(_flatten(item, key))
        else:
            result[key] = _safe(item)
    return result


def _environment_value(value, canonical):
    if not isinstance(value, str):
        return _safe(value)
    default = _SCHEMA.get(canonical, {}).get("default", _MISSING)
    if isinstance(default, str):
        return _safe(value)
    if isinstance(default, list):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return _safe(parsed)
        except (ValueError, TypeError):
            pass
        return [_safe(part.strip()) for part in value.split(",") if part.strip()]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        return float(value) if "." in value else int(value)
    return _safe(value)


def _values(data, *, runtime=False):
    """Normalize only recorded settings; environment names use a fixed allowlist."""
    if not isinstance(data, Mapping):
        return {}
    result = {}
    for field in ("env", "environment"):
        env = data.get(field)
        if isinstance(env, Mapping):
            for name, value in env.items():
                if name in _ENV_KEYS:
                    key = _ENV_KEYS[name]
                    result[key] = _environment_value(value, key)
    for name, value in data.items():
        if name in _ENV_KEYS:
            key = _ENV_KEYS[name]
            result[key] = _environment_value(value, key)
    ordinary = {
        key: value for key, value in data.items()
        if key not in {"env", "environment", "cli"}
        and key not in _ENV_KEYS
        and not (runtime and re.fullmatch(r"[A-Z][A-Z0-9_]+", str(key)))
    }
    result.update(_flatten(ordinary))
    cli = data.get("cli")
    if isinstance(cli, Mapping):
        for name, value in cli.items():
            key = _CLI_KEYS.get(str(name), "cli." + str(name))
            if name == "no_live_fetch" and isinstance(value, bool):
                value = not value
            result[key] = _REDACTED if _secret_key(key) else _safe(value)
    return result


def _group(key):
    if key.startswith("cli."):
        return "execution"
    if key.startswith(("environment.", "credentials.")):
        return "environment"
    if key.startswith(("universal.budget", "universal.model_limits", "universal.extraction_reserve")):
        return "budgets"
    if key.startswith(("llm.extraction", "llm.focused_recovery", "universal.recovery")):
        return "extraction"
    if key.startswith(("source_search.", "universal.historical_discovery")):
        return "discovery"
    if key.startswith(("content_fetch.", "live_web.", "universal.acquisition")):
        return "retrieval"
    if key.startswith(("human_review", "validation.", "anomaly_detection", "llm.source_critic", "llm.source_identity", "llm.source_credibility")):
        return "review"
    if key.startswith(("llm.", "disease_intelligence.")):
        return "model"
    if key.startswith("source_sets.") or key.endswith("overlay_path"):
        return "sources"
    if key.startswith(("output.", "studio.")):
        return "output"
    return "task"


def _applicability(key, effective):
    mode = effective.get("pipeline_mode", _MISSING)
    if key in {"llm.max_chunks", "llm.extraction.soft_primary_calls", "llm.extraction.hard_primary_calls"}:
        return "Legacy compatibility setting (deprecated); scheduler limits take precedence"
    if key.startswith("credentials.") or _secret_key(key):
        return "Credential name only; value never exported"
    if key.startswith("studio.") or key.startswith("environment.LANGFLOW"):
        return "Optional local interface"
    if key == "source_search.fixture_path":
        search_mode = effective.get("source_search.mode", _MISSING)
        return "Inactive in the recorded search mode" if search_mode not in {_MISSING, "fixture"} else "When fixture search is enabled"
    if key == "content_fetch.content_fixture_map_path":
        return "When a local content fixture map is supplied; independent of preloaded fixture documents"
    if key.startswith("human_review.") and key != "human_review.enabled":
        return "Inactive while human review is disabled" if effective.get("human_review.enabled") is False else "When human review is enabled"
    for prefix, switch in (
        ("source_search.", "source_search.enabled"),
        ("source_search.iterative.", "source_search.iterative.enabled"),
        ("source_search.authority_gap_retry.", "source_search.authority_gap_retry.enabled"),
        ("llm.extraction.", "llm.structured_extraction_enabled"),
        ("llm.focused_recovery.", "llm.structured_extraction_enabled"),
        ("llm.source_critic.", "llm.source_critic_enabled"),
        ("llm.source_credibility.", "llm.source_credibility.enabled"),
        ("llm.source_identity.", "llm.source_identity.enabled"),
        ("content_fetch.external_fetch.", "content_fetch.external_fetch.enabled"),
        ("universal.historical_discovery.", "universal.historical_discovery.enabled"),
    ):
        if key.startswith(prefix) and key != switch and effective.get(switch) is False:
            return "Inactive while the corresponding stage is disabled"
    if key.startswith("universal."):
        if mode is not _MISSING and mode != "evidence":
            return "Inactive outside evidence mode"
        policy = effective.get("universal.budget_policy.mode", _MISSING)
        if key in {"universal.budget_limits.fetch", "universal.budget_limits.fetch_ordinary"}:
            return "Inactive under adaptive policy; strict evidence budget only" if policy == "adaptive" else "Strict evidence budget only"
        if key in {"universal.budget_limits.source_targets", "universal.budget_limits.http_requests"}:
            return "Inactive under strict policy; adaptive evidence budget only" if policy == "strict" else "Adaptive evidence budget only"
        return "Evidence mode"
    if key.startswith("source_sets.") or key.endswith("overlay_path"):
        return "Inactive in evidence mode; fixed source sets and overlays are cleared" if mode == "evidence" else "Standard pipeline source policy"
    return "When the corresponding stage is enabled"


def build_run_settings(config: dict | None = None, summary: dict | None = None, state: dict | None = None) -> dict:
    """Describe configured and explicitly recorded effective settings for any run.

    ``config`` should be the selected run configuration after CLI overrides.
    ``runtime_profile`` in summary/state accepts nested settings or ``env`` /
    ``environment`` mappings. State evidence is more recent than the summary;
    explicit search diagnostics and budget limits override profile settings.
    Missing evidence stays missing even when a current default is available.
    """
    configured = _values(config)
    effective = dict(configured)
    basis = {key: "Supplied session configuration" for key in configured}
    statuses = {key: "recorded_configuration" for key in configured}

    def record(key, value, source):
        effective[key] = _REDACTED if _secret_key(key) else _safe(value)
        basis[key] = source
        statuses[key] = "recorded_effective"

    for name, evidence in (("summary", summary), ("state", state)):
        if not isinstance(evidence, Mapping):
            continue
        for key, value in _values(evidence.get("runtime_profile"), runtime=True).items():
            record(key, value, name + ".runtime_profile")
        for field, key in _SUMMARY_KEYS.items():
            if field in evidence:
                record(key, evidence[field], name + "." + field)
        for key, value in _values({"cli": evidence.get("execution_options", {})}).items():
            record(key, value, name + ".execution_options")
        search = evidence.get("source_search_execution_summary")
        if isinstance(search, Mapping):
            for field in ("max_queries", "max_total_results", "max_results_per_query", "timeout_seconds"):
                if field in search:
                    record("source_search." + field, search[field], name + ".source_search_execution_summary." + field)
        fetch = evidence.get("content_fetch_summary")
        if isinstance(fetch, Mapping) and "max_node_seconds" in fetch:
            record("content_fetch.max_node_seconds", fetch["max_node_seconds"], name + ".content_fetch_summary.max_node_seconds")
        budget = evidence.get("run_budget_ledger") or evidence.get("budget")
        if isinstance(budget, Mapping):
            limits = budget.get("limits")
            if isinstance(limits, Mapping):
                for kind, value in limits.items():
                    key = "universal.model_limits." + kind[6:] if kind.startswith("model:") else "universal.budget_limits." + kind
                    record(key, value, name + ".budget.limits." + kind)
            if "extraction_reserve" in budget:
                record("universal.extraction_reserve", budget["extraction_reserve"], name + ".budget.extraction_reserve")

    schema = dict(_SCHEMA)
    for key in configured.keys() | effective.keys():
        schema.setdefault(key, {})
    groups = [{**group, "rows": []} for group in _DESCRIPTOR["groups"]]
    by_group = {group["id"]: group for group in groups}
    for key, spec in sorted(schema.items()):
        # An empty object is retained when explicitly supplied, but a descriptor
        # container does not duplicate the individually documented controls.
        if key not in configured and key not in effective and any(other.startswith(key + ".") for other in schema):
            continue
        raw = configured.get(key, _MISSING)
        value = effective.get(key, _MISSING)
        default = spec.get("default", _MISSING)
        credential = key.startswith("credentials.") or _secret_key(key)
        label = spec.get("label", key.rsplit(".", 1)[-1].replace("_", " ").capitalize())
        row = {
            "key": key,
            "label": label,
            "description": spec.get("description", "Recorded configuration control: " + key + "."),
            "configured_value": None if raw is _MISSING or credential else _safe(raw),
            "configured_display": "Not exported (credential)" if credential else _display(raw),
            "effective_value": None if value is _MISSING or credential else _safe(value),
            "effective_display": "Not exported (credential)" if credential else _display(value),
            "value_status": "credential_redacted" if credential else statuses.get(key, "not_recorded"),
            "basis": "Variable name only; value and availability are not exported" if credential else basis.get(key, "No value recorded in the supplied session evidence"),
            "current_default": None if default is _MISSING or credential else _safe(default),
            "current_default_display": "No fixed default recorded" if default is _MISSING or credential else _display(default),
            "default_status": "not_declared" if default is _MISSING or credential else "current_reference_only",
            "default_basis": spec.get("default_basis", "No default declared for this supplied setting"),
            "applicability": _applicability(key, effective),
            "environment_variables": spec.get("environment_variables", []),
            "cli_aliases": spec.get("cli_aliases", []),
        }
        by_group[spec.get("group", _group(key))]["rows"].append(row)
    all_rows = [row for group in groups for row in group["rows"]]
    return {
        "schema_version": 1,
        "scope_note": "Recorded configuration and recorded effective execution settings are separate. Not recorded does not mean disabled, null, or zero. Current defaults are references only. Credential values are never exported.",
        "groups": [group for group in groups if group["rows"]],
        "counts": {"total": len(all_rows), **dict(Counter(row["value_status"] for row in all_rows))},
    }
