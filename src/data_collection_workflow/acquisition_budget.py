"""Versioned acquisition policy; strict legacy configurations stay unchanged."""
from __future__ import annotations

import math
from urllib.parse import urlsplit, urlunsplit


POLICY_VERSION = 2
ADAPTIVE_DEFAULTS = {"source_targets": 200, "http_requests": 1000}


def nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return value


def is_adaptive_budget(config):
    policy = (config.get("universal") or {}).get("budget_policy")
    if policy is None:
        return False
    if not isinstance(policy, dict) or policy.get("version") != POLICY_VERSION:
        raise ValueError("unsupported acquisition budget policy version")
    if policy.get("mode") not in {"adaptive", "strict"}:
        raise ValueError("unsupported acquisition budget policy mode")
    nonnegative_integer(policy.get("soft_source_target", 50), "soft_source_target")
    return config.get("pipeline_mode") == "evidence" and policy["mode"] == "adaptive"


def canonical_source_target(url):
    parsed = urlsplit(str(url or ""))
    if (parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password):
        raise ValueError("source_target must be an absolute HTTP(S) URL without credentials")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, ""))


def finite_priority(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("frontier priority must be a finite number")
    return float(value)


def graph_recursion_limit(config, limits):
    if not is_adaptive_budget(config):
        return 100
    # Each budgeted action can traverse the seven validation/recovery nodes.
    # Twelve leaves room for acquisition, extraction, and finalization boundaries;
    # this is a finite backstop, not the progress/termination policy.
    dimensions = ("source_targets", "http_requests", "search", "extraction", "browser", "ocr")
    actions = sum(nonnegative_integer(limits.get(key, 0), key) for key in dimensions)
    actions += sum(nonnegative_integer(value, key) for key, value in limits.items() if key.startswith("model:"))
    return 100 + 12 * actions


def apply_interactive_acquisition_preset(config, *, quick_test=False):
    """Only the interactive generator calls this; loaded user configs stay intact."""
    from copy import deepcopy
    result = deepcopy(config)
    universal = result.setdefault("universal", {})
    universal["budget_policy"] = {"version": 2, "mode": "adaptive", "soft_source_target": 50}
    limits = universal.setdefault("budget_limits", {})
    fetch = result.setdefault("content_fetch", {})
    if not quick_test:
        for key, value in ADAPTIVE_DEFAULTS.items():
            limits.setdefault(key, value)
        fetch["max_bytes"] = 20_000_000
        return result
    # Quick mode is an explicit small-run request, including meaningful zeros.
    total = nonnegative_integer(fetch.get("max_total_sources", 5), "max_total_sources")
    derived = nonnegative_integer(fetch.get("max_search_derived_sources", 5), "max_search_derived_sources")
    targets = min(total, derived)
    search = result.get("source_search") or {}
    iterative = search.get("iterative") or {}
    query_limit = iterative.get("max_total_queries", 3) if iterative.get("enabled", True) else search.get("max_queries", 2)
    result_limit = min(search.get("max_total_results", 6), iterative.get("max_total_results", 6))
    extraction_limit = nonnegative_integer((result.get("llm") or {}).get("max_chunks", 5), "max_chunks")
    quick_limits = {"source_targets": targets, "http_requests": 5 * targets,
                    "search": query_limit, "search_results": result_limit, "extraction": extraction_limit}
    for key, value in quick_limits.items():
        value = nonnegative_integer(value, key)
        limits[key] = min(nonnegative_integer(limits.get(key, value), key), value)
    universal["extraction_reserve"] = min(universal.get("extraction_reserve", 1), max(0, limits["extraction"] - 1))
    universal["budget_policy"]["soft_source_target"] = min(50, targets)
    return result


def load_budget_amendment(path):
    import json
    from pathlib import Path
    amendment = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(amendment, dict) or set(amendment) != {"amendment_id", "increases", "reason"}:
        raise ValueError("budget amendment must contain only amendment_id, increases and reason")
    if any(not isinstance(amendment[key], str) or not amendment[key].strip() for key in ("amendment_id", "reason")):
        raise ValueError("budget amendment requires an id and reason")
    if not isinstance(amendment["increases"], dict) or not amendment["increases"]:
        raise ValueError("budget amendment requires explicit new limits")
    for key, value in amendment["increases"].items():
        nonnegative_integer(value, key)
    return amendment
