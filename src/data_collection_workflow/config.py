"""Configuration helpers for the data collection workflow."""

from __future__ import annotations

from data_collection_workflow.environment import get_env

import json
from pathlib import Path

_RESOURCES_DIR = Path(__file__).parent / "resources"
_HANTAVIRUS_PROFILE_PATH = _RESOURCES_DIR / "profiles/hantavirus.json"
_HANTAVIRUS_SCHEMA_PATH = _RESOURCES_DIR / "schemas/hantavirus.json"
_SOURCE_STRATEGY_PATH = _RESOURCES_DIR / "source_strategy.json"
_HANTAVIRUS_SEED_SOURCES_PATH = _RESOURCES_DIR / "fixtures/seed_sources/hantavirus.json"
_SOURCE_SCREENING_POLICY_PATH = _RESOURCES_DIR / "source_screening_policy.json"
_CONTENT_FETCH_POLICY_PATH = _RESOURCES_DIR / "content_fetch_policy.json"
_EVIDENCE_CHUNKING_POLICY_PATH = _RESOURCES_DIR / "evidence_chunking_policy.json"
_STRUCTURED_EXTRACTION_POLICY_PATH = _RESOURCES_DIR / "structured_extraction_policy.json"
_RECORD_NORMALIZATION_POLICY_PATH = _RESOURCES_DIR / "record_normalization_policy.json"
_RECORD_LINKING_POLICY_PATH = _RESOURCES_DIR / "record_linking_policy.json"
_CROSS_SOURCE_CONSISTENCY_POLICY_PATH = _RESOURCES_DIR / "cross_source_consistency_policy.json"
_HANTAVIRUS_FIXTURE_DOCUMENTS_PATH = _RESOURCES_DIR / "fixtures/documents/hantavirus.json"
_HUMAN_REVIEW_POLICY_PATH = _RESOURCES_DIR / "human_review_policy.json"
_FINAL_PACKAGE_POLICY_PATH = _RESOURCES_DIR / "final_package_policy.json"
_LLM_STRUCTURED_EXTRACTION_POLICY_PATH = _RESOURCES_DIR / "llm_structured_extraction_policy.json"
_SOURCE_ROLE_POLICY_PATH = _RESOURCES_DIR / "source_role_policy.json"
_DISEASE_INTELLIGENCE_DIR = _RESOURCES_DIR / "disease_intelligence"
_SEED_SOURCE_OVERLAY_ENV = "SEED_SOURCE_OVERLAY_PATH"
_SOURCE_ROLE_POLICY_OVERLAY_ENV = "SOURCE_ROLE_POLICY_OVERLAY_PATH"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_overlay_path(env_var: str) -> Path | None:
    raw = (get_env(env_var) or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"{env_var} points to a missing file: {path}")
    if not path.is_file():
        raise ValueError(f"{env_var} must point to a JSON file: {path}")
    return path


def _load_overlay_json(env_var: str) -> tuple[dict | None, Path | None]:
    path = _resolve_overlay_path(env_var)
    if path is None:
        return None, None
    try:
        data = _load_json(path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{env_var} points to invalid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{env_var} JSON root must be an object: {path}")
    return data, path


def _source_id_from_seed_source(seed_source: dict) -> str:
    source_id = str(seed_source.get("source_id") or "").strip()
    if source_id:
        return source_id
    seed_source_id = str(seed_source.get("seed_source_id") or "").strip()
    if seed_source_id.startswith("seed_"):
        return "src_" + seed_source_id[len("seed_"):]
    return seed_source_id


def _merge_seed_source_overlay(base: dict, overlay: dict, overlay_path: Path) -> dict:
    overlay_sources = overlay.get("seed_sources")
    if not isinstance(overlay_sources, list):
        raise ValueError(
            f"{_SEED_SOURCE_OVERLAY_ENV} must contain a seed_sources list: "
            f"{overlay_path}"
        )

    merged_sources: list[dict] = []
    seen_source_ids: set[str] = set()
    duplicate_count = 0
    for source in overlay_sources + list(base.get("seed_sources") or []):
        if not isinstance(source, dict):
            raise ValueError(
                f"{_SEED_SOURCE_OVERLAY_ENV} seed_sources entries must be objects: "
                f"{overlay_path}"
            )
        dedupe_key = _source_id_from_seed_source(source)
        if not dedupe_key:
            raise ValueError(
                f"{_SEED_SOURCE_OVERLAY_ENV} seed source missing seed_source_id/source_id: "
                f"{overlay_path}"
            )
        if dedupe_key in seen_source_ids:
            duplicate_count += 1
            continue
        seen_source_ids.add(dedupe_key)
        merged_sources.append(dict(source))

    merged = dict(base)
    merged["seed_sources"] = merged_sources
    merged["overlay_metadata"] = {
        "enabled": True,
        "overlay_path": str(overlay_path),
        "overlay_catalog_name": overlay.get("catalog_name"),
        "overlay_seed_source_count": len(overlay_sources),
        "overlay_precedence": "case_overlay_sources_first",
        "dedupe_key": "derived_source_id",
        "duplicate_seed_source_count": duplicate_count,
    }
    return merged


def _merge_unique_lists(
    base_values,
    overlay_values,
    key: str,
    overlay_path: Path,
) -> list:
    if base_values is None:
        base_values = []
    if overlay_values is None:
        overlay_values = []
    if not isinstance(base_values, list) or not isinstance(overlay_values, list):
        raise ValueError(
            f"{_SOURCE_ROLE_POLICY_OVERLAY_ENV} key {key!r} must be a list: "
            f"{overlay_path}"
        )
    merged: list = []
    seen: set[str] = set()
    for value in list(base_values) + list(overlay_values):
        marker = str(value)
        if marker in seen:
            continue
        seen.add(marker)
        merged.append(value)
    return merged


def _merge_source_role_policy_overlay(
    base: dict,
    overlay: dict,
    overlay_path: Path,
) -> dict:
    list_union_keys = {
        "supported_collection_modes",
        "validation_reserved_source_ids",
        "validation_reserved_domains",
        "collection_allowed_source_ids",
        "context_only_source_ids",
        "excluded_source_ids",
    }
    merged = dict(base)
    for key, value in overlay.items():
        if key == "validation_reserved_domains" and value == []:
            merged[key] = []
        elif key in list_union_keys:
            merged[key] = _merge_unique_lists(
                merged.get(key), value, key, overlay_path
            )
        elif key == "domain_masking_enabled":
            merged[key] = bool(merged.get(key, False)) or bool(value)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value

    merged["policy_overlay_metadata"] = {
        "enabled": True,
        "overlay_path": str(overlay_path),
        "live_case_study_id": overlay.get("live_case_study_id")
        or overlay.get("case_study_id"),
        "validation_reserved_source_id_count": len(
            merged.get("validation_reserved_source_ids") or []
        ),
        "domain_masking_enabled": bool(merged.get("domain_masking_enabled", False)),
    }
    return merged


def load_hantavirus_profile() -> dict:
    """Load the static hantavirus disease profile JSON as a dict."""

    return _load_json(_HANTAVIRUS_PROFILE_PATH)


def disease_intelligence_profile_slug(disease_input: str | None) -> str | None:
    """Return a curated disease-intelligence slug for known disease names."""

    text = " ".join(str(disease_input or "").strip().lower().split())
    if not text:
        return None
    normalized = (
        text.replace("_", " ")
        .replace("/", " ")
        .replace("coronavirus disease 2019", "covid-19")
    )
    compact = normalized.replace("-", "").replace(" ", "")
    if "hantavirus" in normalized or normalized == "hps" or compact == "hfrs":
        return "hantavirus"
    if compact in {"covid19", "sarscov2"} or "covid" in normalized:
        return "covid19"
    if "dengue" in normalized or compact == "denv":
        return "dengue"
    return None


def load_disease_intelligence_profile(disease_input: str | None) -> dict | None:
    """Load a curated disease intelligence profile for a known disease."""

    slug = disease_intelligence_profile_slug(disease_input)
    if not slug:
        return None
    path = _DISEASE_INTELLIGENCE_DIR / f"{slug}.json"
    if not path.exists():
        return None
    return _load_json(path)


def load_hantavirus_collection_schema() -> dict:
    """Load the hantavirus human case/outbreak collection schema JSON as a dict."""

    return _load_json(_HANTAVIRUS_SCHEMA_PATH)


def load_source_strategy() -> dict:
    """Load the source strategy (categories + screening criteria) JSON as a dict."""

    return _load_json(_SOURCE_STRATEGY_PATH)


def load_hantavirus_seed_sources() -> dict:
    """Load the offline hantavirus seed source catalog JSON as a dict."""

    base = _load_json(_HANTAVIRUS_SEED_SOURCES_PATH)
    overlay, overlay_path = _load_overlay_json(_SEED_SOURCE_OVERLAY_ENV)
    if overlay is None or overlay_path is None:
        return base
    return _merge_seed_source_overlay(base, overlay, overlay_path)


def load_source_screening_policy() -> dict:
    """Load the deterministic source screening policy JSON as a dict."""

    return _load_json(_SOURCE_SCREENING_POLICY_PATH)


def load_content_fetch_policy() -> dict:
    """Load the content fetching / document quality policy JSON as a dict."""

    return _load_json(_CONTENT_FETCH_POLICY_PATH)


def load_evidence_chunking_policy() -> dict:
    """Load the evidence chunking / data presence flagging policy JSON as a dict."""

    return _load_json(_EVIDENCE_CHUNKING_POLICY_PATH)


def load_structured_extraction_policy() -> dict:
    """Load the deterministic structured extraction policy JSON as a dict."""

    return _load_json(_STRUCTURED_EXTRACTION_POLICY_PATH)


def load_record_normalization_policy() -> dict:
    """Load the deterministic record normalization policy JSON as a dict."""

    return _load_json(_RECORD_NORMALIZATION_POLICY_PATH)


def load_record_linking_policy() -> dict:
    """Load the deterministic record linking policy JSON as a dict."""

    return _load_json(_RECORD_LINKING_POLICY_PATH)


def load_cross_source_consistency_policy() -> dict:
    """Load the deterministic cross-source consistency policy JSON as a dict."""

    return _load_json(_CROSS_SOURCE_CONSISTENCY_POLICY_PATH)


def load_hantavirus_fixture_documents() -> dict:
    """Load the synthetic local fixture document catalog JSON as a dict."""

    return _load_json(_HANTAVIRUS_FIXTURE_DOCUMENTS_PATH)


def load_human_review_policy() -> dict:
    """Load the deterministic human review policy JSON as a dict."""

    return _load_json(_HUMAN_REVIEW_POLICY_PATH)


def load_final_package_policy() -> dict:
    """Load the deterministic final package assembly policy JSON as a dict."""

    return _load_json(_FINAL_PACKAGE_POLICY_PATH)


def load_llm_structured_extraction_policy() -> dict:
    """Load the optional LLM-based structured extraction policy JSON as a dict."""

    return _load_json(_LLM_STRUCTURED_EXTRACTION_POLICY_PATH)


def load_source_role_policy() -> dict:
    """Load the collection/source-role policy JSON as a dict."""

    base = _load_json(_SOURCE_ROLE_POLICY_PATH)
    overlay, overlay_path = _load_overlay_json(_SOURCE_ROLE_POLICY_OVERLAY_ENV)
    if overlay is None or overlay_path is None:
        return base
    return _merge_source_role_policy_overlay(base, overlay, overlay_path)


def get_collection_mode(policy: dict | None = None) -> str:
    """Resolve the collection mode from policy + environment.

    Invalid values fall back to the policy default so the workflow stays in
    conservative standard mode unless a supported mode is explicitly selected.
    """

    role_policy = policy or load_source_role_policy()
    default = role_policy.get("default_collection_mode") or "standard"
    env_var = role_policy.get("enabled_env_var") or "COLLECTION_MODE"
    raw = (get_env(env_var) or "").strip()
    mode = raw or default
    supported = role_policy.get("supported_collection_modes") or [default]
    if mode not in supported:
        return default
    return mode
