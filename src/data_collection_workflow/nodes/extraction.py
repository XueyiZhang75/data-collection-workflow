"""Structured extraction and schema validation / repair.

Provides deterministic extraction and optional LLM extraction controlled by
ENABLE_LLM_EXTRACTION, with task-specific evidence and schema checks.
"""

from __future__ import annotations

from data_collection_workflow.environment import get_env

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from hashlib import sha256
from math import ceil
from urllib.parse import urlsplit

from .. import llm_clients
from ..geography import EXPLICIT_COUNTRY_NAMES as _OFFICIAL_COUNTRY_NAMES
from ..source_coverage import (
    build_source_coverage_requirements,
    official_report_key_for_url,
)


def _parse_llm_max_chunks() -> int | None:
    """Read LLM_MAX_CHUNKS. Return positive int cap or None when unset/invalid."""

    raw = (get_env("LLM_MAX_CHUNKS") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _parse_positive_int_env(name: str, default: int) -> int:
    raw = (get_env(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _parse_ratio_env(name: str, default: float) -> float:
    raw = (get_env(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value <= 0 or value > 1:
        return default
    return value


@dataclass
class ExtractionBudgetLedger:
    """Keep all extraction model calls inside one auditable scheduler budget."""

    scheduler_mode: str
    max_concurrency: int
    soft_checkpoint_calls: int
    safety_max_calls: int
    rolling_yield_window: int
    focused_recovery_reserved_calls: int
    max_domain_call_share: float
    high_value_source_min_spans: int
    event_source_min_spans: int
    primary_call_count: int = 0
    focused_recovery_call_count: int = 0
    actual_model_call_count: int = 0
    soft_cap_extension_call_count: int = 0
    skipped_primary_hard_cap_count: int = 0
    skipped_domain_cap_count: int = 0
    low_yield_stop_count: int = 0
    incomplete_due_to_safety_cap: bool = False
    legacy_llm_max_chunks_deprecated: bool = False
    primary_calls_by_source: Counter = field(default_factory=Counter)
    calls_by_domain: Counter = field(default_factory=Counter)
    recent_valid_record_counts: list[int] = field(default_factory=list)
    recent_model_call_counts: list[int] = field(default_factory=list)
    primary_model_call_count: int = 0
    primary_result_call_count: int = 0

    @classmethod
    def from_environment(cls) -> "ExtractionBudgetLedger":
        legacy_cap = _parse_llm_max_chunks()
        mode = (get_env("LLM_EXTRACTION_SCHEDULER_MODE") or "").strip()
        if mode not in {"quality_adaptive", "full_audit"}:
            mode = "quality_adaptive"
        soft_default = legacy_cap or 400
        soft = _parse_positive_int_env(
            "LLM_EXTRACTION_SOFT_CHECKPOINT_CALLS",
            _parse_positive_int_env("LLM_SOFT_PRIMARY_CALLS", soft_default),
        )
        safety = _parse_positive_int_env(
            "LLM_EXTRACTION_SAFETY_MAX_CALLS",
            _parse_positive_int_env("LLM_HARD_PRIMARY_CALLS", 2400),
        )
        recovery_default = _parse_positive_int_env(
            "LLM_FOCUSED_RECOVERY_MAX_CALLS",
            120,
        )
        recovery = _parse_positive_int_env(
            "LLM_FOCUSED_RECOVERY_RESERVED_CALLS",
            recovery_default,
        )
        return cls(
            scheduler_mode=mode,
            max_concurrency=_parse_positive_int_env(
                "LLM_EXTRACTION_MAX_CONCURRENCY", 4
            ),
            soft_checkpoint_calls=soft,
            safety_max_calls=safety,
            rolling_yield_window=_parse_positive_int_env(
                "LLM_EXTRACTION_ROLLING_YIELD_WINDOW", 40
            ),
            focused_recovery_reserved_calls=recovery,
            max_domain_call_share=_parse_ratio_env(
                "LLM_MAX_DOMAIN_CALL_SHARE",
                0.20,
            ),
            high_value_source_min_spans=_parse_positive_int_env(
                "LLM_HIGH_VALUE_SOURCE_MIN_SPANS",
                2,
            ),
            event_source_min_spans=_parse_positive_int_env(
                "LLM_EVENT_SOURCE_MIN_SPANS",
                1,
            ),
            legacy_llm_max_chunks_deprecated=legacy_cap is not None,
        )

    @property
    def soft_primary_calls(self) -> int:
        """Compatibility alias for older in-flight extraction callers."""

        return self.soft_checkpoint_calls

    @property
    def hard_primary_calls(self) -> int:
        """Compatibility alias for the total safety ceiling."""

        return self.safety_max_calls

    @property
    def total_model_call_count(self) -> int:
        return self.primary_call_count + self.focused_recovery_call_count

    def _required_source_floor(self, budget_row: dict) -> int:
        if str(budget_row.get("budget_bucket")) == "verified_target_collection":
            return self.high_value_source_min_spans
        if bool(budget_row.get("official_or_high_trust")):
            return self.event_source_min_spans
        return 0

    def _domain_cap(self) -> int:
        return max(8, ceil(self.safety_max_calls * self.max_domain_call_share))

    def can_attempt_primary(
        self,
        chunk: dict,
        budget_row: dict,
        context: dict | None,
        *,
        actual_model_call: bool = True,
        record_skip: bool = True,
    ) -> tuple[bool, str | None]:
        if actual_model_call and self.actual_model_call_count >= self.safety_max_calls:
            if record_skip:
                self.skipped_primary_hard_cap_count += 1
                self.incomplete_due_to_safety_cap = True
            return False, "safety_max_calls_reached"

        source_id = str(chunk.get("source_id") or "unknown")
        required_floor = self._required_source_floor(budget_row)
        source_is_below_floor = self.primary_calls_by_source[source_id] < required_floor
        strong_signal = _chunk_has_strong_record_signal(chunk)
        if (
            self.scheduler_mode == "quality_adaptive"
            and self.primary_call_count >= self.soft_checkpoint_calls
            and not (source_is_below_floor or strong_signal)
        ):
            return False, "primary_soft_cap_no_uncovered_strong_signal"

        domain = _chunk_domain(chunk, context)
        if (
            actual_model_call
            and domain
            and self.calls_by_domain[domain] >= self._domain_cap()
            and not source_is_below_floor
        ):
            if record_skip:
                self.skipped_domain_cap_count += 1
            return False, "primary_domain_cap_reached"
        return True, None

    def register_primary_call(
        self,
        chunk: dict,
        context: dict | None,
        *,
        actual_model_call: bool = True,
    ) -> None:
        if self.primary_call_count >= self.soft_checkpoint_calls:
            self.soft_cap_extension_call_count += 1
        self.primary_call_count += 1
        source_id = str(chunk.get("source_id") or "unknown")
        self.primary_calls_by_source[source_id] += 1
        if actual_model_call:
            self.primary_model_call_count += 1
            self.register_actual_model_call(chunk, context)

    def cancel_undispatched_reservation(self, chunk: dict, context: dict | None, *, focused: bool = False) -> None:
        """Release a scheduler reservation only when the shared ledger did not dispatch."""
        if focused:
            self.focused_recovery_call_count -= 1
        else:
            self.primary_call_count -= 1
            self.primary_model_call_count -= 1
            self.soft_cap_extension_call_count = max(0, self.primary_call_count-self.soft_checkpoint_calls)
            self.primary_calls_by_source[str(chunk.get('source_id') or 'unknown')] -= 1
        self.actual_model_call_count -= 1
        domain = _chunk_domain(chunk, context)
        if domain:
            self.calls_by_domain[domain] -= 1

    def register_primary_result(
        self,
        valid_record_count: int,
        *,
        actual_model_calls: int = 1,
    ) -> None:
        if actual_model_calls <= 0:
            return
        self.primary_result_call_count += int(actual_model_calls)
        self.recent_valid_record_counts.append(max(0, int(valid_record_count)))
        self.recent_model_call_counts.append(int(actual_model_calls))
        while sum(self.recent_model_call_counts) > self.rolling_yield_window:
            overflow = sum(self.recent_model_call_counts) - self.rolling_yield_window
            if self.recent_model_call_counts[0] <= overflow:
                self.recent_model_call_counts.pop(0)
                self.recent_valid_record_counts.pop(0)
                continue
            self.recent_model_call_counts[0] -= overflow
            break
        if len(self.recent_valid_record_counts) > self.rolling_yield_window:
            self.recent_valid_record_counts = self.recent_valid_record_counts[
                -self.rolling_yield_window:
            ]
            self.recent_model_call_counts = self.recent_model_call_counts[
                -self.rolling_yield_window:
            ]

    def register_actual_model_calls(self, count: int) -> None:
        self.actual_model_call_count += max(0, int(count))

    def register_actual_model_call(
        self,
        chunk: dict,
        context: dict | None,
    ) -> None:
        self.actual_model_call_count += 1
        domain = _chunk_domain(chunk, context)
        if domain:
            self.calls_by_domain[domain] += 1

    def should_stop_low_yield(
        self,
        chunk: dict,
        budget_row: dict,
        context: dict | None,
        *,
        record_skip: bool = True,
    ) -> bool:
        """Stop one low-priority span after a full low-yield model-call window."""

        if self.scheduler_mode != "quality_adaptive":
            return False
        if sum(self.recent_model_call_counts) < self.rolling_yield_window:
            return False
        model_calls = sum(self.recent_model_call_counts)
        valid_yield = sum(self.recent_valid_record_counts) / model_calls
        if valid_yield >= 0.05:
            return False
        source_id = str(chunk.get("source_id") or "unknown")
        if self.primary_calls_by_source[source_id] < self._required_source_floor(
            budget_row
        ) or _chunk_has_strong_record_signal(chunk):
            return False
        if record_skip:
            self.low_yield_stop_count += 1
        return True

    def can_attempt_recovery(self, *, actual_model_call: bool = True) -> bool:
        if actual_model_call and self.actual_model_call_count >= self.safety_max_calls:
            self.incomplete_due_to_safety_cap = True
            return False
        return self.focused_recovery_call_count < self.focused_recovery_reserved_calls

    def register_recovery_call(
        self,
        chunk: dict,
        context: dict | None,
        *,
        actual_model_call: bool = True,
    ) -> None:
        self.focused_recovery_call_count += 1
        if actual_model_call:
            self.register_actual_model_call(chunk, context)

    def as_dict(self) -> dict:
        return {
            "scheduler_mode": self.scheduler_mode,
            "max_concurrency": self.max_concurrency,
            "soft_checkpoint_calls": self.soft_checkpoint_calls,
            "safety_max_calls": self.safety_max_calls,
            "rolling_yield_window": self.rolling_yield_window,
            "soft_primary_calls": self.soft_primary_calls,
            "hard_primary_calls": self.hard_primary_calls,
            "focused_recovery_reserved_calls": (
                self.focused_recovery_reserved_calls
            ),
            "max_domain_call_share": self.max_domain_call_share,
            "high_value_source_min_spans": self.high_value_source_min_spans,
            "event_source_min_spans": self.event_source_min_spans,
            "primary_call_count": self.primary_call_count,
            "focused_recovery_call_count": self.focused_recovery_call_count,
            "total_model_call_count": self.total_model_call_count,
            "actual_model_call_count": self.actual_model_call_count,
            "primary_model_call_count": self.primary_model_call_count,
            "primary_result_call_count": self.primary_result_call_count,
            "soft_cap_extension_call_count": self.soft_cap_extension_call_count,
            "skipped_primary_hard_cap_count": self.skipped_primary_hard_cap_count,
            "skipped_domain_cap_count": self.skipped_domain_cap_count,
            "low_yield_stop_count": self.low_yield_stop_count,
            "rolling_valid_record_yield": (
                sum(self.recent_valid_record_counts)
                / sum(self.recent_model_call_counts)
                if sum(self.recent_model_call_counts)
                else None
            ),
            "rolling_actual_model_calls": sum(self.recent_model_call_counts),
            "incomplete_due_to_safety_cap": self.incomplete_due_to_safety_cap,
            "legacy_llm_max_chunks_deprecated": self.legacy_llm_max_chunks_deprecated,
            "primary_calls_by_source": dict(sorted(self.primary_calls_by_source.items())),
            "calls_by_domain": dict(sorted(self.calls_by_domain.items())),
        }


def _llm_semantic_span_hash(chunk: dict, policy: object) -> str:
    """Hash every policy, source, task, and evidence value sent to the LLM."""

    policy_payload = (
        policy.model_dump(mode="json")
        if hasattr(policy, "model_dump")
        else {"policy_type": type(policy).__name__}
    )
    task_context = {
        key: chunk.get(key)
        for key in (
            "task_acceptance_contract",
            "record_acceptance_rules",
            "context_or_quarantine_rules",
            "recovery_reason",
            "case_span_extraction_method",
        )
        if chunk.get(key) is not None
    }
    payload = {
        "policy": policy_payload,
        "task_evidence_context": task_context,
        "source_prompt_metadata": {
            key: chunk.get(key)
            for key in (
                "source_id",
                "source_url",
                "source_type",
                "title",
                "publisher",
                "chunk_id",
                "fetch_purpose",
                "chunk_kind",
                "data_types",
                "context_types",
            )
        },
        "text": str(chunk.get("text") or ""),
        "case_span_quote": str(chunk.get("case_span_quote") or ""),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _run_bounded_llm_calls(
    chunks: list[dict],
    *,
    policy: object,
    max_concurrency: int,
    result_cache: dict[str, tuple[object | None, Exception | None]] | None = None,
) -> tuple[list[dict], dict]:
    """Execute unique prompt/evidence calls in threads and apply in input order.

    The function looks up ``llm_clients.extract_chunk_with_llm`` in each worker
    so the long-standing monkeypatch seam remains intact.
    """

    cache = result_cache if result_cache is not None else {}
    keys = [_llm_semantic_span_hash(chunk, policy) for chunk in chunks]
    scheduled: dict[str, dict] = {}
    cache_hits: set[int] = set()
    for index, (key, chunk) in enumerate(zip(keys, chunks, strict=True)):
        if key in cache or key in scheduled:
            cache_hits.add(index)
            continue
        scheduled[key] = chunk

    dispatched = {}
    def _call(chunk: dict) -> tuple[object | None, Exception | None]:
        from ..session_runtime import capture_model_dispatch, ProviderAccountLimit
        key = _llm_semantic_span_hash(chunk, policy)
        with capture_model_dispatch() as receipt:
            try:
                return llm_clients.extract_chunk_with_llm(chunk, policy), None
            except Exception as exc:  # noqa: BLE001 - caller preserves existing fallback policy
                if isinstance(exc, ProviderAccountLimit):
                    receipt['dispatched'] = exc.dispatched
                return None, exc
            finally:
                dispatched[key] = receipt['dispatched'] is not False

    completed: dict[str, tuple[object | None, Exception | None]] = {}
    if scheduled:
        with ThreadPoolExecutor(max_workers=max(1, int(max_concurrency))) as executor:
            futures = {
                key: executor.submit(_call, chunk)
                for key, chunk in scheduled.items()
            }
            # Read futures by deterministic scheduling order, never completion order.
            for key in scheduled:
                completed[key] = futures[key].result()
                if completed[key][1] is None:
                    cache[key] = completed[key]

    results = []
    for index, (key, chunk) in enumerate(zip(keys, chunks, strict=True)):
        output, error = cache[key] if key in cache else completed[key]
        results.append(
            {
                "chunk": chunk,
                "output": output,
                "error": error,
                "cache_hit": index in cache_hits,
                "model_call": key in scheduled and scheduled[key] is chunk and dispatched.get(key, False),
                "semantic_span_hash": key,
            }
        )
    return results, {
        "model_call_count": sum(dispatched.values()),
        "cache_hit_count": len(cache_hits),
    }


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = get_env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _collection_mode_from_context(context: dict | None) -> str:
    if isinstance(context, dict):
        for key in ("collection_mode",):
            value = context.get(key)
            if value:
                return str(value)
        for key in ("structured_task", "collection_spec"):
            nested = context.get(key)
            if isinstance(nested, dict) and nested.get("collection_mode"):
                return str(nested["collection_mode"])
    return (get_env("COLLECTION_MODE") or "standard").strip() or "standard"


def _direct_collection_enabled(context: dict | None) -> bool:
    return _collection_mode_from_context(context) == "direct_collection"


def _first_text(*values: object) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return None


def _task_location_from_context(context: dict | None) -> str | None:
    if not isinstance(context, dict):
        return None
    contract = context.get("task_acceptance_contract")
    structured_task = context.get("structured_task")
    collection_spec = context.get("collection_spec")
    return _first_text(
        contract.get("location") if isinstance(contract, dict) else None,
        structured_task.get("location") if isinstance(structured_task, dict) else None,
        collection_spec.get("geography") if isinstance(collection_spec, dict) else None,
        collection_spec.get("location") if isinstance(collection_spec, dict) else None,
        context.get("task_location"),
    )


def _is_united_states_location(value: str | None) -> bool:
    normalized = re.sub(r"[^a-z]+", " ", str(value or "").lower()).strip()
    return normalized in {
        "united states",
        "united states of america",
        "usa",
        "us",
        "u s",
        "u s a",
    }
from ..config import (
    load_source_role_policy,
    load_llm_structured_extraction_policy,
    load_structured_extraction_policy,
)
from ..disease_relevance import (
    AMBIGUOUS_DISEASE,
    COMPATIBLE,
    INCOMPATIBLE_DISEASE,
    INSUFFICIENT_TEXT,
    TARGET_DISEASE_MATCH,
    assessment_fields,
    assess_chunk_disease_relevance,
    assess_record_disease_compatibility,
    build_disease_relevance_context,
    record_compatibility_fields,
    update_disease_relevance_summary,
)
from ..models import (
    ObservationRecord,
    HumanReviewItem,
    LLMExtractedRecord,
    LLMExtractionOutput,
    LLMStructuredExtractionPolicy,
    PublicHealthRecord,
    SchemaValidationResult,
    StructuredExtractionPolicy,
)
from ..run_events import emit_workflow_progress
from ..state import DataCollectionState, append_trace

_FIELD_DETECTION_KEYS = (
    "cases_confirmed",
    "cases_probable",
    "cases_suspected",
    "cases_unspecified",
    "deaths",
    "hospitalizations",
    "date_reported",
    "country",
    "subnational_location",
    "virus_or_syndrome",
    "pathogen_or_syndrome",
)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _lower(text: str | None) -> str:
    return (text or "").lower()


def _normalize_number(value: str | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if not s:
        return None
    try:
        num = float(s)
    except (TypeError, ValueError):
        return None
    if num < 0:
        return None
    return num


def _detect_virus_or_syndrome(
    text: str,
    policy: StructuredExtractionPolicy,
) -> str | None:
    lowered = _lower(text)
    for canonical_key, terms in policy.virus_or_syndrome_terms.items():
        for term in terms:
            if not term:
                continue
            if term.lower() in lowered:
                return canonical_key
    return None


def _canonical_disease_name(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if not lowered:
        return None
    if lowered in {
        "hantavirus",
        "hantavirus disease",
        "hps",
        "hantavirus pulmonary syndrome",
    }:
        return "Hantavirus disease"
    if lowered in {"covid", "covid-19", "covid 19", "sars-cov-2"}:
        return "COVID-19"
    if lowered in {"dengue", "dengue fever", "denv", "dengue virus"}:
        return "Dengue"
    return str(value).strip()


def _build_extraction_context(
    state: DataCollectionState | None,
    policy: StructuredExtractionPolicy,
) -> dict:
    state = state or {}
    structured_task = state.get("structured_task") or {}
    collection_spec = state.get("collection_spec") or {}
    disease_intelligence = state.get("disease_intelligence") or {}
    explicit_disease_value = (
        disease_intelligence.get("disease_standard_name")
        or structured_task.get("disease")
        or collection_spec.get("disease")
    )

    disease_standard_name = _canonical_disease_name(
        explicit_disease_value or policy.default_disease
    ) or policy.default_disease

    terms: list[str] = []
    for key in ("disease_input", "disease_standard_name"):
        value = disease_intelligence.get(key)
        if isinstance(value, str):
            terms.append(value)
    for key in ("aliases", "abbreviations", "pathogen_terms", "syndrome_terms"):
        values = disease_intelligence.get(key) or []
        if isinstance(values, list):
            terms.extend(str(v) for v in values if v)
    for value in (structured_task.get("disease"), collection_spec.get("disease")):
        if value:
            terms.append(str(value))

    from ..evidence_qualification import evidence_qualification_enabled
    if evidence_qualification_enabled():
        # Reuse exact disease-name equivalences already accepted by chunk and
        # record relevance; discovery/pathogen aliases remain distinct.
        from ..disease_identity import disease_names
        terms.extend(sorted(disease_names(disease_standard_name)))

    pathogen_terms = [
        str(v)
        for key in ("pathogen_terms", "abbreviations", "syndrome_terms")
        for v in (disease_intelligence.get(key) or [])
        if v
    ]
    if disease_standard_name == "COVID-19":
        terms.extend(["COVID-19", "SARS-CoV-2"])
        pathogen_terms.extend(["SARS-CoV-2", "COVID-19"])
    elif disease_standard_name == "Dengue":
        terms.extend(["dengue", "DENV", "dengue virus"])
        pathogen_terms.extend(["DENV", "dengue virus", "dengue"])
    elif disease_standard_name == "Hantavirus disease":
        terms.extend(["hantavirus", "HPS", "Sin Nombre virus"])

    source_registry_by_id = {
        str(row.get("source_id")): row
        for row in (state.get("source_registry") or [])
        if isinstance(row, dict) and row.get("source_id")
    }
    official_report_key_by_source_id: dict[str, str] = {}
    for source_id, row in source_registry_by_id.items():
        key = official_report_key_for_url(row.get("canonical_url") or row.get("url"))
        if key:
            official_report_key_by_source_id[source_id] = key
    must_fetch_source_ids = {
        str(row.get("source_id"))
        for row in (state.get("must_fetch_sources") or [])
        if isinstance(row, dict) and row.get("source_id")
    }
    must_fetch_source_ids.update(
        str(source_id)
        for source_id, row in source_registry_by_id.items()
        if isinstance(row, dict) and row.get("must_fetch") is True
    )
    documents_by_source_id: dict[str, list[dict]] = {}
    for document in state.get("documents") or []:
        if not isinstance(document, dict):
            continue
        source_id = document.get("source_id")
        if source_id:
            documents_by_source_id.setdefault(str(source_id), []).append(document)

    seen: set[str] = set()
    disease_terms = [
        term
        for term in terms
        if term and not (term.lower() in seen or seen.add(term.lower()))
    ]
    seen_pathogen: set[str] = set()
    pathogen_terms = [
        term
        for term in pathogen_terms
        if term
        and not (term.lower() in seen_pathogen or seen_pathogen.add(term.lower()))
    ]

    return {
        "collection_mode": (
            structured_task.get("collection_mode")
            or collection_spec.get("collection_mode")
            or get_env("COLLECTION_MODE")
            or "standard"
        ),
        "structured_task": {
            "disease": disease_standard_name if explicit_disease_value else None,
            "location": structured_task.get("location"),
            "start_date": structured_task.get("start_date"),
            "end_date": structured_task.get("end_date"),
            "collection_mode": structured_task.get("collection_mode"),
        },
        "collection_spec": {
            "disease": disease_standard_name if explicit_disease_value else None,
            "geography": structured_task.get("location")
            or collection_spec.get("geography"),
            "time_window": collection_spec.get("time_window")
            or structured_task.get("start_date")
            or structured_task.get("end_date"),
            "target_population": collection_spec.get("target_population")
            or "humans",
            "collection_mode": collection_spec.get("collection_mode"),
        },
        "disease_intelligence": {
            "disease_standard_name": (
                disease_standard_name if explicit_disease_value else None
            ),
            "aliases": disease_terms,
            "pathogen_terms": pathogen_terms,
            "syndrome_terms": disease_terms,
        },
        "disease_standard_name": disease_standard_name,
        "disease_terms": disease_terms,
        "pathogen_terms": pathogen_terms,
        "target_population": collection_spec.get("target_population") or "humans",
        "task_location": structured_task.get("location") or collection_spec.get("geography"),
        "time_window": collection_spec.get("time_window")
        or structured_task.get("start_date")
        or structured_task.get("end_date"),
        "target_fields": list(structured_task.get("target_fields") or []),
        "task_acceptance_contract": state.get("task_acceptance_contract") or {},
        "is_hantavirus": disease_standard_name == "Hantavirus disease",
        "must_fetch_source_ids": sorted(must_fetch_source_ids),
        "source_registry_by_id": source_registry_by_id,
        "official_report_key_by_source_id": official_report_key_by_source_id,
        "documents_by_source_id": documents_by_source_id,
        "source_coverage_audit": state.get("source_coverage_audit") or {},
    }


_HANTAVIRUS_ALIAS_TERMS: tuple[tuple[str, str], ...] = (
    ("hantavirus pulmonary syndrome", "hantavirus pulmonary syndrome"),
    ("andes virus", "Andes virus"),
    ("orthohantavirus", "orthohantavirus"),
    ("hantavirus", "hantavirus"),
    ("andv", "ANDV"),
    ("hps", "HPS"),
)

_OFFICIAL_OUTBREAK_SOURCE_HINTS = (
    "who_don",
    "cdc_han",
    "paho_outbreak_update",
    "ecdc_outbreak_update",
    "official_health_alert",
    "official_public_health_outbreak_page",
    "disease outbreak news",
    "disease-outbreak-news",
    "health alert",
    "health advisory",
    "health update",
    "epidemiological alert",
    "outbreak update",
    "outbreak",
)

_OFFICIAL_OUTBREAK_WORD_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty one": 21,
    "twenty-one": 21,
}

_OFFICIAL_COUNT_TOKEN = (
    r"(?:\d+(?:,\d{3})*(?:\.\d+)?|"
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty(?:[- ]one)?)"
)

_OFFICIAL_CASE_RE = re.compile(
    rf"\b(?P<num>{_OFFICIAL_COUNT_TOKEN})\s+"
    r"(?:(?:laboratory[- ]confirmed|lab[- ]confirmed|confirmed|probable|"
    r"suspected|reported|human|total|additional|aggregated)\s+){0,5}"
    r"(?P<label>cases?|infections?)\b",
    re.IGNORECASE,
)
_OFFICIAL_CASE_CONTEXT_RE = re.compile(
    rf"\b(?P<num>{_OFFICIAL_COUNT_TOKEN})\s+"
    r"(?P<context>(?:[\w/-]+\s+){0,8}?)"
    r"(?P<label>cases?|infections?)\b",
    re.IGNORECASE,
)
_OFFICIAL_CASE_TOTAL_RE = re.compile(
    rf"\b(?:a\s+)?(?:aggregated\s+)?total\s+of\s+"
    rf"(?P<num>{_OFFICIAL_COUNT_TOKEN})\s+"
    r"(?:(?:confirmed|probable|suspected|reported|human)\s+){0,4}"
    r"(?P<label>cases?|infections?)\b",
    re.IGNORECASE,
)
_OFFICIAL_DEATH_RE = re.compile(
    rf"\b(?P<num>{_OFFICIAL_COUNT_TOKEN})\s+"
    r"(?:(?:reported|associated|additional|total)\s+){0,4}"
    r"(?P<label>deaths?|fatalities|fatal\s+cases?)\b",
    re.IGNORECASE,
)
_OFFICIAL_CASE_LABEL_RE = re.compile(
    r"\b(?P<label>Cases?\s+(?P<ids>\d+(?:\s*(?:,|and)\s*\d+)*))\b",
    re.IGNORECASE,
)
_OFFICIAL_DIED_RE = re.compile(
    r"\b(?:died|death\s+occurred)(?:\s+on\s+board)?\s+on\s+(?P<date>"
    r"(?:\d{1,2}\s+[A-Z][a-z]+\s+\d{4})|"
    r"(?:[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})|"
    r"(?:\d{1,2}\s+[A-Z][a-z]+))",
    re.IGNORECASE,
)
_OFFICIAL_AS_OF_DATE_RE = re.compile(
    r"\bas\s+of\s+(?P<date>(?:\d{1,2}\s+[A-Z][a-z]+\s+\d{4})|"
    r"(?:[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})|"
    r"(?:\d{1,2}\s+[A-Z][a-z]+))",
    re.IGNORECASE,
)
_OFFICIAL_DATE_RANGE_RE = re.compile(
    r"(?P<start>\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\s+"
    r"(?:to|-|through|until)\s+"
    r"(?P<end>\d{1,2}\s+[A-Z][a-z]+\s+\d{4})",
    re.IGNORECASE,
)
_OFFICIAL_MONTH_DAY_YEAR_RE = re.compile(
    r"\b(?P<month>January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+"
    r"(?P<day>\d{1,2}),?\s+(?P<year>20\d{2})\b",
    re.IGNORECASE,
)
_OFFICIAL_DAY_MONTH_YEAR_RE = re.compile(
    r"\b(?P<day>\d{1,2})\s+"
    r"(?P<month>January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(?P<year>20\d{2})\b",
    re.IGNORECASE,
)
_OFFICIAL_DAY_MONTH_RE = re.compile(
    r"\b(?P<day>\d{1,2})\s+"
    r"(?P<month>January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\b",
    re.IGNORECASE,
)
_OFFICIAL_VESSEL_RE = re.compile(r"\bMV\s+[A-Z][A-Za-z0-9-]*\b")
_OFFICIAL_NEGATIVE_COUNT_CONTEXTS = (
    "countries",
    "country",
    "ports",
    "port",
    "crew",
    "passengers",
    "passenger",
    "contacts",
    "contact",
    "specimens",
    "specimen",
    "samples",
    "sample",
    "tests",
    "test",
    "week",
    "weeks",
    "ew",
    "percent",
    "%",
    "ages",
    "age",
    "years",
    "year",
    "capacity",
    "maternal deaths",
    "preventable maternal",
)


def _official_number(value: str | None) -> float | None:
    if value is None:
        return None
    lowered = re.sub(r"\s+", " ", str(value).strip().lower())
    if lowered in _OFFICIAL_OUTBREAK_WORD_NUMBERS:
        return float(_OFFICIAL_OUTBREAK_WORD_NUMBERS[lowered])
    return _normalize_number(value)


def _chunk_source_text(chunk: dict, context: dict | None = None) -> str:
    source = _source_context_for_chunk(chunk, context)
    values = [
        chunk.get("source_type"),
        chunk.get("source_type_final"),
        chunk.get("source_url"),
        chunk.get("canonical_url"),
        chunk.get("title"),
        chunk.get("publisher"),
        chunk.get("actual_publisher"),
        source.get("source_type"),
        source.get("source_type_final"),
        source.get("canonical_url"),
        source.get("url"),
        source.get("title"),
        source.get("publisher"),
        source.get("actual_publisher"),
    ]
    return " ".join(str(value or "") for value in values)


def _official_outbreak_source_type(chunk: dict, context: dict | None = None) -> str | None:
    text = _lower(_chunk_source_text(chunk, context))
    if "who.int" in text and (
        "disease-outbreak-news" in text
        or "disease outbreak news" in text
        or re.search(r"\bdon\s*\d+", text)
    ):
        return "WHO_DON"
    if "cdc.gov" in text and ("/han/" in text or " han " in f" {text} "):
        return "CDC_HAN"
    if "paho.org" in text and (
        "epidemiological alert" in text
        or "outbreak" in text
        or "hantavirus pulmonary syndrome" in text
    ):
        return "PAHO_outbreak_update"
    if "ecdc.europa.eu" in text and (
        "outbreak" in text
        or "rapid risk assessment" in text
        or "communicable disease threats report" in text
    ):
        return "ECDC_outbreak_update"
    explicit_types = {
        _lower(chunk.get("source_type")),
        _lower(chunk.get("source_type_final")),
    }
    if explicit_types & {
        "who_don",
        "cdc_han",
        "paho_outbreak_update",
        "ecdc_outbreak_update",
        "official_health_alert",
        "official_public_health_outbreak_page",
    }:
        return next(t for t in explicit_types if t)
    if any(hint in text for hint in _OFFICIAL_OUTBREAK_SOURCE_HINTS):
        return "official_public_health_outbreak_page"
    return None


def _is_official_outbreak_page_chunk(chunk: dict, context: dict | None = None) -> bool:
    source_type = _official_outbreak_source_type(chunk, context)
    if not source_type:
        return False
    source_text = _lower(_chunk_source_text(chunk, context))
    official_type = any(
        token in source_text
        for token in (
            "official_public_health_agency",
            "national_public_health_agency",
            "state_or_local_public_health_agency",
            "international_public_health_agency",
            "public_health_agency",
            ".gov",
            "who.int",
            "paho.org",
            "ecdc.europa.eu",
            "world health organization",
            "centers for disease control and prevention",
            "pan american health organization",
        )
    )
    return official_type or _chunk_is_official_or_high_trust(chunk, context)


def _task_is_hantavirus(context: dict | None) -> bool:
    if isinstance(context, dict) and context.get("is_hantavirus") is True:
        return True
    values = [
        (context or {}).get("disease_standard_name") if isinstance(context, dict) else None,
    ]
    if isinstance(context, dict):
        for key in ("structured_task", "collection_spec", "disease_intelligence"):
            nested = context.get(key)
            if isinstance(nested, dict):
                values.extend(
                    nested.get(k)
                    for k in ("disease", "disease_standard_name", "disease_input")
                )
    return any("hantavirus" in _lower(str(value or "")) for value in values)


def _official_hantavirus_alias_match(
    text: str,
    context: dict | None = None,
) -> tuple[bool, str | None]:
    lowered = _lower(text)
    if _task_is_hantavirus(context):
        for needle, label in _HANTAVIRUS_ALIAS_TERMS:
            if re.search(rf"\b{re.escape(needle)}\b", lowered, flags=re.IGNORECASE):
                return True, label
        return False, None

    alias = _detect_disease_alias_used(text, context)
    return bool(alias), alias


def _sentence_for_match(text: str, start: int, end: int) -> str:
    left_candidates = [text.rfind(".", 0, start), text.rfind("\n", 0, start)]
    right_candidates = [
        pos for pos in (text.find(".", end), text.find("\n", end)) if pos >= 0
    ]
    left = max(left_candidates) + 1 if max(left_candidates) >= 0 else 0
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return re.sub(r"\s+", " ", text[left:right]).strip()


def _official_match_disallowed(sentence: str, label: str) -> bool:
    lowered = _lower(sentence)
    if any(
        phrase in lowered
        for phrase in (
            "annual epidemiological report",
            "more on this topic",
            "more information on",
            "related video",
            "research initiative",
            "global perspective on",
            "fact sheet",
        )
    ):
        return True
    if "acquired the infection" in lowered or "acquired infection" in lowered:
        return True
    if "maternal deaths" in lowered or "preventable maternal" in lowered:
        return True
    if label in {"case", "cases", "infection", "infections"}:
        return False
    return any(
        term in lowered
        for term in (
            "maternal deaths",
            "preventable maternal",
            "page number",
            "table row",
            "vessel capacity",
        )
    )


def _official_case_match_has_negative_context(match: re.Match) -> bool:
    context = _lower(match.groupdict().get("context") or match.group(0))
    return any(
        re.search(rf"\b{re.escape(term)}\b", context)
        for term in (
            "countries",
            "country",
            "ports",
            "port",
            "crew",
            "passengers",
            "passenger",
            "contacts",
            "contact",
            "patients",
            "patient",
            "specimens",
            "specimen",
            "samples",
            "sample",
            "tests",
            "test",
            "week",
            "weeks",
            "ew",
            "ages",
            "age",
            "years",
            "year",
            "capacity",
            *list(_MONTHS.keys()),
        )
    ) or "%" in context or bool(re.search(r"\b20\d{2}\b", context))


def _official_count_token_is_year(value: str | None) -> bool:
    token = str(value or "").replace(",", "").strip()
    return bool(re.fullmatch(r"(?:19|20)\d{2}", token))


def _official_sentence_outside_task_years(
    sentence: str,
    context: dict | None = None,
) -> bool:
    years = set(re.findall(r"\b(20\d{2})\b", sentence))
    if not years:
        return False
    task = (context or {}).get("structured_task") or {}
    allowed: set[str] = set()
    for key in ("start_date", "end_date"):
        match = re.search(r"\b(20\d{2})\b", str(task.get(key) or ""))
        if match:
            allowed.add(match.group(1))
    if not allowed:
        return False
    return not bool(years & allowed)


def _official_case_sentence_has_count_cue(sentence: str, label: str) -> bool:
    lowered = _lower(sentence)
    if get_env("PIPELINE_MODE") == "evidence" and re.search(
        r"\b(?:identified|recorded|detected)\b", lowered
    ):
        return True
    if label in {"infection", "infections"} and "case" not in lowered:
        return any(
            cue in lowered
            for cue in (
                "reported",
                "confirmed",
                "identified as",
                "aware of",
                "total",
                "aggregated",
            )
        )
    return any(
        cue in lowered
        for cue in (
            "reported",
            "confirmed",
            "probable",
            "suspected",
            "total",
            "aggregated",
            "including",
            "aware of",
            "as of",
            "have been",
        )
    )


def _legacy_official_best_count_mentions(
    text: str,
    context: dict | None = None,
) -> tuple[dict | None, dict | None, str | None]:
    case_mentions: list[dict] = []
    death_mentions: list[dict] = []
    for match in _OFFICIAL_CASE_LABEL_RE.finditer(text):
        sentence = _sentence_for_match(text, match.start(), match.end())
        if _official_sentence_outside_task_years(sentence, context):
            continue
        ids = re.findall(r"\d+", match.group("ids") or "")
        case_mentions.append(
            {
                "value": float(len(ids) or 1),
                "span": match.group("label"),
                "sentence": sentence,
                "priority": 4,
            }
        )
    for pattern in (_OFFICIAL_CASE_TOTAL_RE, _OFFICIAL_CASE_RE, _OFFICIAL_CASE_CONTEXT_RE):
        for match in pattern.finditer(text):
            if _official_count_token_is_year(match.group("num")):
                continue
            value = _official_number(match.group("num"))
            if value is None:
                continue
            if pattern is _OFFICIAL_CASE_CONTEXT_RE and _official_case_match_has_negative_context(match):
                continue
            sentence = _sentence_for_match(text, match.start(), match.end())
            if _official_match_disallowed(sentence, "cases"):
                continue
            if _official_sentence_outside_task_years(sentence, context):
                continue
            label = _lower(match.group("label"))
            if not _official_case_sentence_has_count_cue(sentence, label):
                continue
            priority = 2 if pattern is _OFFICIAL_CASE_TOTAL_RE else 1
            if "total" in _lower(sentence) or "aggregate" in _lower(sentence):
                priority += 2
            case_mentions.append(
                {
                    "value": value,
                    "span": match.group(0),
                    "sentence": sentence,
                    "priority": priority,
                }
            )
    for match in _OFFICIAL_DEATH_RE.finditer(text):
        if _official_count_token_is_year(match.group("num")):
            continue
        value = _official_number(match.group("num"))
        if value is None:
            continue
        sentence = _sentence_for_match(text, match.start(), match.end())
        if _official_match_disallowed(sentence, "deaths"):
            continue
        if _official_sentence_outside_task_years(sentence, context):
            continue
        death_mentions.append(
            {
                "value": value,
                "span": match.group(0),
                "sentence": sentence,
                "priority": 1 + (1 if "including" in _lower(sentence) else 0),
            }
        )
    for match in _OFFICIAL_DIED_RE.finditer(text):
        sentence = _sentence_for_match(text, match.start(), match.end())
        if _official_sentence_outside_task_years(sentence, context):
            continue
        death_mentions.append(
            {
                "value": 1.0,
                "span": "died on " + match.group("date"),
                "sentence": sentence,
                "priority": 3,
            }
        )
    case = None
    if case_mentions:
        case = max(
            case_mentions,
            key=lambda item: (item["priority"], item["value"], len(item["sentence"])),
        )
    death = None
    if death_mentions:
        death = max(
            death_mentions,
            key=lambda item: (item["priority"], item["value"], len(item["sentence"])),
        )
    failure = None
    if not case and not death:
        numeric_tokens = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b", text)
        word_tokens = [
            word
            for word in _OFFICIAL_OUTBREAK_WORD_NUMBERS
            if re.search(rf"\b{re.escape(word)}\b", _lower(text))
        ]
        if numeric_tokens or word_tokens:
            failure = "ambiguous_numeric_context"
        else:
            failure = "no_case_or_death_signal"
    return case, death, failure



def _official_best_count_mentions(text: str, context: dict | None = None):
    legacy_case, legacy_death, failure = _legacy_official_best_count_mentions(text, context)
    if get_env("PIPELINE_MODE") != "evidence":
        return legacy_case, legacy_death, failure
    from ..source_assertions import count_mentions
    cases, deaths = [], []
    # Explicit named patients and written-out numbers retain their old route.
    if legacy_case and (not re.search(r"\d", legacy_case["span"]) or
                        _OFFICIAL_CASE_LABEL_RE.fullmatch(legacy_case["span"])):
        cases.append(legacy_case)
    if legacy_death and (not re.search(r"\d", legacy_death["span"]) or
                         legacy_death["span"].startswith("died on ")):
        deaths.append(legacy_death)
    for mention in count_mentions(text):
        if mention["field"] not in {"cases_confirmed", "cases_probable", "cases_suspected", "cases_unspecified", "deaths"}:
            continue
        if _official_sentence_outside_task_years(mention["sentence"], context):
            continue
        kind = "deaths" if mention["field"] == "deaths" else "cases"
        if _official_match_disallowed(mention["sentence"], kind):
            continue
        item = {**mention, "priority": 1 + 2 * bool(re.search(r"\b(?:total|aggregate|cumulative)\b", mention["sentence"], re.I))}
        (deaths if kind == "deaths" else cases).append(item)
    key = lambda item: (item["priority"], item["value"], len(item["sentence"]))
    case, death = max(cases, key=key) if cases else None, max(deaths, key=key) if deaths else None
    return case, death, None if case or death else failure or "ambiguous_numeric_context"


_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _iso_date_from_parts(day: str, month: str, year: str) -> str | None:
    try:
        return date(int(year), _MONTHS[month.lower()], int(day)).isoformat()
    except (KeyError, TypeError, ValueError):
        return None


def _infer_year_from_text(text: str) -> str | None:
    match = re.search(r"\b(20\d{2})\b", text)
    return match.group(1) if match else None


def _parse_official_date_text(value: str, surrounding_text: str) -> str | None:
    value = str(value or "").strip()
    if not value:
        return None
    match = _OFFICIAL_DAY_MONTH_YEAR_RE.search(value)
    if match:
        return _iso_date_from_parts(
            match.group("day"), match.group("month"), match.group("year")
        )
    match = _OFFICIAL_MONTH_DAY_YEAR_RE.search(value)
    if match:
        return _iso_date_from_parts(
            match.group("day"), match.group("month"), match.group("year")
        )
    match = _OFFICIAL_DAY_MONTH_RE.search(value)
    if match:
        year = _infer_year_from_text(surrounding_text)
        if year:
            return _iso_date_from_parts(match.group("day"), match.group("month"), year)
    return None


def _official_dates(text: str, title: str, context: dict | None = None) -> dict:
    combined = f"{title}\n{text}"
    result = {
        "date_reported": None,
        "report_date": None,
        "publication_date": None,
        "as_of_date": None,
        "event_start_date": None,
        "event_end_date": None,
        "date_text_span": None,
        "date_anchor_type": None,
        "period_overlap_status": None,
    }
    if get_env("PIPELINE_MODE") == "evidence":
        from ..source_assertions import observation_dates
        result.update(observation_dates(text))
        for field,role in (("as_of_date",r"as\s+of"),("date_reported",r"(?:reported\s+on|report\s+date\s*[:=]?)")):
            match = re.search(r"\b"+role+r"\s+(?P<date>\d{4}-\d{2}-\d{2})\b",text,re.I)
            if match:
                try:
                    date.fromisoformat(match.group("date"))
                except ValueError:
                    continue
                result[field] = match.group("date")
        return result
    range_match = _OFFICIAL_DATE_RANGE_RE.search(combined)
    if range_match:
        result["event_start_date"] = _parse_official_date_text(
            range_match.group("start"), combined
        )
        result["event_end_date"] = _parse_official_date_text(
            range_match.group("end"), combined
        )
        result["date_text_span"] = range_match.group(0)

    as_of_match = _OFFICIAL_AS_OF_DATE_RE.search(combined)
    if as_of_match:
        result["as_of_date"] = _parse_official_date_text(
            as_of_match.group("date"), combined
        )
        result["date_reported"] = result["as_of_date"]
        result["date_anchor_type"] = "as_of_date"
        result["date_text_span"] = result["date_text_span"] or as_of_match.group(0)

    for pattern in (_OFFICIAL_DAY_MONTH_YEAR_RE, _OFFICIAL_MONTH_DAY_YEAR_RE):
        for match in pattern.finditer(title):
            parsed = _parse_official_date_text(match.group(0), title)
            if parsed:
                result["report_date"] = parsed
                result["publication_date"] = parsed
                result["date_reported"] = result["date_reported"] or parsed
                result["date_anchor_type"] = result["date_anchor_type"] or "report_date"
                result["date_text_span"] = result["date_text_span"] or match.group(0)
                break
        if result["report_date"]:
            break

    if not result["date_reported"]:
        for pattern in (_OFFICIAL_DAY_MONTH_YEAR_RE, _OFFICIAL_MONTH_DAY_YEAR_RE):
            match = pattern.search(combined)
            if match:
                parsed = _parse_official_date_text(match.group(0), combined)
                if parsed:
                    result["date_reported"] = parsed
                    result["date_anchor_type"] = "report_date"
                    result["date_text_span"] = result["date_text_span"] or match.group(0)
                    break

    start = result.get("event_start_date")
    end = result.get("event_end_date")
    task = (context or {}).get("structured_task") or {}
    task_start = str(task.get("start_date") or "")
    task_end = str(task.get("end_date") or "")
    if start and end and task_start and task_end:
        if end < task_start or start > task_end:
            result["period_overlap_status"] = "outside_time_window"
        elif start < task_start or end > task_end:
            result["period_overlap_status"] = "partial_overlap"
        else:
            result["period_overlap_status"] = "within_time_window"
    return result


def _official_geography(text: str, title: str, context: dict | None = None) -> dict:
    combined = f"{title}\n{text}"
    if get_env("PIPELINE_MODE") == "evidence":
        from ..geography import explicit_country_matches
        from ..evidence_qualification import _role_bound_occurrence
        local_countries = [(name,key) for name,key in explicit_country_matches(text)
                           if _role_bound_occurrence("country",name,text)]
        keys = {key for _,key in local_countries}
        if len(keys) == 1:
            country = local_countries[0][0]
            return {"country": country, "geographic_scope": country, "geographic_scope_type": "country",
                    "location_type": "country", "geography_text_span": country, "associated_countries": [country]}
        if len(keys) > 1:
            return {"associated_countries": list(dict.fromkeys(name for name,_ in local_countries))}
    vessel_match = _OFFICIAL_VESSEL_RE.search(combined)
    # Explicit non-national scope remains available. The legacy country list
    # cannot restore a location rejected by the evidence source-local role check.
    associated_countries = [] if get_env("PIPELINE_MODE") == "evidence" else [
        country
        for country in _OFFICIAL_COUNTRY_NAMES
        if re.search(rf"\b{re.escape(country)}\b", combined, flags=re.IGNORECASE)
    ]
    if vessel_match:
        return {
            "locality": vessel_match.group(0),
            "geographic_scope": f"{vessel_match.group(0)} cruise ship",
            "geographic_scope_type": "vessel",
            "location_type": "vessel_associated",
            "record_geography_fit_status": (
                "vessel_or_travel_associated_within_global_scope"
                if _lower(_task_location_from_context(context)) == "global"
                else None
            ),
            "geography_text_span": vessel_match.group(0),
            "associated_countries": associated_countries,
            "travel_or_vessel_context": "cruise ship travel",
        }
    lowered = _lower(combined)
    if "cruise ship" in lowered or "travel-associated" in lowered:
        return {
            "locality": None,
            "geographic_scope": "travel-associated outbreak",
            "geographic_scope_type": "travel_associated",
            "location_type": "travel_associated",
            "record_geography_fit_status": (
                "vessel_or_travel_associated_within_global_scope"
                if _lower(_task_location_from_context(context)) == "global"
                else None
            ),
            "geography_text_span": "cruise ship"
            if "cruise ship" in lowered
            else "travel-associated",
            "associated_countries": associated_countries,
            "travel_or_vessel_context": "cruise ship travel",
        }
    if "americas region" in lowered or "region of the americas" in lowered:
        return {
            "locality": None,
            "geographic_scope": "Region of the Americas",
            "geographic_scope_type": "region",
            "location_type": "regional_aggregate",
            "record_geography_fit_status": (
                "regional_within_global_scope"
                if _lower(_task_location_from_context(context)) == "global"
                else None
            ),
            "geography_text_span": "Americas Region",
            "associated_countries": associated_countries,
            "travel_or_vessel_context": None,
        }
    if associated_countries:
        return {
            "country": associated_countries[0],
            "geographic_scope": associated_countries[0],
            "geographic_scope_type": "country",
            "location_type": "country",
            "record_geography_fit_status": (
                "country_within_global_scope"
                if _lower(_task_location_from_context(context)) == "global"
                else None
            ),
            "geography_text_span": associated_countries[0],
            "associated_countries": associated_countries,
            "travel_or_vessel_context": None,
        }
    return {"associated_countries": associated_countries}


def _official_outbreak_evidence_quote(
    text: str,
    case_mention: dict | None,
    death_mention: dict | None,
) -> str:
    sentences = []
    for mention in (case_mention, death_mention):
        sentence = (mention or {}).get("sentence")
        if sentence and sentence not in sentences:
            sentences.append(sentence)
    quote = " ".join(sentences).strip()
    if quote:
        return quote[:1500]
    return re.sub(r"\s+", " ", text).strip()[:1500]


def _official_case_spans(text: str) -> list[dict]:
    """Return explicit case-label spans without mixing adjacent cases."""

    matches = []
    for match in _OFFICIAL_CASE_LABEL_RE.finditer(text):
        prefix = text[max(0, match.start() - 24) : match.start()].lower()
        suffix = text[match.end() : match.end() + 16]
        if re.search(r"(?:of|with|from)\s+$", prefix):
            continue
        if re.search(r"(?:contact|contacts|exposure)\s+(?:of|with)\s+$", prefix):
            continue
        if suffix and not re.match(r"\s*(?::|-|,|\.|\b(?:developed|reported|had|was|were|is|are|died|presented|travelled|traveled|boarded)\b)", suffix, re.IGNORECASE):
            continue
        matches.append(match)
    if not matches:
        return []
    spans: list[dict] = []
    for index, match in enumerate(matches, start=1):
        start = match.start()
        end = matches[index].start() if index < len(matches) else len(text)
        quote = re.sub(r"\s+", " ", text[start:end]).strip()
        if not quote:
            continue
        label = re.sub(r"\s+", " ", match.group("label")).strip()
        span_slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
        spans.append(
            {
                "case_span_id": f"case_span_{index:03d}_{span_slug}",
                "case_span_quote": quote,
                "case_span_start": start,
                "case_span_end": end,
                "case_span_extraction_method": "deterministic_case_label_span",
            }
        )
    return spans


def _expand_case_span_chunks_for_llm(
    chunks: list[dict],
    exclude_parent_chunk_ids: set[str] | None = None,
) -> list[dict]:
    """Replace multi-case narrative chunks with one LLM input per case span."""

    excluded = {str(value) for value in (exclude_parent_chunk_ids or set())}
    expanded: list[dict] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        parent_chunk_id = str(chunk.get("chunk_id") or "")
        if parent_chunk_id in excluded or chunk.get("case_span_id"):
            expanded.append(chunk)
            continue
        spans = _official_case_spans(str(chunk.get("text") or ""))
        if not spans:
            expanded.append(chunk)
            continue
        for span in spans:
            clone = dict(chunk)
            clone.update(span)
            clone["parent_chunk_id"] = parent_chunk_id or None
            clone["chunk_id"] = (
                f"{parent_chunk_id}__{span['case_span_id']}"
                if parent_chunk_id
                else str(span["case_span_id"])
            )
            clone["text"] = span["case_span_quote"]
            clone["case_span_chunk"] = True
            expanded.append(clone)
    return expanded


def _official_outbreak_diagnostic_row(
    chunk: dict,
    *,
    relevant: bool,
    attempted: bool,
    raw_count: int,
    primary_count: int,
    task_aware_count: int,
    quarantine_count: int,
    failure_substage: str | None,
    error: str | None = None,
) -> dict:
    return {
        "source_id": chunk.get("source_id"),
        "chunk_id": chunk.get("chunk_id"),
        "source_url": chunk.get("source_url") or chunk.get("canonical_url"),
        "title": chunk.get("title"),
        "source_type": chunk.get("source_type_final") or chunk.get("source_type"),
        "evidence_chunk_count": 1,
        "relevant_chunk_count": 1 if relevant else 0,
        "extraction_attempted": attempted,
        "extraction_method": "deterministic_official_outbreak_page_extractor"
        if attempted
        else None,
        "raw_records_count": raw_count,
        "primary_candidate_count": primary_count,
        "task_aware_observation_count": task_aware_count,
        "quarantine_count": quarantine_count,
        "extraction_error": error,
        "failure_substage": failure_substage,
    }


def _normalize_official_case_span_text(text: str) -> str:
    """Repair common PDF spacing artifacts before case-local parsing."""

    normalized = str(text or "").replace("\u2019", "'").replace("\u2018", "'")
    normalized = re.sub(
        r"\b(?P<age>\d{1,3})\s*-\s*year\s*-\s*old\b",
        r"\g<age>-year-old",
        normalized,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _official_line_list_details(text: str) -> dict:
    text = _normalize_official_case_span_text(text)
    details: dict[str, object] = {
        "workflow_case_label": None,
        "age": None,
        "gender": None,
        "nationality": None,
        "date_onset": None,
        "date_confirmation": None,
        "date_death": None,
        "outcome": None,
        "symptoms": None,
        "hospitalized": None,
        "intensive_care": None,
        "isolated": None,
        "occupation_or_role": None,
        "cruise_crew": None,
        "cruise_passenger_guest": None,
        "contact_with_case": None,
        "contact_setting": None,
        "travel_from": None,
        "travel_or_vessel_context": None,
        "ship_board_date": None,
        "ship_disembark_date": None,
        "confirmation_method": None,
    }
    label_match = _OFFICIAL_CASE_LABEL_RE.search(text)
    if label_match:
        details["workflow_case_label"] = re.sub(
            r"\s+",
            " ",
            label_match.group("label"),
        ).strip()

    lowered = _lower(text)
    numeric_age_match = re.search(r"\b(?P<age>\d{1,3})-year-old\b", lowered)
    if numeric_age_match:
        details["age"] = numeric_age_match.group("age")
    elif re.search(r"\badult\b", lowered):
        details["age"] = "adult"
    if re.search(r"\bfemale\b", lowered):
        details["gender"] = "female"
    elif re.search(r"\bmale\b", lowered):
        details["gender"] = "male"

    nationality_match = re.search(
        r"\b(?P<nationality>[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+)?)\s+"
        r"(?:male|female)\s+national\b",
        text,
    )
    if nationality_match:
        details["nationality"] = nationality_match.group("nationality")

    died_match = _OFFICIAL_DIED_RE.search(text)
    if died_match:
        details["outcome"] = "death"
        details["date_death"] = _parse_official_date_text(died_match.group("date"), text)

    if details.get("outcome") is None and re.search(r"\brecovered\b", lowered):
        details["outcome"] = "recovered"

    date_pattern = (
        r"(?:\d{1,2}\s+[A-Z][a-z]+\s+\d{4})|"
        r"(?:[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})|"
        r"(?:\d{1,2}\s+[A-Z][a-z]+)"
    )
    onset_patterns = (
        rf"\b(?:onset of symptoms was on|onset of symptoms on|symptom onset on)\s+(?P<date>{date_pattern})",
        rf"\b(?:reported onset of symptoms on|onset of symptoms was reported on)\s+(?P<date>{date_pattern})",
        rf"\bdeveloped\s+symptoms\s+on\s+(?P<date>{date_pattern})",
        rf"\bdeveloped\b.{{0,160}}?\bon\s+(?P<date>{date_pattern})",
        rf"\bpresented\b.*?\bon\s+(?P<date>{date_pattern})",
    )
    for pattern in onset_patterns:
        onset_match = re.search(pattern, text, re.IGNORECASE)
        if onset_match:
            details["date_onset"] = _parse_official_date_text(
                onset_match.group("date"), text
            )
            break

    confirmation_patterns = (
        rf"\bconfirmed\s+by\s+(?P<method>PCR|RT-PCR|serology|laboratory testing|laboratory test)\s+on\s+(?P<date>{date_pattern})",
        rf"\b(?P<method>PCR|RT-PCR|serology|laboratory testing|laboratory test)\s+"
        rf"confirmed\b.{{0,120}}?\bon\s+(?P<date>{date_pattern})",
        rf"\b(?P<method>PCR|RT-PCR|serology|laboratory)\s+(?:testing|test)\s+confirmed\b.{{0,120}}?\bon\s+(?P<date>{date_pattern})",
        rf"\blaboratory\s+testing\s+confirmed\b.{{0,120}}?\bon\s+(?P<date>{date_pattern})",
        rf"\b(?:his\s+|her\s+|the\s+)?(?:laboratory\s+)?samples?\s+confirmed\s+(?P<method>PCR|RT-PCR)\s+positivity\b.{{0,120}}?\bon\s+(?P<date>{date_pattern})",
        rf"\b(?:hantavirus|Andes\s+virus|ANDV)\s+infection\s+was\s+confirmed\s+on\s+(?P<date>{date_pattern})",
        rf"\bon\s+(?P<date>{date_pattern})\s*,?\s+"
        rf"(?:(?:the\s+case|he|she|it)\s+)?(?:was\s+)?(?:subsequently\s+)?"
        rf"confirmed\s+by\s+(?P<method>PCR|RT-PCR|serology|laboratory testing|laboratory test)\b",
    )
    confirmation_match = None
    for pattern in confirmation_patterns:
        confirmation_match = re.search(pattern, text, re.IGNORECASE)
        if confirmation_match:
            break
    if confirmation_match:
        method = confirmation_match.groupdict().get("method") or "laboratory testing"
        details["confirmation_method"] = "PCR" if method.upper() in {"PCR", "RT-PCR"} else method
        details["date_confirmation"] = _parse_official_date_text(
            confirmation_match.group("date"), text
        )

    if re.search(r"\bhospitali[sz]ed\b|\bhospitali[sz]ation\b", lowered):
        details["hospitalized"] = True
    if re.search(r"\bintensive care\b|\bICU\b", text, re.IGNORECASE):
        details["intensive_care"] = True
    if re.search(r"\bisolated\b|\bisolation\b", lowered):
        details["isolated"] = True
    role_match = re.search(
        r"\b(?:serving|employed|working)\s+as\s+(?:the|a|an)\s+"
        r"(?P<role>ship's\s+doctor|doctor|guide|steward(?:ess)?|ornithologist|"
        r"medical\s+officer|captain|crew\s+member)\b",
        text,
        re.IGNORECASE,
    )
    if role_match:
        role = re.sub(r"\s+", " ", role_match.group("role").lower()).strip()
        if role == "ship's doctor":
            role = "ship doctor"
        details["occupation_or_role"] = role
        if role in {
            "ship doctor",
            "doctor",
            "guide",
            "steward",
            "stewardess",
            "medical officer",
            "captain",
            "crew member",
        }:
            details["cruise_crew"] = True
            details["cruise_passenger_guest"] = False
    elif re.search(r"\bcrew(?:\s+member)?\b", lowered):
        details["occupation_or_role"] = "crew"
        details["cruise_crew"] = True
        details["cruise_passenger_guest"] = False
    elif re.search(r"\bpassenger\b|\bguest\b|\btraveller\b|\btraveler\b", lowered):
        details["occupation_or_role"] = "passenger"
        details["cruise_passenger_guest"] = True
        details["cruise_crew"] = False
    contact_match = re.search(
        r"\bclose contact of\s+(?P<label>Case\s+\d+|Cases\s+\d+(?:\s*(?:,|and)\s*\d+)*)",
        text,
        re.IGNORECASE,
    )
    if contact_match:
        details["contact_with_case"] = re.sub(
            r"\s+", " ", contact_match.group("label")
        ).strip()
        if re.search(r"\baboard\b|\bcruise ship\b|\bMV\s+Hondius\b", text, re.IGNORECASE):
            details["contact_setting"] = "aboard cruise ship"

    symptom_terms = []
    for term in (
        "pneumonia",
        "fever",
        "general feeling of being unwell",
        "malaise",
        "respiratory distress",
        "respiratory symptoms",
        "shortness of breath",
        "headache",
        "fatigue",
        "myalgia",
        "diarrhoea",
        "diarrhea",
        "nausea",
        "dizziness",
        "tachycardia",
        "tachypnoea",
        "shock",
        "ards",
    ):
        if term in lowered:
            symptom_terms.append(term)
    if symptom_terms:
        details["symptoms"] = "; ".join(symptom_terms)

    travel_match = re.search(
        r"\btravelled in\s+(?P<travel>.*?)(?:,\s*before| before they boarded| before boarding)",
        text,
        re.IGNORECASE,
    )
    if travel_match:
        details["travel_from"] = re.sub(r"\s+", " ", travel_match.group("travel")).strip()
    board_match = re.search(
        r"\bboarded the (?P<vessel>cruise ship(?:\s+MV\s+Hondius|\s+Hondius)?)\s+on\s+"
        r"(?P<date>(?:\d{1,2}\s+[A-Z][a-z]+\s+\d{4})|"
        r"(?:[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})|"
        r"(?:\d{1,2}\s+[A-Z][a-z]+))",
        text,
        re.IGNORECASE,
    )
    if board_match:
        details["travel_or_vessel_context"] = board_match.group("vessel")
        details["ship_board_date"] = _parse_official_date_text(board_match.group("date"), text)
    elif "cruise ship" in lowered or "hondius" in lowered:
        details["travel_or_vessel_context"] = "cruise ship"
    disembark_match = re.search(
        r"\bdisembark(?:ed)?\s+(?:from\s+the\s+)?(?:cruise ship\s+)?(?:MV\s+Hondius\s+)?on\s+"
        r"(?P<date>(?:\d{1,2}\s+[A-Z][a-z]+\s+\d{4})|"
        r"(?:[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})|"
        r"(?:\d{1,2}\s+[A-Z][a-z]+))",
        text,
        re.IGNORECASE,
    )
    if disembark_match:
        details["ship_disembark_date"] = _parse_official_date_text(
            disembark_match.group("date"), text
        )

    return {key: value for key, value in details.items() if value not in (None, "")}


def _recover_case_fields_from_local_span(
    cleaned: dict,
    case_span_quote: str | None,
) -> tuple[dict, list[str], dict[str, str]]:
    """Fill and validate case fields against one explicit local span."""

    if not case_span_quote:
        return cleaned, [], {}
    recovered = _official_line_list_details(case_span_quote)
    if not recovered.get("workflow_case_label"):
        return cleaned, [], {}

    unsupported: list[str] = []
    methods: dict[str, str] = {}
    for field_name, recovered_value in recovered.items():
        existing_value = cleaned.get(field_name)
        if existing_value not in (None, "", [], {}) and existing_value != recovered_value:
            unsupported.append(field_name)
        if existing_value in (None, "", [], {}) or existing_value != recovered_value:
            cleaned[field_name] = recovered_value
            methods[field_name] = "deterministic_case_span_field_recovery"
    return cleaned, sorted(set(unsupported)), methods


def _official_literal_reporting_period(text: str) -> str | None:
    """Retain a uniquely stated observation year/range without date expansion."""
    periods = []
    for match in re.finditer(r"\b(?:in|during|en)\s+(\d{4}(?:\s*(?:-|\u2013|to)\s*\d{4})?)(?![\d/-])", text, re.I):
        if re.search(r"\b(?:published|updated|released|posted)\s*$",text[:match.start()],re.I):
            continue
        if re.match(r"\s+(?:cases?|deaths?|patients?|tests?)\b",text[match.end():],re.I):
            continue
        periods.append(match.group(1))
    return periods[0] if len(set(periods)) == 1 else None


def _verified_rule_headings(chunk: dict, context: dict | None) -> list[dict]:
    """Return only parser-bound headings from this exact document and chunk."""
    from ..evidence_chunking import validate_bound_context, quote_within_chunk
    spans = chunk.get("bound_context_spans") or []
    if not spans:
        return []
    docs = ((context or {}).get("documents_by_source_id") or {}).get(str(chunk.get("source_id") or ""), [])
    matches = [doc for doc in docs if doc.get("document_id") == chunk.get("document_id") and
               doc.get("content_hash") == chunk.get("document_hash")]
    if len(matches) != 1:
        return []
    doc = matches[0]
    text = str(doc.get("clean_text") or "")
    if sha256(text.encode()).hexdigest() != str(doc.get("text_hash") or doc.get("content_hash") or "").removeprefix("sha256:"):
        return []
    left,right = chunk.get("char_start"),chunk.get("char_end")
    if not (isinstance(left,int) and isinstance(right,int) and
            quote_within_chunk(doc,chunk,left,right,[]) and validate_bound_context(doc,chunk,spans)):
        return []
    return [span for span in spans if span.get("role") == "heading"]


def _official_outbreak_record_from_chunk(
    chunk: dict,
    index: int,
    policy: StructuredExtractionPolicy,
    context: dict | None = None,
) -> tuple[PublicHealthRecord | None, dict]:
    text = str(chunk.get("text") or chunk.get("row_quote") or "").strip()
    title = str(chunk.get("title") or "").strip()
    from ..evidence_qualification import evidence_qualification_enabled
    universal = evidence_qualification_enabled()
    headings = _verified_rule_headings(chunk,context) if universal else []
    heading_text = "\n".join(span["quote"] for span in headings)
    combined = f"{heading_text if universal else title}\n{text}"
    relevant, matched_alias = _official_hantavirus_alias_match(combined, context)
    if universal:
        for heading in reversed(headings):
            assessment = assess_chunk_disease_relevance({"text":heading["quote"],"title":""},_as_disease_relevance_context(context))
            if assessment.get("target_disease_terms_found") or assessment.get("incompatible_disease_terms_found"):
                if not assessment.get("target_disease_terms_found"):
                    relevant = False
                break
    if not relevant:
        return None, _official_outbreak_diagnostic_row(
            chunk,
            relevant=False,
            attempted=True,
            raw_count=0,
            primary_count=0,
            task_aware_count=0,
            quarantine_count=0,
            failure_substage="source_found_but_no_relevant_chunks",
        )

    case_mention, death_mention, count_failure = _official_best_count_mentions(
        text,
        context,
    )
    if not case_mention and not death_mention:
        return None, _official_outbreak_diagnostic_row(
            chunk,
            relevant=True,
            attempted=True,
            raw_count=0,
            primary_count=0,
            task_aware_count=0,
            quarantine_count=0,
            failure_substage=count_failure or "no_case_or_death_signal",
        )

    date_title_context = " ".join(
        str(value or "")
        for value in (title, chunk.get("source_url"), chunk.get("canonical_url"))
    )
    if universal and case_mention and death_mention and case_mention.get("sentence") != death_mention.get("sentence"):
        # Two independently selected maxima do not establish one observation.
        # The model can recover both statements with their own scope/period.
        death_mention = None
    local_evidence = _official_outbreak_evidence_quote(text, case_mention, death_mention)
    dates = _official_dates(local_evidence if universal else text, "" if universal else date_title_context, context)
    geography = _official_geography(local_evidence if universal else text, "" if universal else title, context)
    if universal and not geography.get("associated_countries") and not geography.get("geographic_scope"):
        geography = _official_geography(heading_text,"",context)
    if universal and len(geography.get("associated_countries") or []) > 1:
        # A list of countries is not a binding from the selected count to one.
        geography = {"associated_countries": geography["associated_countries"]}
    warnings = ["official_single_source_not_cross_validated"]
    uncertainty_flags = ["official_single_source_not_cross_validated"]
    if dates.get("period_overlap_status") == "partial_overlap":
        warnings.append("partial_time_window_overlap")
        uncertainty_flags.append("partial_time_window_overlap")
    elif dates.get("period_overlap_status") == "outside_time_window":
        warnings.append("outside_time_window")
        uncertainty_flags.append("outside_time_window")

    if not (
        geography.get("country")
        or geography.get("geographic_scope")
        or geography.get("locality")
    ):
        warnings.append("missing_geography_anchor")
        uncertainty_flags.append("missing_geography_anchor")

    source_id = str(chunk.get("source_id") or "")
    evidence_quote = _official_outbreak_evidence_quote(text, case_mention, death_mention)
    compatibility = assess_record_disease_compatibility(
        {
            "disease": (context or {}).get("disease_standard_name")
            or policy.default_disease,
            "evidence_quote": evidence_quote,
            "heading_context": heading_text if universal else None,
            "source_title": title,
            "source_url": chunk.get("source_url"),
        },
        _as_disease_relevance_context(context),
    )
    source_type = _official_outbreak_source_type(chunk, context) or chunk.get(
        "source_type"
    )
    source_type_final = chunk.get("source_type_final") or chunk.get("source_type")
    canonical_disease = (
        (context or {}).get("disease_standard_name") or policy.default_disease
    )
    cases_unspecified = case_mention.get("value") if case_mention else None
    case_field = _classify_case_bucket(case_mention.get("span") or "") if universal and case_mention else "cases_unspecified"
    case_values = {field:None for field in ("cases_confirmed","cases_probable","cases_suspected","cases_unspecified")}
    case_values[case_field] = cases_unspecified
    deaths = death_mention.get("value") if death_mention else None
    line_list_details = _official_line_list_details(local_evidence if universal else text)
    case_span_quote = str(chunk.get("case_span_quote") or (local_evidence if universal else text)).strip()
    case_span_id = chunk.get("case_span_id")
    case_span_method = (
        chunk.get("case_span_extraction_method")
        or "deterministic_official_case_span"
    )
    field_provenance = {
        field_name: {
            "field_name": field_name,
            "extracted_value": value,
            "source_id": source_id,
            "chunk_id": chunk.get("chunk_id"),
            "case_span_id": case_span_id,
            "supporting_quote": case_span_quote,
            "extraction_method": case_span_method,
        }
        for field_name, value in line_list_details.items()
        if value not in (None, "", [], {})
    }
    pathogen_text = local_evidence if universal else combined
    record = PublicHealthRecord(
        record_id=f"rec_{source_id}_official_outbreak_{index:03d}",
        disease=canonical_disease,
        disease_standard_name=canonical_disease,
        canonical_disease=canonical_disease,
        disease_alias_used=matched_alias,
        disease_alias_match=True,
        matched_alias=matched_alias,
        virus_or_syndrome=(
            "Andes virus"
            if matched_alias in {"Andes virus", "ANDV"}
            or re.search(r"\b(?:Andes virus|ANDV)\b", pathogen_text, re.IGNORECASE)
            else _detect_virus_or_syndrome(pathogen_text, policy)
        ),
        pathogen_or_syndrome=_detect_pathogen_or_syndrome(pathogen_text, context),
        target_population=(context or {}).get("target_population"),
        observation_type="unspecified_case_record",
        observation_types=["unspecified_case_record"],
        primary_case_dataset_eligible=True,
        country=geography.get("country"),
        subnational_location=geography.get("subnational_location"),
        locality=geography.get("locality"),
        geographic_scope=geography.get("geographic_scope"),
        geographic_scope_type=geography.get("geographic_scope_type"),
        location_type=geography.get("location_type"),
        record_geography_fit_status=geography.get("record_geography_fit_status"),
        associated_countries=list(geography.get("associated_countries") or []),
        travel_or_vessel_context=line_list_details.get("travel_or_vessel_context")
        or geography.get("travel_or_vessel_context"),
        geography_text_span=geography.get("geography_text_span"),
        date_reported=dates.get("date_reported"),
        event_start_date=dates.get("event_start_date"),
        event_end_date=dates.get("event_end_date"),
        date_onset=line_list_details.get("date_onset"),
        date_confirmation=line_list_details.get("date_confirmation"),
        date_death=line_list_details.get("date_death"),
        report_date=dates.get("report_date"),
        publication_date=dates.get("publication_date"),
        as_of_date=dates.get("as_of_date"),
        metric_period_start=dates.get("metric_period_start"),
        metric_period_end=dates.get("metric_period_end"),
        date_anchor_type=dates.get("date_anchor_type"),
        date_text_span=dates.get("date_text_span"),
        period_overlap_status=dates.get("period_overlap_status"),
        cases_confirmed=case_values["cases_confirmed"],
        cases_probable=case_values["cases_probable"],
        cases_suspected=case_values["cases_suspected"],
        cases_unspecified=case_values["cases_unspecified"],
        deaths=deaths,
        case_definition=None if universal else "unspecified",
        age=line_list_details.get("age"),
        gender=line_list_details.get("gender"),
        nationality=line_list_details.get("nationality"),
        outcome=line_list_details.get("outcome"),
        symptoms=line_list_details.get("symptoms"),
        hospitalized=line_list_details.get("hospitalized"),
        intensive_care=line_list_details.get("intensive_care"),
        isolated=line_list_details.get("isolated"),
        occupation_or_role=line_list_details.get("occupation_or_role"),
        cruise_crew=line_list_details.get("cruise_crew"),
        cruise_passenger_guest=line_list_details.get("cruise_passenger_guest"),
        contact_with_case=line_list_details.get("contact_with_case"),
        contact_setting=line_list_details.get("contact_setting"),
        travel_from=line_list_details.get("travel_from"),
        ship_board_date=line_list_details.get("ship_board_date"),
        ship_disembark_date=line_list_details.get("ship_disembark_date"),
        confirmation_method=line_list_details.get("confirmation_method"),
        workflow_case_label=line_list_details.get("workflow_case_label"),
        case_span_id=case_span_id,
        case_span_quote=case_span_quote,
        case_span_start=chunk.get("case_span_start"),
        case_span_end=chunk.get("case_span_end"),
        case_span_extraction_method=case_span_method,
        field_provenance_json=(
            json.dumps(field_provenance, ensure_ascii=False, sort_keys=True)
            if field_provenance
            else None
        ),
        unsupported_case_fields=[],
        statistical_count_type=None if universal else "event_total",
        count_semantics=None if universal else "event_total",
        count_value_raw=str(case_mention.get("value")) if case_mention else None,
        count_unit="cases" if case_mention else "deaths",
        count_confidence=0.78,
        count_notes="official outbreak page deterministic extraction",
        reporting_period=(dates.get("reporting_period") or _official_literal_reporting_period(local_evidence)) if universal else None,
        source_id=source_id,
        source_url=chunk.get("source_url") or chunk.get("canonical_url"),
        source_type=source_type,
        source_type_final=source_type_final,
        evidence_quote=evidence_quote,
        heading_context=heading_text if universal else None,
        case_count_text_span=case_mention.get("span") if case_mention else None,
        death_count_text_span=death_mention.get("span") if death_mention else None,
        extraction_confidence=float(chunk.get("confidence") or 0.78),
        missing_fields=[],
        schema_status=None,
        provenance_status=None,
        supporting_chunk_id=chunk.get("chunk_id"),
        source_title=title,
        publisher=chunk.get("publisher") or chunk.get("actual_publisher"),
        actual_publisher=chunk.get("actual_publisher") or chunk.get("publisher"),
        actual_publisher_normalized=chunk.get("actual_publisher_normalized"),
        source_role_final=chunk.get("source_role_final"),
        credibility_score=chunk.get("credibility_score"),
        credibility_level=chunk.get("credibility_level"),
        source_independence_group=chunk.get("source_independence_group"),
        claim_support_role=chunk.get("claim_support_role"),
        recommended_source_role=chunk.get("recommended_source_role"),
        recommended_fetch_use=chunk.get("recommended_fetch_use"),
        recommended_extraction_use=chunk.get("recommended_extraction_use"),
        likely_syndicated_or_aggregated=chunk.get("likely_syndicated_or_aggregated"),
        upstream_source_mentions=list(chunk.get("upstream_source_mentions") or []),
        discovery_method=chunk.get("discovery_method"),
        search_provider=chunk.get("search_provider"),
        query_id=chunk.get("query_id"),
        query_used=chunk.get("query_used"),
        document_id=chunk.get("document_id"),
        document_type=chunk.get("document_type"),
        fetch_purpose=chunk.get("fetch_purpose"),
        chunk_kind=chunk.get("chunk_kind") or "text",
        data_types=list(chunk.get("data_types") or []),
        context_types=list(chunk.get("context_types") or []),
        extraction_method="deterministic_official_outbreak_page_extractor",
        extraction_reason="deterministic extraction from official outbreak page",
        validation_errors=[],
        repair_actions=[],
        requires_human_review=True,
        human_review_reason="official_single_source_not_cross_validated",
        review_status="pending_review",
        quality_status="official_single_source_not_cross_validated",
        uncertainty_flags=uncertainty_flags,
        semantic_warnings=warnings,
        extraction_warnings=warnings,
        llm_used=False,
        llm_model=None,
        llm_provider=None,
        llm_extraction_error=None,
        extraction_mode="deterministic",
        record_schema="official_outbreak_page_extraction_schema",
        legacy_record_type=(
            "ObservationRecord" if ((context or {}).get("is_hantavirus") is True) else None
        ),
        corroboration_status="single_source_unverified",
        corroboration_reason="official_single_source_not_cross_validated",
        independent_source_count=1,
        official_source_support_count=1,
        secondary_source_support_count=0,
        **record_compatibility_fields(compatibility),
    )
    if universal:
        from ..record_identity import stable_record_id
        record.record_id = stable_record_id(record, chunk)
    return record, _official_outbreak_diagnostic_row(
        chunk,
        relevant=True,
        attempted=True,
        raw_count=1,
        primary_count=1,
        task_aware_count=0,
        quarantine_count=0,
        failure_substage=None,
    )


def _official_outbreak_records_from_chunk(
    chunk: dict,
    start_index: int,
    policy: StructuredExtractionPolicy,
    context: dict | None = None,
) -> tuple[list[PublicHealthRecord], dict]:
    text = str(chunk.get("text") or chunk.get("row_quote") or "").strip()
    spans = _official_case_spans(text)
    if not spans:
        record, diagnostic = _official_outbreak_record_from_chunk(
            chunk,
            start_index,
            policy,
            context,
        )
        return ([record] if record is not None else []), diagnostic

    records: list[PublicHealthRecord] = []
    diagnostics: list[dict] = []
    for offset, span in enumerate(spans):
        span_chunk = dict(chunk)
        span_chunk.update(span)
        span_chunk["text"] = span["case_span_quote"]
        record, diagnostic = _official_outbreak_record_from_chunk(
            span_chunk,
            start_index + offset,
            policy,
            context,
        )
        diagnostics.append(diagnostic)
        if record is not None:
            records.append(record)

    if records:
        return records, _official_outbreak_diagnostic_row(
            chunk,
            relevant=True,
            attempted=True,
            raw_count=len(records),
            primary_count=len(records),
            task_aware_count=0,
            quarantine_count=0,
            failure_substage=None,
        )
    failure = next(
        (
            diagnostic.get("failure_substage")
            for diagnostic in diagnostics
            if diagnostic.get("failure_substage")
        ),
        "case_spans_found_but_no_records_extracted",
    )
    return [], _official_outbreak_diagnostic_row(
        chunk,
        relevant=any(diagnostic.get("relevant_chunk_count") for diagnostic in diagnostics),
        attempted=True,
        raw_count=0,
        primary_count=0,
        task_aware_count=0,
        quarantine_count=sum(
            int(diagnostic.get("quarantine_count") or 0) for diagnostic in diagnostics
        ),
        failure_substage=failure,
    )


def extract_official_outbreak_records_from_chunks(
    evidence_chunks: list[dict],
    *,
    policy: StructuredExtractionPolicy,
    context: dict | None = None,
    start_index: int = 1,
) -> tuple[list[PublicHealthRecord], list[dict]]:
    """Extract reviewable primary candidates from official outbreak pages.

    This deterministic repair is source-type aware, but intentionally count- and
    URL-agnostic. It only promotes explicit case/death sentences with preserved
    provenance into raw primary candidates requiring human review.
    """

    if not (context or {}).get("disease_standard_name") or any(
        field not in (context or {}) for field in ("disease_terms", "pathogen_terms")
    ):
        # Public callers may supply task state rather than the prepared context.
        # Derive missing task terms without replacing caller metadata or evidence.
        context = dict(context or {})
        prepared = _build_extraction_context(context, policy)
        if not context.get("disease_standard_name"):
            context["disease_standard_name"] = prepared["disease_standard_name"]
        for field in ("disease_terms", "pathogen_terms"):
            context.setdefault(field, prepared[field])

    records: list[PublicHealthRecord] = []
    diagnostics: list[dict] = []
    index_by_source: dict[str, int] = {}
    for chunk in evidence_chunks:
        if not isinstance(chunk, dict):
            continue
        if not _is_official_outbreak_page_chunk(chunk, context):
            continue
        source_id = str(chunk.get("source_id") or "")
        index_by_source.setdefault(source_id, start_index - 1)
        index_by_source[source_id] += 1
        try:
            chunk_records, diagnostic = _official_outbreak_records_from_chunk(
                chunk,
                index_by_source[source_id],
                policy,
                context,
            )
        except Exception as exc:  # noqa: BLE001 - harness records error detail.
            diagnostic = _official_outbreak_diagnostic_row(
                chunk,
                relevant=True,
                attempted=True,
                raw_count=0,
                primary_count=0,
                task_aware_count=0,
                quarantine_count=1,
                failure_substage="raw_record_extracted_but_schema_invalid",
                error=f"{type(exc).__name__}: {exc}",
            )
            chunk_records = []
        if chunk_records:
            records.extend(chunk_records)
            index_by_source[source_id] += len(chunk_records) - 1
        diagnostics.append(diagnostic)
    return records, diagnostics


def _chunk_with_task_context(chunk: dict, context: dict | None) -> dict:
    updated = dict(chunk)
    contract = (context or {}).get("task_acceptance_contract") or {}
    if contract:
        updated["task_acceptance_contract"] = contract
        updated["record_acceptance_rules"] = list(
            contract.get("record_acceptance_rules") or []
        )
        updated["context_or_quarantine_rules"] = list(
            contract.get("context_or_quarantine_rules") or []
        )
    return updated


def _metric_row_batch_size() -> int:
    return _parse_positive_int_env("METRIC_ROW_BATCH_SIZE", 8)


def _direct_min_target_metric_records() -> int:
    return _parse_positive_int_env("DIRECT_MIN_TARGET_METRIC_RECORDS", 6)


def _direct_text_fallback_after_row_extraction_enabled() -> bool:
    return _env_flag(
        "DIRECT_ENABLE_TEXT_FALLBACK_AFTER_ROW_EXTRACTION",
        default=False,
    )


def _chunk_is_metric_row(chunk: dict) -> bool:
    return str(chunk.get("chunk_kind") or "").lower() == "metric_row"


def _metric_row_key(chunk: dict) -> tuple[str, ...]:
    from ..evidence_qualification import evidence_qualification_enabled
    key = (str(chunk.get("source_id") or "unknown"), str(chunk.get("table_id") or "table"))
    if evidence_qualification_enabled():
        # Table row numbers recur across revisions of the same source.
        return key + (str(chunk.get("document_id") or ""), str(chunk.get("document_hash") or ""))
    return key


def _metric_row_id(chunk: dict, fallback_index: int) -> str:
    return str(
        chunk.get("row_id")
        or chunk.get("chunk_id")
        or f"metric_row_{fallback_index}"
    )


def _numeric_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?", str(text or "")):
        compact = match.replace(",", "")
        tokens.add(compact)
        if compact.endswith("%"):
            tokens.add(compact[:-1])
    return {token for token in tokens if token}


def _make_metric_row_batch_chunk(
    rows: list[dict],
    *,
    batch_index: int,
) -> dict:
    first = rows[0]
    source_id = str(first.get("source_id") or "unknown")
    table_id = str(first.get("table_id") or "table")
    metric_rows: list[dict] = []
    row_quote_by_id: dict[str, str] = {}
    row_chunk_id_by_id: dict[str, str] = {}
    row_metadata_by_id: dict[str, dict] = {}
    text_lines: list[str] = []
    for index, row in enumerate(rows, start=1):
        row_id = _metric_row_id(row, index)
        if row_id in row_metadata_by_id:
            from ..evidence_qualification import evidence_qualification_enabled
            if evidence_qualification_enabled():
                row_id = f"{row_id}@{row.get('chunk_id') or index}"
                while row_id in row_metadata_by_id:
                    row_id += f"@{index}"
        row_quote = str(row.get("row_quote") or row.get("text") or "")
        metric_rows.append(
            {
                "row_id": row_id,
                "chunk_id": row.get("chunk_id"),
                "text": row.get("text"),
                "row_quote": row_quote,
                "table_id": row.get("table_id"),
                "source_id": row.get("source_id"),
                "bound_context_spans": list(row.get("bound_context_spans") or []),
                "document_id": row.get("document_id"),
                "document_hash": row.get("document_hash"),
                "char_start": row.get("char_start"),
                "char_end": row.get("char_end"),
                "json_pointer": row.get("json_pointer"),
                "structured_data_kind": row.get("structured_data_kind"),
                "source_column_label": row.get("source_column_label"),
                "metric_column_label": row.get("metric_column_label"),
                "source_column_labels": list(row.get("source_column_labels") or []),
                "table_header": row.get("table_header"),
                "heading_context": row.get("heading_context"),
                "row_context_type": row.get("row_context_type"),
                "reporting_period_start": row.get("reporting_period_start"),
                "reporting_period_end": row.get("reporting_period_end"),
                "reporting_period_label": row.get("reporting_period_label"),
            }
        )
        row_quote_by_id[row_id] = row_quote
        if row.get("chunk_id"):
            row_chunk_id_by_id[row_id] = str(row.get("chunk_id"))
        row_metadata_by_id[row_id] = {
            key: row.get(key)
            for key in (
                "chunk_id",
                "document_id",
                "document_hash",
                "char_start",
                "char_end",
                "bound_context_spans",
                "json_pointer",
                "structured_data_kind",
                "chunk_kind",
                "data_types",
                "context_types",
                "contains_target_data",
                "disease_relevance_status",
                "extraction_eligible_for_task_disease",
                "presence_reason",
                "table_id",
                "row_id",
                "row_quote",
                "source_column_labels",
                "source_column_label",
                "metric_column_label",
                "table_header",
                "heading_context",
                "row_context_type",
                "reporting_period_start",
                "reporting_period_end",
                "reporting_period_label",
                "period_basis",
            )
        }
        row_metadata_by_id[row_id]["row_numeric_tokens"] = sorted(
            _numeric_tokens(row_quote or row.get("text") or "")
        )
        text_lines.append(f"[{row_id}] {row.get('text') or row_quote}")
    batch = dict(first)
    from ..evidence_qualification import evidence_qualification_enabled
    version_suffix = ""
    if evidence_qualification_enabled():
        version = json.dumps(_metric_row_key(first), ensure_ascii=False)
        version_suffix = "_" + hashlib.sha256(version.encode()).hexdigest()[:12]
    batch.update(
        {
            "chunk_id": f"batch_{source_id}_{table_id}_{batch_index}{version_suffix}",
            "chunk_kind": "metric_row_batch",
            "text": "\n".join(text_lines),
            "metric_rows": metric_rows,
            "row_quote_by_id": row_quote_by_id,
            "row_chunk_id_by_id": row_chunk_id_by_id,
            "row_metadata_by_id": row_metadata_by_id,
            "metric_row_batch_size": len(rows),
            "context_types": sorted(
                set(
                    [
                        *(first.get("context_types") or []),
                        "metric_row_batch",
                    ]
                )
            ),
        }
    )
    return batch


def _clean_metric_cell_text(value: str | None) -> str:
    text = re.sub(r"[*_`]+", "", str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _pipe_table_cells(row_text: str | None) -> list[str]:
    text = str(row_text or "").strip()
    if "|" not in text:
        return []
    cells = [_clean_metric_cell_text(cell) for cell in text.strip("|").split("|")]
    return [cell for cell in cells if cell]


def _parse_metric_cell_value(cell: str) -> tuple[float | None, float | None]:
    text = _clean_metric_cell_text(cell)
    count_match = re.search(r"(-?\d[\d,]*(?:\.\d+)?)", text)
    percent_match = re.search(r"\((-?\d[\d,]*(?:\.\d+)?)\s*%\)", text)
    if not percent_match:
        percent_match = re.search(r"(-?\d[\d,]*(?:\.\d+)?)\s*%", text)

    def _to_float(match: re.Match[str] | None) -> float | None:
        if not match:
            return None
        try:
            return float(match.group(1).replace(",", ""))
        except (TypeError, ValueError):
            return None

    return _to_float(count_match), _to_float(percent_match)


def _metric_row_column_labels(row: dict, value_count: int) -> list[str]:
    raw = row.get("source_column_labels") or row.get("column_labels") or []
    labels: list[str] = []
    if isinstance(raw, list):
        labels = [str(item).strip() for item in raw if str(item or "").strip()]
    elif isinstance(raw, str) and raw.strip():
        labels = [part.strip() for part in re.split(r"\s*\|\s*|\s*,\s*", raw) if part.strip()]
    if len(labels) >= value_count:
        return labels[:value_count]
    if _can_infer_weekly_clinical_lab_comparison_columns(row, value_count):
        return ["Current Week", "Previous Week"]
    if value_count == 1:
        fallback = row.get("source_column_label") or row.get("metric_column_label")
        return [
            str(fallback).strip()
            if fallback and not _is_weak_column_label(str(fallback))
            else "Current Week"
        ]
    defaults = [f"Column {index + 1}" for index in range(value_count)]
    while len(defaults) < value_count:
        defaults.append(f"Column {len(defaults) + 1}")
    merged = labels + defaults[len(labels) :]
    return merged[:value_count]


def _can_infer_weekly_clinical_lab_comparison_columns(
    row: dict,
    value_count: int,
) -> bool:
    """Infer current/previous week only for narrow weekly lab comparison rows.

    Some public-health pages expose table rows without the original header, but
    keep a section heading such as "Results of tests from Clinical Laboratories".
    In that recurring surveillance layout, two numeric columns represent the
    current reporting week and the previous week. Keep this deliberately narrow
    so unrelated two-column tables remain quarantined as ambiguous.
    """

    if value_count != 2:
        return False
    if not (row.get("reporting_period_start") and row.get("reporting_period_end")):
        return False
    context_text = " ".join(
        str(value or "")
        for value in (
            row.get("heading_context"),
            row.get("table_header"),
            row.get("table_id"),
            row.get("title"),
            row.get("reporting_period_label"),
        )
    ).lower()
    if "clinical laborator" not in context_text:
        return False
    if any(
        marker in context_text
        for marker in (
            "season to date",
            "season-to-date",
            "cumulative",
            "public health laborator",
        )
    ):
        return False
    cells = _pipe_table_cells(str(row.get("row_quote") or row.get("text") or ""))
    if len(cells) != 3:
        return False
    row_label = cells[0].lower()
    return bool(
        any(term in row_label for term in ("specimen", "test", "positive", "percent"))
        and any(re.search(r"\d", cell) for cell in cells[1:])
    )


def _deterministic_metric_payloads_from_row(row: dict) -> list[dict]:
    row_text = str(row.get("row_quote") or row.get("text") or "")
    cells = _pipe_table_cells(row_text)
    if len(cells) < 2:
        return []
    label = cells[0]
    label_lower = label.lower()
    value_cells = cells[1:]
    column_labels = _metric_row_column_labels(row, len(value_cells))
    payloads: list[dict] = []

    positive_specimen_row = (
        "positive" in label_lower
        and ("specimen" in label_lower or "test" in label_lower)
    )
    tested_specimen_row = (
        ("specimen" in label_lower or "test" in label_lower)
        and ("tested" in label_lower or "performed" in label_lower)
        and "positive" not in label_lower
    )
    if not positive_specimen_row and not tested_specimen_row:
        return []
    parsed_cells = [
        (*_parse_metric_cell_value(cell), cell)
        for cell in value_cells
    ]
    has_percent = any(percent_value is not None for _, percent_value, _ in parsed_cells)
    has_count = any(count_value is not None for count_value, _, _ in parsed_cells)
    if not has_percent and not (
        tested_specimen_row
        and has_count
        and _can_infer_weekly_clinical_lab_comparison_columns(row, len(value_cells))
    ):
        return []

    for index, (count_value, percent_value, _cell) in enumerate(parsed_cells):
        column_label = column_labels[index] if index < len(column_labels) else None
        if tested_specimen_row and count_value is not None:
            payloads.append(
                {
                    "metric_name": "Number of specimens tested",
                    "metric_value": count_value,
                    "metric_unit": "count",
                    "metric_category": "lab_test_count",
                    "tests_total": count_value,
                    "source_column_label": column_label,
                    "source_row_id": _metric_row_id(row, 1),
                    "observation_type": "surveillance_summary",
                    "observation_types": ["surveillance_summary"],
                    "primary_case_dataset_eligible": False,
                    "count_semantics": "laboratory test count",
                    "statistical_count_type": "newly_reported",
                }
            )
        if positive_specimen_row and count_value is not None:
            payloads.append(
                {
                    "metric_name": "Number of positive specimens",
                    "metric_value": count_value,
                    "metric_unit": "count",
                    "metric_category": "lab_positive_count",
                    "tests_positive": count_value,
                    "source_column_label": column_label,
                    "source_row_id": _metric_row_id(row, 1),
                    "observation_type": "surveillance_summary",
                    "observation_types": ["surveillance_summary"],
                    "primary_case_dataset_eligible": False,
                    "count_semantics": "laboratory positive specimen count",
                    "statistical_count_type": "newly_reported",
                }
            )
        if positive_specimen_row and percent_value is not None:
            payloads.append(
                {
                    "metric_name": "Percent positive specimens",
                    "metric_value": percent_value,
                    "metric_unit": "percent",
                    "metric_category": "lab_positivity_percent",
                    "metric_denominator": "specimens_tested",
                    "positivity_rate": percent_value,
                    "source_column_label": column_label,
                    "source_row_id": _metric_row_id(row, 1),
                    "observation_type": "surveillance_summary",
                    "observation_types": ["surveillance_summary"],
                    "primary_case_dataset_eligible": False,
                    "count_semantics": "laboratory positivity percent",
                    "statistical_count_type": "newly_reported",
                }
            )
    return payloads


def _deterministic_metric_row_records(
    row: dict,
    *,
    start_index: int,
    llm_policy: LLMStructuredExtractionPolicy,
    settings: dict,
    context: dict | None,
) -> list[PublicHealthRecord]:
    payloads = _deterministic_metric_payloads_from_row(row)
    if not payloads:
        return []
    batch_chunk = _make_metric_row_batch_chunk([row], batch_index=0)
    records: list[PublicHealthRecord] = []
    for offset, payload in enumerate(payloads):
        llm_record = LLMExtractedRecord(**payload)
        record = _build_record_from_llm_output(
            llm_record,
            batch_chunk,
            start_index + offset,
            llm_policy,
            settings,
            context,
        )
        if record is None:
            continue
        warnings = list(record.semantic_warnings or [])
        if "deterministic_metric_row_splitter" not in warnings:
            warnings.append("deterministic_metric_row_splitter")
        record = record.model_copy(
            update={
                "llm_used": False,
                "llm_model": None,
                "llm_provider": None,
                "extraction_mode": "deterministic",
                "extraction_method": "deterministic_metric_row_splitter",
                "extraction_reason": (
                    "Deterministic metric-row split from explicit table row"
                ),
                "semantic_warnings": warnings,
                "extraction_warnings": warnings,
            }
        )
        records.append(record)
    return records


def _record_task_fit_assessments(
    records: list[dict],
    context: dict | None,
) -> list[dict]:
    contract = (context or {}).get("task_acceptance_contract") or {}
    task_location = str(contract.get("location") or (context or {}).get("task_location") or "")
    task_location_lower = task_location.lower()
    assessments: list[dict] = []
    for record in records:
        text = " ".join(
            str(record.get(key) or "")
            for key in (
                "subnational_location",
                "geographic_scope",
                "country",
                "evidence_quote",
                "source_title",
            )
        ).lower()
        geography_fit = (
            "matches_task_location"
            if not task_location_lower or task_location_lower in text
            else "needs_final_gate_review"
        )
        has_date = any(
            record.get(key)
            for key in (
                "date_reported",
                "date_anchor",
                "event_start_date",
                "reporting_period",
                "metric_period_start",
                "metric_period_end",
            )
        )
        has_count = any(
            record.get(key) not in (None, "")
            for key in (
                "cases_confirmed",
                "cases_probable",
                "cases_suspected",
                "cases_unspecified",
                "deaths",
                "hospitalizations",
                "tests_positive",
                "positivity_rate",
                "cumulative_count",
                "new_count",
                "metric_value",
            )
        )
        assessments.append(
            {
                "record_id": record.get("record_id"),
                "source_id": record.get("source_id"),
                "record_task_fit_agent": "record_task_fit_assessor",
                "contract_version": contract.get("contract_version"),
                "geography_fit": geography_fit,
                "date_fit": "has_date_or_period" if has_date else "missing_date_anchor",
                "count_semantics_fit": (
                    "has_interpretable_metric"
                    if has_count
                    else "missing_interpretable_metric"
                ),
                "record_acceptance_rules": list(
                    contract.get("record_acceptance_rules") or []
                ),
                "context_or_quarantine_rules": list(
                    contract.get("context_or_quarantine_rules") or []
                ),
            }
        )
    return assessments


def _detect_disease_alias_used(text: str, context: dict | None) -> str | None:
    from ..evidence_qualification import evidence_qualification_enabled
    universal = evidence_qualification_enabled()
    lowered = _lower(text)
    for term in (context or {}).get("disease_terms") or []:
        if not term:
            continue
        if (re.search(r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)", lowered)
                if universal else term.lower() in lowered):
            return term
    return None


def _detect_pathogen_or_syndrome(text: str, context: dict | None) -> str | None:
    from ..disease_relevance import _find_terms

    lowered = _lower(text)
    universal = get_env("PIPELINE_MODE") == "evidence"
    for term in (context or {}).get("pathogen_terms") or []:
        # A profile alias is a search vocabulary, not evidence inside a longer word.
        matched = _find_terms(text, [term]) if universal and term else term and term.lower() in lowered
        if matched:
            canonical = _canonical_disease_name(term)
            if canonical in {"COVID-19", "Dengue"} and term.lower() not in {
                "sars-cov-2",
                "denv",
                "dengue virus",
            }:
                return canonical
            return term
    return None


_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def _extract_year_or_date(
    text: str,
    policy: StructuredExtractionPolicy,
) -> str | None:
    if not text:
        return None
    iso = _ISO_DATE_RE.search(text)
    if iso:
        return iso.group(1)
    year_start = int(policy.date_patterns.get("year_range_start", 2020))
    year_end = int(policy.date_patterns.get("year_range_end", 2026))
    for year in _YEAR_RE.findall(text):
        if year_start <= int(year) <= year_end:
            return year
    return None


# Country / location patterns. Order matters — longer / more specific first.
_LOCATION_PATTERNS: list[tuple[re.Pattern, tuple[str | None, str | None]]] = [
    (re.compile(r"\bin\s+the\s+United\s+States\b", re.IGNORECASE), ("United States of America", None)),
    (re.compile(r"\bin\s+United\s+States\b", re.IGNORECASE), ("United States of America", None)),
    (re.compile(r"\bin\s+USA\b"), ("United States of America", None)),
    (re.compile(r"\bin\s+New\s+Mexico\b", re.IGNORECASE), ("United States of America", "New Mexico")),
    (re.compile(r"\bNew\s+York\s+City\b|\bNYC\b", re.IGNORECASE), ("United States of America", "New York City")),
    (re.compile(r"\bNew\s+York\b", re.IGNORECASE), ("United States of America", "New York")),
    (re.compile(r"\bFlorida\b", re.IGNORECASE), ("United States of America", "Florida")),
    (re.compile(r"\bin\s+China\b", re.IGNORECASE), ("China", None)),
    (re.compile(r"\bin\s+Chile\b", re.IGNORECASE), ("Chile", None)),
    (re.compile(r"\bin\s+Argentina\b", re.IGNORECASE), ("Argentina", None)),
    (re.compile(r"\bin\s+Europe\b", re.IGNORECASE), (None, "Europe")),
    (re.compile(r"\bin\s+Germany\b", re.IGNORECASE), ("Germany", None)),
    (re.compile(r"\bin\s+Sweden\b", re.IGNORECASE), ("Sweden", None)),
    (re.compile(r"\bin\s+Finland\b", re.IGNORECASE), ("Finland", None)),
    (re.compile(r"\bin\s+France\b", re.IGNORECASE), ("France", None)),
    (re.compile(r"\bin\s+Spain\b", re.IGNORECASE), ("Spain", None)),
]
# Generic "Country X" / "Country Y" style placeholders (uppercase token).
_COUNTRY_X_RE = re.compile(r"\bin\s+(Country\s+[A-Z][\w-]*)\b")


def _extract_country_or_location(text: str) -> tuple[str | None, str | None]:
    if not text:
        return None, None
    for pattern, result in _LOCATION_PATTERNS:
        if pattern.search(text):
            return result
    m = _COUNTRY_X_RE.search(text)
    if m:
        return m.group(1), None
    return None, None


# ---------------------------------------------------------------------------
# Case + death extraction
# ---------------------------------------------------------------------------


# "12 ... cases" — up to 5 short tokens between the number and "case(s)".
_CASE_NUMERIC_RE = re.compile(
    r"(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)\s+"
    r"(?P<between>(?:[\w-]+\s+){0,5})?"
    r"cases?\b",
    re.IGNORECASE,
)
# "(confirmed/probable/suspected/laboratory-confirmed) cases?: 12"  OR  "cases: 12"
_CASE_COLON_RE = re.compile(
    r"(?P<prefix>(?:laboratory-?\s*)?confirmed\s+cases?"
    r"|probable\s+cases?"
    r"|suspected\s+cases?"
    r"|cases?)\s*:\s*"
    r"(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)",
    re.IGNORECASE,
)


def _classify_case_bucket(prefix_or_between: str) -> str:
    if get_env("PIPELINE_MODE") == "evidence":
        from ..source_assertions import case_bucket
        return case_bucket(prefix_or_between)
    lowered = prefix_or_between.lower()
    if "confirmed" in lowered or "laboratory" in lowered:
        return "cases_confirmed"
    if "probable" in lowered:
        return "cases_probable"
    if "suspected" in lowered:
        return "cases_suspected"
    return "cases_unspecified"


_BUCKET_TO_LABEL = {
    "cases_confirmed": "confirmed",
    "cases_probable": "probable",
    "cases_suspected": "suspected",
    "cases_unspecified": "unspecified",
}


def _extract_case_counts(
    text: str,
    policy: StructuredExtractionPolicy,  # noqa: ARG001 — reserved for future LLM step
) -> dict:
    result = {
        "cases_confirmed": None,
        "cases_probable": None,
        "cases_suspected": None,
        "cases_unspecified": None,
        "case_definition": None,
    }
    if not text:
        return result

    ordered_labels: list[str] = []

    for m in _CASE_NUMERIC_RE.finditer(text):
        num = _normalize_number(m.group("num"))
        if num is None:
            continue
        between = m.group("between") or ""
        bucket = _classify_case_bucket(between)
        if result[bucket] is None:
            result[bucket] = num
            ordered_labels.append(_BUCKET_TO_LABEL[bucket])

    for m in _CASE_COLON_RE.finditer(text):
        num = _normalize_number(m.group("num"))
        if num is None:
            continue
        bucket = _classify_case_bucket(m.group("prefix"))
        if result[bucket] is None:
            result[bucket] = num
            ordered_labels.append(_BUCKET_TO_LABEL[bucket])

    if ordered_labels:
        seen: set[str] = set()
        unique = [lbl for lbl in ordered_labels if not (lbl in seen or seen.add(lbl))]
        result["case_definition"] = ",".join(unique)
    return result


def _extract_deaths(
    text: str,
    policy: StructuredExtractionPolicy,
) -> float | None:
    if not text:
        return None
    keywords = [k for k in policy.death_keywords if k and k != "died"]
    if not keywords:
        return None
    pattern = re.compile(
        rf"(?P<num>\d+(?:,\d{{3}})*(?:\.\d+)?)\s+(?:[\w-]+\s+){{0,3}}"
        rf"(?:{'|'.join(re.escape(k) for k in keywords)})\b",
        re.IGNORECASE,
    )
    m = pattern.search(text)
    if m:
        return _normalize_number(m.group("num"))

    colon_pattern = re.compile(
        rf"(?:{'|'.join(re.escape(k) for k in keywords)})\s*:\s*"
        rf"(?P<num>\d+(?:,\d{{3}})*(?:\.\d+)?)",
        re.IGNORECASE,
    )
    m = colon_pattern.search(text)
    if m:
        return _normalize_number(m.group("num"))
    return None


_HOSPITALIZATION_NUMERIC_RE = re.compile(
    r"(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)\s+"
    r"(?:[\w-]+\s+){0,3}hospitali[sz]ations?\b",
    re.IGNORECASE,
)
_HOSPITALIZATION_COLON_RE = re.compile(
    r"hospitali[sz]ations?\s*:\s*"
    r"(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)",
    re.IGNORECASE,
)
_SOURCE_HOSPITALIZATION_RE = re.compile(
    r"\b(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)\s+(?:reported\s+)?hospitali[sz]ations?\b",re.I
)
_CORE_CASE_BURDEN_RE = re.compile(
    r"(?:\b(?:reported|notified|recorded|confirmed)\s+)?"
    r"(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)\s+"
    r"(?:[\w-]+\s+){0,6}?"
    r"(?P<label>(?:tb|tuberculosis|measles|influenza|flu|covid-19|covid)\s+)?"
    r"cases?\b",
    re.IGNORECASE,
)
_CORE_INCIDENCE_RATE_RE = re.compile(
    r"\bincidence\s+rate(?:\s+(?:was|of|is))?\s+"
    r"(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)\s+"
    r"(?:cases?\s+)?per\s+100,?000(?:\s+(?:persons?|population))?",
    re.IGNORECASE,
)
_CORE_MORTALITY_RATE_RE = re.compile(
    r"\bmortality\s+rate(?:\s+(?:was|of|is))?\s+"
    r"(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)\s+"
    r"(?:deaths?\s+)?per\s+100,?000(?:\s+(?:persons?|population))?",
    re.IGNORECASE,
)
_CORE_METRIC_CONTEXT_TYPES = {
    "narrative_metric",
    "markdown_metric_line",
    "annual_metric",
    "public_health_metric",
}
_CORE_METRIC_DATA_TYPES = {
    "case_count",
    "incidence_rate",
    "mortality_rate",
    "death_count",
    "burden_metric",
}
_EDGE_DEMOGRAPHIC_QUALIFIERS = {
    "unknown",
    "missing",
    "birth origin",
    "race",
    "ethnicity",
    "age",
    "sex",
    "gender",
}


def _extract_hospitalizations(text: str) -> float | None:
    if not text:
        return None
    for pattern in (_HOSPITALIZATION_NUMERIC_RE, _HOSPITALIZATION_COLON_RE):
        m = pattern.search(text)
        if m:
            return _normalize_number(m.group("num"))
    return None


def _chunk_looks_like_core_metric_text(chunk: dict) -> bool:
    if get_env("PIPELINE_MODE") == "evidence":
        text = str(chunk.get("text") or "")
        if any(pattern.search(text) for pattern in (_CORE_INCIDENCE_RATE_RE,_CORE_MORTALITY_RATE_RE,_SOURCE_HOSPITALIZATION_RE)):
            return True
    context_types = {
        str(value).lower() for value in (chunk.get("context_types") or []) if value
    }
    data_types = {
        str(value).lower() for value in (chunk.get("data_types") or []) if value
    }
    if context_types.intersection(_CORE_METRIC_CONTEXT_TYPES):
        return True
    if data_types.intersection(_CORE_METRIC_DATA_TYPES):
        return True
    return str(chunk.get("chunk_kind") or "").lower() in {
        "narrative_metric",
        "metric_text",
    }


def _core_metric_period_fields(
    chunk: dict,
    context: dict | None,
) -> tuple[str | None, str | None, str | None]:
    if get_env("PIPELINE_MODE") == "evidence":
        return None,None,_official_literal_reporting_period(str(chunk.get("text") or ""))
    start, end, label = _source_period_from_chunk(chunk, context)
    if start and end:
        return start, end, label
    structured = (context or {}).get("structured_task") or {}
    collection = (context or {}).get("collection_spec") or {}
    start = structured.get("start_date") or collection.get("start_date")
    end = structured.get("end_date") or collection.get("end_date") or start
    return (
        str(start) if start else None,
        str(end) if end else None,
        str(label) if label else None,
    )


def _core_metric_location_fields(
    chunk: dict,
    context: dict | None,
) -> tuple[str | None, str | None, str | None]:
    if get_env("PIPELINE_MODE") == "evidence":
        local = _official_geography(str(chunk.get("text") or ""),"",context)
        if not local.get("associated_countries") and not local.get("geographic_scope"):
            local = _official_geography("\n".join(s["quote"] for s in _verified_rule_headings(chunk,context)),"",context)
        return local.get("country"),local.get("geographic_scope"),local.get("geographic_scope_type")
    task_location = _task_location_from_context(context)
    country = _first_text(chunk.get("country"))
    geographic_scope = _first_text(
        chunk.get("geographic_scope"),
        chunk.get("subnational_location"),
        country,
        task_location,
    )
    geographic_scope_type = _first_text(chunk.get("geographic_scope_type"))
    if _is_united_states_location(geographic_scope):
        country = country or "United States"
        geographic_scope = "United States"
        geographic_scope_type = geographic_scope_type or "country"
    elif geographic_scope and not geographic_scope_type:
        geographic_scope_type = "country" if geographic_scope == country else "subnational"
    return country, geographic_scope, geographic_scope_type


def _match_has_edge_demographic_context(match: re.Match[str], text: str) -> bool:
    start = max(0, match.start() - 80)
    end = min(len(text), match.end() + 80)
    window = text[start:end].lower()
    return any(marker in window for marker in _EDGE_DEMOGRAPHIC_QUALIFIERS)


def _core_metric_payloads_from_chunk(
    chunk: dict,
    context: dict | None,
) -> list[dict]:
    text = str(chunk.get("row_quote") or chunk.get("text") or "")
    if not text.strip() or not _chunk_looks_like_core_metric_text(chunk):
        return []
    payloads: list[dict] = []
    period_start, period_end, period_label = _core_metric_period_fields(chunk, context)
    period_source = (
        "filled_from_source_reporting_period"
        if period_start and period_end
        else "unresolved"
    )

    case_match = None if get_env("PIPELINE_MODE") == "evidence" else _CORE_CASE_BURDEN_RE.search(text)
    if case_match and not _match_has_edge_demographic_context(case_match, text):
        value = _normalize_number(case_match.group("num"))
        if value is not None:
            payloads.append(
                {
                    "metric_name": "reported case count",
                    "metric_value": value,
                    "metric_unit": "count",
                    "metric_category": "case_count",
                    "count_semantics": "annual public health case aggregate",
                    "statistical_count_type": "annual",
                }
            )

    if get_env("PIPELINE_MODE") == "evidence":
        hospitalization = _SOURCE_HOSPITALIZATION_RE.search(text)
        if hospitalization:
            payloads.append({"metric_name":"hospitalizations","metric_value":_normalize_number(hospitalization.group("num")),
                             "metric_unit":"count","metric_category":"hospitalization_count"})

    incidence_match = _CORE_INCIDENCE_RATE_RE.search(text)
    if incidence_match:
        value = _normalize_number(incidence_match.group("num"))
        if value is not None:
            payloads.append(
                {
                    "metric_name": "incidence rate",
                    "metric_value": value,
                    "metric_unit": "per 100,000 population",
                    "metric_category": "incidence_rate",
                    "incidence_rate": value,
                    "metric_denominator": "100,000 population",
                    "count_semantics": "annual public health incidence rate",
                    "statistical_count_type": "annual",
                }
            )

    mortality_match = _CORE_MORTALITY_RATE_RE.search(text)
    if mortality_match:
        value = _normalize_number(mortality_match.group("num"))
        if value is not None:
            payloads.append(
                {
                    "metric_name": "mortality rate",
                    "metric_value": value,
                    "metric_unit": "per 100,000 population",
                    "metric_category": "mortality_rate",
                    "metric_denominator": "100,000 population",
                    "count_semantics": "annual public health mortality rate",
                    "statistical_count_type": "annual",
                }
            )

    if not payloads:
        return []
    for payload in payloads:
        if get_env("PIPELINE_MODE") == "evidence":
            payload.pop("count_semantics",None)
            payload.pop("statistical_count_type",None)
        if period_start:
            payload["metric_period_start"] = period_start
        if period_end:
            payload["metric_period_end"] = period_end
            payload["date_reported"] = period_end
        if period_label:
            payload["metric_period_label"] = period_label
            payload["reporting_period"] = period_label
        payload["metric_period_source"] = period_source
    return payloads


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------


def _classify_table_column(header_lower: str) -> str | None:
    if not header_lower:
        return None
    if "confirmed case" in header_lower or header_lower == "confirmed":
        return "cases_confirmed"
    if "probable case" in header_lower or header_lower == "probable":
        return "cases_probable"
    if "suspected case" in header_lower or header_lower == "suspected":
        return "cases_suspected"
    if "case" in header_lower or header_lower == "cases":
        return "cases_unspecified"
    if "death" in header_lower or "fatality" in header_lower:
        return "deaths"
    if "hospitalization" in header_lower or "hospitalisation" in header_lower:
        return "hospitalizations"
    if "country" in header_lower:
        return "country"
    if (
        "location" in header_lower
        or "region" in header_lower
        or "state" in header_lower
        or "province" in header_lower
        or "district" in header_lower
    ):
        return "subnational_location"
    if "year" in header_lower or "date" in header_lower:
        return "date_reported"
    return None


def _extract_from_table_text(
    text: str,
    policy: StructuredExtractionPolicy,  # noqa: ARG001 — reserved
) -> dict | None:
    if not text or "|" not in text:
        return None
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return None

    header_cells = [c.strip() for c in lines[0].split("|")]
    column_map: dict[str, int] = {}
    for i, header in enumerate(header_cells):
        field = _classify_table_column(header.lower())
        if field and field not in column_map:
            column_map[field] = i
    if not column_map:
        return None

    data_cells = [c.strip() for c in lines[1].split("|")]
    result: dict = {}
    for field, idx in column_map.items():
        if idx >= len(data_cells):
            continue
        value = data_cells[idx]
        if not value:
            continue
        if field in (
            "cases_confirmed",
            "cases_probable",
            "cases_suspected",
            "cases_unspecified",
            "deaths",
            "hospitalizations",
        ):
            num = _normalize_number(value)
            if num is not None:
                result[field] = num
        elif field == "date_reported":
            if re.fullmatch(r"20\d{2}", value):
                result[field] = value
            else:
                result[field] = value
        else:
            result[field] = value
    return result or None


# ---------------------------------------------------------------------------
# Chunk → record
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 16: post-extraction semantic guardrails (shared by LLM + rule paths)
# ---------------------------------------------------------------------------


_GENERIC_VIRUS_OR_SYNDROME_VALUES = {
    "hantavirus",
    "hantavirus disease",
    "hantavirus infection",
    "hantaviridae",
    "orthohantavirus",
    "orthohantavirus infection",
}

_CANONICAL_VIRUS_OR_SYNDROME = {
    "hps": "HPS",
    "hfrs": "HFRS",
    "andes virus": "Andes virus",
    "seoul virus": "Seoul virus",
    "sin nombre virus": "Sin Nombre virus",
    "hantaan virus": "Hantaan virus",
    "puumala virus": "Puumala virus",
    "dobrava-belgrade virus": "Dobrava-Belgrade virus",
    "dobrava virus": "Dobrava-Belgrade virus",
}

_REGION_TERMS = {
    "eu/eea": "EU/EEA",
    "eu/eea (28 countries)": "EU/EEA",
    "european union": "EU/EEA",
    "european union/european economic area": "EU/EEA",
    "eea": "EU/EEA",
    "europe": "Europe",
    "americas": "Americas",
    "north america": "North America",
    "south america": "South America",
}

_ALLOWED_STATISTICAL_COUNT_TYPES = {
    "cumulative",
    "annual",
    "newly_reported",
    "historical_total",
    "subset",
    "unknown",
}

_ALLOWED_GEOGRAPHIC_SCOPE_TYPES = {
    "country",
    "subnational",
    "region",
    "multi_country",
    "global",
    "unknown",
}

# Step 16.1.1: LLMs sometimes emit hyphenated / free-text variants for the
# semantic enum fields. Canonicalize them to the internal vocabulary used by
# downstream nodes so e.g. "multi-country" still triggers the regional-scope
# exception in normalization / linking. Mapping is exhaustive enough for the
# real-world variants we have observed; unmapped inputs are surfaced as a
# warning rather than silently dropped.
_GEOGRAPHIC_SCOPE_TYPE_ALIASES: dict[str, str] = {
    "country": "country",
    "national": "country",
    "nation": "country",
    "single country": "country",
    "single_country": "country",
    "subnational": "subnational",
    "sub-national": "subnational",
    "sub national": "subnational",
    "sub_national": "subnational",
    "state": "subnational",
    "province": "subnational",
    "region": "region",
    "regional": "region",
    "multi_country": "multi_country",
    "multi-country": "multi_country",
    "multi country": "multi_country",
    "multicountry": "multi_country",
    "multinational": "multi_country",
    "multi-national": "multi_country",
    "multi national": "multi_country",
    "global": "global",
    "worldwide": "global",
    "world wide": "global",
    "world-wide": "global",
    "unknown": "unknown",
}

_STATISTICAL_COUNT_TYPE_ALIASES: dict[str, str] = {
    "cumulative": "cumulative",
    "total": "cumulative",
    "cumulative total": "cumulative",
    "cumulative_total": "cumulative",
    "annual": "annual",
    "yearly": "annual",
    "newly_reported": "newly_reported",
    "newly reported": "newly_reported",
    "newly-reported": "newly_reported",
    "new cases": "newly_reported",
    "additional cases": "newly_reported",
    "historical_total": "historical_total",
    "historical total": "historical_total",
    "historic total": "historical_total",
    "reported through": "historical_total",
    "subset": "subset",
    "subgroup": "subset",
    "of which": "subset",
    "unknown": "unknown",
}


def _normalize_enum_lookup_key(value: str) -> str:
    """Lowercase + collapse internal whitespace; keep '-' and '_' intact for
    direct alias lookup. Used to find an LLM-emitted value in the alias maps.
    """

    return re.sub(r"\s+", " ", value.strip().lower())


def _canonicalize_geographic_scope_type(
    value: str | None,
) -> tuple[str | None, list[str]]:
    if value is None:
        return None, []
    if not isinstance(value, str):
        return value, []
    stripped = value.strip()
    if not stripped:
        return None, []
    key = _normalize_enum_lookup_key(stripped)
    canonical = _GEOGRAPHIC_SCOPE_TYPE_ALIASES.get(key)
    if canonical is None:
        return stripped, ["unrecognized_geographic_scope_type"]
    if canonical != stripped:
        return canonical, ["canonicalized_geographic_scope_type"]
    return canonical, []


def _canonicalize_statistical_count_type(
    value: str | None,
) -> tuple[str | None, list[str]]:
    if value is None:
        return None, []
    if not isinstance(value, str):
        return value, []
    stripped = value.strip()
    if not stripped:
        return None, []
    key = _normalize_enum_lookup_key(stripped)
    canonical = _STATISTICAL_COUNT_TYPE_ALIASES.get(key)
    if canonical is None:
        return stripped, ["unrecognized_statistical_count_type"]
    if canonical != stripped:
        return canonical, ["canonicalized_statistical_count_type"]
    return canonical, []


def _standardize_disease_for_context(
    value: str | None,
    context: dict | None = None,
) -> tuple[str, list[str]]:
    """Standardize disease to the active task disease.

    The legacy no-context behavior remains Hantavirus disease. Stage 8 passes
    context from structured task / disease intelligence so non-hantavirus
    records keep their disease label.
    """

    from ..evidence_qualification import evidence_qualification_enabled
    if evidence_qualification_enabled():
        # A task label may filter a source fact, never replace or create it.
        return value.strip() if isinstance(value, str) else "", []

    canonical = (
        _canonical_disease_name((context or {}).get("disease_standard_name"))
        if context
        else None
    ) or "Hantavirus disease"
    warnings: list[str] = []
    if value is None or not isinstance(value, str):
        return canonical, warnings
    input_canonical = _canonical_disease_name(value)
    if input_canonical and input_canonical != canonical:
        warnings.append("standardized_disease_name")
    elif value.strip() != canonical:
        warnings.append("standardized_disease_name")
    return canonical, warnings


def _clean_virus_or_syndrome(
    value: str | None,
) -> tuple[str | None, list[str]]:
    if value is None:
        return None, []
    if not isinstance(value, str):
        return None, []
    stripped = value.strip()
    if not stripped:
        return None, []
    lowered = stripped.lower()
    if lowered in _GENERIC_VIRUS_OR_SYNDROME_VALUES:
        return None, ["removed_generic_virus_or_syndrome"]
    if lowered in _CANONICAL_VIRUS_OR_SYNDROME:
        canonical = _CANONICAL_VIRUS_OR_SYNDROME[lowered]
        return canonical, []
    return stripped, ["unrecognized_virus_or_syndrome_semantics"]


def _standardize_statistical_count_type(
    value: str | None,
    text: str | None,
    aliases: dict[str, list[str]] | None = None,
) -> tuple[str | None, list[str]]:
    """Normalize or infer statistical_count_type from value or chunk text."""

    if isinstance(value, str) and value.strip():
        canonical, w = _canonicalize_statistical_count_type(value)
        if canonical in _ALLOWED_STATISTICAL_COUNT_TYPES:
            return canonical, w
    if text:
        lowered = text.lower()
        # Order matters: more specific first.
        if any(m in lowered for m in ("newly reported", "new cases", "additional cases")):
            return "newly_reported", ["inferred_statistical_count_type_from_text"]
        if any(m in lowered for m in ("of which", "among", "subset")):
            return "subset", ["inferred_statistical_count_type_from_text"]
        if any(
            m in lowered
            for m in (
                "reported through",
                "as of december",
                "as of january",
                "as of february",
                "as of march",
                "as of april",
                "as of may",
                "as of june",
                "as of july",
                "as of august",
                "as of september",
                "as of october",
                "as of november",
                "prior to",
            )
        ):
            return "historical_total", ["inferred_statistical_count_type_from_text"]
        if any(
            m in lowered
            for m in (
                "annual",
                "during 2020",
                "during 2021",
                "during 2022",
                "during 2023",
                "during 2024",
                "during 2025",
                "during 2026",
                "in 2020",
                "in 2021",
                "in 2022",
                "in 2023",
                "in 2024",
                "in 2025",
                "in 2026",
            )
        ):
            return "annual", ["inferred_statistical_count_type_from_text"]
        if any(m in lowered for m in ("cumulative", "total", "through ", "since ")):
            return "cumulative", ["inferred_statistical_count_type_from_text"]
    return None, []


def _standardize_geographic_scope(
    record: dict,
) -> tuple[dict, list[str]]:
    out = dict(record)
    warnings: list[str] = []

    # Step 16.1.1: canonicalize any LLM-emitted scope_type variant (e.g.
    # "multi-country", "national") before downstream nodes inspect it.
    raw_scope_type = out.get("geographic_scope_type")
    canonical_scope_type, scope_type_warnings = _canonicalize_geographic_scope_type(
        raw_scope_type
    )
    if canonical_scope_type != raw_scope_type:
        out["geographic_scope_type"] = canonical_scope_type
    warnings.extend(scope_type_warnings)

    country = out.get("country")
    scope = out.get("geographic_scope")
    scope_type = out.get("geographic_scope_type")
    subnational = out.get("subnational_location")

    if isinstance(country, str) and country.strip():
        lowered = country.strip().lower()
        if lowered in _REGION_TERMS:
            canonical_region = _REGION_TERMS[lowered]
            if not scope:
                out["geographic_scope"] = canonical_region
            if not scope_type:
                # Single-region term like Europe maps to region; aggregates
                # like EU/EEA are multi_country at semantic level.
                out["geographic_scope_type"] = (
                    "multi_country" if canonical_region == "EU/EEA" else "region"
                )
            out["country"] = None
            warnings.append("regional_geographic_scope_not_country")

    # Fill scope from country when country is a single nation.
    if (
        out.get("country")
        and not out.get("geographic_scope")
        and isinstance(out.get("country"), str)
    ):
        out["geographic_scope"] = out["country"]
        if not out.get("geographic_scope_type"):
            out["geographic_scope_type"] = "country"

    if subnational and out.get("country") and not out.get("aggregation_level"):
        out["aggregation_level"] = "subnational"

    if out.get("geographic_scope") and not out.get("geographic_scope_type"):
        out["geographic_scope_type"] = "unknown"

    return out, warnings


def _apply_extraction_semantic_guardrails(
    record_dict: dict,
    chunk: dict | None,
    context: dict | None = None,
) -> dict:
    out = dict(record_dict)
    existing_warnings = list(out.get("semantic_warnings") or [])

    disease, w_d = _standardize_disease_for_context(out.get("disease"), context)
    out["disease"] = disease
    out["disease_standard_name"] = disease
    existing_warnings.extend(w_d)

    if context and not context.get("is_hantavirus"):
        pathogen = out.get("pathogen_or_syndrome")
        if not pathogen and chunk and get_env("PIPELINE_MODE") != "evidence":
            pathogen = _detect_pathogen_or_syndrome(chunk.get("text") or "", context)
        out["pathogen_or_syndrome"] = pathogen
        if not out.get("virus_or_syndrome"):
            out["virus_or_syndrome"] = pathogen
    else:
        vos, w_v = _clean_virus_or_syndrome(out.get("virus_or_syndrome"))
        out["virus_or_syndrome"] = vos
        if not out.get("pathogen_or_syndrome"):
            out["pathogen_or_syndrome"] = vos
        existing_warnings.extend(w_v)

    chunk_text = chunk.get("text") if isinstance(chunk, dict) else None
    # A nearby statement may describe a different indicator or period. In evidence
    # only normalize an extracted label; the shared evidence assessor decides
    # whether that label is supported. Do not manufacture one from a whole chunk.
    semantics_text = None if get_env("PIPELINE_MODE") == "evidence" else chunk_text
    sct, w_s = _standardize_statistical_count_type(
        out.get("statistical_count_type"), semantics_text
    )
    out["statistical_count_type"] = sct
    existing_warnings.extend(w_s)

    out, w_g = _standardize_geographic_scope(out)
    existing_warnings.extend(w_g)

    outbreak_text = " ".join(
        str(value or "")
        for value in (
            out.get("count_semantics"),
            out.get("statistical_count_type"),
            out.get("case_definition"),
            out.get("source_section"),
            chunk_text,
        )
    ).lower()
    explicit_outbreak_count = any(
        token in outbreak_text
        for token in (
            "outbreak_count",
            "outbreak count",
            "number of outbreaks",
            "reported outbreaks",
            "outbreaks reported",
            "outbreaks were reported",
        )
    ) or bool(
        re.search(r"\b\d[\d,]*(?:\s+\w+){0,4}\s+outbreaks?\b", outbreak_text)
        or re.search(r"\boutbreaks?\s+(?:reported|total(?:ed)?|count(?:ed)?)\s+\d[\d,]*\b", outbreak_text)
    )
    explicit_case_count = bool(
        re.search(r"\b\d[\d,]*(?:\s+\w+){0,4}\s+(?:cases?|patients?|infections?)\b", outbreak_text)
        or re.search(r"\b(?:cases?|patients?|infections?)\s+(?:reported|total(?:ed)?|count(?:ed)?)\s+\d[\d,]*\b", outbreak_text)
    )
    if (
        out.get("cases_unspecified") is not None
        and explicit_outbreak_count
        and not explicit_case_count
    ):
        outbreak_count = out.get("cases_unspecified")
        if out.get("cumulative_count") is None and out.get("new_count") is None:
            out["cumulative_count"] = outbreak_count
        out["cases_unspecified"] = None
        out["observation_type"] = "outbreak_summary"
        observation_types = list(out.get("observation_types") or [])
        if "outbreak_summary" not in observation_types:
            observation_types.append("outbreak_summary")
        out["observation_types"] = observation_types
        out["primary_case_dataset_eligible"] = False
        if not out.get("count_semantics"):
            out["count_semantics"] = "outbreak_count"
        existing_warnings.append("outbreak_count_not_case_count")

    metric_text = " ".join(
        str(value or "")
        for value in (
            out.get("metric_name"),
            out.get("metric_category"),
            out.get("metric_unit"),
            out.get("metric_denominator"),
            out.get("count_semantics"),
            out.get("statistical_count_type"),
            out.get("source_section"),
            chunk_text,
        )
    ).lower()
    ed_visit_metric = any(
        token in metric_text
        for token in (
            "emergency department",
            "ed visit",
            "ed visits",
            "nssp",
            "emergency room",
            "er visit",
        )
    )
    if ed_visit_metric and out.get("positivity_rate") is not None:
        if out.get("metric_value") is None:
            out["metric_value"] = out.get("positivity_rate")
        if not out.get("metric_name"):
            out["metric_name"] = "nssp_ed_visit_percent"
        if not out.get("metric_category"):
            out["metric_category"] = "ed_visit_percent"
        if not out.get("metric_unit"):
            out["metric_unit"] = "percent"
        if not out.get("metric_denominator"):
            out["metric_denominator"] = "emergency_department_visits"
        out["positivity_rate"] = None
        existing_warnings.append("ed_visit_percent_moved_from_positivity_rate")

    if out.get("metric_value") is None:
        metric_candidates = [
            ("tests_positive", "lab_positive_count", "positive_tests", "count"),
            ("tests_total", "lab_test_count", "tests_total", "count"),
            ("hospitalizations", "hospitalization_count", "hospitalizations", "count"),
            ("icu_admissions", "icu_admission_count", "icu_admissions", "count"),
            ("deaths", "death_count", "deaths", "count"),
            ("cumulative_count", "aggregate_count", "cumulative_count", "count"),
            ("new_count", "new_count", "new_count", "count"),
            ("incidence_rate", "incidence_rate", "incidence_rate", "rate"),
            ("positivity_rate", "lab_positivity_percent", "positivity_rate", "percent"),
        ]
        for field, category, name, unit in metric_candidates:
            if out.get(field) is not None:
                out["metric_value"] = out.get(field)
                out["metric_category"] = out.get("metric_category") or category
                out["metric_name"] = out.get("metric_name") or name
                out["metric_unit"] = out.get("metric_unit") or unit
                if field == "tests_positive" and out.get("tests_total") is not None:
                    out["metric_denominator"] = (
                        out.get("metric_denominator") or "tests_total"
                    )
                break

    if out.get("metric_value") is not None:
        metric_category = str(out.get("metric_category") or "").lower()
        metric_name = str(out.get("metric_name") or "").lower()
        if not out.get("metric_name"):
            out["metric_name"] = out.get("metric_category") or "public_health_metric"
        if not out.get("metric_category"):
            out["metric_category"] = out.get("metric_name")
        if not out.get("metric_unit"):
            if "percent" in metric_category or "percent" in metric_name:
                out["metric_unit"] = "percent"
            elif "rate" in metric_category or "rate" in metric_name:
                out["metric_unit"] = "rate"
            else:
                out["metric_unit"] = "count"
        from ..evidence_qualification import evidence_qualification_enabled
        if not out.get("count_semantics") and not evidence_qualification_enabled():
            out["count_semantics"] = out.get("metric_category") or out.get("metric_name")

    out["semantic_warnings"] = existing_warnings
    return out


def _chunk_is_extractable(
    chunk: dict,
    policy: StructuredExtractionPolicy,
    context: dict | None = None,
) -> bool:
    conditions = policy.extractable_chunk_conditions or {}
    disease_blocked, _ = _chunk_blocked_by_disease_gate(chunk, context)
    if disease_blocked:
        return False
    if conditions.get("requires_contains_target_data", True):
        if not chunk.get("contains_target_data"):
            if _lower(chunk.get("chunk_kind")) != "table":
                return False
            table_data = _extract_from_table_text(
                str(chunk.get("text") or ""),
                policy,
            )
            if not table_data or not any(
                table_data.get(field) is not None
                for field in (
                    "cases_confirmed",
                    "cases_probable",
                    "cases_suspected",
                    "cases_unspecified",
                    "deaths",
                    "hospitalizations",
                )
            ):
                return False
    allowed_purposes = conditions.get("allowed_fetch_purposes") or []
    if allowed_purposes and chunk.get("fetch_purpose") not in allowed_purposes:
        if not (
            not _direct_collection_enabled(context)
            and chunk.get("fetch_purpose") == "context_grounding"
            and chunk.get("contains_target_data")
        ):
            return False
    allowed_kinds = conditions.get("allowed_chunk_kinds") or []
    if allowed_kinds and (chunk.get("chunk_kind") or "text") not in allowed_kinds:
        return False
    text = chunk.get("text") or ""
    if not text.strip():
        return False
    return True


def _context_only_source_ids(role_policy: dict | None) -> set[str]:
    if not role_policy:
        return set()
    return {str(source_id) for source_id in role_policy.get("context_only_source_ids") or []}


def _chunk_routing_flags(chunk: dict) -> list[str]:
    flags = list(chunk.get("routing_flags") or [])
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    for flag in metadata.get("routing_flags") or []:
        if flag not in flags:
            flags.append(flag)
    return flags


def _is_context_only_chunk(chunk: dict, role_policy: dict | None = None) -> bool:
    if chunk.get("source_id") in _context_only_source_ids(role_policy):
        return True
    flags = set(_chunk_routing_flags(chunk))
    if "context_only" in flags or "blocked_from_structured_extraction" in flags:
        return True
    if chunk.get("fetch_purpose") == "context_grounding":
        return True
    return chunk.get("source_role") in {"context_source", "context_only"}


def _chunk_has_explicit_context_only_role(
    chunk: dict,
    role_policy: dict | None = None,
) -> bool:
    if chunk.get("source_id") in _context_only_source_ids(role_policy):
        return True
    flags = set(_chunk_routing_flags(chunk))
    if "context_only" in flags or "blocked_from_structured_extraction" in flags:
        return True
    return chunk.get("source_role") in {"context_source", "context_only"}


def _build_record_from_chunk(
    chunk: dict,
    index: int,
    policy: StructuredExtractionPolicy,
    context: dict | None = None,
) -> PublicHealthRecord | None:
    if not _chunk_is_extractable(chunk, policy, context):
        return None

    if get_env("PIPELINE_MODE") == "evidence":
        record,_ = _official_outbreak_record_from_chunk(chunk,index,policy,context)
        return record

    text = chunk.get("text") or ""
    chunk_kind = chunk.get("chunk_kind") or "text"

    universal = get_env("PIPELINE_MODE") == "evidence"
    virus_or_syndrome = None if universal else _detect_virus_or_syndrome(text, policy)
    pathogen_or_syndrome = None if universal else _detect_pathogen_or_syndrome(text, context)
    disease_alias_used = _detect_disease_alias_used(text, context)
    date_reported = _extract_year_or_date(text, policy)
    country, subnational_location = _extract_country_or_location(text)
    case_counts = _extract_case_counts(text, policy)
    deaths = _extract_deaths(text, policy)
    hospitalizations = _extract_hospitalizations(text)

    if chunk_kind == "table":
        table_data = _extract_from_table_text(text, policy)
        if table_data:
            if table_data.get("date_reported") is not None:
                date_reported = table_data["date_reported"]
            if table_data.get("country") is not None:
                country = table_data["country"]
            if table_data.get("subnational_location") is not None:
                subnational_location = table_data["subnational_location"]
            for f in ("cases_confirmed", "cases_probable", "cases_suspected", "cases_unspecified"):
                if table_data.get(f) is not None:
                    case_counts[f] = table_data[f]
            if table_data.get("deaths") is not None:
                deaths = table_data["deaths"]
            if table_data.get("hospitalizations") is not None:
                hospitalizations = table_data["hospitalizations"]
            # Refresh case_definition if table populated any case bucket.
            if any(case_counts[b] is not None for b in (
                "cases_confirmed", "cases_probable", "cases_suspected", "cases_unspecified"
            )) and not case_counts.get("case_definition"):
                labels = [
                    _BUCKET_TO_LABEL[b] for b in (
                        "cases_confirmed", "cases_probable", "cases_suspected", "cases_unspecified"
                    ) if case_counts[b] is not None
                ]
                case_counts["case_definition"] = ",".join(labels) or None

    has_any_signal = (
        case_counts["cases_confirmed"] is not None
        or case_counts["cases_probable"] is not None
        or case_counts["cases_suspected"] is not None
        or case_counts["cases_unspecified"] is not None
        or deaths is not None
        or hospitalizations is not None
        or date_reported is not None
        or country is not None
        or subnational_location is not None
    )
    if not has_any_signal:
        return None

    source_id = chunk.get("source_id") or ""
    record_id = f"rec_{source_id}_{index:03d}"

    chunk_confidence = chunk.get("confidence")
    extraction_confidence = (
        float(chunk_confidence) if chunk_confidence is not None else 0.50
    )

    # Apply Step 16 semantic guardrails to the rule-based pre-record so that
    # statistical_count_type / geographic_scope / virus_or_syndrome are kept
    # consistent with the LLM path.
    pre_record = {
        "disease": (context or {}).get("disease_standard_name") or policy.default_disease,
        "disease_standard_name": (context or {}).get("disease_standard_name") or policy.default_disease,
        "disease_alias_used": disease_alias_used,
        "virus_or_syndrome": virus_or_syndrome,
        "pathogen_or_syndrome": pathogen_or_syndrome,
        "country": country,
        "subnational_location": subnational_location,
        "date_reported": date_reported,
        "cases_confirmed": case_counts["cases_confirmed"],
        "cases_probable": case_counts["cases_probable"],
        "cases_suspected": case_counts["cases_suspected"],
        "cases_unspecified": case_counts["cases_unspecified"],
        "deaths": deaths,
        "hospitalizations": hospitalizations,
        "case_definition": case_counts["case_definition"],
    }
    cleaned = _apply_extraction_semantic_guardrails(pre_record, chunk, context)

    required_core = policy.required_core_fields_for_valid_record
    field_values = {
        "disease": cleaned.get("disease") or policy.default_disease,
        "source_url": chunk.get("source_url"),
        "source_type": chunk.get("source_type"),
        "evidence_quote": text,
    }
    missing_fields = [
        f for f in required_core
        if not field_values.get(f) or (isinstance(field_values.get(f), str) and not str(field_values[f]).strip())
    ]

    semantic_warnings = list(cleaned.get("semantic_warnings") or [])
    if _has_any_case_or_death({**cleaned, "hospitalizations": hospitalizations}):
        if not cleaned.get("date_reported"):
            semantic_warnings.append("missing_date_for_count_bearing_record")
        if not (cleaned.get("country") or cleaned.get("subnational_location") or cleaned.get("geographic_scope")):
            semantic_warnings.append("missing_location_for_count_bearing_record")
    compatibility = assess_record_disease_compatibility(
        {
            **cleaned,
            "disease": cleaned.get("disease") or policy.default_disease,
            "evidence_quote": text,
            "table_header": chunk.get("table_header"),
            "heading_context": chunk.get("heading_context"),
            "source_title": chunk.get("title"),
            "source_url": chunk.get("source_url"),
        },
        _as_disease_relevance_context(context),
    )

    return PublicHealthRecord(
        record_id=record_id,
        disease=cleaned.get("disease") or policy.default_disease,
        disease_standard_name=cleaned.get("disease_standard_name")
        or cleaned.get("disease")
        or policy.default_disease,
        disease_alias_used=cleaned.get("disease_alias_used"),
        virus_or_syndrome=cleaned.get("virus_or_syndrome"),
        pathogen_or_syndrome=cleaned.get("pathogen_or_syndrome"),
        target_population=(context or {}).get("target_population"),
        observation_type=cleaned.get("observation_type"),
        observation_types=list(cleaned.get("observation_types") or []),
        primary_case_dataset_eligible=cleaned.get(
            "primary_case_dataset_eligible"
        ),
        country=cleaned.get("country"),
        subnational_location=cleaned.get("subnational_location"),
        date_reported=cleaned.get("date_reported"),
        event_start_date=None,
        event_end_date=None,
        cases_confirmed=cleaned.get("cases_confirmed"),
        cases_probable=cleaned.get("cases_probable"),
        cases_suspected=cleaned.get("cases_suspected"),
        cases_unspecified=cleaned.get("cases_unspecified"),
        deaths=cleaned.get("deaths"),
        hospitalizations=cleaned.get("hospitalizations"),
        icu_admissions=cleaned.get("icu_admissions"),
        tests_positive=cleaned.get("tests_positive"),
        tests_total=cleaned.get("tests_total"),
        positivity_rate=cleaned.get("positivity_rate"),
        incidence_rate=cleaned.get("incidence_rate"),
        cumulative_count=cleaned.get("cumulative_count"),
        new_count=cleaned.get("new_count"),
        case_definition=cleaned.get("case_definition"),
        source_id=source_id,
        source_url=chunk.get("source_url"),
        source_type=chunk.get("source_type"),
        evidence_quote=text,
        extraction_confidence=extraction_confidence,
        missing_fields=missing_fields,
        schema_status=None,
        provenance_status=None,
        supporting_chunk_id=chunk.get("chunk_id"),
        table_header=chunk.get("table_header"),
        heading_context=chunk.get("heading_context"),
        row_context_type=chunk.get("row_context_type"),
        source_title=chunk.get("title"),
        publisher=chunk.get("publisher"),
        source_role_final=chunk.get("source_role_final"),
        credibility_score=chunk.get("credibility_score"),
        credibility_level=chunk.get("credibility_level"),
        actual_publisher=chunk.get("actual_publisher"),
        actual_publisher_normalized=chunk.get("actual_publisher_normalized"),
        source_type_final=chunk.get("source_type_final"),
        source_independence_group=chunk.get("source_independence_group"),
        claim_support_role=chunk.get("claim_support_role"),
        recommended_source_role=chunk.get("recommended_source_role"),
        recommended_fetch_use=chunk.get("recommended_fetch_use"),
        recommended_extraction_use=chunk.get("recommended_extraction_use"),
        likely_syndicated_or_aggregated=chunk.get(
            "likely_syndicated_or_aggregated"
        ),
        upstream_source_mentions=list(chunk.get("upstream_source_mentions") or []),
        discovery_method=chunk.get("discovery_method"),
        search_provider=chunk.get("search_provider"),
        query_id=chunk.get("query_id"),
        query_used=chunk.get("query_used"),
        document_id=chunk.get("document_id"),
        document_type=chunk.get("document_type"),
        fetch_purpose=chunk.get("fetch_purpose"),
        chunk_kind=chunk_kind,
        data_types=list(chunk.get("data_types") or []),
        context_types=list(chunk.get("context_types") or []),
        extraction_method=policy.extraction_method,
        extraction_reason="deterministic extraction from evidence chunk",
        validation_errors=[],
        repair_actions=[],
        requires_human_review=False,
        statistical_count_type=cleaned.get("statistical_count_type"),
        reporting_period=cleaned.get("reporting_period"),
        as_of_date=cleaned.get("as_of_date"),
        count_semantics=cleaned.get("count_semantics") or cleaned.get("statistical_count_type") or "unspecified",
        aggregation_level=cleaned.get("aggregation_level"),
        geographic_scope=cleaned.get("geographic_scope"),
        geographic_scope_type=cleaned.get("geographic_scope_type"),
        population_scope=cleaned.get("population_scope"),
        source_section=cleaned.get("source_section"),
        semantic_warnings=semantic_warnings,
        extraction_warnings=semantic_warnings,
        **record_compatibility_fields(compatibility),
        record_schema="generic_public_health_record",
        legacy_record_type=(
            "ObservationRecord" if ((context or {}).get("is_hantavirus") is True) else None
        ),
    )


def _build_core_metric_records_from_chunk(
    chunk: dict,
    start_index: int,
    policy: StructuredExtractionPolicy,
    context: dict | None = None,
) -> list[PublicHealthRecord]:
    payloads = _core_metric_payloads_from_chunk(chunk, context)
    if not payloads:
        return []

    text = str(chunk.get("row_quote") or chunk.get("text") or "")
    source_id = chunk.get("source_id") or ""
    disease = (context or {}).get("disease_standard_name") or policy.default_disease
    universal = get_env("PIPELINE_MODE") == "evidence"
    virus_or_syndrome = None if universal else _detect_virus_or_syndrome(text, policy)
    pathogen_or_syndrome = None if universal else _detect_pathogen_or_syndrome(text, context)
    disease_alias_used = _detect_disease_alias_used(text, context)
    country, geographic_scope, geographic_scope_type = _core_metric_location_fields(
        chunk,
        context,
    )
    if not country and geographic_scope_type == "country":
        country = geographic_scope
    compatibility = assess_record_disease_compatibility(
        {
            "disease": disease,
            "evidence_quote": text,
            "source_title": chunk.get("title"),
            "source_url": chunk.get("source_url"),
        },
        _as_disease_relevance_context(context),
    )
    records: list[PublicHealthRecord] = []
    for offset, payload in enumerate(payloads):
        record_id = f"rec_{source_id}_{start_index + offset:03d}"
        warnings = list(chunk.get("semantic_warnings") or [])
        warnings.append("deterministic_core_metric_text_fallback")
        records.append(
            PublicHealthRecord(
                record_id=record_id,
                disease=disease,
                disease_standard_name=disease,
                disease_alias_used=disease_alias_used,
                virus_or_syndrome=virus_or_syndrome,
                pathogen_or_syndrome=pathogen_or_syndrome,
                target_population=(context or {}).get("target_population"),
                observation_type="surveillance_summary",
                observation_types=["surveillance_summary"],
                primary_case_dataset_eligible=False,
                country=country,
                subnational_location=chunk.get("subnational_location"),
                date_reported=payload.get("date_reported"),
                event_start_date=None,
                event_end_date=None,
                cases_confirmed=None,
                cases_probable=None,
                cases_suspected=None,
                cases_unspecified=None,
                deaths=None,
                hospitalizations=None,
                icu_admissions=None,
                tests_positive=None,
                tests_total=None,
                positivity_rate=None,
                incidence_rate=payload.get("incidence_rate"),
                cumulative_count=None,
                new_count=None,
                metric_name=payload.get("metric_name"),
                metric_value=payload.get("metric_value"),
                metric_unit=payload.get("metric_unit"),
                metric_category=payload.get("metric_category"),
                metric_denominator=payload.get("metric_denominator"),
                metric_period_start=payload.get("metric_period_start"),
                metric_period_end=payload.get("metric_period_end"),
                metric_period_source=payload.get("metric_period_source"),
                metric_period_label=payload.get("metric_period_label"),
                case_definition=None,
                source_id=source_id,
                source_url=chunk.get("source_url"),
                source_type=chunk.get("source_type"),
                evidence_quote=text,
                extraction_confidence=float(chunk.get("confidence") or 0.55),
                missing_fields=[],
                schema_status=None,
                provenance_status=None,
                supporting_chunk_id=chunk.get("chunk_id"),
                source_title=chunk.get("title"),
                publisher=chunk.get("publisher"),
                source_role_final=chunk.get("source_role_final"),
                credibility_score=chunk.get("credibility_score"),
                credibility_level=chunk.get("credibility_level"),
                actual_publisher=chunk.get("actual_publisher"),
                actual_publisher_normalized=chunk.get(
                    "actual_publisher_normalized"
                ),
                source_type_final=chunk.get("source_type_final"),
                source_independence_group=chunk.get("source_independence_group"),
                claim_support_role=chunk.get("claim_support_role"),
                recommended_source_role=chunk.get("recommended_source_role"),
                recommended_fetch_use=chunk.get("recommended_fetch_use"),
                recommended_extraction_use=chunk.get(
                    "recommended_extraction_use"
                ),
                likely_syndicated_or_aggregated=chunk.get(
                    "likely_syndicated_or_aggregated"
                ),
                upstream_source_mentions=list(
                    chunk.get("upstream_source_mentions") or []
                ),
                discovery_method=chunk.get("discovery_method"),
                search_provider=chunk.get("search_provider"),
                query_id=chunk.get("query_id"),
                query_used=chunk.get("query_used"),
                document_id=chunk.get("document_id"),
                document_type=chunk.get("document_type"),
                fetch_purpose=chunk.get("fetch_purpose"),
                chunk_kind=chunk.get("chunk_kind") or "text",
                data_types=list(chunk.get("data_types") or []),
                context_types=list(chunk.get("context_types") or []),
                extraction_method="deterministic_core_metric_text_fallback",
                extraction_reason="deterministic extraction of core public-health metric from narrative text",
                validation_errors=[],
                repair_actions=[],
                requires_human_review=False,
                statistical_count_type=payload.get("statistical_count_type"),
                reporting_period=payload.get("reporting_period"),
                as_of_date=None,
                count_semantics=payload.get("count_semantics"),
                aggregation_level="national",
                geographic_scope=geographic_scope,
                geographic_scope_type=geographic_scope_type,
                population_scope=chunk.get("population_scope"),
                source_section=chunk.get("source_section"),
                semantic_warnings=warnings,
                extraction_warnings=warnings,
                **record_compatibility_fields(compatibility),
                record_schema="generic_public_health_record",
                legacy_record_type=(
                    "ObservationRecord"
                    if ((context or {}).get("is_hantavirus") is True)
                    else None
                ),
            )
        )
    if universal:
        from ..record_identity import stable_record_id
        for record in records:
            record.record_id = stable_record_id(record, chunk)
    return records


def _filter_redundant_core_metric_records(
    primary_record: PublicHealthRecord | None,
    core_records: list[PublicHealthRecord],
    context: dict | None = None,
) -> list[PublicHealthRecord]:
    if _direct_collection_enabled(context):
        return core_records
    if primary_record is None or not core_records:
        return core_records
    primary_values = {
        float(value)
        for value in (
            primary_record.cases_confirmed,
            primary_record.cases_probable,
            primary_record.cases_suspected,
            primary_record.cases_unspecified,
        )
        if value is not None
    }
    if not primary_values:
        return core_records
    filtered: list[PublicHealthRecord] = []
    for record in core_records:
        category = str(record.metric_category or record.metric_name or "").lower()
        value = record.metric_value
        if "case" in category and value is not None:
            try:
                if float(value) in primary_values:
                    continue
            except (TypeError, ValueError):
                pass
        filtered.append(record)
    return filtered


# ---------------------------------------------------------------------------
# Schema validation + repair
# ---------------------------------------------------------------------------


_CONTENT_FIELDS = (
    "cases_confirmed",
    "cases_probable",
    "cases_suspected",
    "cases_unspecified",
    "deaths",
    "hospitalizations",
    "date_reported",
    "country",
    "subnational_location",
    "geographic_scope",
)


def _has_any_case_or_death(record: dict) -> bool:
    for f in (
        "cases_confirmed",
        "cases_probable",
        "cases_suspected",
        "cases_unspecified",
        "deaths",
        "hospitalizations",
    ):
        if record.get(f) is not None:
            return True
    return False


def _has_minimum_content(record: dict) -> bool:
    for f in _CONTENT_FIELDS:
        value = record.get(f)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return True
    return False


def _repair_record(
    record: dict,
    policy: StructuredExtractionPolicy,
) -> tuple[dict, list[str]]:
    repaired = dict(record)
    actions: list[str] = []

    disease = repaired.get("disease")
    if policy.default_disease and (not disease or (isinstance(disease, str) and not disease.strip())):
        repaired["disease"] = policy.default_disease
        actions.append("set_disease_to_default")

    if repaired.get("extraction_confidence") is None:
        repaired["extraction_confidence"] = 0.50
        actions.append("set_default_extraction_confidence")

    return repaired, actions


_ISO_DATE_FULL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PERIOD_WEEK_RE = re.compile(r"\b(?:mmwr\s*)?week\s+(\d{1,2})\b", re.IGNORECASE)


def _metric_period_start_after_end(record: dict) -> bool:
    start = str(record.get("metric_period_start") or "").strip()
    end = str(record.get("metric_period_end") or "").strip()
    if not (_ISO_DATE_FULL_RE.match(start) and _ISO_DATE_FULL_RE.match(end)):
        return False
    return start > end


def _has_strong_metric_period_disambiguator(record: dict) -> bool:
    for key in ("source_column_label", "metric_column_label"):
        value = str(record.get(key) or "").strip()
        if value and not _is_weak_column_label(value):
            return True
    return str(record.get("metric_period_source") or "").strip() in {
        "llm_extracted",
        "filled_from_row_column_label",
        "filled_from_column_label",
    }


def _ambiguous_multi_period_metric_quote(record: dict) -> bool:
    if record.get("metric_value") in (None, "") and not record.get("metric_name"):
        return False
    quote = str(record.get("evidence_quote") or "").strip()
    if not quote:
        return False
    mentioned_weeks = {
        int(match.group(1))
        for match in _PERIOD_WEEK_RE.finditer(quote)
        if match.group(1).isdigit()
    }
    if len(mentioned_weeks) < 2:
        return False
    return not _has_strong_metric_period_disambiguator(record)


def _validate_record(
    record: dict,
    policy: StructuredExtractionPolicy,
) -> tuple[dict, SchemaValidationResult]:
    repaired, repair_actions = _repair_record(record, policy)

    validation_errors: list[str] = []
    try:
        PublicHealthRecord(**repaired)
        pydantic_ok = True
    except Exception as exc:
        pydantic_ok = False
        validation_errors.append(f"pydantic_validation_failed: {exc}")

    def _missing(field: str) -> bool:
        value = repaired.get(field)
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        return False

    missing_core = [f for f in policy.required_core_fields_for_valid_record if _missing(f)]
    missing_provenance = [f for f in policy.required_provenance_fields if _missing(f)]
    provenance_status = "verified" if not missing_provenance else "incomplete"
    has_min_content = _has_minimum_content(repaired)
    review_trigger_missing = [
        f for f in policy.fields_that_trigger_human_review_if_missing if _missing(f)
    ]
    has_count_signal = _has_any_case_or_death(repaired)
    has_location_signal = any(
        not _missing(f)
        for f in ("country", "subnational_location", "locality", "geographic_scope")
    )
    has_date_signal = any(
        not _missing(f)
        for f in (
            "date_reported",
            "event_start_date",
            "event_end_date",
            "reporting_period",
            "as_of_date",
        )
    )
    if has_count_signal and not has_location_signal:
        review_trigger_missing.append("location")
    if has_count_signal and not has_date_signal:
        review_trigger_missing.append("date")

    missing_fields = sorted({*missing_core, *review_trigger_missing})
    metric_period_invalid = _metric_period_start_after_end(repaired)
    ambiguous_metric_period = _ambiguous_multi_period_metric_quote(repaired)

    if (
        not pydantic_ok
        or missing_core
        or not has_min_content
        or metric_period_invalid
        or ambiguous_metric_period
    ):
        schema_status = "rejected"
        if missing_core:
            validation_errors.append(f"missing_required_core_fields: {missing_core}")
        if not has_min_content:
            validation_errors.append("no_minimum_content")
        if metric_period_invalid:
            validation_errors.append("metric_period_invalid_start_after_end")
        if ambiguous_metric_period:
            validation_errors.append("ambiguous_multi_period_metric_quote")
    elif review_trigger_missing or provenance_status == "incomplete":
        schema_status = "needs_review"
        if review_trigger_missing:
            validation_errors.append(
                f"missing_review_trigger_fields: {review_trigger_missing}"
            )
        if provenance_status == "incomplete":
            validation_errors.append(
                f"incomplete_provenance: {missing_provenance}"
            )
    else:
        schema_status = "valid"

    requires_human_review = schema_status == "needs_review"

    repaired["schema_status"] = schema_status
    repaired["provenance_status"] = provenance_status
    repaired["missing_fields"] = missing_fields
    repaired["validation_errors"] = validation_errors
    repaired["repair_actions"] = repair_actions
    repaired["requires_human_review"] = requires_human_review

    result = SchemaValidationResult(
        record_id=repaired.get("record_id", ""),
        schema_status=schema_status,
        provenance_status=provenance_status,
        validation_errors=validation_errors,
        missing_fields=missing_fields,
        repair_actions=repair_actions,
        requires_human_review=requires_human_review,
    )
    return repaired, result


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _load_llm_policy() -> LLMStructuredExtractionPolicy:
    return LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy())


def _chunk_allowed_for_llm(
    chunk: dict, llm_policy: LLMStructuredExtractionPolicy
) -> bool:
    if not chunk.get("contains_target_data"):
        return False
    if chunk.get("extraction_eligible_for_task_disease") is False:
        return False
    disease_status = chunk.get("disease_relevance_status")
    if disease_status and disease_status != TARGET_DISEASE_MATCH:
        return False
    if chunk.get("fetch_purpose") not in llm_policy.allowed_fetch_purposes:
        return False
    if (chunk.get("chunk_kind") or "text") not in llm_policy.allowed_chunk_kinds:
        return False
    text = chunk.get("text") or ""
    if not text.strip():
        return False
    return True


def _as_disease_relevance_context(context: dict | None) -> dict:
    if not context:
        return build_disease_relevance_context(None)
    if "target_disease_terms" in context or "guard_enabled" in context:
        return context
    return build_disease_relevance_context(context)


def _assess_chunk_for_extraction(chunk: dict, context: dict | None) -> dict:
    if chunk.get("disease_relevance_status"):
        cached_assessment = {
            "status": chunk.get("disease_relevance_status"),
            "score": chunk.get("disease_relevance_score"),
            "target_disease_terms_found": list(
                chunk.get("target_disease_terms_found") or []
            ),
            "incompatible_disease_terms_found": list(
                chunk.get("incompatible_disease_terms_found") or []
            ),
            "reason": chunk.get("disease_relevance_reason"),
            "data_signal_count": chunk.get("disease_relevance_data_signal_count", 0),
        }
        chunk_kind = _lower(chunk.get("chunk_kind"))
        chunk_type = _lower(chunk.get("chunk_type"))
        table_context = chunk_kind == "table" or chunk_type in {
            "table_row",
            "metric_row",
        }
        if table_context:
            row_text = str(chunk.get("row_quote") or chunk.get("text") or "")
            row_assessment = assess_chunk_disease_relevance(
                {"text": row_text, "title": ""},
                _as_disease_relevance_context(context),
            )
            row_incompatible_terms = list(
                row_assessment.get("incompatible_disease_terms_found") or []
            )
            row_target_terms = list(
                row_assessment.get("target_disease_terms_found") or []
            )
            if row_incompatible_terms and not row_target_terms:
                row_assessment["status"] = INCOMPATIBLE_DISEASE
                row_assessment["reason"] = (
                    "local_evidence_disease_mismatch: the table row explicitly "
                    "names an incompatible disease and does not name the target disease."
                )
                return row_assessment
        # A table row can inherit an explicit disease identity from its table or
        # document title. Content processing may have assessed the row text alone,
        # where column values legitimately omit the disease name. Never use this
        # inheritance when the row already names an incompatible disease.
        if (
            cached_assessment.get("status") != TARGET_DISEASE_MATCH
            and table_context
            and not cached_assessment.get("incompatible_disease_terms_found")
        ):
            refreshed = assess_chunk_disease_relevance(
                chunk,
                _as_disease_relevance_context(context),
            )
            if refreshed.get("status") == TARGET_DISEASE_MATCH:
                refreshed["reason"] = "target disease inherited from verified table context"
                return refreshed
        return cached_assessment
    return assess_chunk_disease_relevance(chunk, _as_disease_relevance_context(context))


def _chunk_blocked_by_disease_gate(
    chunk: dict,
    context: dict | None,
) -> tuple[bool, dict]:
    assessment = _assess_chunk_for_extraction(chunk, context)
    if (
        chunk.get("extraction_eligible_for_task_disease") is False
        and assessment.get("status") != TARGET_DISEASE_MATCH
    ):
        return True, assessment
    return assessment.get("status") != TARGET_DISEASE_MATCH, assessment


def _source_context_for_chunk(chunk: dict, context: dict | None) -> dict:
    source_id = str(chunk.get("source_id") or "")
    if isinstance(context, dict):
        source = (context.get("source_registry_by_id") or {}).get(source_id)
        if isinstance(source, dict):
            return source
    return {}


def _chunk_is_must_fetch(chunk: dict, context: dict | None) -> bool:
    source_id = str(chunk.get("source_id") or "")
    must_fetch_ids = set()
    if isinstance(context, dict):
        must_fetch_ids = {str(v) for v in (context.get("must_fetch_source_ids") or [])}
    source = _source_context_for_chunk(chunk, context)
    return (
        bool(source_id and source_id in must_fetch_ids)
        or chunk.get("must_fetch") is True
        or source.get("must_fetch") is True
    )


_EXPLICIT_LOW_TRUST_SOURCE_MARKERS = {
    "discussion_board",
    "forum",
    "personal_blog",
    "social_media",
    "social_network",
    "user_generated",
}


def _chunk_has_explicit_low_trust_source_type(chunk: dict) -> bool:
    """Return true only for an explicit, upstream low-trust source identity."""

    direct_type = _lower(chunk.get("source_type"))
    normalized = re.sub(r"[\s/-]+", "_", direct_type)
    return any(marker in normalized for marker in _EXPLICIT_LOW_TRUST_SOURCE_MARKERS)


def _source_context_verifies_high_trust(source: dict) -> bool:
    if not source:
        return False
    values = [
        source.get("source_type"),
        source.get("source_type_final"),
        source.get("publisher"),
        source.get("actual_publisher"),
        source.get("canonical_url"),
        source.get("url"),
        source.get("credibility_level"),
    ]
    text = _lower(" ".join(str(value or "") for value in values))
    return (
        _lower(source.get("credibility_level")) == "high"
        or "official_public_health_agency" in text
        or "public_health_agency" in text
        or "department of health" in text
        or "state_or_local_public_health_agency" in text
        or ".gov" in text
    )


def _chunk_is_official_or_high_trust(chunk: dict, context: dict | None) -> bool:
    source = _source_context_for_chunk(chunk, context)
    if (
        _chunk_has_explicit_low_trust_source_type(chunk)
        and not _source_context_verifies_high_trust(source)
    ):
        return False
    values = [
        chunk.get("source_type"),
        chunk.get("source_type_final"),
        chunk.get("publisher"),
        chunk.get("actual_publisher"),
        chunk.get("source_url"),
        chunk.get("credibility_level"),
        source.get("source_type"),
        source.get("source_type_final"),
        source.get("publisher"),
        source.get("actual_publisher"),
        source.get("canonical_url"),
        source.get("url"),
        source.get("credibility_level"),
    ]
    text = _lower(" ".join(str(value or "") for value in values))
    return (
        "high" in {_lower(chunk.get("credibility_level")), _lower(source.get("credibility_level"))}
        or "official_public_health_agency" in text
        or "public_health_agency" in text
        or "department of health" in text
        or "state_or_local_public_health_agency" in text
        or ".gov" in text
    )


_TASK_COLLECTION_CANDIDATE_LABELS = {
    "verified_target_collection",
    "search_verified_target_collection",
    "fetch_verified_target_collection",
    "task_record_collection_candidate",
}
_TASK_COLLECTION_REJECT_LABELS = {
    "context",
    "context_only",
    "validation",
    "validation_only",
    "best_available_context_candidate",
    "wrong_period_context",
    "excluded",
    "non_target_or_context",
}


def _chunk_is_task_collection_candidate(chunk: dict, context: dict | None) -> bool:
    source = _source_context_for_chunk(chunk, context)
    source_has_verified_collection_identity = bool(
        source.get("usable_for_task_collection") is True
        or {
            _lower(source.get("target_fit_status")),
            _lower(source.get("triage_role")),
            _lower(source.get("source_role_final")),
            _lower(source.get("source_role")),
        }
        & _TASK_COLLECTION_CANDIDATE_LABELS
    )
    if (
        _chunk_has_explicit_low_trust_source_type(chunk)
        and not source_has_verified_collection_identity
    ):
        return False
    labels = {
        _lower(chunk.get("target_fit_status")),
        _lower(chunk.get("triage_role")),
        _lower(chunk.get("source_role_final")),
        _lower(chunk.get("source_role")),
        _lower(source.get("target_fit_status")),
        _lower(source.get("triage_role")),
        _lower(source.get("source_role_final")),
        _lower(source.get("source_role")),
    }
    if labels & _TASK_COLLECTION_REJECT_LABELS:
        return False
    if chunk.get("usable_for_task_collection") is True or source.get("usable_for_task_collection") is True:
        return True
    if labels & _TASK_COLLECTION_CANDIDATE_LABELS:
        return True
    return (
        "collection" in labels
        and _lower(source.get("geography_fit") or chunk.get("geography_fit"))
        in {"match", "candidate", "possible", ""}
        and _lower(source.get("date_fit") or chunk.get("date_fit"))
        in {"match", "candidate", "possible", ""}
    )


def _domain_from_text_url(value: str) -> str:
    domain = urlsplit(str(value or "")).netloc.lower()
    return domain[4:] if domain.startswith("www.") else domain


def _domain_matches(domain: str, official_domains: list[str]) -> bool:
    if not domain:
        return False
    for official in official_domains:
        official = str(official or "").strip().lower()
        if official and (domain == official or domain.endswith("." + official)):
            return True
    return False


def _chunk_text_for_target_matching(chunk: dict, context: dict | None) -> str:
    source = _source_context_for_chunk(chunk, context)
    values = [
        chunk.get("source_url"),
        chunk.get("canonical_url"),
        chunk.get("title"),
        chunk.get("publisher"),
        chunk.get("text"),
        source.get("canonical_url"),
        source.get("url"),
        source.get("title"),
        source.get("publisher"),
        source.get("actual_publisher"),
    ]
    return _lower(" ".join(str(value or "") for value in values))


def _chunk_domain(chunk: dict, context: dict | None) -> str:
    source = _source_context_for_chunk(chunk, context)
    for value in (
        chunk.get("source_url"),
        chunk.get("canonical_url"),
        source.get("canonical_url"),
        source.get("url"),
    ):
        domain = _domain_from_text_url(str(value or ""))
        if domain:
            return domain
    return ""


def _chunk_official_report_key(chunk: dict, context: dict | None) -> str | None:
    source_id = str(chunk.get("source_id") or "")
    key_by_source = (context or {}).get("official_report_key_by_source_id") or {}
    if source_id and isinstance(key_by_source, dict):
        key = key_by_source.get(source_id)
        if key:
            return str(key)
    source = _source_context_for_chunk(chunk, context)
    for value in (
        chunk.get("source_url"),
        chunk.get("canonical_url"),
        source.get("canonical_url"),
        source.get("url"),
    ):
        key = official_report_key_for_url(str(value or ""))
        if key:
            return key
    return None


def _chunk_matches_target_official_requirement(
    chunk: dict,
    context: dict | None,
) -> bool:
    if not _direct_collection_enabled(context):
        return False
    requirements = build_source_coverage_requirements(context or {})
    if not requirements:
        return False
    domain = _chunk_domain(chunk, context)
    text = _chunk_text_for_target_matching(chunk, context)
    for requirement in requirements:
        if not _domain_matches(domain, list(requirement.get("official_domains") or [])):
            continue
        week = str(requirement.get("week") or "")
        year = str(requirement.get("year") or "")
        week_hints = {
            f"week-{week}",
            f"week_{week}",
            f"week {week}",
        }
        try:
            week_int = int(week)
            week_hints.update(
                {
                    f"week-{week_int:02d}",
                    f"week_{week_int:02d}",
                    f"week {week_int:02d}",
                }
            )
        except (TypeError, ValueError):
            pass
        date_match = any(
            str(value or "").strip().lower() in text
            for value in (requirement.get("date_hints") or [])
            if str(value or "").strip()
        )
        title_match = any(
            str(value or "").strip().lower() in text
            for value in (requirement.get("title_hints") or [])
            if str(value or "").strip()
        )
        week_match = any(hint and hint in text for hint in week_hints)
        if (date_match or week_match or title_match) and (not year or year in text or title_match):
            return True
    return False


def _chunk_priority(chunk: dict, context: dict | None) -> int:
    if (
        _chunk_matches_target_official_requirement(chunk, context)
        or _chunk_is_task_collection_candidate(chunk, context)
    ):
        return 1
    if _chunk_is_official_or_high_trust(chunk, context):
        return 2
    role = _lower(chunk.get("source_role_final") or chunk.get("source_role"))
    if role in {"collection", "collection_support"}:
        return 3
    return 4


def _chunk_is_llm_ready(chunk: dict, context: dict | None) -> bool:
    source = _source_context_for_chunk(chunk, context)
    readiness = str(
        chunk.get("extraction_readiness")
        or (chunk.get("metadata") or {}).get("extraction_readiness")
        or source.get("extraction_readiness")
        or ""
    ).strip().lower()
    if readiness not in {"context_only", "not_ready"}:
        return True
    return bool(chunk.get("case_span_quote")) and bool(
        chunk.get("contains_target_data")
    )



def extraction_skip_reason(chunk, context=None):
    """Common source-span eligibility for extraction and recovery planning."""
    if not str(chunk.get("text") or chunk.get("row_quote") or "").strip():
        return "empty_span"
    if not _chunk_is_llm_ready(chunk, context):
        return "extraction_not_ready"
    from ..evidence_qualification import evidence_qualification_enabled
    if evidence_qualification_enabled():
        from ..evidence_chunking import structural_heading_skip_reason, local_content_skip_reason
        local_reason = local_content_skip_reason(chunk)
        if local_reason:
            return local_reason
        documents = (context or {}).get("documents_by_source_id", {}).get(chunk.get("source_id"), [])
        heading_reason = structural_heading_skip_reason(chunk, documents)
        if heading_reason:
            return heading_reason
    if chunk.get("contains_target_data") is False:
        return "no_target_data"
    if chunk.get("extraction_eligible_for_task_disease") is False:
        return "disease_not_eligible"
    if chunk.get("disease_relevance_status") not in {None, "", "target_disease_match", "unknown"}:
        return "disease_mismatch"
    role_policy = load_source_role_policy()
    if _is_context_only_chunk(chunk, role_policy) and (
            _direct_collection_enabled(context) or _chunk_has_explicit_context_only_role(chunk, role_policy)):
        return "context_only"
    return None


def _chunk_information_priority(chunk: dict) -> int:
    """Rank extractable content within a source before using document order."""

    configured_priority = chunk.get("extraction_priority")
    if configured_priority is not None:
        try:
            return max(0, int(configured_priority))
        except (TypeError, ValueError):
            pass
    text = re.sub(r"\s+", " ", str(chunk.get("text") or "")).strip()
    lowered = text.lower()
    if _OFFICIAL_CASE_LABEL_RE.search(text):
        return 0
    if str(chunk.get("chunk_kind") or "").lower() in {
        "metric_row",
        "metric_row_batch",
        "table_row",
    } or chunk.get("row_id"):
        return 1
    if re.search(
        r"\b(?:patient|adult|male|female|crew|passenger|hospitali[sz]ed|"
        r"intensive care|symptom onset|developed symptoms|died on|recovered)\b",
        lowered,
    ):
        return 2
    if re.search(
        r"\b\d[\d,]*(?:\.\d+)?\s+(?:confirmed\s+|probable\s+|suspected\s+)?"
        r"(?:cases?|deaths?|patients?|hospitali[sz]ations?)\b",
        lowered,
    ):
        return 3
    navigation_terms = {
        "home",
        "about",
        "contact",
        "privacy",
        "menu",
        "publications",
    }
    if text and len(navigation_terms & set(re.findall(r"[a-z]+", lowered))) >= 3:
        return 9
    return 4


_TARGET_SOURCE_UNUSABLE_COVERAGE_STATUSES = {
    "partial_target_coverage",
    "target_official_source_missing",
    "target_source_missing_or_unverified",
    "target_official_source_fetch_failed",
    "target_official_source_unusable",
    "target_official_all_aliases_unusable",
    "target_alias_error_page",
    "target_source_unusable_error_page",
    "fallback_target_fetch_failed",
    "no_task_collection_document",
}


def _direct_should_skip_non_target_extraction(
    chunks: list[dict],
    context: dict | None,
) -> bool:
    if not _direct_collection_enabled(context):
        return False
    coverage_status = _lower(
        ((context or {}).get("source_coverage_audit") or {}).get("coverage_status")
    )
    if coverage_status not in _TARGET_SOURCE_UNUSABLE_COVERAGE_STATUSES:
        return False
    return not any(
        isinstance(chunk, dict) and _chunk_priority(chunk, context) <= 1
        for chunk in chunks
    )


def _ordered_llm_chunks(evidence_chunks: list[dict], context: dict | None) -> list[dict]:
    min_per_source = _parse_positive_int_env(
        "LLM_MUST_FETCH_MIN_CHUNKS_PER_SOURCE",
        6,
    )
    official_max = _parse_positive_int_env(
        "LLM_OFFICIAL_EXTRACTION_MAX_CHUNKS",
        30,
    )
    indexed = [
        (index, chunk)
        for index, chunk in enumerate(evidence_chunks)
        if isinstance(chunk, dict)
    ]
    protected = [
        (idx, chunk)
        for idx, chunk in indexed
        if _chunk_priority(chunk, context) <= 1
    ]
    others = [
        (idx, chunk)
        for idx, chunk in indexed
        if _chunk_priority(chunk, context) > 1
    ]

    grouped: dict[str, list[tuple[int, dict]]] = {}
    first_index_by_source: dict[str, int] = {}
    priority_by_source: dict[str, int] = {}
    for item in protected:
        idx, chunk = item
        source_id = str(chunk.get("source_id") or f"unknown_{idx}")
        grouped.setdefault(source_id, []).append(item)
        first_index_by_source[source_id] = min(
            first_index_by_source.get(source_id, idx),
            idx,
        )
        priority_by_source[source_id] = min(
            priority_by_source.get(source_id, _chunk_priority(chunk, context)),
            _chunk_priority(chunk, context),
        )
    for group in grouped.values():
        group.sort(
            key=lambda item: (
                _chunk_priority(item[1], context),
                _chunk_information_priority(item[1]),
                item[1].get("chunk_index") or item[0],
                item[0],
            )
        )
    source_order = sorted(
        grouped,
        key=lambda source_id: (
            priority_by_source.get(source_id, 99),
            first_index_by_source.get(source_id, 0),
            source_id,
        ),
    )

    protected_ordered: list[tuple[int, dict]] = []
    for round_index in range(min_per_source):
        for source_id in source_order:
            group = grouped.get(source_id) or []
            if round_index < len(group):
                protected_ordered.append(group[round_index])
    for source_id in source_order:
        group = grouped.get(source_id) or []
        protected_ordered.extend(group[min_per_source:])

    protected_ids = {id(chunk) for _, chunk in protected_ordered}
    remaining = [
        item
        for item in others
        if id(item[1]) not in protected_ids
    ]
    remaining.sort(
        key=lambda item: (
            _chunk_priority(item[1], context),
            _chunk_information_priority(item[1]),
            item[1].get("source_id") or "",
            item[1].get("chunk_index") or item[0],
            item[0],
        )
    )
    if _direct_collection_enabled(context):
        protected_first_round = [
            grouped[source_id][0]
            for source_id in source_order
            if grouped.get(source_id)
        ]
        first_round_ids = {id(chunk) for _, chunk in protected_first_round}
        protected_extras = [
            item for item in protected_ordered if id(item[1]) not in first_round_ids
        ]
        preferred_extra_count = max(0, official_max - len(protected_first_round))
        return [
            chunk
            for _, chunk in [
                *protected_first_round,
                *protected_extras[:preferred_extra_count],
                *remaining,
                *protected_extras[preferred_extra_count:],
            ]
        ]
    return [chunk for _, chunk in [*protected_ordered, *remaining]]


def _direct_llm_chunk_allowed(
    chunk: dict,
    llm_policy: LLMStructuredExtractionPolicy,
    context: dict | None,
) -> bool:
    if not _chunk_is_llm_ready(chunk, context):
        return False
    if not _direct_collection_enabled(context) or not _chunk_is_must_fetch(chunk, context):
        return _chunk_allowed_for_llm(chunk, llm_policy)
    if chunk.get("fetch_purpose") not in llm_policy.allowed_fetch_purposes:
        return False
    if (chunk.get("chunk_kind") or "text") not in llm_policy.allowed_chunk_kinds:
        return False
    return bool((chunk.get("text") or "").strip())


def _source_budget_row(
    budget_by_source: dict[str, dict],
    chunk: dict,
    context: dict | None,
) -> dict:
    source_id = str(chunk.get("source_id") or "unknown")
    official_report_key = _chunk_official_report_key(chunk, context)
    priority = _chunk_priority(chunk, context)
    if priority <= 1:
        budget_bucket = "verified_target_collection"
        target_fit_status = "verified_target"
    elif _chunk_is_official_or_high_trust(chunk, context):
        budget_bucket = "official_or_high_trust"
        target_fit_status = "non_target_or_context"
    elif _is_context_only_chunk(chunk, None):
        budget_bucket = "context"
        target_fit_status = "non_target_or_context"
    else:
        budget_bucket = "other"
        target_fit_status = "non_target_or_context"
    row = budget_by_source.setdefault(
        source_id,
        {
            "source_id": source_id,
            "official_report_key": official_report_key,
            "must_fetch": _chunk_is_must_fetch(chunk, context),
            "official_or_high_trust": _chunk_is_official_or_high_trust(chunk, context),
            "budget_bucket": budget_bucket,
            "target_fit_status": target_fit_status,
            "attempted_before_target_sources": False,
            "queued_count": 0,
            "eligible_count": 0,
            "attempted_count": 0,
            "skipped_due_to_cap_count": 0,
            "skipped_context_only_count": 0,
            "skipped_disease_mismatch_count": 0,
            "record_count": 0,
        },
    )
    row["must_fetch"] = bool(row.get("must_fetch") or _chunk_is_must_fetch(chunk, context))
    row["official_or_high_trust"] = bool(
        row.get("official_or_high_trust")
        or _chunk_is_official_or_high_trust(chunk, context)
    )
    if official_report_key and not row.get("official_report_key"):
        row["official_report_key"] = official_report_key
    if priority <= 1:
        row["budget_bucket"] = "verified_target_collection"
        row["target_fit_status"] = "verified_target"
    return row


def _has_any_llm_content_signal(record_data: dict) -> bool:
    for field in (
        "cases_confirmed",
        "cases_probable",
        "cases_suspected",
        "cases_unspecified",
        "deaths",
        "hospitalizations",
        "icu_admissions",
        "tests_positive",
        "tests_total",
        "positivity_rate",
        "incidence_rate",
        "cumulative_count",
        "new_count",
        "metric_name",
        "metric_value",
        "metric_period_start",
        "metric_period_end",
        "date_reported",
        "country",
        "subnational_location",
        "workflow_case_label",
        "age",
        "gender",
        "nationality",
        "outcome",
        "symptoms",
        "hospitalized",
        "intensive_care",
        "isolated",
        "occupation_or_role",
        "cruise_crew",
        "cruise_passenger_guest",
        "contact_with_case",
        "contact_setting",
        "ship_board_date",
        "ship_disembark_date",
        "travel_from",
        "travel_to",
        "confirmation_method",
        "accession_id",
    ):
        if record_data.get(field) is not None:
            return True
    return False


def _source_period_from_chunk(
    chunk: dict,
    context: dict | None = None,
) -> tuple[str | None, str | None, str | None]:
    start = chunk.get("reporting_period_start")
    end = chunk.get("reporting_period_end")
    label = chunk.get("reporting_period_label")
    if start and end:
        return str(start), str(end), str(label) if label else None
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    start = metadata.get("reporting_period_start")
    end = metadata.get("reporting_period_end")
    label = metadata.get("reporting_period_label")
    if start and end:
        return str(start), str(end), str(label) if label else None
    source = _source_context_for_chunk(chunk, context)
    for container in (source,):
        start = container.get("reporting_period_start")
        end = container.get("reporting_period_end")
        label = container.get("reporting_period_label")
        if start and end:
            return str(start), str(end), str(label) if label else None
    source_values = [
        chunk.get("source_url"),
        chunk.get("canonical_url"),
        source.get("canonical_url"),
        source.get("url"),
    ]
    source_keys = {
        official_report_key_for_url(str(value or ""))
        for value in source_values
        if value
    }
    source_keys.discard(None)
    for requirement in build_source_coverage_requirements(context or {}):
        req_start = requirement.get("reporting_period_start")
        req_end = requirement.get("reporting_period_end")
        if not req_start or not req_end:
            continue
        req_keys = {
            official_report_key_for_url(str(value or ""))
            for value in (requirement.get("official_candidate_urls") or [])
            if value
        }
        req_keys.discard(None)
        if source_keys and req_keys and source_keys.intersection(req_keys):
            return (
                str(req_start),
                str(req_end),
                str(requirement.get("reporting_period_label") or ""),
            )
    return None, None, None


def _source_requirement_from_chunk(
    chunk: dict,
    context: dict | None = None,
) -> dict | None:
    source = _source_context_for_chunk(chunk, context)
    source_values = [
        chunk.get("source_url"),
        chunk.get("canonical_url"),
        source.get("canonical_url"),
        source.get("url"),
    ]
    source_keys = {
        official_report_key_for_url(str(value or ""))
        for value in source_values
        if value
    }
    source_keys.discard(None)
    for requirement in build_source_coverage_requirements(context or {}):
        req_keys = {
            official_report_key_for_url(str(value or ""))
            for value in (requirement.get("official_candidate_urls") or [])
            if value
        }
        req_keys.discard(None)
        if source_keys and req_keys and source_keys.intersection(req_keys):
            return requirement
    return None


def _inherit_task_source_context_for_metric_record(
    cleaned: dict,
    chunk: dict,
    row_metadata: dict,
    context: dict | None,
) -> dict:
    from ..evidence_qualification import evidence_qualification_enabled

    # Task constraints and planner metadata are not evidence for record facts.
    if evidence_qualification_enabled():
        return cleaned
    if cleaned.get("metric_value") in (None, "") and not cleaned.get("metric_name"):
        return cleaned
    out = dict(cleaned)
    requirement = _source_requirement_from_chunk(chunk, context)
    task_location = _task_location_from_context(context)
    inherited_location = _first_text(
        row_metadata.get("country"),
        chunk.get("country"),
        requirement.get("location") if isinstance(requirement, dict) else None,
        task_location,
    )
    if not out.get("country") and _is_united_states_location(inherited_location):
        out["country"] = "United States"
    if not out.get("geographic_scope"):
        out["geographic_scope"] = _first_text(
            row_metadata.get("geographic_scope"),
            chunk.get("geographic_scope"),
            inherited_location,
        )
    if not out.get("geographic_scope_type") and out.get("geographic_scope"):
        out["geographic_scope_type"] = (
            "country"
            if _is_united_states_location(out.get("geographic_scope"))
            else "subnational"
        )

    period_start = _first_text(
        row_metadata.get("reporting_period_start"),
        chunk.get("reporting_period_start"),
        requirement.get("reporting_period_start")
        if isinstance(requirement, dict)
        else None,
    )
    period_end = _first_text(
        row_metadata.get("reporting_period_end"),
        chunk.get("reporting_period_end"),
        requirement.get("reporting_period_end")
        if isinstance(requirement, dict)
        else None,
    )
    period_label = _first_text(
        row_metadata.get("reporting_period_label"),
        chunk.get("reporting_period_label"),
        requirement.get("reporting_period_label")
        if isinstance(requirement, dict)
        else None,
    )
    if period_label and not out.get("reporting_period"):
        out["reporting_period"] = period_label
    if period_end and not out.get("date_reported"):
        out["date_reported"] = period_end
    if period_end and not out.get("date_anchor"):
        out["date_anchor"] = period_end
    return out


def _fill_metric_period_from_verified_source(
    cleaned: dict,
    chunk: dict,
    context: dict | None,
) -> dict:
    from ..evidence_qualification import evidence_qualification_enabled

    # Task constraints and planner metadata are not evidence for record facts.
    if evidence_qualification_enabled():
        return cleaned
    if not _direct_collection_enabled(context):
        return cleaned
    if cleaned.get("metric_value") in (None, "") and not cleaned.get("metric_name"):
        return cleaned
    if cleaned.get("metric_period_start") and cleaned.get("metric_period_end"):
        out = dict(cleaned)
        out["metric_period_source"] = out.get("metric_period_source") or "llm_extracted"
        return out
    if _chunk_priority(chunk, context) > 1:
        return cleaned
    start, end, _label = _source_period_from_chunk(chunk, context)
    if not start or not end:
        return cleaned
    out = dict(cleaned)
    if not out.get("metric_period_start"):
        out["metric_period_start"] = start
    if not out.get("metric_period_end"):
        out["metric_period_end"] = end
    out["metric_period_source"] = (
        out.get("metric_period_source") or "filled_from_source_reporting_period"
    )
    warnings = list(out.get("semantic_warnings") or [])
    if "filled_metric_period_from_source_reporting_period" not in warnings:
        warnings.append("filled_metric_period_from_source_reporting_period")
    out["semantic_warnings"] = warnings
    return out


_WEAK_COLUMN_LABEL_RE = re.compile(r"^column[_\s-]?\d+$", re.IGNORECASE)


def _is_weak_column_label(value: str | None) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    return bool(_WEAK_COLUMN_LABEL_RE.match(text))


def _preferred_column_label(
    llm_value: str | None,
    row_or_chunk_value: str | None,
) -> str | None:
    if _is_weak_column_label(llm_value) and not _is_weak_column_label(
        row_or_chunk_value
    ):
        return str(row_or_chunk_value)
    return str(llm_value).strip() if str(llm_value or "").strip() else (
        str(row_or_chunk_value).strip()
        if str(row_or_chunk_value or "").strip()
        else None
    )


def _column_semantics_text(
    cleaned: dict,
    row_metadata: dict,
    chunk: dict,
) -> str:
    values = [
        cleaned.get("source_column_label"),
        cleaned.get("metric_column_label"),
        row_metadata.get("source_column_label"),
        row_metadata.get("metric_column_label"),
        chunk.get("source_column_label"),
        chunk.get("metric_column_label"),
        " ".join(str(item) for item in (row_metadata.get("source_column_labels") or [])),
        " ".join(str(item) for item in (chunk.get("source_column_labels") or [])),
        row_metadata.get("table_header"),
        chunk.get("table_header"),
        row_metadata.get("heading_context"),
        chunk.get("heading_context"),
        cleaned.get("reporting_period"),
        row_metadata.get("reporting_period_label"),
        chunk.get("reporting_period_label"),
    ]
    return " ".join(str(value or "") for value in values).lower()


def _cumulative_reporting_period_label(
    cleaned: dict,
    row_metadata: dict,
    chunk: dict,
) -> str | None:
    labels = [
        cleaned.get("metric_column_label"),
        cleaned.get("source_column_label"),
        row_metadata.get("metric_column_label"),
        row_metadata.get("source_column_label"),
        chunk.get("metric_column_label"),
        chunk.get("source_column_label"),
    ]
    clean_labels = [
        str(label).strip()
        for label in labels
        if str(label or "").strip() and not _is_weak_column_label(str(label))
    ]
    for label in clean_labels:
        lowered = label.lower()
        if "through" in lowered and any(
            marker in lowered for marker in ("season", "cumulative", "to date")
        ):
            return label
    cumulative_label = next(
        (
            label
            for label in clean_labels
            if any(
                marker in label.lower()
                for marker in ("season", "cumulative", "to date")
            )
        ),
        None,
    )
    period_label = (
        cleaned.get("reporting_period")
        or row_metadata.get("reporting_period_label")
        or chunk.get("reporting_period_label")
    )
    if cumulative_label and period_label:
        return f"{cumulative_label} through {period_label}"
    if period_label:
        from ..evidence_qualification import evidence_qualification_enabled
        if evidence_qualification_enabled():
            return str(period_label)
        return f"Season-to-date through {period_label}"
    return cumulative_label


def _parse_iso_date_value(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except (TypeError, ValueError):
        return None


def _previous_period_from_source(
    chunk: dict,
    context: dict | None,
) -> tuple[str | None, str | None]:
    start, end, _label = _source_period_from_chunk(chunk, context)
    start_date = _parse_iso_date_value(start)
    end_date = _parse_iso_date_value(end)
    if not start_date or not end_date:
        return None, None
    if end_date < start_date:
        start_date, end_date = end_date, start_date
    period_length = (end_date - start_date).days + 1
    previous_end = start_date - timedelta(days=1)
    previous_start = previous_end - timedelta(days=period_length - 1)
    return previous_start.isoformat(), previous_end.isoformat()


def _metric_column_label_text(
    cleaned: dict,
    row_metadata: dict,
    chunk: dict,
) -> str:
    labels = [
        cleaned.get("source_column_label"),
        cleaned.get("metric_column_label"),
        row_metadata.get("source_column_label"),
        row_metadata.get("metric_column_label"),
        chunk.get("source_column_label"),
        chunk.get("metric_column_label"),
    ]
    return " ".join(str(label or "") for label in labels).strip().lower()


_NARRATIVE_WEEK_RE = re.compile(
    r"\b(?:during|for|in)\s+(?:mmwr\s+)?week\s+\d{1,2}\b|\bthis week\b",
    re.IGNORECASE,
)


def _row_context_type_set(row_metadata: dict, chunk: dict) -> set[str]:
    values = {
        str(row_metadata.get("row_context_type") or ""),
        str(chunk.get("row_context_type") or ""),
    }
    values.update(str(item) for item in (row_metadata.get("context_types") or []))
    values.update(str(item) for item in (chunk.get("context_types") or []))
    return {value for value in values if value}


def _narrative_metric_text(row_metadata: dict, chunk: dict) -> str:
    return str(
        row_metadata.get("row_quote")
        or row_metadata.get("text")
        or chunk.get("row_quote")
        or chunk.get("text")
        or ""
    )


def _is_narrative_metric_line(row_metadata: dict, chunk: dict) -> bool:
    text = _narrative_metric_text(row_metadata, chunk)
    if not text or "|" in text:
        return False
    context_types = _row_context_type_set(row_metadata, chunk)
    return bool(
        "markdown_metric_line" in context_types
        or row_metadata.get("row_context_type") == "markdown_metric_line"
        or chunk.get("row_context_type") == "markdown_metric_line"
    )


def _source_period_available(chunk: dict, context: dict | None) -> bool:
    start, end, _label = _source_period_from_chunk(chunk, context)
    return bool(start and end)


def _narrative_metric_resolves_current_period(
    row_metadata: dict,
    chunk: dict,
    context: dict | None,
) -> bool:
    if not _is_narrative_metric_line(row_metadata, chunk):
        return False
    if not _source_period_available(chunk, context):
        return False
    text = _narrative_metric_text(row_metadata, chunk)
    return bool(_NARRATIVE_WEEK_RE.search(text))


def _resolve_metric_column_period_type(
    cleaned: dict,
    row_metadata: dict,
    chunk: dict,
    context: dict | None = None,
) -> tuple[str, str, str, list[str]]:
    label_text = _metric_column_label_text(cleaned, row_metadata, chunk)
    semantics_text = _column_semantics_text(cleaned, row_metadata, chunk)
    source_label = str(cleaned.get("source_column_label") or "").strip()
    metric_label = str(cleaned.get("metric_column_label") or "").strip()
    warning_flags: list[str] = []
    cumulative_markers = (
        "cumulative",
        "season-to-date",
        "season to date",
        "season_to_date",
        "to date",
        "through week",
        "since season",
    )
    previous_markers = (
        "previous week",
        "prior week",
        "last week",
        "previous period",
        "prior period",
    )
    current_markers = (
        "current week",
        "this week",
        "reporting week",
        "current period",
    )
    if any(marker in semantics_text for marker in cumulative_markers):
        return (
            "resolved",
            "cumulative_period",
            "resolved_from_cumulative_column_label",
            warning_flags,
        )
    if any(marker in label_text for marker in previous_markers):
        return (
            "resolved",
            "previous_period",
            "resolved_from_previous_column_label",
            warning_flags,
        )
    if any(marker in label_text for marker in current_markers):
        return (
            "resolved",
            "current_period",
            "resolved_from_current_column_label",
            warning_flags,
        )
    if _narrative_metric_resolves_current_period(row_metadata, chunk, context):
        return (
            "resolved",
            "current_period",
            "resolved_from_narrative_week_phrase",
            warning_flags,
        )
    if _is_weak_column_label(source_label) or _is_weak_column_label(metric_label):
        warning_flags.append("ambiguous_metric_column_semantics")
        return (
            "ambiguous",
            "ambiguous_column",
            "weak_column_label_without_header_semantics",
            warning_flags,
        )
    if source_label or metric_label:
        return (
            "resolved",
            "current_period",
            "treated_as_current_period_from_explicit_metric_column_label",
            warning_flags,
        )
    warning_flags.append("ambiguous_metric_column_semantics")
    return (
        "ambiguous",
        "ambiguous_column",
        "missing_column_label_semantics",
        warning_flags,
    )


def _apply_metric_column_semantics(
    cleaned: dict,
    row_metadata: dict,
    chunk: dict,
    context: dict | None = None,
) -> dict:
    if cleaned.get("metric_value") in (None, "") and not cleaned.get("metric_name"):
        return cleaned
    out = dict(cleaned)
    source_label = _preferred_column_label(
        out.get("source_column_label"),
        row_metadata.get("source_column_label") or chunk.get("source_column_label"),
    )
    metric_label = _preferred_column_label(
        out.get("metric_column_label"),
        row_metadata.get("metric_column_label") or chunk.get("metric_column_label"),
    )
    if source_label:
        out["source_column_label"] = source_label
    if metric_label:
        out["metric_column_label"] = metric_label

    (
        semantics_status,
        period_type,
        resolution_reason,
        column_warning_flags,
    ) = _resolve_metric_column_period_type(out, row_metadata, chunk, context)
    out["metric_column_semantics_status"] = semantics_status
    out["resolved_column_period_type"] = period_type
    out["column_period_resolution_reason"] = resolution_reason
    if resolution_reason == "resolved_from_narrative_week_phrase":
        out["column_semantics_resolution_method"] = "narrative_week_phrase"
        out["column_semantics_confidence"] = 0.85
    elif semantics_status == "resolved":
        out["column_semantics_resolution_method"] = "column_label"
        out["column_semantics_confidence"] = 0.95
    else:
        out["column_semantics_resolution_method"] = "unresolved"
        out["column_semantics_confidence"] = 0.0
    merged_column_warnings = list(out.get("column_period_warning_flags") or [])
    for warning in column_warning_flags:
        if warning not in merged_column_warnings:
            merged_column_warnings.append(warning)
    out["column_period_warning_flags"] = merged_column_warnings

    semantic_warnings = list(out.get("semantic_warnings") or [])
    for warning in column_warning_flags:
        if warning not in semantic_warnings:
            semantic_warnings.append(warning)
    out["semantic_warnings"] = semantic_warnings

    if period_type == "previous_period":
        previous_start, previous_end = _previous_period_from_source(chunk, context)
        if previous_start and previous_end:
            out["metric_period_start"] = previous_start
            out["metric_period_end"] = previous_end
            out["metric_period_source"] = "filled_from_previous_column_label"
            out["date_reported"] = previous_end
            out["date_anchor"] = previous_end
        else:
            out["metric_column_semantics_status"] = "ambiguous"
            out["resolved_column_period_type"] = "ambiguous_previous_period"
            out["column_period_resolution_reason"] = (
                "previous_column_label_without_source_reporting_period"
            )
            if "ambiguous_previous_period" not in out["column_period_warning_flags"]:
                out["column_period_warning_flags"].append("ambiguous_previous_period")
            if "ambiguous_previous_period" not in out["semantic_warnings"]:
                out["semantic_warnings"].append("ambiguous_previous_period")
        return out

    if period_type == "current_period":
        if resolution_reason == "resolved_from_narrative_week_phrase":
            out["metric_period_source"] = "filled_from_narrative_week_phrase"
        else:
            out["metric_period_source"] = (
                out.get("metric_period_source") or "filled_from_source_reporting_period"
            )
        return out

    if period_type == "cumulative_period":
        out["statistical_count_type"] = "cumulative"
        out["count_semantics"] = "cumulative"
        out["metric_period_source"] = "filled_from_column_label"
        cumulative_period = _cumulative_reporting_period_label(
            out,
            row_metadata,
            chunk,
        )
        if cumulative_period:
            out["reporting_period"] = cumulative_period
            out["metric_period_label"] = cumulative_period
        if "inferred_cumulative_metric_from_column_label" not in out["semantic_warnings"]:
            out["semantic_warnings"].append(
                "inferred_cumulative_metric_from_column_label"
            )
        return out

    return out


_METRIC_ROW_STOPWORDS = {
    "number",
    "specimens",
    "specimen",
    "total",
    "count",
    "rate",
    "percent",
    "percentage",
    "reported",
    "reports",
    "with",
    "from",
    "this",
    "that",
    "week",
}


def _metric_text_tokens(value: str | None) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) >= 3 and token not in _METRIC_ROW_STOPWORDS
    }
    return tokens


def _record_value_tokens(cleaned: dict) -> set[str]:
    tokens: set[str] = set()
    for key in (
        "metric_value",
        "tests_positive",
        "tests_total",
        "hospitalizations",
        "deaths",
        "cases_confirmed",
        "cases_probable",
        "cases_suspected",
        "cases_unspecified",
        "positivity_rate",
        "incidence_rate",
    ):
        value = cleaned.get(key)
        if value in (None, ""):
            continue
        text = str(value)
        tokens.add(text)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric.is_integer():
            tokens.add(str(int(numeric)))
        tokens.add(f"{numeric:g}")
    return tokens


def _row_match_score(row_metadata: dict, row_quote: str, cleaned: dict) -> int:
    row_text = str(row_quote or row_metadata.get("row_quote") or row_metadata.get("text") or "")
    row_lower = row_text.lower()
    row_numbers = set(row_metadata.get("row_numeric_tokens") or _numeric_tokens(row_text))
    value_tokens = _record_value_tokens(cleaned)
    score = 0
    if value_tokens and row_numbers.intersection(value_tokens):
        score += 6
    metric_tokens = (
        _metric_text_tokens(cleaned.get("metric_name"))
        | _metric_text_tokens(cleaned.get("metric_category"))
    )
    if metric_tokens:
        overlap = {token for token in metric_tokens if token in row_lower}
        score += len(overlap) * 2
        if len(overlap) >= min(2, len(metric_tokens)):
            score += 2
    return score


def _resolve_metric_row_binding(
    cleaned: dict,
    chunk: dict,
    requested_row_id: str | None,
) -> tuple[str | None, dict, str, list[str]]:
    row_quote_by_id = chunk.get("row_quote_by_id") or {}
    row_metadata_by_id = chunk.get("row_metadata_by_id") or {}
    warnings: list[str] = []
    if not isinstance(row_metadata_by_id, dict) or not row_metadata_by_id:
        return requested_row_id, {}, "not_applicable", warnings

    def _metadata_for(row_id: str | None) -> dict:
        if not row_id:
            return {}
        metadata = row_metadata_by_id.get(str(row_id))
        return metadata if isinstance(metadata, dict) else {}

    def _quote_for(row_id: str | None, metadata: dict) -> str:
        if row_id and isinstance(row_quote_by_id, dict):
            quote = row_quote_by_id.get(str(row_id))
            if quote:
                return str(quote)
        return str(metadata.get("row_quote") or metadata.get("text") or "")

    requested_metadata = _metadata_for(requested_row_id)
    if requested_metadata:
        requested_score = _row_match_score(
            requested_metadata,
            _quote_for(requested_row_id, requested_metadata),
            cleaned,
        )
        if requested_score > 0:
            return requested_row_id, requested_metadata, "resolved", warnings

    best_row_id: str | None = None
    best_metadata: dict = {}
    best_score = 0
    for candidate_row_id, metadata in row_metadata_by_id.items():
        if not isinstance(metadata, dict):
            continue
        score = _row_match_score(
            metadata,
            _quote_for(str(candidate_row_id), metadata),
            cleaned,
        )
        if score > best_score:
            best_score = score
            best_row_id = str(candidate_row_id)
            best_metadata = metadata

    if best_row_id and best_score > 0:
        if requested_row_id and best_row_id != str(requested_row_id):
            warnings.append("metric_row_rebound_from_llm_source_row_id")
            return best_row_id, best_metadata, "rebinding_resolved", warnings
        return best_row_id, best_metadata, "resolved", warnings

    warnings.append("metric_row_binding_unresolved")
    return requested_row_id, requested_metadata, "unresolved", warnings


def _bind_universal_field_provenance(cleaned, chunk, row_metadata, chunk_id, quote, legacy):
    """Preserve quotations, but bind identifiers to the actual extracted row."""
    from ..evidence_qualification import _NUMERIC, _DATES, _SUBSTANTIVE, _GEO, _IDENTITY_FIELDS
    supplied = cleaned.get('field_provenance_json') or {}
    if isinstance(supplied, str):
        try:
            supplied = json.loads(supplied)
        except (ValueError, TypeError):
            supplied = {}
    if not isinstance(supplied, dict):
        supplied = {}
    factual = set(('disease',) + _NUMERIC + _DATES + _SUBSTANTIVE + _GEO) | _IDENTITY_FIELDS
    result = {}
    for name in sorted(factual):
        value = cleaned.get(name)
        if value in (None, '', [], {}):
            continue
        entry = supplied.get(name)
        entry = dict(entry) if isinstance(entry, dict) else {}
        citation = next((entry[k] for k in ('quote','evidence_quote','supporting_quote') if k in entry), quote)
        bound = {**(legacy.get(name) or {}), **entry,
                 'field_name':name, 'extracted_value':value, 'quote':citation,
                 'supporting_quote':citation, 'source_id':chunk.get('source_id'), 'chunk_id':chunk_id}
        # Model-supplied document references cannot select another patient or row.
        for key in ('document_id','document_hash'):
            actual = row_metadata.get(key) or chunk.get(key)
            if actual:
                bound[key] = actual
            else:
                bound.pop(key,None)
        result[name] = bound
    return result


def _build_record_from_llm_output(
    llm_record: LLMExtractedRecord,
    chunk: dict,
    index: int,
    llm_policy: LLMStructuredExtractionPolicy,
    settings: dict,
    context: dict | None = None,
) -> PublicHealthRecord | None:
    from ..evidence_qualification import evidence_qualification_enabled
    universal = evidence_qualification_enabled()
    data = llm_record.model_dump()
    if not _has_any_llm_content_signal(data):
        return None

    # Bind the transport batch before interpreting any words or columns.
    # Neighboring rows may describe entirely different indicators.
    row_quote_by_id = chunk.get("row_quote_by_id") or {}
    row_chunk_id_by_id = chunk.get("row_chunk_id_by_id") or {}
    if universal:
        requested_source_row_id = data.get("source_row_id") or chunk.get("row_id")
        source_row_id, row_metadata, metric_row_binding_status, binding_warnings = (
            _resolve_metric_row_binding(data, chunk, requested_source_row_id)
        )
        if chunk.get("chunk_kind") == "metric_row_batch":
            local_quote = row_quote_by_id.get(str(source_row_id), "")
            chunk = {**chunk, **row_metadata, "text": local_quote,
                     "chunk_id": row_chunk_id_by_id.get(str(source_row_id)),
                     "bound_context_spans": list(row_metadata.get("bound_context_spans") or [])}
            # Never fall back to the first row's context for an unresolved batch.
            if not row_metadata:
                for key in ("table_header", "heading_context", "source_column_label",
                            "metric_column_label", "source_column_labels",
                            "reporting_period_start", "reporting_period_end",
                            "char_start", "char_end"):
                    chunk[key] = None

    source_id = chunk.get("source_id") or ""
    record_id = f"rec_{source_id}_{index:03d}"
    text = chunk.get("text") or ""
    extraction_confidence = chunk.get("confidence")
    if extraction_confidence is None:
        extraction_confidence = 0.50

    cleaned = _apply_extraction_semantic_guardrails(data, chunk, context)
    cleaned = _fill_metric_period_from_verified_source(cleaned, chunk, context)
    if not universal:
        requested_source_row_id = cleaned.get("source_row_id") or chunk.get("row_id")
        source_row_id, row_metadata, metric_row_binding_status, binding_warnings = (
            _resolve_metric_row_binding(cleaned, chunk, requested_source_row_id)
        )
    if binding_warnings:
        warnings = list(cleaned.get("semantic_warnings") or [])
        for warning in binding_warnings:
            if warning not in warnings:
                warnings.append(warning)
        cleaned["semantic_warnings"] = warnings
    if (
        row_metadata
        and not cleaned.get("metric_period_start")
        and row_metadata.get("reporting_period_start")
    ):
        cleaned["metric_period_start"] = row_metadata.get("reporting_period_start")
        cleaned["metric_period_end"] = row_metadata.get("reporting_period_end")
        cleaned["metric_period_source"] = "filled_from_row_column_label"
    cleaned = _inherit_task_source_context_for_metric_record(
        cleaned,
        chunk,
        row_metadata,
        context,
    )

    def _row_or_chunk(key: str):
        value = row_metadata.get(key)
        return value if value not in (None, "") else chunk.get(key)

    cleaned = _apply_metric_column_semantics(cleaned, row_metadata, chunk, context)

    evidence_quote = text
    supporting_chunk_id = chunk.get("chunk_id")
    if source_row_id and isinstance(row_quote_by_id, dict):
        row_quote = row_quote_by_id.get(str(source_row_id))
        if row_quote:
            evidence_quote = str(row_quote)
    elif chunk.get("row_quote"):
        evidence_quote = str(chunk.get("row_quote"))
    if source_row_id and isinstance(row_chunk_id_by_id, dict):
        supporting_chunk_id = row_chunk_id_by_id.get(str(source_row_id)) or supporting_chunk_id
    case_span_quote = _first_text(
        cleaned.get("case_span_quote"),
        chunk.get("case_span_quote"),
    )
    case_span_id = _first_text(
        cleaned.get("case_span_id"),
        chunk.get("case_span_id"),
    )
    case_span_start = cleaned.get("case_span_start")
    if case_span_start is None:
        case_span_start = chunk.get("case_span_start")
    case_span_end = cleaned.get("case_span_end")
    if case_span_end is None:
        case_span_end = chunk.get("case_span_end")
    case_span_extraction_method = _first_text(
        cleaned.get("case_span_extraction_method"),
        chunk.get("case_span_extraction_method"),
    )
    if case_span_quote:
        evidence_quote = case_span_quote
    elif universal and cleaned.get("evidence_quote") is not None:
        evidence_quote = cleaned["evidence_quote"]
    cleaned, unsupported_case_fields, case_field_methods = (
        _recover_case_fields_from_local_span(cleaned, case_span_quote)
    )
    record_chunk_kind = _row_or_chunk("chunk_kind") or "text"
    record_data_types = list(_row_or_chunk("data_types") or [])
    record_context_types = sorted(
        {
            *[str(item) for item in (chunk.get("context_types") or [])],
            *[str(item) for item in (row_metadata.get("context_types") or [])],
        }
    )
    row_context_type = _row_or_chunk("row_context_type")
    if not row_context_type and "markdown_metric_line" in record_context_types:
        row_context_type = "markdown_metric_line"
    elif not row_context_type and "markdown_table_row" in record_context_types:
        row_context_type = "markdown_table_row"
    disease = (
        cleaned.get("disease")
        or ("" if universal else (context or {}).get("disease_standard_name") or "Hantavirus disease")
    )

    required_core = ("disease", "source_url", "source_type", "evidence_quote")
    field_values = {
        "disease": disease,
        "source_url": chunk.get("source_url"),
        "source_type": chunk.get("source_type"),
        "evidence_quote": evidence_quote,
    }
    missing_fields = [
        f
        for f in required_core
        if not field_values.get(f)
        or (isinstance(field_values.get(f), str) and not str(field_values[f]).strip())
    ]
    compatibility = assess_record_disease_compatibility(
        {
            **cleaned,
            "disease": disease,
            "evidence_quote": evidence_quote,
            "source_title": chunk.get("title"),
            "source_url": chunk.get("source_url"),
        },
        _as_disease_relevance_context(context),
    )
    case_field_names = (
        "workflow_case_label",
        "age",
        "gender",
        "nationality",
        "date_onset",
        "date_confirmation",
        "date_death",
        "outcome",
        "symptoms",
        "hospitalized",
        "intensive_care",
        "isolated",
        "occupation_or_role",
        "cruise_crew",
        "cruise_passenger_guest",
        "contact_with_case",
        "contact_setting",
        "ship_board_date",
        "ship_disembark_date",
        "travel_from",
        "travel_to",
        "confirmation_method",
        "accession_id",
    )
    provenance_quote = case_span_quote or evidence_quote
    field_provenance = {
        field_name: {
            "field_name": field_name,
            "extracted_value": cleaned.get(field_name),
            "source_id": source_id,
            "chunk_id": supporting_chunk_id,
            "case_span_id": case_span_id,
            "supporting_quote": provenance_quote,
            "extraction_method": (
                case_field_methods.get(field_name)
                or case_span_extraction_method
                or llm_policy.llm_extraction_method
            ),
        }
        for field_name in case_field_names
        if cleaned.get(field_name) not in (None, "", [], {})
    }

    if universal:
        # Cite normalized count semantics through the same source span as its origin.
        cleaned["count_semantics"] = cleaned.get("count_semantics") or cleaned.get("statistical_count_type")
        field_provenance = _bind_universal_field_provenance(
            cleaned, chunk, row_metadata, supporting_chunk_id, evidence_quote, field_provenance
        )

    record = PublicHealthRecord(
        record_id=record_id,
        disease=disease,
        disease_standard_name=cleaned.get("disease_standard_name") or disease,
        disease_alias_used=cleaned.get("disease_alias_used")
        or _detect_disease_alias_used(text, context),
        virus_or_syndrome=cleaned.get("virus_or_syndrome"),
        pathogen_or_syndrome=cleaned.get("pathogen_or_syndrome")
        or (None if universal else _detect_pathogen_or_syndrome(text, context)),
        target_population=(context or {}).get("target_population"),
        observation_type=cleaned.get("observation_type"),
        observation_types=list(cleaned.get("observation_types") or []),
        primary_case_dataset_eligible=cleaned.get(
            "primary_case_dataset_eligible"
        ),
        country=cleaned.get("country"),
        subnational_location=cleaned.get("subnational_location"),
        date_reported=cleaned.get("date_reported"),
        event_start_date=cleaned.get("event_start_date"),
        event_end_date=cleaned.get("event_end_date"),
        date_onset=cleaned.get("date_onset"),
        date_confirmation=cleaned.get("date_confirmation"),
        date_death=cleaned.get("date_death"),
        cases_confirmed=cleaned.get("cases_confirmed"),
        cases_probable=cleaned.get("cases_probable"),
        cases_suspected=cleaned.get("cases_suspected"),
        cases_unspecified=cleaned.get("cases_unspecified"),
        deaths=cleaned.get("deaths"),
        age=cleaned.get("age"),
        gender=cleaned.get("gender"),
        nationality=cleaned.get("nationality"),
        outcome=cleaned.get("outcome"),
        symptoms=cleaned.get("symptoms"),
        hospitalized=cleaned.get("hospitalized"),
        intensive_care=cleaned.get("intensive_care"),
        isolated=cleaned.get("isolated"),
        occupation_or_role=cleaned.get("occupation_or_role"),
        cruise_crew=cleaned.get("cruise_crew"),
        cruise_passenger_guest=cleaned.get("cruise_passenger_guest"),
        contact_with_case=cleaned.get("contact_with_case"),
        contact_setting=cleaned.get("contact_setting"),
        ship_board_date=cleaned.get("ship_board_date"),
        ship_disembark_date=cleaned.get("ship_disembark_date"),
        travel_from=cleaned.get("travel_from"),
        travel_to=cleaned.get("travel_to"),
        confirmation_method=cleaned.get("confirmation_method"),
        accession_id=cleaned.get("accession_id"),
        workflow_case_label=cleaned.get("workflow_case_label"),
        patient_links=cleaned.get("patient_links") or [],
        disjoint_partition=cleaned.get("disjoint_partition"),
        case_span_id=case_span_id,
        case_span_quote=case_span_quote,
        case_span_start=case_span_start,
        case_span_end=case_span_end,
        case_span_extraction_method=case_span_extraction_method,
        field_provenance_json=json.dumps(
            field_provenance,
            ensure_ascii=False,
            sort_keys=True,
        ),
        unsupported_case_fields=unsupported_case_fields,
        focused_recovery_status=chunk.get("focused_recovery_status"),
        hospitalizations=cleaned.get("hospitalizations"),
        icu_admissions=cleaned.get("icu_admissions"),
        tests_positive=cleaned.get("tests_positive"),
        tests_total=cleaned.get("tests_total"),
        positivity_rate=cleaned.get("positivity_rate"),
        incidence_rate=cleaned.get("incidence_rate"),
        cumulative_count=cleaned.get("cumulative_count"),
        new_count=cleaned.get("new_count"),
        metric_name=cleaned.get("metric_name"),
        metric_value=cleaned.get("metric_value"),
        metric_unit=cleaned.get("metric_unit"),
        metric_category=cleaned.get("metric_category"),
        metric_denominator=cleaned.get("metric_denominator"),
        metric_period_start=cleaned.get("metric_period_start"),
        metric_period_end=cleaned.get("metric_period_end"),
        metric_period_source=cleaned.get("metric_period_source"),
        source_column_label=cleaned.get("source_column_label")
        or _row_or_chunk("source_column_label"),
        metric_column_label=cleaned.get("metric_column_label")
        or _row_or_chunk("metric_column_label"),
        metric_row_binding_status=metric_row_binding_status,
        metric_column_semantics_status=cleaned.get(
            "metric_column_semantics_status"
        ),
        resolved_column_period_type=cleaned.get("resolved_column_period_type"),
        column_period_resolution_reason=cleaned.get(
            "column_period_resolution_reason"
        ),
        column_period_warning_flags=list(
            cleaned.get("column_period_warning_flags") or []
        ),
        metric_period_label=cleaned.get("metric_period_label"),
        column_semantics_resolution_method=cleaned.get(
            "column_semantics_resolution_method"
        ),
        column_semantics_confidence=cleaned.get("column_semantics_confidence"),
        source_column_labels=list(_row_or_chunk("source_column_labels") or []),
        table_header=_row_or_chunk("table_header"),
        heading_context=_row_or_chunk("heading_context"),
        row_context_type=row_context_type,
        case_definition=cleaned.get("case_definition"),
        source_id=source_id,
        source_url=chunk.get("source_url"),
        source_type=chunk.get("source_type"),
        evidence_quote=evidence_quote,
        extraction_confidence=float(extraction_confidence),
        missing_fields=missing_fields,
        schema_status=None,
        provenance_status=None,
        supporting_chunk_id=supporting_chunk_id,
        source_row_id=source_row_id,
        source_title=chunk.get("title"),
        publisher=chunk.get("publisher"),
        source_role_final=chunk.get("source_role_final"),
        credibility_score=chunk.get("credibility_score"),
        credibility_level=chunk.get("credibility_level"),
        actual_publisher=chunk.get("actual_publisher"),
        actual_publisher_normalized=chunk.get("actual_publisher_normalized"),
        source_type_final=chunk.get("source_type_final"),
        source_independence_group=chunk.get("source_independence_group"),
        claim_support_role=chunk.get("claim_support_role"),
        recommended_source_role=chunk.get("recommended_source_role"),
        recommended_fetch_use=chunk.get("recommended_fetch_use"),
        recommended_extraction_use=chunk.get("recommended_extraction_use"),
        likely_syndicated_or_aggregated=chunk.get(
            "likely_syndicated_or_aggregated"
        ),
        upstream_source_mentions=list(chunk.get("upstream_source_mentions") or []),
        discovery_method=chunk.get("discovery_method"),
        search_provider=chunk.get("search_provider"),
        query_id=chunk.get("query_id"),
        query_used=chunk.get("query_used"),
        document_id=chunk.get("document_id"),
        document_type=chunk.get("document_type"),
        fetch_purpose=chunk.get("fetch_purpose"),
        chunk_kind=record_chunk_kind,
        data_types=record_data_types,
        context_types=record_context_types,
        extraction_method=llm_policy.llm_extraction_method,
        extraction_reason="LLM structured extraction from evidence chunk",
        validation_errors=[],
        repair_actions=[],
        requires_human_review=False,
        llm_used=True,
        llm_model=settings.get("model"),
        llm_provider=settings.get("provider"),
        llm_extraction_error=None,
        extraction_mode="llm",
        statistical_count_type=cleaned.get("statistical_count_type"),
        count_semantics=cleaned.get("count_semantics") or cleaned.get("statistical_count_type") or (None if get_env("PIPELINE_MODE") == "evidence" else "unspecified"),
        reporting_period=cleaned.get("reporting_period"),
        as_of_date=cleaned.get("as_of_date"),
        aggregation_level=cleaned.get("aggregation_level"),
        geographic_scope=cleaned.get("geographic_scope"),
        geographic_scope_type=cleaned.get("geographic_scope_type"),
        population_scope=cleaned.get("population_scope"),
        source_section=cleaned.get("source_section"),
        semantic_warnings=list(cleaned.get("semantic_warnings") or []),
        extraction_warnings=list(cleaned.get("semantic_warnings") or []),
        **record_compatibility_fields(compatibility),
        record_schema="generic_public_health_record",
        legacy_record_type=(
            "ObservationRecord" if ((context or {}).get("is_hantavirus") is True) else None
        ),
    )
    if universal:
        from ..record_identity import stable_record_id
        record.record_id = stable_record_id(record, chunk)
    return record


def _rule_based_extract_records_from_chunks(
    evidence_chunks: list[dict],
    deterministic_policy: StructuredExtractionPolicy,
    start_index: int = 1,
    only_chunk_ids: set[str] | None = None,
    context: dict | None = None,
) -> tuple[list[PublicHealthRecord], dict]:
    """Refactored deterministic loop; preserves Step 7 behavior."""

    records: list[PublicHealthRecord] = []
    attempted_chunk_ids = []
    chunk_index_by_source: dict[str, int] = {}
    target_data_count = 0
    extractable_count = 0
    skipped_count = 0
    skipped_context_only_chunk_count = 0
    skipped_context_only_source_ids: set[str] = set()
    skipped_disease_mismatch_chunk_count = 0
    skipped_disease_mismatch_source_ids: set[str] = set()
    field_counters: Counter = Counter()
    role_policy = load_source_role_policy()

    for chunk in evidence_chunks:
        if only_chunk_ids is not None and chunk.get("chunk_id") not in only_chunk_ids:
            continue
        if _is_context_only_chunk(chunk, role_policy) and (
            _direct_collection_enabled(context)
            or _chunk_has_explicit_context_only_role(chunk, role_policy)
        ):
            skipped_count += 1
            skipped_context_only_chunk_count += 1
            if chunk.get("source_id"):
                skipped_context_only_source_ids.add(chunk.get("source_id"))
            continue
        disease_blocked, disease_assessment = _chunk_blocked_by_disease_gate(
            chunk, context
        )
        if disease_blocked:
            skipped_count += 1
            skipped_disease_mismatch_chunk_count += 1
            if chunk.get("source_id"):
                skipped_disease_mismatch_source_ids.add(chunk.get("source_id"))
            continue
        if chunk.get("contains_target_data"):
            target_data_count += 1
        if not _chunk_is_extractable(chunk, deterministic_policy, context):
            skipped_count += 1
            continue
        extractable_count += 1
        if chunk.get("chunk_id"):
            attempted_chunk_ids.append(str(chunk["chunk_id"]))
        source_id = chunk.get("source_id") or ""
        if source_id not in chunk_index_by_source:
            chunk_index_by_source[source_id] = start_index - 1
        chunk_index_by_source[source_id] += 1
        idx = chunk_index_by_source[source_id]

        record = _build_record_from_chunk(
            chunk,
            idx,
            deterministic_policy,
            context,
        )
        if record is None:
            core_records = _build_core_metric_records_from_chunk(
                chunk,
                idx,
                deterministic_policy,
                context,
            )
            if not core_records:
                skipped_count += 1
                chunk_index_by_source[source_id] -= 1
                continue
            records.extend(core_records)
            chunk_index_by_source[source_id] = (
                chunk_index_by_source.get(source_id, 0) + len(core_records) - 1
            )
            for core_record in core_records:
                rec_dict = core_record.model_dump()
                for field in _FIELD_DETECTION_KEYS:
                    if rec_dict.get(field) is not None:
                        field_counters[field] += 1
            continue
        records.append(record)
        rec_dict = record.model_dump()
        for field in _FIELD_DETECTION_KEYS:
            if rec_dict.get(field) is not None:
                field_counters[field] += 1
        core_records = _build_core_metric_records_from_chunk(
            chunk,
            idx + 1,
            deterministic_policy,
            context,
        )
        core_records = _filter_redundant_core_metric_records(
            record,
            core_records,
            context,
        )
        if core_records:
            records.extend(core_records)
            chunk_index_by_source[source_id] = (
                chunk_index_by_source.get(source_id, 0) + len(core_records)
            )
            for core_record in core_records:
                rec_dict = core_record.model_dump()
                for field in _FIELD_DETECTION_KEYS:
                    if rec_dict.get(field) is not None:
                        field_counters[field] += 1

    stats = {
        "attempted_chunk_ids": attempted_chunk_ids,
        "target_data_chunk_count": target_data_count,
        "extractable_chunk_count": extractable_count,
        "skipped_chunk_count": skipped_count,
        "skipped_context_only_chunk_count": skipped_context_only_chunk_count,
        "skipped_context_only_source_ids": sorted(skipped_context_only_source_ids),
        "skipped_disease_mismatch_chunk_count": skipped_disease_mismatch_chunk_count,
        "skipped_disease_mismatch_source_ids": sorted(
            skipped_disease_mismatch_source_ids
        ),
        "field_detection_counts": {
            f: field_counters.get(f, 0) for f in _FIELD_DETECTION_KEYS
        },
    }
    return records, stats


def _chunk_text_preview(chunk: dict, *, max_chars: int = 280) -> str:
    text = str(chunk.get("row_quote") or chunk.get("text") or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def _sample_chunks_for_source(
    evidence_chunks: list[dict],
    source_id: str,
    *,
    max_samples: int = 3,
) -> list[dict]:
    samples: list[dict] = []
    for chunk in evidence_chunks:
        if str(chunk.get("source_id") or "") != source_id:
            continue
        samples.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "chunk_kind": chunk.get("chunk_kind") or "text",
                "chunk_index": chunk.get("chunk_index"),
                "row_id": chunk.get("row_id"),
                "table_id": chunk.get("table_id"),
                "contains_target_data": chunk.get("contains_target_data"),
                "extraction_eligible_for_task_disease": (
                    chunk.get("extraction_eligible_for_task_disease")
                ),
                "disease_relevance_status": chunk.get("disease_relevance_status"),
                "data_types": list(chunk.get("data_types") or []),
                "context_types": list(chunk.get("context_types") or []),
                "text_preview": _chunk_text_preview(chunk),
            }
        )
        if len(samples) >= max_samples:
            break
    return samples


def _core_metric_extraction_gaps(
    evidence_chunks: list[dict],
    extraction_budget_by_source: dict[str, dict],
    context: dict | None,
) -> list[dict]:
    chunks_by_source: dict[str, list[dict]] = {}
    for chunk in evidence_chunks:
        source_id = str(chunk.get("source_id") or "")
        if source_id:
            chunks_by_source.setdefault(source_id, []).append(chunk)

    gaps: list[dict] = []
    for source_id, row in sorted(extraction_budget_by_source.items()):
        if int(row.get("attempted_count") or 0) <= 0:
            continue
        if int(row.get("record_count") or 0) > 0:
            continue
        source_chunks = chunks_by_source.get(str(source_id), [])
        core_chunks = [
            chunk
            for chunk in source_chunks
            if _chunk_looks_like_core_metric_text(chunk)
            and not _is_context_only_chunk(chunk, None)
            and (
                _chunk_is_task_collection_candidate(chunk, context)
                or _chunk_is_official_or_high_trust(chunk, context)
            )
        ]
        if not core_chunks:
            continue
        gaps.append(
            {
                "source_id": source_id,
                "reason": "core_metric_text_attempted_but_no_records_extracted",
                "budget": dict(row),
                "sample_chunks": _sample_chunks_for_source(
                    evidence_chunks, source_id
                ),
                "core_metric_chunk_count": len(core_chunks),
            }
        )
    return gaps


def _chunk_has_strong_record_signal(chunk: dict) -> bool:
    from ..source_assertions import count_mentions, contextual_number_role
    from ..evidence_qualification import evidence_qualification_enabled

    universal = evidence_qualification_enabled()
    if str(chunk.get("case_span_quote") or "").strip():
        return True
    text = re.sub(
        r"\s+",
        " ",
        str(chunk.get("row_quote") or chunk.get("text") or ""),
    ).strip()
    if not text or _chunk_information_priority(chunk) >= 9:
        return False
    lowered = text.lower()
    return bool(
        _OFFICIAL_CASE_LABEL_RE.search(text)
        or re.search(r"\bpatient\s+(?:no\.?\s*)?\d+\b", lowered)
        or re.search(
            r"\b(?:adult|\d{1,3}-year-old)\b.{0,120}\b(?:male|female|patient|"
            r"passenger|crew)\b",
            lowered,
        )
        or any(
            not universal or not contextual_number_role(lowered, match.start("number"), match.end("number"))
            for match in re.finditer(
                r"\b(?P<number>\d[\d,]*(?:\.\d+)?)\s+(?:confirmed\s+|probable\s+|"
                r"suspected\s+)?(?:cases?|deaths?|patients?|hospitali[sz]ations?)\b",
                lowered,
            )
        )
        or (universal and count_mentions(text))
        or (
            not universal
            and re.search(r"\b(?:PCR|RT-PCR|confirmed|died|hospitali[sz]ed|ICU)\b", text, re.IGNORECASE)
            and re.search(r"\b(?:hantavirus|Andes virus|ANDV|HPS|Hondius)\b", text, re.IGNORECASE)
        )
    )


def _classify_llm_empty_output_reason(chunk: dict) -> str:
    if _chunk_information_priority(chunk) >= 9:
        return "navigation_or_boilerplate"
    if _chunk_has_strong_record_signal(chunk):
        return "llm_empty_strong_signal"
    if _is_context_only_chunk(chunk, None):
        return "legitimate_no_record_context"
    return "case_span_not_detected"


def _focused_recovery_queue(
    candidates: list[dict],
    extraction_budget_by_source: dict[str, dict],
    context: dict | None,
) -> list[dict]:
    max_sources = _parse_positive_int_env(
        "LLM_FOCUSED_RECOVERY_MAX_SOURCES",
        30,
    )
    spans_per_source = _parse_positive_int_env(
        "LLM_FOCUSED_RECOVERY_SPANS_PER_SOURCE",
        3,
    )
    grouped: dict[str, list[dict]] = {}
    first_index: dict[str, int] = {}
    for index, chunk in enumerate(candidates):
        source_id = str(chunk.get("source_id") or f"unknown_{index}")
        explicit_span = bool(chunk.get("case_span_id") or chunk.get("case_span_quote"))
        if _chunk_priority(chunk, context) > 3 and not explicit_span:
            continue
        if not _chunk_has_strong_record_signal(chunk):
            continue
        grouped.setdefault(source_id, []).append(chunk)
        first_index.setdefault(source_id, index)
    source_order = sorted(
        grouped,
        key=lambda source_id: (
            min(_chunk_priority(chunk, context) for chunk in grouped[source_id]),
            first_index[source_id],
            source_id,
        ),
    )[:max_sources]
    queued: list[dict] = []
    for source_id in source_order:
        rows = sorted(
            grouped[source_id],
            key=lambda chunk: (
                _chunk_information_priority(chunk),
                chunk.get("chunk_index") or 0,
                str(chunk.get("chunk_id") or ""),
            ),
        )
        queued.extend(rows[:spans_per_source])
    return queued


def _llm_extract_records_from_chunks(
    evidence_chunks: list[dict],
    llm_policy: LLMStructuredExtractionPolicy,
    deterministic_policy: StructuredExtractionPolicy,
    fallback_to_rule_based: bool,
    context: dict | None = None,
) -> tuple[list[PublicHealthRecord], dict]:
    """LLM extraction loop with optional per-chunk deterministic fallback."""

    settings = llm_clients.get_llm_settings()
    records: list[PublicHealthRecord] = []
    observation_keys = set()
    observation_chunks = {str(c.get('chunk_id') or ''): c for c in evidence_chunks}

    def _append_observations(incoming, chunk=None):
        """One immutable observation per source position and complete facts."""
        if get_env('PIPELINE_MODE') != 'evidence':
            records.extend(incoming)
            return incoming
        from ..record_identity import record_identity_payload
        added = []
        for record in incoming:
            row = record.model_dump()
            original_chunk = observation_chunks.get(str(row.get('supporting_chunk_id') or '')) or chunk or {}
            payload = record_identity_payload(row, original_chunk)
            if not payload.get('evidence_quote') and not any((payload.get('field_positions') or {}).values()):
                # Equal numbers without bound evidence do not prove identity.
                records.append(record)
                added.append(record)
                continue
            key = json.dumps(payload,
                             sort_keys=True, ensure_ascii=True, separators=(',', ':'))
            if key in observation_keys:
                continue
            observation_keys.add(key)
            records.append(record)
            added.append(record)
        return added

    chunk_index_by_source: dict[str, int] = {}
    llm_eligible_chunk_count = 0
    llm_call_count = 0
    llm_success_count = 0
    llm_empty_output_count = 0
    llm_empty_output_reasons: Counter = Counter()
    llm_empty_output_diagnostics: list[dict] = []
    focused_recovery_candidates: list[dict] = []
    primary_empty_inputs: dict[str, dict] = {}
    focused_recovery_call_count = 0
    focused_retry_succeeded_count = 0
    focused_retry_empty_count = 0
    deterministic_recovery_succeeded_count = 0
    focused_recovery_valid_record_count = 0
    focused_recovery_field_gain_count = 0
    focused_recovery_status_by_source: dict[str, str] = {}
    llm_error_count = 0
    llm_fallback_count = 0
    rule_based_fallback_record_count = 0
    llm_error_messages: list[str] = []
    target_data_count = 0
    max_chunks = _parse_llm_max_chunks()
    budget_ledger = ExtractionBudgetLedger.from_environment()
    llm_result_cache: dict[str, tuple[object | None, Exception | None]] = {}
    scheduler_model_call_count = 0
    scheduler_cache_hit_count = 0
    llm_model_call_credits: Counter = Counter()
    pending_prefetch_errors: dict[str, Exception] = {}
    primary_reserved_keys: set[str] = set()
    provider_deferred_ids: set[str] = set()

    def _call_llm(
        chunk_for_llm: dict,
        prefetch_candidates: list[dict] | None = None,
    ):
        nonlocal scheduler_model_call_count, scheduler_cache_hit_count
        current_key = _llm_semantic_span_hash(chunk_for_llm, llm_policy)
        pending_error = pending_prefetch_errors.pop(current_key, None)
        if pending_error is not None:
            raise pending_error
        call_batch = [chunk_for_llm]
        batch_keys = {current_key}
        for candidate in prefetch_candidates or []:
            if len(call_batch) >= budget_ledger.max_concurrency:
                break
            candidate_key = _llm_semantic_span_hash(candidate, llm_policy)
            if candidate_key in batch_keys or candidate_key in llm_result_cache:
                continue
            call_batch.append(candidate)
            batch_keys.add(candidate_key)
        results, diagnostics = _run_bounded_llm_calls(
            call_batch,
            policy=llm_policy,
            max_concurrency=budget_ledger.max_concurrency,
            result_cache=llm_result_cache,
        )
        scheduler_model_call_count += diagnostics["model_call_count"]
        scheduler_cache_hit_count += diagnostics["cache_hit_count"]
        for prefetched_result in results:
            from ..provider_failures import ProviderAccountLimit
            blocked = prefetched_result['error']
            key = prefetched_result['semantic_span_hash']
            if isinstance(blocked, ProviderAccountLimit) and not blocked.dispatched and key in primary_reserved_keys:
                budget_ledger.cancel_undispatched_reservation(prefetched_result['chunk'], context)
                primary_reserved_keys.remove(key)
            if prefetched_result["model_call"]:
                llm_model_call_credits[prefetched_result["semantic_span_hash"]] += 1
        for prefetched_result in results[1:]:
            if prefetched_result["error"] is not None:
                pending_prefetch_errors[prefetched_result["semantic_span_hash"]] = (
                    prefetched_result["error"]
                )
        result = results[0]
        if result["error"] is not None:
            raise result["error"]
        return result["output"]

    def _actual_model_call_credit(chunk_for_llm: dict) -> int:
        semantic_key = _llm_semantic_span_hash(chunk_for_llm, llm_policy)
        return int(llm_model_call_credits.pop(semantic_key, 0))

    def _provider_deferred(chunk_for_llm: dict, budget_row: dict, *, local_fallback=False) -> bool:
        nonlocal rule_based_fallback_record_count
        from ..session_runtime import get_runtime, ProviderAccountLimit
        runtime = get_runtime()
        if runtime is None:
            return False
        stopped = runtime.ledger.provider_status(settings.get('provider'))
        if not stopped or stopped.get('status') != 'halted':
            return False
        key = _llm_semantic_span_hash(chunk_for_llm, llm_policy)
        pending = pending_prefetch_errors.get(key)
        if key in llm_result_cache or pending is not None and getattr(pending, 'dispatched', True):
            return False  # consume already-dispatched results/errors once
        try:
            _call_llm(chunk_for_llm)  # shared ledger permits completed cache, never a new request
            return False
        except ProviderAccountLimit as exc:
            if exc.dispatched:
                raise
        provider_deferred_ids.update(str(row.get('chunk_id')) for row in chunk_for_llm.get('metric_rows') or [] if row.get('chunk_id'))
        if not chunk_for_llm.get('metric_rows'):
            provider_deferred_ids.add(str(chunk_for_llm.get('parent_chunk_id') or chunk_for_llm.get('chunk_id')))
        budget_row['provider_status'] = 'provider_deferred'
        budget_row['last_budget_skip_reason'] = 'provider_account_limit'
        if local_fallback and fallback_to_rule_based:
            source_id = str(chunk_for_llm.get('source_id') or '')
            recovered, _ = _rule_based_extract_records_from_chunks(
                [chunk_for_llm], deterministic_policy,
                start_index=chunk_index_by_source.get(source_id, 0)+1, context=context)
            recovered = _append_observations(recovered, chunk_for_llm)
            rule_based_fallback_record_count += len(recovered)
            budget_row['record_count'] += len(recovered)
            chunk_index_by_source[source_id] = chunk_index_by_source.get(source_id, 0)+len(recovered)
        return True

    primary_llm_call_count = 0
    llm_skipped_due_to_chunk_cap_count = 0
    skipped_context_only_chunk_count = 0
    skipped_context_only_source_ids: set[str] = set()
    skipped_disease_mismatch_chunk_count = 0
    skipped_disease_mismatch_source_ids: set[str] = set()
    must_fetch_disease_gate_bypass_count = 0
    extraction_budget_by_source: dict[str, dict] = {}
    official_extraction_queue: list[dict] = []
    role_policy = load_source_role_policy()
    ordered_chunks = _ordered_llm_chunks(evidence_chunks, context)
    processed_metric_row_chunk_ids: set[str] = set()
    batched_metric_row_call_count = 0
    metric_row_batch_record_count = 0
    metric_row_record_count_by_source: dict[str, int] = {}
    deterministic_metric_row_record_count = 0
    fallback_text_chunk_call_count = 0
    skipped_text_fallback_after_row_record_count = 0
    target_source_llm_call_count = 0
    skipped_context_extraction_count = 0
    rolling_low_yield_stop_triggered = False
    no_task_collection_document = False
    extraction_blocking_reason: str | None = None
    has_target_source_chunks = any(
        _chunk_priority(chunk, context) <= 1
        for chunk in ordered_chunks
        if isinstance(chunk, dict)
    )
    if _direct_should_skip_non_target_extraction(ordered_chunks, context):
        for chunk in ordered_chunks:
            if not isinstance(chunk, dict):
                continue
            budget_row = _source_budget_row(extraction_budget_by_source, chunk, context)
            budget_row["queued_count"] += 1
            budget_row["skipped_context_only_count"] += 1
        skipped_context_extraction_count += sum(
            1 for chunk in ordered_chunks if isinstance(chunk, dict)
        )
        no_task_collection_document = True
        extraction_blocking_reason = "no_task_collection_document"
        ordered_chunks = []
        has_target_source_chunks = False
    target_source_attempted = False
    official_outbreak_deterministic_record_count = 0
    official_outbreak_deterministic_diagnostics: list[dict] = []
    official_outbreak_processed_chunk_ids: set[str] = set()
    official_outbreak_records, official_outbreak_diagnostics = (
        extract_official_outbreak_records_from_chunks(
            ordered_chunks,
            policy=deterministic_policy,
            context=context,
        )
    )
    if official_outbreak_diagnostics:
        official_outbreak_deterministic_diagnostics = official_outbreak_diagnostics
        chunk_by_id = {
            str(chunk.get("chunk_id") or ""): chunk
            for chunk in ordered_chunks
            if isinstance(chunk, dict)
        }
        official_outbreak_records = _append_observations(official_outbreak_records)
        official_outbreak_deterministic_record_count = len(official_outbreak_records)
        record_count_by_chunk_id = Counter(
            str(record.supporting_chunk_id or "")
            for record in official_outbreak_records
            if record.supporting_chunk_id
        )
        for diagnostic in official_outbreak_diagnostics:
            chunk_id = str(diagnostic.get("chunk_id") or "")
            chunk = chunk_by_id.get(chunk_id)
            if not chunk:
                continue
            source_id = str(chunk.get("source_id") or "")
            budget_row = _source_budget_row(extraction_budget_by_source, chunk, context)
            budget_row["queued_count"] += 1
            budget_row["eligible_count"] += 1
            budget_row["attempted_count"] += 1
            record_count = int(record_count_by_chunk_id.get(chunk_id, 0))
            budget_row["record_count"] += record_count
            if record_count > 0:
                from ..evidence_qualification import evidence_qualification_enabled
                # A heuristic candidate is not proof that the whole span was
                # extracted. EVIDENCE still gives the model its bounded opportunity.
                if not evidence_qualification_enabled():
                    official_outbreak_processed_chunk_ids.add(chunk_id)
                chunk_index_by_source[source_id] = (
                    chunk_index_by_source.get(source_id, 0) + record_count
                )
                if _chunk_priority(chunk, context) <= 1:
                    target_source_attempted = True
            if diagnostic.get("relevant_chunk_count"):
                target_data_count += 1

    ordered_chunks = _ordered_llm_chunks(
        _expand_case_span_chunks_for_llm(
            ordered_chunks,
            exclude_parent_chunk_ids=official_outbreak_processed_chunk_ids,
        ),
        context,
    )

    # Metric tables can contain hundreds of rows from one source. Reserve the
    # soft-budget coverage floor for explicit case/patient spans before letting
    # metric batches consume primary calls.
    protected_case_source_ids = {
        str(chunk.get("source_id") or "")
        for chunk in ordered_chunks
        if isinstance(chunk, dict)
        and not _chunk_is_metric_row(chunk)
        and _chunk_has_strong_record_signal(chunk)
        and _chunk_information_priority(chunk) <= 3
        and not _is_context_only_chunk(chunk, role_policy)
    }
    protected_case_source_ids.discard("")
    metric_share_limit = max(1, int(budget_ledger.soft_primary_calls * 0.25))
    if protected_case_source_ids:
        reserved_case_calls = min(
            len(protected_case_source_ids), budget_ledger.soft_primary_calls
        )
        metric_primary_call_limit = min(
            metric_share_limit,
            max(0, budget_ledger.soft_primary_calls - reserved_case_calls),
        )
    else:
        metric_primary_call_limit = metric_share_limit
    metric_row_batch_deferred_for_case_coverage_count = 0

    metric_row_groups: dict[tuple[str, ...], list[dict]] = {}
    metric_row_group_order: list[tuple[str, ...]] = []
    if _direct_collection_enabled(context):
        for chunk in ordered_chunks:
            if not isinstance(chunk, dict) or not _chunk_is_metric_row(chunk):
                continue
            chunk_id = str(chunk.get("chunk_id") or "")
            if chunk_id and chunk_id in official_outbreak_processed_chunk_ids:
                processed_metric_row_chunk_ids.add(chunk_id)
                continue
            source_id = str(chunk.get("source_id") or "")
            budget_row = _source_budget_row(extraction_budget_by_source, chunk, context)
            budget_row["queued_count"] += 1
            if _is_context_only_chunk(chunk, role_policy) and (
                _direct_collection_enabled(context)
                or _chunk_has_explicit_context_only_role(chunk, role_policy)
            ):
                skipped_context_only_chunk_count += 1
                budget_row["skipped_context_only_count"] += 1
                if source_id:
                    skipped_context_only_source_ids.add(source_id)
                processed_metric_row_chunk_ids.add(str(chunk.get("chunk_id") or ""))
                continue
            disease_blocked, _ = _chunk_blocked_by_disease_gate(chunk, context)
            if disease_blocked:
                skipped_disease_mismatch_chunk_count += 1
                budget_row["skipped_disease_mismatch_count"] += 1
                if source_id:
                    skipped_disease_mismatch_source_ids.add(source_id)
                processed_metric_row_chunk_ids.add(str(chunk.get("chunk_id") or ""))
                continue
            if not _direct_llm_chunk_allowed(chunk, llm_policy, context):
                continue
            llm_eligible_chunk_count += 1
            budget_row["eligible_count"] += 1
            deterministic_records = _deterministic_metric_row_records(
                chunk,
                start_index=chunk_index_by_source.get(source_id, 0) + 1,
                llm_policy=llm_policy,
                settings=settings,
                context=context,
            )
            if isinstance(deterministic_records, tuple):
                candidate_records = deterministic_records[0]
                deterministic_records = (
                    candidate_records if isinstance(candidate_records, list) else []
                )
            if deterministic_records:
                deterministic_records = _append_observations(deterministic_records, chunk)
                metric_row_record_count_by_source[source_id] = (
                    metric_row_record_count_by_source.get(source_id, 0)
                    + len(deterministic_records)
                )
                chunk_index_by_source[source_id] = (
                    chunk_index_by_source.get(source_id, 0)
                    + len(deterministic_records)
                )
                deterministic_metric_row_record_count += len(deterministic_records)
                metric_row_batch_record_count += len(deterministic_records)
                budget_row["attempted_count"] += 1
                budget_row["record_count"] += len(deterministic_records)
                processed_metric_row_chunk_ids.add(str(chunk.get("chunk_id") or ""))
                if _chunk_priority(chunk, context) <= 1:
                    target_source_attempted = True
                emit_workflow_progress(
                    "structured_extraction",
                    "Deterministic metric-row split completed",
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "source_id": source_id,
                        "record_count": len(deterministic_records),
                    },
                    status="completed",
                )
                continue
            key = _metric_row_key(chunk)
            if key not in metric_row_groups:
                metric_row_groups[key] = []
                metric_row_group_order.append(key)
            metric_row_groups[key].append(chunk)

    metric_row_batch_size = max(1, _metric_row_batch_size())
    for key in metric_row_group_order:
        rows = metric_row_groups.get(key) or []
        for offset in range(0, len(rows), metric_row_batch_size):
            batch_rows = rows[offset : offset + metric_row_batch_size]
            if not batch_rows:
                continue
            batch_budget_row = _source_budget_row(
                extraction_budget_by_source, batch_rows[0], context
            )
            if batched_metric_row_call_count >= metric_primary_call_limit:
                metric_row_batch_deferred_for_case_coverage_count += len(batch_rows)
                llm_skipped_due_to_chunk_cap_count += len(batch_rows)
                for row in batch_rows:
                    budget_row = _source_budget_row(
                        extraction_budget_by_source, row, context
                    )
                    budget_row["skipped_due_to_cap_count"] += 1
                    budget_row["last_budget_skip_reason"] = (
                        "metric_batch_deferred_for_case_coverage"
                    )
                    processed_metric_row_chunk_ids.add(str(row.get("chunk_id") or ""))
                    if _chunk_has_strong_record_signal(row):
                        focused_candidate = dict(row)
                        focused_candidate["recovery_reason"] = (
                            "metric_batch_deferred_for_case_coverage"
                        )
                        focused_recovery_candidates.append(focused_candidate)
                continue
            batch_chunk = _make_metric_row_batch_chunk(
                batch_rows,
                batch_index=(offset // metric_row_batch_size) + 1,
            )
            chunk_for_llm = _chunk_with_task_context(batch_chunk, context)
            batch_key = _llm_semantic_span_hash(chunk_for_llm, llm_policy)
            if _provider_deferred(chunk_for_llm, batch_budget_row, local_fallback=True):
                processed_metric_row_chunk_ids.update(str(row.get("chunk_id") or "") for row in batch_rows)
                continue
            batch_model_call = batch_key not in llm_result_cache
            can_attempt, skip_reason = budget_ledger.can_attempt_primary(
                batch_rows[0],
                batch_budget_row,
                context,
                actual_model_call=batch_model_call,
            )
            if not can_attempt:
                llm_skipped_due_to_chunk_cap_count += len(batch_rows)
                for row in batch_rows:
                    budget_row = _source_budget_row(
                        extraction_budget_by_source, row, context
                    )
                    budget_row["skipped_due_to_cap_count"] += 1
                    budget_row["last_budget_skip_reason"] = skip_reason
                    processed_metric_row_chunk_ids.add(str(row.get("chunk_id") or ""))
                continue
            target_source_attempted_before_call = target_source_attempted
            llm_call_count += 1
            primary_llm_call_count += 1
            budget_ledger.register_primary_call(
                batch_rows[0],
                context,
                actual_model_call=batch_model_call,
            )
            batched_metric_row_call_count += 1
            source_id = str(batch_chunk.get("source_id") or "")
            if _chunk_priority(batch_chunk, context) <= 1:
                target_source_attempted = True
                target_source_llm_call_count += 1
            for row in batch_rows:
                budget_row = _source_budget_row(
                    extraction_budget_by_source, row, context
                )
                budget_row["attempted_count"] += 1
                processed_metric_row_chunk_ids.add(str(row.get("chunk_id") or ""))
            emit_workflow_progress(
                "structured_extraction",
                "LLM metric-row batch extraction started",
                {
                    "chunk_id": batch_chunk.get("chunk_id"),
                    "source_id": source_id,
                    "row_count": len(batch_rows),
                    "llm_call_count": llm_call_count,
                    "max_chunks": max_chunks,
                },
            )
            actual_call_credit = 0
            try:
                output = _call_llm(chunk_for_llm)
                actual_call_credit = _actual_model_call_credit(chunk_for_llm)
                llm_success_count += 1
                if not output.records:
                    llm_empty_output_count += 1
                    reason = _classify_llm_empty_output_reason(batch_chunk)
                    llm_empty_output_reasons[reason] += 1
                    llm_empty_output_diagnostics.append(
                        {
                            "source_id": source_id,
                            "chunk_id": batch_chunk.get("chunk_id"),
                            "reason": reason,
                            "stage": "primary_metric_row_batch",
                        }
                    )
                built_metric_record_count = 0
                record_limit = max(llm_policy.max_records_per_chunk, len(batch_rows))
                for llm_record in output.records[:record_limit]:
                    chunk_index_by_source[source_id] = chunk_index_by_source.get(source_id, 0) + 1
                    idx = chunk_index_by_source[source_id]
                    record = _build_record_from_llm_output(
                        llm_record, chunk_for_llm, idx, llm_policy, settings, context
                    )
                    if record is None:
                        chunk_index_by_source[source_id] -= 1
                        continue
                    if not _append_observations([record], chunk_for_llm):
                        chunk_index_by_source[source_id] -= 1
                        continue
                    built_metric_record_count += 1
                    metric_row_batch_record_count += 1
                    metric_row_record_count_by_source[source_id] = (
                        metric_row_record_count_by_source.get(source_id, 0) + 1
                    )
                    for row in batch_rows:
                        if row.get("row_id") == record.source_row_id:
                            budget_row = _source_budget_row(
                                extraction_budget_by_source, row, context
                            )
                            budget_row["record_count"] += 1
                            break
                budget_ledger.register_primary_result(
                    built_metric_record_count,
                    actual_model_calls=actual_call_credit,
                )
                emit_workflow_progress(
                    "structured_extraction",
                    "LLM metric-row batch extraction completed",
                    {
                        "chunk_id": batch_chunk.get("chunk_id"),
                        "source_id": source_id,
                        "record_count": len(output.records),
                        "llm_success_count": llm_success_count,
                    },
                    status="completed",
                )
            except Exception as exc:  # noqa: BLE001
                from ..provider_failures import ProviderAccountLimit
                if isinstance(exc, ProviderAccountLimit) and not exc.dispatched:
                    budget_ledger.cancel_undispatched_reservation(batch_rows[0], context)
                    llm_call_count -= 1
                    primary_llm_call_count -= 1
                    batched_metric_row_call_count -= 1
                    if _chunk_priority(batch_chunk, context) <= 1:
                        target_source_llm_call_count -= 1
                    target_source_attempted = target_source_attempted_before_call
                    for row in batch_rows:
                        _source_budget_row(extraction_budget_by_source, row, context)['attempted_count'] -= 1
                    _provider_deferred(chunk_for_llm, batch_budget_row, local_fallback=True)
                    continue
                if not actual_call_credit:
                    actual_call_credit = _actual_model_call_credit(chunk_for_llm)
                budget_ledger.register_primary_result(
                    0,
                    actual_model_calls=actual_call_credit,
                )
                llm_error_count += 1
                llm_error_messages.append(f"{type(exc).__name__}: {exc}")
                emit_workflow_progress(
                    "structured_extraction",
                    "LLM metric-row batch extraction failed",
                    {
                        "chunk_id": batch_chunk.get("chunk_id"),
                        "source_id": source_id,
                        "error_type": exc.__class__.__name__,
                    },
                    status="error",
                )

    def _metric_primary_skip_reason(chunk: dict, budget_row: dict) -> str | None:
        from ..evidence_qualification import evidence_qualification_enabled

        # EVIDENCE continues collecting independent facts, periods and fields after
        # metric rows; a raw row count is not proof of qualified coverage.
        if evidence_qualification_enabled():
            return None
        # Prefetch and result consumption must agree on this legacy direct-mode
        # filter; a discarded prefetched result cannot advance a checkpoint.
        if not _direct_collection_enabled(context) or (
            metric_row_batch_record_count < _direct_min_target_metric_records()
        ):
            return None
        if not _direct_text_fallback_after_row_extraction_enabled() and (
            metric_row_record_count_by_source.get(str(chunk.get("source_id") or ""), 0)
            >= _direct_min_target_metric_records()
        ):
            return "source_metric_records_sufficient"
        if str(budget_row.get("budget_bucket")) != "verified_target_collection":
            return "context_after_metric_rows"
        return None

    primary_prefetch_chunks: list[dict] = []
    for candidate in ordered_chunks:
        if not isinstance(candidate, dict):
            continue
        candidate_chunk_id = str(candidate.get("chunk_id") or "")
        if (
            candidate_chunk_id in processed_metric_row_chunk_ids
            or candidate_chunk_id in official_outbreak_processed_chunk_ids
            or _is_context_only_chunk(candidate, role_policy)
        ):
            continue
        candidate_disease_blocked, _ = _chunk_blocked_by_disease_gate(
            candidate, context
        )
        if candidate_disease_blocked or not _direct_llm_chunk_allowed(
            candidate, llm_policy, context
        ):
            continue
        candidate_budget_row = _source_budget_row(
            extraction_budget_by_source, candidate, context,
        )
        if _metric_primary_skip_reason(candidate, candidate_budget_row):
            continue
        primary_prefetch_chunks.append(
            _chunk_with_task_context(candidate, context)
        )
    primary_prefetch_index = {
        _llm_semantic_span_hash(chunk, llm_policy): index
        for index, chunk in enumerate(primary_prefetch_chunks)
    }
    def _reserve_primary_prefetch(current_key: str) -> list[dict]:
        prefetch_position = primary_prefetch_index.get(current_key, -1)
        if prefetch_position < 0:
            return []
        capacity = max(0, budget_ledger.max_concurrency - 1)
        if budget_ledger.scheduler_mode == "quality_adaptive":
            # Keep rolling quality history intact. Reservations include responses
            # not yet applied, so no work crosses an unevaluated checkpoint.
            window = max(1, budget_ledger.rolling_yield_window)
            next_checkpoint = (
                budget_ledger.primary_result_call_count // window + 1
            ) * window
            calls_until_checkpoint = max(
                0, next_checkpoint - budget_ledger.primary_model_call_count,
            )
            capacity = min(capacity, calls_until_checkpoint)
        if capacity <= 0:
            return []

        reserved: list[dict] = []
        batch_keys = {current_key}
        for candidate in primary_prefetch_chunks[prefetch_position + 1 :]:
            if len(reserved) >= capacity:
                break
            candidate_key = _llm_semantic_span_hash(candidate, llm_policy)
            if (
                candidate_key in batch_keys
                or candidate_key in primary_reserved_keys
                or candidate_key in llm_result_cache
            ):
                continue
            candidate_budget_row = _source_budget_row(
                extraction_budget_by_source,
                candidate,
                context,
            )
            if budget_ledger.should_stop_low_yield(
                candidate, candidate_budget_row, context, record_skip=False,
            ):
                continue
            can_reserve, _ = budget_ledger.can_attempt_primary(
                candidate,
                candidate_budget_row,
                context,
                actual_model_call=True,
                record_skip=False,
            )
            if not can_reserve:
                continue
            budget_ledger.register_primary_call(
                candidate,
                context,
                actual_model_call=True,
            )
            primary_reserved_keys.add(candidate_key)
            batch_keys.add(candidate_key)
            reserved.append(candidate)
        return reserved

    for queue_index, chunk in enumerate(ordered_chunks, start=1):
        chunk_id = str(chunk.get("chunk_id") or "")
        if chunk_id and chunk_id in processed_metric_row_chunk_ids:
            continue
        if chunk_id and chunk_id in official_outbreak_processed_chunk_ids:
            continue
        source_id = str(chunk.get("source_id") or "")
        budget_row = _source_budget_row(extraction_budget_by_source, chunk, context)
        budget_row["queued_count"] += 1
        priority = _chunk_priority(chunk, context)
        if priority <= 1:
            official_extraction_queue.append(
                {
                    "queue_index": queue_index,
                    "source_id": source_id,
                    "official_report_key": budget_row.get("official_report_key"),
                    "chunk_id": chunk.get("chunk_id"),
                    "priority": "must_fetch" if priority == 0 else "official_or_high_trust",
                    "must_fetch": _chunk_is_must_fetch(chunk, context),
                    "official_or_high_trust": _chunk_is_official_or_high_trust(chunk, context),
                    "chunk_index": chunk.get("chunk_index"),
                    "title": chunk.get("title"),
                    "source_url": chunk.get("source_url"),
                }
            )
        if _is_context_only_chunk(chunk, role_policy) and (
            _direct_collection_enabled(context)
            or _chunk_has_explicit_context_only_role(chunk, role_policy)
        ):
            skipped_context_only_chunk_count += 1
            budget_row["skipped_context_only_count"] += 1
            if chunk.get("source_id"):
                skipped_context_only_source_ids.add(chunk.get("source_id"))
            continue
        disease_blocked, disease_assessment = _chunk_blocked_by_disease_gate(
            chunk, context
        )
        if disease_blocked:
            skipped_disease_mismatch_chunk_count += 1
            budget_row["skipped_disease_mismatch_count"] += 1
            if chunk.get("source_id"):
                skipped_disease_mismatch_source_ids.add(chunk.get("source_id"))
            continue
        if chunk.get("contains_target_data"):
            target_data_count += 1
        if not _direct_llm_chunk_allowed(chunk, llm_policy, context):
            continue
        metric_skip_reason = _metric_primary_skip_reason(chunk, budget_row)
        if metric_skip_reason == "source_metric_records_sufficient":
            skipped_text_fallback_after_row_record_count += 1
            budget_row["skipped_text_fallback_after_row_record_count"] = (
                budget_row.get("skipped_text_fallback_after_row_record_count", 0) + 1
            )
            continue
        if metric_skip_reason == "context_after_metric_rows":
            skipped_context_extraction_count += 1
            budget_row["skipped_context_only_count"] += 1
            continue
        llm_eligible_chunk_count += 1
        budget_row["eligible_count"] += 1
        chunk_for_llm = _chunk_with_task_context(chunk, context)
        current_key = _llm_semantic_span_hash(chunk_for_llm, llm_policy)
        if _provider_deferred(chunk_for_llm, budget_row, local_fallback=True):
            continue
        was_prefetch_reserved = current_key in primary_reserved_keys
        if not was_prefetch_reserved and budget_ledger.should_stop_low_yield(
            chunk,
            budget_row,
            context,
        ):
            rolling_low_yield_stop_triggered = True
            budget_row["last_budget_skip_reason"] = "rolling_low_yield_stop"
            continue
        # Enforce the optional per-run LLM chunk cap. Once the cap has been
        # reached, additional eligible chunks are counted but skipped without
        # incurring further LLM calls.
        current_model_call = current_key not in llm_result_cache
        if not was_prefetch_reserved:
            can_attempt, skip_reason = budget_ledger.can_attempt_primary(
                chunk,
                budget_row,
                context,
                actual_model_call=current_model_call,
            )
            if not can_attempt:
                llm_skipped_due_to_chunk_cap_count += 1
                budget_row["skipped_due_to_cap_count"] += 1
                budget_row["last_budget_skip_reason"] = skip_reason
                if _chunk_has_strong_record_signal(chunk):
                    recovery_chunk = dict(chunk)
                    recovery_chunk["recovery_reason"] = (
                        "high_value_cap_skipped_span"
                        if _chunk_priority(chunk, context) <= 2
                        else "cap_skipped_strong_signal"
                    )
                    focused_recovery_candidates.append(recovery_chunk)
                continue
            budget_ledger.register_primary_call(
                chunk,
                context,
                actual_model_call=current_model_call,
            )
        target_source_attempted_before_call = target_source_attempted
        llm_call_count += 1
        primary_llm_call_count += 1
        if str(chunk.get("chunk_kind") or "") != "metric_row_batch":
            fallback_text_chunk_call_count += 1
        if (
            has_target_source_chunks
            and not target_source_attempted
            and str(budget_row.get("budget_bucket")) != "verified_target_collection"
        ):
            budget_row["attempted_before_target_sources"] = True
        if str(budget_row.get("budget_bucket")) == "verified_target_collection":
            target_source_attempted = True
            target_source_llm_call_count += 1
        budget_row["attempted_count"] += 1
        emit_workflow_progress(
            "structured_extraction",
            "LLM extraction chunk started",
            {
                "chunk_id": chunk.get("chunk_id"),
                "source_id": source_id,
                "llm_call_count": llm_call_count,
                "max_chunks": max_chunks,
            },
        )

        actual_call_credit = 0
        try:
            reserved_prefetch = (
                []
                if current_key in pending_prefetch_errors
                else _reserve_primary_prefetch(current_key)
            )
            output = _call_llm(
                chunk_for_llm,
                reserved_prefetch,
            )
            actual_call_credit = _actual_model_call_credit(chunk_for_llm)
            llm_success_count += 1
            call_valid_record_count = 0
            if not output.records:
                primary_empty_inputs[str(chunk.get("chunk_id") or "")] = dict(chunk_for_llm)
                llm_empty_output_count += 1
                reason = _classify_llm_empty_output_reason(chunk)
                llm_empty_output_reasons[reason] += 1
                llm_empty_output_diagnostics.append(
                    {
                        "source_id": source_id,
                        "chunk_id": chunk.get("chunk_id"),
                        "reason": reason,
                        "stage": "primary_llm_extraction",
                    }
                )
                if reason in {"llm_empty_strong_signal", "case_span_not_detected"}:
                    recovery_chunk = dict(chunk)
                    recovery_chunk["recovery_reason"] = reason
                    focused_recovery_candidates.append(recovery_chunk)
                if (
                    fallback_to_rule_based
                    and str(chunk.get("chunk_kind") or "") != "metric_row_batch"
                    and _core_metric_payloads_from_chunk(chunk, context)
                ):
                    llm_fallback_count += 1
                    start_idx = chunk_index_by_source.get(source_id, 0) + 1
                    fallback_records, _ = _rule_based_extract_records_from_chunks(
                        [chunk],
                        deterministic_policy,
                        start_index=start_idx,
                        context=context,
                    )
                    fallback_records = _append_observations(fallback_records, chunk)
                    rule_based_fallback_record_count += len(fallback_records)
                    budget_row["record_count"] += len(fallback_records)
                    call_valid_record_count += len(fallback_records)
                    chunk_index_by_source[source_id] = (
                        chunk_index_by_source.get(source_id, 0)
                        + len(fallback_records)
                    )
            emit_workflow_progress(
                "structured_extraction",
                "LLM extraction chunk completed",
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source_id": source_id,
                    "record_count": len(output.records),
                    "llm_success_count": llm_success_count,
                },
                status="completed",
            )
            for llm_record in output.records[: llm_policy.max_records_per_chunk]:
                chunk_index_by_source[source_id] = chunk_index_by_source.get(source_id, 0) + 1
                idx = chunk_index_by_source[source_id]
                record = _build_record_from_llm_output(
                    llm_record, chunk_for_llm, idx, llm_policy, settings, context
                )
                if record is None:
                    chunk_index_by_source[source_id] -= 1
                    continue
                if not _append_observations([record], chunk_for_llm):
                    chunk_index_by_source[source_id] -= 1
                    continue
                budget_row["record_count"] += 1
                call_valid_record_count += 1
            budget_ledger.register_primary_result(
                call_valid_record_count,
                actual_model_calls=actual_call_credit,
            )
        except Exception as exc:  # noqa: BLE001 — caller decides fallback
            from ..provider_failures import ProviderAccountLimit
            if isinstance(exc, ProviderAccountLimit) and not exc.dispatched:
                if not was_prefetch_reserved:
                    budget_ledger.cancel_undispatched_reservation(chunk, context)
                llm_call_count -= 1
                primary_llm_call_count -= 1
                budget_row['attempted_count'] -= 1
                if str(chunk.get('chunk_kind') or '') != 'metric_row_batch':
                    fallback_text_chunk_call_count -= 1
                if str(budget_row.get('budget_bucket')) == 'verified_target_collection':
                    target_source_llm_call_count -= 1
                target_source_attempted = target_source_attempted_before_call
                _provider_deferred(chunk_for_llm, budget_row, local_fallback=True)
                emit_workflow_progress('structured_extraction', 'LLM extraction deferred: provider account limit',
                    {'chunk_id':chunk.get('chunk_id'),'source_id':source_id,'reason':'provider_account_limit'}, status='deferred')
                continue
            llm_error_count += 1
            llm_error_messages.append(f"{type(exc).__name__}: {exc}")
            if fallback_to_rule_based:
                fallback_valid_count = 0
                llm_fallback_count += 1
                start_idx = chunk_index_by_source.get(source_id, 0) + 1
                fallback_records, _ = _rule_based_extract_records_from_chunks(
                    [chunk],
                    deterministic_policy,
                    start_index=start_idx,
                    context=context,
                )
                fallback_records = _append_observations(fallback_records, chunk)
                rule_based_fallback_record_count += len(fallback_records)
                fallback_valid_count = len(fallback_records)
                budget_row["record_count"] += len(fallback_records)
                chunk_index_by_source[source_id] = (
                    chunk_index_by_source.get(source_id, 0) + len(fallback_records)
                )
            else:
                fallback_valid_count = 0
            if not actual_call_credit:
                actual_call_credit = _actual_model_call_credit(chunk_for_llm)
            budget_ledger.register_primary_result(
                fallback_valid_count,
                actual_model_calls=actual_call_credit,
            )
            emit_workflow_progress(
                "structured_extraction",
                "LLM extraction chunk failed",
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source_id": source_id,
                    "error_type": exc.__class__.__name__,
                    "fallback_to_rule_based": fallback_to_rule_based,
                },
                status="error",
            )

    focused_queue = _focused_recovery_queue(
        focused_recovery_candidates,
        extraction_budget_by_source,
        context,
    )
    focused_max_calls = min(
        budget_ledger.focused_recovery_reserved_calls,
        _parse_positive_int_env(
        "LLM_FOCUSED_RECOVERY_MAX_CALLS",
        budget_ledger.focused_recovery_reserved_calls,
        ),
    )

    def _prepared_focused_llm_chunk(chunk: dict, recovery_index: int) -> dict | None:
        focused_chunk = dict(chunk)
        focused_quote = re.sub(
            r"\s+",
            " ",
            str(chunk.get("case_span_quote") or chunk.get("text") or ""),
        ).strip()
        if get_env("PIPELINE_MODE") == "evidence":
            previous_input = primary_empty_inputs.get(str(chunk.get("chunk_id") or ""))
            skip_reason = None
            if previous_input is not None:
                previous_text = str(previous_input.get("text") or "")
                if focused_quote == re.sub(r"\s+", " ", previous_text).strip():
                    skip_reason = "focused_retry_skipped_unchanged_empty"
                else:
                    # A shorter prompt must point back to the unchanged source,
                    # not invent positions in its whitespace-normalized copy.
                    original_chunk = observation_chunks.get(str(
                        chunk.get("parent_chunk_id") or chunk.get("chunk_id") or ""
                    )) or chunk
                    original_text = str(original_chunk.get("text") or "")
                    pattern = r"\s+".join(re.escape(word) for word in focused_quote.split())
                    matches = list(re.finditer(pattern, original_text)) if pattern else []
                    if not pattern or not re.search(pattern, previous_text):
                        matches = []
                    if previous_input.get("parent_chunk_id"):
                        previous_start = previous_input.get("case_span_start")
                        previous_end = previous_input.get("case_span_end")
                        if isinstance(previous_start, int) and isinstance(previous_end, int):
                            matches = [m for m in matches if previous_start <= m.start() < m.end() <= previous_end]
                        else:
                            matches = []
                    declared = (chunk.get("case_span_start"), chunk.get("case_span_end"))
                    located = next((m for m in matches if m.span() == declared), None)
                    if located is None and len(matches) == 1:
                        located = matches[0]
                    if located is None:
                        skip_reason = "focused_retry_skipped_unlocated_span"
                    else:
                        focused_chunk["case_span_start"], focused_chunk["case_span_end"] = located.span()
                        focused_chunk["case_span_quote"] = located.group()
            if skip_reason:
                source_id = str(chunk.get("source_id") or "")
                _source_budget_row(extraction_budget_by_source, chunk, context)["focused_recovery_status"] = skip_reason
                focused_recovery_status_by_source[source_id] = skip_reason
                llm_empty_output_diagnostics.append({
                    "source_id": source_id, "chunk_id": chunk.get("chunk_id"),
                    "reason": skip_reason, "stage": "focused_llm_recovery",
                    "status": "skipped", "input_changed": False,
                })
                return None
        focused_chunk.setdefault(
            "case_span_id",
            f"focused_span_{recovery_index:03d}",
        )
        focused_chunk.setdefault("case_span_quote", focused_quote)
        focused_chunk.setdefault("case_span_start", 0)
        focused_chunk.setdefault("case_span_end", len(focused_quote))
        focused_chunk.setdefault(
            "case_span_extraction_method",
            "focused_llm_local_span",
        )
        focused_chunk["focused_recovery_status"] = "focused_retry_attempted"
        recovery_reason = str(
            chunk.get("recovery_reason") or "uncovered_strong_signal_span"
        )
        focused_chunk["recovery_reason"] = recovery_reason
        focused_chunk["focused_recovery_input_changed"] = True
        focused_chunk["text"] = (
            f"Focused recovery reason: {recovery_reason}. "
            "Extract only claims explicitly supported by this local evidence span.\n"
            f"Local evidence span: {focused_quote}"
        )
        return _chunk_with_task_context(focused_chunk, context)

    for recovery_index, chunk in enumerate(focused_queue, start=1):
        source_id = str(chunk.get("source_id") or "")
        budget_row = _source_budget_row(extraction_budget_by_source, chunk, context)
        start_idx = chunk_index_by_source.get(source_id, 0) + 1
        recovered_records, _ = _rule_based_extract_records_from_chunks(
            [chunk],
            deterministic_policy,
            start_index=start_idx,
            context=context,
        )
        if recovered_records:
            recovered_records = [
                record.model_copy(
                    update={"focused_recovery_status": "deterministic_recovery_succeeded"}
                )
                for record in recovered_records
            ]
            recovered_records = _append_observations(recovered_records, chunk)
            recovered_count = len(recovered_records)
            deterministic_recovery_succeeded_count += recovered_count
            focused_recovery_valid_record_count += recovered_count
            focused_recovery_field_gain_count += sum(
                1
                for record in recovered_records
                for field in _FIELD_DETECTION_KEYS
                if record.model_dump().get(field) is not None
            )
            budget_row["record_count"] += recovered_count
            status = "deterministic_recovery_succeeded" if recovered_count else "deterministic_recovery_no_new_observation"
            budget_row["focused_recovery_status"] = status
            focused_recovery_status_by_source[source_id] = status
            chunk_index_by_source[source_id] = (
                chunk_index_by_source.get(source_id, 0) + recovered_count
            )
            if recovered_count:
                continue

        if focused_recovery_call_count >= focused_max_calls:
            budget_row["focused_recovery_status"] = "focused_retry_budget_exhausted"
            focused_recovery_status_by_source[source_id] = (
                "focused_retry_budget_exhausted"
            )
            continue

        chunk_for_llm = _prepared_focused_llm_chunk(chunk, recovery_index)
        if chunk_for_llm is None:
            continue
        focused_chunk = chunk_for_llm
        focused_key = _llm_semantic_span_hash(chunk_for_llm, llm_policy)
        if _provider_deferred(chunk_for_llm, budget_row):
            budget_row["focused_recovery_status"] = "provider_deferred"
            focused_recovery_status_by_source[source_id] = "provider_deferred"
            continue
        focused_model_call = focused_key not in llm_result_cache
        if not budget_ledger.can_attempt_recovery(
            actual_model_call=focused_model_call
        ):
            budget_row["focused_recovery_status"] = "focused_retry_budget_exhausted"
            focused_recovery_status_by_source[source_id] = (
                "focused_retry_budget_exhausted"
            )
            continue
        recovery_reason = str(focused_chunk.get("recovery_reason") or "")
        focused_recovery_call_count += 1
        llm_call_count += 1
        budget_ledger.register_recovery_call(
            focused_chunk,
            context,
            actual_model_call=focused_model_call,
        )
        budget_row["attempted_count"] += 1
        try:
            output = _call_llm(chunk_for_llm)
            llm_success_count += 1
            if not output.records:
                llm_empty_output_count += 1
                focused_retry_empty_count += 1
                llm_empty_output_reasons["focused_retry_empty"] += 1
                llm_empty_output_diagnostics.append(
                    {
                        "source_id": source_id,
                        "chunk_id": chunk.get("chunk_id"),
                        "reason": "focused_retry_empty",
                        "recovery_reason": recovery_reason,
                        "input_changed": True,
                        "stage": "focused_llm_recovery",
                    }
                )
                budget_row["focused_recovery_status"] = "focused_retry_empty"
                focused_recovery_status_by_source[source_id] = "focused_retry_empty"
                continue

            built_records: list[PublicHealthRecord] = []
            for llm_record in output.records[: llm_policy.max_records_per_chunk]:
                chunk_index_by_source[source_id] = (
                    chunk_index_by_source.get(source_id, 0) + 1
                )
                successful_chunk = dict(chunk_for_llm)
                successful_chunk["focused_recovery_status"] = (
                    "focused_retry_succeeded"
                )
                record = _build_record_from_llm_output(
                    llm_record,
                    successful_chunk,
                    chunk_index_by_source[source_id],
                    llm_policy,
                    settings,
                    context,
                )
                if record is None:
                    chunk_index_by_source[source_id] -= 1
                    continue
                built_records.append(record)
            built_records = _append_observations(built_records, focused_chunk)
            if built_records:
                focused_retry_succeeded_count += 1
                focused_recovery_valid_record_count += len(built_records)
                focused_recovery_field_gain_count += sum(
                    1
                    for record in built_records
                    for field in _FIELD_DETECTION_KEYS
                    if record.model_dump().get(field) is not None
                )
                budget_row["record_count"] += len(built_records)
                budget_row["focused_recovery_status"] = "focused_retry_succeeded"
                focused_recovery_status_by_source[source_id] = (
                    "focused_retry_succeeded"
                )
            else:
                focused_retry_empty_count += 1
                llm_empty_output_reasons["focused_retry_empty"] += 1
                budget_row["focused_recovery_status"] = "focused_retry_empty"
                focused_recovery_status_by_source[source_id] = "focused_retry_empty"
        except Exception as exc:  # noqa: BLE001
            from ..provider_failures import ProviderAccountLimit
            if isinstance(exc, ProviderAccountLimit) and not exc.dispatched:
                budget_ledger.cancel_undispatched_reservation(focused_chunk, context, focused=True)
                focused_recovery_call_count -= 1
                llm_call_count -= 1
                budget_row['attempted_count'] -= 1
                _provider_deferred(chunk_for_llm, budget_row)
                budget_row['focused_recovery_status'] = 'provider_deferred'
                focused_recovery_status_by_source[source_id] = 'provider_deferred'
                continue
            llm_error_count += 1
            llm_error_messages.append(f"{type(exc).__name__}: {exc}")
            budget_row["focused_recovery_status"] = "focused_retry_error"
            focused_recovery_status_by_source[source_id] = "focused_retry_error"

    field_counters: Counter = Counter()
    for record in records:
        rec_dict = record.model_dump()
        for field in _FIELD_DETECTION_KEYS:
            if rec_dict.get(field) is not None:
                field_counters[field] += 1
    successful_official_report_keys = {
        str(row.get("official_report_key"))
        for row in extraction_budget_by_source.values()
        if row.get("official_report_key") and int(row.get("record_count") or 0) > 0
    }
    official_extraction_failures: list[dict] = []
    official_extraction_resolved_by_equivalent_sources: list[dict] = []
    must_fetch_ids = {str(v) for v in (context or {}).get("must_fetch_source_ids") or []}
    for source_id in sorted(must_fetch_ids):
        row = extraction_budget_by_source.get(source_id)
        if not row:
            continue
        if int(row.get("record_count") or 0) > 0:
            continue
        official_report_key = row.get("official_report_key")
        if official_report_key and official_report_key in successful_official_report_keys:
            official_extraction_resolved_by_equivalent_sources.append(
                {
                    "source_id": source_id,
                    "official_report_key": official_report_key,
                    "reason": "resolved_by_equivalent_source",
                    "budget": dict(row),
                }
            )
            continue
        reason = "must_fetch_source_produced_no_records"
        if int(row.get("attempted_count") or 0) == 0:
            reason = "must_fetch_source_not_attempted_for_extraction"
        elif int(row.get("skipped_due_to_cap_count") or 0) > 0:
            reason = "must_fetch_source_partially_skipped_due_to_chunk_cap"
        official_extraction_failures.append(
            {
                "source_id": source_id,
                "reason": reason,
                "budget": dict(row),
                "sample_chunks": _sample_chunks_for_source(evidence_chunks, source_id),
            }
        )

    core_metric_extraction_gaps = _core_metric_extraction_gaps(
        evidence_chunks,
        extraction_budget_by_source,
        context,
    )

    stats = {
        "extraction_mode": "llm_structured_output",
        "llm_enabled": True,
        "llm_provider": settings.get("provider"),
        "llm_model": settings.get("model"),
        "llm_eligible_chunk_count": llm_eligible_chunk_count,
        "llm_call_count": llm_call_count,
        "primary_llm_call_count": primary_llm_call_count,
        "llm_success_count": llm_success_count,
        "llm_transport_success_count": llm_success_count,
        "llm_empty_output_count": llm_empty_output_count,
        "llm_non_empty_output_count": max(
            0, llm_success_count - llm_empty_output_count
        ),
        "valid_record_count": len(records),
        "unique_valid_record_count": len(
            {
                str(record.record_id)
                for record in records
                if getattr(record, "record_id", None)
            }
        ),
        "llm_empty_output_reasons": dict(sorted(llm_empty_output_reasons.items())),
        "llm_empty_output_diagnostics": llm_empty_output_diagnostics,
        "focused_recovery_candidate_count": len(focused_queue),
        "focused_recovery_call_count": focused_recovery_call_count,
        "focused_retry_succeeded_count": focused_retry_succeeded_count,
        "focused_retry_empty_count": focused_retry_empty_count,
        "deterministic_recovery_succeeded_count": (
            deterministic_recovery_succeeded_count
        ),
        "focused_recovery_valid_record_count": (
            focused_recovery_valid_record_count
        ),
        "focused_recovery_field_gain_count": focused_recovery_field_gain_count,
        "focused_recovery_status_by_source": dict(
            sorted(focused_recovery_status_by_source.items())
        ),
        "llm_error_count": llm_error_count,
        "llm_fallback_count": llm_fallback_count,
        "llm_error_messages": llm_error_messages,
        "rule_based_fallback_record_count": rule_based_fallback_record_count,
        "official_outbreak_deterministic_record_count": (
            official_outbreak_deterministic_record_count
        ),
        "official_outbreak_deterministic_diagnostics": (
            official_outbreak_deterministic_diagnostics
        ),
        "target_data_chunk_count": target_data_count,
        "extractable_chunk_count": llm_eligible_chunk_count,
        "field_detection_counts": {
            f: field_counters.get(f, 0) for f in _FIELD_DETECTION_KEYS
        },
        "llm_max_chunks": max_chunks,
        "deprecated_llm_max_chunks": max_chunks is not None,
        "scheduler_model_call_count": scheduler_model_call_count,
        "provider_deferred_chunk_count": len(provider_deferred_ids),
        "provider_deferred_chunk_ids": sorted(provider_deferred_ids),
        "scheduler_cache_hit_count": scheduler_cache_hit_count,
        "rolling_low_yield_stop_triggered": rolling_low_yield_stop_triggered,
        "extraction_budget_ledger": budget_ledger.as_dict(),
        "llm_skipped_due_to_chunk_cap_count": llm_skipped_due_to_chunk_cap_count,
        "skipped_context_only_chunk_count": skipped_context_only_chunk_count,
        "skipped_context_only_source_ids": sorted(skipped_context_only_source_ids),
        "skipped_disease_mismatch_chunk_count": skipped_disease_mismatch_chunk_count,
        "skipped_disease_mismatch_source_ids": sorted(
            skipped_disease_mismatch_source_ids
        ),
        "must_fetch_disease_gate_bypass_count": must_fetch_disease_gate_bypass_count,
        "official_extraction_queue": official_extraction_queue,
        "batched_metric_row_call_count": batched_metric_row_call_count,
        "metric_primary_call_limit": metric_primary_call_limit,
        "metric_row_batch_deferred_for_case_coverage_count": (
            metric_row_batch_deferred_for_case_coverage_count
        ),
        "metric_row_batch_record_count": metric_row_batch_record_count,
        "metric_row_record_count_by_source": dict(
            sorted(metric_row_record_count_by_source.items())
        ),
        "deterministic_metric_row_record_count": deterministic_metric_row_record_count,
        "fallback_text_chunk_call_count": fallback_text_chunk_call_count,
        "skipped_text_fallback_after_row_record_count": (
            skipped_text_fallback_after_row_record_count
        ),
        "target_source_llm_call_count": target_source_llm_call_count,
        "skipped_context_extraction_count": skipped_context_extraction_count,
        "no_task_collection_document": no_task_collection_document,
        "extraction_blocking_reason": extraction_blocking_reason,
        "official_extraction_failures": official_extraction_failures,
        "core_metric_extraction_gaps": core_metric_extraction_gaps,
        "core_metric_extraction_gap_count": len(core_metric_extraction_gaps),
        "official_extraction_resolved_by_equivalent_sources": (
            official_extraction_resolved_by_equivalent_sources
        ),
        "official_extraction_resolved_by_equivalent_source_count": len(
            official_extraction_resolved_by_equivalent_sources
        ),
        "extraction_budget_by_source": extraction_budget_by_source,
    }
    return records, stats


def structured_extraction(state: DataCollectionState) -> dict:
    """Convert target-data evidence chunks into generic public-health records.

    Default = deterministic rule-based extraction. When
    `ENABLE_LLM_EXTRACTION=true`, run the optional LLM extractor with
    per-chunk fallback to the deterministic extractor when configured.
    """

    deterministic_policy = StructuredExtractionPolicy(
        **load_structured_extraction_policy()
    )
    llm_policy = _load_llm_policy()
    chunks = list(state.get("evidence_chunks") or [])
    extraction_context = _build_extraction_context(state, deterministic_policy)
    if get_env("PIPELINE_MODE") == "evidence":
        chunks = [chunk for chunk in chunks if not extraction_skip_reason(chunk, extraction_context)]

    llm_enabled = llm_clients.llm_extraction_enabled()
    fallback_to_rule_based = llm_clients.llm_fallback_to_rule_based()

    if llm_enabled:
        records, llm_stats = _llm_extract_records_from_chunks(
            chunks,
            llm_policy,
            deterministic_policy,
            fallback_to_rule_based,
            extraction_context,
        )
        raw_record_dicts = [r.model_dump() for r in records]
        skipped_count = max(
            0,
            len(chunks)
            - int(llm_stats.get("extractable_chunk_count", 0)),
        )
        summary = {
            "input_chunk_count": len(chunks),
            "target_data_chunk_count": llm_stats.get("target_data_chunk_count", 0),
            "extractable_chunk_count": llm_stats.get("extractable_chunk_count", 0),
            "raw_record_count": len(records),
            "skipped_chunk_count": skipped_count,
            "extraction_method": llm_policy.llm_extraction_method,
            "field_detection_counts": llm_stats.get("field_detection_counts") or {},
            "extraction_mode": "llm_structured_output",
            "llm_enabled": True,
            "llm_provider": llm_stats.get("llm_provider"),
            "llm_model": llm_stats.get("llm_model"),
            "llm_eligible_chunk_count": llm_stats.get("llm_eligible_chunk_count", 0),
            "llm_call_count": llm_stats.get("llm_call_count", 0),
            "primary_llm_call_count": llm_stats.get(
                "primary_llm_call_count", 0
            ),
            "llm_success_count": llm_stats.get("llm_success_count", 0),
            "llm_transport_success_count": llm_stats.get(
                "llm_transport_success_count", llm_stats.get("llm_success_count", 0)
            ),
            "llm_empty_output_count": llm_stats.get("llm_empty_output_count", 0),
            "llm_non_empty_output_count": llm_stats.get(
                "llm_non_empty_output_count", 0
            ),
            "valid_record_count": llm_stats.get("valid_record_count", len(records)),
            "unique_valid_record_count": llm_stats.get(
                "unique_valid_record_count", len(records)
            ),
            "llm_empty_output_reasons": llm_stats.get(
                "llm_empty_output_reasons"
            )
            or {},
            "llm_empty_output_diagnostics": llm_stats.get(
                "llm_empty_output_diagnostics"
            )
            or [],
            "focused_recovery_candidate_count": llm_stats.get(
                "focused_recovery_candidate_count", 0
            ),
            "focused_recovery_call_count": llm_stats.get(
                "focused_recovery_call_count", 0
            ),
            "focused_retry_succeeded_count": llm_stats.get(
                "focused_retry_succeeded_count", 0
            ),
            "focused_retry_empty_count": llm_stats.get(
                "focused_retry_empty_count", 0
            ),
            "deterministic_recovery_succeeded_count": llm_stats.get(
                "deterministic_recovery_succeeded_count", 0
            ),
            "focused_recovery_valid_record_count": llm_stats.get(
                "focused_recovery_valid_record_count", 0
            ),
            "focused_recovery_field_gain_count": llm_stats.get(
                "focused_recovery_field_gain_count", 0
            ),
            "focused_recovery_status_by_source": llm_stats.get(
                "focused_recovery_status_by_source"
            )
            or {},
            "llm_error_count": llm_stats.get("llm_error_count", 0),
            "llm_error_messages": llm_stats.get("llm_error_messages") or [],
            "llm_fallback_count": llm_stats.get("llm_fallback_count", 0),
            "rule_based_fallback_record_count": llm_stats.get(
                "rule_based_fallback_record_count", 0
            ),
            "official_outbreak_deterministic_record_count": llm_stats.get(
                "official_outbreak_deterministic_record_count", 0
            ),
            "official_outbreak_deterministic_diagnostics": llm_stats.get(
                "official_outbreak_deterministic_diagnostics"
            )
            or [],
            "core_metric_extraction_gaps": llm_stats.get(
                "core_metric_extraction_gaps"
            )
            or [],
            "core_metric_extraction_gap_count": llm_stats.get(
                "core_metric_extraction_gap_count", 0
            ),
            "skipped_context_only_chunk_count": llm_stats.get(
                "skipped_context_only_chunk_count", 0
            ),
            "skipped_context_only_source_ids": llm_stats.get(
                "skipped_context_only_source_ids"
            )
            or [],
            "skipped_disease_mismatch_chunk_count": llm_stats.get(
                "skipped_disease_mismatch_chunk_count", 0
            ),
            "skipped_disease_mismatch_source_ids": llm_stats.get(
                "skipped_disease_mismatch_source_ids"
            )
            or [],
            "llm_max_chunks": llm_stats.get("llm_max_chunks"),
            "deprecated_llm_max_chunks": bool(
                llm_stats.get("deprecated_llm_max_chunks", False)
            ),
            "scheduler_model_call_count": llm_stats.get(
                "scheduler_model_call_count", 0
            ),
            "provider_deferred_chunk_count": llm_stats.get("provider_deferred_chunk_count", 0),
            "provider_deferred_chunk_ids": llm_stats.get("provider_deferred_chunk_ids") or [],
            "scheduler_cache_hit_count": llm_stats.get(
                "scheduler_cache_hit_count", 0
            ),
            "extraction_budget_ledger": llm_stats.get(
                "extraction_budget_ledger"
            )
            or {},
            "llm_skipped_due_to_chunk_cap_count": llm_stats.get(
                "llm_skipped_due_to_chunk_cap_count", 0
            ),
            "must_fetch_disease_gate_bypass_count": llm_stats.get(
                "must_fetch_disease_gate_bypass_count", 0
            ),
            "official_extraction_queue": llm_stats.get("official_extraction_queue")
            or [],
            "batched_metric_row_call_count": llm_stats.get(
                "batched_metric_row_call_count", 0
            ),
            "metric_primary_call_limit": llm_stats.get(
                "metric_primary_call_limit", 0
            ),
            "metric_row_batch_deferred_for_case_coverage_count": llm_stats.get(
                "metric_row_batch_deferred_for_case_coverage_count", 0
            ),
            "metric_row_batch_record_count": llm_stats.get(
                "metric_row_batch_record_count", 0
            ),
            "metric_row_record_count_by_source": llm_stats.get(
                "metric_row_record_count_by_source"
            )
            or {},
            "deterministic_metric_row_record_count": llm_stats.get(
                "deterministic_metric_row_record_count", 0
            ),
            "fallback_text_chunk_call_count": llm_stats.get(
                "fallback_text_chunk_call_count", 0
            ),
            "skipped_text_fallback_after_row_record_count": llm_stats.get(
                "skipped_text_fallback_after_row_record_count", 0
            ),
            "target_source_llm_call_count": llm_stats.get(
                "target_source_llm_call_count", 0
            ),
            "skipped_context_extraction_count": llm_stats.get(
                "skipped_context_extraction_count", 0
            ),
            "no_task_collection_document": bool(
                llm_stats.get("no_task_collection_document", False)
            ),
            "extraction_blocking_reason": llm_stats.get(
                "extraction_blocking_reason"
            ),
            "official_extraction_failures": llm_stats.get(
                "official_extraction_failures"
            )
            or [],
            "core_metric_extraction_gaps": llm_stats.get(
                "core_metric_extraction_gaps"
            )
            or [],
            "core_metric_extraction_gap_count": llm_stats.get(
                "core_metric_extraction_gap_count", 0
            ),
            "official_extraction_resolved_by_equivalent_sources": llm_stats.get(
                "official_extraction_resolved_by_equivalent_sources"
            )
            or [],
            "official_extraction_resolved_by_equivalent_source_count": llm_stats.get(
                "official_extraction_resolved_by_equivalent_source_count",
                0,
            ),
            "extraction_budget_by_source": llm_stats.get(
                "extraction_budget_by_source"
            )
            or {},
        }
        llm_summary = {
            "extraction_mode": "llm_structured_output",
            "llm_enabled": True,
            "llm_provider": llm_stats.get("llm_provider"),
            "llm_model": llm_stats.get("llm_model"),
            "llm_eligible_chunk_count": llm_stats.get("llm_eligible_chunk_count", 0),
            "llm_call_count": llm_stats.get("llm_call_count", 0),
            "primary_llm_call_count": llm_stats.get(
                "primary_llm_call_count", 0
            ),
            "llm_success_count": llm_stats.get("llm_success_count", 0),
            "llm_transport_success_count": llm_stats.get(
                "llm_transport_success_count", llm_stats.get("llm_success_count", 0)
            ),
            "llm_empty_output_count": llm_stats.get("llm_empty_output_count", 0),
            "llm_non_empty_output_count": llm_stats.get(
                "llm_non_empty_output_count", 0
            ),
            "valid_record_count": llm_stats.get("valid_record_count", len(records)),
            "unique_valid_record_count": llm_stats.get(
                "unique_valid_record_count", len(records)
            ),
            "llm_empty_output_reasons": llm_stats.get(
                "llm_empty_output_reasons"
            )
            or {},
            "llm_empty_output_diagnostics": llm_stats.get(
                "llm_empty_output_diagnostics"
            )
            or [],
            "focused_recovery_candidate_count": llm_stats.get(
                "focused_recovery_candidate_count", 0
            ),
            "focused_recovery_call_count": llm_stats.get(
                "focused_recovery_call_count", 0
            ),
            "focused_retry_succeeded_count": llm_stats.get(
                "focused_retry_succeeded_count", 0
            ),
            "focused_retry_empty_count": llm_stats.get(
                "focused_retry_empty_count", 0
            ),
            "deterministic_recovery_succeeded_count": llm_stats.get(
                "deterministic_recovery_succeeded_count", 0
            ),
            "focused_recovery_valid_record_count": llm_stats.get(
                "focused_recovery_valid_record_count", 0
            ),
            "focused_recovery_field_gain_count": llm_stats.get(
                "focused_recovery_field_gain_count", 0
            ),
            "focused_recovery_status_by_source": llm_stats.get(
                "focused_recovery_status_by_source"
            )
            or {},
            "llm_error_count": llm_stats.get("llm_error_count", 0),
            "llm_fallback_count": llm_stats.get("llm_fallback_count", 0),
            "llm_error_messages": llm_stats.get("llm_error_messages") or [],
            "rule_based_fallback_record_count": llm_stats.get(
                "rule_based_fallback_record_count", 0
            ),
            "skipped_context_only_chunk_count": llm_stats.get(
                "skipped_context_only_chunk_count", 0
            ),
            "skipped_context_only_source_ids": llm_stats.get(
                "skipped_context_only_source_ids"
            )
            or [],
            "skipped_disease_mismatch_chunk_count": llm_stats.get(
                "skipped_disease_mismatch_chunk_count", 0
            ),
            "skipped_disease_mismatch_source_ids": llm_stats.get(
                "skipped_disease_mismatch_source_ids"
            )
            or [],
            "fallback_to_rule_based": fallback_to_rule_based,
            "llm_max_chunks": llm_stats.get("llm_max_chunks"),
            "deprecated_llm_max_chunks": bool(
                llm_stats.get("deprecated_llm_max_chunks", False)
            ),
            "scheduler_model_call_count": llm_stats.get(
                "scheduler_model_call_count", 0
            ),
            "provider_deferred_chunk_count": llm_stats.get("provider_deferred_chunk_count", 0),
            "provider_deferred_chunk_ids": llm_stats.get("provider_deferred_chunk_ids") or [],
            "scheduler_cache_hit_count": llm_stats.get(
                "scheduler_cache_hit_count", 0
            ),
            "extraction_budget_ledger": llm_stats.get(
                "extraction_budget_ledger"
            )
            or {},
            "llm_skipped_due_to_chunk_cap_count": llm_stats.get(
                "llm_skipped_due_to_chunk_cap_count", 0
            ),
            "must_fetch_disease_gate_bypass_count": llm_stats.get(
                "must_fetch_disease_gate_bypass_count", 0
            ),
            "official_extraction_failure_count": len(
                llm_stats.get("official_extraction_failures") or []
            ),
            "batched_metric_row_call_count": llm_stats.get(
                "batched_metric_row_call_count", 0
            ),
            "metric_primary_call_limit": llm_stats.get(
                "metric_primary_call_limit", 0
            ),
            "metric_row_batch_deferred_for_case_coverage_count": llm_stats.get(
                "metric_row_batch_deferred_for_case_coverage_count", 0
            ),
            "metric_row_batch_record_count": llm_stats.get(
                "metric_row_batch_record_count", 0
            ),
            "metric_row_record_count_by_source": llm_stats.get(
                "metric_row_record_count_by_source"
            )
            or {},
            "deterministic_metric_row_record_count": llm_stats.get(
                "deterministic_metric_row_record_count", 0
            ),
            "fallback_text_chunk_call_count": llm_stats.get(
                "fallback_text_chunk_call_count", 0
            ),
            "target_source_llm_call_count": llm_stats.get(
                "target_source_llm_call_count", 0
            ),
            "skipped_context_extraction_count": llm_stats.get(
                "skipped_context_extraction_count", 0
            ),
            "no_task_collection_document": bool(
                llm_stats.get("no_task_collection_document", False)
            ),
            "extraction_blocking_reason": llm_stats.get(
                "extraction_blocking_reason"
            ),
            "official_extraction_resolved_by_equivalent_source_count": llm_stats.get(
                "official_extraction_resolved_by_equivalent_source_count",
                0,
            ),
        }
    else:
        records, det_stats = _rule_based_extract_records_from_chunks(
            chunks, deterministic_policy, context=extraction_context
        )
        deterministic_metric_row_record_count = 0
        chunk_index_by_source: dict[str, int] = {}
        for chunk in chunks:
            if not isinstance(chunk, dict) or not _chunk_is_metric_row(chunk):
                continue
            source_id = str(chunk.get("source_id") or "")
            deterministic_records = _deterministic_metric_row_records(
                chunk,
                start_index=chunk_index_by_source.get(source_id, 0) + 1,
                llm_policy=llm_policy,
                settings=llm_clients.get_llm_settings(),
                context=extraction_context,
            )
            if not deterministic_records:
                continue
            records.extend(deterministic_records)
            chunk_index_by_source[source_id] = (
                chunk_index_by_source.get(source_id, 0)
                + len(deterministic_records)
            )
            deterministic_metric_row_record_count += len(deterministic_records)
        raw_record_dicts = [r.model_dump() for r in records]
        summary = {
            "input_chunk_count": len(chunks),
            "target_data_chunk_count": det_stats["target_data_chunk_count"],
            "extractable_chunk_count": det_stats["extractable_chunk_count"],
            "raw_record_count": len(records),
            "skipped_chunk_count": det_stats["skipped_chunk_count"],
            "extraction_method": deterministic_policy.extraction_method,
            "field_detection_counts": det_stats["field_detection_counts"],
            "skipped_context_only_chunk_count": det_stats.get(
                "skipped_context_only_chunk_count", 0
            ),
            "skipped_context_only_source_ids": det_stats.get(
                "skipped_context_only_source_ids"
            )
            or [],
            "skipped_disease_mismatch_chunk_count": det_stats.get(
                "skipped_disease_mismatch_chunk_count", 0
            ),
            "skipped_disease_mismatch_source_ids": det_stats.get(
                "skipped_disease_mismatch_source_ids"
            )
            or [],
            "extraction_mode": "deterministic_rule_based",
            "llm_enabled": False,
            "llm_call_count": 0,
            "primary_llm_call_count": 0,
            "llm_success_count": 0,
            "llm_transport_success_count": 0,
            "llm_empty_output_count": 0,
            "llm_non_empty_output_count": 0,
            "valid_record_count": len(records),
            "unique_valid_record_count": len(
                {
                    str(record.record_id)
                    for record in records
                    if getattr(record, "record_id", None)
                }
            ),
            "llm_empty_output_reasons": {},
            "llm_empty_output_diagnostics": [],
            "focused_recovery_candidate_count": 0,
            "focused_recovery_call_count": 0,
            "focused_retry_succeeded_count": 0,
            "focused_retry_empty_count": 0,
            "deterministic_recovery_succeeded_count": 0,
            "focused_recovery_valid_record_count": 0,
            "focused_recovery_field_gain_count": 0,
            "focused_recovery_status_by_source": {},
            "llm_error_count": 0,
            "llm_fallback_count": 0,
            "llm_max_chunks": _parse_llm_max_chunks(),
            "extraction_budget_ledger": {},
            "llm_skipped_due_to_chunk_cap_count": 0,
            "must_fetch_disease_gate_bypass_count": 0,
            "official_extraction_queue": [],
            "official_extraction_failures": [],
            "extraction_budget_by_source": {},
            "batched_metric_row_call_count": 0,
            "metric_primary_call_limit": 0,
            "metric_row_batch_deferred_for_case_coverage_count": 0,
            "metric_row_batch_record_count": deterministic_metric_row_record_count,
            "deterministic_metric_row_record_count": deterministic_metric_row_record_count,
            "fallback_text_chunk_call_count": 0,
            "target_source_llm_call_count": 0,
            "skipped_context_extraction_count": 0,
        }
        llm_summary = {
            "extraction_mode": "deterministic_rule_based",
            "llm_enabled": False,
            "llm_provider": None,
            "llm_model": None,
            "llm_eligible_chunk_count": 0,
            "llm_call_count": 0,
            "primary_llm_call_count": 0,
            "llm_success_count": 0,
            "llm_transport_success_count": 0,
            "llm_empty_output_count": 0,
            "llm_non_empty_output_count": 0,
            "valid_record_count": len(records),
            "unique_valid_record_count": len(
                {
                    str(record.record_id)
                    for record in records
                    if getattr(record, "record_id", None)
                }
            ),
            "llm_empty_output_reasons": {},
            "llm_empty_output_diagnostics": [],
            "focused_recovery_candidate_count": 0,
            "focused_recovery_call_count": 0,
            "focused_retry_succeeded_count": 0,
            "focused_retry_empty_count": 0,
            "deterministic_recovery_succeeded_count": 0,
            "focused_recovery_valid_record_count": 0,
            "focused_recovery_field_gain_count": 0,
            "focused_recovery_status_by_source": {},
            "llm_error_count": 0,
            "llm_fallback_count": 0,
            "llm_error_messages": [],
            "rule_based_fallback_record_count": 0,
            "fallback_to_rule_based": fallback_to_rule_based,
            "skipped_disease_mismatch_chunk_count": det_stats.get(
                "skipped_disease_mismatch_chunk_count", 0
            ),
            "skipped_disease_mismatch_source_ids": det_stats.get(
                "skipped_disease_mismatch_source_ids"
            )
            or [],
            "llm_max_chunks": _parse_llm_max_chunks(),
            "extraction_budget_ledger": {},
            "llm_skipped_due_to_chunk_cap_count": 0,
            "must_fetch_disease_gate_bypass_count": 0,
            "official_extraction_failure_count": 0,
            "batched_metric_row_call_count": 0,
            "metric_primary_call_limit": 0,
            "metric_row_batch_deferred_for_case_coverage_count": 0,
            "metric_row_batch_record_count": deterministic_metric_row_record_count,
            "deterministic_metric_row_record_count": deterministic_metric_row_record_count,
            "fallback_text_chunk_call_count": 0,
            "target_source_llm_call_count": 0,
            "skipped_context_extraction_count": 0,
        }

    record_task_fit_assessments = _record_task_fit_assessments(
        raw_record_dicts,
        extraction_context,
    )
    disease_counts = dict(
        Counter(str(r.get("disease") or "unknown") for r in raw_record_dicts)
    )
    source_type_counts = dict(
        Counter(str(r.get("source_type") or "unknown") for r in raw_record_dicts)
    )
    extraction_method_counts = dict(
        Counter(str(r.get("extraction_method") or "unknown") for r in raw_record_dicts)
    )
    warning_counter: Counter = Counter()
    for record in raw_record_dicts:
        for warning in (record.get("semantic_warnings") or []) + (
            record.get("extraction_warnings") or []
        ):
            warning_counter[warning] += 1
    metric_row_extraction_audit = [
        {
            "record_id": record.get("record_id"),
            "source_id": record.get("source_id"),
            "document_id": record.get("document_id"),
            "supporting_chunk_id": record.get("supporting_chunk_id"),
            "chunk_kind": record.get("chunk_kind"),
            "metric_name": record.get("metric_name"),
            "metric_value": record.get("metric_value"),
            "metric_unit": record.get("metric_unit"),
            "metric_category": record.get("metric_category"),
            "metric_denominator": record.get("metric_denominator"),
            "metric_period_start": record.get("metric_period_start"),
            "metric_period_end": record.get("metric_period_end"),
            "metric_period_source": record.get("metric_period_source"),
            "source_row_id": record.get("source_row_id"),
            "source_column_label": record.get("source_column_label"),
            "metric_column_label": record.get("metric_column_label"),
            "metric_row_binding_status": record.get("metric_row_binding_status"),
            "metric_column_semantics_status": record.get(
                "metric_column_semantics_status"
            ),
            "resolved_column_period_type": record.get(
                "resolved_column_period_type"
            ),
            "column_period_resolution_reason": record.get(
                "column_period_resolution_reason"
            ),
            "column_period_warning_flags": record.get(
                "column_period_warning_flags"
            )
            or [],
            "metric_period_label": record.get("metric_period_label"),
            "column_semantics_resolution_method": record.get(
                "column_semantics_resolution_method"
            ),
            "column_semantics_confidence": record.get(
                "column_semantics_confidence"
            ),
            "table_header": record.get("table_header"),
            "heading_context": record.get("heading_context"),
            "row_context_type": record.get("row_context_type"),
            "row_quote": record.get("row_quote") or record.get("evidence_quote"),
            "semantic_warnings": record.get("semantic_warnings") or [],
            "reporting_period": record.get("reporting_period"),
            "source_url": record.get("source_url"),
            "evidence_quote": record.get("evidence_quote"),
        }
        for record in raw_record_dicts
        if record.get("metric_value") not in (None, "")
    ]
    metric_category_counts = dict(
        Counter(
            str(row.get("metric_category") or "unknown")
            for row in metric_row_extraction_audit
        )
    )
    extraction_budget_rows = list(
        (summary.get("extraction_budget_by_source") or {}).values()
    )
    task_source_extraction_attempted_count = sum(
        1
        for row in extraction_budget_rows
        if int(row.get("attempted_count") or 0) > 0
        and (
            row.get("budget_bucket") == "verified_target_collection"
            or row.get("target_fit_status") == "verified_target"
            or row.get("must_fetch") is True
        )
    )
    best_available_extraction_count = sum(
        1
        for row in extraction_budget_rows
        if int(row.get("attempted_count") or 0) > 0
        and row.get("budget_bucket") == "official_or_high_trust"
    )
    target_metric_row_chunk_count = sum(
        1
        for chunk in chunks
        if (chunk.get("chunk_kind") or "text") == "metric_row"
        and _chunk_priority(chunk, extraction_context) <= 1
    )
    fallback_metric_row_chunk_count = sum(
        1
        for chunk in chunks
        if (chunk.get("chunk_kind") or "text") == "metric_row"
        and _chunk_priority(chunk, extraction_context) > 1
        and not _is_context_only_chunk(chunk, None)
        and _chunk_is_official_or_high_trust(chunk, extraction_context)
    )
    context_skipped_count = int(summary.get("skipped_context_only_chunk_count") or 0) + int(
        summary.get("skipped_context_extraction_count") or 0
    )
    metric_extraction_plan = {
        "metric_record_count": len(metric_row_extraction_audit),
        "metric_category_counts": metric_category_counts,
        "metric_source_count": len(
            {
                str(row.get("source_id") or "")
                for row in metric_row_extraction_audit
                if row.get("source_id")
            }
        ),
        "table_chunk_count": sum(
            1
            for chunk in chunks
            if (chunk.get("chunk_kind") or "text") in {"table", "metric_row"}
        ),
        "official_extraction_queue_count": len(
            summary.get("official_extraction_queue") or []
        ),
        "llm_call_count": summary.get("llm_call_count", 0),
        "llm_max_chunks": summary.get("llm_max_chunks"),
        "target_source_llm_call_count": summary.get("target_source_llm_call_count", 0),
        "task_source_extraction_attempted_count": (
            task_source_extraction_attempted_count
        ),
        "skipped_context_extraction_count": summary.get(
            "skipped_context_extraction_count", 0
        ),
        "context_extraction_skipped_count": summary.get(
            "skipped_context_extraction_count", 0
        ),
        "context_skipped_count": context_skipped_count,
        "best_available_extraction_count": best_available_extraction_count,
        "batched_metric_row_call_count": summary.get(
            "batched_metric_row_call_count", 0
        ),
        "metric_row_record_count_by_source": summary.get(
            "metric_row_record_count_by_source"
        )
        or {},
        "deterministic_metric_row_record_count": summary.get(
            "deterministic_metric_row_record_count", 0
        ),
        "fallback_text_chunk_call_count": summary.get(
            "fallback_text_chunk_call_count", 0
        ),
        "target_text_fallback_attempted_count": summary.get(
            "fallback_text_chunk_call_count", 0
        ),
        "skipped_text_fallback_after_row_record_count": summary.get(
            "skipped_text_fallback_after_row_record_count", 0
        ),
        "core_metric_extraction_gap_count": summary.get(
            "core_metric_extraction_gap_count", 0
        ),
        "core_metric_extraction_gaps": summary.get("core_metric_extraction_gaps")
        or [],
        "metric_row_chunk_count": sum(
            1
            for chunk in chunks
            if (chunk.get("chunk_kind") or "text") == "metric_row"
        ),
        "target_metric_row_chunk_count": target_metric_row_chunk_count,
        "fallback_metric_row_chunk_count": fallback_metric_row_chunk_count,
    }
    generic_record_count = sum(
        1
        for record in raw_record_dicts
        if record.get("record_schema") == "generic_public_health_record"
    )
    legacy_hantavirus_record_count = sum(
        1 for record in raw_record_dicts if record.get("disease") == "Hantavirus disease"
    )
    summary.update(
        {
            "generic_record_count": generic_record_count,
            "legacy_hantavirus_record_count": legacy_hantavirus_record_count,
            "disease_counts": disease_counts,
            "source_type_counts": source_type_counts,
            "extraction_method_counts": extraction_method_counts,
            "rejected_record_count": 0,
            "review_required_record_count": sum(
                1 for record in raw_record_dicts if record.get("requires_human_review")
            ),
            "unsupported_target_field_count": 0,
            "warnings": dict(warning_counter),
            "active_disease": extraction_context.get("disease_standard_name"),
            "record_schema": "generic_public_health_record",
            "record_task_fit_assessment_count": len(record_task_fit_assessments),
            "metric_extraction_plan": metric_extraction_plan,
            "metric_row_extraction_audit_count": len(metric_row_extraction_audit),
            "metric_category_counts": metric_category_counts,
        }
    )
    emit_workflow_progress(
        "structured_extraction",
        "structured extraction summary ready",
        {
            "input_chunk_count": summary.get("input_chunk_count"),
            "extractable_chunk_count": summary.get("extractable_chunk_count"),
            "raw_record_count": len(raw_record_dicts),
            "extraction_mode": summary.get("extraction_mode"),
            "llm_call_count": summary.get("llm_call_count", 0),
            "llm_error_count": summary.get("llm_error_count", 0),
        },
    )

    trace = append_trace(
        state,
        node_name="structured_extraction",
        message=(
            f"Built {len(raw_record_dicts)} raw records "
            f"(extraction_mode={summary['extraction_mode']}, "
            f"llm_enabled={summary['llm_enabled']})."
        ),
        metadata=summary,
    )
    from ..session_runtime import get_runtime
    runtime = get_runtime()
    attempted_ids = set(state.get("extraction_attempted_chunk_ids") or [])
    if not llm_enabled:
        attempted_ids.update(det_stats.get("attempted_chunk_ids") or [])
    if runtime:
        attempted_ids.update(runtime.ledger.attempted_chunk_ids())
    return {
        "extraction_attempted_chunk_ids": sorted(attempted_ids),
        "raw_records": raw_record_dicts,
        "structured_extraction_summary": summary,
        "llm_extraction_summary": llm_summary,
        "record_task_fit_assessments": record_task_fit_assessments,
        "metric_extraction_plan": metric_extraction_plan,
        "metric_row_extraction_audit": metric_row_extraction_audit,
        "official_extraction_queue": summary.get("official_extraction_queue") or [],
        "official_extraction_failures": summary.get("official_extraction_failures")
        or [],
        "official_extraction_resolved_by_equivalent_sources": summary.get(
            "official_extraction_resolved_by_equivalent_sources"
        )
        or [],
        "extraction_budget_by_source": summary.get("extraction_budget_by_source")
        or {},
        "disease_relevance_summary": update_disease_relevance_summary(
            {**state, "raw_records": raw_record_dicts},
            skipped_disease_mismatch_chunk_count=summary.get(
                "skipped_disease_mismatch_chunk_count", 0
            ),
        ),
        "collection_trace": trace,
    }


def schema_validation_and_repair(state: DataCollectionState) -> dict:
    """Validate raw records, apply deterministic repair, route to review/reject."""

    policy = StructuredExtractionPolicy(**load_structured_extraction_policy())
    raw_records = list(state.get("raw_records") or [])
    disease_context = build_disease_relevance_context(state)
    existing_queue = list(state.get("human_review_queue") or [])
    existing_review_ids = {item.get("review_id") for item in existing_queue}

    validated: list[dict] = []
    rejected: list[dict] = []
    new_review_items: list[HumanReviewItem] = []

    status_counter: Counter = Counter()
    prov_counter: Counter = Counter()
    missing_field_counter: Counter = Counter()
    repair_action_counter: Counter = Counter()
    needs_review_count = 0
    disease_counter: Counter = Counter()
    source_type_counter: Counter = Counter()
    extraction_method_counter: Counter = Counter()
    generic_record_count = 0
    legacy_hantavirus_record_count = 0
    disease_mismatch_rejected_count = 0
    disease_uncertain_review_count = 0

    for record in raw_records:
        validated_record, _result = _validate_record(record, policy)
        compatibility = assess_record_disease_compatibility(
            validated_record,
            disease_context,
        )
        validated_record.update(record_compatibility_fields(compatibility))
        compatibility_status = compatibility.get("status")
        if compatibility.get("reject_record"):
            validated_record["schema_status"] = "rejected"
            validated_record["requires_human_review"] = False
            errors = list(validated_record.get("validation_errors") or [])
            if "disease_mismatch" not in errors:
                errors.append("disease_mismatch")
            error = f"disease_mismatch: {compatibility.get('reason')}"
            if error not in errors:
                errors.append(error)
            validated_record["validation_errors"] = errors
            disease_mismatch_rejected_count += 1
        elif compatibility_status in {AMBIGUOUS_DISEASE, INSUFFICIENT_TEXT}:
            if validated_record.get("schema_status") != "rejected":
                validated_record["schema_status"] = "needs_review"
                validated_record["requires_human_review"] = True
                errors = list(validated_record.get("validation_errors") or [])
                if "disease_relevance_uncertain" not in errors:
                    errors.append("disease_relevance_uncertain")
                error = f"disease_relevance_uncertain: {compatibility.get('reason')}"
                if error not in errors:
                    errors.append(error)
                validated_record["validation_errors"] = errors
                disease_uncertain_review_count += 1
        status = validated_record.get("schema_status") or "rejected"
        status_counter[status] += 1
        prov_counter[validated_record.get("provenance_status") or "unknown"] += 1
        disease_counter[validated_record.get("disease") or "unknown"] += 1
        source_type_counter[validated_record.get("source_type") or "unknown"] += 1
        extraction_method_counter[
            validated_record.get("extraction_method") or "unknown"
        ] += 1
        if validated_record.get("record_schema") == "generic_public_health_record":
            generic_record_count += 1
        if validated_record.get("disease") == "Hantavirus disease":
            legacy_hantavirus_record_count += 1
        for f in validated_record.get("missing_fields") or []:
            missing_field_counter[f] += 1
        for a in validated_record.get("repair_actions") or []:
            repair_action_counter[a] += 1

        if status == "rejected":
            rejected.append(validated_record)
            continue

        validated.append(validated_record)
        if validated_record.get("requires_human_review"):
            needs_review_count += 1
            record_id = validated_record.get("record_id") or ""
            review_id = f"review_record_{record_id}"
            if review_id and review_id not in existing_review_ids:
                errs = validated_record.get("validation_errors") or []
                reason = (
                    "Record requires review after schema validation: "
                    + ", ".join(errs)
                ) if errs else "Record requires review after schema validation."
                new_review_items.append(
                    HumanReviewItem(
                        review_id=review_id,
                        item_type="record_schema_validation",
                        related_ids=[record_id],
                        reason=reason,
                        status="pending",
                    )
                )
                existing_review_ids.add(review_id)

    human_review_queue = list(existing_queue) + [
        item.model_dump() for item in new_review_items
    ]

    summary = {
        "raw_record_count": len(raw_records),
        "validated_record_count": len(validated),
        "rejected_record_count": len(rejected),
        "needs_review_count": needs_review_count,
        "human_review_item_count": len(new_review_items),
        "schema_status_counts": dict(status_counter),
        "provenance_status_counts": dict(prov_counter),
        "missing_field_counts": dict(missing_field_counter),
        "repair_action_counts": dict(repair_action_counter),
        "generic_record_count": generic_record_count,
        "legacy_hantavirus_record_count": legacy_hantavirus_record_count,
        "disease_counts": dict(disease_counter),
        "source_type_counts": dict(source_type_counter),
        "extraction_method_counts": dict(extraction_method_counter),
        "review_required_record_count": needs_review_count,
        "unsupported_target_field_count": 0,
        "warnings": {},
        "disease_mismatch_rejected_record_count": disease_mismatch_rejected_count,
        "disease_uncertain_review_record_count": disease_uncertain_review_count,
    }
    emit_workflow_progress(
        "schema_validation_and_repair",
        "schema validation summary ready",
        {
            "raw_record_count": len(raw_records),
            "validated_record_count": len(validated),
            "rejected_record_count": len(rejected),
            "needs_review_count": needs_review_count,
            "schema_status_counts": dict(status_counter),
            "repair_action_counts": dict(repair_action_counter),
        },
    )

    trace = append_trace(
        state,
        node_name="schema_validation_and_repair",
        message=(
            f"Validated {len(raw_records)} raw records: "
            f"{len(validated)} validated ({needs_review_count} need review), "
            f"{len(rejected)} rejected."
        ),
        metadata=summary,
    )
    return {
        "validated_records": validated,
        "rejected_records": rejected,
        "human_review_queue": human_review_queue,
        "schema_validation_summary": summary,
        "disease_relevance_summary": update_disease_relevance_summary(
            {**state, "validated_records": validated, "rejected_records": rejected},
            disease_mismatch_rejected_record_count=disease_mismatch_rejected_count,
            disease_uncertain_review_record_count=disease_uncertain_review_count,
        ),
        "collection_trace": trace,
    }
