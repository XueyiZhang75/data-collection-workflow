"""Task scoping nodes: scope planning, profile/schema setup, query strategy."""

from __future__ import annotations

from data_collection_workflow.environment import get_env

import json
import re
from collections import Counter

from .. import llm_clients
from ..query_policy import universal_queries_enabled
from ..config import (
    load_disease_intelligence_profile,
    load_hantavirus_collection_schema,
    load_hantavirus_profile,
    load_source_strategy,
)
from ..localized_source_planning import (
    build_localized_source_planning_hints,
    public_summary as localized_source_planning_public_summary,
)
from ..source_coverage import (
    build_source_coverage_requirements,
    build_task_evidence_contract,
)
from ..models import (
    CollectionSchema,
    CollectionSpec,
    DataFieldSpec,
    DiseaseIntelligenceProfile,
    DiseaseProfile,
    ExecutableSourcePlan,
    ObservationRecord,
    PlannedSearchQuery,
    SearchQuery,
    SearchQuerySet,
    ScreeningCriteria,
    SourceCategory,
    SourceStrategy,
    StructuredTaskInput,
)
from ..state import DataCollectionState, append_trace

_TIME_WINDOW_PATTERN = re.compile(r"(\d{4})\s*(?:-|to|–|—)\s*(\d{4})", re.IGNORECASE)
_US_TOKEN_PATTERN = re.compile(r"\b(US|USA|United States)\b")
_DEFAULT_DISEASE = "Hantavirus disease"
_DEFAULT_TARGET_POPULATION = "human"
_DEFAULT_TASK_TYPE = "public_health_case_and_outbreak_collection"
_DEFAULT_REQUIRED_FIELDS = [
    "disease",
    "virus_or_syndrome",
    "country",
    "subnational_location",
    "date_reported",
    "event_start_date",
    "event_end_date",
    "cases_confirmed",
    "cases_probable",
    "cases_suspected",
    "cases_unspecified",
    "deaths",
    "case_definition",
    "source_url",
    "source_type",
    "evidence_quote",
]
_DEFAULT_SOURCE_PRIORITY = [
    "official_public_health_agency",
    "international_organization_report",
    "peer_reviewed_literature",
    "structured_database",
    "news_and_situation_report",
]
_STRUCTURED_TASK_FIELDS = {
    "disease",
    "disease_aliases",
    "location",
    "start_date",
    "end_date",
    "target_fields",
    "source_preferences",
    "collection_mode",
    "user_request",
    "run_label",
}
_PLANNED_QUERY_URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_EXECUTABLE_SOURCE_TYPES = [
    "official_public_health_agency",
    "international_organization_report",
    "peer_reviewed_literature",
    "structured_database",
    "news_and_situation_report",
]


def _infer_geography(user_request: str) -> str | None:
    lowered = user_request.lower()
    if "global" in lowered or "worldwide" in lowered:
        return "global"
    if _US_TOKEN_PATTERN.search(user_request):
        return "United States of America"
    return None


def _infer_time_window(user_request: str) -> str | None:
    match = _TIME_WINDOW_PATTERN.search(user_request)
    if not match:
        return None
    start, end = match.group(1), match.group(2)
    return f"{start}-{end}"


def _clean_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _time_window(
    start_date: str | None,
    end_date: str | None,
    user_request: str,
) -> str | None:
    start = _clean_str(start_date)
    end = _clean_str(end_date)
    if start and end:
        return start if start == end else f"{start}-{end}"
    if start:
        return start
    if end:
        return end
    return _infer_time_window(user_request)


def _source_priority_from_preferences(value) -> list[str]:
    if isinstance(value, dict):
        for key in ("source_types", "preferred_source_types", "source_priority"):
            items = _as_str_list(value.get(key))
            if items:
                return items
        keys = [str(key).strip() for key in value.keys() if str(key).strip()]
        return keys or list(_DEFAULT_SOURCE_PRIORITY)
    return _as_str_list(value) or list(_DEFAULT_SOURCE_PRIORITY)


def _structured_task_from_state(
    state: DataCollectionState,
) -> tuple[StructuredTaskInput, str]:
    payload: dict = {}
    source = "legacy_user_request"

    nested = state.get("structured_task")
    if isinstance(nested, dict):
        payload.update(nested)
        source = "state.structured_task"

    for field in _STRUCTURED_TASK_FIELDS:
        value = state.get(field)  # type: ignore[literal-required]
        if value not in (None, "", [], {}):
            payload[field] = value
            source = "state.structured_fields"

    if "user_request" not in payload and state.get("user_request"):
        payload["user_request"] = state.get("user_request")

    if payload:
        return StructuredTaskInput(**payload), source

    return StructuredTaskInput(user_request=state.get("user_request")), "default"


def _data_focus(disease: str) -> str:
    if disease == _DEFAULT_DISEASE:
        return "human hantavirus case, outbreak, and surveillance data"
    return f"human {disease} case, outbreak, and surveillance data"


def _is_hantavirus_like(disease: str | None) -> bool:
    lowered = (disease or "").strip().lower()
    return lowered in {"hantavirus", "hantavirus disease", "hps"}


def task_intake_and_scope_planning(state: DataCollectionState) -> dict:
    """Build a deterministic CollectionSpec with light scope inference."""

    user_request = state.get("user_request", "") or ""
    task, input_source = _structured_task_from_state(state)
    task_user_request = task.user_request or user_request

    disease = _clean_str(task.disease) or _DEFAULT_DISEASE
    geography = _clean_str(task.location) or _infer_geography(task_user_request or "")
    time_window = _time_window(task.start_date, task.end_date, task_user_request or "")
    required_fields = list(task.target_fields or []) or list(_DEFAULT_REQUIRED_FIELDS)
    source_priority = _source_priority_from_preferences(task.source_preferences)
    warnings: list[str] = []

    spec = CollectionSpec(
        task_type=_DEFAULT_TASK_TYPE,
        disease=disease,
        target_population=_DEFAULT_TARGET_POPULATION,
        data_focus=_data_focus(disease),
        geography=geography,
        time_window=time_window,
        required_fields=required_fields,
        source_priority=source_priority,
        start_date=_clean_str(task.start_date),
        end_date=_clean_str(task.end_date),
        target_fields=required_fields,
        source_preferences=task.source_preferences,
        collection_mode=_clean_str(task.collection_mode),
        user_request=task_user_request,
        run_label=_clean_str(task.run_label),
        task_input_source=input_source,
        task_input_warnings=warnings,
    )
    summary = {
        "input_source": input_source,
        "structured_task_present": bool(state.get("structured_task")),
        "disease": spec.disease,
        "location": spec.geography,
        "start_date": spec.start_date,
        "end_date": spec.end_date,
        "time_window": spec.time_window,
        "target_field_count": len(spec.required_fields),
        "source_preference_count": len(spec.source_priority),
        "collection_mode": spec.collection_mode,
        "run_label": spec.run_label,
        "warnings": warnings,
    }
    message = (
        "Built CollectionSpec from structured task input."
        if input_source != "legacy_user_request"
        else "Built backward-compatible CollectionSpec from legacy user_request/defaults."
    )
    trace = append_trace(
        state,
        node_name="task_intake_and_scope_planning",
        message=message,
        metadata={
            "task_type": spec.task_type,
            "disease": spec.disease,
            "geography": spec.geography,
            "time_window": spec.time_window,
            "task_input_source": input_source,
            "target_field_count": len(spec.required_fields),
            "warnings": warnings,
        },
    )
    return {
        "structured_task": task.model_dump(),
        "collection_spec": spec.model_dump(),
        "task_intake_summary": summary,
        "collection_trace": trace,
    }


def _generic_disease_intelligence(spec: dict) -> DiseaseIntelligenceProfile:
    disease = _clean_str(spec.get("disease")) or "unknown disease"
    location = _clean_str(spec.get("geography")) or "{location}"
    time_window = _clean_str(spec.get("time_window")) or "{time_window}"
    fields = _as_str_list(spec.get("required_fields"))
    return DiseaseIntelligenceProfile(
        disease_input=disease,
        disease_standard_name=disease,
        disease_category="infectious disease",
        aliases=[disease],
        case_count_terms=["cases", "case counts"],
        death_terms=["deaths"],
        surveillance_terms=[f"{disease} surveillance"],
        outbreak_terms=["outbreak", "public health alert"],
        official_source_terms=["public health agency", "health department"],
        likely_reporting_agencies=["public health agency", "health department"],
        preferred_source_categories=[
            "official_public_health_agency",
            "international_organization_report",
            "peer_reviewed_literature",
        ],
        validation_source_categories=[
            "official_public_health_agency",
            "structured_database",
        ],
        suggested_geographic_granularity="task_defined",
        suggested_time_granularity="task_defined",
        extraction_priority_fields=fields or _DEFAULT_REQUIRED_FIELDS,
        count_semantics_notes=[
            "Generic deterministic fallback; disease-specific count semantics are not curated."
        ],
        disambiguation_risks=[
            "Disease-specific exclusions are not curated for this task."
        ],
        exclusion_terms=["unrelated disease", "animal-only data without human cases"],
        suggested_query_terms=[disease, f"{disease} cases", f"{disease} deaths"],
        suggested_query_templates=[
            f'"{disease}" cases deaths "{location}" {time_window}'.strip(),
            f'"{disease}" surveillance "{location}" {time_window}'.strip(),
        ],
        confidence=0.5,
        generation_method="generic_deterministic_fallback",
        warnings=["generic_disease_intelligence_fallback_used"],
    )


def _render_query_templates(profile: DiseaseIntelligenceProfile, spec: dict) -> list[str]:
    location = _clean_str(spec.get("geography")) or "{location}"
    time_window = _clean_str(spec.get("time_window")) or "{time_window}"
    rendered = []
    for template in profile.suggested_query_templates:
        rendered.append(
            str(template)
            .replace("{location}", location)
            .replace("{time_window}", time_window)
            .strip()
        )
    return rendered


def _profile_from_curated_or_generic(spec: dict) -> DiseaseIntelligenceProfile:
    disease = _clean_str(spec.get("disease"))
    curated = None if universal_queries_enabled() else load_disease_intelligence_profile(disease)
    if curated is not None:
        profile = DiseaseIntelligenceProfile(**curated)
    else:
        profile = _generic_disease_intelligence(spec)
    profile.suggested_query_templates = _render_query_templates(profile, spec)
    return profile


def _disease_intelligence_prompt(spec: dict, curated_profile: dict | None) -> str:
    payload = {
        "instruction": (
            "Generate disease intelligence for source planning only. Do not "
            "perform web search, do not provide URLs, and do not claim source "
            "discovery. Return structured terminology, source needs, query "
            "terms, validation source categories, extraction priorities, and "
            "warnings."
        ),
        "collection_spec": spec,
        "curated_profile_available": curated_profile is not None,
        "curated_profile_hint": curated_profile or {},
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _llm_force_enabled() -> bool:
    return (
        get_env("DISEASE_INTELLIGENCE_FORCE_LLM") or ""
    ).strip().lower() == "true"


def _llm_fallback_to_curated() -> bool:
    return (
        get_env("DISEASE_INTELLIGENCE_FALLBACK_TO_CURATED") or "true"
    ).strip().lower() != "false"


def _build_disease_intelligence_summary(
    profile: DiseaseIntelligenceProfile,
) -> dict:
    return {
        "disease_input": profile.disease_input,
        "disease_standard_name": profile.disease_standard_name,
        "generation_method": profile.generation_method,
        "alias_count": len(profile.aliases),
        "pathogen_term_count": len(profile.pathogen_terms),
        "source_category_count": len(profile.preferred_source_categories),
        "query_term_count": len(profile.suggested_query_terms),
        "source_need_count": len(profile.likely_reporting_agencies)
        + len(profile.official_source_terms),
        "warnings": list(profile.warnings),
    }


def disease_intelligence_builder(state: DataCollectionState) -> dict:
    """Build disease-specific terminology and source-need intelligence."""

    spec = state.get("collection_spec") or {}
    disease = _clean_str(spec.get("disease"))
    curated_raw = None if universal_queries_enabled() else load_disease_intelligence_profile(disease)
    llm_enabled = llm_clients.llm_disease_intelligence_enabled()
    generation_error: Exception | None = None

    if llm_enabled:
        try:
            raw = llm_clients.run_pydantic_structured_llm(
                system_prompt=(
                    "You create auditable disease intelligence for the data "
                    "collection workflow. Return only structured source-planning "
                    "intelligence; do not search the web and do not invent URLs."
                ),
                user_prompt=_disease_intelligence_prompt(spec, curated_raw),
                schema_model=DiseaseIntelligenceProfile,
                temperature=0.0,
            )
            profile = DiseaseIntelligenceProfile(**raw)
            profile.generation_method = "llm_generated"
            profile.suggested_query_templates = _render_query_templates(profile, spec)
        except Exception as exc:  # noqa: BLE001 - advisory LLM falls back
            generation_error = exc
        else:
            summary = _build_disease_intelligence_summary(profile)
            trace = append_trace(
                state,
                node_name="disease_intelligence_builder",
                message=(
                    "Built disease intelligence from LLM output "
                    f"({profile.disease_standard_name})."
                ),
                metadata={
                    **summary,
                    "llm_enabled": True,
                    "llm_failed": False,
                },
            )
            return {
                "disease_intelligence": profile.model_dump(),
                "disease_intelligence_summary": summary,
                "collection_trace": trace,
            }

    from ..provider_failures import ProviderAccountLimit
    provider_halted = isinstance(generation_error, ProviderAccountLimit)
    if generation_error is not None and not _llm_fallback_to_curated() and not provider_halted:
        raise RuntimeError(
            "disease intelligence LLM required but failed; "
            "set DISEASE_INTELLIGENCE_FALLBACK_TO_CURATED=true only for "
            "offline/debug fallback runs."
        ) from generation_error

    if generation_error is not None and (_llm_fallback_to_curated() or provider_halted):
        profile = _profile_from_curated_or_generic(spec)
        if curated_raw is not None:
            profile.generation_method = "llm_failed_curated_fallback"
            warning = "llm_disease_intelligence_failed_curated_fallback"
        else:
            profile.generation_method = "generic_deterministic_fallback"
            warning = "llm_disease_intelligence_failed_generic_fallback"
        if warning not in profile.warnings:
            profile.warnings.append(warning)
        profile.warnings.append(
            f"llm_failure_type:{type(generation_error).__name__}"
        )
    elif _llm_force_enabled() and curated_raw is None:
        profile = _generic_disease_intelligence(spec)
    else:
        profile = _profile_from_curated_or_generic(spec)

    summary = _build_disease_intelligence_summary(profile)
    trace = append_trace(
        state,
        node_name="disease_intelligence_builder",
        message=(
            "Built disease intelligence "
            f"({profile.disease_standard_name}, generation_method={profile.generation_method})."
        ),
        metadata={
            **summary,
            "llm_enabled": llm_enabled,
            "llm_failed": generation_error is not None,
        },
    )
    return {
        "disease_intelligence": profile.model_dump(),
        "disease_intelligence_summary": summary,
        "collection_trace": trace,
    }


def _unique_preserve_order(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in _as_str_list(values):
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _schema_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "generic_disease"


def _primary_data_objects(
    target_fields: list[str],
    extraction_priority_fields: list[str],
) -> list[str]:
    joined = " ".join(target_fields + extraction_priority_fields).lower()
    objects = ["human case record", "surveillance summary"]
    if "death" in joined:
        objects.append("death record")
    if "hospital" in joined or "icu" in joined:
        objects.append("hospitalization record")
    if "outbreak" in joined:
        objects.append("outbreak event")
    return _unique_preserve_order(objects)


def _build_generated_disease_profile(
    intelligence: DiseaseIntelligenceProfile,
    spec: CollectionSpec,
) -> DiseaseProfile:
    include_terms = _unique_preserve_order(
        [
            intelligence.disease_standard_name,
            *intelligence.aliases,
            *intelligence.abbreviations,
            *intelligence.suggested_query_terms,
            *intelligence.pathogen_terms,
        ]
    )
    required_fields = _unique_preserve_order(
        [
            *spec.required_fields,
            *spec.target_fields,
            *intelligence.extraction_priority_fields,
        ]
    )
    return DiseaseProfile(
        disease_standard_name=intelligence.disease_standard_name,
        disease_family=intelligence.disease_category,
        include_terms=include_terms,
        syndrome_terms=_unique_preserve_order(intelligence.syndrome_terms),
        virus_terms=_unique_preserve_order(intelligence.pathogen_terms),
        exclude_terms=_unique_preserve_order(intelligence.exclusion_terms),
        target_population=spec.target_population or _DEFAULT_TARGET_POPULATION,
        primary_data_objects=_primary_data_objects(
            required_fields,
            intelligence.extraction_priority_fields,
        ),
        required_record_fields=required_fields,
    )


_GENERIC_SCHEMA_BASE_FIELDS = [
    "disease",
    "virus_or_syndrome",
    "country",
    "subnational_location",
    "date_reported",
    "event_start_date",
    "event_end_date",
    "cases_confirmed",
    "cases_probable",
    "cases_suspected",
    "cases_unspecified",
    "deaths",
    "case_definition",
    "source_id",
    "source_url",
    "source_type",
    "evidence_quote",
    "supporting_chunk_id",
    "statistical_count_type",
    "reporting_period",
    "as_of_date",
    "geographic_scope",
    "geographic_scope_type",
]

_REQUIRED_SCHEMA_FIELDS = {
    "disease",
    "source_url",
    "source_type",
    "evidence_quote",
}

_FIELD_TYPES = {
    "cases_confirmed": "non_negative_number_or_null",
    "cases_probable": "non_negative_number_or_null",
    "cases_suspected": "non_negative_number_or_null",
    "cases_unspecified": "non_negative_number_or_null",
    "deaths": "non_negative_number_or_null",
    "source_url": "string",
    "source_type": "string",
    "evidence_quote": "string",
}

_FIELD_DESCRIPTIONS = {
    "disease": "Standard disease name for the requested task.",
    "virus_or_syndrome": "Specific pathogen, syndrome, or disease subtype if stated by the source.",
    "country": "Country where the human cases, deaths, outbreak, or surveillance data occurred.",
    "subnational_location": "State, province, city, county, district, hospital, or other subnational location as stated.",
    "date_reported": "Date when the data were reported or published; preserve partial dates when needed.",
    "event_start_date": "Start date of the case event, outbreak, or reporting period if explicitly stated.",
    "event_end_date": "End date of the case event, outbreak, or reporting period if explicitly stated.",
    "cases_confirmed": "Laboratory-confirmed or source-defined confirmed human cases.",
    "cases_probable": "Source-defined probable human cases.",
    "cases_suspected": "Source-defined suspected human cases.",
    "cases_unspecified": "Human cases where confirmation status is not clearly stated.",
    "deaths": "Deaths attributed to the reported disease event or surveillance period.",
    "case_definition": "Case definition or count label used by the source.",
    "source_id": "Workflow source identifier when available.",
    "source_url": "URL of the source from which the record was extracted.",
    "source_type": "Source category used by the data collection workflow.",
    "evidence_quote": "Verbatim source text supporting the extracted record.",
    "supporting_chunk_id": "Evidence chunk identifier supporting the extracted record.",
    "statistical_count_type": "Count semantics such as cumulative, annual, newly_reported, or point-in-time.",
    "reporting_period": "Reporting period or temporal aggregation window stated by the source.",
    "as_of_date": "Cutoff date for cumulative or dashboard-style counts when stated.",
    "geographic_scope": "Regional, multi-country, national, or subnational geographic scope if not represented by country alone.",
    "geographic_scope_type": "Canonical scope type such as country, subnational, region, multi_country, or global.",
}


def _field_spec(name: str) -> DataFieldSpec:
    return DataFieldSpec(
        name=name,
        type=_FIELD_TYPES.get(name, "string_or_null"),
        required=name in _REQUIRED_SCHEMA_FIELDS,
        description=_FIELD_DESCRIPTIONS.get(
            name,
            "User-requested target field retained for profile/schema planning.",
        ),
    )


def _unsupported_target_field_warnings(fields: list[str]) -> list[str]:
    supported = set(ObservationRecord.model_fields)
    return [
        f"target_field_not_yet_supported_by_record_model:{field}"
        for field in fields
        if field not in supported
    ]


def _build_generated_collection_schema(
    intelligence: DiseaseIntelligenceProfile,
    spec: CollectionSpec,
    generation_method: str,
    warnings: list[str],
) -> CollectionSchema:
    field_names = _unique_preserve_order(
        [
            *_GENERIC_SCHEMA_BASE_FIELDS,
            *spec.required_fields,
            *spec.target_fields,
            *intelligence.extraction_priority_fields,
        ]
    )
    disease_name = intelligence.disease_standard_name
    rules = [
        f"Extract only human public-health records about {disease_name}.",
        "Keep confirmed, probable, suspected, unspecified, death, hospitalization, and surveillance metrics separate when the source distinguishes them.",
        "Do not infer missing values or calculate derived metrics unless explicitly stated in the source.",
        "Every extracted record must preserve source_url, source_type, evidence_quote, and supporting_chunk_id when available.",
        "Preserve reporting_period, as_of_date, statistical_count_type, geographic_scope, and geographic_scope_type when the source states count semantics or aggregate geography.",
        "Current extraction record model remains ObservationRecord; unsupported target fields are retained in schema planning with warnings.",
        *warnings,
    ]
    return CollectionSchema(
        schema_name=f"{_schema_slug(disease_name)}_public_health_collection_schema",
        schema_version="0.3",
        description=(
            "Disease-aware generic public-health collection schema generated "
            f"for {disease_name} from structured task input and disease intelligence."
        ),
        record_type="public_health_case_or_outbreak_record",
        core_fields=[_field_spec(name) for name in field_names],
        extraction_rules=rules,
    )


_CANONICAL_SOURCE_TYPES = [
    "official_public_health_agency",
    "international_organization_report",
    "peer_reviewed_literature",
    "structured_database",
    "news_and_situation_report",
]


def _source_type_priorities(spec: CollectionSpec, intelligence: DiseaseIntelligenceProfile) -> dict[str, int]:
    ordered = _unique_preserve_order(
        [
            *spec.source_priority,
            *intelligence.preferred_source_categories,
            *intelligence.validation_source_categories,
            *_CANONICAL_SOURCE_TYPES,
        ]
    )
    return {source_type: index + 1 for index, source_type in enumerate(ordered)}


def _build_generated_source_strategy(
    intelligence: DiseaseIntelligenceProfile,
    spec: CollectionSpec,
) -> SourceStrategy:
    priorities = _source_type_priorities(spec, intelligence)
    source_need_text = ", ".join(
        _unique_preserve_order(
            [*intelligence.likely_reporting_agencies, *intelligence.official_source_terms]
        )[:5]
    )
    categories = [
        SourceCategory(
            source_type="official_public_health_agency",
            priority=priorities.get("official_public_health_agency", 1),
            description=(
                "Official public health agencies reporting disease-specific "
                f"case, death, hospitalization, outbreak, or surveillance data. "
                f"Likely source needs: {source_need_text or 'health department reports'}."
            ),
            example_domains=["cdc.gov", "who.int", "ecdc.europa.eu"],
        ),
        SourceCategory(
            source_type="international_organization_report",
            priority=priorities.get("international_organization_report", 2),
            description=(
                "International or regional health organization reports with "
                "disease-specific surveillance or situation updates."
            ),
            example_domains=["who.int", "paho.org", "ecdc.europa.eu"],
        ),
        SourceCategory(
            source_type="peer_reviewed_literature",
            priority=priorities.get("peer_reviewed_literature", 3),
            description=(
                "Peer-reviewed studies containing human case series, outbreak, "
                "surveillance, or epidemiological data for the requested disease."
            ),
            example_databases=["PubMed", "Europe PMC", "OpenAlex"],
        ),
        SourceCategory(
            source_type="structured_database",
            priority=priorities.get("structured_database", 4),
            description=(
                "Structured public-health dashboards, line lists, open data "
                "portals, or surveillance datasets for requested metrics."
            ),
            example_sources=["line lists", "surveillance datasets", "open data portals"],
        ),
        SourceCategory(
            source_type="news_and_situation_report",
            priority=priorities.get("news_and_situation_report", 5),
            description=(
                "Reputable news reports, situation reports, or alerts that may "
                "contain early disease-specific outbreak signals."
            ),
            example_sources=["news reports", "situation updates", "public alerts"],
        ),
    ]
    disease_name = intelligence.disease_standard_name
    location = spec.geography or "the requested geography"
    time_window = spec.time_window or "the requested time window"
    criteria = ScreeningCriteria(
        include_if_all_apply=[
            f"The source is about {disease_name} or a listed alias/pathogen term from the disease intelligence profile.",
            f"The source concerns human cases, deaths, hospitalizations, outbreaks, or surveillance data for {location} during {time_window}.",
            "The source contains or is likely to contain at least one requested target field or a supporting evidence quote.",
        ],
        exclude_if_any_apply=[
            "The source is only background, prevention-only, clinical overview, vector/animal surveillance, or laboratory-only content without human extractable data.",
            f"The source is unrelated to {disease_name}.",
            "The source lacks extractable counts, dates, locations, provenance, or evidence text for the requested task.",
            "The source is a duplicate of an already registered source.",
        ],
        uncertain_if_any_apply=[
            "Disease match is unclear from the title, snippet, or document text.",
            "Geography, time window, source provenance, or count semantics are unclear.",
            "The source may mix disease-specific counts with broader syndrome, pathogen-family, or all-condition counts.",
        ],
    )
    return SourceStrategy(source_categories=categories, screening_criteria=criteria)


def _profile_schema_generation_method(intelligence: DiseaseIntelligenceProfile) -> str:
    if intelligence.generation_method == "generic_deterministic_fallback":
        return "generic_fallback_profile_schema"
    return "disease_intelligence_generated_profile_schema"


def _profile_schema_summary(
    *,
    spec: CollectionSpec,
    profile: DiseaseProfile,
    schema: CollectionSchema,
    strategy: SourceStrategy,
    generation_method: str,
    warnings: list[str],
    task_acceptance_contract: dict | None = None,
) -> dict:
    return {
        "disease": spec.disease,
        "profile_generation_method": generation_method,
        "schema_generation_method": generation_method,
        "source_strategy_generation_method": generation_method,
        "disease_profile_standard_name": profile.disease_standard_name,
        "collection_schema_name": schema.schema_name,
        "collection_schema_version": schema.schema_version,
        "core_field_count": len(schema.core_fields),
        "target_field_count": len(spec.required_fields),
        "target_fields": list(spec.required_fields),
        "source_category_count": len(strategy.source_categories),
        "task_acceptance_contract_version": (
            (task_acceptance_contract or {}).get("contract_version")
        ),
        "warnings": list(warnings),
    }


def _infer_task_geography_scope(location: str | None) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", (location or "").lower()).strip()
    if not normalized or normalized in {"global", "world", "worldwide"}:
        return "global"
    if normalized in {
        "united states",
        "united states of america",
        "usa",
        "us",
        "u s",
        "u s a",
    }:
        return "country"
    return "subnational_or_local"


def _build_task_acceptance_contract(
    *,
    spec: CollectionSpec,
    intelligence: DiseaseIntelligenceProfile | None,
    schema: CollectionSchema,
) -> dict:
    standard_name = (
        intelligence.disease_standard_name
        if intelligence is not None
        else spec.disease
    )
    aliases = (
        _unique_preserve_order(
            [
                *intelligence.aliases,
                *intelligence.abbreviations,
                *intelligence.pathogen_terms,
                *intelligence.syndrome_terms,
            ]
        )
        if intelligence is not None
        else []
    )
    field_names = [field.name for field in schema.core_fields]
    return {
        "contract_version": "v1",
        "contract_scope": "task_compatible_collection_record",
        "disease": spec.disease,
        "disease_standard_name": standard_name,
        "accepted_disease_terms": _unique_preserve_order(
            [spec.disease, standard_name, *aliases]
        ),
        "excluded_disease_terms": (
            list(intelligence.exclusion_terms) if intelligence is not None else []
        ),
        "location": spec.geography,
        "target_geography_scope": _infer_task_geography_scope(spec.geography),
        "start_date": spec.start_date,
        "end_date": spec.end_date,
        "time_window": spec.time_window,
        "target_fields": list(spec.required_fields or spec.target_fields),
        "available_schema_fields": field_names,
        "record_acceptance_rules": [
            "must_match_task_disease_or_accepted_synonym",
            "must_match_task_location",
            "must_match_task_date_window_or_report_period",
            "must_have_interpretable_numeric_or_zero_metric",
            "must_include_source_provenance_and_evidence_quote",
        ],
        "context_or_quarantine_rules": [
            "national_or_broader_aggregate_without_target_location_fit",
            "reporting_period_substantially_broader_than_task_window",
            "source_context_only_without_extractable_target_record",
            "non_target_subtype_for_default_disease_scope",
        ],
        "semantic_decision_owner": (
            "LLM agents judge source/chunk/record task fit; deterministic "
            "rules enforce dates, schema, provenance, and final hard boundaries."
        ),
    }


def profile_and_schema_setup(state: DataCollectionState) -> dict:
    """Build active disease profile, collection schema, and source strategy."""

    spec = CollectionSpec(**(state.get("collection_spec") or {}))
    intelligence_for_contract: DiseaseIntelligenceProfile | None = None
    if _is_hantavirus_like(spec.disease) and not universal_queries_enabled():
        profile = DiseaseProfile(**load_hantavirus_profile())
        schema = CollectionSchema(**load_hantavirus_collection_schema())
        strategy = SourceStrategy(**load_source_strategy())
        generation_method = "legacy_hantavirus_profile_schema"
        warnings: list[str] = []
    else:
        intelligence = DiseaseIntelligenceProfile(**(state.get("disease_intelligence") or {}))
        intelligence_for_contract = intelligence
        warnings = _unique_preserve_order(
            [
                *_unsupported_target_field_warnings(
                    _unique_preserve_order(
                        [
                            *spec.required_fields,
                            *spec.target_fields,
                            *intelligence.extraction_priority_fields,
                        ]
                    )
                ),
            ]
        )
        generation_method = _profile_schema_generation_method(intelligence)
        profile = _build_generated_disease_profile(intelligence, spec)
        schema = _build_generated_collection_schema(
            intelligence,
            spec,
            generation_method,
            warnings,
        )
        strategy = _build_generated_source_strategy(intelligence, spec)

    task_acceptance_contract = _build_task_acceptance_contract(
        spec=spec,
        intelligence=intelligence_for_contract,
        schema=schema,
    )
    task_evidence_contract = build_task_evidence_contract(state)
    summary = _profile_schema_summary(
        spec=spec,
        profile=profile,
        schema=schema,
        strategy=strategy,
        generation_method=generation_method,
        warnings=warnings,
        task_acceptance_contract=task_acceptance_contract,
    )
    trace = append_trace(
        state,
        node_name="profile_and_schema_setup",
        message=(
            "Built active disease profile, collection schema, and source "
            f"strategy ({profile.disease_standard_name}, generation_method={generation_method})."
        ),
        metadata=summary,
    )
    return {
        "disease_profile": profile.model_dump(),
        "collection_schema": schema.model_dump(),
        "source_strategy": strategy.model_dump(),
        "screening_criteria": strategy.screening_criteria.model_dump(),
        "task_acceptance_contract": task_acceptance_contract,
        "task_evidence_contract": task_evidence_contract,
        "source_coverage_requirements": list(task_evidence_contract.get("requirements") or []),
        "profile_schema_summary": summary,
        "collection_trace": trace,
    }


def hantavirus_profile_and_schema_setup(state: DataCollectionState) -> dict:
    """Backward-compatible alias for profile_and_schema_setup."""

    return profile_and_schema_setup(state)


def _source_plan_target_fields(spec: dict, schema_dict: dict | None) -> list[str]:
    fields = _as_str_list(spec.get("required_fields")) or _as_str_list(
        spec.get("target_fields")
    )
    if fields:
        return fields
    schema_fields = []
    for field in (schema_dict or {}).get("core_fields") or []:
        if isinstance(field, dict) and field.get("name"):
            schema_fields.append(str(field["name"]))
    return schema_fields or _expected_fields_default()


def _source_plan_disease_terms(
    profile: DiseaseProfile,
    disease_intelligence: dict,
) -> list[str]:
    return _unique_preserve_order(
        [
            *(_as_str_list(disease_intelligence.get("suggested_query_terms"))[:6]),
            *(_as_str_list(disease_intelligence.get("aliases"))[:4]),
            *(_as_str_list(disease_intelligence.get("abbreviations"))[:4]),
            *(_as_str_list(disease_intelligence.get("pathogen_terms"))[:4]),
            *profile.include_terms[:6],
            *profile.syndrome_terms[:4],
            *profile.virus_terms[:4],
            profile.disease_standard_name,
        ]
    )


def _source_plan_location_terms(geography: str | None) -> list[str]:
    if not geography:
        return []
    if geography.lower() == "global":
        return ["global"]
    return [geography]


def _source_plan_time_terms(time_window: str | None) -> list[str]:
    if not time_window:
        return []
    years = re.findall(r"\d{4}", time_window)
    return _unique_preserve_order([time_window, *years])


def _plan_slug(value: str | None) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")
    return slug or "unspecified"


def _source_role_hint(source_type: str) -> str:
    mapping = {
        "official_public_health_agency": "collection",
        "international_organization_report": "validation",
        "peer_reviewed_literature": "context",
        "structured_database": "validation",
        "news_and_situation_report": "collection_support",
    }
    return mapping.get(source_type, "collection_support")


def _source_query_type(source_type: str) -> str:
    mapping = {
        "official_public_health_agency": "official_site",
        "international_organization_report": "domain_limited",
        "peer_reviewed_literature": "literature",
        "structured_database": "database",
        "news_and_situation_report": "news",
    }
    return mapping.get(source_type, "general_web")


def _source_provider_channel(source_type: str) -> str:
    mapping = {
        "official_public_health_agency": "official_site_search",
        "international_organization_report": "official_site_search",
        "peer_reviewed_literature": "literature_api",
        "structured_database": "database_search",
        "news_and_situation_report": "news_search",
    }
    return mapping.get(source_type, "web_search")


def _category_expected_fields(source_type: str, target_fields: list[str]) -> list[str]:
    if source_type == "peer_reviewed_literature":
        return _unique_preserve_order(
            [*target_fields, "case_definition", "evidence_quote"]
        )
    if source_type == "structured_database":
        return _unique_preserve_order(
            [*target_fields, "cases_confirmed", "deaths", "date_reported"]
        )
    if source_type == "news_and_situation_report":
        return _unique_preserve_order(
            [*target_fields, "subnational_location", "evidence_quote"]
        )
    return list(target_fields)


def _source_plan_objectives(
    disease_name: str,
    geography: str | None,
    time_window: str | None,
    localized_hints: dict | None = None,
) -> list[dict]:
    location = geography or "the requested geography"
    period = time_window or "the requested time window"
    objectives = [
        {
            "objective_id": "obj_collection_001",
            "objective": (
                f"Identify primary reporting sources for {disease_name} human "
                f"cases and deaths in {location} during {period}."
            ),
            "source_role_hint": "collection",
            "rationale": "Collection sources should contain extractable records.",
            "priority": 1,
        },
        {
            "objective_id": "obj_validation_001",
            "objective": (
                "Identify independent summary or structured sources to compare "
                "against collected records."
            ),
            "source_role_hint": "validation",
            "rationale": "Held-back or independent sources support masked validation.",
            "priority": 2,
        },
        {
            "objective_id": "obj_context_001",
            "objective": (
                "Identify context sources for disease terminology, case definitions, "
                "and count semantics."
            ),
            "source_role_hint": "context",
            "rationale": "Context sources help interpret extraction and normalization.",
            "priority": 3,
        },
        {
            "objective_id": "obj_review_001",
            "objective": (
                "Flag ambiguous source roles, validation leakage risk, or unclear "
                "count semantics for human review."
            ),
            "source_role_hint": "human_review",
            "rationale": "Uncertain sources should not silently enter extraction.",
            "priority": 4,
        },
    ]
    if localized_hints and localized_hints.get("enabled"):
        objectives.insert(
            0,
            {
                "objective_id": "obj_localized_official_001",
                "objective": (
                    "Prioritize localized Shanghai / China official public-health "
                    "sources and Chinese HFRS terminology before broad web or news "
                    "queries."
                ),
                "source_role_hint": "collection",
                "rationale": (
                    "HFRS/hantavirus reporting in China may appear in Chinese "
                    "official notifiable infectious disease sources."
                ),
                "priority": 1,
            },
        )
    return objectives


def _apply_localized_category_hints(
    categories: list[dict],
    localized_hints: dict | None,
) -> list[dict]:
    if not localized_hints or not localized_hints.get("enabled"):
        return categories
    updated = []
    for category in categories:
        category_copy = dict(category)
        if category_copy.get("source_type") == "official_public_health_agency":
            category_copy["risk_notes"] = _unique_preserve_order(
                [
                    *(category_copy.get("risk_notes") or []),
                    "localized_official_sources_prioritized_for_shanghai_china",
                    "chinese_hfrs_terms_required_for_source_discovery",
                ]
            )
            category_copy["why_relevant"] = (
                f"{category_copy.get('why_relevant') or ''} "
                "For Shanghai/HFRS tasks, local and national Chinese official "
                "public-health sources are the highest-priority collection sources."
            ).strip()
        updated.append(category_copy)
    return updated


def _planned_source_categories(
    strategy: SourceStrategy,
    target_fields: list[str],
) -> list[dict]:
    categories = []
    for index, category in enumerate(strategy.source_categories, start=1):
        source_type = category.source_type
        if source_type not in _EXECUTABLE_SOURCE_TYPES:
            continue
        categories.append(
            {
                "source_category_id": f"cat_{index:03d}_{source_type}",
                "source_type": source_type,
                "role_hint": _source_role_hint(source_type),
                "priority": category.priority,
                "expected_fields": _category_expected_fields(
                    source_type, target_fields
                ),
                "why_relevant": category.description,
                "risk_notes": [
                    "planned_only_not_executed",
                    "requires_later_search_result_screening",
                ],
            }
        )
    return categories


def _query_location_suffix(location_terms: list[str]) -> str:
    return f" {' '.join(location_terms)}" if location_terms else ""


def _query_time_suffix(time_terms: list[str]) -> str:
    return f" {' '.join(time_terms[:2])}" if time_terms else ""


def _planned_query_text(
    source_type: str,
    term: str,
    location_terms: list[str],
    time_terms: list[str],
) -> str:
    location_suffix = _query_location_suffix(location_terms)
    time_suffix = _query_time_suffix(time_terms)
    if source_type == "official_public_health_agency":
        return f'"{term}" cases deaths public health{location_suffix}{time_suffix}'.strip()
    if source_type == "international_organization_report":
        return f'"{term}" outbreak report surveillance{location_suffix}{time_suffix}'.strip()
    if source_type == "peer_reviewed_literature":
        return f'"{term}" epidemiology cases deaths{location_suffix}{time_suffix}'.strip()
    if source_type == "structured_database":
        return f'"{term}" surveillance dataset cases deaths{location_suffix}{time_suffix}'.strip()
    return f'"{term}" outbreak report cases deaths{location_suffix}{time_suffix}'.strip()


def _localized_planned_query_dicts(
    localized_hints: dict | None,
    *,
    start_index: int = 1,
) -> list[dict]:
    if not localized_hints or not localized_hints.get("enabled"):
        return []
    query_dicts: list[dict] = []
    for spec in localized_hints.get("planned_query_specs") or []:
        if not isinstance(spec, dict):
            continue
        query = str(spec.get("query") or "").strip()
        if not query:
            continue
        query_dicts.append(
            {
                "query_id": f"q_exec_{start_index + len(query_dicts):03d}",
                "query": query,
                "query_type": spec.get("query_type") or "general_web",
                "provider_channel": spec.get("provider_channel") or "web_search",
                "source_type": spec.get("source_type")
                or "official_public_health_agency",
                "role_hint": spec.get("role_hint") or "collection",
                "priority": int(spec.get("priority") or 1),
                "expected_fields": list(
                    spec.get("expected_fields") or _expected_fields_default()
                ),
                "disease_terms_used": _as_str_list(spec.get("disease_terms_used")),
                "location_terms_used": _as_str_list(spec.get("location_terms_used")),
                "time_terms_used": _as_str_list(spec.get("time_terms_used")),
                "query_language": spec.get("query_language"),
                "jurisdiction_hint": spec.get("jurisdiction_hint"),
                "official_domain_hint": spec.get("official_domain_hint"),
                "localized_source_hint": bool(spec.get("localized_source_hint")),
                "source_priority_reason": spec.get("source_priority_reason"),
                "rationale": spec.get("rationale")
                or "Localized official source planning query.",
                "execution_status": "planned_not_executed",
            }
        )
    return query_dicts


def _source_discovery_template_query_metadata(query: str) -> dict:
    lowered = query.lower()
    if any(
        token in lowered
        for token in [
            "ministry of health",
            "national public health",
            "health ministry",
            "public health agency",
        ]
    ):
        return {
            "source_type": "official_public_health_agency",
            "provider_channel": "official_site_search",
            "query_type": "official_site",
        }
    if any(
        token in lowered
        for token in [
            "pathoplexus",
            "genomic",
            "sequence",
            "line list",
            "surveillance dataset",
            "outbreak data",
        ]
    ):
        return {
            "source_type": "structured_database",
            "provider_channel": "database_search",
            "query_type": "database",
        }
    if any(
        token in lowered
        for token in [
            "nejm",
            "eurosurveillance",
            "case report",
            "case series",
            "peer-reviewed",
            "journal",
        ]
    ):
        return {
            "source_type": "peer_reviewed_literature",
            "provider_channel": "literature_api",
            "query_type": "literature",
        }
    if any(
        token in lowered
        for token in [
            "who disease outbreak news",
            "who ",
            "paho",
            "ecdc",
        ]
    ):
        return {
            "source_type": "international_organization_report",
            "provider_channel": "official_site_search",
            "query_type": "official_site",
        }
    if any(token in lowered for token in ["cdc", " han", "health department"]):
        return {
            "source_type": "official_public_health_agency",
            "provider_channel": "official_site_search",
            "query_type": "official_site",
        }
    if any(token in lowered for token in ["pubmed", "literature", "journal"]):
        return {
            "source_type": "peer_reviewed_literature",
            "provider_channel": "literature_api",
            "query_type": "literature",
        }
    if any(
        token in lowered
        for token in [
            "vessel-associated",
            "travel-associated",
            "cruise ship",
            "multi-country",
            "outbreak",
        ]
    ):
        return {
            "source_type": "international_organization_report",
            "provider_channel": "web_search",
            "query_type": "news",
        }
    return {
        "source_type": "news_and_situation_report",
        "provider_channel": "web_search",
        "query_type": "general_web",
    }


def _global_outbreak_source_class_queries(
    disease_intelligence: dict | None,
    *,
    disease_terms: list[str],
    location_terms: list[str],
    time_terms: list[str],
    target_fields: list[str],
    seen: set[str],
    start_index: int = 1,
) -> list[dict]:
    location_text = " ".join(location_terms).strip().lower()
    intelligence_text = " ".join(
        str(value)
        for key in (
            "source_discovery_query_templates",
            "official_source_terms",
            "surveillance_terms",
            "disambiguation_risks",
        )
        for value in ((disease_intelligence or {}).get(key) or [])
    ).lower()
    outbreak_like = (
        any(
            token in location_text
            for token in ("global", "worldwide", "multi-country", "multicountry")
        )
        or any(
            token in intelligence_text
            for token in ("outbreak", "cluster", "travel-associated", "cruise")
        )
    )
    if not outbreak_like:
        return []

    primary_terms = disease_terms[:2] or [_DEFAULT_DISEASE]
    time_suffix = _query_time_suffix(time_terms)
    geo_suffix = _query_location_suffix(location_terms)
    specs: list[tuple[str, str, str]] = []
    for term in primary_terms:
        specs.extend(
            [
                (
                    f'"{term}" "ministry of health" "confirmed cases"{time_suffix}',
                    "official_public_health_agency",
                    "Search national ministry reporting, not only international aggregate reports.",
                ),
                (
                    f'"{term}" "national public health" outbreak cases deaths{time_suffix}',
                    "official_public_health_agency",
                    "Search national public-health authority updates for outbreak case counts.",
                ),
                (
                    f'"{term}" genomic epidemiology outbreak data{time_suffix}',
                    "structured_database",
                    "Search structured genomic or outbreak databases for case-linked evidence.",
                ),
                (
                    f'"{term}" site:pathoplexus.org sequence cases{time_suffix}',
                    "structured_database",
                    "Search a pathogen genomic database domain without using benchmark rows.",
                ),
                (
                    f'"{term}" site:eurosurveillance.org outbreak{time_suffix}',
                    "peer_reviewed_literature",
                    "Search public-health surveillance journal sources for outbreak reports.",
                ),
                (
                    f'"{term}" site:nejm.org outbreak{time_suffix}',
                    "peer_reviewed_literature",
                    "Search medical journal case-report sources for outbreak evidence.",
                ),
                (
                    f'"{term}" line list case series cases deaths{time_suffix}',
                    "structured_database",
                    "Search line-list or case-series data products for outbreak records.",
                ),
                (
                    f'"{term}" "case report" outbreak{time_suffix}',
                    "peer_reviewed_literature",
                    "Search peer-reviewed case reports for individual or clustered cases.",
                ),
                (
                    f'"{term}" "case series" outbreak{time_suffix}',
                    "peer_reviewed_literature",
                    "Search peer-reviewed case series before relying on media summaries.",
                ),
                (
                    f'"{term}" outbreak report cases deaths{geo_suffix}{time_suffix}',
                    "news_and_situation_report",
                    "Retain reputable media and situation reports as supporting evidence.",
                ),
            ]
        )

    query_dicts: list[dict] = []
    for query, source_type, rationale in specs:
        query = re.sub(r"\s+", " ", query).strip()
        if not query or query in seen:
            continue
        seen.add(query)
        query_dicts.append(
            {
                "query_id": f"q_exec_{start_index + len(query_dicts):03d}",
                "query": query,
                "query_type": _source_query_type(source_type),
                "provider_channel": _source_provider_channel(source_type),
                "source_type": source_type,
                "role_hint": _source_role_hint(source_type),
                "priority": 1 if source_type != "news_and_situation_report" else 4,
                "expected_fields": list(target_fields or _expected_fields_default()),
                "disease_terms_used": [str(query).split('"')[1] if '"' in query else query],
                "location_terms_used": list(location_terms),
                "time_terms_used": list(time_terms),
                "query_language": "en",
                "jurisdiction_hint": None,
                "official_domain_hint": None,
                "localized_source_hint": False,
                "source_priority_reason": "global_outbreak_authority_source_class",
                "rationale": rationale,
                "execution_status": "planned_not_executed",
            }
        )
    return query_dicts


def _source_discovery_template_queries(
    disease_intelligence: dict | None,
    *,
    disease_terms: list[str],
    location_terms: list[str],
    time_terms: list[str],
    target_fields: list[str],
    seen: set[str],
    start_index: int = 1,
) -> list[dict]:
    if not disease_intelligence:
        return []
    templates = _as_str_list(
        disease_intelligence.get("source_discovery_query_templates")
    )
    if not templates:
        return []

    time_window = " ".join(time_terms[:2]).strip()
    location = " ".join(location_terms).strip()
    query_dicts: list[dict] = []
    for template in templates:
        query = (
            template.replace("{location}", location)
            .replace("{time_window}", time_window)
            .strip()
        )
        query = re.sub(r"\s+", " ", query)
        if not query or query in seen:
            continue
        seen.add(query)
        metadata = _source_discovery_template_query_metadata(query)
        query_dicts.append(
            {
                "query_id": f"q_exec_{start_index + len(query_dicts):03d}",
                "query": query,
                "query_type": metadata["query_type"],
                "provider_channel": metadata["provider_channel"],
                "source_type": metadata["source_type"],
                "role_hint": "collection",
                "priority": 1,
                "expected_fields": list(target_fields or _expected_fields_default()),
                "disease_terms_used": list(disease_terms[:4]),
                "location_terms_used": list(location_terms),
                "time_terms_used": list(time_terms),
                "query_language": "en",
                "jurisdiction_hint": None,
                "official_domain_hint": None,
                "localized_source_hint": False,
                "source_priority_reason": (
                    "Disease-intelligence source-class template for official "
                    "or outbreak source discovery."
                ),
                "rationale": (
                    "Disease intelligence requested source-class-aware search "
                    "coverage before generic category expansion."
                ),
                "execution_status": "planned_not_executed",
            }
        )
    return query_dicts


def _planned_queries(
    categories: list[dict],
    disease_terms: list[str],
    location_terms: list[str],
    time_terms: list[str],
    localized_hints: dict | None = None,
    disease_intelligence: dict | None = None,
    target_fields: list[str] | None = None,
) -> list[dict]:
    queries = _localized_planned_query_dicts(localized_hints)
    seen: set[str] = set()
    for query in queries:
        seen.add(str(query.get("query") or ""))
    queries.extend(
        _source_discovery_template_queries(
            disease_intelligence,
            disease_terms=disease_terms,
            location_terms=location_terms,
            time_terms=time_terms,
            target_fields=list(target_fields or _expected_fields_default()),
            seen=seen,
            start_index=len(queries) + 1,
        )
    )
    queries.extend(
        _global_outbreak_source_class_queries(
            disease_intelligence,
            disease_terms=disease_terms,
            location_terms=location_terms,
            time_terms=time_terms,
            target_fields=list(target_fields or _expected_fields_default()),
            seen=seen,
            start_index=len(queries) + 1,
        )
    )
    primary_terms = disease_terms[:3] or [_DEFAULT_DISEASE]
    for category in categories:
        source_type = category["source_type"]
        for term in primary_terms[:2]:
            query = _planned_query_text(
                source_type, term, location_terms, time_terms
            )
            if query in seen:
                continue
            seen.add(query)
            queries.append(
                {
                    "query_id": f"q_exec_{len(queries) + 1:03d}",
                    "query": query,
                    "query_type": _source_query_type(source_type),
                    "provider_channel": _source_provider_channel(source_type),
                    "source_type": source_type,
                    "role_hint": category["role_hint"],
                    "priority": category["priority"],
                    "expected_fields": list(category["expected_fields"]),
                    "disease_terms_used": [term],
                    "location_terms_used": list(location_terms),
                    "time_terms_used": list(time_terms),
                    "query_language": "en",
                    "jurisdiction_hint": None,
                    "official_domain_hint": None,
                    "localized_source_hint": False,
                    "source_priority_reason": None,
                    "rationale": (
                        "Executable search query planned for later source "
                        f"discovery against {source_type}; not executed in Stage 4."
                    ),
                    "execution_status": "planned_not_executed",
                }
            )
    return queries


def _source_planning_risks() -> list[dict]:
    return [
        {
            "risk_id": "risk_semantic_leakage_001",
            "risk": (
                "A source intended for validation may leak into collection if roles "
                "are not enforced during later discovery."
            ),
            "severity": "high",
            "applies_to": ["validation", "collection"],
            "mitigation": "Keep validation-reserved sources separate until evaluation.",
            "human_review_trigger": True,
        },
        {
            "risk_id": "risk_count_semantics_001",
            "risk": (
                "Cumulative, annual, newly reported, and suspected counts may be "
                "mixed across source categories."
            ),
            "severity": "medium",
            "applies_to": ["collection", "validation", "context"],
            "mitigation": "Require evidence quotes and count-semantics fields.",
            "human_review_trigger": True,
        },
    ]


def _deterministic_executable_source_plan(
    *,
    spec: dict,
    profile: DiseaseProfile,
    strategy: SourceStrategy,
    schema_dict: dict | None,
    disease_intelligence: dict,
    localized_hints: dict | None = None,
    generation_method: str = "deterministic_executable_source_plan",
    llm_enabled: bool = False,
    extra_warnings: list[str] | None = None,
) -> ExecutableSourcePlan:
    target_fields = _source_plan_target_fields(spec, schema_dict)
    disease_name = (
        disease_intelligence.get("disease_standard_name")
        or profile.disease_standard_name
        or spec.get("disease")
        or _DEFAULT_DISEASE
    )
    geography = spec.get("geography")
    time_window = spec.get("time_window")
    if universal_queries_enabled():
        # Preserve the task disease; curated subtype/event templates are not
        # evidence for the current session. LLM queries still face the guard.
        disease_name = str(spec.get("disease") or disease_name)
        disease_terms = [disease_name]
        disease_intelligence = {}
        localized_hints = None
    else:
        disease_terms = _source_plan_disease_terms(profile, disease_intelligence)
    location_terms = _source_plan_location_terms(geography)
    time_terms = _source_plan_time_terms(time_window)
    categories = _apply_localized_category_hints(
        _planned_source_categories(strategy, target_fields),
        localized_hints,
    )
    planned_queries = _planned_queries(
        categories,
        disease_terms,
        location_terms,
        time_terms,
        localized_hints,
        disease_intelligence,
        target_fields,
    )
    warnings = [
        "source_plan_created_not_executed",
        *(extra_warnings or []),
    ]
    if localized_hints and localized_hints.get("enabled"):
        warnings.append("localized_source_planning_hints_applied")
    plan = ExecutableSourcePlan(
        plan_id=(
            "exec_source_plan_"
            f"{_plan_slug(disease_name)}_{_plan_slug(geography)}_"
            f"{_plan_slug(time_window)}"
        ),
        disease=disease_name,
        location=geography,
        time_window=time_window,
        target_fields=target_fields,
        generation_method=generation_method,
        llm_enabled=llm_enabled,
        execution_status="planned_not_executed",
        warnings=_unique_preserve_order(warnings),
        source_discovery_objectives=_source_plan_objectives(
            disease_name,
            geography,
            time_window,
            localized_hints,
        ),
        planned_source_categories=categories,
        planned_queries=planned_queries,
        source_planning_risks=_source_planning_risks(),
    )
    return _sanitize_executable_source_plan(plan)


def _sanitize_planned_query_text(query: str) -> str:
    cleaned = _PLANNED_QUERY_URL_PATTERN.sub("", query or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _fallback_query_text(plan_dict: dict, query: dict) -> str:
    parts = _unique_preserve_order(
        [
            *(_as_str_list(query.get("disease_terms_used"))),
            plan_dict.get("disease"),
            *(_as_str_list(query.get("location_terms_used"))),
            plan_dict.get("location"),
            *(_as_str_list(query.get("time_terms_used"))[:1]),
            "cases deaths",
        ]
    )
    return " ".join(parts)


def _sanitize_executable_source_plan(plan: ExecutableSourcePlan) -> ExecutableSourcePlan:
    plan_dict = plan.model_dump()
    warnings = list(plan_dict.get("warnings") or [])
    plan_dict["execution_status"] = "planned_not_executed"
    for index, query in enumerate(plan_dict.get("planned_queries") or [], start=1):
        query["execution_status"] = "planned_not_executed"
        query["query_id"] = query.get("query_id") or f"q_exec_{index:03d}"
        original = str(query.get("query") or "")
        sanitized = _sanitize_planned_query_text(original)
        if sanitized != original:
            warnings.append(f"llm_planned_query_url_sanitized:{query['query_id']}")
        if not sanitized:
            sanitized = _fallback_query_text(plan_dict, query)
            warnings.append(f"empty_planned_query_replaced:{query['query_id']}")
        query["query"] = sanitized
    plan_dict["warnings"] = _unique_preserve_order(warnings)
    return ExecutableSourcePlan(**plan_dict)


def _is_localized_planned_query(query: dict) -> bool:
    if bool(query.get("localized_source_hint")):
        return True
    rationale = str(query.get("rationale") or "").lower()
    if "localized official source planning" in rationale:
        return True
    query_text = str(query.get("query") or "")
    return any(
        token in query_text
        for token in (
            "肾综合征出血热",
            "汉坦病毒",
            "流行性出血热",
            "wsjkw.sh.gov.cn",
            "shcdc.sh.cn",
        )
    )


def _prepend_missing_localized_queries(
    base_queries: list[dict],
    current_queries: list[dict],
) -> list[dict]:
    current = [dict(query) for query in current_queries if isinstance(query, dict)]
    localized = [
        dict(query) for query in base_queries if isinstance(query, dict)
        and _is_localized_planned_query(query)
    ]
    if not localized:
        return current

    seen = {
        _sanitize_planned_query_text(str(query.get("query") or "")).casefold()
        for query in current
    }
    additions = []
    for query in localized:
        key = _sanitize_planned_query_text(str(query.get("query") or "")).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        additions.append(query)
    return [*additions, *current]


def _build_executable_source_plan_prompt(
    *,
    user_request: str,
    spec: dict,
    profile: dict,
    strategy: dict,
    schema_dict: dict,
    deterministic_plan: ExecutableSourcePlan,
    task_acceptance_contract: dict | None = None,
) -> tuple[str, str]:
    system_prompt = (
        "You create executable source discovery plans for the data collection "
        "workflow. Return only a JSON object matching the ExecutableSourcePlan "
        "schema. Do not search the web, fetch URLs, claim that sources were "
        "found, or add discovered source URLs. Every planned query must have "
        "execution_status='planned_not_executed'."
    )
    payload = {
        "user_request": user_request,
        "collection_spec": {
            "disease": spec.get("disease"),
            "geography": spec.get("geography"),
            "time_window": spec.get("time_window"),
            "target_fields": _source_plan_target_fields(spec, schema_dict),
        },
        "task_acceptance_contract": task_acceptance_contract or {},
        "disease_profile": {
            "disease_standard_name": profile.get("disease_standard_name"),
            "include_terms": _as_str_list(profile.get("include_terms")),
            "syndrome_terms": _as_str_list(profile.get("syndrome_terms")),
            "virus_terms": _as_str_list(profile.get("virus_terms")),
        },
        "source_strategy": {
            "source_categories": [
                {
                    "source_type": item.get("source_type"),
                    "priority": item.get("priority"),
                }
                for item in strategy.get("source_categories") or []
                if isinstance(item, dict)
            ]
        },
        "allowed_generation_methods": [
            "llm_executable_source_plan",
            "deterministic_executable_source_plan",
            "llm_failed_deterministic_fallback",
            "invalid_llm_output_deterministic_fallback",
        ],
        "allowed_execution_status": ["planned_not_executed"],
        "allowed_source_types": list(_EXECUTABLE_SOURCE_TYPES),
        "allowed_role_hints": [
            "collection",
            "validation",
            "context",
            "collection_support",
            "human_review",
        ],
        "reference_deterministic_plan_shape": deterministic_plan.model_dump(),
    }
    if universal_queries_enabled():
        core_fields = ("query_id", "query", "provider_channel", "source_type", "role_hint", "rationale")
        routing_fields = ("query_type", "priority", "query_language", "jurisdiction_hint",
                          "official_domain_hint", "localized_source_hint")
        reference = deterministic_plan.model_dump()
        payload["reference_deterministic_plan_shape"] = {
            **{key: reference[key] for key in ("plan_id", "disease", "generation_method")},
            "planned_queries": [
                {key: query[key] for key in (*core_fields, *routing_fields)
                 if key in query and (key in core_fields or (query[key] is not None and query[key] != "" and query[key] is not False))}
                for query in reference["planned_queries"]
            ],
        }
        payload["response_contract"] = {
            "top_level_fields": ["plan_id", "disease", "generation_method", "planned_queries"],
            "required_query_fields": list(core_fields),
            "routing_fields_when_needed": list(routing_fields),
            "omit_repeated_query_fields": ["expected_fields", "disease_terms_used", "location_terms_used", "time_terms_used"],
            "omit_top_level_sections": ["source_discovery_objectives", "planned_source_categories", "source_planning_risks"],
            "preserve_query_breadth": True,
            "rationale_style": "one short sentence",
        }
        system_prompt += (
            " Follow response_contract: return only the required plan fields and concise planned_queries, "
            "preserving useful distinct queries across the reference source families and necessary routing hints. "
            "Do not reduce query breadth to save tokens. Omit repeated optional query metadata and the listed "
            "top-level sections; the workflow retains its deterministic objectives, categories, and risks. "
            "Use one short sentence per rationale and compact JSON without indentation."
        )
    return system_prompt, json.dumps(payload, ensure_ascii=True, indent=2)


def _merge_llm_executable_plan(
    raw: dict,
    deterministic_plan: ExecutableSourcePlan,
) -> ExecutableSourcePlan:
    base = deterministic_plan.model_dump()
    raw_dict = dict(raw or {})
    structured_output_mode = raw_dict.pop(
        "_structured_output_mode",
        raw_dict.get("structured_output_mode"),
    )
    merged = {**base, **raw_dict}
    merged["plan_id"] = merged.get("plan_id") or base["plan_id"]
    merged["disease"] = merged.get("disease") or base["disease"]
    merged["location"] = merged.get("location") or base.get("location")
    merged["time_window"] = merged.get("time_window") or base.get("time_window")
    merged["target_fields"] = merged.get("target_fields") or base["target_fields"]
    merged["generation_method"] = "llm_executable_source_plan"
    merged["llm_enabled"] = True
    merged["execution_status"] = "planned_not_executed"
    merged["source_discovery_objectives"] = (
        merged.get("source_discovery_objectives")
        or base["source_discovery_objectives"]
    )
    merged["planned_source_categories"] = (
        merged.get("planned_source_categories") or base["planned_source_categories"]
    )
    merged["planned_queries"] = _prepend_missing_localized_queries(
        base.get("planned_queries") or [],
        merged.get("planned_queries") or base["planned_queries"],
    )
    merged["source_planning_risks"] = (
        merged.get("source_planning_risks") or base["source_planning_risks"]
    )
    merged["warnings"] = _unique_preserve_order(
        [
            "source_plan_created_not_executed",
            *(
                ["localized_source_planning_hints_preserved_after_llm_plan"]
                if any(
                    _is_localized_planned_query(query)
                    for query in base.get("planned_queries") or []
                )
                else []
            ),
            *(_as_str_list(merged.get("warnings"))),
        ]
    )
    merged["structured_output_mode"] = structured_output_mode
    return _sanitize_executable_source_plan(ExecutableSourcePlan(**merged))


def _executable_source_plan_summary(plan: ExecutableSourcePlan) -> dict:
    plan_dict = plan.model_dump()
    queries = plan_dict.get("planned_queries") or []
    categories = plan_dict.get("planned_source_categories") or []
    return {
        "plan_id": plan.plan_id,
        "disease": plan.disease,
        "location": plan.location,
        "time_window": plan.time_window,
        "target_field_count": len(plan.target_fields),
        "generation_method": plan.generation_method,
        "llm_enabled": plan.llm_enabled,
        "execution_status": plan.execution_status,
        "objective_count": len(plan.source_discovery_objectives),
        "planned_source_category_count": len(categories),
        "planned_query_count": len(queries),
        "risk_count": len(plan.source_planning_risks),
        "role_hint_counts": dict(Counter(q.get("role_hint") for q in queries)),
        "source_type_counts": dict(Counter(q.get("source_type") for q in queries)),
        "provider_channel_counts": dict(
            Counter(q.get("provider_channel") for q in queries)
        ),
        "warnings": list(plan.warnings),
    }


def _localized_source_planning_summary(
    plan: ExecutableSourcePlan,
    localized_hints: dict | None,
) -> dict:
    summary = localized_source_planning_public_summary(localized_hints)
    plan_queries = [
        query.model_dump()
        for query in plan.planned_queries
        if _is_localized_planned_query(query.model_dump())
    ]
    official_domains = _unique_preserve_order(
        [
            str(query.get("official_domain_hint") or "")
            for query in plan_queries
            if query.get("official_domain_hint")
        ]
        + _as_str_list(summary.get("official_domain_hints"))
    )
    summary.update(
        {
            "localized_query_count": len(plan_queries),
            "official_domain_hint_count": len(official_domains),
            "official_domain_hints": official_domains,
            "planned_query_ids": [
                str(query.get("query_id") or "") for query in plan_queries
            ],
            "example_queries": [
                str(query.get("query") or "") for query in plan_queries[:5]
            ],
        }
    )
    if plan_queries and not summary.get("enabled"):
        summary["enabled"] = True
    return summary


def _source_planning_agent_summary(
    *,
    plan: ExecutableSourcePlan,
    planning_enabled: bool,
    status: str,
    failure_type: str | None = None,
    failure_message: str | None = None,
) -> dict:
    summary = {
        "llm_source_planning_enabled": planning_enabled,
        "status": status,
        "generation_method": plan.generation_method,
        "execution_status": plan.execution_status,
        "agent_name": "source_planning_agent",
        "agent_version": "0.4",
        "agent_query_count": len(plan.planned_queries),
        "agent_query_added_count": 0,
        "agent_candidate_hint_count": 0,
        "human_review_recommended": any(
            risk.human_review_trigger for risk in plan.source_planning_risks
        ),
        "warnings": list(plan.warnings),
        "structured_output_mode": plan.structured_output_mode,
    }
    if failure_type:
        summary["failure_type"] = failure_type
    if failure_message:
        summary["failure_message"] = failure_message
    return summary


def executable_source_planning(state: DataCollectionState) -> dict:
    """Create an auditable executable source discovery plan without executing it."""

    spec_dict = state.get("collection_spec") or {}
    profile_dict = state.get("disease_profile") or load_hantavirus_profile()
    strategy_dict = state.get("source_strategy") or load_source_strategy()
    schema_dict = state.get("collection_schema") or load_hantavirus_collection_schema()
    disease_intelligence = state.get("disease_intelligence") or {}
    profile = DiseaseProfile(**profile_dict)
    strategy = SourceStrategy(**strategy_dict)
    planning_enabled = llm_clients.llm_source_planning_enabled()
    localized_hints = {} if universal_queries_enabled() else build_localized_source_planning_hints(
        structured_task=state.get("structured_task") or {},
        collection_spec=spec_dict,
        disease_intelligence=disease_intelligence,
        preferred_source_categories=spec_dict.get("source_priority"),
    )

    deterministic_plan = _deterministic_executable_source_plan(
        spec=spec_dict,
        profile=profile,
        strategy=strategy,
        schema_dict=schema_dict,
        disease_intelligence=disease_intelligence,
        localized_hints=localized_hints,
        llm_enabled=False,
    )
    plan = deterministic_plan
    status = "deterministic_plan_created"
    failure_type: str | None = None
    failure_message: str | None = None

    if planning_enabled:
        system_prompt, user_prompt = _build_executable_source_plan_prompt(
            user_request=state.get("user_request", "") or "",
            spec=spec_dict,
            profile=profile_dict,
            strategy=strategy_dict,
            schema_dict=schema_dict,
            deterministic_plan=deterministic_plan,
            task_acceptance_contract=state.get("task_acceptance_contract") or {},
        )
        try:
            raw = llm_clients.run_pydantic_structured_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_model=ExecutableSourcePlan,
                temperature=0.0,
            )
            plan = _merge_llm_executable_plan(raw, deterministic_plan)
            status = "success"
        except Exception as exc:  # noqa: BLE001 - fallback keeps offline path usable
            failure_type = type(exc).__name__
            failure_message = str(exc)
            fallback_method = "llm_failed_deterministic_fallback"
            plan = _deterministic_executable_source_plan(
                spec=spec_dict,
                profile=profile,
                strategy=strategy,
                schema_dict=schema_dict,
                disease_intelligence=disease_intelligence,
                localized_hints=localized_hints,
                generation_method=fallback_method,
                llm_enabled=True,
                extra_warnings=[
                    "llm_source_planning_failed_deterministic_fallback_used",
                    f"llm_source_planning_failure_type:{failure_type}",
                ],
            )
            status = "failed_deterministic_fallback"

    plan_summary = _executable_source_plan_summary(plan)
    localized_summary = _localized_source_planning_summary(plan, localized_hints)
    plan_summary["localized_source_planning"] = localized_summary
    source_planning_summary = _source_planning_agent_summary(
        plan=plan,
        planning_enabled=planning_enabled,
        status=status,
        failure_type=failure_type,
        failure_message=failure_message,
    )
    source_planning_summary["localized_source_planning_summary"] = localized_summary
    trace = append_trace(
        state,
        node_name="executable_source_planning",
        message=(
            f"Built executable source plan with {len(plan.planned_queries)} "
            "planned queries; no searches were executed."
        ),
        metadata=plan_summary,
    )
    return {
        "agentic_source_plan": plan.model_dump(),
        "evidence_strategy_plan": plan.model_dump(),
        "executable_source_plan_summary": plan_summary,
        "localized_source_planning_summary": localized_summary,
        "source_planning_agent_summary": source_planning_summary,
        "collection_trace": trace,
    }


def _priority_for(source_type: str, strategy: SourceStrategy) -> int:
    for category in strategy.source_categories:
        if category.source_type == source_type:
            return category.priority
    return 99


def _expected_fields_default() -> list[str]:
    return [
        "cases",
        "deaths",
        "date",
        "location",
        "source_url",
        "source_type",
        "evidence_quote",
    ]


def _add_query(
    inventory: list[SearchQuery],
    seen_queries: set[str],
    next_index: dict[str, int],
    bucket_key: str,
    query_string: str,
    source_type: str,
    priority: int,
    rationale: str,
    expected_fields: list[str],
) -> None:
    if query_string in seen_queries:
        return
    seen_queries.add(query_string)
    next_index[bucket_key] = next_index.get(bucket_key, 0) + 1
    query_id = f"q_{bucket_key}_{next_index[bucket_key]:03d}"
    inventory.append(
        SearchQuery(
            query_id=query_id,
            query=query_string,
            source_type=source_type,
            priority=priority,
            rationale=rationale,
            expected_fields=expected_fields,
        )
    )


def _append_executable_plan_queries(
    inventory_dicts: list[dict],
    seen_queries: set[str],
    agentic_source_plan: dict | None,
) -> int:
    if not agentic_source_plan:
        return 0
    planned = agentic_source_plan.get("planned_queries") or []
    if not isinstance(planned, list):
        raise ValueError("agentic_source_plan.planned_queries must be a list")

    added_count = 0
    for index, item in enumerate(planned, start=1):
        if not isinstance(item, dict):
            raise ValueError("planned source query must be an object")
        planned_query = PlannedSearchQuery(**item)
        query = _sanitize_planned_query_text(planned_query.query)
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)
        added_count += 1
        inventory_dicts.append(
            {
                "query_id": planned_query.query_id or f"q_exec_{index:03d}",
                "query": query,
                "source_type": planned_query.source_type,
                "priority": planned_query.priority,
                "rationale": planned_query.rationale,
                "expected_fields": list(planned_query.expected_fields),
                "query_source": "executable_source_plan",
                "discovery_method": "executable_source_plan",
                "query_type": planned_query.query_type,
                "provider_channel": planned_query.provider_channel,
                "role_hint": planned_query.role_hint,
                "execution_status": planned_query.execution_status,
                "disease_terms_used": list(planned_query.disease_terms_used),
                "location_terms_used": list(planned_query.location_terms_used),
                "time_terms_used": list(planned_query.time_terms_used),
                "query_language": planned_query.query_language,
                "jurisdiction_hint": planned_query.jurisdiction_hint,
                "official_domain_hint": planned_query.official_domain_hint,
                "localized_source_hint": planned_query.localized_source_hint,
                "source_priority_reason": planned_query.source_priority_reason,
            }
        )
    return added_count


def _state_collection_mode(state: DataCollectionState) -> str:
    structured_task = state.get("structured_task") or {}
    collection_spec = state.get("collection_spec") or {}
    workflow = state.get("workflow") or {}
    return str(
        structured_task.get("collection_mode")
        or collection_spec.get("collection_mode")
        or workflow.get("collection_mode")
        or state.get("collection_mode")
        or ""
    ).strip()


def _direct_collection_llm_plan_primary(state: DataCollectionState) -> bool:
    if _state_collection_mode(state) != "direct_collection" and not universal_queries_enabled():
        return False
    plan = state.get("agentic_source_plan") or {}
    planned = plan.get("planned_queries") if isinstance(plan, dict) else None
    return bool(planned)


def _search_query_set_from_inventory(inventory_dicts: list[dict]) -> SearchQuerySet:
    official_source_queries = [
        item["query"]
        for item in inventory_dicts
        if item.get("source_type") == "official_public_health_agency"
    ]
    literature_queries = [
        item["query"]
        for item in inventory_dicts
        if item.get("source_type") == "peer_reviewed_literature"
    ]
    news_and_report_queries = [
        item["query"]
        for item in inventory_dicts
        if item.get("source_type")
        in (
            "news_and_situation_report",
            "international_organization_report",
        )
    ]
    database_queries = [
        item["query"]
        for item in inventory_dicts
        if item.get("source_type") == "structured_database"
    ]
    return SearchQuerySet(
        official_source_queries=official_source_queries,
        literature_queries=literature_queries,
        news_and_report_queries=news_and_report_queries,
        database_queries=database_queries,
    )


def query_strategy_builder(state: DataCollectionState) -> dict:
    """Build deterministic queries grouped both as SearchQuerySet and a typed inventory."""

    profile_dict = state.get("disease_profile") or load_hantavirus_profile()
    profile = DiseaseProfile(**profile_dict)

    strategy_dict = state.get("source_strategy") or load_source_strategy()
    strategy = SourceStrategy(**strategy_dict)

    spec_dict = state.get("collection_spec") or {}
    geography = spec_dict.get("geography")
    time_window = spec_dict.get("time_window")
    schema_dict = state.get("collection_schema") or load_hantavirus_collection_schema()

    disease_intelligence = state.get("disease_intelligence") or {}
    if disease_intelligence:
        include_terms = _as_str_list(
            disease_intelligence.get("suggested_query_terms")
        ) or _as_str_list(disease_intelligence.get("aliases"))
        syndrome_terms = _as_str_list(disease_intelligence.get("syndrome_terms"))
        virus_terms = _as_str_list(disease_intelligence.get("pathogen_terms"))
        outbreak_terms = _as_str_list(disease_intelligence.get("outbreak_terms"))
        official_source_terms = _as_str_list(
            disease_intelligence.get("official_source_terms")
        )
        disease_name_for_rationale = (
            disease_intelligence.get("disease_standard_name")
            or profile.disease_standard_name
        )
    else:
        include_terms = profile.include_terms
        syndrome_terms = profile.syndrome_terms
        virus_terms = profile.virus_terms
        outbreak_terms = []
        official_source_terms = []
        disease_name_for_rationale = profile.disease_standard_name

    geo_suffix = ""
    if geography and geography.lower() != "global":
        geo_suffix = f" {geography}"

    time_suffix = ""
    if time_window:
        time_suffix = f" {time_window}"

    expected_fields = _expected_fields_default()

    inventory: list[SearchQuery] = []
    seen: set[str] = set()
    next_index: dict[str, int] = {}
    agentic_source_plan = state.get("agentic_source_plan")
    direct_llm_plan_primary = _direct_collection_llm_plan_primary(state)
    source_planning_agent_summary = dict(
        state.get("source_planning_agent_summary")
        or {
            "llm_source_planning_enabled": llm_clients.llm_source_planning_enabled(),
            "status": "not_run",
            "agent_query_count": 0,
            "agent_query_added_count": 0,
            "agent_candidate_hint_count": 0,
            "warnings": ["executable_source_planning_node_not_run"],
        }
    )
    localized_summary = state.get("localized_source_planning_summary") or (
        (state.get("executable_source_plan_summary") or {}).get(
            "localized_source_planning"
        )
        or {}
    )
    planned_query_count = len(
        (agentic_source_plan or {}).get("planned_queries") or []
    )

    if direct_llm_plan_primary:
        inventory_dicts: list[dict] = []
        executable_query_added_count = _append_executable_plan_queries(
            inventory_dicts,
            seen,
            agentic_source_plan,
        )
        source_planning_agent_summary.update(
            {
                "agent_query_count": planned_query_count,
                "agent_query_added_count": executable_query_added_count,
                "query_strategy_consumed_executable_plan": True,
                "deterministic_query_template_suppressed": True,
                "deterministic_query_template_suppression_reason": (
                    "direct_collection uses LLM evidence strategy queries as the "
                    "primary search inventory; deterministic templates are a "
                    "fallback only when no executable plan queries exist."
                ),
            }
        )
        if localized_summary:
            source_planning_agent_summary[
                "localized_source_planning_summary"
            ] = localized_summary
        query_set = _search_query_set_from_inventory(inventory_dicts)
        trace = append_trace(
            state,
            node_name="query_strategy_builder",
            message=(
                f"Built {len(inventory_dicts)} direct_collection search queries "
                "from the LLM evidence strategy plan."
            ),
            metadata={
                "inventory_size": len(inventory_dicts),
                "deterministic_inventory_size": 0,
                "official_source_query_count": len(query_set.official_source_queries),
                "literature_query_count": len(query_set.literature_queries),
                "news_and_report_query_count": len(query_set.news_and_report_queries),
                "database_query_count": len(query_set.database_queries),
                "geography": geography,
                "time_window": time_window,
                "executable_source_plan_present": True,
                "localized_source_planning_summary": localized_summary,
                **source_planning_agent_summary,
            },
        )
        return {
            "search_queries": query_set.model_dump(),
            "search_query_inventory": inventory_dicts,
            "evidence_strategy_plan": agentic_source_plan,
            "localized_source_planning_summary": localized_summary,
            "source_planning_agent_summary": source_planning_agent_summary,
            "collection_trace": trace,
        }

    official_priority = _priority_for("official_public_health_agency", strategy)
    international_priority = _priority_for("international_organization_report", strategy)
    literature_priority = _priority_for("peer_reviewed_literature", strategy)
    database_priority = _priority_for("structured_database", strategy)
    news_priority = _priority_for("news_and_situation_report", strategy)

    # --- official_public_health_agency ---
    official_sites = ["cdc.gov", "who.int", "ecdc.europa.eu"]
    for site in official_sites:
        for term in include_terms[:4]:
            q = f'"{term}" cases deaths site:{site}{geo_suffix}{time_suffix}'.strip()
            _add_query(
                inventory, seen, next_index, "official",
                q, "official_public_health_agency", official_priority,
                f"Target official agency content on {site} for {disease_name_for_rationale} human cases and deaths.",
                expected_fields,
            )
    for source_term in official_source_terms[:8]:
        for term in include_terms[:2] or [profile.disease_standard_name]:
            q = f'"{source_term}" "{term}"{geo_suffix}{time_suffix}'.strip()
            _add_query(
                inventory,
                seen,
                next_index,
                "official",
                q,
                "official_public_health_agency",
                official_priority,
                (
                    "Search source-class-aware official public health channels "
                    f"for {disease_name_for_rationale}."
                ),
                expected_fields,
            )
        for official_site in official_sites:
            for term in syndrome_terms[:3]:
                q = (
                    f'"{term}" surveillance site:{official_site}'
                    f'{geo_suffix}{time_suffix}'
                ).strip()
                _add_query(
                    inventory, seen, next_index, "official",
                    q, "official_public_health_agency", official_priority,
                    f"Find {term} surveillance content from {official_site}.",
                    expected_fields,
                )

    coverage_requirements = build_source_coverage_requirements(state)
    for requirement in coverage_requirements:
        candidate_urls = list(requirement.get("official_candidate_urls") or [])
        candidate_filenames = [
            str(url).rsplit("/", 1)[-1]
            for url in candidate_urls
            if str(url).strip()
        ]
        title_hints = [
            str(value).strip()
            for value in (requirement.get("title_hints") or [])
            if str(value).strip()
        ]
        date_hints = [
            str(value).strip()
            for value in (requirement.get("date_hints") or [])
            if str(value).strip()
        ]
        for domain in requirement.get("official_domains") or []:
            week = requirement.get("week")
            year = requirement.get("year")
            for filename in candidate_filenames:
                q = f'site:{domain} "{filename}"'.strip()
                _add_query(
                    inventory,
                    seen,
                    next_index,
                    "official",
                    q,
                    "official_public_health_agency",
                    official_priority,
                    requirement.get("reason")
                    or "Find target jurisdiction official weekly surveillance report.",
                    expected_fields,
                )
            for hint in title_hints[:5]:
                q = f'site:{domain} "{hint}" "{year}"'.strip()
                _add_query(
                    inventory,
                    seen,
                    next_index,
                    "official",
                    q,
                    "official_public_health_agency",
                    official_priority,
                    requirement.get("reason")
                    or "Find target jurisdiction official weekly surveillance report.",
                    expected_fields,
                )
            for date_hint in date_hints[-2:]:
                for term in include_terms[:1] or ["influenza"]:
                    q = f'site:{domain} "{term}" "{date_hint}"'.strip()
                    _add_query(
                        inventory,
                        seen,
                        next_index,
                        "official",
                        q,
                        "official_public_health_agency",
                        official_priority,
                        requirement.get("reason")
                        or "Find target jurisdiction official weekly surveillance report.",
                        expected_fields,
                    )
            for term in include_terms[:2] or ["influenza"]:
                q = (
                    f'site:{domain} "{term}" "Week-{week}" "{year}"'
                ).strip()
                _add_query(
                    inventory,
                    seen,
                    next_index,
                    "official",
                    q,
                    "official_public_health_agency",
                    official_priority,
                    requirement.get("reason")
                    or "Find target jurisdiction official weekly surveillance report.",
                    expected_fields,
                )
            q = (
                f'site:{domain} "respiratory disease surveillance" '
                f'"Week-{week}" "{year}"'
            ).strip()
            _add_query(
                inventory,
                seen,
                next_index,
                "official",
                q,
                "official_public_health_agency",
                official_priority,
                requirement.get("reason")
                or "Find target jurisdiction official weekly surveillance report.",
                expected_fields,
            )

    # --- international_organization_report ---
    intl_sites = ["who.int", "paho.org", "ecdc.europa.eu"]
    for site in intl_sites:
        for term in include_terms[:3]:
            q = f'"{term}" outbreak report site:{site}{geo_suffix}{time_suffix}'.strip()
            _add_query(
                inventory, seen, next_index, "international",
                q, "international_organization_report", international_priority,
                f"Retrieve outbreak/situation reports about {term} from {site}.",
                expected_fields,
            )
    for outbreak_term in outbreak_terms[:8]:
        for term in include_terms[:2] or [profile.disease_standard_name]:
            q = f'"{term}" "{outbreak_term}"{geo_suffix}{time_suffix}'.strip()
            _add_query(
                inventory,
                seen,
                next_index,
                "international",
                q,
                "international_organization_report",
                international_priority,
                (
                    "Search generic outbreak, vessel-associated, travel-associated, "
                    "or multi-country cluster terms against international sources."
                ),
                expected_fields,
            )
        for intl_site in intl_sites:
            for term in syndrome_terms[:2]:
                q = (
                    f'"{term}" surveillance report site:{intl_site}'
                    f'{geo_suffix}{time_suffix}'
                ).strip()
                _add_query(
                    inventory, seen, next_index, "international",
                    q, "international_organization_report", international_priority,
                    f"Retrieve {term} surveillance reports from {intl_site}.",
                    expected_fields,
                )

    # --- peer_reviewed_literature ---
    for term in include_terms:
        q = f'"{term}" outbreak cases deaths epidemiology{geo_suffix}{time_suffix}'.strip()
        _add_query(
            inventory, seen, next_index, "literature",
            q, "peer_reviewed_literature", literature_priority,
            f"Find peer-reviewed epidemiology studies of {term}.",
            expected_fields,
        )
    for term in syndrome_terms:
        q = f'"{term}" human cases outbreak{geo_suffix}{time_suffix}'.strip()
        _add_query(
            inventory, seen, next_index, "literature",
            q, "peer_reviewed_literature", literature_priority,
            f"Find peer-reviewed human case series for {term}.",
            expected_fields,
        )
    for virus in virus_terms:
        q = f'"{virus}" human cases outbreak{geo_suffix}{time_suffix}'.strip()
        _add_query(
            inventory, seen, next_index, "literature",
            q, "peer_reviewed_literature", literature_priority,
            f"Find peer-reviewed studies reporting human {virus} cases.",
            expected_fields,
        )

    # --- structured_database ---
    for term in include_terms[:4]:
        q = f'"{term}" surveillance dataset{geo_suffix}{time_suffix}'.strip()
        _add_query(
            inventory, seen, next_index, "database",
            q, "structured_database", database_priority,
            f"Locate structured surveillance datasets for {term}.",
            expected_fields,
        )
        q2 = f'"{term}" outbreak data{geo_suffix}{time_suffix}'.strip()
        _add_query(
            inventory, seen, next_index, "database",
            q2, "structured_database", database_priority,
            f"Locate structured outbreak datasets for {term}.",
            expected_fields,
        )
    for term in syndrome_terms[:3]:
        q = f'"{term}" line list cases deaths{geo_suffix}{time_suffix}'.strip()
        _add_query(
            inventory, seen, next_index, "database",
            q, "structured_database", database_priority,
            f"Locate line lists or case-level data for {term}.",
            expected_fields,
        )

    # --- news_and_situation_report ---
    for term in include_terms[:4]:
        q = f'"{term}" outbreak report cases deaths{geo_suffix}{time_suffix}'.strip()
        _add_query(
            inventory, seen, next_index, "news",
            q, "news_and_situation_report", news_priority,
            f"Pick up news / situation reports about {term} outbreaks.",
            expected_fields,
        )
    for term in syndrome_terms[:3]:
        q = f'"{term}" human cases outbreak report{geo_suffix}{time_suffix}'.strip()
        _add_query(
            inventory, seen, next_index, "news",
            q, "news_and_situation_report", news_priority,
            f"Pick up news / situation reports about {term}.",
            expected_fields,
        )
    for virus in virus_terms:
        q = f'"{virus}" confirmed cases deaths{geo_suffix}{time_suffix}'.strip()
        _add_query(
            inventory, seen, next_index, "news",
            q, "news_and_situation_report", news_priority,
            f"Pick up news / alerts about confirmed {virus} cases.",
            expected_fields,
        )

    inventory_dicts = [q.model_dump() for q in inventory]
    executable_query_added_count = _append_executable_plan_queries(
        inventory_dicts, seen, agentic_source_plan
    )
    source_planning_agent_summary.update(
        {
            "agent_query_count": planned_query_count,
            "agent_query_added_count": executable_query_added_count,
            "query_strategy_consumed_executable_plan": bool(agentic_source_plan),
            "deterministic_query_template_suppressed": False,
        }
    )
    if localized_summary:
        source_planning_agent_summary[
            "localized_source_planning_summary"
        ] = localized_summary

    # Group into the backward-compatible SearchQuerySet structure.
    query_set = _search_query_set_from_inventory(inventory_dicts)

    trace = append_trace(
        state,
        node_name="query_strategy_builder",
        message=(
            f"Built {len(inventory_dicts)} search queries across 5 source "
            f"categories (executable_source_plan_present={bool(agentic_source_plan)})."
        ),
        metadata={
            "inventory_size": len(inventory_dicts),
            "deterministic_inventory_size": len(inventory),
            "official_source_query_count": len(query_set.official_source_queries),
            "literature_query_count": len(query_set.literature_queries),
            "news_and_report_query_count": len(query_set.news_and_report_queries),
            "database_query_count": len(query_set.database_queries),
            "geography": geography,
            "time_window": time_window,
            "executable_source_plan_present": bool(agentic_source_plan),
            "localized_source_planning_summary": localized_summary,
            **source_planning_agent_summary,
        },
    )
    return {
        "search_queries": query_set.model_dump(),
        "search_query_inventory": inventory_dicts,
        "evidence_strategy_plan": agentic_source_plan,
        "localized_source_planning_summary": localized_summary,
        "source_planning_agent_summary": source_planning_agent_summary,
        "collection_trace": trace,
    }
