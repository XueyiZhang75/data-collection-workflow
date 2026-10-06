"""Deterministic disease relevance gates for the data collection workflow."""

from __future__ import annotations

from collections import Counter
import re
from urllib.parse import urlsplit


TARGET_DISEASE_MATCH = "target_disease_match"
RELATED_CONTEXT_ONLY = "related_context_only"
AMBIGUOUS_DISEASE = "ambiguous_disease"
UNRELATED_DISEASE = "unrelated_disease"
INSUFFICIENT_TEXT = "insufficient_text"
COMPATIBLE = "compatible"
INCOMPATIBLE_DISEASE = "incompatible_disease"

EXTRACTABLE_DISEASE_STATUSES = {TARGET_DISEASE_MATCH}
REJECT_DISEASE_STATUSES = {UNRELATED_DISEASE, INCOMPATIBLE_DISEASE}

_TERM_GROUPS: dict[str, dict[str, object]] = {
    "hantavirus": {
        "canonical": "Hantavirus disease",
        "triggers": ["hantavirus", "hps", "hfrs", "orthohantavirus"],
        "terms": [
            "hantavirus",
            "hantaviruses",
            "hantavirus disease",
            "hantavirus infection",
            "hantavirus pulmonary syndrome",
            "HPS",
            "hemorrhagic fever with renal syndrome",
            "haemorrhagic fever with renal syndrome",
            "HFRS",
            "orthohantavirus",
            "Sin Nombre virus",
            "Seoul virus",
            "Hantaan virus",
            "Puumala virus",
            "Dobrava-Belgrade virus",
            "Andes virus",
            "han virus",
            "汉坦病毒",
            "肾综合征出血热",
            "流行性出血热",
        ],
    },
    "covid19": {
        "canonical": "COVID-19",
        "triggers": ["covid", "sars-cov-2", "coronavirus disease 2019"],
        "terms": [
            "COVID-19",
            "COVID",
            "coronavirus disease 2019",
            "SARS-CoV-2",
            "SARS CoV 2",
            "2019-nCoV",
        ],
    },
    "dengue": {
        "canonical": "Dengue",
        "triggers": ["dengue", "denv"],
        "terms": [
            "dengue",
            "dengue fever",
            "DENV",
            "dengue virus",
        ],
    },
    "west_nile": {
        "canonical": "West Nile virus disease",
        "triggers": ["west nile", "wnv"],
        "terms": [
            "West Nile virus",
            "West Nile virus disease",
            "West Nile disease",
            "West Nile fever",
            "West Nile neuroinvasive disease",
            "WNV",
        ],
    },
    "tuberculosis": {
        "canonical": "Tuberculosis",
        "triggers": ["tuberculosis", "tb", "mycobacterium tuberculosis"],
        "terms": [
            "Tuberculosis",
            "TB",
            "TB disease",
            "tuberculosis disease",
            "Mycobacterium tuberculosis",
        ],
    },
    "legionnaires": {
        "canonical": "Legionnaires' disease",
        "triggers": ["legionnaires", "legionella"],
        "terms": [
            "Legionnaires' disease",
            "Legionnaires disease",
            "Legionella",
            "Legionella pneumophila",
            "legionellosis",
        ],
    },
    "ebola": {
        "canonical": "Ebola disease",
        "triggers": ["ebola", "orthoebolavirus", "zairense", "zaire ebolavirus"],
        "terms": [
            "Ebola",
            "Ebola virus disease",
            "EVD",
            "Zaire ebolavirus",
            "Orthoebolavirus zairense",
            "ebolavirus",
        ],
    },
    "influenza": {
        "canonical": "Influenza",
        "triggers": ["flu", "influenza", "ili", "h5n6", "h5n1"],
        "terms": [
            "Influenza",
            "flu",
            "seasonal influenza",
            "influenza-like illness",
            "ILI",
            "avian influenza",
            "H5N6",
            "H5N1",
            "influenza A",
            "influenza B",
        ],
    },
    "yellow_fever": {
        "canonical": "Yellow fever",
        "triggers": ["yellow fever"],
        "terms": [
            "Yellow fever",
            "yellow fever virus",
        ],
    },
    "measles": {
        "canonical": "Measles",
        "triggers": ["measles", "rubeola"],
        "terms": [
            "Measles",
            "measles virus",
            "rubeola",
        ],
    },
    "candidemia": {
        "canonical": "Candidemia",
        "triggers": ["candidemia", "candidaemia", "candida"],
        "terms": [
            "Candidemia",
            "candidaemia",
            "Candida bloodstream infection",
            "Candida infection",
            "Candida auris",
        ],
    },
}

_DATA_SIGNAL_TERMS = [
    "case",
    "cases",
    "confirmed",
    "probable",
    "suspected",
    "death",
    "deaths",
    "fatality",
    "fatalities",
    "died",
    "hospitalization",
    "hospitalizations",
    "hospitalisation",
    "hospitalisations",
    "outbreak",
    "cluster",
    "surveillance",
    "reported",
    "report",
    "reports",
    "notification",
    "notifications",
    "incidence",
    "prevalence",
    "dataset",
    "table",
]

_SOURCE_TEXT_FIELDS = [
    "title",
    "snippet",
    "publisher",
    "domain",
    "source_type",
    "source_purpose",
    "notes",
    "result_source",
    "canonical_url",
    "url",
]

_RECORD_TEXT_FIELDS = [
    "disease",
    "disease_standard_name",
    "disease_alias_used",
    "virus_or_syndrome",
    "pathogen_or_syndrome",
    "source_title",
    "evidence_quote",
    "evidence_context",
    "source_url",
]


def _norm(value: str | None) -> str:
    text = str(value or "")
    text = text.replace("\u2010", "-").replace("\u2011", "-")
    text = text.replace("\u2012", "-").replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", text).strip()


def _term_key(value: str) -> str:
    return _norm(value).lower().replace("-", " ")


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _norm(value)
        key = _term_key(cleaned)
        if cleaned and key not in seen:
            result.append(cleaned)
            seen.add(key)
    return result


def canonical_disease_name(value: str | None) -> str | None:
    key = _term_key(str(value or ""))
    if not key:
        return None
    for group in _TERM_GROUPS.values():
        for term in list(group.get("triggers") or []) + list(group.get("terms") or []):
            if _term_key(str(term)) == key or _term_key(str(term)) in key:
                return str(group["canonical"])
    return _norm(str(value))


def _group_for_disease(value: str | None) -> str | None:
    key = _term_key(str(value or ""))
    if not key:
        return None
    for group_name, group in _TERM_GROUPS.items():
        for term in list(group.get("triggers") or []) + list(group.get("terms") or []):
            term_key = _term_key(str(term))
            if term_key and (term_key == key or term_key in key):
                return group_name
    return None


def _task_source_dict(state: dict | None) -> tuple[dict, dict, dict]:
    state = state or {}
    return (
        state.get("structured_task") or {},
        state.get("collection_spec") or {},
        state.get("disease_intelligence") or {},
    )


def build_disease_relevance_context(state: dict | None = None) -> dict:
    """Build deterministic target/incompatible term sets from workflow state."""

    structured_task, collection_spec, disease_intelligence = _task_source_dict(state)
    raw_values = [
        structured_task.get("disease"),
        collection_spec.get("disease"),
        disease_intelligence.get("disease_standard_name"),
        disease_intelligence.get("disease_input"),
    ]
    target_raw = next((str(v) for v in raw_values if v), None)
    target_disease = (
        canonical_disease_name(target_raw) or _norm(target_raw)
        if target_raw
        else None
    )
    target_group = _group_for_disease(target_disease) or _group_for_disease(target_raw)

    from .disease_identity import disease_names
    identity_terms: list[str] = [str(v) for v in raw_values if v]
    identity_terms.extend(sorted(disease_names(target_raw)))
    for key in (
        "aliases",
        "abbreviations",
        "pathogen_terms",
        "syndrome_terms",
    ):
        values = disease_intelligence.get(key) or []
        if isinstance(values, list):
            identity_terms.extend(str(v) for v in values if v)
    if target_group and target_group in _TERM_GROUPS:
        identity_terms.extend(str(v) for v in _TERM_GROUPS[target_group]["terms"])

    data_signal_terms: list[str] = list(_DATA_SIGNAL_TERMS)
    for key in (
        "case_count_terms",
        "death_terms",
        "hospitalization_terms",
        "outbreak_terms",
    ):
        values = disease_intelligence.get(key) or []
        if isinstance(values, list):
            data_signal_terms.extend(str(v) for v in values if v)

    incompatible_terms: list[str] = []
    if target_raw:
        for group_name, group in _TERM_GROUPS.items():
            if target_group and group_name == target_group:
                continue
            incompatible_terms.extend(str(v) for v in group["terms"])

    return {
        "guard_enabled": bool(target_raw),
        "target_disease": target_disease,
        "target_group": target_group,
        "target_disease_identity_terms": _dedupe(identity_terms),
        "target_disease_terms": _dedupe(identity_terms),
        "data_signal_terms": _dedupe(data_signal_terms),
        "incompatible_disease_terms": _dedupe(incompatible_terms),
    }


def _term_pattern(term: str) -> re.Pattern | None:
    cleaned = _norm(term)
    if not cleaned:
        return None
    if re.search(r"[^\x00-\x7F]", cleaned):
        return re.compile(re.escape(cleaned), re.IGNORECASE)
    escaped = re.sub(
        r"(?:\\ |\\-)+",
        lambda _match: r"[\s\-]+",
        re.escape(cleaned),
    )
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\s\-/.]*", cleaned):
        return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
    return re.compile(escaped, re.IGNORECASE)


def _find_terms(text: str, terms: list[str]) -> list[str]:
    found: list[str] = []
    for term in terms:
        pattern = _term_pattern(term)
        if pattern is not None and pattern.search(text):
            found.append(_norm(term))
    return _dedupe(found)


def detect_data_signal_count(text: str) -> int:
    lowered = text.lower()
    count = 0
    for term in _DATA_SIGNAL_TERMS:
        pattern = _term_pattern(term)
        if pattern is not None and pattern.search(lowered):
            count += 1
    if re.search(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", text):
        count += 1
    if re.search(r"\b(?:19|20)\d{2}\b", text):
        count += 1
    return count


def _target_identity_terms(context: dict) -> list[str]:
    terms = list(context.get("target_disease_identity_terms") or [])
    if terms:
        return terms
    return list(context.get("target_disease_terms") or [])


def assess_text_disease_relevance(
    text: str | None,
    context: dict,
    *,
    require_data_signal: bool = False,
    evidence_fields_used: list[str] | None = None,
) -> dict:
    """Assess text against the target disease and known incompatible diseases."""

    clean = _norm(text)
    target_terms = _target_identity_terms(context)
    incompatible_terms = list(context.get("incompatible_disease_terms") or [])
    data_signal_count = detect_data_signal_count(clean)
    has_data_signal = data_signal_count > 0
    if not context.get("guard_enabled"):
        status = TARGET_DISEASE_MATCH if clean else INSUFFICIENT_TEXT
        score = 1.0 if clean else 0.0
        reason = (
            "No active task disease was provided; disease relevance gate was not enforced."
        )
        return {
            "status": status,
            "score": round(score, 4),
            "target_disease": context.get("target_disease"),
            "target_disease_terms_found": [],
            "incompatible_disease_terms_found": [],
            "has_data_signal": has_data_signal,
            "data_signal_count": data_signal_count,
            "evidence_fields_used": list(evidence_fields_used or []),
            "reason": reason,
        }

    target_found = _find_terms(clean, target_terms)
    incompatible_found = _find_terms(clean, incompatible_terms)

    if incompatible_found and not target_found:
        status = UNRELATED_DISEASE
        score = 0.0
        reason = "Text names an incompatible disease and does not name the target task disease."
    elif incompatible_found and target_found:
        status = AMBIGUOUS_DISEASE
        score = 0.45
        reason = "Text names both the target task disease and an incompatible disease."
    elif target_found and (has_data_signal or not require_data_signal):
        status = TARGET_DISEASE_MATCH
        score = 0.95 if has_data_signal else 0.82
        reason = "Text names the target task disease."
    elif target_found:
        status = RELATED_CONTEXT_ONLY
        score = 0.62
        reason = "Text names the target task disease but lacks extractable data signals."
    elif clean and has_data_signal:
        status = AMBIGUOUS_DISEASE
        score = 0.20
        reason = "Text has data signals but does not name the target task disease."
    else:
        status = INSUFFICIENT_TEXT
        score = 0.0 if len(clean) < 30 else 0.10
        reason = "Text does not provide enough target-disease evidence."

    return {
        "status": status,
        "score": round(score, 4),
        "target_disease": context.get("target_disease"),
        "target_disease_terms_found": target_found,
        "incompatible_disease_terms_found": incompatible_found,
        "has_data_signal": has_data_signal,
        "data_signal_count": data_signal_count,
        "evidence_fields_used": list(evidence_fields_used or []),
        "reason": reason,
    }


def _domain(value: str | None) -> str:
    netloc = urlsplit(str(value or "")).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def source_relevance_text(entry: dict) -> tuple[str, list[str]]:
    parts: list[str] = []
    used: list[str] = []
    for field in _SOURCE_TEXT_FIELDS:
        value = entry.get(field)
        if value:
            if field in {"canonical_url", "url"}:
                domain = _domain(str(value))
                if domain:
                    parts.append(domain)
                    used.append(field)
                continue
            parts.append(str(value))
            used.append(field)
    return " ".join(parts), used


def assess_source_disease_relevance(entry: dict, context: dict) -> dict:
    text, used = source_relevance_text(entry)
    return assess_text_disease_relevance(
        text,
        context,
        require_data_signal=False,
        evidence_fields_used=used,
    )


def assess_document_disease_relevance(document: dict, context: dict) -> dict:
    text = " ".join(
        str(document.get(field) or "")
        for field in ("title", "clean_text")
        if document.get(field)
    )
    return assess_text_disease_relevance(
        text,
        context,
        require_data_signal=False,
        evidence_fields_used=["title", "clean_text"],
    )


def assess_chunk_disease_relevance(chunk: dict, context: dict) -> dict:
    text = " ".join(
        str(chunk.get(field) or "")
        for field in ("title", "text")
        if chunk.get(field)
    )
    return assess_text_disease_relevance(
        text,
        context,
        require_data_signal=True,
        evidence_fields_used=["title", "text"],
    )


def _canonical_group_from_record_values(record: dict) -> str | None:
    for field in (
        "disease",
        "disease_standard_name",
        "virus_or_syndrome",
        "pathogen_or_syndrome",
        "disease_alias_used",
    ):
        group = _group_for_disease(record.get(field))
        if group:
            return group
    return None


def record_compatibility_text(record: dict) -> tuple[str, list[str]]:
    parts: list[str] = []
    used: list[str] = []
    for field in _RECORD_TEXT_FIELDS:
        value = record.get(field)
        if value:
            parts.append(str(value))
            used.append(field)
    return " ".join(parts), used


def record_local_evidence_text(record: dict) -> tuple[str, list[str]]:
    """Return the narrow evidence span that carries the extracted value."""

    context_parts: list[str] = []
    evidence_parts: list[str] = []
    used: list[str] = []
    for field in ("heading_context", "table_header"):
        value = record.get(field)
        if value:
            context_parts.append(str(value))
            used.append(field)
    for field in (
        "evidence_quote",
        "evidence_context",
        "case_count_text_span",
        "death_count_text_span",
    ):
        value = record.get(field)
        if value:
            evidence_parts.append(str(value))
            used.append(field)
    parts = ([" ".join(context_parts)] if context_parts else []) + evidence_parts
    return ". ".join(parts), used


def _record_has_count_bearing_value(record: dict) -> bool:
    for field in (
        "case_count",
        "cases_confirmed",
        "cases_probable",
        "cases_suspected",
        "cases_unspecified",
        "death_count",
        "deaths",
        "hospitalizations",
    ):
        value = record.get(field)
        if value not in (None, "", [], {}):
            return True
    return any(
        token in _term_key(str(record.get("metric_category") or ""))
        for token in ("case", "death", "hospital")
    )


def _evidence_sentences(text: str) -> list[str]:
    cleaned = _norm(text)
    if not cleaned:
        return []
    # Internal dots in numeric values and identifier segments are not sentence
    # boundaries. Keep the source characters intact while splitting run-on prose.
    internal_dots = set()
    for match in re.finditer(
        r"\b[A-Z][A-Z0-9]*(?:\.\d+[A-Z0-9]*)+\b|\b\d+(?:,\d{3})*(?:\.\d+)+\b",
        cleaned,
    ):
        token = match.group()
        # A single acronym.count dot can also join two prose sentences. Only
        # explicit identifier wording makes that ambiguous form an identifier.
        if token[0].isalpha() and token.count(".") == 1 and not re.search(
            r"\b(?:lineage|variant|clade|genotype|subtype|strain|version|identifier|id|code)\s*[:=]?\s*$",
            cleaned[:match.start()], re.I,
        ):
            continue
        internal_dots.update(match.start() + offset for offset, character in enumerate(token) if character == ".")
    cleaned = re.sub(
        r"(?<=[.!?;])(?=[A-Z0-9])",
        lambda match: "" if match.start() - 1 in internal_dots else " ",
        cleaned,
    )
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?;。！？；])\s+", cleaned)
        if sentence.strip()
    ] or [cleaned]


def _sentence_has_count_signal(sentence: str) -> bool:
    lowered = sentence.lower()
    has_standalone_number = bool(
        re.search(r"(?<![A-Za-z0-9-])\d+(?:[.,]\d+)?(?![A-Za-z0-9-])", sentence)
    )
    return has_standalone_number and any(
        re.search(rf"\b{re.escape(term)}\b", lowered)
        for term in _DATA_SIGNAL_TERMS
    )


def _count_bearing_sentence_disease_mismatch(
    local_text: str,
    context: dict,
) -> tuple[bool, bool, list[str], list[str], str | None]:
    target_terms = _target_identity_terms(context)
    incompatible_terms = list(context.get("incompatible_disease_terms") or [])
    incompatible_found_all: list[str] = []
    target_found_all: list[str] = []
    count_sentence_seen = False
    count_sentence_text: str | None = None
    sentences = _evidence_sentences(local_text)
    for index, sentence in enumerate(sentences):
        if not _sentence_has_count_signal(sentence):
            continue
        count_sentence_seen = True
        if count_sentence_text is None:
            count_sentence_text = sentence
        previous = sentences[index - 1] if index > 0 else ""
        disease_window = f"{previous} {sentence}".strip()
        sentence_target_found = _find_terms(sentence, target_terms)
        sentence_incompatible_found = _find_terms(sentence, incompatible_terms)
        target_found = _find_terms(disease_window, target_terms)
        incompatible_found = _find_terms(disease_window, incompatible_terms)
        incompatible_found_all.extend(incompatible_found)
        target_found_all.extend(target_found)
        if (
            sentence_incompatible_found
            and not sentence_target_found
        ) or (incompatible_found and not target_found):
            return (
                True,
                count_sentence_seen,
                _dedupe(target_found_all),
                _dedupe(incompatible_found_all),
                count_sentence_text,
            )
    return (
        False,
        count_sentence_seen,
        _dedupe(target_found_all),
        _dedupe(incompatible_found_all),
        count_sentence_text,
    )


def _record_identity_disease_terms(record: dict, context: dict) -> tuple[list[str], list[str]]:
    identity_text = " ".join(
        str(record.get(field) or "")
        for field in (
            "disease",
            "disease_standard_name",
            "disease_alias_used",
            "virus_or_syndrome",
            "pathogen_or_syndrome",
            "metric_name",
            "metric_category",
            "count_semantics",
            "statistical_count_type",
        )
        if record.get(field)
    )
    return (
        _find_terms(identity_text, _target_identity_terms(context)),
        _find_terms(
            identity_text, list(context.get("incompatible_disease_terms") or [])
        ),
    )


def assess_record_disease_compatibility(record: dict, context: dict) -> dict:
    """Return a compatibility assessment for an extracted record."""

    text, used = record_compatibility_text(record)
    local_text, local_used = record_local_evidence_text(record)
    assessment = assess_text_disease_relevance(
        text,
        context,
        require_data_signal=False,
        evidence_fields_used=used,
    )
    local_assessment = assess_text_disease_relevance(
        local_text,
        context,
        require_data_signal=True,
        evidence_fields_used=local_used,
    ) if local_text.strip() else {}
    if not context.get("guard_enabled"):
        result = dict(assessment)
        result.update(
            {
                "status": COMPATIBLE,
                "reason": (
                    "No active task disease was provided; record disease compatibility gate was not enforced."
                ),
                "record_group": _canonical_group_from_record_values(record),
                "target_group": context.get("target_group"),
                "is_compatible": True,
                "reject_record": False,
            }
        )
        return result

    target_group = context.get("target_group")
    record_group = _canonical_group_from_record_values(record)

    incompatible_values = assessment["incompatible_disease_terms_found"]
    target_values = assessment["target_disease_terms_found"]
    local_target_values = list(local_assessment.get("target_disease_terms_found") or [])
    local_incompatible_values = list(
        local_assessment.get("incompatible_disease_terms_found") or []
    )
    local_status = str(local_assessment.get("status") or "")
    identity_target_values, identity_incompatible_values = _record_identity_disease_terms(
        record, context
    )
    record_level_target_identity_override = False
    count_sentence_mismatch = False
    count_sentence_seen = False
    count_sentence_text: str | None = None
    count_sentence_target_values: list[str] = []
    count_sentence_incompatible_values: list[str] = []
    if local_text.strip() and _record_has_count_bearing_value(record):
        (
            count_sentence_mismatch,
            count_sentence_seen,
            count_sentence_target_values,
            count_sentence_incompatible_values,
            count_sentence_text,
        ) = _count_bearing_sentence_disease_mismatch(local_text, context)
    count_sentence_lacks_target = bool(
        count_sentence_seen
        and not count_sentence_target_values
    )
    local_numeric_status = (
        INCOMPATIBLE_DISEASE
        if count_sentence_mismatch or count_sentence_lacks_target
        else TARGET_DISEASE_MATCH
        if count_sentence_target_values or local_target_values
        else local_status
    )
    local_mismatch = bool(
        local_text.strip()
        and local_incompatible_values
        and not local_target_values
    )
    if count_sentence_mismatch:
        status = INCOMPATIBLE_DISEASE
        reason = (
            "local_evidence_disease_mismatch: count-bearing evidence sentence "
            "names an incompatible disease and does not name the target task disease."
        )
    elif local_mismatch:
        status = INCOMPATIBLE_DISEASE
        reason = (
            "Record local evidence quote names an incompatible disease and "
            "does not name the target task disease."
        )
    elif count_sentence_lacks_target:
        status = INCOMPATIBLE_DISEASE
        reason = (
            "local_evidence_disease_mismatch: count-bearing evidence sentence "
            "does not name the target task disease or one of its aliases."
        )
    elif record_group and target_group and record_group != target_group:
        status = INCOMPATIBLE_DISEASE
        reason = "Record disease/pathogen field is incompatible with the target task disease."
    elif assessment["status"] == UNRELATED_DISEASE:
        status = INCOMPATIBLE_DISEASE
        reason = assessment["reason"]
    elif assessment["status"] == AMBIGUOUS_DISEASE and incompatible_values:
        if identity_target_values and not identity_incompatible_values:
            status = COMPATIBLE
            record_level_target_identity_override = True
            reason = (
                "Record disease/metric identity matches the target task disease; "
                "incompatible terms appear in broader mixed-disease evidence."
            )
        else:
            status = INCOMPATIBLE_DISEASE
            reason = "Record evidence contains incompatible disease terms."
    elif target_values or (
        record_group is not None and target_group is not None and record_group == target_group
    ):
        status = COMPATIBLE
        reason = "Record is compatible with the target task disease."
    elif not text.strip():
        status = INSUFFICIENT_TEXT
        reason = "Record lacks text evidence for disease compatibility."
    else:
        status = AMBIGUOUS_DISEASE
        reason = "Record does not provide enough target-disease evidence."

    result = dict(assessment)
    result.update(
        {
            "status": status,
            "reason": reason,
            "record_group": record_group,
            "target_group": target_group,
            "record_level_target_identity_override": record_level_target_identity_override,
            "record_local_disease_relevance_status": (
                INCOMPATIBLE_DISEASE
                if (local_mismatch or count_sentence_mismatch or count_sentence_lacks_target)
                else local_status
            ),
            "record_local_evidence_fields_used": local_used,
            "record_local_target_terms_found": (
                count_sentence_target_values or local_target_values
            ),
            "record_local_incompatible_terms_found": (
                count_sentence_incompatible_values or local_incompatible_values
            ),
            "record_local_numeric_disease_status": local_numeric_status,
            "record_local_numeric_sentence": count_sentence_text,
            "record_local_numeric_disease_decision": reason
            if (
                count_sentence_mismatch
                or count_sentence_lacks_target
                or local_numeric_status == INCOMPATIBLE_DISEASE
            )
            else local_numeric_status,
            "record_local_numeric_target_terms_found": count_sentence_target_values,
            "record_local_numeric_incompatible_terms_found": (
                count_sentence_incompatible_values
            ),
            "is_compatible": status == COMPATIBLE,
            "reject_record": status in {INCOMPATIBLE_DISEASE},
        }
    )
    return result


def assessment_fields(assessment: dict, prefix: str = "") -> dict:
    pre = f"{prefix}_" if prefix else ""
    return {
        f"{pre}disease_relevance_status": assessment.get("status"),
        f"{pre}disease_relevance_score": assessment.get("score"),
        f"{pre}target_disease_terms_found": list(
            assessment.get("target_disease_terms_found") or []
        ),
        f"{pre}incompatible_disease_terms_found": list(
            assessment.get("incompatible_disease_terms_found") or []
        ),
        f"{pre}disease_relevance_reason": assessment.get("reason"),
        f"{pre}disease_relevance_data_signal_count": assessment.get(
            "data_signal_count", 0
        ),
    }


def record_compatibility_fields(assessment: dict) -> dict:
    return {
        "record_disease_compatibility_status": assessment.get("status"),
        "record_disease_compatibility_reason": assessment.get("reason"),
        "record_target_disease_terms_found": list(
            assessment.get("target_disease_terms_found") or []
        ),
        "record_incompatible_disease_terms_found": list(
            assessment.get("incompatible_disease_terms_found") or []
        ),
        "record_disease_compatibility_reject": bool(assessment.get("reject_record")),
        "record_local_disease_relevance_status": assessment.get(
            "record_local_disease_relevance_status"
        ),
        "record_local_target_terms_found": list(
            assessment.get("record_local_target_terms_found") or []
        ),
        "record_local_incompatible_terms_found": list(
            assessment.get("record_local_incompatible_terms_found") or []
        ),
        "record_local_numeric_disease_status": assessment.get(
            "record_local_numeric_disease_status"
        ),
        "record_local_numeric_sentence": assessment.get(
            "record_local_numeric_sentence"
        ),
        "record_local_numeric_disease_decision": assessment.get(
            "record_local_numeric_disease_decision"
        ),
        "record_local_numeric_target_terms_found": list(
            assessment.get("record_local_numeric_target_terms_found") or []
        ),
        "record_local_numeric_incompatible_terms_found": list(
            assessment.get("record_local_numeric_incompatible_terms_found") or []
        ),
    }


def update_disease_relevance_summary(state: dict, **extra_counts) -> dict:
    """Build a compact summary across available workflow artifacts."""

    context = build_disease_relevance_context(state)

    def _count(items: list[dict], field: str) -> dict[str, int]:
        return dict(Counter(str(item.get(field) or "unknown") for item in items))

    rejected = list(state.get("rejected_records") or [])
    normalized = list(state.get("normalized_records") or [])
    summary = {
        "target_disease": context.get("target_disease"),
        "target_group": context.get("target_group"),
        "source_status_counts": _count(
            list(state.get("source_registry") or []),
            "source_disease_relevance_status",
        ),
        "document_status_counts": _count(
            list(state.get("documents") or []),
            "document_disease_relevance_status",
        ),
        "chunk_status_counts": _count(
            list(state.get("evidence_chunks") or []),
            "disease_relevance_status",
        ),
        "record_compatibility_status_counts": _count(
            list(state.get("raw_records") or [])
            + list(state.get("validated_records") or [])
            + normalized
            + rejected,
            "record_disease_compatibility_status",
        ),
        "rejected_incompatible_record_count": sum(
            1
            for record in rejected
            if record.get("record_disease_compatibility_status")
            in {INCOMPATIBLE_DISEASE, UNRELATED_DISEASE}
        ),
        "normalized_incompatible_record_count": sum(
            1
            for record in normalized
            if record.get("record_disease_compatibility_status")
            in {INCOMPATIBLE_DISEASE, UNRELATED_DISEASE}
        ),
    }
    summary.update(extra_counts)
    return summary
