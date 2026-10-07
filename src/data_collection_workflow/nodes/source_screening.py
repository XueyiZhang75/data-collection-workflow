"""Source screening, critic review, and source-level routing (Step 4).

Deterministic rule-based implementation. No LLM, no network. Later steps can
replace the rule-based logic with structured-output LLM agents without
changing node names or graph topology.
"""

from __future__ import annotations

from data_collection_workflow.environment import get_env

import re
from collections import Counter
from datetime import date, datetime
from time import monotonic
from urllib.parse import urlsplit

from .. import llm_clients
from ..query_policy import universal_queries_enabled
from ..agents.source_critic_agent import assess_source_with_llm
from ..config import (
    get_collection_mode,
    load_source_role_policy,
    load_source_screening_policy,
)
from ..models import (
    HumanReviewItem,
    SourceCriticResult,
    SourceFinalRoutingDecision,
    SourceRegistryEntry,
    SourceScreeningPolicy,
    SourceScreeningResult,
)
from ..source_credibility import (
    RUBRIC_VERSION as SOURCE_CREDIBILITY_RUBRIC_VERSION,
    apply_source_credibility_assessment,
    build_source_credibility_summary,
    source_credibility_runtime_from_env,
)
from ..source_identity import (
    _is_search_endpoint as _identity_search_endpoint,
    apply_source_identity_routing_guardrails,
    apply_source_identity_to_registry,
)
from ..source_coverage import annotate_source_coverage
from ..state import DataCollectionState, append_trace


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _parse_csv_env(name: str) -> set[str] | None:
    raw = get_env(name) or ""
    if not raw.strip():
        return None
    values = {part.strip() for part in raw.split(",") if part.strip()}
    return values or None


def _parse_positive_int_env(name: str) -> int | None:
    raw = (get_env(name) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _parse_positive_float_env(name: str) -> float | None:
    raw = (get_env(name) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = (get_env(name) or "").strip().lower()
    if not raw:
        return default
    return raw == "true"


def _collection_mode_from_state(
    state: DataCollectionState | dict,
    role_policy: dict | None = None,
) -> str:
    """Resolve collection mode from the active workflow state before policy.

    The direct-collection fast path is a per-run behavior. Reading only the
    static source-role policy lets old defaults leak into new runs and causes
    identity/critic agents to spend time on context sources after a target
    collection source is already verified.
    """

    structured_task = state.get("structured_task") or {}
    collection_spec = state.get("collection_spec") or {}
    for value in (
        collection_spec.get("collection_mode"),
        structured_task.get("collection_mode"),
        state.get("collection_mode"),
    ):
        if str(value or "").strip():
            return str(value).strip()
    return get_collection_mode(role_policy or load_source_role_policy())


def _parse_date(value) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d{4}", text):
        text = f"{text}-01-01"
    try:
        parts = [int(part) for part in text.split("-")]
        if len(parts) == 3:
            return date(parts[0], parts[1], parts[2])
    except (TypeError, ValueError):
        return None
    try:
        return datetime.fromisoformat(text).date()
    except (TypeError, ValueError):
        return None


def _task_week_pairs(state: DataCollectionState | dict) -> set[tuple[int, int]]:
    structured_task = state.get("structured_task") or {}
    collection_spec = state.get("collection_spec") or {}
    start = _parse_date(
        structured_task.get("start_date") or collection_spec.get("start_date")
    )
    end = _parse_date(
        structured_task.get("end_date")
        or collection_spec.get("end_date")
        or structured_task.get("start_date")
        or collection_spec.get("start_date")
    )
    if not start or not end:
        return set()
    if end < start:
        start, end = end, start
    pairs: set[tuple[int, int]] = set()
    current = start
    while current <= end:
        days_until_saturday = (5 - current.weekday()) % 7
        anchor = date.fromordinal(current.toordinal() + days_until_saturday)
        iso = anchor.isocalendar()
        pairs.add((int(iso.year), int(iso.week)))
        current = date.fromordinal(current.toordinal() + 1)
    return pairs


def _entry_target_text(entry: dict) -> str:
    return " ".join(
        str(entry.get(key) or "")
        for key in (
            "canonical_url",
            "url",
            "title",
            "name",
            "source_title",
            "snippet",
            "publisher",
            "published_date",
        )
    ).lower()


_YEAR_WEEK_PATTERNS = (
    r"\b(?P<year>20\d{2})[-_/ ]+week[-_/ ]?(?P<week>\d{1,2})\b",
    r"\bweek[-_/ ]?(?P<week>\d{1,2})\b.{0,40}?\b(?P<year>20\d{2})\b",
    r"\b(?P<year>20\d{2})[-_/ ]+w(?P<week>\d{1,2})\b",
)


def _explicit_year_week_pairs(text: str) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for pattern in _YEAR_WEEK_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                pairs.add((int(match.group("year")), int(match.group("week"))))
            except (TypeError, ValueError):
                continue
    return pairs


def _source_reporting_date_fit(texts: list[str], state: dict) -> str:
    """Compare periods stated by a source, independently of its publication date."""
    task = _task_for_screening(state)
    start = _parse_date(task.get("start_date"))
    end = _parse_date(task.get("end_date")) or start
    if not start or not end:
        return "candidate"
    if end < start:
        start, end = end, start
    # Bibliography years and publishing metadata do not describe observations.
    texts = [re.split(r"\b(?:references|bibliography|literature cited)\s*:|(?:^|\n)\s*(?:references|bibliography)\s*(?:\n|$)", text, maxsplit=1, flags=re.IGNORECASE)[0]
             for text in texts]
    # Remove only a date-shaped value immediately following a metadata label.
    # A label used in prose ("Updated case counts...") or a completed sentence
    # ("Published online.") must not consume a later observation year.
    year = r"(?:19|20)\d{2}"
    month = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
             r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
             r"janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)\.?" )
    day = r"\d{1,2}(?:st|nd|rd|th)?\b"
    metadata_date = (rf"(?:{year}(?:[-/]\d{{1,2}}){{0,2}}|"
                     rf"(?:{day}\s+)?{month}\s+(?:{day},?\s+)?{year}|"
                     rf"\d{{1,2}}[-/]\d{{1,2}}[-/]{year})")
    publication_metadata = (r"\b(?:published|updated|reviewed|accessed|retrieved|copyright)\b"
                            r"(?:[ \t]+(?:online|electronically|on|in|date)){0,3}[ \t]*:?[ \t]*"
                            rf"{metadata_date}\b")
    texts = [re.sub(publication_metadata, " ", text, flags=re.IGNORECASE) for text in texts]
    # A header or an agency's history cannot borrow a later case-count
    # sentence as its observation scope. This is retrieval triage only.
    from ..disease_identity import disease_names
    from ..source_assertions import NUMBER_TOKEN, count_mentions
    names = set(disease_names(task.get("disease"))) | set(task.get("disease_aliases") or [])
    observation_signal = re.compile(
        r"\b(?:cases?|deaths?|patients?|surveillance|outbreaks?|incidence|"
        r"epidemiolog\w*|notifications?|reporting\s+(?:period|week)|observation\s+period|"
        r"cas|décès|épidémi\w*|semaine\w*|casos|óbitos|brote\w*)\b", re.IGNORECASE)
    background_date = (r"\b(?:established|founded|created|launched|"
                       r"first\s+(?:identified|discovered|described)|fondée?|créée?)"
                       r"(?:\s+(?:in|on|en|was|the)){0,3}\s+" + metadata_date)
    current_signal = re.compile(
        r"\b(?:current(?:ly)?|ongoing|now|latest|recent(?:ly)?|today|this\s+(?:week|month|year)|"
        r"to\s+date|actuelle?\w*|désormais|ces\s+dernières|à\s+ce\s+jour|"
        r"actualmente|recientes?|últimas?\s+semanas?)\b|s['’]intensifie",
        re.IGNORECASE)
    current_assertion = re.compile(
        r"\b\d+(?:[.,\s]\d+)*\s+(?:(?:confirmed|suspected|reported|new|active)\s+)?"
        r"(?:cases?|deaths?|patients?|cas|décès|casos)\b|"
        r"\b(?:cases?|deaths?)\s+(?:(?:have|has|are|is)\s+)?"
        r"(?:(?:now|currently|recently)\s+)?(?:increas|decreas|ris|fall)\w*\b|"
        r"\b(?:cas|décès|casos)\s+(?:ont\s+|han\s+)?(?:augment|diminu|aument|disminu)\w*\b|"
        r"s['’]intensifie", re.IGNORECASE)
    # An administrative declaration is a dated event, not an observation
    # interval. Keep dates when the same statement actually reports data.
    declaration = re.compile(r"\b(?:declar\w*|déclar\w*)\b", re.IGNORECASE)
    emergency = re.compile(r"\b(?:emergenc\w*|urgence\w*)\b", re.IGNORECASE)
    reported_observation = re.compile(
        r"\b(?:cases?|deaths?|patients?|cas|décès|casos)\s+"
        r"(?:(?:were|was|are|is|have|has|been|ont|été|fueron|han|sido)\s+)*"
        r"(?:report\w*|record\w*|confirm\w*|signal\w*|registr\w*)\b|"
        r"\b(?:surveillance|reporting|observation)\s+period\b", re.IGNORECASE)
    # Only onset language attached to an outbreak can open its interval.
    # A start with an explicit end still describes a closed observation.
    outbreak_onset = re.compile(
        rf"\b(?:outbreak|épidémie|epidemia|brote)\b[^.;!?]{{0,100}}?"
        rf"\b(?:began|started|commenc\w*|début\w*|comenz\w*|empez\w*)\s+"
        rf"(?:(?:in|on|en|le|the)\s+)?(?P<date>{metadata_date})\b", re.IGNORECASE)
    outbreak_end = re.compile(
        rf"\b(?:ended|ceased|resolved|termin\w*|achev\w*)\s+"
        rf"(?:(?:in|on|en|le|the)\s+)?{metadata_date}\b", re.IGNORECASE)
    statements = []
    for text in texts:
        text = re.sub(background_date, " ", text, flags=re.IGNORECASE)
        for statement in re.split(r"(?<!\b[A-Za-z])\.(?:\s+|$)|[!?;|\n]+", text):
            named_disease = any(name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)",
                                                  statement, re.IGNORECASE) for name in names)
            if observation_signal.search(statement) or named_disease or _explicit_year_week_pairs(statement):
                statements.append(statement)
    texts = statements
    pairs = set().union(*(_explicit_year_week_pairs(text) for text in texts)) if texts else set()
    periods = []
    for reporting_year, reporting_week in pairs:
        try:
            periods.append((date.fromisocalendar(reporting_year, reporting_week, 1),
                            date.fromisocalendar(reporting_year, reporting_week, 7)))
        except ValueError:
            continue
    unknown_open_period = False
    iso_range = r"\b((?:19|20)\d{2}-\d{2}-\d{2})\s*(?:to|through|[-\u2013\u2014\u2212])\s*((?:19|20)\d{2}-\d{2}-\d{2})\b"
    for text in texts:
        clauses = re.split(
            r"(?<![,\s])(?=,?\s+(?:but|whereas|while|mais|pero|and|et|y|after|before|après|avant|for)\b)",
            text, flags=re.IGNORECASE)
        for index, clause in enumerate(clauses):
            # A conjunction may continue data under the preceding date. Only
            # an independently dated continuation has its own temporal scope.
            following = clauses[index + 1] if index + 1 < len(clauses) else ""
            following_counts = count_mentions(following)
            independently_dated = any(
                not any(item["char_start"] <= match.start() < item["char_end"]
                        for item in following_counts)
                for match in re.finditer(metadata_date, following, re.IGNORECASE))
            shared_date = bool(re.match(r",?\s*(?:and|et|y|after|before|après|avant|for)\b", following, re.IGNORECASE)
                and (current_assertion.search(following) or following_counts or reported_observation.search(following))
                and not independently_dated)
            if (declaration.search(clause) and emergency.search(clause) and not shared_date
                    and not current_assertion.search(clause) and not count_mentions(clause)
                    and not reported_observation.search(clause)
                    and not _explicit_year_week_pairs(clause)):
                clauses[index] = re.sub(metadata_date, " ", clause, flags=re.IGNORECASE)
        text = "".join(clauses)
        current_observation = bool((current_assertion.search(text) or count_mentions(text))
                                   and current_signal.search(text))
        future_onset = False
        if current_observation and not outbreak_end.search(text):
            onset = outbreak_onset.search(text)
            if onset:
                onset_year = re.search(year, onset.group("date"))
                # Unknown current time does not override an explicit start
                # after the requested window, nor dates on the count itself.
                lower = _parse_date(onset.group("date"))
                if lower is None and onset_year:
                    lower = date(int(onset_year.group()), 1, 1)
                future_onset = bool(lower and lower > end)
                if not future_onset:
                    left, right = onset.span("date")
                    text = text[:left] + " " * (right - left) + text[right:]
        # An open lower bound cannot prove that observations continued into
        # the task year, or that they ended at the end of the starting year.
        open_start = (rf"\b(?:since|depuis|desde)\s+(?:(?:le|the|el)\s+)?(?P<since>{metadata_date})\b|"
                      rf"\bfrom\s+(?P<from>{metadata_date})\s+(?:onwards?|to\s+(?:date|present))\b")
        def remove_open(match):
            nonlocal unknown_open_period
            expression = match.group('since') or match.group('from')
            lower = _parse_date(expression)
            if lower is None:
                observed_year = re.search(year, expression)
                lower = date(int(observed_year.group()), 1, 1) if observed_year else None
            if lower and lower > end:
                periods.append((lower, date.max))
            else:
                unknown_open_period = True
            return " "
        text = re.sub(open_start, remove_open, text, flags=re.IGNORECASE)
        # Reuse the shared count recognizer and its original offsets. Remove
        # both endpoints when the measured quantity is a range, rather than
        # leaving its first endpoint behind as a spurious observation year.
        connector = r"(?:[-\u2013\u2014\u2212]|to|through|and|et|à)"
        count_spans = []
        for mention in count_mentions(text):
            left, right = mention['char_start'], mention['char_end']
            leading = re.search(NUMBER_TOKEN + rf"\s*{connector}\s*$", text[:left], re.IGNORECASE)
            if leading:
                left = leading.start()
            if not re.search(r"\b(?:in|during|for|year|en|em)\s+$", text[max(0, left - 24):left], re.IGNORECASE):
                count_spans.append((left, right))
        metric = r"(?:cases?|deaths?|patients?|samples?|cas|décès|casos|óbitos)"
        reverse_count = rf"\b{metric}\s*[:=]\s*{NUMBER_TOKEN}(?:\s*{connector}\s*{NUMBER_TOKEN})?"
        count_spans.extend(match.span() for match in re.finditer(reverse_count, text, re.IGNORECASE))
        for left, right in sorted(set(count_spans), reverse=True):
            text = text[:left] + " " * (right - left) + text[right:]
        # A measured four-digit count is not a date. Check only after its
        # quantity span is masked, but before genuine week dates are removed.
        if current_observation and not future_onset and not re.search(year, text):
            unknown_open_period = True
        # Explicit week/year pairs already carry their narrower interval.
        for week_pattern in _YEAR_WEEK_PATTERNS:
            text = re.sub(week_pattern, " ", text, flags=re.IGNORECASE)
        # A full date/range is more specific than the year embedded in it.
        for match in re.finditer(iso_range, text, re.IGNORECASE):
            first, last = _parse_date(match.group(1)), _parse_date(match.group(2))
            if first and last and first <= last:
                periods.append((first, last))
        remaining = re.sub(iso_range, " ", text, flags=re.IGNORECASE)
        for match in re.finditer(r"\b((?:19|20)\d{2}-\d{2}-\d{2})\b", remaining):
            value = _parse_date(match.group(1))
            if value:
                periods.append((value, value))
        remaining = re.sub(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b", " ", remaining)
        for match in re.finditer(r"\b((?:19|20)\d{2})\s*(?:[-\u2013\u2014\u2212]|to|through)\s*((?:19|20)\d{2})\b", remaining):
            first, last = int(match.group(1)), int(match.group(2))
            if first <= last:
                periods.append((date(first, 1, 1), date(last, 12, 31)))
        remaining = re.sub(r"\b(?:19|20)\d{2}\s*(?:[-\u2013\u2014\u2212]|to|through)\s*(?:19|20)\d{2}\b", " ", remaining)
        periods.extend((date(int(year), 1, 1), date(int(year), 12, 31))
                       for year in re.findall(r"\b((?:19|20)\d{2})\b", remaining))
    if any(first <= end and last >= start for first, last in periods):
        return "match"
    if unknown_open_period or not periods:
        return "candidate"
    return "mismatch"


def _source_positive_target_verification(entry: dict, state: dict) -> dict:
    from ..disease_identity import disease_names
    task = _task_for_screening(state)
    # These are source statements. Publisher, URL, query and classifier hints
    # cannot establish the disease, place or observation period being reported.
    texts = [str(entry.get(key) or "") for key in ("title", "name", "source_title", "snippet")]
    text = " ".join(texts)
    def contains(value, source_text=text):
        return bool(value and re.search(r"(?<!\w)" + re.escape(str(value)) + r"(?!\w)", source_text, re.IGNORECASE))
    names = set(disease_names(task.get("disease"))) | set(task.get("disease_aliases") or [])
    disease_fit = "match" if any(contains(name) for name in names) else "candidate"
    location = " ".join(str(task.get("location") or "").split())
    place_names = [location]
    us_target = location.casefold() in {"united states", "united states of america", "usa", "u.s.a.", "us", "u.s."}
    if us_target:
        place_names = ["United States", "United States of America", "USA", "U.S.A.", "U.S."]
    place_match = any(contains(name) for name in place_names) or (us_target and re.search(r"(?<!\w)US(?!\w)", text))
    geography_fit = "match" if place_match else "candidate"
    def reporting_date_fit(source_texts):
        # A year inside a disease name (e.g. coronavirus disease 2019) is not a period.
        period_texts = list(source_texts)
        for name in sorted(names, key=len, reverse=True):
            without_year = re.sub(r"\b(?:19|20)\d{2}\b", " ", name)
            period_texts = [re.sub(r"(?<!\w)" + re.escape(name) + r"(?!\w)", lambda _: without_year, value, flags=re.IGNORECASE)
                            for value in period_texts]
        return _source_reporting_date_fit(period_texts, state)

    date_fit = reporting_date_fit(texts)
    # Separate source observations cannot supply different parts of one match.
    # Preserve initials such as U.S. while splitting sentences and report items.
    statements = [part for value in texts
                  for part in re.split(r"(?<!\b[A-Za-z])\.(?:\s+|$)|[!?;\n]+", value) if part.strip()]
    associated_target = any(
        any(contains(name, statement) for name in names)
        and (any(contains(name, statement) for name in place_names)
             or (us_target and re.search(r"(?<!\w)US(?!\w)", statement)))
        and reporting_date_fit([statement]) == "match"
        for statement in statements
    )
    result = {"target_verification_status": "unverified_candidate",
              "target_verification_reason": "source disease, geography or reporting period remains unverified",
              "triage_role": "task_record_collection_candidate", "source_role": None,
              "screening_decision": None, "disease_fit": disease_fit,
              "geography_fit": geography_fit, "date_fit": date_fit}
    if str(entry.get("source_disease_relevance_status") or "").strip().lower() in {"unrelated_disease", "incompatible_disease"}:
        result.update(target_verification_status="unrelated_disease", disease_fit="mismatch",
                      target_verification_reason="source disease identity is incompatible with the task disease",
                      triage_role="context_only", source_role="context_source", screening_decision="include_for_context_fetch")
    elif date_fit == "mismatch":
        result.update(target_verification_status="temporal_mismatch",
                      target_verification_reason="explicit source reporting period does not overlap task window",
                      triage_role="context_only", source_role="context_source", screening_decision="include_for_context_fetch")
    elif disease_fit == geography_fit == date_fit == "match" and associated_target:
        result.update(target_verification_status="verified_target",
                      target_verification_reason="source explicitly states target disease, geography and an overlapping reporting period",
                      triage_role="verified_target_collection", source_role="data_source", screening_decision="include")
    return result


def assess_source_task_fit(entry: dict, state: dict, document: dict | None = None) -> dict:
    """Assess source statements, retaining the origin of provisional judgments.

    Only complete readable content bound to this source can replace discovery
    metadata. Explicit and unversioned negative constraints remain boundaries.
    A document title may be inherited from search, so only its body is evidence.
    """
    source = dict(entry)
    origin = "discovery_metadata"
    content_hash = None
    usable_document = False
    if document is not None:
        bound = bool(entry.get("source_id") and document.get("source_id") and entry["source_id"] == document["source_id"])
        issues = " ".join(str(value) for value in document.get("quality_issues") or []).lower()
        try:
            status = int(document.get("http_status_code") if document.get("http_status_code") is not None else 200)
        except (TypeError, ValueError):
            status = 0
        readable = (document.get("content_readable") is True
                    and 200 <= status < 300
                    and document.get("request_success") is not False
                    and document.get("parse_eligible") is not False
                    and document.get("raw_content_complete") is not False
                    and not document.get("acquisition_incomplete")
                    and not document.get("parse_error")
                    and document.get("parse_status") != "parsed_partial"
                    and document.get("quality_status") not in {"unusable", "error_page"}
                    and document.get("acquisition_status") not in {"blocked", "shell", "request_error", "budget_exhausted", "parse_error"}
                    and bool(str(document.get("clean_text") or "").strip())
                    and not any(term in issues for term in ("error_page", "access_denied", "blocked_page")))
        usable_document = bound and readable
        if usable_document:
            source = {**source, "title": "", "name": "", "source_title": "",
                      "snippet": document.get("clean_text") or ""}
            origin = "fetched_content"
            content_hash = document.get("content_hash")

    previous_origin = entry.get("task_fit_evidence_origin")
    replace_provisional = usable_document and previous_origin == "discovery_metadata"
    negative_values = {"mismatch", "wrong_geography", "wrong_period", "wrong_disease", "unrelated"}
    negatives = {key: entry[key] for key in ("disease_fit", "geography_fit", "date_fit", "period_fit")
                 if entry.get(key) in negative_values}
    negative_classifier = str(entry.get("source_disease_relevance_status") or "").strip().lower() in {
        "unrelated_disease", "incompatible_disease"}
    aggregate_dimensions = {"wrong_period": "date_fit", "temporal_mismatch": "date_fit",
        "wrong_geography": "geography_fit", "geography_mismatch": "geography_fit", "broader_than_task": "geography_fit",
        "wrong_disease": "disease_fit", "unrelated_disease": "disease_fit"}
    aggregate_value = str(entry.get("target_fit_status") or "").strip().lower()
    aggregate_negative = aggregate_value in (set(aggregate_dimensions) | {
        "context_only", "non_target_or_context", "partial_overlap", "task_fit_mismatch"})
    if aggregate_value in aggregate_dimensions:
        negatives.setdefault(aggregate_dimensions[aggregate_value], "mismatch")
    if replace_provisional:
        source.pop("source_disease_relevance_status", None)
    elif negatives or negative_classifier or aggregate_negative:
        # Do not relabel an old unexplained rejection as provisional metadata:
        # doing so would make it erasable on the next call.
        origin = previous_origin if previous_origin in {"discovery_metadata", "fetched_content", "explicit_constraint"} else "explicit_constraint"
        content_hash = entry.get("task_fit_content_hash") if origin == "fetched_content" else None

    result = _source_positive_target_verification(source, state)
    if not replace_provisional:
        for key, value in negatives.items():
            result["date_fit" if key == "period_fit" else key] = value
            result.update(target_verification_status="task_fit_mismatch",
                          target_verification_reason="preserved source task-fit contradiction",
                          triage_role="context_only", source_role="context_source",
                          screening_decision="include_for_context_fetch")
    if aggregate_negative and not replace_provisional:
        result.update(target_verification_status="task_fit_mismatch",
                      target_verification_reason="preserved aggregate source task-fit contradiction",
                      triage_role="context_only", source_role="context_source",
                      screening_decision="include_for_context_fetch")
    result.update(task_fit_evidence_origin=origin, task_fit_content_hash=content_hash,
                  task_fit_assessment_version=2,
                  target_fit_status=(aggregate_value if aggregate_negative and not replace_provisional else
                                     "verified_target" if result["target_verification_status"] == "verified_target"
                                     else result["triage_role"]))
    if replace_provisional:
        # The discovery classifier must not retain contrary proof after its own
        # provisional inputs have been replaced by the received source body.
        result["source_disease_relevance_status"] = ("target_disease_match" if result["disease_fit"] == "match" else "unknown")
        final_role = str(entry.get("source_role_final") or "").lower()
        flags = set(entry.get("routing_flags") or []) | set(entry.get("screening_flags") or [])
        boundary = (entry.get("blocked_from_fetch") or entry.get("source_excluded_by_human_review")
                    or entry.get("requires_human_review")
                    or final_role in {"excluded", "context_only", "validation", "validation_reserved", "validation_source", "search_endpoint", "needs_human_review"}
                    or entry.get("final_screening_decision") in {"exclude", "reserved_for_validation"}
                    or bool(flags & {"context_only", "blocked_from_structured_extraction", "user_excluded"}))
        if not boundary and result["target_verification_status"] == "verified_target":
            result.update(source_role_final="collection", final_screening_decision="include_for_content_fetch",
                          fetch_purpose="data_extraction", ready_for_content_fetch=True)
        elif not boundary and result["triage_role"] == "context_only":
            result.update(source_role_final="context", final_screening_decision="include_for_context_fetch",
                          fetch_purpose="context_grounding")
    return result


def _direct_target_verification(entry: dict, state: DataCollectionState | dict) -> dict:
    if universal_queries_enabled():
        return assess_source_task_fit(entry, state)
    task_pairs = _task_week_pairs(state)
    explicit_pairs = _explicit_year_week_pairs(_entry_target_text(entry))
    if str(entry.get("source_disease_relevance_status") or "").strip().lower() in {
        "unrelated_disease",
        "incompatible_disease",
    }:
        return {
            "target_verification_status": "unrelated_disease",
            "target_verification_reason": (
                "source disease identity is incompatible with the task disease"
            ),
            "triage_role": "context_only",
            "date_fit": "candidate",
            "source_role": "context_source",
            "screening_decision": "include_for_context_fetch",
        }
    if task_pairs and explicit_pairs:
        if task_pairs & explicit_pairs:
            return {
                "target_verification_status": "verified_target",
                "target_verification_reason": "explicit source year/week overlaps task window",
                "triage_role": "verified_target_collection",
                "date_fit": "match",
                "source_role": "data_source",
                "screening_decision": "include",
            }
        return {
            "target_verification_status": "temporal_mismatch",
            "target_verification_reason": (
                "explicit source year/week does not overlap task window"
            ),
            "triage_role": "context_only",
            "date_fit": "mismatch",
            "source_role": "context_source",
            "screening_decision": "include_for_context_fetch",
        }
    return {
        "target_verification_status": "unverified_candidate",
        "target_verification_reason": "no explicit source year/week was parsed",
        "triage_role": "task_record_collection_candidate",
        "date_fit": "candidate",
        "source_role": None,
        "screening_decision": None,
    }


def _apply_direct_triage_verification(
    entry: dict,
    result: SourceScreeningResult,
    state: DataCollectionState | dict,
    *,
    collection_mode: str,
) -> tuple[dict, dict]:
    if (entry.get("historical_snapshot") or {}).get("verification_status") == "index_only":
        updated = _preserve_historical_metadata_only_routing(entry)
        return updated, {key: updated[key] for key in (
            "target_verification_status", "target_verification_reason", "triage_role",
            "disease_fit", "geography_fit", "date_fit",
        )}
    verification = _direct_target_verification(entry, state)
    triage = dict(verification)
    if universal_queries_enabled():
        updated = dict(entry)
        updated.update({key: verification[key] for key in (
            "target_verification_status", "target_verification_reason", "triage_role",
            "disease_fit", "geography_fit", "date_fit", "task_fit_evidence_origin",
            "task_fit_content_hash", "task_fit_assessment_version")})
        updated["target_fit_status"] = ("verified_target" if verification["target_verification_status"] == "verified_target"
                                         else verification["triage_role"])
        flags = list(updated.get("screening_flags") or result.screening_flags or [])
        provisional = "ambiguous_source_role" in flags
        provisional_block = provisional and updated.get("blocked_from_fetch_reason") in {None, "irrelevant_source", "ambiguous_source_role"}
        explicit_boundary = bool(updated.get("blocked_from_fetch") and not provisional_block)
        if not explicit_boundary:
            if verification["target_verification_status"] in {"unrelated_disease", "temporal_mismatch", "task_fit_mismatch"}:
                updated.update(source_role="context_source", source_role_fit="context_source",
                               screening_decision="include", screening_reason=verification["target_verification_reason"])
            elif provisional or verification["target_verification_status"] == "verified_target":
                updated.update(source_role="data_source", source_role_fit="data_source",
                               screening_decision="include", screening_reason=verification["target_verification_reason"])
                flags = [flag for flag in flags if flag != "ambiguous_source_role"]
                if provisional:
                    flags.append("source_function_unverified")
                    if provisional_block:
                        updated.update(blocked_from_fetch=False, blocked_from_fetch_reason=None)
            updated["screening_flags"] = flags
        return updated, triage
    if collection_mode != "direct_collection":
        return entry, triage

    updated = dict(entry)
    flags = list(updated.get("screening_flags") or result.screening_flags or [])
    if verification["target_verification_status"] == "unrelated_disease":
        if "target_disease_mismatch" not in flags:
            flags.append("target_disease_mismatch")
        updated.update(
            {
                "source_role": "context_source",
                "source_role_final": "context",
                "screening_decision": "include_for_context_fetch",
                "screening_confidence": min(
                    float(result.screening_confidence or 0.0), 0.40
                ),
                "screening_reason": verification["target_verification_reason"],
                "screening_flags": flags,
                "target_fit_status": "unrelated_disease",
                "target_verification_status": "unrelated_disease",
                "target_verification_reason": verification[
                    "target_verification_reason"
                ],
                "triage_role": "context_only",
                "date_fit": verification["date_fit"],
                "geography_fit": "candidate",
                "disease_fit": "mismatch",
                "source_role_fit": "context_source",
            }
        )
    elif verification["target_verification_status"] == "temporal_mismatch":
        if "target_temporal_mismatch" not in flags:
            flags.append("target_temporal_mismatch")
        updated["source_role"] = verification["source_role"]
        updated["screening_decision"] = verification["screening_decision"]
        updated["screening_confidence"] = min(
            float(result.screening_confidence or 0.0), 0.60
        )
        updated["screening_reason"] = verification["target_verification_reason"]
        updated["screening_flags"] = flags
        updated["target_fit_status"] = "temporal_mismatch"
        updated["target_verification_status"] = verification["target_verification_status"]
        updated["target_verification_reason"] = verification["target_verification_reason"]
        updated["triage_role"] = verification["triage_role"]
        updated["date_fit"] = verification["date_fit"]
        updated["geography_fit"] = "candidate"
        updated["disease_fit"] = "candidate"
        updated["source_role_fit"] = verification["source_role"]
    elif verification["target_verification_status"] == "verified_target":
        updated["target_fit_status"] = "verified_target"
        updated["target_verification_status"] = verification["target_verification_status"]
        updated["target_verification_reason"] = verification["target_verification_reason"]
        updated["triage_role"] = verification["triage_role"]
        updated["date_fit"] = verification["date_fit"]
        updated["geography_fit"] = "match"
        updated["disease_fit"] = "candidate"
        updated["source_role_fit"] = verification["source_role"]
        updated["source_role"] = updated.get("source_role") or verification["source_role"]
        updated["screening_decision"] = updated.get("screening_decision") or verification[
            "screening_decision"
        ]
    else:
        updated["target_fit_status"] = "task_record_collection_candidate"
        updated["target_verification_status"] = "candidate_task_record_source"
        updated["target_verification_reason"] = verification["target_verification_reason"]
        updated["triage_role"] = verification["triage_role"]
        updated["date_fit"] = verification["date_fit"]
        updated["geography_fit"] = "candidate"
        updated["disease_fit"] = "candidate"
        updated["source_role_fit"] = updated.get("source_role") or result.source_role
    if universal_queries_enabled():
        updated.update({key: verification[key] for key in ("disease_fit", "geography_fit", "date_fit")})
    if (
        verification["target_verification_status"] != "unrelated_disease"
        and _entry_requires_source_trust_review(updated)
    ):
        reason = (
            "Source identity is news/social/secondary/unknown; task-compatible "
            "metrics require human review before strict collection use."
        )
        updated = _apply_source_trust_review_routing(updated, reason=reason)
        triage.update(
            {
                "target_verification_status": "source_trust_requires_human_review",
                "target_verification_reason": reason,
                "triage_role": "needs_human_review",
                "date_fit": updated.get("date_fit") or triage.get("date_fit"),
                "source_role": "context_source",
                "screening_decision": "needs_human_review",
            }
        )
    return updated, triage


def _llm_source_critic_review_blocks_fetch() -> bool:
    value = (
        get_env("LLM_SOURCE_CRITIC_REVIEW_BLOCKS_FETCH") or ""
    ).strip().lower()
    return value != "false"


_SOURCE_CRITIC_SEARCH_DISCOVERY_METHODS = {
    "live_search_result",
    "fixture_search_result",
}
_SOURCE_CRITIC_BLOCK_DECISIONS = {
    "not_task_relevant",
    "exclude_from_task",
    "no_extractable_data",
}
_SOURCE_CRITIC_CONTEXT_DECISIONS = {
    "no_extractable_data",
    "suitable_for_context",
    "collection_support_only",
}
_SOURCE_CRITIC_BLOCK_FETCH_RECOMMENDATIONS = {
    "block_fetch",
    "fetch_only_after_human_review",
    "context_fetch_only",
}
_SOURCE_CRITIC_HARD_BLOCK_FLAGS = {
    "critical_disease_mismatch",
    "disease_mismatch",
    "source_not_about_task",
    "outside_task_scope",
}
_SOURCE_CRITIC_ALLOWED_ROLES = {
    "collection",
    "validation",
    "context",
    "collection_support",
    "search_endpoint",
    "excluded",
    "needs_human_review",
}
_SOURCE_CRITIC_DECISION_ALIASES = {
    "exclude": "not_task_relevant",
    "excluded": "not_task_relevant",
    "not_relevant": "not_task_relevant",
    "context_only": "no_extractable_data",
}


def _text_blob(entry: dict) -> str:
    """Combine relevant metadata fields into a single lowercase text blob."""

    parts: list[str] = []
    keys = (("title", "snippet", "canonical_url", "url") if universal_queries_enabled()
            else ("title", "publisher", "source_type", "source_purpose", "notes"))
    for key in keys:
        value = entry.get(key)
        if value:
            parts.append(str(value))
    for key in ("matched_terms", "expected_fields"):
        items = entry.get(key) or []
        if items:
            parts.append(" ".join(str(item) for item in items))
    query_used = entry.get("query_used")
    if query_used:
        parts.append(str(query_used))
    return " ".join(parts).lower()


def _source_content_text(entry: dict) -> str:
    """Source-only text for signal scans.

    Excludes `query_used` because the query string describes *how we found*
    the source, not *what the source itself contains*. Including the query
    would otherwise make every candidate inherit terms like "cases" or
    "deaths" from the discovery query and incorrectly classify context pages
    as data sources.
    """

    parts: list[str] = []
    for key in ("title", "publisher", "source_type", "source_purpose", "notes"):
        value = entry.get(key)
        if value:
            parts.append(str(value))
    matched_terms = entry.get("matched_terms") or []
    if matched_terms:
        parts.append(" ".join(str(item) for item in matched_terms))
    return " ".join(parts).lower()


def _contains_any(text: str, terms: list[str]) -> bool:
    for term in terms:
        if not term:
            continue
        if term.lower() in text:
            return True
    return False


# ---------------------------------------------------------------------------
# Role classification
# ---------------------------------------------------------------------------


def _classify_source_role(
    entry: dict,
    policy: SourceScreeningPolicy,
) -> tuple[str, list[str]]:
    """Return (source_role, flags) using rule-based metadata classification."""

    flags: list[str] = []

    url = entry.get("canonical_url") or entry.get("url") or ""
    publisher = entry.get("publisher")
    expected_fields = entry.get("expected_fields") or []
    # Signal scans use source-only text so the discovery query string does
    # not bleed data-source terms into every candidate.
    source_text = _source_content_text(entry)

    # 1. Placeholder URI
    for prefix in policy.placeholder_uri_prefixes:
        if url.startswith(prefix):
            return "placeholder_source", flags

    # 2. The page URL/role, not its publisher, identifies a query interface.
    if universal_queries_enabled():
        if _identity_search_endpoint(entry, urlsplit(url).hostname):
            return "search_endpoint", flags
    elif publisher in policy.search_endpoint_publishers:
        return "search_endpoint", flags

    # 3. expected_fields indicates case/death/date/location
    data_fields = {"cases", "deaths", "date", "location"}
    if not universal_queries_enabled() and any(f in data_fields for f in expected_fields):
        return "data_source", flags

    # 4. Source-only text contains data-source signals
    if _contains_any(source_text, policy.data_source_signals):
        return "data_source", flags

    # 5. expected_fields indicates context/definition material
    context_fields = {"case_definition", "disease", "virus_or_syndrome"}
    if any(f in context_fields for f in expected_fields):
        return "context_source", flags

    # 6. Source-only text contains context signals
    if _contains_any(source_text, policy.context_source_signals):
        return "context_source", flags

    # Unknown function is a reason to inspect content, not proof of mismatch.
    if universal_queries_enabled():
        flags.append("source_function_unverified")
        return "data_source", flags
    # Legacy classification contract.
    flags.append("ambiguous_source_role")
    return "irrelevant_source", flags


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------


def _task_for_screening(screening_criteria: dict | None) -> dict:
    criteria = screening_criteria or {}
    structured = criteria.get("structured_task") or criteria.get("task") or {}
    collection = criteria.get("collection_spec") or {}
    return {
        "disease": str(
            criteria.get("disease")
            or structured.get("disease")
            or collection.get("disease")
            or ""
        ).strip(),
        "location": str(
            criteria.get("location")
            or structured.get("location")
            or structured.get("geography")
            or collection.get("geography")
            or collection.get("location")
            or ""
        ).strip(),
        "start_date": str(
            criteria.get("start_date")
            or structured.get("start_date")
            or collection.get("start_date")
            or ""
        ).strip(),
        "end_date": str(
            criteria.get("end_date")
            or structured.get("end_date")
            or collection.get("end_date")
            or ""
        ).strip(),
    }


def _screening_profile_for_role(
    role: str,
    *,
    entry: dict,
    task: dict,
) -> tuple[str, float, str]:
    disease = task.get("disease") or "the requested disease"
    location = task.get("location") or "the requested location"
    date_range = " to ".join(
        value for value in (task.get("start_date"), task.get("end_date")) if value
    )
    title = entry.get("title") or entry.get("canonical_url") or entry.get("url") or "source"
    if role == "data_source":
        return (
            "include",
            0.90,
            (
                f"Source metadata for {title} indicates likely extractable "
                f"{disease} public-health data for {location}"
                + (f" during {date_range}" if date_range else "")
                + "."
            ),
        )
    if role == "context_source":
        return (
            "include",
            0.78,
            (
                f"Source appears useful as context for {disease} collection "
                f"for {location}, but may not contain direct extractable counts."
            ),
        )
    if role == "search_endpoint":
        return (
            "uncertain",
            0.70,
            (
                "Source is a search endpoint and should be expanded into "
                "document-level candidates before content fetch."
            ),
        )
    if role == "placeholder_source":
        return (
            "uncertain",
            0.65,
            "Source is an internal placeholder URI and should be deferred.",
        )
    return (
        "exclude",
        0.60,
        (
            f"Source metadata does not clearly match the requested "
            f"{disease} data collection task for {location}."
        ),
    )


def _screen_entry(
    entry: dict,
    policy: SourceScreeningPolicy,
    screening_criteria: dict | None,  # noqa: ARG001 — reserved for future LLM step
) -> SourceScreeningResult:
    role, role_flags = _classify_source_role(entry, policy)
    decision, confidence, reason = _screening_profile_for_role(
        role,
        entry=entry,
        task=_task_for_screening(screening_criteria),
    )

    flags = list(role_flags)
    if universal_queries_enabled() and "source_function_unverified" in flags:
        confidence = 0.60
        reason = "Source function remains unknown; acquire content to assess task data and navigation value."

    if not (entry.get("canonical_url") or entry.get("url")):
        flags.append("missing_url")
        confidence = min(confidence, 0.50)
    if not entry.get("source_type"):
        flags.append("missing_source_type")
        confidence = min(confidence, 0.50)

    allowed_fields = set(policy.target_data_fields) | set(policy.context_fields)
    expected_extractable = [
        f for f in (entry.get("expected_fields") or []) if f in allowed_fields
    ]

    return SourceScreeningResult(
        source_id=entry.get("source_id", ""),
        screening_decision=decision,
        screening_confidence=confidence,
        screening_reason=reason,
        source_role=role,
        screening_flags=flags,
        expected_extractable_fields=expected_extractable,
    )


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


_CRITIC_PROFILE: dict[str, tuple[str, float, str, str | None]] = {
    "data_source": (
        "include",
        0.88,
        "Critic confirms: source has clear data-oriented signals and expected "
        "extractable case/death/date/location fields.",
        None,
    ),
    "context_source": (
        "include",
        0.74,
        "Critic confirms: source is relevant context for the task disease terminology "
        "or case definition, but is not necessarily a case-count source.",
        "context_only_or_low_case_data_signal",
    ),
    "search_endpoint": (
        "uncertain",
        0.76,
        "Critic flags: search endpoints should not be parsed as final source "
        "documents — they must be expanded into article-level candidates first.",
        "requires_article_level_expansion",
    ),
    "placeholder_source": (
        "uncertain",
        0.72,
        "Critic flags: internal seed placeholder URI; should be deferred until "
        "real connector implementation provides actual sources.",
        "requires_connector_implementation",
    ),
    "irrelevant_source": (
        "exclude",
        0.65,
        "Critic confirms: source metadata does not clearly tie it to human "
        "case or outbreak data for the task disease.",
        None,
    ),
}


def _critic_review_entry(
    entry: dict,
    policy: SourceScreeningPolicy,  # noqa: ARG001 — reserved for future LLM step
) -> SourceCriticResult:
    role = entry.get("source_role") or "irrelevant_source"
    screening_decision = entry.get("screening_decision")

    critic_decision, critic_confidence, critic_reason, critic_flag = _CRITIC_PROFILE[role]
    if universal_queries_enabled() and "source_function_unverified" in (entry.get("screening_flags") or []):
        critic_confidence = 0.60
        critic_reason = "Content inspection is needed; metadata has not established extractable task observations."

    if role == "data_source":
        agrees = screening_decision == "include"
    elif role == "context_source":
        agrees = screening_decision == "include"
    elif role == "search_endpoint":
        agrees = screening_decision == "uncertain"
    elif role == "placeholder_source":
        agrees = screening_decision == "uncertain"
    else:  # irrelevant_source
        agrees = screening_decision == "exclude"

    flags = [critic_flag] if critic_flag else []

    return SourceCriticResult(
        source_id=entry.get("source_id", ""),
        critic_decision=critic_decision,
        critic_confidence=critic_confidence,
        critic_reason=critic_reason,
        critic_agrees_with_screening=agrees,
        critic_flags=flags,
    )


def _append_flag(flags: list[str], flag: str) -> list[str]:
    if flag not in flags:
        flags.append(flag)
    return flags


def _as_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _as_confidence(value) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, numeric))


def _source_critic_role(value: str | None, *, default: str = "needs_human_review") -> str:
    role = str(value or "").strip().lower()
    return role if role in _SOURCE_CRITIC_ALLOWED_ROLES else default


def _source_critic_decision_from_legacy(assessment: dict) -> str:
    proposed_decision = str(
        assessment.get("proposed_screening_decision") or ""
    ).strip().lower()
    proposed_role = str(assessment.get("proposed_source_role") or "").strip().lower()
    if proposed_decision in {"exclude", "excluded"}:
        return "not_task_relevant"
    if bool(assessment.get("validation_candidate_risk", False)):
        return "suitable_for_validation"
    if bool(assessment.get("context_only_risk", False)) or proposed_role in {
        "context",
        "collection_support",
    }:
        return "suitable_for_context"
    if bool(assessment.get("needs_human_review", False)):
        return "needs_human_review"
    return "suitable_for_collection"


def _normalize_source_critic_assessment(assessment: dict, source_id: str) -> dict:
    decision = str(
        assessment.get("critic_decision")
        or assessment.get("source_critic_decision")
        or _source_critic_decision_from_legacy(assessment)
    ).strip().lower()
    decision = _SOURCE_CRITIC_DECISION_ALIASES.get(decision, decision)
    risk_flags = _as_str_list(assessment.get("risk_flags"))
    if bool(assessment.get("semantic_leakage_risk", False)):
        risk_flags.append("semantic_leakage_risk")
    if bool(assessment.get("context_only_risk", False)):
        risk_flags.append("only_background_or_context")
    if bool(assessment.get("validation_candidate_risk", False)):
        risk_flags.append("validation_candidate_risk")
    risk_flags = list(dict.fromkeys(risk_flags))

    reason = str(
        assessment.get("reasoning_summary")
        or assessment.get("reason")
        or assessment.get("credibility_reason")
        or assessment.get("human_review_reason")
        or ""
    ).strip()
    recommended_role = _source_critic_role(
        assessment.get("recommended_role")
        or assessment.get("proposed_source_role")
        or ("context" if decision in _SOURCE_CRITIC_CONTEXT_DECISIONS else None),
        default="needs_human_review",
    )
    fetch_recommendation = str(
        assessment.get("fetch_recommendation") or ""
    ).strip().lower()
    if not fetch_recommendation:
        if decision in {"not_task_relevant", "exclude_from_task"}:
            fetch_recommendation = "block_fetch"
        elif decision in _SOURCE_CRITIC_CONTEXT_DECISIONS:
            fetch_recommendation = "context_fetch_only"
        elif bool(assessment.get("needs_human_review", False)):
            fetch_recommendation = "fetch_only_after_human_review"
        else:
            fetch_recommendation = "allow_fetch"
    review_required = bool(
        assessment.get("review_required", False)
        or assessment.get("needs_human_review", False)
        or fetch_recommendation == "fetch_only_after_human_review"
    )
    return {
        **assessment,
        "source_id": str(assessment.get("source_id") or source_id or ""),
        "critic_decision": decision,
        "risk_flags": risk_flags,
        "recommended_role": recommended_role,
        "fetch_recommendation": fetch_recommendation,
        "review_required": review_required,
        "confidence": _as_confidence(assessment.get("confidence")),
        "reasoning_summary": reason,
        "warnings": _as_str_list(assessment.get("warnings")),
    }


def _source_critic_fetch_policy(
    assessment: dict,
    *,
    review_blocks_fetch: bool,
) -> dict:
    decision = str(assessment.get("critic_decision") or "").strip().lower()
    fetch_recommendation = str(
        assessment.get("fetch_recommendation") or ""
    ).strip().lower()
    risk_flags = set(_as_str_list(assessment.get("risk_flags")))
    hard_block = (
        decision in {"not_task_relevant", "exclude_from_task"}
        or fetch_recommendation == "block_fetch"
        or bool(risk_flags & _SOURCE_CRITIC_HARD_BLOCK_FLAGS)
    )
    policy_block = (
        hard_block
        or decision in _SOURCE_CRITIC_BLOCK_DECISIONS
        or fetch_recommendation in _SOURCE_CRITIC_BLOCK_FETCH_RECOMMENDATIONS
    )
    should_block = policy_block and (review_blocks_fetch or hard_block)
    context_only = (
        decision in _SOURCE_CRITIC_CONTEXT_DECISIONS
        or fetch_recommendation == "context_fetch_only"
    )
    review_required = bool(assessment.get("review_required", False)) or (
        should_block and not hard_block
    )
    if hard_block:
        final_decision = "exclude"
        final_role = "excluded"
        status = "excluded"
    elif should_block and context_only:
        final_decision = "include_for_context_fetch"
        final_role = _source_critic_role(
            assessment.get("recommended_role"), default="context"
        )
        if final_role not in {"context", "collection_support"}:
            final_role = "context"
        status = "ready_for_context_fetch"
    elif should_block or review_required:
        final_decision = "needs_human_review"
        final_role = "needs_human_review"
        status = "needs_human_review"
    else:
        final_decision = None
        final_role = _source_critic_role(
            assessment.get("recommended_role"), default=""
        )
        status = None
    return {
        "should_block_fetch": should_block,
        "hard_block": hard_block,
        "context_only": context_only,
        "review_required": review_required,
        "final_decision": final_decision,
        "final_role": final_role,
        "status": status,
    }


def _is_search_endpoint_for_source_critic(entry: dict) -> bool:
    url = str(entry.get("canonical_url") or entry.get("url") or "").lower()
    if universal_queries_enabled():
        return _identity_search_endpoint(entry, urlsplit(url).hostname)
    role_values = {
        str(entry.get("source_role") or "").lower(),
        str(entry.get("source_role_final") or "").lower(),
        str(entry.get("role_hint") or "").lower(),
    }
    if role_values & {"search_endpoint", "placeholder_source"}:
        return True
    text = _source_content_text(entry)
    return (
        "pubmed" in text
        or "openalex" in text
        or "europe pmc" in text
        or "/search" in url
        or "search?" in url
        or "search=" in url
    )


def _source_critic_sort_key(entry: dict) -> tuple[int, int, int, str]:
    discovery_method = str(entry.get("discovery_method") or "")
    if discovery_method == "live_search_result":
        discovery_bucket = 0
    elif discovery_method == "fixture_search_result":
        discovery_bucket = 1
    else:
        discovery_bucket = 2
    search_rank = entry.get("search_rank")
    try:
        search_rank_int = int(search_rank)
    except (TypeError, ValueError):
        search_rank_int = 999_999
    priority = entry.get("priority")
    try:
        priority_int = int(priority)
    except (TypeError, ValueError):
        priority_int = 999_999
    return (
        discovery_bucket,
        search_rank_int,
        priority_int,
        str(entry.get("source_id") or ""),
    )


def _is_clearly_low_value_direct_context(entry: dict) -> bool:
    source_type = str(entry.get("source_type_final") or entry.get("source_type") or "").lower()
    product_type = str(entry.get("data_product_type") or "").lower()
    evidence_role = str(entry.get("expected_evidence_role") or "").lower()
    role = str(entry.get("source_role_final") or entry.get("source_role") or "").lower()
    specificity = str(entry.get("task_specificity") or "").lower()
    if source_type in {"social_media", "search_endpoint", "placeholder_source"}:
        return True
    if product_type in {
        "background_fact_sheet",
        "generic_background",
        "general_overview",
    }:
        return True
    return bool(
        role in {"context", "excluded"}
        and (
            specificity in {"generic", "background", "not_task_specific"}
            or evidence_role in {"background_context", "context_only"}
        )
    )


def _select_llm_source_critic_candidates(
    registry: list[dict],
    *,
    allowlist: set[str] | None,
    max_sources: int | None,
    collection_mode: str = "standard",
) -> tuple[set[str], dict[str, str], dict]:
    explicit_allowlist_used = allowlist is not None
    skipped_reasons: dict[str, str] = {}
    eligible: list[dict] = []
    target_official_must_fetch_ids = {
        str(entry.get("source_id") or "")
        for entry in registry
        if entry.get("must_fetch")
        and _is_known_official_public_health_source(entry)
        and entry.get("source_id")
    }
    verified_target_ids = {
        str(entry.get("source_id") or "")
        for entry in registry
        if str(entry.get("target_verification_status") or "").strip().lower()
        in {"verified_target", "verified"}
        and entry.get("source_id")
    }
    direct_fast_path_active = (
        not universal_queries_enabled()
        and collection_mode == "direct_collection"
        and not explicit_allowlist_used
        and (
            bool(target_official_must_fetch_ids)
            or bool(verified_target_ids)
        )
    )

    for entry in registry:
        source_id = str(entry.get("source_id") or "")
        if not source_id:
            continue
        if explicit_allowlist_used and source_id not in allowlist:
            skipped_reasons[source_id] = "source_not_in_explicit_critic_allowlist"
            continue
        if direct_fast_path_active:
            if source_id in target_official_must_fetch_ids:
                skipped_reasons[source_id] = (
                    "target_official_must_fetch_skips_llm_source_critic"
                )
                continue
            if source_id in verified_target_ids:
                skipped_reasons[source_id] = (
                    "verified_target_skips_llm_source_critic"
                )
                continue
            if _is_clearly_low_value_direct_context(entry):
                skipped_reasons[source_id] = (
                    "direct_low_value_context_skips_source_critic"
                )
                continue
        if not explicit_allowlist_used and _is_search_endpoint_for_source_critic(entry):
            skipped_reasons[source_id] = "search_endpoint_or_placeholder_not_critic_candidate"
            continue
        eligible.append(entry)

    selected_entries = sorted(eligible, key=_source_critic_sort_key)
    if max_sources is not None:
        selected_entries = selected_entries[:max_sources]
    selected_ids = {str(entry.get("source_id") or "") for entry in selected_entries}
    for entry in eligible:
        source_id = str(entry.get("source_id") or "")
        if source_id and source_id not in selected_ids:
            skipped_reasons[source_id] = "not_selected_by_source_critic_limit"

    skipped_reason_counts = Counter(skipped_reasons.values())
    summary = {
        "selection_mode": "explicit_allowlist"
        if explicit_allowlist_used
        else "auto_priority",
        "explicit_allowlist_used": explicit_allowlist_used,
        "collection_mode": collection_mode,
        "direct_target_official_fast_path": direct_fast_path_active,
        "max_sources": max_sources,
        "eligible_candidate_count": len(eligible),
        "selected_candidate_count": len(selected_ids),
        "skipped_candidate_count": len(skipped_reasons),
        "selected_source_ids": [str(entry.get("source_id") or "") for entry in selected_entries],
        "skipped_reason_counts": dict(skipped_reason_counts),
    }
    return selected_ids, skipped_reasons, summary


def _apply_llm_source_critic_assessment(
    entry: dict,
    assessment: dict,
    *,
    review_blocks_fetch: bool,
) -> dict:
    updated = dict(entry)
    source_id = str(updated.get("source_id") or "")
    assessment = _normalize_source_critic_assessment(assessment, source_id)
    fetch_policy = _source_critic_fetch_policy(
        assessment,
        review_blocks_fetch=review_blocks_fetch,
    )
    routing_flags = list(updated.get("routing_flags") or [])
    critic_flags = list(updated.get("critic_flags") or [])

    semantic_leakage_risk = bool(assessment.get("semantic_leakage_risk", False))
    needs_human_review = bool(
        assessment.get("needs_human_review", False)
        or assessment.get("review_required", False)
    )
    context_only_risk = bool(assessment.get("context_only_risk", False))
    validation_candidate_risk = bool(
        assessment.get("validation_candidate_risk", False)
    )
    risk_flags = _as_str_list(assessment.get("risk_flags"))
    for risk_flag in risk_flags:
        _append_flag(routing_flags, f"llm_source_critic:{risk_flag}")
        _append_flag(critic_flags, f"llm_source_critic:{risk_flag}")

    if semantic_leakage_risk:
        _append_flag(routing_flags, "llm_semantic_leakage_risk")
        _append_flag(critic_flags, "llm_semantic_leakage_risk")
    if context_only_risk:
        _append_flag(routing_flags, "llm_context_only_risk")
        _append_flag(critic_flags, "llm_context_only_risk")
    if validation_candidate_risk:
        _append_flag(routing_flags, "llm_validation_candidate_risk")
        _append_flag(critic_flags, "llm_validation_candidate_risk")
    if needs_human_review:
        _append_flag(routing_flags, "llm_human_review_recommended")
        _append_flag(critic_flags, "llm_human_review_recommended")

    if fetch_policy["should_block_fetch"]:
        _append_flag(routing_flags, "llm_source_critic_block_fetch")
        _append_flag(critic_flags, "llm_source_critic_block_fetch")
        reason = (
            assessment.get("reasoning_summary")
            or assessment.get("human_review_reason")
            or "LLM source critic blocked content fetch."
        )
        updated.update(
            {
                "final_screening_decision": fetch_policy["final_decision"],
                "final_screening_reason": _append_note(
                    updated.get("final_screening_reason"), str(reason)
                ),
                "ready_for_content_fetch": False,
                "requires_human_review": bool(fetch_policy["review_required"]),
                "status": fetch_policy["status"],
                "source_role_final": fetch_policy["final_role"],
                "blocked_from_fetch": True,
                "blocked_from_fetch_reason": (
                    "llm_source_critic_block_fetch: " + str(reason)
                ),
            }
        )
    elif needs_human_review and review_blocks_fetch:
        reason = (
            assessment.get("human_review_reason")
            or assessment.get("reasoning_summary")
            or "LLM source critic recommended human review."
        )
        updated.update(
            {
                "final_screening_decision": "needs_human_review",
                "final_screening_reason": _append_note(
                    updated.get("final_screening_reason"), str(reason)
                ),
                "ready_for_content_fetch": False,
                "requires_human_review": True,
                "status": "needs_human_review",
                "source_role_final": "needs_human_review",
                "blocked_from_fetch": True,
                "blocked_from_fetch_reason": (
                    "llm_source_critic_block_fetch: " + str(reason)
                ),
            }
        )

    updated.update(
        {
            "llm_source_critic_enabled": True,
            "llm_source_critic_attempted": True,
            "llm_source_critic_assessed": True,
            "llm_source_critic_status": "assessed",
            "llm_source_critic_failed": False,
            "llm_source_critic_error": None,
            "llm_source_critic_error_type": None,
            "source_routing_method": "llm_source_critic_enriched",
            "llm_proposed_source_role": assessment.get("proposed_source_role"),
            "llm_proposed_screening_decision": assessment.get(
                "proposed_screening_decision"
            ),
            "llm_credibility_level": assessment.get("credibility_level"),
            "llm_credibility_reason": assessment.get("credibility_reason"),
            "llm_expected_extractable_fields": list(
                assessment.get("expected_extractable_fields") or []
            ),
            "llm_semantic_leakage_risk": semantic_leakage_risk,
            "llm_semantic_leakage_reason": assessment.get(
                "semantic_leakage_reason"
            ),
            "llm_context_only_risk": context_only_risk,
            "llm_validation_candidate_risk": validation_candidate_risk,
            "llm_needs_human_review": needs_human_review,
            "llm_human_review_reason": assessment.get("human_review_reason"),
            "llm_source_critic_decision": assessment.get("critic_decision"),
            "llm_source_critic_confidence": assessment.get("confidence"),
            "llm_source_critic_reason": assessment.get("reasoning_summary"),
            "llm_source_critic_risk_flags": risk_flags,
            "llm_source_critic_recommended_role": assessment.get(
                "recommended_role"
            ),
            "llm_source_critic_fetch_recommendation": assessment.get(
                "fetch_recommendation"
            ),
            "llm_source_critic_review_required": bool(
                fetch_policy["review_required"]
            ),
            "llm_source_critic_block_fetch": bool(
                fetch_policy["should_block_fetch"]
            ),
            "llm_source_critic_warnings": _as_str_list(
                assessment.get("warnings")
            ),
            "llm_reasoning_summary": assessment.get("reasoning_summary"),
            "routing_flags": routing_flags,
            "critic_flags": critic_flags,
            "blocked_from_fetch": bool(
                updated.get("blocked_from_fetch", False)
            ),
        }
    )
    return updated


def _is_timeout_exception(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "timeout" in text or "timed out" in text


def _is_known_official_public_health_source(entry: dict) -> bool:
    source_type = str(
        entry.get("source_type_final") or entry.get("source_type") or ""
    ).strip().lower()
    publisher = str(
        entry.get("actual_publisher") or entry.get("publisher") or ""
    ).strip().lower()
    domain = str(
        entry.get("domain") or entry.get("canonical_url") or entry.get("url") or ""
    ).strip().lower()
    official_types = {
        "official_public_health_agency",
        "national_public_health_agency",
        "state_or_local_public_health_agency",
        "international_public_health_agency",
        "international_organization_report",
        "structured_database",
    }
    official_publishers = {
        "centers for disease control and prevention",
        "world health organization",
        "pan american health organization",
        "european centre for disease prevention and control",
        "new mexico department of health",
        "california department of public health",
        "virginia department of health",
    }
    official_domains = {
        "cdc.gov",
        "who.int",
        "paho.org",
        "ecdc.europa.eu",
        "nmhealth.org",
        "cdph.ca.gov",
        "vdh.virginia.gov",
        "health.ny.gov",
        "floridahealth.gov",
    }
    return (
        source_type in official_types
        or publisher in official_publishers
        or any(domain_name in domain for domain_name in official_domains)
    )


def _apply_llm_source_critic_failure(entry: dict, exc: Exception) -> dict:
    updated = dict(entry)
    flags = list(updated.get("routing_flags") or [])
    _append_flag(flags, "llm_source_critic_failed")
    status = "timeout" if _is_timeout_exception(exc) else "failed"
    if status == "timeout":
        _append_flag(flags, "llm_source_critic_timeout")
    if not _is_known_official_public_health_source(updated):
        updated.update(
            {
                "requires_human_review": True,
                "status": "needs_human_review",
                "final_screening_decision": "needs_human_review",
                "source_role_final": "needs_human_review",
                "final_screening_reason": _append_note(
                    updated.get("final_screening_reason"),
                    "LLM source critic failed; deterministic source identity "
                    "could not establish enough confidence for automatic collection routing.",
                ),
            }
        )
    updated.update(
        {
            "llm_source_critic_enabled": True,
            "llm_source_critic_attempted": True,
            "llm_source_critic_assessed": False,
            "llm_source_critic_status": status,
            "llm_source_critic_failed": True,
            "llm_source_critic_error": f"{type(exc).__name__}: {exc}",
            "llm_source_critic_error_type": type(exc).__name__,
            "source_routing_method": "deterministic_fallback",
            "blocked_from_fetch": False,
            "blocked_from_fetch_reason": None,
            "routing_flags": flags,
        }
    )
    return updated


def _mark_llm_source_critic_skipped(entry: dict, reason: str) -> dict:
    warnings = list(entry.get("llm_source_critic_warnings") or [])
    if reason not in warnings:
        warnings.append(reason)
    updated = dict(entry)
    updated.update(
        {
            "llm_source_critic_enabled": True,
            "llm_source_critic_attempted": False,
            "llm_source_critic_assessed": False,
            "llm_source_critic_status": "skipped",
            "llm_source_critic_failed": False,
            "llm_source_critic_error": None,
            "llm_source_critic_error_type": None,
            "llm_source_critic_warnings": warnings,
            "source_routing_method": entry.get("source_routing_method")
            or "deterministic",
        }
    )
    return updated


def _mark_llm_source_critic_fail_open_skipped(entry: dict, reason: str) -> dict:
    warnings = list(entry.get("llm_source_critic_warnings") or [])
    if reason not in warnings:
        warnings.append(reason)
    flags = list(entry.get("routing_flags") or [])
    _append_flag(flags, "llm_source_critic_fail_open")
    updated = dict(entry)
    updated.update(
        {
            "llm_source_critic_enabled": True,
            "llm_source_critic_attempted": False,
            "llm_source_critic_assessed": False,
            "llm_source_critic_status": "source_critic_skipped_timeout_fail_open",
            "llm_source_critic_failed": False,
            "llm_source_critic_error": None,
            "llm_source_critic_error_type": None,
            "llm_source_critic_warnings": warnings,
            "source_routing_method": "deterministic_fallback",
            "blocked_from_fetch": False,
            "blocked_from_fetch_reason": None,
            "routing_flags": flags,
        }
    )
    return updated


def _mark_llm_source_critic_disabled(entry: dict) -> dict:
    updated = dict(entry)
    updated.update(
        {
            "llm_source_critic_enabled": False,
            "llm_source_critic_attempted": False,
            "llm_source_critic_assessed": False,
            "llm_source_critic_status": "disabled",
            "llm_source_critic_failed": False,
            "llm_source_critic_error": None,
            "llm_source_critic_error_type": None,
            "source_routing_method": entry.get("source_routing_method")
            or "deterministic",
        }
    )
    return updated


def _preserve_historical_metadata_only_routing(entry: dict) -> dict:
    """Preserve index leads without claiming verification of archived content."""
    if (entry.get("historical_snapshot") or {}).get("verification_status") != "index_only":
        return entry
    return {
        **entry,
        "blocked_from_fetch": True,
        "blocked_from_fetch_reason": "historical_index_only",
        "ready_for_content_fetch": False,
        "target_verification_status": "unverified_candidate",
        "target_verification_reason": (
            "Archive index metadata only; snapshot content and reporting period "
            "have not been verified."
        ),
        "target_fit_status": "task_record_collection_candidate",
        "triage_role": "task_record_collection_candidate",
        "disease_fit": "unknown",
        "geography_fit": "unknown",
        "date_fit": "unknown",
        "task_fit_evidence_origin": "historical_index_metadata",
        "task_fit_content_hash": None,
        "published_date": None,
        "reporting_period_start": None,
        "reporting_period_end": None,
        "reporting_period_label": None,
        "period_basis": None,
        "must_fetch": False,
        "must_fetch_reason": None,
        "coverage_requirement_ids": [],
    }


def _reapply_source_critic_fetch_block(entry: dict) -> dict:
    if not entry.get("llm_source_critic_block_fetch"):
        return entry
    updated = dict(entry)
    role = _source_critic_role(
        updated.get("source_role_final")
        or updated.get("llm_source_critic_recommended_role"),
        default="needs_human_review",
    )
    decision = str(updated.get("llm_source_critic_decision") or "").lower()
    fetch_recommendation = str(
        updated.get("llm_source_critic_fetch_recommendation") or ""
    ).lower()
    hard_exclude = (
        role == "excluded"
        or decision in {"not_task_relevant", "exclude_from_task"}
        or fetch_recommendation == "block_fetch"
    )
    if hard_exclude:
        updated["source_role_final"] = "excluded"
        updated["final_screening_decision"] = "exclude"
        updated["status"] = "excluded"
    elif role in {"context", "collection_support"}:
        updated["source_role_final"] = role
        updated["final_screening_decision"] = "include_for_context_fetch"
        updated["status"] = "ready_for_context_fetch"
    else:
        updated["source_role_final"] = "needs_human_review"
        updated["final_screening_decision"] = "needs_human_review"
        updated["status"] = "needs_human_review"
    updated["ready_for_content_fetch"] = False
    updated["blocked_from_fetch"] = True
    updated["blocked_from_fetch_reason"] = updated.get(
        "blocked_from_fetch_reason"
    ) or (
        "llm_source_critic_block_fetch: "
        + str(updated.get("llm_source_critic_reason") or "blocked by critic")
    )
    if updated.get("llm_source_critic_review_required"):
        updated["requires_human_review"] = True
    return updated


# ---------------------------------------------------------------------------
# Final routing
# ---------------------------------------------------------------------------


_FINAL_STATUS_MAP: dict[str, str] = {
    "include_for_content_fetch": "ready_for_content_fetch",
    "include_for_context_fetch": "ready_for_context_fetch",
    "defer_to_search_expansion": "deferred_search_expansion",
    "needs_human_review": "needs_human_review",
    "exclude": "excluded",
    "reserved_for_validation": "reserved_for_validation",
}

_VALIDATION_RESERVED_REASON = (
    "Held out for masked validation ground-truth comparison."
)
_CONTEXT_ONLY_REASON = (
    "Configured as context-only by source-role policy; fetch for context "
    "grounding only, not structured record extraction."
)

_SOURCE_TRUST_REVIEW_TYPES = {
    "blog",
    "community_forum",
    "forum",
    "media_report",
    "news",
    "news_and_situation_report",
    "news_media",
    "secondary_media",
    "social_media",
    "unknown",
}

_SOURCE_TRUST_REVIEW_PUBLISHERS = {
    "facebook",
    "instagram",
    "reddit",
    "tiktok",
    "twitter",
    "x",
    "youtube",
}

_SOURCE_TRUST_REVIEW_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "reddit.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "youtu.be",
}


def _domain_for_entry(entry: dict) -> str:
    url = entry.get("canonical_url") or entry.get("url") or ""
    if not url:
        return ""
    netloc = urlsplit(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _entry_requires_source_trust_review(entry: dict) -> bool:
    source_type = str(
        entry.get("source_type_final")
        or entry.get("source_type")
        or entry.get("planned_query_source_type")
        or ""
    ).strip().lower()
    publisher = str(
        entry.get("actual_publisher") or entry.get("publisher") or ""
    ).strip().lower()
    domain = _domain_for_entry(entry)
    if source_type in _SOURCE_TRUST_REVIEW_TYPES:
        return True
    if any(value and value in publisher for value in _SOURCE_TRUST_REVIEW_PUBLISHERS):
        return True
    return any(domain == value or domain.endswith("." + value) for value in _SOURCE_TRUST_REVIEW_DOMAINS)


def _apply_source_trust_review_routing(entry: dict, *, reason: str) -> dict:
    updated = dict(entry)
    flags = list(updated.get("screening_flags") or [])
    if "source_trust_requires_human_review" not in flags:
        flags.append("source_trust_requires_human_review")
    routing_flags = list(updated.get("routing_flags") or [])
    if "source_trust_requires_human_review" not in routing_flags:
        routing_flags.append("source_trust_requires_human_review")
    updated.update(
        {
            "target_fit_status": "needs_human_review",
            "target_verification_status": "source_trust_requires_human_review",
            "target_verification_reason": reason,
            "triage_role": "needs_human_review",
            "source_role_final": "needs_human_review",
            "source_role": "context_source",
            "source_role_fit": "needs_human_review",
            "screening_decision": "needs_human_review",
            "final_screening_decision": "needs_human_review",
            "screening_reason": reason,
            "final_screening_reason": reason,
            "status": "needs_human_review",
            "ready_for_content_fetch": False,
            "requires_human_review": True,
            "human_review_recommended": True,
            "human_review_reason": reason,
            "screening_flags": flags,
            "routing_flags": routing_flags,
        }
    )
    return updated


def _domain_matches_reserved(domain: str, reserved_domains: list[str]) -> bool:
    if not domain:
        return False
    for reserved in reserved_domains:
        normalized = str(reserved or "").strip().lower()
        if not normalized:
            continue
        if domain == normalized or domain.endswith("." + normalized):
            return True
    return False


def _is_validation_reserved_source(
    entry: dict,
    role_policy: dict,
    collection_mode: str,
) -> bool:
    if collection_mode != "masked_validation":
        return False
    reserved_ids = set(role_policy.get("validation_reserved_source_ids") or [])
    source_id = entry.get("source_id")
    if source_id in reserved_ids:
        return True
    if not role_policy.get("domain_masking_enabled", False):
        return False
    reserved_domains = list(role_policy.get("validation_reserved_domains") or [])
    return _domain_matches_reserved(_domain_for_entry(entry), reserved_domains)


def _is_context_only_source(entry: dict, role_policy: dict) -> bool:
    context_only_ids = set(role_policy.get("context_only_source_ids") or [])
    return entry.get("source_id") in context_only_ids


def _append_note(existing: str | None, note: str) -> str:
    if not existing:
        return note
    if note in existing:
        return existing
    return f"{existing} {note}"


def _entry_text_for_credibility(entry: dict) -> str:
    parts = [
        str(entry.get("publisher") or ""),
        str(entry.get("source_type") or ""),
        str(entry.get("title") or ""),
        str(entry.get("canonical_url") or entry.get("url") or ""),
    ]
    return " ".join(parts).lower()


def _apply_validation_reserved_override(
    entry: dict,
    role_policy: dict,
) -> dict:
    behavior = role_policy.get("collection_blocking_behavior") or {}
    flags = list(entry.get("routing_flags") or [])
    for flag in behavior.get("routing_flags") or [
        "validation_reserved",
        "blocked_from_collection",
    ]:
        if flag not in flags:
            flags.append(flag)

    final_decision = behavior.get(
        "final_screening_decision", "reserved_for_validation"
    )
    status = behavior.get("status", "reserved_for_validation")
    updated = dict(entry)
    updated.update(
        {
            "source_role": "validation_reserved",
            "final_screening_decision": final_decision,
            "final_screening_confidence": entry.get("final_screening_confidence"),
            "final_screening_reason": _VALIDATION_RESERVED_REASON,
            "ready_for_content_fetch": bool(
                behavior.get("set_ready_for_content_fetch", False)
            ),
            "requires_human_review": False,
            "routing_flags": flags,
            "status": status,
            "notes": _append_note(entry.get("notes"), _VALIDATION_RESERVED_REASON),
        }
    )
    return updated


def _apply_context_only_override(entry: dict) -> dict:
    flags = list(entry.get("routing_flags") or [])
    for flag in ("context_only", "blocked_from_structured_extraction"):
        if flag not in flags:
            flags.append(flag)

    updated = dict(entry)
    updated.update(
        {
            "source_role": "context_source",
            "final_screening_decision": "include_for_context_fetch",
            "final_screening_reason": _CONTEXT_ONLY_REASON,
            "ready_for_content_fetch": True,
            "requires_human_review": False,
            "routing_flags": flags,
            "status": "ready_for_context_fetch",
            "notes": _append_note(entry.get("notes"), _CONTEXT_ONLY_REASON),
        }
    )
    return updated


def _final_route_entry(
    entry: dict,
    critic: SourceCriticResult,
    policy: SourceScreeningPolicy,
) -> SourceFinalRoutingDecision:
    role = entry.get("source_role") or "irrelevant_source"
    screening_decision = entry.get("screening_decision")
    screening_confidence = float(entry.get("screening_confidence") or 0.0)
    screening_reason = entry.get("screening_reason") or ""

    avg_confidence = (screening_confidence + critic.critic_confidence) / 2.0
    routing_flags: list[str] = []

    # 1. Role-based tentative decision.
    if role == "data_source" and screening_decision == "include" and critic.critic_decision == "include":
        final_decision = "include_for_content_fetch"
        ready_for_content_fetch = True
        requires_human_review = False
        final_reason = (
            "Source classified as data_source; both screening and critic agree "
            "to include for content fetch."
        )
    elif role == "context_source" and screening_decision == "include" and critic.critic_decision == "include":
        final_decision = "include_for_context_fetch"
        ready_for_content_fetch = True
        requires_human_review = False
        final_reason = (
            "Source classified as context_source; both screening and critic "
            "agree to include for context fetch."
        )
    elif role == "search_endpoint":
        final_decision = "defer_to_search_expansion"
        ready_for_content_fetch = False
        requires_human_review = False
        final_reason = (
            "Search endpoint deferred: must be expanded into article-level "
            "candidates by a later real-search step."
        )
    elif role == "placeholder_source":
        final_decision = "defer_to_search_expansion"
        ready_for_content_fetch = False
        requires_human_review = False
        final_reason = (
            "Internal seed placeholder deferred: awaiting real connector "
            "implementation."
        )
    else:
        # Fallback for irrelevant_source or unrecognized combinations.
        final_decision = "exclude"
        ready_for_content_fetch = False
        requires_human_review = False
        final_reason = screening_reason or "Source excluded by default fallback."

    # 2. Disagreement override.
    if (
        screening_decision is not None
        and critic.critic_decision is not None
        and screening_decision != critic.critic_decision
    ):
        final_decision = "needs_human_review"
        ready_for_content_fetch = False
        requires_human_review = True
        routing_flags.append("screening_and_critic_disagree")
        final_reason = (
            "Screening and critic disagree on this source; routing to human "
            "review for resolution."
        )

    # 3. Low-confidence override.
    low_threshold = float(policy.thresholds.get("low_confidence", 0.50))
    if avg_confidence < low_threshold:
        final_decision = "needs_human_review"
        ready_for_content_fetch = False
        requires_human_review = True
        if "low_screening_confidence" not in routing_flags:
            routing_flags.append("low_screening_confidence")
        final_reason = (
            f"Combined screening+critic confidence ({avg_confidence:.2f}) is "
            f"below the low_confidence threshold ({low_threshold:.2f})."
        )

    # 4. Irrelevant-source final override (after disagree/low-conf checks).
    if role == "irrelevant_source":
        final_decision = "exclude"
        ready_for_content_fetch = False
        requires_human_review = False
        final_reason = (
            "Source role is irrelevant for the task collection scope; "
            "excluded from downstream content fetching."
        )

    return SourceFinalRoutingDecision(
        source_id=entry.get("source_id", ""),
        final_decision=final_decision,
        final_confidence=round(avg_confidence, 4),
        final_reason=final_reason,
        ready_for_content_fetch=ready_for_content_fetch,
        requires_human_review=requires_human_review,
        routing_flags=routing_flags,
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def source_screening(state: DataCollectionState) -> dict:
    """Deterministic source screening agent (rule-based, no LLM)."""

    policy = SourceScreeningPolicy(**load_source_screening_policy())
    registry = list(state.get("source_registry") or [])
    screening_criteria = dict(state.get("screening_criteria") or {})
    screening_criteria.setdefault("structured_task", state.get("structured_task") or {})
    screening_criteria.setdefault("collection_spec", state.get("collection_spec") or {})
    task_for_triage = _task_for_screening(screening_criteria)
    collection_mode = _collection_mode_from_state(state)
    low_threshold = float(policy.thresholds.get("low_confidence", 0.50))

    updated: list[dict] = []
    decision_counter: Counter = Counter()
    role_counter: Counter = Counter()
    triage_role_counter: Counter = Counter()
    target_verification_counter: Counter = Counter()
    low_confidence_count = 0
    missing_url_count = 0
    missing_source_type_count = 0
    triage_results: list[dict] = []

    for entry in registry:
        result = _screen_entry(entry, policy, screening_criteria)
        new_entry = dict(entry)
        new_entry.update(
            {
                "screening_decision": result.screening_decision,
                "screening_confidence": result.screening_confidence,
                "screening_reason": result.screening_reason,
                "source_role": result.source_role,
                "screening_flags": list(result.screening_flags),
                "expected_extractable_fields": list(result.expected_extractable_fields),
                "status": "screened",
            }
        )
        new_entry, target_triage = _apply_direct_triage_verification(
            new_entry,
            result,
            state,
            collection_mode=collection_mode,
        )
        # Validate via the model so any future field changes fail loudly here.
        new_entry = _preserve_historical_metadata_only_routing(new_entry)
        validated = SourceRegistryEntry(**new_entry).model_dump()
        updated.append(validated)

        validated_decision = validated.get("screening_decision") or result.screening_decision
        validated_role = validated.get("source_role") or result.source_role
        decision_counter[validated_decision] += 1
        role_counter[validated_role] += 1
        if float(validated.get("screening_confidence") or result.screening_confidence) < low_threshold:
            low_confidence_count += 1
        validated_flags = list(validated.get("screening_flags") or result.screening_flags)
        if "missing_url" in validated_flags:
            missing_url_count += 1
        if "missing_source_type" in validated_flags:
            missing_source_type_count += 1
        triage_results.append(
            {
                "source_id": validated.get("source_id"),
                "source_url": validated.get("canonical_url")
                or validated.get("url"),
                "canonical_url": validated.get("canonical_url")
                or validated.get("url"),
                "source_title": validated.get("title"),
                "title": validated.get("title"),
                "query_id": validated.get("query_id"),
                "query_used": validated.get("query_used"),
                "task_disease": task_for_triage.get("disease"),
                "task_location": task_for_triage.get("location"),
                "task_start_date": task_for_triage.get("start_date"),
                "task_end_date": task_for_triage.get("end_date"),
                "triage_agent": "source_triage_agent",
                "triage_method": (
                    "deterministic_safety_checks_with_llm_triage_contract"
                ),
                "triage_decision": validated_decision,
                "triage_role": target_triage.get("triage_role"),
                "source_role": validated_role,
                "source_role_final": validated.get("source_role_final"),
                "source_role_compat": validated.get("source_role"),
                "source_type": validated.get("source_type"),
                "source_type_final": validated.get("source_type_final"),
                "publisher": validated.get("publisher"),
                "actual_publisher": validated.get("actual_publisher"),
                "domain": validated.get("domain"),
                "target_fit_status": validated.get("target_fit_status"),
                "target_verification_status": target_triage.get(
                    "target_verification_status"
                ),
                "target_verification_reason": target_triage.get(
                    "target_verification_reason"
                ),
                "disease_fit": validated.get("disease_fit")
                or ("candidate" if validated_decision != "exclude" else "unknown"),
                "geography_fit": validated.get("geography_fit")
                or ("candidate" if validated_decision != "exclude" else "unknown"),
                "date_fit": validated.get("date_fit")
                or target_triage.get("date_fit")
                or ("candidate" if validated_decision != "exclude" else "unknown"),
                "source_role_fit": validated.get("source_role_fit") or validated_role,
                "requires_human_review": bool(validated.get("requires_human_review")),
                "human_review_recommended": bool(
                    validated.get("human_review_recommended")
                ),
                "human_review_reason": validated.get("human_review_reason"),
                "routing_flags": list(validated.get("routing_flags") or []),
                "confidence": validated.get("screening_confidence"),
                "reason": validated.get("screening_reason"),
                "flags": validated_flags,
                "expected_extractable_fields": list(
                    result.expected_extractable_fields
                ),
                "llm_semantic_triage_required": True,
            }
        )
        triage_role_counter[target_triage.get("triage_role") or "unknown"] += 1
        target_verification_counter[
            target_triage.get("target_verification_status") or "unknown"
        ] += 1

    summary = {
        "input_registry_count": len(registry),
        "screened_count": len(updated),
        "screening_decision_counts": dict(decision_counter),
        "source_role_counts": dict(role_counter),
        "low_confidence_count": low_confidence_count,
        "missing_url_count": missing_url_count,
        "missing_source_type_count": missing_source_type_count,
        "triage_role_counts": dict(triage_role_counter),
        "target_verification_status_counts": dict(target_verification_counter),
    }

    trace = append_trace(
        state,
        node_name="source_screening",
        message=f"Screened {len(updated)} sources using deterministic policy.",
        metadata=summary,
    )
    return {
        "source_registry": updated,
        "source_screening_summary": summary,
        "source_triage_results": triage_results,
        "collection_trace": trace,
    }


def source_critic_and_uncertainty_routing(state: DataCollectionState) -> dict:
    """Deterministic source critic + final routing (rule-based, no LLM)."""

    policy = SourceScreeningPolicy(**load_source_screening_policy())
    role_policy = load_source_role_policy()
    collection_mode = _collection_mode_from_state(state, role_policy)
    registry = list(state.get("source_registry") or [])
    existing_queue = list(state.get("human_review_queue") or [])
    existing_review_ids = {item.get("review_id") for item in existing_queue}
    llm_source_identity_enabled = llm_clients.llm_source_identity_enabled()
    llm_source_identity_max_sources = _parse_positive_int_env(
        "LLM_SOURCE_IDENTITY_MAX_SOURCES"
    )
    llm_source_identity_require_llm = _env_flag(
        "LLM_SOURCE_IDENTITY_REQUIRE_LLM"
    )
    llm_source_identity_allow_fallback = _env_flag(
        "LLM_SOURCE_IDENTITY_ALLOW_DETERMINISTIC_FALLBACK",
        default=True,
    )
    (
        registry,
        source_coverage_requirements,
        source_coverage_audit,
    ) = annotate_source_coverage(registry, state)
    registry, source_identity_assessments, source_identity_summary = (
        apply_source_identity_to_registry(
            registry,
            collection_spec=state.get("collection_spec"),
            llm_enabled=llm_source_identity_enabled,
            max_sources=llm_source_identity_max_sources,
            require_llm=llm_source_identity_require_llm,
            allow_deterministic_fallback=llm_source_identity_allow_fallback,
        )
    )
    # Re-apply coverage after identity so advisory identity metadata can never
    # demote a task-critical verified collection source.
    (
        registry,
        source_coverage_requirements,
        source_coverage_audit,
    ) = annotate_source_coverage(registry, state)
    if (
        llm_source_identity_enabled
        and llm_source_identity_require_llm
        and not llm_source_identity_allow_fallback
        and source_identity_summary.get("blocked_llm_required_count", 0) > 0
    ):
        blocked_source_ids = [
            item.get("source_id")
            for item in source_identity_assessments
            if item.get("source_identity_status") == "blocked_llm_required"
        ]
        blocked_preview = ", ".join(str(sid) for sid in blocked_source_ids if sid)
        raise RuntimeError(
            "source identity LLM required but unavailable for "
            f"{source_identity_summary.get('blocked_llm_required_count')} source(s)"
            + (f": {blocked_preview}" if blocked_preview else "")
        )
    llm_source_critic_enabled = llm_clients.llm_source_critic_enabled()
    llm_source_critic_allowlist = _parse_csv_env(
        "LLM_SOURCE_CRITIC_SOURCE_ID_ALLOWLIST"
    )
    llm_source_critic_max_sources = _parse_positive_int_env(
        "LLM_SOURCE_CRITIC_MAX_SOURCES"
    )
    llm_source_critic_max_failures = _parse_positive_int_env(
        "LLM_SOURCE_CRITIC_MAX_FAILURES"
    )
    llm_source_critic_max_node_seconds = _parse_positive_float_env(
        "LLM_SOURCE_CRITIC_MAX_NODE_SECONDS"
    )
    llm_source_critic_started_at = monotonic()
    llm_review_blocks_fetch = (
        _llm_source_critic_review_blocks_fetch()
        and collection_mode != "direct_collection"
    )
    credibility_runtime = source_credibility_runtime_from_env()

    updated: list[dict] = []
    credibility_assessments: list[dict] = []
    critic_decision_counter: Counter = Counter()
    final_decision_counter: Counter = Counter()
    critic_flag_counter: Counter = Counter()
    agreement_count = 0
    disagreement_count = 0
    ready_for_content_fetch_count = 0
    requires_human_review_count = 0
    deferred_search_expansion_count = 0
    excluded_count = 0
    validation_reserved_source_ids: list[str] = []
    context_only_source_ids: list[str] = []
    llm_attempted_source_count = 0
    llm_assessed_source_count = 0
    llm_skipped_source_count = 0
    llm_skipped_source_ids: list[str] = []
    llm_source_critic_failure_count = 0
    llm_semantic_leakage_count = 0
    llm_human_review_recommended_count = 0
    credibility_level_counter: Counter = Counter()
    low_credibility_source_count = 0
    llm_critic_decision_counter: Counter = Counter()
    llm_fetch_recommendation_counter: Counter = Counter()
    llm_risk_flag_counter: Counter = Counter()
    llm_skipped_reason_counter: Counter = Counter()
    llm_blocked_fetch_count = 0
    llm_review_required_count = 0
    llm_allowed_fetch_count = 0
    llm_context_only_count = 0
    llm_blocked_source_ids: list[str] = []
    llm_selected_source_ids: list[str] = []
    fail_open_skipped_source_count = 0
    fail_open_reason_counter: Counter = Counter()
    source_routing_method_counter: Counter = Counter()
    source_critic_results: list[dict] = []

    if llm_source_critic_enabled:
        (
            llm_selected_source_id_set,
            llm_candidate_skip_reasons,
            source_critic_selection_summary,
        ) = _select_llm_source_critic_candidates(
            registry,
            allowlist=llm_source_critic_allowlist,
            max_sources=llm_source_critic_max_sources,
            collection_mode=collection_mode,
        )
        llm_selected_source_ids = list(
            source_critic_selection_summary.get("selected_source_ids") or []
        )
    else:
        llm_selected_source_id_set = set()
        llm_candidate_skip_reasons = {}
        source_critic_selection_summary = {
            "selection_mode": "disabled",
            "explicit_allowlist_used": False,
            "max_sources": llm_source_critic_max_sources,
            "eligible_candidate_count": 0,
            "selected_candidate_count": 0,
            "skipped_candidate_count": 0,
            "selected_source_ids": [],
            "skipped_reason_counts": {},
        }

    new_review_items: list[HumanReviewItem] = []

    for entry in registry:
        critic = _critic_review_entry(entry, policy)
        routing = _final_route_entry(entry, critic, policy)

        new_entry = dict(entry)
        new_entry.update(
            {
                "critic_decision": critic.critic_decision,
                "critic_confidence": critic.critic_confidence,
                "critic_reason": critic.critic_reason,
                "critic_agrees_with_screening": critic.critic_agrees_with_screening,
                "critic_flags": list(critic.critic_flags),
                "final_screening_decision": routing.final_decision,
                "final_screening_confidence": routing.final_confidence,
                "final_screening_reason": routing.final_reason,
                "ready_for_content_fetch": routing.ready_for_content_fetch,
                "requires_human_review": routing.requires_human_review,
                "routing_flags": list(routing.routing_flags),
                "status": _FINAL_STATUS_MAP.get(routing.final_decision, "screened"),
            }
        )
        if llm_source_critic_enabled:
            source_id = new_entry.get("source_id") or ""
            if source_id not in llm_selected_source_id_set:
                skip_reason = llm_candidate_skip_reasons.get(
                    source_id, "not_selected_by_source_critic_policy"
                )
                llm_skipped_source_count += 1
                llm_skipped_reason_counter[skip_reason] += 1
                if source_id:
                    llm_skipped_source_ids.append(source_id)
                new_entry = _mark_llm_source_critic_skipped(new_entry, skip_reason)
            else:
                fail_open_reason = None
                if (
                    llm_source_critic_max_failures
                    and llm_source_critic_failure_count
                    >= llm_source_critic_max_failures
                ):
                    fail_open_reason = "source_critic_failure_cap_reached"
                elif (
                    llm_source_critic_max_node_seconds
                    and monotonic() - llm_source_critic_started_at
                    >= llm_source_critic_max_node_seconds
                ):
                    fail_open_reason = "source_critic_node_timeout_fail_open"

                if fail_open_reason:
                    llm_skipped_source_count += 1
                    fail_open_skipped_source_count += 1
                    llm_skipped_reason_counter[fail_open_reason] += 1
                    fail_open_reason_counter[fail_open_reason] += 1
                    if source_id:
                        llm_skipped_source_ids.append(source_id)
                    new_entry = _mark_llm_source_critic_fail_open_skipped(
                        new_entry,
                        fail_open_reason,
                    )
                else:
                    llm_attempted_source_count += 1
                    try:
                        assessment = assess_source_with_llm(
                            source_entry=new_entry,
                            collection_spec=state.get("collection_spec"),
                            screening_policy=policy.model_dump(),
                            source_role_policy=role_policy,
                        )
                        llm_assessed_source_count += 1
                        if assessment.get("semantic_leakage_risk"):
                            llm_semantic_leakage_count += 1
                        if assessment.get("needs_human_review"):
                            llm_human_review_recommended_count += 1
                        new_entry = _apply_llm_source_critic_assessment(
                            new_entry,
                            assessment,
                            review_blocks_fetch=llm_review_blocks_fetch,
                        )
                    except Exception as exc:  # noqa: BLE001 - advisory fallback
                        llm_source_critic_failure_count += 1
                        new_entry = _apply_llm_source_critic_failure(new_entry, exc)
        else:
            new_entry = _mark_llm_source_critic_disabled(new_entry)

        if _is_validation_reserved_source(new_entry, role_policy, collection_mode):
            new_entry = _apply_validation_reserved_override(new_entry, role_policy)
            validation_reserved_source_ids.append(new_entry.get("source_id", ""))
        elif _is_context_only_source(new_entry, role_policy):
            new_entry = _apply_context_only_override(new_entry)
            context_only_source_ids.append(new_entry.get("source_id", ""))

        new_entry, credibility_assessment = apply_source_credibility_assessment(
            new_entry,
            state,
            credibility_runtime,
        )
        new_entry = _reapply_source_critic_fetch_block(new_entry)
        new_entry = apply_source_identity_routing_guardrails(new_entry)
        new_entry = annotate_source_coverage([new_entry], state)[0][0]
        credibility_assessments.append(credibility_assessment)
        new_entry = _preserve_historical_metadata_only_routing(new_entry)
        validated = SourceRegistryEntry(**new_entry).model_dump()
        updated.append(validated)

        critic_decision_counter[critic.critic_decision] += 1
        for flag in critic.critic_flags:
            critic_flag_counter[flag] += 1
        if critic.critic_agrees_with_screening:
            agreement_count += 1
        else:
            disagreement_count += 1

        final_decision = validated.get("final_screening_decision") or routing.final_decision
        final_decision_counter[final_decision] += 1
        source_routing_method_counter[
            validated.get("source_routing_method") or "deterministic"
        ] += 1
        if validated.get("ready_for_content_fetch"):
            ready_for_content_fetch_count += 1
        if validated.get("requires_human_review"):
            requires_human_review_count += 1
        if final_decision == "defer_to_search_expansion":
            deferred_search_expansion_count += 1
        if final_decision == "exclude":
            excluded_count += 1
        credibility_level = validated.get("credibility_level") or "unknown"
        credibility_level_counter[credibility_level] += 1
        if credibility_level == "low":
            low_credibility_source_count += 1

        llm_status = validated.get("llm_source_critic_status")
        if llm_status == "assessed":
            llm_decision = (
                validated.get("llm_source_critic_decision") or "unknown"
            )
            llm_fetch_recommendation = (
                validated.get("llm_source_critic_fetch_recommendation")
                or "unknown"
            )
            llm_critic_decision_counter[llm_decision] += 1
            llm_fetch_recommendation_counter[llm_fetch_recommendation] += 1
            for risk_flag in validated.get("llm_source_critic_risk_flags") or []:
                llm_risk_flag_counter[risk_flag] += 1
            if validated.get("llm_source_critic_block_fetch"):
                llm_blocked_fetch_count += 1
                llm_blocked_source_ids.append(validated.get("source_id", ""))
            elif llm_fetch_recommendation == "allow_fetch":
                llm_allowed_fetch_count += 1
            if validated.get("llm_source_critic_review_required"):
                llm_review_required_count += 1
            if (
                llm_fetch_recommendation == "context_fetch_only"
                or validated.get("llm_source_critic_decision")
                in _SOURCE_CRITIC_CONTEXT_DECISIONS
            ):
                llm_context_only_count += 1

        source_critic_results.append(
            {
                "source_id": validated.get("source_id"),
                "source_url": validated.get("canonical_url"),
                "llm_source_critic_enabled": validated.get(
                    "llm_source_critic_enabled"
                ),
                "llm_source_critic_attempted": validated.get(
                    "llm_source_critic_attempted"
                ),
                "llm_source_critic_assessed": validated.get(
                    "llm_source_critic_assessed"
                ),
                "llm_source_critic_status": llm_status,
                "source_routing_method": validated.get("source_routing_method"),
                "llm_source_critic_decision": validated.get(
                    "llm_source_critic_decision"
                ),
                "llm_source_critic_confidence": validated.get(
                    "llm_source_critic_confidence"
                ),
                "llm_source_critic_reason": validated.get(
                    "llm_source_critic_reason"
                ),
                "llm_source_critic_risk_flags": validated.get(
                    "llm_source_critic_risk_flags"
                )
                or [],
                "llm_source_critic_recommended_role": validated.get(
                    "llm_source_critic_recommended_role"
                ),
                "llm_source_critic_fetch_recommendation": validated.get(
                    "llm_source_critic_fetch_recommendation"
                ),
                "llm_source_critic_review_required": validated.get(
                    "llm_source_critic_review_required"
                ),
                "llm_source_critic_block_fetch": validated.get(
                    "llm_source_critic_block_fetch"
                ),
                "blocked_from_fetch": validated.get("blocked_from_fetch"),
                "blocked_from_fetch_reason": validated.get(
                    "blocked_from_fetch_reason"
                ),
                "reason": validated.get("llm_source_critic_reason")
                or validated.get("blocked_from_fetch_reason"),
            }
        )

        if validated.get("human_review_recommended"):
            review_id = f"review_source_credibility_{validated.get('source_id', '')}"
            if review_id and review_id not in existing_review_ids:
                new_review_items.append(
                    HumanReviewItem(
                        review_id=review_id,
                        item_type="source_credibility",
                        related_ids=[validated.get("source_id", "")],
                        reason=validated.get("human_review_reason")
                        or validated.get("final_score_explanation")
                        or "Source credibility assessment recommended human review.",
                        status="pending",
                    )
                )
                existing_review_ids.add(review_id)

        if validated.get("llm_source_critic_block_fetch"):
            review_id = f"review_source_critic_{validated.get('source_id', '')}"
            if review_id and review_id not in existing_review_ids:
                new_review_items.append(
                    HumanReviewItem(
                        review_id=review_id,
                        item_type="source_critic_blocked_source",
                        related_ids=[validated.get("source_id", "")],
                        reason=validated.get("blocked_from_fetch_reason")
                        or validated.get("llm_source_critic_reason")
                        or "LLM source critic blocked content fetch.",
                        status="pending",
                        priority=1,
                        review_packet={
                            "source_id": validated.get("source_id"),
                            "source_url": validated.get("canonical_url"),
                            "title": validated.get("title"),
                            "llm_source_critic_decision": validated.get(
                                "llm_source_critic_decision"
                            ),
                            "llm_source_critic_risk_flags": validated.get(
                                "llm_source_critic_risk_flags"
                            )
                            or [],
                            "llm_source_critic_fetch_recommendation": validated.get(
                                "llm_source_critic_fetch_recommendation"
                            ),
                            "blocked_from_fetch_reason": validated.get(
                                "blocked_from_fetch_reason"
                            ),
                        },
                        source_ids=[validated.get("source_id", "")],
                        source_urls=[validated.get("canonical_url", "")],
                        severity="high"
                        if validated.get("source_role_final") == "excluded"
                        else "medium",
                    )
                )
                existing_review_ids.add(review_id)

        if validated.get("requires_human_review"):
            review_id = f"review_source_{validated.get('source_id', '')}"
            if review_id and review_id not in existing_review_ids:
                new_review_items.append(
                    HumanReviewItem(
                        review_id=review_id,
                        item_type="source_screening",
                        related_ids=[validated.get("source_id", "")],
                        reason=validated.get("final_screening_reason")
                        or routing.final_reason,
                        status="pending",
                    )
                )
                existing_review_ids.add(review_id)

    (
        updated,
        source_coverage_requirements,
        source_coverage_audit,
    ) = annotate_source_coverage(updated, state)
    updated = [_preserve_historical_metadata_only_routing(row) for row in updated]
    must_fetch_sources = [
        {
            "source_id": row.get("source_id"),
            "canonical_url": row.get("canonical_url") or row.get("url"),
            "must_fetch_reason": row.get("must_fetch_reason"),
            "coverage_requirement_ids": row.get("coverage_requirement_ids") or [],
            "routing_conflict_warnings": row.get("routing_conflict_warnings") or [],
        }
        for row in updated
        if row.get("must_fetch")
    ]
    direct_fast_path_summary = {
        "collection_mode": collection_mode,
        "direct_collection_enabled": collection_mode == "direct_collection",
        "target_source_count": len(must_fetch_sources),
        "target_source_ids": [
            str(row.get("source_id") or "")
            for row in must_fetch_sources
            if row.get("source_id")
        ],
        "identity_assessed_source_count": source_identity_summary.get(
            "identity_assessed_count", 0
        ),
        "identity_llm_assessed_source_count": source_identity_summary.get(
            "llm_identity_assessed_count", 0
        ),
        "identity_skipped_source_count": source_identity_summary.get(
            "direct_identity_fast_path_skipped_count", 0
        ),
        "critic_attempted_source_count": llm_attempted_source_count,
        "critic_assessed_source_count": llm_assessed_source_count,
        "critic_skipped_source_count": llm_skipped_source_count,
        "critic_skipped_reason_counts": dict(llm_skipped_reason_counter),
        "critic_fail_open_skipped_source_count": fail_open_skipped_source_count,
        "critic_fail_open_reason_counts": dict(fail_open_reason_counter),
        "source_critic_selection_summary": source_critic_selection_summary,
    }

    human_review_queue = list(existing_queue) + [
        item.model_dump() for item in new_review_items
    ]
    credibility_summary = build_source_credibility_summary(
        credibility_assessments,
        credibility_runtime,
    )
    identity_skip_reason_counter = Counter(
        item.get("source_identity_llm_skipped_reason")
        for item in source_identity_assessments
        if item.get("source_identity_llm_skipped_reason")
    )
    credibility_skip_reason_counter = Counter(
        item.get("source_credibility_llm_skipped_reason")
        for item in credibility_assessments
        if item.get("source_credibility_llm_skipped_reason")
    )
    direct_fast_path_summary.update(
        {
            "identity_llm_skipped_reason_counts": dict(
                identity_skip_reason_counter
            ),
            "source_credibility_llm_assessed_count": credibility_summary.get(
                "llm_assessed_count", 0
            ),
            "source_credibility_llm_skipped_count": credibility_summary.get(
                "llm_skipped_count", 0
            ),
            "credibility_llm_skipped_reason_counts": dict(
                credibility_skip_reason_counter
                or Counter(credibility_summary.get("llm_skipped_reason_counts") or {})
            ),
        }
    )

    critic_summary = {
        "input_registry_count": len(registry),
        "critic_reviewed_count": len(updated),
        "critic_decision_counts": dict(critic_decision_counter),
        "decision_counts": dict(llm_critic_decision_counter),
        "fetch_recommendation_counts": dict(llm_fetch_recommendation_counter),
        "risk_flag_counts": dict(llm_risk_flag_counter),
        "agreement_count": agreement_count,
        "disagreement_count": disagreement_count,
        "critic_flag_counts": dict(critic_flag_counter),
        "collection_mode": collection_mode,
        "source_critic_selection_summary": source_critic_selection_summary,
        "llm_source_critic_enabled": llm_source_critic_enabled,
        "attempted_source_count": llm_attempted_source_count,
        "assessed_source_count": llm_assessed_source_count,
        "skipped_source_count": llm_skipped_source_count,
        "failed_source_count": llm_source_critic_failure_count,
        "fail_open_skipped_source_count": fail_open_skipped_source_count,
        "fail_open_reason_counts": dict(fail_open_reason_counter),
        "blocked_fetch_count": llm_blocked_fetch_count,
        "needs_review_count": llm_review_required_count,
        "allowed_fetch_count": llm_allowed_fetch_count,
        "context_only_count": llm_context_only_count,
        "selected_source_ids": list(llm_selected_source_ids),
        "blocked_source_ids": list(llm_blocked_source_ids),
        "source_routing_method_counts": dict(source_routing_method_counter),
        "skipped_reason_counts": dict(llm_skipped_reason_counter),
        "llm_source_critic_attempted_source_count": llm_attempted_source_count,
        "llm_assessed_source_count": llm_assessed_source_count,
        "llm_skipped_source_count": llm_skipped_source_count,
        "llm_skipped_source_ids": llm_skipped_source_ids,
        "llm_source_critic_allowlist_enabled": (
            llm_source_critic_allowlist is not None
        ),
        "llm_source_critic_allowlist": (
            sorted(llm_source_critic_allowlist)
            if llm_source_critic_allowlist is not None
            else []
        ),
        "llm_source_critic_max_sources": llm_source_critic_max_sources,
        "llm_source_critic_max_failures": llm_source_critic_max_failures,
        "llm_source_critic_max_node_seconds": llm_source_critic_max_node_seconds,
        "llm_source_critic_review_blocks_fetch": llm_review_blocks_fetch,
        "llm_source_critic_failure_count": llm_source_critic_failure_count,
        "llm_semantic_leakage_count": llm_semantic_leakage_count,
        "llm_human_review_recommended_count": llm_human_review_recommended_count,
        "llm_source_critic_blocked_fetch_count": llm_blocked_fetch_count,
        "llm_source_critic_review_required_count": llm_review_required_count,
        "llm_source_critic_allowed_fetch_count": llm_allowed_fetch_count,
        "llm_source_critic_context_only_count": llm_context_only_count,
        "llm_source_critic_decision_counts": dict(llm_critic_decision_counter),
        "llm_source_critic_fetch_recommendation_counts": dict(
            llm_fetch_recommendation_counter
        ),
        "llm_source_critic_risk_flag_counts": dict(llm_risk_flag_counter),
        "credibility_rubric_version": SOURCE_CREDIBILITY_RUBRIC_VERSION,
        "credibility_level_counts": dict(credibility_level_counter),
        "low_credibility_source_count": low_credibility_source_count,
        "source_credibility_assessed_count": credibility_summary.get(
            "assessed_source_count", 0
        ),
        "source_credibility_llm_enabled": credibility_summary.get("llm_enabled"),
        "source_credibility_llm_assessed_count": credibility_summary.get(
            "llm_assessed_count", 0
        ),
        "source_credibility_llm_failure_count": credibility_summary.get(
            "llm_failure_count", 0
        ),
        "source_identity_summary": source_identity_summary,
        "source_coverage_requirement_count": len(source_coverage_requirements),
        "must_fetch_source_count": len(must_fetch_sources),
        "source_coverage_audit": source_coverage_audit,
        "direct_fast_path_summary": direct_fast_path_summary,
    }
    routing_summary = {
        "collection_mode": collection_mode,
        "final_decision_counts": dict(final_decision_counter),
        "ready_for_content_fetch_count": ready_for_content_fetch_count,
        "requires_human_review_count": requires_human_review_count,
        "deferred_search_expansion_count": deferred_search_expansion_count,
        "excluded_count": excluded_count,
        "validation_reserved_source_count": len(validation_reserved_source_ids),
        "validation_reserved_source_ids": list(validation_reserved_source_ids),
        "context_only_source_count": len(context_only_source_ids),
        "context_only_source_ids": list(context_only_source_ids),
        "llm_source_critic_enabled": llm_source_critic_enabled,
        "source_critic_selection_summary": source_critic_selection_summary,
        "source_critic_blocked_fetch_count": llm_blocked_fetch_count,
        "source_critic_blocked_source_ids": list(llm_blocked_source_ids),
        "source_critic_review_required_count": llm_review_required_count,
        "source_critic_fail_open_skipped_source_count": (
            fail_open_skipped_source_count
        ),
        "source_critic_fail_open_reason_counts": dict(fail_open_reason_counter),
        "llm_source_critic_attempted_source_count": llm_attempted_source_count,
        "llm_assessed_source_count": llm_assessed_source_count,
        "llm_skipped_source_count": llm_skipped_source_count,
        "llm_source_critic_review_blocks_fetch": llm_review_blocks_fetch,
        "llm_semantic_leakage_count": llm_semantic_leakage_count,
        "llm_human_review_recommended_count": llm_human_review_recommended_count,
        "credibility_rubric_version": SOURCE_CREDIBILITY_RUBRIC_VERSION,
        "credibility_level_counts": dict(credibility_level_counter),
        "low_credibility_source_count": low_credibility_source_count,
        "source_credibility_role_counts": credibility_summary.get("role_counts", {}),
        "source_credibility_human_review_count": credibility_summary.get(
            "needs_review_count", 0
        ),
        "source_identity_assessed_count": source_identity_summary.get(
            "identity_assessed_count", 0
        ),
        "source_identity_llm_assessed_count": source_identity_summary.get(
            "llm_identity_assessed_count", 0
        ),
        "source_coverage_requirement_count": len(source_coverage_requirements),
        "must_fetch_source_count": len(must_fetch_sources),
        "source_coverage_audit": source_coverage_audit,
        "direct_fast_path_summary": direct_fast_path_summary,
    }

    trace = append_trace(
        state,
        node_name="source_critic_and_uncertainty_routing",
        message=(
            f"Critic reviewed {len(updated)} sources; "
            f"{ready_for_content_fetch_count} ready for fetch, "
            f"{deferred_search_expansion_count} deferred, "
            f"{requires_human_review_count} flagged for human review."
        ),
        metadata={**critic_summary, **routing_summary},
    )
    return {
        "source_registry": updated,
        "human_review_queue": human_review_queue,
        "source_critic_summary": critic_summary,
        "source_critic_results": source_critic_results,
        "source_routing_summary": routing_summary,
        "source_credibility_assessments": credibility_assessments,
        "source_credibility_summary": credibility_summary,
        "source_identity_assessments": source_identity_assessments,
        "source_identity_summary": source_identity_summary,
        "source_coverage_requirements": source_coverage_requirements,
        "source_coverage_audit": source_coverage_audit,
        "must_fetch_sources": must_fetch_sources,
        "source_critic_selection_summary": source_critic_selection_summary,
        "direct_fast_path_summary": direct_fast_path_summary,
        "collection_trace": trace,
    }
