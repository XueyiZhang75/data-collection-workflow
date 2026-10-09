"""Deterministic geography anchoring helpers for public-health records."""

from __future__ import annotations

from typing import Any
from functools import lru_cache
import re
import unicodedata


EXPLICIT_COUNTRY_NAMES = (
    "Argentina",
    "Bolivia",
    "Brazil",
    "Canada",
    "Chile",
    "Germany",
    "Netherlands",
    "Panama",
    "Paraguay",
    "South Africa",
    "Switzerland",
    "United States",
    "Uruguay",
)


US_STATES = {
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new hampshire",
    "new jersey",
    "new mexico",
    "new york",
    "north carolina",
    "north dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode island",
    "south carolina",
    "south dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west virginia",
    "wisconsin",
    "wyoming",
    "district of columbia",
}


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_us_state(value: Any) -> bool:
    return _lower(value) in US_STATES


def _is_county(value: Any) -> bool:
    return "county" in _lower(value)


def _task_location(task: dict | None) -> str | None:
    if not isinstance(task, dict):
        return None
    return _clean(task.get("location") or task.get("geography") or task.get("task_location"))


def _source_jurisdiction(source_identity: dict | None) -> tuple[str | None, str | None]:
    if not isinstance(source_identity, dict):
        return None, None
    level = _clean(source_identity.get("jurisdiction_level"))
    name = _clean(
        source_identity.get("jurisdiction_name")
        or source_identity.get("jurisdiction")
        or source_identity.get("jurisdiction_scope_name")
    )
    scope = _clean(source_identity.get("jurisdiction_scope"))
    if name:
        return name, level or scope
    if scope and scope not in {"subnational", "national", "international", "regional"}:
        return scope, level
    return None, level or scope


def _set_state_scope(out: dict, state_name: str, *, method: str, warning: bool) -> dict:
    out["subnational_location"] = out.get("subnational_location") or state_name
    out["country"] = out.get("country") or "United States"
    out["geographic_scope"] = out.get("geographic_scope") or state_name
    out["geographic_scope_type"] = out.get("geographic_scope_type") or "subnational"
    out["geography_inference_method"] = method
    out["geography_inference_warning"] = warning
    return out


def resolve_record_geography(
    record: dict,
    *,
    source_identity: dict | None = None,
    task: dict | None = None,
) -> dict:
    """Anchor record geography using evidence first, source second, task third.

    Source/task fallback is explicitly marked with geography_inference_warning.
    The helper never collapses a state or county task to a United States-only
    record.
    """

    out = dict(record)
    from .evidence_qualification import evidence_qualification_enabled
    if evidence_qualification_enabled():
        # Only explicit record geography may survive; source jurisdiction and
        # task scope are search constraints, not observation evidence.
        return out
    out.setdefault("geography_inference_method", "evidence_geography")
    out.setdefault("geography_inference_warning", False)

    locality = _clean(out.get("locality"))
    subnational = _clean(out.get("subnational_location"))
    scope = _clean(out.get("geographic_scope"))
    scope_type = _clean(out.get("geographic_scope_type"))
    country = _clean(out.get("country"))

    if locality:
        out["locality"] = locality
        if _is_county(locality):
            out["geographic_scope"] = locality
            out["geographic_scope_type"] = "county"
        elif not scope:
            out["geographic_scope"] = locality
            out["geographic_scope_type"] = scope_type or "locality"
        if subnational and _is_us_state(subnational):
            out["subnational_location"] = subnational
            out["country"] = country or "United States"
        return out

    if subnational:
        out["subnational_location"] = subnational
        if _is_us_state(subnational):
            out["country"] = country or "United States"
        if not scope:
            out["geographic_scope"] = subnational
            out["geographic_scope_type"] = scope_type or "subnational"
        return out

    if scope:
        out["geographic_scope"] = scope
        if _is_us_state(scope):
            out["subnational_location"] = scope
            out["country"] = country or "United States"
            out["geographic_scope_type"] = scope_type or "subnational"
        elif _lower(scope) in {"united states", "united states of america", "usa", "us"}:
            out["country"] = country or "United States"
            out["geographic_scope"] = "United States"
            out["geographic_scope_type"] = scope_type or "country"
        elif scope_type:
            out["geographic_scope_type"] = scope_type
        return out

    if country:
        out["country"] = country
        out["geographic_scope"] = country
        out["geographic_scope_type"] = scope_type or "country"
        return out

    source_area, source_level = _source_jurisdiction(source_identity)
    if source_area and _lower(source_level) in {"state", "local", "subnational"}:
        return _set_state_scope(
            out,
            source_area,
            method="source_jurisdiction",
            warning=True,
        )
    if source_area and _lower(source_level) == "national":
        out["country"] = "United States" if _lower(source_area) in {"us", "usa", "united states"} else source_area
        out["geographic_scope"] = out["country"]
        out["geographic_scope_type"] = "country"
        out["geography_inference_method"] = "source_jurisdiction"
        out["geography_inference_warning"] = True
        return out

    task_area = _task_location(task)
    if task_area and _is_us_state(task_area):
        return _set_state_scope(
            out,
            task_area,
            method="task_geography",
            warning=True,
        )
    if task_area and _lower(task_area) in {"united states", "united states of america", "usa", "us"}:
        out["country"] = "United States"
        out["geographic_scope"] = "United States"
        out["geographic_scope_type"] = "country"
        out["geography_inference_method"] = "task_geography"
        out["geography_inference_warning"] = True
    return out


def _country_name_key(value):
    """Keep accents while matching Unicode canonical and regex case equivalents."""
    key = unicodedata.normalize('NFC', str(value or '')).casefold()
    # Python's Unicode IGNORECASE treats I/i/İ/ı as equivalent; casefold alone
    # does not. Apply the same equivalence to vocabulary and lookup keys.
    return re.sub('i\u0307+', 'i', key.replace('\u0131', 'i'))


@lru_cache(maxsize=1)
def _explicit_country_vocabulary():
    """Standard CLDR country/territory names; no task, source or publisher facts."""
    from babel import Locale
    from .config import load_record_normalization_policy
    english = Locale.parse('en').territories
    excluded = {'EU','EZ','UN','QO','XA','XB','ZZ'}
    names = {}
    for language in ('en','fr','es','pt','zh_Hans'):
        for code, name in Locale.parse(language).territories.items():
            if len(code) == 2 and code.isalpha() and code not in excluded and code in english:
                names[_country_name_key(name)] = _country_name_key(english[code])
    for alias, target in (load_record_normalization_policy().get('country_aliases') or {}).items():
        if '/' not in target:
            canonical = names.get(_country_name_key(target),names.get(_country_name_key(alias),_country_name_key(target)))
            names[_country_name_key(alias)] = canonical
            names[_country_name_key(target)] = canonical
    # Match canonical decompositions in place, retaining original quotes and
    # offsets. No blanket accent stripping or new country aliases are added.
    variants = {unicodedata.normalize(form, name) for name in names for form in ('NFC', 'NFD')}
    terms = [re.escape(name).replace('i', 'i\u0307?') for name in sorted(variants, key=lambda name: (-len(name), name))]
    pattern = re.compile(r'(?<!\w)(?:'+'|'.join(terms)+r')(?!\w)',re.I)
    return names, pattern


def explicit_country_key(value):
    names, _ = _explicit_country_vocabulary()
    key = _country_name_key(value)
    return names.get(key,key)


def explicit_country_matches(text):
    names, pattern = _explicit_country_vocabulary()
    for match in pattern.finditer(text):
        name = match.group()
        key = _country_name_key(name)
        if key.isascii() and len(key.replace('.','')) <= 2 and not name.isupper():
            continue
        # A homonymous explicitly labelled subnational place is not a country.
        if re.search(r'\b(?:state|province|city|district|county)\s+of\s*$',text[max(0,match.start()-50):match.start()],re.I):
            continue
        yield name, names[key]


CANADIAN_PROVINCES = (
    "Alberta", "British Columbia", "Manitoba", "New Brunswick",
    "Newfoundland and Labrador", "Northwest Territories", "Nova Scotia",
    "Nunavut", "Ontario", "Prince Edward Island", "Quebec", "Saskatchewan", "Yukon",
)


def explicit_statistical_scope(text, *, country=None):
    """Bind a named province to its count; parent country needs source evidence.

    A travel origin, institution, or list of contributing provinces does not
    establish the statistical scope. No task or publisher metadata is used.
    """
    from .source_assertions import count_mentions, sentence_spans
    if country and explicit_country_key(country) != explicit_country_key("Canada"):
        return {}
    matches = []
    for canonical in CANADIAN_PROVINCES:
        label = r"Qu[e\u00e9]bec" if canonical == "Quebec" else re.escape(canonical)
        for match in re.finditer(r"(?<!\w)" + label + r"(?!\w)", text, re.I):
            # A row label binds the row's values independently of document scope.
            row = text.strip().strip("|").strip()
            cells = [cell.strip() for cell in row.split("|")]
            if (len(cells) > 1 and re.fullmatch(label, cells[0], re.I)
                    and any(re.search(r"\d", cell) for cell in cells[1:])):
                matches.append((canonical, match.group()))
                continue
            left, right = next(((a, b) for a, b in sentence_spans(text)
                                if a <= match.start() < b), (0, len(text)))
            sentence = text[left:right]
            if not count_mentions(sentence):
                continue
            named_provinces = {name for name in CANADIAN_PROVINCES
                               if re.search(r"(?<!\w)" + (r"Qu[e\u00e9]bec" if name == "Quebec" else re.escape(name)) + r"(?!\w)", sentence, re.I)}
            if len(named_provinces) > 1:
                continue
            before, after = text[left:match.start()], text[match.end():right]
            inclusion = re.search(r",\s*(?:including|of\s+which|dont|y\s+compris)\b", before, re.I)
            if inclusion and count_mentions(before[:inclusion.start()]):
                continue
            actor_location = re.search(r"\b(?:University|Hospital|Ministry|Department|Government|Laboratory|Agency|Institute|Office|Center|Centre)\b[^.!?;]*\b(?:in|of)\s*$", before, re.I)
            if actor_location and re.match(r"\s+(?:reported|reports|recorded|identified|confirmed|published)\b", after, re.I):
                continue
            if re.search(r"\b(?:travel(?:led|ed)?|visited|return(?:ed)?|imported|exposed|exposure)\b[^.!?;]*$", before, re.I):
                continue
            if re.search(r"\b(?:University|Hospital|Ministry|Department|Government)\s+of\s+$", before, re.I):
                continue
            if re.match(r"\s+(?:Health|University|Hospital|Ministry|government)\b", after, re.I):
                continue
            local = bool(re.search(r"\b(?:in|across|within)\s+(?:the province of\s+)?$", before, re.I)
                         or re.match(r"\s+(?:reported|reports|recorded|recording|had|has|identified|confirmed)\b", after, re.I))
            if local:
                matches.append((canonical, match.group()))
    if len({name for name, _ in matches}) != 1:
        return {}
    canonical, literal = matches[0]
    parent = {key for _, key in explicit_country_matches(text)}
    result = {"country": None, "subnational_location": literal,
              "geographic_scope": literal, "geographic_scope_type": "subnational",
              "location_type": "subnational", "geography_text_span": literal,
              "geography_inference_method": "explicit_province_scope",
              "geography_inference_warning": False}
    if parent == {explicit_country_key("Canada")}:
        result["country"] = "Canada"
        result["geography_inference_method"] = "explicit_province_parent_hierarchy"
    return result
