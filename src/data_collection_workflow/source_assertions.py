"""Source-only lexical projections and typed assertions shared by evidence consumers.

Offsets always address the immutable original string. Normalization changes
presentation, never adds a geography, a date component, or a measured value.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
import unicodedata

# Explicit three-digit groups include typographic apostrophe/prime variants.
# Keep one separator contract for recognition, scalar conversion and proof.
_GROUP_SEPARATOR = r"[,\u00a0\u202f '\u2019\u02bc\u02b9\u2032]"
NUMBER_TOKEN = r"(?:\d{1,3}(?:" + _GROUP_SEPARATOR + r"\d{3})+|\d+)(?:\.\d+)?"
_BOUNDED = re.compile(r"(?:at least|more than|over|at most|fewer than|less than|approximately|about|"
                      r"plus de|au moins|moins de|au plus|environ|pr[e\u00e8]s de)\s*$", re.I)


def sentence_spans(text, *, semicolons=False, paragraphs=False):
    """Yield literal sentence offsets without splitting an abbreviated month.

    The exception is deliberately narrow: an explicit month abbreviation must
    be followed by a day. Other full stops remain evidence boundaries.
    """
    separator = r'(?<=[.!?])\s+'
    if semicolons:
        separator += r'|;\s*'
    if paragraphs:
        separator += r'|\n\s*\n'
    start = 0
    for match in re.finditer(separator, text):
        if (re.search(r'\b(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.$',
                      text[max(0, match.start()-6):match.start()], re.I)
                and re.match(r'\d{1,2}(?:\b|st\b|nd\b|rd\b|th\b)', text[match.end():], re.I)):
            continue
        end = match.start() + (1 if text[match.start():match.start()+1] == ';' else 0)
        if start < end:
            yield start, end
        start = match.end()
    if start < len(text):
        yield start, len(text)


def sentence_bounds(text, start, end, **options):
    return next(((left, right) for left, right in sentence_spans(text, **options)
                 if left <= start < end <= right), (start, end))


@dataclass(frozen=True)
class TextProjection:
    text: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]

    def original_span(self, start, end):
        if not (0 <= start < end <= len(self.text)):
            raise ValueError("invalid analysis span")
        return self.starts[start], self.ends[end - 1]


def _line_labels(text):
    lines = list(re.finditer(r"(?m)^([ \t]*)(\d{1,4})[ \t]+([^\r\n]+)", text))
    removed = []
    for left, right in zip(lines, lines[1:]):
        # Year-indexed data rows and table rows retain their numeric scope.
        if any(1900 <= int(item.group(2)) <= 2099 or "|" in item.group(3) for item in (left, right)):
            continue
        if int(right.group(2)) != int(left.group(2)) + 1:
            continue
        if text[left.end():right.start()].strip():
            continue
        # A numbered list of measurements is not manuscript line numbering.
        if re.match(r"(?:confirmed|suspected|probable|cases?|cas|deaths?)\b", left.group(3), re.I):
            continue
        if not re.search(r"\d", left.group(3) + right.group(3)):
            continue
        for item in (left, right):
            removed.append((item.start(), item.start(3)))
    return sorted(set(removed))


def analysis_projection(text, *, collapse_whitespace=False, fold=False):
    removed = _line_labels(text)
    output, starts, ends = [], [], []
    removed_index = 0
    for at, char in enumerate(text):
        while removed_index < len(removed) and at >= removed[removed_index][1]:
            removed_index += 1
        if removed_index < len(removed) and removed[removed_index][0] <= at:
            continue
        normalized = unicodedata.normalize("NFKC", char)
        if fold:
            normalized = "".join(value for value in unicodedata.normalize("NFKD", normalized.casefold())
                                 if not unicodedata.combining(value))
        for value in normalized:
            if value in "\u00a0\u202f":
                value = " "
            if collapse_whitespace and value.isspace():
                value = " "
                if output and output[-1] == " ":
                    ends[-1] = at + 1
                    continue
            output.append(value)
            starts.append(at)
            ends.append(at + 1)
    return TextProjection("".join(output), tuple(starts), tuple(ends))


def normalized_quote_spans(text, quote):
    projection = analysis_projection(text, collapse_whitespace=True)
    requested = analysis_projection(quote, collapse_whitespace=True).text.strip()
    if not requested:
        return []
    return [projection.original_span(match.start(), match.end())
            for match in re.finditer(re.escape(requested), projection.text)]


def number_value(token):
    if not re.fullmatch(NUMBER_TOKEN, token):
        return None
    value = re.sub(_GROUP_SEPARATOR, "", token)
    try:
        return int(value) if re.fullmatch(r"\d+", value) else float(value)
    except ValueError:
        return None


def number_pattern(value):
    token = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
    if re.fullmatch(r"\d+", token):
        grouped = format(int(token), ",").split(",")
        alternatives = [re.escape(token)]
        if len(grouped) > 1:
            alternatives.append(_GROUP_SEPARATOR.join(grouped))
        token = "(?:" + "|".join(alternatives) + ")"
    else:
        token = re.escape(token)
    # Neither the beginning nor the last group of a grouped number is a count.
    return r"(?<![\w.,])(?<!\d" + _GROUP_SEPARATOR + ")" + token + r"(?![\d.,%]|" + _GROUP_SEPARATOR + r"\d)"


def numeric_span_is_complete(text, start, end):
    """Reject a numeric fragment across a full run of grouping glyphs.

    Explicit date punctuation is a clause boundary, not a number group. Ordinary
    sentence quotes with no preceding numeric group also remain valid.
    """
    prefix = re.search(r"(?P<number>\d+)(?P<marks>" + _GROUP_SEPARATOR + r"+)$", text[:start])
    if prefix:
        marks = prefix.group('marks')
        date_punctuation = False
        if re.fullmatch(r",[ \u00a0\u202f]+['\u2019]?", marks):
            date_prefix = text[max(0, prefix.start('number') - 80):prefix.end('number')]
            date_punctuation = (any(m.end() == len(date_prefix) for m in _DATE.finditer(date_prefix))
                                or contextual_number_role(text, *prefix.span('number')) == 'year')
        if not date_punctuation:
            return False
    return not re.match(_GROUP_SEPARATOR + r"+\d", text[end:])


def case_bucket(text):
    if re.search(r"\b(?:confirmed|confirm[e\u00e9]s?|laboratory|laboratoire)\b", text, re.I):
        return "cases_confirmed"
    if re.search(r"\bprobables?\b", text, re.I):
        return "cases_probable"
    if re.search(r"\b(?:suspected|suspects?)\b", text, re.I):
        return "cases_suspected"
    return "cases_unspecified"



def bounded_count(text, start, end):
    """Bounds and range endpoints cannot be promoted to exact measurements."""
    prefix, suffix = text[max(0, start - 80):start], text[end:end + 50]
    if _BOUNDED.search(prefix) or re.search(r"(?:[<>]=?|[\u2264\u2265])\s*$", prefix):
        return True
    connector = r"(?:[-\u2013\u2014]|to|through|and|et|a|\u00e0)"
    if re.search(NUMBER_TOKEN + r"\s*" + connector + r"\s*$", prefix, re.I):
        return True
    return bool(re.match(r"\s*" + connector + r"\s*" + NUMBER_TOKEN + r"(?!\w)", suffix, re.I))


_CONTEXT_NUMBER_PREFIXES = (
    ("page", r"pages?|p{1,2}\."),
    ("line", r"lines?|lignes?|ln\."),
    ("section", r"sections?|sec\.|chapters?|chapitres?"),
    ("reference", r"references?|refs?\.?|citations?|numero"),
    ("age", r"ages?|aged"),
    ("table", r"tables?|tableaux?"),
    ("figure", r"figures?|fig\."),
)


def contextual_number_role(text, start, end):
    """Recognize an explicit role immediately attached to this number token."""
    prefix = _fold(text[max(0, start - 80):start])
    for role, label in _CONTEXT_NUMBER_PREFIXES:
        if re.search(r"\b(?:" + label + r")\s*(?:(?:no\.?|number)\s*)?[:#]?\s*$", prefix):
            return role
    if re.search(r"\[\s*$", prefix) and re.match(r"\s*\]", text[end:]):
        return "reference"
    suffix = _fold(text[end:end + 50])
    if re.match(r"\s+(?:years?|months?|weeks?|days?|hours?|minutes?|ans?|annees?|mois|semaines?|jours?|heures?)\b", suffix):
        return "duration"
    # Only an explicitly temporal prefix establishes a year role. A four-digit
    # quantity such as "reported 2025 cases" remains a possible measured count.
    if re.fullmatch(r"\d{4}", text[start:end]) and (
            re.search(r"\b(?:during|since|until|through|year|en|depuis|durant|pendant)\s*(?:the\s+year\s*)?$", prefix)
            or (re.search(r"\bin\s*(?:the\s+year\s*)?$", prefix)
                and not re.search(r"\bresult(?:s|ed|ing)?\s+in\s*$", prefix))):
        return "year"
    return None


# Explicit counted objects, not page topics. A disease vaccination article can
# contain both disease cases and adverse-event cases with different numerators.
_CASE_OBJECT_MODIFIER = r"(?:(?:confirmed|suspected|reported|serious|severe|new|total)\s+)*"
_ADVERSE_OBJECT = r"(?:(?:serious|severe)\s+)?adverse\s+(?:events?|reactions?|effects?)"
_NON_DISEASE_CASE_LABEL = (
    _CASE_OBJECT_MODIFIER + r"(?:cases?\s+of\s+" + _ADVERSE_OBJECT +
    r"|" + _ADVERSE_OBJECT + r"\s+cases?)"
    r"|cas(?:\s+(?:confirm[eé]\u0301?s?|suspects?|graves?))?\s+(?:d['’]\s*|de\s+)"
    r"(?:effets?\s+ind[eé]\u0301?sirables?|r[eé]\u0301?actions?\s+ind[eé]\u0301?sirables?)(?:\s+graves?)?"
)


def case_count_label_role(label):
    """Recognize an explicit non-disease object in one complete column label."""
    label = _fold(str(label)).strip()
    # Count-unit wrappers describe the same column object, not another scope.
    label = re.sub(r"^(?:number of\s+|nombre de\s+|nombre d['’]\s*)", "", label)
    label = re.sub(r"\s*\(\s*(?:n|count|nombre)\s*\)\s*$", "", label)
    bare_object = (_CASE_OBJECT_MODIFIER + _ADVERSE_OBJECT +
                   r"|(?:effets?\s+indesirables?|reactions?\s+indesirables?)(?:\s+graves?)?")
    return "adverse_event" if re.fullmatch(r"(?:" + _NON_DISEASE_CASE_LABEL + r"|" + bare_object + r")",
                                         label, re.I) else None


def case_count_object_role(text, start, end):
    """Bind an explicit object to this numeric token, never the whole paragraph."""
    after = re.compile(r"\s+(?:" + _NON_DISEASE_CASE_LABEL + r")(?!\w)", re.I)
    if after.match(text, end):
        return "adverse_event"
    # Test a reverse label only when this token is directly after its separator.
    # No fixed character window may hide an object behind long layout spacing.
    boundary = start
    while boundary and text[boundary - 1].isspace():
        boundary -= 1
    if boundary and text[boundary - 1] in ":=":
        before = re.compile(r"(?<!\w)(?:" + _NON_DISEASE_CASE_LABEL + r")\s*[:=]\s*$", re.I)
        if before.search(text, 0, start):
            return "adverse_event"
    return None


def count_mentions(text):
    """Find measured count phrases, retaining original offsets and qualifiers."""
    projection = analysis_projection(text)
    normalized = projection.text
    result = []
    # The intervening words may name the disease, but never another quantity.
    patterns = [
        ("cases", re.compile(r"(?P<number>" + NUMBER_TOKEN + r")\s+"
            r"(?P<before>(?:[^\W\d_][\w/-]*\s+){0,5}?)(?P<label>cases?|cas|infections?)\b"
            r"(?P<after>(?:\s+(?:confirm[e\u00e9]s?|suspects?|probables?))?)", re.I)),
        ("deaths", re.compile(r"(?P<number>" + NUMBER_TOKEN + r")\s+"
            r"(?:(?:reported|associated|additional|total)\s+){0,4}(?P<label>deaths?|fatalities|d[e\u00e9]c[e\u00e8]s)\b", re.I)),
        ("hospitalizations", re.compile(r"(?P<number>" + NUMBER_TOKEN + r")\s+"
            r"(?:reported\s+)?(?P<label>hospitali[sz]ations?)\b", re.I)),
    ]
    for kind, pattern in patterns:
        for match in pattern.finditer(normalized):
            left, right = match.span()
            if contextual_number_role(normalized, *match.span("number")):
                continue
            if left and (normalized[left - 1].isalnum() or normalized[left - 1] in ".,"):
                continue
            if not numeric_span_is_complete(normalized, *match.span("number")):
                continue
            if kind == "cases":
                if case_count_object_role(normalized, *match.span("number")):
                    continue
                before = match.group("before")
                if re.search(r"\b(?:tests?|samples?|specimens?|countries|contacts?|passengers?|crew|percent)\b", before, re.I):
                    continue
                if re.match(r"\s+(?:investigation|report|study|series|definition|management|history)\b", normalized[right:], re.I):
                    continue
                field = case_bucket(match.group())
            else:
                field = kind
            sentence_left, sentence_right = sentence_bounds(normalized, left, right)
            start, end = projection.original_span(left, right)
            s_start, s_end = projection.original_span(sentence_left, sentence_right)
            result.append({"field": field, "value": number_value(match.group("number")),
                "span": text[start:end], "sentence": text[s_start:s_end],
                "char_start": start, "char_end": end,
                "bounded": bounded_count(normalized, left, left + len(match.group("number")))})
    return result


_MONTHS = {
    "january": 1, "janvier": 1, "jan": 1, "february": 2, "fevrier": 2, "feb": 2,
    "march": 3, "mars": 3, "mar": 3, "april": 4, "avril": 4, "apr": 4,
    "may": 5, "mai": 5, "june": 6, "juin": 6, "jun": 6,
    "july": 7, "juillet": 7, "jul": 7, "august": 8, "aout": 8, "aug": 8,
    "september": 9, "septembre": 9, "sep": 9, "sept": 9,
    "october": 10, "octobre": 10, "oct": 10, "november": 11, "novembre": 11, "nov": 11,
    "december": 12, "decembre": 12, "dec": 12,
}
_MONTH = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_DAY = r"\d{1,2}(?:st|nd|rd|th|er)?"
_YEAR = r"(?:19|20)\d{2}"
_DATE_LABEL = rf"(?:{_YEAR}-\d{{2}}-\d{{2}}|(?:{_DAY}\s+)?{_MONTH}(?:\s+{_DAY},?)?(?:\s+{_YEAR})?)"
_INTERVAL = re.compile(rf"(?<![\w./\u2013\u2014-])(?:(?P<range_prefix>between|entre)\s+)?(?P<start>{_DATE_LABEL}|{_DAY})\s*(?P<connector>to|through|until|and|et|au|a|[-\u2013\u2014])\s*(?P<end>{_DATE_LABEL})(?!\w)", re.I)
_DATE = re.compile(rf"(?<!\w){_DATE_LABEL}(?!\w)", re.I)


def _fold(text):
    return "".join(char for char in unicodedata.normalize("NFKD", text.casefold())
                   if not unicodedata.combining(char))


def _date_parts(token):
    token = token.strip().lower().rstrip(".")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
        try:
            value = date.fromisoformat(token)
            return value.year, value.month, value.day
        except ValueError:
            return None
    month = re.search(_MONTH, token, re.I)
    if not month:
        return None
    month_value = _MONTHS[month.group().rstrip(".")]
    before, after = token[:month.start()], token[month.end():]
    year = re.search(r"\b" + _YEAR + r"\b", after)
    day = re.search(_DAY, before) or re.match(r"\s*(" + _DAY + r")(?!\d)", re.sub(_YEAR, "", after))
    day_value = int(re.match(r"\d+", day.group().strip()).group()) if day else None
    year_value = int(year.group()) if year else None
    if year_value and day_value:
        try:
            date(year_value, month_value, day_value)
        except ValueError:
            return None
    return year_value, month_value, day_value


def _metadata_date(text, start):
    prefix = text[max(0, start - 65):start]
    return bool(re.search(r"\b(?:publish|published|publication|updated|posted|accessed|copyright|"
                          r"publie|publication|mis a jour)\s*(?:(?:online|on|le|date)\s*)?[:\-]?\s*$", prefix, re.I))


def publication_dates(text):
    """Return explicitly labelled publication dates, never observation dates."""
    projection = analysis_projection(text, fold=True)
    result = []
    for match in _DATE.finditer(projection.text):
        parts = _date_parts(match.group())
        prefix = projection.text[max(0, match.start()-65):match.start()]
        if (not parts or any(part is None for part in parts) or not re.search(
                r'\b(?:publish|published|publication|publie)\s*(?:(?:online|on|le|date)\s*)?[:\-]?\s*$', prefix)):
            continue
        start, end = projection.original_span(match.start(), match.end())
        result.append({'value':date(*parts).isoformat(), 'quote':text[start:end],
                       'char_start':start, 'char_end':end})
    return result


def _source_intervals(text):
    projection = analysis_projection(text, fold=True)
    folded = projection.text
    result = []
    for match in _INTERVAL.finditer(folded):
        first, last = _date_parts(match.group("start")), _date_parts(match.group("end"))
        # A day-only first endpoint shares the explicit second endpoint's month.
        # It never guesses a prior month when the resulting range is reversed.
        day_only = first is None and re.fullmatch(_DAY, match.group("start"))
        discrete_connector = match.group("connector").casefold() in {"and", "et"}
        if day_only and discrete_connector and not match.group("range_prefix"):
            continue
        if day_only and last and last[2] is not None:
            first = (None, last[1], int(re.match(r"\d+", match.group("start")).group()))
        if not first or not last or _metadata_date(folded, match.start()):
            continue
        year = first[0] or last[0]
        if not year:
            continue
        first = (first[0] or year, *first[1:])
        last = (last[0] or year, *last[1:])
        # Missing years may be inherited only within this explicit interval;
        # validate again now that formerly partial dates have a calendar year.
        try:
            for parts in (first, last):
                if parts[2] is not None:
                    date(*parts)
        except ValueError:
            continue
        if first[:2] > last[:2] or (first[:2] == last[:2] and first[2] is not None and last[2] is not None and first[2] > last[2]):
            continue
        # Retain the required between/entre marker so the saved source label
        # remains an interval when it is parsed independently later.
        quote_start = match.start() if day_only and discrete_connector else match.start("start")
        start, end = projection.original_span(quote_start, match.end())
        result.append({"period": (first, last), "quote": text[start:end]})
    return result


def date_intervals(text):
    return [item["period"] for item in _source_intervals(text)]


def observation_dates(text):
    """Recover precise metric endpoints while retaining incomplete source labels."""
    periods = _source_intervals(text)
    result = {}
    if len({item["period"] for item in periods}) == 1:
        item = periods[0]
        first, last = item["period"]
        result["reporting_period"] = item["quote"]
        result["date_text_span"] = item["quote"]
        for field, parts in (("metric_period_start", first), ("metric_period_end", last)):
            if all(value is not None for value in parts):
                result[field] = date(*parts).isoformat()
    elif not periods:
        projection = analysis_projection(text, fold=True)
        dates = {field: {} for field in ("as_of_date", "metric_period_start", "metric_period_end")}
        for match in _DATE.finditer(projection.text):
            parts = _date_parts(match.group())
            if not parts or any(value is None for value in parts):
                continue
            value = date(*parts).isoformat()
            start, end = projection.original_span(match.start(), match.end())
            for field, values in dates.items():
                # A standalone as-of observation is not an inferred interval.
                if field == "metric_period_end" and typed_date_support("as_of_date", value, text):
                    continue
                if typed_date_support(field, value, text):
                    values[value] = text[start:end]
        for field, values in dates.items():
            if len(values) == 1:
                value, quote = next(iter(values.items()))
                result[field] = value
                if "date_anchor_type" not in result:
                    result.update(date_text_span=quote, date_anchor_type=field)
    return result


def typed_date_support(name, value, text):
    """Return a decision for typed period conversions, otherwise no opinion."""
    token = str(value).strip()
    source_periods = date_intervals(text)
    if name == "reporting_period":
        requested = date_intervals(token)
        if requested:
            return bool(source_periods and any(period in source_periods for period in requested))
        return None
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
        return None
    try:
        expected = date.fromisoformat(token)
    except ValueError:
        return False
    target = (expected.year, expected.month, expected.day)
    if name in {"metric_period_start", "period_start_date"} and source_periods:
        return any(first == target for first, _ in source_periods)
    if name in {"metric_period_end", "period_end_date"} and source_periods:
        return any(last == target for _, last in source_periods)
    folded = _fold(text)
    candidates = [(match, _date_parts(match.group())) for match in _DATE.finditer(folded)]
    if any(parts == target and _metadata_date(folded, match.start()) for match, parts in candidates):
        # A separate correctly typed observation occurrence may still exist.
        candidates = [(match, parts) for match, parts in candidates if not _metadata_date(folded, match.start())]
        if not candidates:
            return False
    if name in {"date_reported", "report_date"}:
        weekday = r"(?:(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\s+)?"
        report_verb = r"\b(?:reported|notified|signale[es]?|notifie[es]?)\b"
        competing_role = r"\b(?:onset|died|death|diagnosed|occurred|started|began|symptomes|deces|debut)\b"
        for match, parts in candidates:
            if parts != target or _metadata_date(folded, match.start()):
                continue
            # A report date belongs to this reporting predicate, not another
            # sentence, a publication label, or a clinical event in the report.
            left, right = sentence_bounds(folded, match.start(), match.end(), semicolons=True)
            before = folded[left:match.start()].split('\n')[-1]
            after = folded[match.end():right].split('\n')[0]
            direct = r"\b(?:reported|report date|date reported|date de signalement|date de notification)\s*(?:(?:on|le)\s+|[:=]\s*)?" + weekday + r"$"
            if re.search(direct, before):
                return True
            if not re.search(r"\b(?:on|le)\s+" + weekday + r"$", before):
                continue
            verbs = list(re.finditer(report_verb, before))
            if verbs and not re.search(competing_role, before[verbs[-1].end():]):
                return True
            following_report = re.search(report_verb, after)
            if (re.fullmatch(r"\s*(?:on|le)\s+" + weekday, before) and following_report
                    and not re.search(competing_role, after[:following_report.start()])):
                return True
        return False
    # A generic article/preposition cannot override an explicit start/end role
    # immediately before this same date occurrence.
    start_role = (r"\b(?:since|depuis|a partir (?:du|de))\s+(?:(?:the|le|la|on)\s+)?$"
                  r"|(?:^|[.!?;\n]\s*)from\s+(?:the\s+)?$")
    end_role = r"\b(?:until|through|ending|jusqu['’]au)\s+(?:(?:the|le|la|on)\s+)?$"
    roles = {
        "as_of_date": r"\b(?:as of|on|au|le|a la date du)\s*$",
        "metric_period_end": r"\b(?:as of|au)\s*$|" + end_role,
        "period_end_date": r"\b(?:au)\s*$|" + end_role,
        "metric_period_start": start_role,
        "period_start_date": start_role,
    }
    if name in {"event_start_date", "event_end_date"}:
        action = r"(?:started|began|onset|start|debut)" if name == "event_start_date" else r"(?:ended|ceased|end|fin)"
        role = r"\b(?:event|outbreak|epidemic|epidemie|flambee)\b[^.!?;\n]{0,35}\b" + action + r"\s*(?:on|le|date|[:=])?\s*$"
        return any(parts == target and re.search(role, folded[max(0, match.start() - 90):match.start()])
                   and not _metadata_date(folded, match.start()) for match, parts in candidates)
    if name in roles:
        for match, parts in candidates:
            if parts != target or _metadata_date(folded, match.start()):
                continue
            prefix = folded[max(0, match.start() - 80):match.start()]
            if name == "as_of_date":
                event_role = (r"\b(?:confirmed|diagnosed|died|began|started|"
                              r"developed\s+symptoms|confirme[es]*|diagnostique[es]*|"
                              r"decede[es]*|commence[es]*|debute[es]*)\s+(?:on|le)\s*$")
                if re.search(start_role, prefix) or re.search(end_role, prefix) or re.search(event_role, prefix):
                    continue
            if re.search(roles[name], prefix):
                return True
        return False
    return None


def annual_period_support(year, text):
    """A year label cannot replace an explicitly shorter observation window."""
    if not re.fullmatch(r"\d{4}", str(year)):
        return False
    year = int(year)
    periods = date_intervals(text)
    if periods:
        return all(first == (year, 1, 1) and last == (year, 12, 31)
                   for first, last in periods)
    folded = _fold(text)
    # Any source-local observation day/month retains its finer precision.
    # Metadata dates do not describe the measured interval.
    if any(parts and parts[0] == year and not _metadata_date(folded, match.start())
           for match in _DATE.finditer(folded) if (parts := _date_parts(match.group()))):
        return False
    return bool(re.search(r"(?<!\d)" + str(year) + r"(?!\d)", folded))
