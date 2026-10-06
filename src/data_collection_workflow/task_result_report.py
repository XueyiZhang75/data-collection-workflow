"""Evidence-bound answers to the user's collection task, separate from run diagnostics."""
from __future__ import annotations

from datetime import date
import math
import re
import unicodedata
from pathlib import Path


COUNT_FIELDS = (
    'cases_confirmed', 'cases_probable', 'cases_suspected',
    'cases_unspecified', 'deaths', 'hospitalizations',
)
LABELS = {
    "cases_confirmed": "confirmed cases",
    "cases_probable": "probable cases",
    "cases_suspected": "suspected cases",
    "cases_unspecified": "cases of unspecified classification",
    "deaths": "deaths",
    "hospitalizations": "hospitalizations",
}
NON_TOTAL_WORDS = re.compile(
    r'\b(?:average|mean|per day|each day|daily|weekly|monthly|rate|percent|'
    r'percentage|sample|subset|proportion|incidence)\b|%', re.I,
)


def _norm(value):
    text = unicodedata.normalize('NFKD', str(value or '').casefold()).replace('_', ' ')
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return ' '.join(re.findall(r'\w+', text))


def _number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


def _value(row, field):
    value = _number(row.get(field))
    if field in COUNT_FIELDS or value is not None:
        return value
    raw = row.get(field)
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _disease_matches(row, task):
    names = [_norm(name) for name in [task.get('disease'), *(task.get('disease_aliases') or [])]]
    observed = [_norm(row.get(field)) for field in
                ('disease_standard_name', 'disease', 'virus_or_syndrome')]
    return any(name and re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text)
               for name in names for text in observed if text)


def _location_matches(row, task, *, exact):
    from .geography import explicit_country_key

    def country_key(value):
        return _norm(explicit_country_key(value))

    raw_location = str(task.get('location') or '')
    location = _norm(raw_location)
    if not location:
        return False
    country = _norm(row.get('country'))
    scope = _norm(row.get('geographic_scope'))
    subnational = _norm(row.get('subnational_location'))
    country_canonical = country_key(row.get('country')) if country else ''
    scope_canonical = country_key(row.get('geographic_scope')) if scope else ''
    location_canonical = country_key(raw_location)
    scope_type = _norm(row.get('geographic_scope_type'))
    local_types = {'subnational', 'locality', 'state', 'province', 'county',
                   'district', 'city', 'municipality', 'local'}
    parts = [_norm(part) for part in re.split(r'[,;]', raw_location) if _norm(part)]
    if len(parts) > 1:
        values = {country, scope, subnational, country_canonical, scope_canonical} - {''}
        if not all(part in values or country_key(part) in values for part in parts):
            return False
        return not exact or parts[0] in {scope, subnational}
    if country and country_canonical == location_canonical:
        if scope_type in {'country', 'national'} and scope and scope_canonical != country_canonical:
            return False
        if not exact:
            return True
        return (not subnational and scope_canonical in {'', country_canonical}
                and scope_type not in local_types | {'multi country', 'region'})
    if location in {scope, subnational}:
        # A contradictory explicit country may not be silently replaced by a
        # same-named scope. A local scope is allowed under a different country.
        if country and scope_type not in local_types:
            return False
        return not exact or location in {scope, subnational}
    return False


def _full_period(row, task):
    start, end = str(task.get('start_date') or ''), str(task.get('end_date') or '')
    if not start or not end:
        return False
    as_of = str(row.get('as_of_date') or '')
    if as_of:
        try:
            if date.fromisoformat(as_of) < date.fromisoformat(end):
                return False
        except ValueError:
            return False
    if row.get('metric_period_start') == start and row.get('metric_period_end') == end:
        return True
    period = str(row.get('reporting_period') or '').strip()
    try:
        full_year = start == date.fromisoformat(start).replace(month=1, day=1).isoformat() and\
            end == date.fromisoformat(start).replace(month=12, day=31).isoformat()
    except ValueError:
        full_year = False
    if full_year and period == start[:4]:
        return True
    return bool(re.fullmatch(re.escape(start) + r'\s*(?:to|through|–|—)\s*' + re.escape(end), period, re.I))


def _evidence_fields(row):
    qualification = row.get('evidence_qualification') or {}
    return {item.get('field') for item in qualification.get('field_evidence') or []
            if isinstance(item, dict) and item.get('document_hash') and item.get('locator')}


def _source(row, field):
    qualification = row.get('evidence_qualification') or {}
    entries = [entry for entry in qualification.get('field_evidence') or []
               if isinstance(entry, dict) and entry.get('field') == field
               and entry.get('document_hash') and entry.get('locator')]
    entry = entries[0] if entries else {}
    return {
        'record_id': row.get('record_id'), 'url': row.get('source_url'),
        'quote': str(entry.get('quote') or row.get('evidence_quote') or '').strip(),
        'document_hash': entry.get('document_hash'),
        'locator': entry.get('locator'),
    }


def _task_measure_field(field):
    if field in COUNT_FIELDS:
        return True
    tokens = set(_norm(field).split())
    return bool(tokens & {'rate', 'ratio', 'percent', 'percentage', 'incidence',
                          'prevalence', 'coverage', 'positivity', 'mortality', 'fatality'})


def _number_match(quote, number):
    if number is None:
        return None
    patterns = {str(number), f"{number:,}", f"{number:_}".replace('_', ' ')}
    for pattern in sorted(patterns, key=len, reverse=True):
        match = re.search(r'(?<!\d)' + re.escape(pattern) + r'(?!\d)', quote)
        if match:
            return match
    return None


def _claim_window(quote, match):
    base = max(0, match.start() - 180)
    start = base
    end = min(len(quote), match.end() + 180)
    for boundary in re.finditer(r'[.!?;]\s+(?=[A-Z])|\n', quote[base:match.start()]):
        start = base + boundary.end()
    after = quote[match.end():end]
    boundary = re.search(r'[.!?;]\s+(?=[A-Z])|\n', after)
    if boundary:
        end = match.end() + boundary.start()
    return quote[start:end]


def _qualifier(quote, match):
    before = quote[max(0, match.start() - 35):match.start()].casefold()
    patterns = (
        (r'(?:over|more than|above|exceeding)\s*$', 'over'),
        (r'(?:at least|no fewer than)\s*$', 'at_least'),
        (r'(?:about|around|approximately|roughly|estimated)\s*$', 'about'),
        (r'(?:nearly|almost)\s*$', 'nearly'),
        (r'(?:up to|at most|no more than)\s*$', 'at_most'),
        (r'(?:under|less than|fewer than)\s*$', 'under'),
    )
    for pattern, label in patterns:
        if re.search(pattern, before):
            return label
    return 'exact'


def _exact_answer_eligible(row, task, field):
    if (row.get('evidence_qualification') or {}).get('status') != 'qualified':
        return False
    if not _task_measure_field(field):
        return False
    value = _value(row, field)
    if value is None or not _disease_matches(row, task):
        return False
    if not _location_matches(row, task, exact=True) or not _full_period(row, task):
        return False
    supported = _evidence_fields(row)
    if field not in supported or not ({'country', 'geographic_scope', 'subnational_location'} & supported):
        return False
    if not ({'reporting_period'} & supported or {'metric_period_start', 'metric_period_end'} <= supported):
        return False
    if field not in COUNT_FIELDS:
        field_quote = _source(row, field)['quote']
        match = _number_match(field_quote, value)
        semantic_tokens = [token for token in _norm(field).split()
                           if token not in {'rate', 'ratio', 'percent', 'percentage'}]
        if not match or not semantic_tokens or not all(token in _norm(field_quote) for token in semantic_tokens):
            return False
    if field in COUNT_FIELDS:
        semantics = _norm(row.get('count_semantics') or row.get('statistical_count_type'))
        if NON_TOTAL_WORDS.search(semantics) or semantics in {'new', 'incremental', 'incident', 'unknown', 'unspecified'}:
            return False
        population = _norm(row.get('population_scope'))
        if population and population not in {'all', 'all cases', 'all reported cases',
                                             'all confirmed cases', 'general population',
                                             'national population', 'whole population'}:
            return False
        quote = str(row.get('evidence_quote') or '')
        match = _number_match(quote, value)
        if not match:
            return False
        before = quote[max(0, match.start() - 45):match.start()]
        after = quote[match.end():match.end() + 65]
        if re.search(r'(?:averag\w*|mean|median|rate(?: of)?|per day|daily)\s*$', before, re.I):
            return False
        if re.match(r'\s*(?:%|percent\b|percentage\b)', after, re.I):
            return False
        if re.match(r'\s*(?:new\s+)?(?:cases?|deaths?)\s+(?:each|per)\s+(?:day|week|month)\b', after, re.I):
            return False
    return bool(row.get('source_url'))


def _candidate_lead(row, task, field):
    if (row.get('evidence_qualification') or {}).get('status') != 'candidate':
        return None
    if not _task_measure_field(field):
        return None
    value = _value(row, field)
    quote = str(row.get('evidence_quote') or '').strip()
    if value is None or not quote or not row.get('source_url'):
        return None
    if not _disease_matches(row, task) or not _location_matches(row, task, exact=True):
        return None
    if field in COUNT_FIELDS and NON_TOTAL_WORDS.search(_norm(row.get('count_semantics') or row.get('statistical_count_type'))):
        return None
    match = _number_match(quote, value)
    if not match:
        return None
    # Candidate metadata is not proof. It only helps route an explicitly
    # labelled numeric claim into a lead, never into a confirmed answer.
    claim = _norm(_claim_window(quote, match))
    if field in COUNT_FIELDS and re.search(
        r'\b(?:we analyzed|we included|sample of|study sample|cohort of|'
        r'genomes|sequenced|participants enrolled)\b', claim,
    ):
        return None
    diseases = [task.get('disease'), *(task.get('disease_aliases') or [])]
    if not any(_norm(name) and _norm(name) in claim for name in diseases):
        return None
    if _norm(task.get('location')) not in claim:
        return None
    years = set(re.findall(r'\b\d{4}\b', str(task.get('start_date') or '') + ' ' + str(task.get('end_date') or '')))
    date_context = ' '.join(str(row.get(key) or '') for key in
                            ('reporting_period', 'metric_period_start', 'metric_period_end', 'as_of_date'))
    if years and not any(year in quote or year in date_context for year in years):
        return None
    return {
        'value': value, 'qualifier': _qualifier(quote, match),
        'period': row.get('reporting_period') or None,
        'as_of_date': row.get('as_of_date') or row.get('metric_period_end') or None,
        'source': _source(row, field),
        'qualification_reasons': list((row.get('evidence_qualification') or {}).get('reasons') or []),
    }


def _date_rank(lead):
    value = str(lead.get('as_of_date') or '')
    match = re.search(r'\b\d{4}-\d{2}-\d{2}\b', value)
    if match:
        try:
            return date.fromisoformat(match.group())
        except ValueError:
            pass
    months = {name.lower(): number for number, name in enumerate(
        ('January', 'February', 'March', 'April', 'May', 'June', 'July',
         'August', 'September', 'October', 'November', 'December'), 1)}
    match = re.search(r'\b([A-Za-z]+)\s+(\d{4})\b', value)
    if match and match.group(1).lower() in months:
        return date(int(match.group(2)), months[match.group(1).lower()], 1)
    return date.min


def _observations(package, task, fields):
    rows = []
    for row in package.get('final_dataset') or []:
        if (row.get('evidence_qualification') or {}).get('status') != 'qualified':
            continue
        if not _disease_matches(row, task) or not _location_matches(row, task, exact=False):
            continue
        values = {field: _value(row, field) for field in fields}
        values = {field: value for field, value in values.items() if value is not None}
        if not values:
            continue
        quote = str(row.get('evidence_quote') or '')
        rows.append({
            'record_id': row.get('record_id'), 'scope': row.get('geographic_scope') or row.get('country'),
            'period': row.get('reporting_period') or None, 'as_of_date': row.get('as_of_date') or None,
            'values': values, 'metric_name': row.get('metric_name'),
            'count_semantics': row.get('count_semantics') or row.get('statistical_count_type'),
            'interpretation_note': 'daily_average' if re.search(r'\b(?:average|each day|per day|daily)\b', quote, re.I) else None,
            'source': _source(row, next(iter(values))),
        })
    return sorted(rows, key=lambda item: (str(item['as_of_date'] or item['period'] or ''), str(item['record_id'])), reverse=True)


def _label(field):
    return LABELS.get(field, field.replace("_", " "))


def build_task_result(package, manifest):
    """Build task answers from qualified observations; candidates remain leads."""
    task = manifest.get("task") or {}
    records = list(package.get("final_dataset") or []) + list(
        package.get("candidate_records") or []
    )
    requested = list(task.get("target_fields") or [])
    if requested:
        fields = [
            field
            for field in requested
            if field in COUNT_FIELDS
            or any((_value(row, field) is not None for row in records))
        ]
    else:
        fields = [
            field
            for field in COUNT_FIELDS
            if any((_value(row, field) is not None for row in records))
        ]
    fields = list(dict.fromkeys(fields))
    answers = {}
    for field in fields:
        exact = [
            row
            for row in package.get("final_dataset") or []
            if _exact_answer_eligible(row, task, field)
        ]
        by_value = {}
        for row in exact:
            by_value.setdefault(_value(row, field), []).append(_source(row, field))
        leads = {}
        for row in package.get("candidate_records") or []:
            lead = _candidate_lead(row, task, field)
            if lead:
                leads.setdefault((lead["value"], lead["qualifier"]), []).append(lead)
        grouped_leads = []
        for (candidate_value, qualifier), entries in leads.items():
            entries = sorted(entries, key=_date_rank, reverse=True)
            sources = []
            seen = set()
            for entry in entries:
                url = entry["source"]["url"]
                if url not in seen:
                    sources.append(entry["source"])
                    seen.add(url)
            grouped_leads.append(
                {
                    "value": candidate_value,
                    "qualifier": qualifier,
                    "sources": sources[:3],
                    "period": entries[0]["period"],
                    "as_of_date": entries[0]["as_of_date"],
                    "rank_date": (
                        _date_rank(entries[0]).isoformat()
                        if _date_rank(entries[0]) != date.min
                        else None
                    ),
                }
            )
        grouped_leads.sort(
            key=lambda item: (
                item["rank_date"] or "",
                len(item["sources"]),
                item["value"],
            ),
            reverse=True,
        )
        if len(by_value) == 1:
            value, sources = next(iter(by_value.items()))
            status = "confirmed"
        elif by_value:
            value, sources, status = (None, [], "conflict")
        else:
            value, sources, status = (None, [], "unconfirmed")
        answers[field] = {
            "status": status,
            "value": value,
            "sources": sources,
            "conflicting_values": (
                [
                    {"value": number, "sources": refs}
                    for number, refs in sorted(by_value.items())
                ]
                if status == "conflict"
                else []
            ),
            "candidate_leads": grouped_leads[:3] if status != "confirmed" else [],
        }
    confirmed = [
        field for field, answer in answers.items() if answer["status"] == "confirmed"
    ]
    if confirmed:
        headline = (
            "Confirmed: "
            + "; ".join(
                (f"{_label(field)} {answers[field]['value']:,}" for field in confirmed)
            )
            + "."
        )
    elif not answers:
        headline = "No numeric measure was available for this task; consult the data and sources."
    elif any((field in COUNT_FIELDS for field in answers)):
        headline = (
            "No total for the requested scope is confirmed by qualified evidence."
        )
    else:
        headline = (
            "No value for the requested scope is confirmed by qualified evidence."
        )
    if not confirmed:
        pending = []
        priority_fields = [
            field for field in ("cases_confirmed", "deaths") if field in answers
        ]
        priority_fields += [field for field in fields if field not in priority_fields]
        for field in priority_fields:
            leads = answers[field]["candidate_leads"]
            if leads:
                pending.append((field, leads[0]))
            if len(pending) == 2:
                break
        if pending:
            headline += (
                " Unverified leads (not task answers): "
                + "; ".join(
                    (
                        f"{_label(field)} {_format_lead(lead)}"
                        + (f" ({lead['as_of_date']})" if lead.get("as_of_date") else "")
                        for field, lead in pending
                    )
                )
                + "."
            )
    return {
        "schema_version": "task-result/1",
        "task": task,
        "coverage_status": manifest.get("coverage_status") or "not_assessed",
        "collection_status": manifest.get("collection_status") or "not_assessed",
        "answers": answers,
        "qualified_observations": _observations(package, task, fields),
        "headline": headline,
    }


def _source_link(source):
    url = str(source.get("url") or "")
    return (
        f"[source]({url})"
        if url.startswith(("http://", "https://"))
        else url or "source unavailable"
    )


def _clean_cell(value):
    return str(value or '—').replace('|', '\\|').replace('\r', ' ').replace('\n', ' ')[:240]


def _format_lead(lead):
    value = f"{lead['value']:,}"
    qualifier = lead.get("qualifier") or "exact"
    return {
        "over": f"over {value}",
        "at_least": f"at least {value}",
        "about": f"about {value}",
        "nearly": f"nearly {value}",
        "at_most": f"at most {value}",
        "under": f"under {value}",
    }.get(qualifier, value)


def render_task_result(result):
    task = result["task"]
    scope = f"{task.get('disease') or '?'} · {task.get('location') or '?'} · {task.get('start_date') or '?'} — {task.get('end_date') or '?'}"
    lines = [
        "# Collection task results",
        "",
        f"**Task:** {scope}",
        "",
        result["headline"],
        "",
        "## Requested measures",
        "",
        "| Measure | Supported result |",
        "| --- | --- |",
    ]
    for field, answer in result["answers"].items():
        label = _label(field)
        if answer["status"] == "confirmed":
            status = f"{answer['value']:,}" + " (supported by qualified evidence)"
        elif answer["status"] == "conflict":
            status = "Conflicting qualified totals; review required"
        else:
            status = "Unconfirmed; missing does not mean zero"
        lines.append(f"| {_clean_cell(label)} | {_clean_cell(status)} |")
    for field, answer in result["answers"].items():
        label = _label(field)
        if answer["status"] == "confirmed":
            lines += ["", f"### {label}: {answer['value']:,}", ""]
            for source in answer["sources"][:5]:
                lines.append(
                    f"- {_source_link(source)} — {_clean_cell(source.get('quote'))}"
                )
        elif answer["status"] == "conflict":
            lines += ["", f"### {label}: " + "conflicting totals", ""]
            for item in answer["conflicting_values"]:
                for source in item["sources"][:3]:
                    lines.append(
                        f"- {item['value']:,}: {_source_link(source)} — {_clean_cell(source.get('quote'))}"
                    )
        if answer["candidate_leads"]:
            lines += ["", f"### {label}: " + "unverified leads", ""]
            lines.append(
                "These figures did not pass evidence qualification and are not task answers."
            )
            for lead in answer["candidate_leads"]:
                when = (
                    lead.get("as_of_date") or lead.get("period") or "period unresolved"
                )
                refs = ", ".join((_source_link(source) for source in lead["sources"]))
                lines.append(f"- {_format_lead(lead)} ({_clean_cell(when)}): {refs}")
                if lead["sources"]:
                    lines.append("  - " + _clean_cell(lead["sources"][0].get("quote")))
    observations = result["qualified_observations"]
    lines += ["", "## Evidence-qualified observations", ""]
    if observations:
        lines += [
            "Each observation applies to its own area, period and measure; partial figures are not full-period totals.",
            "",
            "| "
            + "Measure and value | Area | Period or as of | Interpretation | Source"
            + " |",
            "| --- | --- | --- | --- | --- |",
        ]
        for item in observations[:30]:
            values = ", ".join(
                (
                    f"{(item['metric_name'] if field == 'metric_value' and item.get('metric_name') else _label(field))} {value:,}"
                    for field, value in item["values"].items()
                )
            )
            note = (
                "daily average"
                if item["interpretation_note"] == "daily_average"
                else item.get("count_semantics") or "—"
            )
            lines.append(
                f"| {_clean_cell(values)} | {_clean_cell(item['scope'])} | {_clean_cell(item['as_of_date'] or item['period'])} | {_clean_cell(note)} | {_source_link(item['source'])} |"
            )
        if len(observations) > 30:
            lines += [
                "",
                f"{len(observations) - 30} more qualified observations are in collection/final_dataset.csv.",
            ]
    else:
        lines.append("No qualified numeric observation matches this task.")
    if (
        result["coverage_status"] != "complete"
        or result["collection_status"] == "partial"
    ):
        lines += [
            "",
            "**Coverage:** Collection or coverage is incomplete; results reflect only processed, qualified material.",
        ]
    return "\n".join(lines) + "\n"


def write_task_result_artifacts(result, output_dir):
    from .export import write_json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "task_result.md"
    report_path.write_text(render_task_result(result), encoding="utf-8")
    return {
        "task_result_json": str(write_json(result, out / "task_result.json")),
        "task_result_english": str(report_path),
    }
