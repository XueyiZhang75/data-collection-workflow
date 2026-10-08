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
CUMULATIVE_SEMANTICS = {'cumulative', 'cumulative count', 'cumulative total', 'total', 'outbreak total'}
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
            if isinstance(item, dict) and item.get('document_hash') and item.get('locator')
            and item.get('supported') is not False}


def _source(row, field):
    qualification = row.get('evidence_qualification') or {}
    entries = [entry for entry in qualification.get('field_evidence') or []
               if isinstance(entry, dict) and entry.get('field') == field
               and entry.get('document_hash') and entry.get('locator')
               and entry.get('supported') is not False]
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


def _answer_eligible(row, task, field, *, full_period):
    if (row.get('evidence_qualification') or {}).get('status') != 'qualified':
        return False
    if not _task_measure_field(field):
        return False
    value = _value(row, field)
    if value is None or not _disease_matches(row, task):
        return False
    if not _location_matches(row, task, exact=True):
        return False
    if full_period and not _full_period(row, task):
        return False
    supported = _evidence_fields(row)
    if field not in supported or not ({'country', 'geographic_scope', 'subnational_location'} & supported):
        return False
    if full_period and not ('reporting_period' in supported or {'metric_period_start', 'metric_period_end'} <= supported):
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
        checked_semantics = semantics if full_period else re.sub(r'\b(?:weekly|monthly|daily)\b', '', semantics)
        if NON_TOTAL_WORDS.search(checked_semantics) or semantics in {'unknown', 'unspecified'}:
            return False
        if full_period and semantics in {'new', 'incremental', 'incident'}:
            return False
        population = _norm(row.get('population_scope'))
        if population and population not in {'all', 'all cases', 'all reported cases',
                                             'all confirmed cases', 'general population',
                                             'national population', 'whole population'}:
            return False
        quote = _source(row, field)['quote']
        match = _number_match(quote, value)
        if not match:
            return False
        if full_period and _qualifier(quote, match) != 'exact':
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


def _exact_answer_eligible(row, task, field):
    return _answer_eligible(row, task, field, full_period=True)


def _iso_date(value):
    try:
        return date.fromisoformat(str(value or ''))
    except ValueError:
        return None


def _period_bounds(value):
    """Return comparison bounds alongside source-precision display labels."""
    from .evidence_qualification import _period_label_bounds
    from .source_assertions import _DATE, _date_parts, _fold, date_intervals

    token = str(value or '').strip()
    lower, upper = _period_label_bounds(token)
    if lower is None or upper is None:
        return None
    labels = [lower.isoformat(), upper.isoformat()]
    if re.fullmatch(r'\d{4}-\d{2}', token):
        labels = [token, token]
    else:
        intervals = date_intervals(token)
        parts = intervals[0] if len(intervals) == 1 else None
        if parts is None and _DATE.fullmatch(_fold(token)):
            component = _date_parts(_fold(token))
            parts = (component, component)
        if parts:
            labels = [f'{part[0]:04d}-{part[1]:02d}' + (f'-{part[2]:02d}' if part[2] else '')
                      for part in parts]
    return lower, upper, labels[0], labels[1]


def _supported_window(row, task, *, allow_outside=False):
    """Use observation dates, never the publication date, to locate a result."""
    fields = _evidence_fields(row)
    start_bounds = _period_bounds(row.get('metric_period_start')) if 'metric_period_start' in fields else None
    end_bounds = _period_bounds(row.get('metric_period_end')) if 'metric_period_end' in fields else None
    start, start_label = (start_bounds[0], start_bounds[2]) if start_bounds else (None, None)
    end, end_label = (end_bounds[1], end_bounds[3]) if end_bounds else (None, None)
    if 'reporting_period' in fields and (start is None or end is None):
        bounds = _period_bounds(row.get('reporting_period'))
        if bounds:
            start, end, start_label, end_label = bounds
    as_of_bounds = _period_bounds(row.get('as_of_date')) if 'as_of_date' in fields else None
    as_of, as_of_label = (as_of_bounds[1], as_of_bounds[3]) if as_of_bounds else (None, None)
    if as_of and (end is None or as_of < end):
        end, end_label = as_of, as_of_label
    task_start, task_end = _iso_date(task.get('start_date')), _iso_date(task.get('end_date'))
    if not task_start or not task_end or not end:
        return None
    if not allow_outside and not task_start <= end <= task_end:
        return None
    if start and (start > end or (not allow_outside and start < task_start)):
        return None
    return {'period_start': start_label, 'period_end': end_label, 'as_of_date': as_of_label}


def _verified_closure_anchor(row, task, field):
    """A closure event date is separate from the count's statistical cutoff."""
    value = str(row.get('outbreak_closure_date') or '')
    closure_date = _iso_date(value)
    start, end = _iso_date(task.get('start_date')), _iso_date(task.get('end_date'))
    if not closure_date or not start or not end or not start <= closure_date <= end:
        return None
    entries = (row.get('evidence_qualification') or {}).get('field_evidence') or []
    numeric = [entry for entry in entries if isinstance(entry, dict) and entry.get('field') == field
               and entry.get('supported') is True and entry.get('value') == _value(row, field)
               and entry.get('document_hash') and (entry.get('locator') or {}).get('chunk_id')]
    for entry in entries:
        if (not isinstance(entry, dict) or entry.get('field') != 'outbreak_closure_date'
                or entry.get('supported') is not True or entry.get('value') != value):
            continue
        locator = entry.get('locator') or {}
        if not any(item['document_hash'] == entry.get('document_hash')
                   and item['locator']['chunk_id'] == locator.get('chunk_id') for item in numeric):
            continue
        quote = str(entry.get('quote') or '').strip()
        if not quote or len(quote) > 900:
            continue
        return {'record_id': row.get('record_id'), 'url': row.get('source_url'),
                'quote': quote, 'document_hash': entry['document_hash'], 'locator': locator,
                'value': value, 'date_role': 'outbreak_closure_event',
                'derivation': entry.get('derivation') or row.get('outbreak_closure_derivation') or None}
    return None


def _closure_sources(row, field, task):
    """Only source-local closure evidence linked to this numeric observation."""
    source = _source(row, field)
    sources = [source]
    anchor = _verified_closure_anchor(row, task, field)
    if anchor:
        sources.append(anchor)
    numeric_chunk = (source.get('locator') or {}).get('chunk_id')
    for entry in (row.get('evidence_qualification') or {}).get('closure_evidence') or []:
        if not isinstance(entry, dict) or entry.get('supported') is not True:
            continue
        locator = entry.get('locator') or {}
        if (not numeric_chunk or locator.get('chunk_id') != numeric_chunk
                or entry.get('document_hash') != source.get('document_hash')):
            continue
        quote = str(entry.get('quote') or '').strip()
        if quote and len(quote) <= 900:
            sources.append({**source, 'quote': quote, 'locator': locator})
    return sources


def _completed_outbreak_sources(row, task, field, window):
    from .source_assertions import _DATE, sentence_spans

    anchor = _verified_closure_anchor(row, task, field)
    if field not in COUNT_FIELDS or (not window['period_start'] and not anchor):
        return []
    semantics = _norm(row.get('count_semantics') or row.get('statistical_count_type'))
    if semantics not in CUMULATIVE_SEMANTICS:
        return []
    count_source = _source(row, field)
    count_quote = count_source['quote']
    # A closure declaration does not make a current-week or unrelated count a final total.
    if not re.search(r'\boutbreak\b|\bsince\s+(?:the\s+)?first\b', count_quote, re.I):
        return []
    if re.search(r'\b(?:new|another|separate|previous|earlier)\s+outbreak\b', count_quote, re.I):
        return []

    def authority(text):
        return (re.search(r'\b(?:ministry of health|health ministry|health minister|minister of health|'
                          r'department of health|health department|public health agency|'
                          r'health authorities|world health organization)\b', text, re.I)
                or re.search(r'\bWHO\b', text))

    def uncertain(text):
        text = re.sub(r'\bMay(?=\s+\d{1,4}\b)|(?<=in )May\b|(?<=during )May\b', '', text, flags=re.I)
        return re.search(r"\b(?:not|never|denied|denies|denying|if|may|might|could|would|will|"
                         r"expects?|expected|plans?|planned|premature|falsely|rumou?r|hasn't|hadn't|didn't|"
                         r"exercise|scenario|simulation|hypothetical)\b", text, re.I)

    def official_actor(text):
        match = authority(text)
        if not match:
            return False
        # Between the authority and its verb allow names, titles and auxiliaries,
        # not a different subject or an intervening reporting predicate.
        return bool(re.fullmatch(r"(?:\s|[,()]|[A-Z][\w'-]*\.?|has|have|had|of|in|for|"
                                 r"officially|formally|publicly|now|today)*", text[match.end():]))

    def direct_authority(text):
        match = authority(text)
        if not match:
            return False
        location = _norm(task.get('location'))
        return _norm(text[:match.start()]) in {'', 'the', location, location + ' s', 'the country s'}

    sources = _closure_sources(row, field, task)
    for source in sources:
        quote = source['quote']
        if len(quote) > 900:
            continue
        sentences = []
        for left, right in sentence_spans(quote):
            sentence = quote[left:right]
            if sentences and re.search(r'\b(?:Dr|Mr|Mrs|Ms|Prof)\.$', sentences[-1]):
                sentences[-1] += ' ' + sentence
            else:
                sentences.append(sentence)
        for index, sentence in enumerate(sentences):
            years = re.findall(r'\b(?:19|20)\d{2}\b', sentence)
            lower_year = (window['period_start'] or anchor['value'])[:4]
            upper_year = (window['period_end'] or anchor['value'])[:4]
            if any(not lower_year <= year <= upper_year for year in years):
                continue
            dates = [_period_bounds(match.group()) for match in _DATE.finditer(sentence)]
            start_bound = _period_bounds(window['period_start'])[0] if window['period_start'] else None
            if start_bound and any(bounds and bounds[1] < start_bound for bounds in dates):
                continue
            attribution = sentences[index + 1] if index + 1 < len(sentences) else ''
            linked_attribution = bool(re.match(r'\s*(?:The|This) declaration\b', attribution, re.I))
            by = re.search(r'\b(?:announced|issued|made|confirmed|declared)\s+by\b', attribution, re.I)
            announced_by = bool(by and direct_authority(attribution[by.end():]))
            statement_by = any(official_actor(attribution[:verb.start()]) for verb in
                               re.finditer(r'\b(?:said|told|announced|confirmed)\b', attribution, re.I))
            if uncertain(sentence) or (linked_attribution and uncertain(attribution)):
                continue
            for declaration in re.finditer(r'\b(?:declared|announced|confirmed)\b', sentence, re.I):
                before = sentence[:declaration.start()]
                after = re.split(r';|\b(?:while|whereas|but|and)\b', sentence[declaration.end():], flags=re.I)[0]
                actor = re.split(r';|\b(?:while|whereas|but|and)\b', before, flags=re.I)[-1]
                passive = bool(re.search(r'\b(?:was|is|has been|had been)\s*$', actor, re.I))
                event = before if passive else after
                normalized_event = _norm(event)
                names = [_norm(name) for name in [task.get('disease'), *(task.get('disease_aliases') or [])] if _norm(name)]
                disease = '(?:' + '|'.join(re.escape(name) for name in names) + ')'
                outbreak = '(?:' + disease + r'\s+outbreak|outbreak\s+of\s+' + disease + ')'
                if not names or not re.search(r'\b' + outbreak + r'\b', normalized_event):
                    continue
                location = _norm(task.get('location'))
                location_bound = bool(location and (
                    re.search(outbreak + r'\s+in\s+' + re.escape(location) + r'\b', normalized_event)
                    or re.search(re.escape(location) + r'\s+s\s+' + outbreak + r'\b', normalized_event)))
                if (not location_bound and re.search(outbreak + r'\s+in (?:the|this) country\b', normalized_event)
                        and re.match(re.escape(location) + r'\b', _norm(actor))):
                    location_bound = True
                if not location_bound or not re.search(r'\boutbreak\b', event, re.I):
                    continue
                normalized_after = _norm(after)
                ending = r'(?:over|ended|closed)(?=$|\s+(?:on|in|after|following|with|as|by|at)\b)'
                end_of = re.search(r'\bend\s+(?:of|to)\s+(?:(?:a|an|the)\s+)?' + outbreak, normalized_after)
                event_phrase = ('(?:' + outbreak + r'\s+in\s+(?:' + re.escape(location) + r'|the country|this country)'
                                + '|' + re.escape(location) + r'\s+s\s+' + outbreak + ')')
                ended = (re.match(ending, normalized_after) if passive else re.search(
                    event_phrase + r'\s+(?:(?:was|is|has|had|been|officially)\s+)*' + ending, normalized_after))
                if not (end_of or ended):
                    continue
                passive_by = re.search(r'\bby\b', after, re.I)
                attributed = official_actor(actor) if not passive else (passive_by and direct_authority(after[passive_by.end():]))
                if not attributed and not (linked_attribution and (announced_by or statement_by) and authority(attribution)):
                    continue
                ordinal = r'\b(first|second|third|fourth|fifth|\d+(?:st|nd|rd|th))\s+(?:\w+\s+){0,2}outbreak\b'
                count_episode = re.search(ordinal, count_quote, re.I)
                closure_episode = re.search(ordinal, event, re.I)
                if count_episode and (not closure_episode or count_episode.group(1).lower() != closure_episode.group(1).lower()):
                    continue
                context = {'closure_statement_quote': sentence,
                           'closure_attribution_quote': attribution if linked_attribution else None}
                refs = [count_source] if source == count_source else [count_source, source]
                return [{**ref, **context} for ref in refs]
    return []


RESULT_KIND_LABELS = {
    'full_period_total': 'Full-period total',
    'completed_outbreak_total': 'Completed outbreak total',
    'latest_available': 'Latest available cumulative count',
    'reported_period': 'Reported-period value',
}


def _supported_results(records, task, field):
    results = []
    for row in records:
        if not _answer_eligible(row, task, field, full_period=False):
            continue
        anchor = _verified_closure_anchor(row, task, field)
        window = _supported_window(row, task)
        event_basis = window is None and anchor is not None
        if event_basis:
            window = _supported_window(row, task, allow_outside=True) or {
                'period_start': None, 'period_end': None, 'as_of_date': None}
        if window is None:
            continue
        source = _source(row, field)
        match = _number_match(source['quote'], _value(row, field))
        if not match:
            continue
        qualifier = _qualifier(source['quote'], match)
        closure_sources = _completed_outbreak_sources(row, task, field, window)
        if event_basis and not closure_sources:
            continue
        semantics = _norm(row.get('count_semantics') or row.get('statistical_count_type'))
        if _full_period(row, task) and semantics not in {'new', 'incremental', 'incident'}:
            kind = 'full_period_total'
            note = 'The evidence covers the requested reporting period.'
        elif closure_sources:
            kind = 'completed_outbreak_total'
            note = ('This is the final count for the documented outbreak; it does not establish zero cases outside this outbreak '
                    'or a complete calendar-period total. Separate outbreaks are not added together.')
            if event_basis:
                note += (' The closure event falls within the query window, but the count may include cases before the query window. '
                         'The closure date does not supply a statistical cutoff or an unknown start year.')
        elif field in COUNT_FIELDS and semantics in CUMULATIVE_SEMANTICS:
            kind = 'latest_available'
            note = ('Cumulative count for the stated period or as-of date, among processed qualified evidence; '
                    'not a full-period total. Uncovered dates are not zero and overlapping counts are not added.')
        else:
            kind = 'reported_period'
            note = 'This value applies only to the stated reporting period; it is not a cumulative or full-period total.'
        results.append({'result_kind': kind, 'value': _value(row, field), 'qualifier': qualifier,
                        'record_id': row.get('record_id'), 'recovered_from_record_id': row.get('recovered_from_record_id'),
                        'scope': row.get('geographic_scope') or row.get('country'),
                        'period': row.get('reporting_period') or None, **window,
                        'temporal_basis': 'outbreak_closure_event' if event_basis else 'observation_period',
                        'closure_date': anchor['value'] if anchor and closure_sources else None,
                        'closure_date_source': anchor if closure_sources else None,
                        'sources': closure_sources or [source], 'boundary_note': note})
    results.sort(key=lambda item: (item['result_kind'] == 'full_period_total',
                                 _period_bounds(item['closure_date'] if item['temporal_basis'] == 'outbreak_closure_event'
                                                else item['period_end'])[1],
                                 _period_bounds(item['period_start'])[0] if item['period_start'] else date.min,
                                 item['result_kind'] == 'completed_outbreak_total'), reverse=True)
    selected = results[0] if results else None
    comparable = [item for item in results if selected and all(item[key] == selected[key] for key in
                  ('result_kind', 'period_start', 'period_end', 'scope', 'closure_date'))]
    conflict = len({(item['value'], item['qualifier']) for item in comparable}) > 1
    return results, None if conflict else selected, conflict


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
        first_field = next(iter(values))
        closure_anchor = _verified_closure_anchor(row, task, first_field)
        window = _supported_window(row, task, allow_outside=True) or {
            'period_start': None, 'period_end': None, 'as_of_date': None}
        if closure_anchor and not _completed_outbreak_sources(row, task, first_field, window):
            closure_anchor = None
        quote = str(row.get('evidence_quote') or '')
        rows.append({
            'record_id': row.get('record_id'), 'scope': row.get('geographic_scope') or row.get('country'),
            'period': row.get('reporting_period') or None, 'as_of_date': row.get('as_of_date') or None,
            'values': values, 'metric_name': row.get('metric_name'),
            'count_semantics': row.get('count_semantics') or row.get('statistical_count_type'),
            'closure_date': closure_anchor['value'] if closure_anchor else None,
            'closure_date_derived': bool(closure_anchor and closure_anchor.get('derivation')),
            'interpretation_note': 'daily_average' if re.search(r'\b(?:average|each day|per day|daily)\b', quote, re.I) else None,
            'source': _source(row, first_field),
        })
    return sorted(rows, key=lambda item: (str(item['closure_date'] or item['as_of_date'] or item['period'] or ''), str(item['record_id'])), reverse=True)


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
        supported_results, selected_result, scoped_conflict = _supported_results(
            package.get('final_dataset') or [], task, field,
        )
        for lead in grouped_leads:
            superseded = []
            for source in lead['sources']:
                matches = [item['record_id'] for item in supported_results
                           if item['recovered_from_record_id'] == source['record_id']
                           and item['value'] == lead['value'] and item['qualifier'] == lead['qualifier']
                           and any(ref['url'] == source['url'] for ref in item['sources'])]
                if not matches:
                    break
                superseded.extend(matches)
            else:
                if superseded:
                    lead['superseded_by_qualified_record_ids'] = sorted(set(superseded))
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
            "supported_results": supported_results,
            "selected_result": selected_result,
            "scoped_result_conflict": scoped_conflict,
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
    scoped = [(field, answer['selected_result']) for field, answer in answers.items()
              if answer['status'] != 'confirmed' and answer['selected_result']]
    if scoped:
        scoped_headline = '; '.join(
            f"{RESULT_KIND_LABELS[item['result_kind']]}: {_label(field)} {_format_lead(item)}"
            f" ({_result_time(item)})"
            for field, item in scoped
        ) + '.'
        headline = headline + ' ' + scoped_headline if confirmed else scoped_headline
    if not confirmed:
        pending = []
        priority_fields = [
            field for field in ("cases_confirmed", "deaths") if field in answers
        ]
        priority_fields += [field for field in fields if field not in priority_fields]
        for field in priority_fields:
            leads = [lead for lead in answers[field]['candidate_leads']
                     if not lead.get('superseded_by_qualified_record_ids')]
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


def _result_time(item):
    if item.get('period_end'):
        period = f"{item['period_start'] or 'start unresolved'} to {item['period_end']}"
    else:
        period = 'statistical period unresolved'
    if item.get('closure_date'):
        derived = ' (closure date derived from source)' if (item.get('closure_date_source') or {}).get('derivation') else ''
        return f"outbreak closed {item['closure_date']}{derived}; {period}"
    return period


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
        elif answer.get('selected_result'):
            selected = answer['selected_result']
            status = (f"{_format_lead(selected)} — {RESULT_KIND_LABELS[selected['result_kind']]} "
                      f"({_result_time(selected)})")
        elif answer.get('scoped_result_conflict'):
            status = "Conflicting qualified values for the same reporting period; review required"
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
        scoped_results = answer.get('supported_results') or []
        if answer['status'] != 'confirmed' and scoped_results:
            lines += ['', f'### {label}: evidence-supported scoped results', '']
            for item in scoped_results[:10]:
                lines.append(f"- **{_format_lead(item)} — {RESULT_KIND_LABELS[item['result_kind']]}** "
                             f"({_result_time(item)}). "
                             + item['boundary_note'])
                date_source = item.get('closure_date_source') or {}
                derivation = date_source.get('derivation') or {}
                if derivation:
                    lines.append(f"  - Closure date derived from {derivation.get('weekday') or 'the relative date in the source'} "
                                 f"and this source's publication date {derivation.get('publication_date') or 'unresolved'} "
                                 f"({_source_link(date_source)}); this is a source-specific event-date inference, not a statistical cutoff.")
                for source in item['sources']:
                    lines.append(f"  - {_source_link(source)} — {_clean_cell(source.get('quote'))}")
            if len(scoped_results) > 10:
                lines.append(f"{len(scoped_results) - 10} more supported results are in task_result.json.")
        visible_leads = [lead for lead in answer['candidate_leads']
                         if not lead.get('superseded_by_qualified_record_ids')]
        if visible_leads:
            lines += ["", f"### {label}: " + "unverified leads", ""]
            lines.append(
                "These figures did not pass evidence qualification and are not task answers."
            )
            for lead in visible_leads:
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
            when = item['as_of_date'] or item['period']
            if item.get('closure_date'):
                derived = ' (derived from source)' if item.get('closure_date_derived') else ''
                when = f"outbreak closed {item['closure_date']}{derived}; " + (when or 'statistical period unresolved')
            lines.append(
                f"| {_clean_cell(values)} | {_clean_cell(item['scope'])} | {_clean_cell(when)} | {_clean_cell(note)} | {_source_link(item['source'])} |"
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
    return {
        "task_result_json": str(write_json(result, out / "task_result.json")),
    }
