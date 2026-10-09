"""Temporal scope consistency for already source-bound simple table rows."""
from __future__ import annotations

import re

from .source_assertions import date_intervals, number_value, observation_dates


_NUMERIC = {'cases_confirmed', 'cases_probable', 'cases_suspected', 'cases_unspecified',
            'deaths', 'hospitalizations', 'tests_positive', 'tests_total', 'metric_value'}
_TEMPORAL = {'reporting_period', 'metric_period_start', 'metric_period_end',
             'period_start_date', 'period_end_date', 'as_of_date', 'date_reported', 'report_date'}
_SEMANTICS = {'count_semantics', 'statistical_count_type'}
_WEEK = re.compile(r'\bweek\s+(\d{1,2})\b', re.I)


def _key(value):
    return re.sub(r'\s+', ' ', str(value or '')).strip().casefold()


def _metric_column(label, field):
    """Exclude axis cells and competing units before using an equal number."""
    if re.search(r'\b(?:week|date|year|age)\b', label, re.I) and not re.search(
            r'\b(?:cases?|deaths?|tests?|hospitali[sz]ations?)\b', label, re.I):
        return False
    if field.startswith('cases_') and re.search(r'\b(?:deaths?|tests?|hospitali[sz]ations?)\b', label, re.I):
        return False
    if field == 'deaths' and re.search(r'\bcases?\b', label, re.I):
        return False
    kind = re.search(r'\b(confirmed|probable|suspected)\b', label, re.I)
    if kind and field in {'cases_confirmed', 'cases_probable', 'cases_suspected'}:
        return field == 'cases_' + kind[1].lower()
    return True


def _period_support(name, value, week, scope):
    if name in {'as_of_date', 'date_reported', 'report_date'}:
        # An explicit interval is not an observation cutoff or report date.
        return False
    dates = observation_dates(scope)
    if name != 'reporting_period':
        field = {'period_start_date': 'metric_period_start', 'period_end_date': 'metric_period_end'}.get(name, name)
        return bool(dates.get(field) and str(value) == dates[field])
    token = str(value).strip()
    stated_weeks = {int(match) for match in _WEEK.findall(token)}
    if stated_weeks and stated_weeks != {week}:
        return False
    requested = date_intervals(token)
    if requested:
        return bool(date_intervals(scope) and all(period in date_intervals(scope) for period in requested))
    if week is not None:
        # A week number alone never supplies a year or calendar convention.
        return bool(re.fullmatch(r'week\s+0*' + str(week), token, re.I))
    # A selected annual column may state its year despite a weekly report title.
    return bool(re.fullmatch(r'\d{4}', token) and re.search(r'(?<!\d)' + re.escape(token) + r'(?!\d)', scope))


def table_observation_scope_support(name, value, text, record):
    """Check a selected numeric cell's temporal scope, not its overall validity.

    ``text`` must already be resolved through source-local provenance checks.
    True for a numeric field means only that its supplied temporal fields agree;
    callers must still validate the number, metric, disease and geography.
    """
    if name not in _NUMERIC | _TEMPORAL | _SEMANTICS:
        return None
    lines = str(text).splitlines()
    table_lines = [line for line in lines if '|' in line]
    if len(table_lines) != 2:
        return None
    labels, cells = ([cell.strip() for cell in line.strip().strip('|').split('|')] for line in table_lines)
    if len(labels) != len(cells) or not labels:
        return None
    context = [line for line in lines[:lines.index(table_lines[0])] if '|' not in line]
    if name not in _SEMANTICS and not any(re.search(r'\bweek\b', line, re.I) for line in [*context, *labels]):
        return None
    values = [(name, value)] if name in _NUMERIC else [
        (field, record[field]) for field in _NUMERIC if record.get(field) is not None]
    if not values:
        return None
    if not any(number_value(cell) is not None and any(_metric_column(label, field) for field, _ in values)
               for label, cell in zip(labels, cells)):
        # Key/value rows and prose cells carry their own explicit assertions;
        # this helper only binds scalar numeric cells to column headings.
        return None
    selected = set()
    for field, number in values:
        matches = [index for index, (label, cell) in enumerate(zip(labels, cells))
                   if number_value(cell) == number and _metric_column(label, field)]
        if len(matches) != 1:
            return False
        selected.add(matches[0])
    if len(selected) != 1:
        return False
    index = selected.pop()
    claimed_label = record.get('source_column_label')
    if claimed_label and _key(claimed_label) != _key(labels[index]):
        return False
    if name in _SEMANTICS:
        # Use only this numerator's verified header, never a neighboring total
        # or a model-provided label. A generic total does not state its interval.
        patterns = {
            'newly_reported': r'\b(?:new|newly\s+reported)\s+(?:(?:confirmed|probable|suspected)\s+)?(?:cases?|deaths?|tests?|hospitali[sz]ations?)\b',
            'cumulative': r'\bcumulative\s+(?:(?:confirmed|probable|suspected)\s+)?(?:cases?|deaths?|tests?|hospitali[sz]ations?)\b',
            'annual': r'\b(?:annual|full[ -]year)\b',
        }
        pattern = patterns.get(str(value))
        if pattern is None:
            return None
        header = labels[index]
        if re.search(r'\b(?:not|non)[ -]+(?:new|cumulative|annual)\b', header, re.I):
            return False
        return bool(re.search(pattern, header, re.I))

    row_week = None
    if re.search(r'\bweek\b', labels[0], re.I):
        match = re.fullmatch(r'(?:week\s+)?(\d{1,2})(?:\s*\(.*\))?', cells[0], re.I)
        if match:
            row_week = int(match[1])
    column_weeks = {int(match) for match in _WEEK.findall(labels[index])}
    if len(column_weeks) > 1:
        return False
    week = row_week if row_week is not None else next(iter(column_weeks), None)
    if week is not None and not 1 <= week <= 53:
        return False
    scope = cells[0] if row_week is not None else labels[index]
    if row_week is None and week is not None:
        # Only the matching observation column can bind its named report week.
        matching = [line for line in context if {int(match) for match in _WEEK.findall(line)} == {week}]
        intervals = {tuple(date_intervals(line)) for line in matching if date_intervals(line)}
        if len(intervals) == 1:
            scope += '\n' + '\n'.join(matching)
    if name in _TEMPORAL:
        return _period_support(name, value, week, scope)
    return all(_period_support(field, record[field], week, scope)
               for field in _TEMPORAL if record.get(field) not in (None, ''))
