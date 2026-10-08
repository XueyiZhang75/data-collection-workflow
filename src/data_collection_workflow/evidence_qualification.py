"""Fail-closed, source-local qualification for the opt-in universal workflow.

Task contracts constrain observations; they never supply missing evidence.
"""
from __future__ import annotations

from data_collection_workflow.environment import get_env

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from datetime import date

from .disease_identity import disease_names, same_disease_name
from .source_assertions import analysis_projection, normalized_quote_spans, number_pattern, number_value, numeric_span_is_complete, case_count_object_role, case_count_label_role, NUMBER_TOKEN, bounded_count, contextual_number_role, typed_date_support, annual_period_support
from .source_assertions import sentence_spans, sentence_bounds, publication_dates


class ProductKind(str, Enum):
    CASE_LEVEL = 'case_level'
    AGGREGATE = 'aggregate'
    CONTEXT = 'context'
    UNRESOLVED = 'unresolved'


@dataclass(frozen=True)
class FieldEvidence:
    field: str
    value: object
    document_hash: str
    locator: dict
    quote: str
    supported: bool
    reason: str = ''

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class RecordQualification:
    record_id: str
    status: str
    product_kind: ProductKind
    reasons: list[str] = field(default_factory=list)
    field_evidence: list[FieldEvidence] = field(default_factory=list)

    def to_dict(self):
        result = asdict(self)
        result['product_kind'] = self.product_kind.value
        return result


def evidence_qualification_enabled():
    return get_env('PIPELINE_MODE', '').strip() == 'evidence'


def build_evidence_index(state):
    chunks = state.get('evidence_chunks') or []
    return {'documents': list(state.get('documents') or []),
            'evidence_chunks': {str(c.get('chunk_id') or c.get('evidence_span_id')): c for c in chunks}}


_NUMERIC = ('cases_confirmed','cases_probable','cases_suspected','cases_unspecified','deaths','hospitalizations','icu_admissions','tests_positive','tests_total','positivity_rate','incidence_rate','cumulative_count','new_count','metric_value')
_DATES = ('reporting_period','as_of_date','event_start_date','event_end_date','period_start_date','period_end_date','metric_period_start','metric_period_end','onset_date','confirmation_date','date_onset','date_confirmation','date_death','date_reported','report_date','ship_board_date','ship_disembark_date')
_SUBSTANTIVE = ('virus_or_syndrome','age','gender','nationality','outcome','symptoms','hospitalized','intensive_care','isolated','occupation_or_role','cruise_crew','cruise_passenger_guest','contact_with_case','contact_setting','travel_from','travel_to','confirmation_method','accession_id','population_scope','associated_countries','travel_or_vessel_context','metric_denominator','case_definition','disease_standard_name','pathogen_or_syndrome','metric_name','metric_unit','metric_category','metric_period_label','count_unit','count_semantics','unit','count_basis','statistical_count_type','value_kind','denominator_scope','population_denominator','scope','geographic_scope_type','patient_links','disjoint_partition')
_GEO = ('country','subnational_location','locality','geographic_scope')


def _present(value):
    return value is not None and value != '' and value != [] and value != {}


def _contains(text, value):
    return bool(re.search(r'(?<!\w)' + re.escape(str(value).strip()) + r'(?!\w)', text, re.I))


def _provenance(record):
    value = record.get('field_provenance_json') or {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def _local_span(record, entry, index):
    chunks = index.get('evidence_chunks') or index.get('spans') or {}
    if isinstance(chunks, list):
        chunks = {str(c.get('chunk_id') or c.get('evidence_span_id')): c for c in chunks}
    chunk_id = entry.get('chunk_id') or entry.get('evidence_span_id') or record.get('chunk_id') or record.get('evidence_chunk_id') or record.get('supporting_chunk_id')
    chunk = chunks.get(str(chunk_id)) or {}
    source = chunk.get('source_id') or record.get('source_id')
    documents = index.get('documents') or []
    if isinstance(documents, dict):
        documents = list(documents.values())
    matches = [d for d in documents if d.get('source_id') == source and d.get('content_hash')]
    requested_hash = entry.get('document_hash') or chunk.get('document_hash') or chunk.get('content_hash')
    if requested_hash:
        matches = [d for d in matches if d.get('content_hash') == requested_hash]
    requested_id = entry.get('document_id') or chunk.get('document_id')
    if requested_id:
        matches = [d for d in matches if d.get('document_id') == requested_id]
    if len(matches) != 1 or not chunk:
        return '', '', {}, 'missing_or_ambiguous_document_locator'
    doc = matches[0]
    if ('low_ocr_confidence_candidate' in (doc.get('quality_issues') or []) and
            not any('quality_issues' in span for span in doc.get('locator_spans') or [])):
        return '', doc['content_hash'], {}, 'low_ocr_confidence_candidate'
    text = str(doc.get('clean_text') or '')
    expected_hash = str(doc.get('text_hash') or doc.get('content_hash') or '').removeprefix('sha256:')
    if hashlib.sha256(text.encode('utf-8')).hexdigest() != expected_hash:
        return '', '', {}, 'document_hash_mismatch'
    quote = str(next((entry[key] for key in ('quote','evidence_quote','supporting_quote') if key in entry),
                     record.get('case_span_quote') or chunk.get('row_quote') or chunk.get('text') or '') or '')
    # Real evidence chunks preserve offsets; a field quote cannot borrow a sibling row.
    if not quote or len(quote) > 900:
        return '', doc['content_hash'], {}, 'unresolvable_or_nonlocal_span'
    contexts = chunk.get('bound_context_spans') or []
    from .evidence_chunking import validate_bound_context, quote_within_chunk, combine_bound_quote, source_unit_for_quote
    if contexts and not validate_bound_context(doc, chunk, contexts):
        return '', doc['content_hash'], {}, 'invalid_bound_context'
    spans = [(match.start(), match.end()) for match in re.finditer(re.escape(quote), text)]
    if not spans:
        spans = normalized_quote_spans(text, quote)
    if isinstance(chunk.get('char_start'), int):
        spans = [(left,right) for left,right in spans if quote_within_chunk(doc, chunk, left, right, contexts)]
    if len(spans) != 1:
        expected = entry.get('char_start', chunk.get('char_start'))
        located = [(left,right) for left,right in spans if left == expected]
        if len(located) == 1:
            spans = located
        else:
            return '', doc['content_hash'], {}, 'unresolvable_or_nonlocal_span' if not spans else 'ambiguous_span'
    start, end = spans[0]
    quote = text[start:end]
    if any(key in entry for key in ('quote','evidence_quote','supporting_quote')):
        unit_start, unit_end = source_unit_for_quote(doc,chunk,start,start+len(quote))
        unit_quote = text[unit_start:unit_end]
        if len(combine_bound_quote(contexts,unit_quote)) <= 900:
            start,quote = unit_start,unit_quote
    resolved_length = len(quote)
    evidence_ranges = [(start,start+resolved_length)] + [(s['char_start'],s['char_end']) for s in contexts]
    if any('low_ocr_confidence_candidate' in (span.get('quality_issues') or []) and
           any(left < span['char_end'] and span['char_start'] < right for left,right in evidence_ranges)
           for span in doc.get('locator_spans') or [] if isinstance(span.get('char_start'),int) and isinstance(span.get('char_end'),int)):
        return '', doc['content_hash'], {}, 'low_ocr_confidence_candidate'
    locator = {'chunk_id':chunk_id,'document_id':doc.get('document_id'),'parser_version':doc.get('parser_version'),
               'text_hash':doc.get('text_hash'),'char_start':start,'char_end':start+resolved_length,
               'table_id':chunk.get('table_id'),'row_id':chunk.get('row_id')}
    if chunk.get('structured_data_kind'):
        locator.update(json_pointer=chunk.get('json_pointer'),structured_data_kind=chunk['structured_data_kind'])
    if contexts:
        # Contexts are exact structural source spans, even when a title is distant.
        quote = combine_bound_quote(contexts,quote)
        locator['bound_context_spans'] = contexts
    elif chunk.get('table_id') and chunk.get('row_id'):
        surrounding = text[max(0,start-900):start+len(quote)]
        headers = [str(chunk.get(k) or '') for k in ('heading_context','table_header')]
        if all(not h or h in surrounding for h in headers):
            quote = '\n'.join([h for h in headers if h] + [quote])
    if len(quote) > 900:
        return '', doc['content_hash'], {}, 'unresolvable_or_nonlocal_span'
    return quote, doc['content_hash'], locator, ''




_CLINICAL_LABELS = {
    'age': r'age(?:d)?', 'gender':r'(?:gender|sex)', 'nationality':r'nationality',
    'outcome':r'outcome', 'symptoms':r'symptoms?',
    'hospitalized':r'hospitali[sz]ed', 'intensive_care':r'intensive care',
    'isolated':r'isolated', 'occupation_or_role':r'(?:occupation|role)',
    'cruise_crew':r'cruise crew', 'cruise_passenger_guest':r'(?:cruise passenger|guest)',
    'contact_with_case':r'contact with case', 'contact_setting':r'contact setting',
    'travel_from':r'(?:travelled from|traveled from|travel from)',
    'travel_to':r'(?:travelled to|traveled to|travel to)',
    'confirmation_method':r'confirmation method', 'accession_id':r'accession(?: id)?',
    'date_onset':r'(?:symptom onset|onset|symptoms began)',
    'onset_date':r'(?:symptom onset|onset|symptoms began)',
    'date_confirmation':r'(?:confirmed|confirmation date|date of confirmation)',
    'confirmation_date':r'(?:confirmed|confirmation date|date of confirmation)',
    'date_death':r'(?:died|death date|date of death)',
    'date_reported':r'(?:reported|report date|date reported)',
    'report_date':r'(?:reported|report date|date reported)',
    'ship_board_date':r'(?:boarded|boarding date)',
    'ship_disembark_date':r'(?:disembarked|disembarkation date)',
}
_IDENTITY_FIELDS = {'case_id','patient_id','case_label','case_identifier','workflow_case_label'}


def _clinical_support(name, value, text):
    """Require a typed role adjacent to a value, never mere token presence."""
    values = value if isinstance(value,list) else [value]
    if name in _IDENTITY_FIELDS:
        token = str(value).strip()
        if re.match(r'^(?:patient|case)\s+(?:no\.?\s*|#\s*)?\w+',token,re.I):
            return _contains(text,token)
        return bool(re.search(r'\b(?:patient|case)\s+(?:id\s*[:=]?\s*|no\.?\s*|#\s*)?'+re.escape(token)+r'(?!\w)',text,re.I))
    label = _CLINICAL_LABELS[name]
    for item in values:
        token = re.escape(str(item).strip())
        if isinstance(item,bool):
            token = r'(?:yes|true)' if item else r'(?:no|false)'
        # Exact typed table header/cell correspondence also supports a field.
        table_lines = [line for line in text.splitlines() if '|' in line]
        table_supported = any(
            re.fullmatch(label,h.strip(),re.I) and re.fullmatch(token,c.strip(),re.I) and
            _affirmative_assertion(h + ': ' + c, 0, len(h + ': ' + c), negative_value=item is False)
            for header,row in zip(table_lines,table_lines[1:])
            if len(header.strip('|').split('|')) == len(row.strip('|').split('|'))
            for h,c in zip(header.strip('|').split('|'),row.strip('|').split('|')))
        if table_supported:
            continue
        if name == 'age' and any(_affirmative_assertion(text,m.start(),m.end()) for m in re.finditer(r'(?<![\w.])'+token+r'\s*(?:-\s*)?years?\s*(?:-\s*)?old\b',text,re.I)):
            continue
        # No intervening observation words allowed between clinical label/value.
        if not any(_affirmative_assertion(text,m.start(),m.end(),negative_value=item is False) for m in re.finditer(r'\b'+label+r'\s*(?:(?:on|in|was|is|of)\s+|[:=]\s*)?'+token+r'(?![\w-]|\.\d)',text,re.I)):
            return False
    return True


_UNCERTAIN_RELATION = re.compile(r"\b(?:not|no|never|neither|nor|cannot|can't|isn't|aren't|wasn't|weren't|may|might|could|possibly|perhaps|uncertain|unclear|unknown|whether|if|unless|unconfirmed|unlikely|unproven|unproved|doubt)\b",re.I)


def _affirmative_assertion(text, start, end, *, negative_value=False):
    # Polarity belongs to the measured predicate, not adjacent measurements.
    # Conditions can govern the whole sentence even before a conjunction.
    if re.search(r'\b(?:if|unless)\b|(?:^|[,;]\s*)si\s+', text, re.I):
        return False
    sentence_left, sentence_right = sentence_bounds(text, start, end)
    separators = list(re.finditer(r';|\n|\b(?:and|but|while|whereas)\b', text, re.I))
    left = max([sentence_left] + [m.end() for m in separators if m.end() <= start])
    right = min([sentence_right] + [m.start() for m in separators if m.start() >= end])
    clause = text[left:right]
    # May with a date/temporal preposition is a month, not a modal verb.
    clause = re.sub(r'\bMay(?=\s+\d{1,4}\b)|(?<=in )May\b|(?<=during )May\b', '', clause, flags=re.I)
    if negative_value:
        clause = re.sub(r'[:=]\s*(?:no|false)\b', '', clause, flags=re.I)
    french_uncertain = (r"\b(?:ne\s+|n['\u2019]\w+\s+)[^.!?;]{0,45}\b(?:pas|jamais|plus)\b|"
                        r"\b(?:aucun[es]*|pourrait|pourraient|serait|seraient|peut[\s-]\u00eatre|peut[\s-]etre|hypoth[e\u00e9]tique|incertain[es]*)\b")
    return not (_UNCERTAIN_RELATION.search(clause) or re.search(french_uncertain, clause, re.I))


def supports_patient_relation(text, left, right):
    """Only an affirmative, explicitly bound identity relation can merge cases."""
    if not left or not right:
        return False
    for clause in re.split(r'(?<=[.!?;])\s+',text):
        if _UNCERTAIN_RELATION.search(clause):
            continue
        for a,b in ((left,right),(right,left)):
            if re.search(r'(?<!\w)'+re.escape(str(a))+r'\s+(?:is|was)\s+the same (?:patient|case) as\s+'+re.escape(str(b))+r'(?!\w)',clause,re.I):
                return True
    return False


def supports_partition_relation(text,members,target):
    if not isinstance(members,list) or not members or not target:
        return False
    for clause in re.split(r'(?<=[.!?;])\s+',text):
        if _UNCERTAIN_RELATION.search(clause):
            continue
        if (all(isinstance(label,str) and _contains(clause,label) for label in members+[target]) and
            re.search(r'\b(?:mutually exclusive|non.overlapping)\b',clause,re.I) and
            re.search(r'\b(?:exhaustive|complete partition)\b',clause,re.I)):
            return True
    return False


def _patient_local_text(text, record):
    """Ambiguous multi-patient clauses cannot support individual attributes."""
    identity = next((str(record[k]) for k in _IDENTITY_FIELDS if _present(record.get(k))),None)
    if not identity:
        return text
    selected = []
    current_subject = False
    for clause in re.split(r'(?<=[.!?;])\s+',text):
        labels = re.findall(r'\b(?:patient|case)\s+(?:id\s*[:=]?\s*|no\.?\s*|#\s*)?[A-Za-z0-9][\w.-]*',clause,re.I)
        if labels:
            current_subject = all(label.rstrip('.,;:').casefold() == identity.casefold() or
                                  re.fullmatch(r'(?:patient|case)\s+(?:id\s*[:=]?\s*|no\.?\s*|#\s*)?'+re.escape(identity),label.rstrip('.,;:'),re.I)
                                  for label in labels)
        elif current_subject:
            # Only explicit clinical continuations inherit the named patient.
            # A new unlabeled subject (for example a relative) is not that patient.
            continuation = r'(?:he|she|they)\b|(?:' + '|'.join(_CLINICAL_LABELS.values()) + r')\b'
            if not re.match(continuation, clause.strip(), re.I):
                current_subject = False
        if current_subject:
            selected.append(clause)
    return '\n'.join(selected)


def _valid_calendar_value(name, value):
    token = str(value).strip()
    # Reporting periods can retain year/month precision; event dates cannot.
    if name == 'reporting_period':
        if re.fullmatch(r'\d{4}(?:-\d{2})?', token):
            token += '-01-01' if len(token) == 4 else '-01'
        elif not re.fullmatch(r'\d{4}-\d{2}-\d{2}', token):
            # This field is a source label, including weeks, quarters and ranges.
            # Validate embedded ISO dates/weeks without inventing a conversion.
            try:
                for iso_date in re.findall(r'(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)', token):
                    date.fromisoformat(iso_date)
                for year, week in re.findall(r'(?<!\d)(\d{4})-W(\d{2})(?!\d)', token, re.I):
                    date.fromisocalendar(int(year), int(week), 1)
            except ValueError:
                return False
            return bool(token)
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', token):
        return False
    try:
        date.fromisoformat(token)
    except ValueError:
        return False
    return True


def _role_bound_occurrence(name, value, text):
    if name in _DATES:
        typed = typed_date_support(name, value, text)
        if typed is not None:
            return typed
    # Reject explicit competing roles; repeated values can still have a separate
    # affirmative event occurrence in the same source-local span.
    for match in re.finditer(r'(?<!\w)' + re.escape(str(value).strip()) + r'(?!\w)', text, re.I):
        prefix = text[max(0, match.start()-80):match.start()]
        if name == 'reporting_period' and re.fullmatch(r'\d{4}',str(value).strip()):
            # An endpoint of an explicit multi-year period is not an annual total.
            join = r'(?:-|\u2013|\u2014|to|through)'
            if (re.search(r'\b\d{4}\s*'+join+r'\s*$',prefix,re.I) or
                    re.match(r'\s*'+join+r'\s*\d{4}\b',text[match.end():],re.I)):
                continue
        if name in _GEO:
            if re.search(r'\b(?:majority|most|majorit[e\u00e9]|plupart)\b.{0,35}\b(?:in|from|at|a|\u00e0)\s*$', prefix, re.I):
                continue
            wrong_role = (r'\b(?:from|to|nationals? of|citizens? of|native of|born in|'
                          r'nationality(?: is)?|originat(?:ed|ing) in|'
                          r'(?:pro)?ven(?:aient|ait|ant|us?|ues?) de|originaires? de)'
                          r'\s*[:=]?\s*(?:(?:the|le|la|les)\s+)?$')
        else:
            wrong_role = r'\b(?:(?:published|updated|posted|publication)\s*(?:on|date|:)?|numbered|(?:study|report|reference|patient|case|figure|table)\s*(?:id|number|no\.?|#)|(?:reference|identifier|version|id))\s*[:=#]?\s*$'
        if not re.search(wrong_role, prefix, re.I):
            return True
    return False


_COUNT_LABELS = {
    'cases_confirmed': r'(?:confirmed\s+(?:cases?|cas)|cas\s+confirm[e\u00e9]s?)',
    'cases_probable': r'(?:probable\s+cases?|cas\s+probables?)',
    'cases_suspected': r'(?:suspected\s+cases?|cas\s+suspects?)',
    'cases_unspecified': r'(?:cases?|cas)',
    'deaths': r'(?:deaths?|fatalities|deces|d\u00e9c\u00e8s)',
    'hospitalizations': r'hospitali[sz]ations?',
    'icu_admissions': r'(?:icu admissions?|intensive care admissions?)',
    'tests_positive': r'positive tests?', 'tests_total': r'(?:total tests?|tests?)',
}


def _canonical_metric(value, disease=None):
    # A disease-qualified display label denotes the same metric, provided its
    # count semantics are checked separately on the metric_name field.
    if disease:
        label = str(value)
        for alias in sorted(disease_names(disease), key=len, reverse=True):
            label = re.sub(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', '', label, flags=re.I)
        label = re.sub(r'^\s*(?:cumulative|annual|total|newly reported|new)\s+', '', label, flags=re.I)
        value = label
    key = re.sub(r'[^a-z0-9]+','_',str(value).casefold()).strip('_')
    aliases = {'confirmed_cases':'cases_confirmed','probable_cases':'cases_probable',
               'suspected_cases':'cases_suspected','cases':'cases_unspecified',
               'positive_tests':'tests_positive','total_tests':'tests_total'}
    key = aliases.get(key,key)
    return key if key in _COUNT_LABELS or key == 'incidence_rate' else None


_INCIDENCE_ASSERTION = re.compile(
    r"\bincidence\s+rate\s*(?:(?:was|is|of)\s+|[:=]\s*)?"
    r"(?P<value>"+NUMBER_TOKEN+r")\s+(?:cases?\s+)?per\s+"
    r"(?P<denominator>"+NUMBER_TOKEN+r")(?![\d.,%])"
    r"(?:\s+(?P<scope>population|persons?|people))?(?![\w-])",re.I)


def _incidence_units_match(record, assertion):
    """Match every stated unit to the denominator of this one rate predicate."""
    source_denominator = number_value(assertion.group('denominator'))
    if source_denominator <= 0:
        return False
    source_scope = (assertion.group('scope') or '').casefold()
    supplied = False
    for field in ('metric_unit','count_unit','unit','metric_denominator'):
        if not _present(record.get(field)):
            continue
        supplied = True
        prefix = r'(?:per\s+)?' if field == 'metric_denominator' else r'per\s+'
        expected = re.fullmatch(prefix+r'(?P<denominator>'+NUMBER_TOKEN+r')(?:\s+(?P<scope>population|persons?|people))?',str(record[field]).strip(),re.I)
        if not expected or number_value(expected.group('denominator')) != source_denominator:
            return False
        if expected.group('scope') and expected.group('scope').casefold() != source_scope:
            return False
    return supplied


def _json_fields(text):
    try:
        value = json.loads(text)
    except (ValueError,TypeError):
        return {}
    if not isinstance(value,dict) or any(isinstance(v,(dict,list)) for v in value.values()):
        return {}
    return {str(k).casefold():v for k,v in value.items()}


def _table_scope_fields(text):
    lines = [line for line in text.splitlines() if '|' in line]
    if len(lines) != 2:
        return {}
    headers, cells = ([cell.strip() for cell in line.strip('|').split('|')] for line in lines)
    if len(headers) != len(cells):
        return {}
    aliases = {'state':'subnational_location','region':'subnational_location',
               'province':'subnational_location','year':'reporting_period'}
    result = {}
    for header, value in zip(headers,cells):
        key = re.sub(r'[^a-z0-9]+','_',header.casefold()).strip('_')
        key = aliases.get(key,key)
        if key in ('disease',) + _GEO + _DATES:
            result[key] = value
    return result


def _count_statements(text, label, record=None):
    """Keep competing metric assertions in separate local clauses.

    Co-occurrence in a sentence does not bind France's count to Germany.
    Ambiguous coordinated prose remains a candidate for finer extraction.
    """
    from .source_assertions import count_mentions
    counter = r'(?<![\w.])\d[\d,.]*\s+(?:' + label + r')|(?:' + label + r')\s*:\s*\d[\d,.]*'
    # Split a new named numeric assertion only AFTER an existing count.
    # This preserves enclosing geography, periods, units and count semantics.
    separator = r';\s*|\s+(?:and|while|whereas|but|et|y)\s+|,\s+'
    subject_count = r'(?=(?P<subject>(?:[^\W\d_]+[ \t]+){1,8})(?P<count>\d[\d,.]*))'
    temporal_or_id = r'\b(?:during|in|on|as of|since|until|period|ending|week|month|year|reference|identifier|id|number)\s*$'
    sentences=[]
    for left, right in sentence_spans(text):
        original = text[left:right]
        # Explicit included groups inherit the parent's scope in one sentence;
        # the parent is evaluated separately and cannot borrow child geography.
        relation = re.search(r',\s*(?:including|of\s+which|dont|y\s+compris)\s+', original, re.I)
        if relation and original[:relation.start()].strip():
            from .source_assertions import count_mentions, _DATE, _fold
            from .geography import explicit_country_matches
            from .disease_relevance import build_disease_relevance_context
            parent, child = original[:relation.start()], original[relation.end():]
            parents, children = count_mentions(parent), count_mentions(child)
            if children:
                parent_scope = parent if parents else ''
                for item in reversed(parents):
                    number = re.match(NUMBER_TOKEN, parent[item['char_start']:])
                    if number:
                        left, right = item['char_start'], item['char_start'] + number.end()
                        parent_scope = parent_scope[:left] + ' ' * (right - left) + parent_scope[right:]
                # Each coordinated included member gets only the enclosing
                # parent scope. No member lends scope back to its parent or peers.
                groups, start = [], 0
                for boundary in re.finditer(r',\s+|\s+(?:and|et)\s+', child, re.I):
                    if count_mentions(child[start:boundary.start()]) and count_mentions(child[boundary.end():]):
                        groups.append(child[start:boundary.start()])
                        start = boundary.end()
                groups.append(child[start:])
                yield from _count_statements(parent, label, record)
                for group in groups:
                    members = count_mentions(group)
                    if not members:
                        continue
                    scope = parent_scope
                    new_subject = re.search(r'\b(?:reported|reports|recorded|notified|confirmed|has|have|had|signale|declare|comptabilise)\b', _fold(group[:members[0]['char_start']]))
                    if new_subject:
                        scope = ''
                    else:
                        if list(explicit_country_matches(group)):
                            for place, _ in explicit_country_matches(scope):
                                scope = re.sub(re.escape(place), ' ', scope, flags=re.I)
                        if record and record.get('disease'):
                            context = build_disease_relevance_context({'structured_task': {'disease': record['disease']}})
                            terms = list(disease_names(record['disease'])) + context.get('incompatible_disease_terms', [])
                            if any(_contains(group, term) for term in terms):
                                for term in sorted(terms, key=len, reverse=True):
                                    scope = re.sub(r'(?<!\w)' + re.escape(term) + r'(?!\w)', ' ', scope, flags=re.I)
                        if re.search(r'(?<!\d)(?:19|20)\d{2}(?!\d)(?!\s+(?:cases?|cas|deaths?|patients?|tests?))', group, re.I):
                            scope = _DATE.sub(' ', scope)
                            scope = re.sub(r'(?<!\d)(?:19|20)\d{2}(?!\d)', ' ', scope)
                        if record and re.search(r'\b(?:in|from|at|en|dans)\s+[A-Z]', group):
                            for key in ('subnational_location', 'locality'):
                                place = record.get(key)
                                if place and not _contains(group, place):
                                    scope = re.sub(re.escape(str(place)), ' ', scope, flags=re.I)
                    yield from _count_statements(scope + original[relation.start():relation.end()].lstrip(',') + group, label, record)
                continue
        start=0
        for match in re.finditer('(?:'+separator+')'+subject_count,original,re.I):
            if (re.search(counter,original[start:match.start()],re.I) or count_mentions(original[start:match.start()])) and not re.search(temporal_or_id,match.group('subject'),re.I):
                sentences.append(original[start:match.start()])
                start=match.end()
        sentences.append(original[start:])
    for sentence in sentences:
        if len(list(re.finditer(counter, sentence, re.I))) <= 1:
            yield sentence
            continue
        for clause in re.split(r';\s*|\s+(?:and|while|whereas|but|et|y)\s+|,\s+(?=[A-Z])', sentence):
            if len(list(re.finditer(counter, clause, re.I))) == 1:
                yield clause


def _source_table_grid(locator, index):
    """Read complex-cell geometry from the resolved source, never model metadata."""
    if locator.get('table_id') is None or locator.get('row_id') is None:
        return None
    documents = index.get('documents') or []
    if isinstance(documents, dict):
        documents = list(documents.values())
    documents = [doc for doc in documents if doc.get('text_hash') == locator.get('text_hash')]
    if len(documents) != 1:
        return None
    doc = documents[0]
    table = next((table for table in doc.get('tables') or []
                  if str(table.get('table_id')) == str(locator['table_id'])), None)
    if not table or not table.get('cell_spans'):
        return None
    cells = table['cell_spans']
    complex_grid = len(table.get('header_row_ids') or []) > 1 or any(
        cell.get('row_end', 0) > cell.get('row_id', 0) + 1 or
        cell.get('column_end', 0) > cell.get('column_start', 0) + 1 for cell in cells)
    if not complex_grid:
        return None
    return {'table': table, 'text': doc.get('clean_text') or '',
            'row_id': str(locator['row_id'])}


def _grid_numeric_support(name, value, record):
    grid = record['_validated_table_grid']
    table, text = grid['table'], grid['text']
    try:
        row_id = int(grid['row_id'])
    except (ValueError, TypeError):
        return False
    cells = table['cell_spans']
    for cell in cells:
        start, end = cell.get('char_start'), cell.get('char_end')
        if (not isinstance(start, int) or not isinstance(end, int) or
                not 0 <= start <= end <= len(text) or text[start:end] != cell.get('quote')):
            return False
    headers = [cell for cell in cells if cell.get('is_header')]
    active = [cell for cell in cells if not cell.get('is_header') and
              cell['row_id'] <= row_id < cell['row_end']]
    metric = _canonical_metric(record.get('metric_name'), record.get('disease')) if name == 'metric_value' else name
    metric_pattern = _COUNT_LABELS.get(metric)
    if not metric_pattern:
        return False
    def column_headers(cell):
        return [header for header in headers if
                header['column_start'] <= cell['column_start'] < header['column_end']]
    def column_matches(cell):
        labels = [header['quote'] for header in column_headers(cell)]
        if metric and metric.startswith('cases_') and any(case_count_label_role(label) for label in labels):
            return False
        typed = [_canonical_metric(label) for label in labels]
        known = [kind for kind in typed if kind]
        return metric in known if known else any(re.search(metric_pattern, item, re.I) for item in labels)
    row_scope = []
    explicit_scope = {}
    for cell in active:
        header_text = ' '.join(header['quote'] for header in column_headers(cell))
        if re.search(r'\b(?:country|territory|region|state|province|district|locality|location|period|year|date|week|disease|condition|pathogen|syndrome)\b', header_text, re.I):
            if any(_canonical_metric(header['quote']) for header in column_headers(cell)):
                continue
            row_scope.append(cell['quote'])
            for field in ('disease',) + _GEO + _DATES:
                if ((field == 'disease' and re.search(r'\b(?:disease|condition|pathogen|syndrome)\b',header_text,re.I)) or
                        (field == 'country' and re.search(r'\b(?:country|territory)\b',header_text,re.I)) or
                        (field in _DATES and re.search(r'\b(?:period|year|date|week)\b',header_text,re.I))):
                    explicit_scope.setdefault(field,[]).append(cell['quote'])
    provenance = _provenance(record)
    citation = provenance.get(name) or (provenance.get('metric_value') if name == metric else {}) or {}
    citation = citation if isinstance(citation,dict) else {}
    quote = next((str(citation[key]) for key in ('quote','evidence_quote','supporting_quote')
                  if citation.get(key)), '')
    hits = []
    for cell in active:
        # A count spanning different data rows is not a separate per-row count.
        if cell['row_id'] != row_id or cell['row_end'] != row_id + 1 or cell['column_end'] != cell['column_start'] + 1:
            continue
        if not column_matches(cell):
            continue
        number = re.fullmatch(r'\s*(' + NUMBER_TOKEN + r')\s*(?:\(\s*\d+(?:\.\d+)?%?\s*\))?\s*',cell['quote'])
        if not number or number_value(number.group(1)) != float(value):
            continue
        if quote:
            occurrences = [m for m in re.finditer(re.escape(quote),text)]
            if not any(m.start() < cell['char_end'] and cell['char_start'] < m.end() for m in occurrences):
                continue
        selected_headers = [header['quote'] for header in column_headers(cell)]
        selected = '\n'.join(selected_headers)
        local = '\n'.join([record.get('_validated_bound_scope') or '', *row_scope, selected])
        if not _affirmative_assertion(local,0,len(local)) or re.search(r'\bhypothetical\b',local,re.I):
            continue
        fields = ['disease'] + [field for field in _GEO + _DATES if _present(record.get(field))]
        valid = True
        for field in fields:
            expected = record.get(field)
            if not _supports(field,expected,local,record):
                valid = False; break
            if field in explicit_scope and not _supports(field,expected,'\n'.join(explicit_scope[field]),record):
                valid = False; break
            # A value named by another column's header cannot support this cell.
            if field in _GEO + _DATES and any(_supports(field,expected,header['quote'],record) for header in headers):
                if not _supports(field,expected,selected,record):
                    valid = False; break
        if valid:
            hits.append(cell)
    return len(hits) == 1



def _count_semantics_support(semantics, text, record):
    """Bind canonical count meaning to the same supported numeric assertion."""
    from .source_assertions import count_mentions, _fold, _DATE_LABEL

    counts = [(field, record[field]) for field in _COUNT_LABELS if _present(record.get(field))]
    canonical = _canonical_metric(record.get('metric_name'), record.get('disease'))
    if canonical in _COUNT_LABELS and _present(record.get('metric_value')):
        counts.append((canonical, record['metric_value']))
    if not counts:
        return False
    bound_scope = record.get('_validated_bound_scope') or ''
    statement_text = text[len(bound_scope)+1:] if bound_scope and text.startswith(bound_scope+'\n') else text
    for field, value in set(counts):
        matched = False
        scoped_mentions = []
        for assertion in _count_statements(statement_text, _COUNT_LABELS[field], record):
            scoped_mentions.extend((assertion, item) for item in count_mentions(assertion))
        for text, mention in scoped_mentions:
            mentions = count_mentions(text)
            if mention['field'] != field or mention['value'] != value or mention['bounded']:
                continue
            left, right = mention['char_start'], mention['char_end']
            sentence_start, sentence_end = sentence_bounds(text, left, right, semicolons=True, paragraphs=True)
            sentence = text[sentence_start:sentence_end]
            # A same-valued count in another country's statement is not support.
            scope = record.get('_validated_bound_scope') or ''
            local = scope + '\n' + sentence if scope and not sentence.startswith(scope) else sentence
            if re.search(r'\b(?:hypothetical|would|could|might)\b', sentence, re.I) or not _supports(field, value, local, record):
                continue
            previous = [m for m in mentions if sentence_start <= m['char_start'] < left]
            following = [m for m in mentions if right <= m['char_start'] < sentence_end]
            prefix_start = previous[-1]['char_end'] if previous else sentence_start
            suffix_end = following[0]['char_start'] if following else sentence_end
            prefix = _fold(text[prefix_start:left])
            suffix = _fold(text[right:suffix_end])
            phrase = _fold(text[left:right])
            if semantics == 'newly_reported':
                new = r'(?:new|newly\s+reported|nouveaux?|nouvelles?)'
                matched = bool(re.search(r'\b' + new + r'\b', phrase) or
                    re.search(r'\b' + new + r'\s*:?\s*$', prefix) or
                    re.search(r'(?:^|,)\s*(?:(?:all|were|are|have been|ont ete|tous|toutes)\s+){0,3}(?:newly\s+reported|nouvellement\s+declares)\b', suffix))
            elif semantics == 'cumulative':
                explicit = re.search(r'\b(?:cumulative(?:\s+(?:total|count|number))?(?:\s+of)?|(?:un\s+)?total\s+cumule(?:\s+de)?|cumul(?:\s+de)?)\s*:?\s*$', prefix)
                total = re.search(r'\b(?:total(?:\s+(?:count|number))?(?:\s+of)?|(?:un\s+)?total(?:\s+de)?|au\s+total)\s*[:,]?\s*$', prefix)
                start = re.search(r'\b(?:since|depuis)\s+(?:' + _DATE_LABEL + r'|(?:19|20)\d{2})(?!\w)', prefix + ' ' + suffix, re.I)
                unit = record.get('metric_unit') or record.get('count_unit') or record.get('unit')
                unit_tail = (r'(?:\s*' + re.escape(_fold(str(unit))) + r'\s*;)?') if unit else ''
                # Preserve an explicitly attached compact metric qualifier,
                # without interpreting an unrelated later clause as its label.
                declared_tail = re.fullmatch(r'\s*;' + unit_tail + r'\s*cumulative\s*\.?\s*', _fold(text[right:]))
                first_case = re.match(r'\s*since\s+(?:the\s+)?(?:first|index)\s+'
                    r'(?:[a-z][a-z-]*\s+){0,5}cases?\s+(?:(?:was|were)\s+)?'
                    r'(?:recorded|reported|confirmed|detected)\s+(?:on\s+)?' + _DATE_LABEL,
                    _fold(sentence), re.I)
                # An explicitly included death count shares this cumulative
                # reporting predicate; a new assertion cannot inherit it.
                inherited = (not previous or bool(re.fullmatch(r'\s*,?\s*(?:including|of which)\s*', prefix)))
                incident = re.search(r'\b(?:new|newly\s+reported|today|this\s+(?:week|month))\b', _fold(sentence))
                matched = bool(explicit or (total and start) or declared_tail or
                               (first_case and inherited and not incident))
            elif semantics == 'subset':
                # The denominator of an explicit fraction is never its subset.
                denominator = bool(previous and re.search(r'\b(?:out\s+of|of|sur)\s*$', prefix))
                numerator = re.match(r'\s*(?:out\s+of|of\s+a\s+total\s+of|sur|as\s+part\s+of\s+(?:a\s+)?(?:larger\s+)?group\s+of)\s+' + NUMBER_TOKEN, text[right:sentence_end], re.I)
                relation_prefix = text[sentence_start:left]
                projection = analysis_projection(relation_prefix, fold=True)
                cues = list(re.finditer(r'\b(?:including|of\s+which|dont|y\s+compris|among|parmi|subset\s+of)\b', projection.text))
                included = False
                if cues:
                    cue_end = sentence_start + projection.original_span(cues[-1].start(), cues[-1].end())[1]
                    # Coordinated included members may share one explicit cue,
                    # but a new reporting predicate starts another assertion.
                    segment = text[cue_end:left]
                    for earlier in reversed(previous):
                        if earlier['char_start'] >= cue_end:
                            a, b = earlier['char_start'] - cue_end, earlier['char_end'] - cue_end
                            segment = segment[:a] + ' ' * (b - a) + segment[b:]
                    included = not re.search(r'\b(?:reported|reports|recorded|notified|confirmed|has|have|had|was|were|while|whereas|but|signale|declare|comptabilise)\b', _fold(segment))
                matched = bool(not denominator and (numerator or included))
            if matched:
                break
        if not matched:
            return False
    return True


def _supports(name, value, text, record):
    text = analysis_projection(text).text
    if not _present(value):
        return False
    if name in _DATES and not _valid_calendar_value(name, value):
        return False
    if name in {'metric_unit','count_unit','unit'} and str(value).casefold() == 'count' and (
            _canonical_metric(record.get('metric_name'), record.get('disease')) == 'incidence_rate' or _present(record.get('incidence_rate'))):
        return False
    if name != 'patient_links':
        text = _patient_local_text(text,record)
    table_scope = _table_scope_fields(text)
    if name in table_scope:
        if name in {'disease','disease_standard_name'}:
            return same_disease_name(table_scope[name],value)
        return str(table_scope[name]).casefold() == str(value).casefold()
    structured = record.get('_validated_json_fields') or _json_fields(text)
    source_metric = _canonical_metric(structured.get('metric_name'))
    if source_metric and _present(structured.get('metric_value')):
        if source_metric in structured and structured[source_metric] != structured['metric_value']:
            return False
        structured = {**structured,source_metric:structured['metric_value']}
    structured_name = _canonical_metric(record.get('metric_name'), record.get('disease')) if name == 'metric_value' else name
    if structured_name in structured:
        actual = structured[structured_name]
        if name in {'disease','disease_standard_name'}:
            return same_disease_name(actual,value)
        if name == 'reporting_period' and type(actual) is int and re.fullmatch(r'\d{4}',str(actual)):
            actual = str(actual)
        if isinstance(actual,str) and isinstance(value,str):
            actual, value = actual.casefold(), value.casefold()
        if isinstance(actual,bool) != isinstance(value,bool) or actual != value:
            return False
        if name in _NUMERIC:
            binding_fields = ['disease'] + [k for k in _GEO + _DATES if _present(record.get(k))]
            return all(_supports(k,record.get(k),text,record) for k in binding_fields)
        return True
    canonical = _canonical_metric(record.get('metric_name'), record.get('disease'))
    if name == 'metric_name' and canonical:
        metric_value = record.get('metric_value', record.get(canonical))
        qualifier = re.match(r'^\s*(cumulative|annual|newly reported|new)\b', str(value), re.I)
        if qualifier:
            semantics = {'new':'newly_reported', 'newly reported':'newly_reported'}.get(qualifier[1].lower(), qualifier[1].lower())
            if not _supports('count_semantics', semantics, text, record):
                return False
        return _present(metric_value) and _supports(canonical,metric_value,text,record)
    if name in {'metric_unit','count_unit','unit','metric_denominator'} and (canonical == 'incidence_rate' or _present(record.get('incidence_rate'))):
        rate = record.get('incidence_rate') if _present(record.get('incidence_rate')) else record.get('metric_value')
        return _present(rate) and _supports('incidence_rate',rate,text,record)
    if name in {'metric_unit','count_unit','unit'} and str(value).casefold() == 'count':
        count_values = [(field,record[field]) for field in _COUNT_LABELS if _present(record.get(field))]
        if canonical in _COUNT_LABELS and _present(record.get('metric_value')):
            count_values.append((canonical,record['metric_value']))
        return bool(count_values) and all(_supports(field,number,text,record) for field,number in count_values)

    if name in {'date_reported','report_date'}:
        typed = typed_date_support(name, value, text)
        if typed is not None:
            return bool(typed and _affirmative_assertion(text, 0, len(text)))
    if name in _CLINICAL_LABELS or name in _IDENTITY_FIELDS:
        return _clinical_support(name,value,text)
    if name == 'geographic_scope_type':
        scope = record.get('geographic_scope')
        geographic_field = {'country':'country','national':'country','subnational':'subnational_location','local':'locality','locality':'locality'}.get(str(value))
        return bool(geographic_field and scope and scope == record.get(geographic_field) and _contains(text,scope))
    if name == 'patient_links':
        identity = next((record.get(k) for k in _IDENTITY_FIELDS if record.get(k)),None)
        return bool(identity and isinstance(value,list) and value and
                    _contains(text,identity) and
                    all(isinstance(link,dict) and link.get('identity') and link.get('document_hash') and
                        supports_patient_relation(text,identity,link['identity']) for link in value))
    if name == 'disjoint_partition':
        if not isinstance(value,dict) or value.get('axis') != 'population_scope':
            return False
        members = value.get('members')
        metric_labels = {'cases_confirmed':'confirmed cases','cases_probable':'probable cases','cases_suspected':'suspected cases','cases_unspecified':'cases','deaths':'deaths','hospitalizations':'hospitalizations'}
        label = metric_labels.get(value.get('metric'))
        return bool(isinstance(members,list) and len(members) >= 2 and len(set(members)) == len(members) and
                    value.get('mutually_exclusive') is True and value.get('exhaustive') is True and
                    value.get('target') and label and _contains(text,label) and
                    all(isinstance(member,str) and member and _contains(text,member) for member in members) and
                    _contains(text,value['target']) and
                    supports_partition_relation(text,members,value['target']))
    if isinstance(value, list):
        return all(_contains(text,item) for item in value)
    if name in {'count_semantics','statistical_count_type'} and str(value) in {'subset','cumulative','newly_reported'}:
        return _count_semantics_support(str(value), text, record)
    if name in {'count_semantics','statistical_count_type'} and str(value) == 'annual':
        period = str(record.get('reporting_period') or '')
        return annual_period_support(period, text)
    if name in {'count_semantics','statistical_count_type'} and str(value) in {'unspecified','unknown'}:
        return True
    if name == 'case_definition':
        return all(_contains(text,item.strip()) for item in str(value).split(','))
    if name == 'metric_category':
        return str(value) in _metric_families({k:v for k,v in record.items() if k != 'metric_category'})
    if name in _NUMERIC:
        if record.get('_validated_table_grid'):
            return _grid_numeric_support(name,value,record)
        single_column = record.get('_validated_single_column')
        if single_column:
            header, cell = single_column
            table_text = '\n'.join((record.get('_validated_bound_scope') or '',header + ' |',cell + ' |'))
            return _supports(name,value,table_text,{k:v for k,v in record.items() if k != '_validated_single_column'})
        number = number_pattern(value)
        if re.search(r'(?:at least|more than|over|at most|fewer than|less than|approximately|about|plus de|au moins|moins de|au plus|environ|pr[e\u00e8]s de)\s+' + number,text,re.I):
            return False
        metric_field = _canonical_metric(record.get('metric_name'), record.get('disease')) if name == 'metric_value' else name
        label = _COUNT_LABELS.get(metric_field) or {'positivity_rate':r'positivity','incidence_rate':r'incidence'}.get(name, re.escape(str(record.get('metric_name') or name.replace('_',' '))))
        if metric_field and metric_field.startswith('cases_') and record.get('disease'):
            source_disease = '(?:' + '|'.join(re.escape(alias) for alias in sorted(disease_names(record['disease']),key=len,reverse=True)) + ')'
            label = label.replace('(?:cases?|cas)', '(?:' + source_disease + r'\s+)?(?:cases?|cas)')
            if '(?:cases?|cas)' not in label:
                label = label.replace('cases?', '(?:' + source_disease + r'\s+)?cases?')
        if metric_field and metric_field.startswith('cases_'):
            # Explicit incident-count modifiers do not change case definition.
            label = r'(?:(?:new|newly reported|nouveaux|nouveau|nouvelles)\s+)?(?:' + label + ')'
        # Known typed columns must agree before considering natural-language labels.
        def column_matches(cell):
            if metric_field and metric_field.startswith("cases_") and case_count_label_role(cell):
                return False
            typed = _canonical_metric(cell)
            return typed == metric_field if typed else bool(re.search(label,cell,re.I))
        # Verify the numeric cell's own column, not another column's label.
        if '|' in text:
            lines = text.splitlines()
            first_table = next(i for i,line in enumerate(lines) if '|' in line)
            context = '\n'.join(lines[:first_table])
            headings = None
            binding_fields = ['disease'] + [k for k in _GEO + _DATES if _present(record.get(k))]
            for line in lines[first_table:]:
                if '|' not in line:
                    headings = None
                    context = line
                    continue
                cells = [cell.strip() for cell in line.strip('|').split('|')]
                if any(column_matches(cell) for cell in cells):
                    headings = cells
                    continue
                if not headings or len(headings) != len(cells):
                    continue
                local = '\n'.join((context, ' | '.join(headings), line))
                if not _affirmative_assertion(context,0,len(context)) or re.search(r'\bhypothetical\b',context,re.I):
                    continue
                if not all(_supports(k,record.get(k),local,record) for k in binding_fields):
                    continue
                if any(column_matches(h) and re.fullmatch(number,c) and _affirmative_assertion(h + ': ' + c,0,len(h + ': ' + c)) for h,c in zip(headings,cells)):
                    return True
            return False
        # Prose must bind disease and geographic/time facts to this statement.
        bound_scope = record.get('_validated_bound_scope') or ''
        if bound_scope and (not _affirmative_assertion(bound_scope,0,len(bound_scope)) or re.search(r'\bhypothetical\b',bound_scope,re.I)):
            return False
        # Bound headings supply missing scope, but cannot override an explicit
        # observation year in the original count sentence.
        statement_text = text[len(bound_scope)+1:] if bound_scope and text.startswith(bound_scope+'\n') else text
        from .geography import explicit_country_matches, explicit_country_key
        for sentence in _count_statements(statement_text,label,record):
            country_mentions = list(explicit_country_matches(sentence))
            local_countries = {canonical for name,canonical in country_mentions
                               if _role_bound_occurrence('country',name,sentence)}
            expected_country = explicit_country_key(record.get('country'))
            # An explicit travel/origin role cannot borrow a heading to become
            # the observation location of this numeric assertion.
            if expected_country and expected_country in {key for _,key in country_mentions} and expected_country not in local_countries:
                continue
            if any(_present(record.get(field)) and _contains(sentence, record[field])
                   and not _role_bound_occurrence(field, record[field], sentence)
                   for field in _GEO):
                continue
            permitted_countries = {expected_country}
            # Some territorial names also have ISO country codes. A source's
            # explicit locality (country) relationship binds the two roles;
            # simple co-occurrence still cannot override a competing country.
            for geo in ('subnational_location', 'locality'):
                place = record.get(geo)
                country = record.get('country')
                if place and country and re.search(re.escape(str(place)) +
                        r'\s*(?:\(\s*|,\s*)' + re.escape(str(country)) + r'(?!\w)', sentence, re.I):
                    permitted_countries.update(key for _,key in explicit_country_matches(str(place)))
            if expected_country and local_countries and local_countries - permitted_countries:
                continue
            observation_years = set()
            for observed in re.finditer(r'\b(?:in|during|since|until|through|for|year)\s+(?:the\s+year\s+)?(\d{4})(?!\d)(?:\s*(?:-|\u2013|to|through)\s*(\d{4})(?!\d))?',sentence,re.I):
                if re.match(r'\s+(?:cases?|deaths?|patients?|tests?)\b',sentence[observed.end():],re.I):
                    continue
                observation_years.update(year for year in observed.groups() if year)
            if observation_years and any(
                    set(re.findall(r'(?<!\d)\d{4}(?!\d)',str(record.get(field) or ''))) - observation_years
                    for field in ('reporting_period','period_start_date','period_end_date','metric_period_start','metric_period_end')
                    if _present(record.get(field))):
                continue
            local = '\n'.join((bound_scope,sentence)) if bound_scope else sentence
            binding_fields = ['disease'] + [k for k in _GEO + _DATES if _present(record.get(k))]
            from .disease_relevance import build_disease_relevance_context
            disease_context = build_disease_relevance_context({'structured_task':{'disease':record.get('disease')}})
            if any(_contains(sentence,term) for term in disease_context.get('incompatible_disease_terms', [])):
                continue
            target = re.escape(str(record.get('disease') or ''))
            if re.search(target + r'\s+(?:and|or|et|ou)\s+[A-Za-z]', sentence, re.I):
                continue
            if not all(_supports(k,record.get(k),local,record) for k in binding_fields):
                continue
            if metric_field == 'incidence_rate':
                if any(numeric_span_is_complete(sentence,match.start('value'),match.end('value')) and
                       numeric_span_is_complete(sentence,match.start('denominator'),match.end('denominator')) and
                       number_value(match.group('value')) == float(value) and
                       _incidence_units_match(record,match) and _affirmative_assertion(sentence,match.start(),match.end())
                       for match in _INCIDENCE_ASSERTION.finditer(sentence)):
                    return True
                continue
            if re.search(number + r'\s+case\s+(?:investigation|report|study|series|definition|management|history)\b', sentence, re.I):
                continue
            patterns = (r'(?P<count>'+number+r')\s+(?:'+label+r')', r'(?:'+label+r')\s*:\s*(?P<count>'+number+r')')
            if any(_affirmative_assertion(sentence,m.start(),m.end())
                   and numeric_span_is_complete(sentence,m.start('count'),m.end('count'))
                   and not (metric_field and metric_field.startswith('cases_')
                            and case_count_object_role(sentence,m.start('count'),m.end('count')))
                   and not contextual_number_role(sentence,m.start('count'),m.end('count'))
                   and not bounded_count(sentence,m.start('count'),m.end('count'))
                   for pattern in patterns for m in re.finditer(pattern,sentence,re.I)):
                return True
        return False
    if name in {'disease','disease_standard_name'}:
        return any(_contains(text,alias) for alias in disease_names(value))
    if name == 'virus_or_syndrome' and same_disease_name(value, record.get('disease')):
        return any(_contains(text, alias) for alias in disease_names(value))
    if name in _DATES or name in _GEO:
        return _role_bound_occurrence(name,value,text)
    return _contains(text,value)


def assess_record_evidence(record, *, contract, evidence_index):
    numeric = [k for k in _NUMERIC if _present(record.get(k))]
    identity = next((k for k in ('case_id','patient_id','case_label','case_identifier','workflow_case_label') if _present(record.get(k))), None)
    kind = ProductKind.CASE_LEVEL if identity else ProductKind.AGGREGATE if numeric else ProductKind.CONTEXT
    reasons = []
    closure = None
    if record.get('outbreak_closure_date'):
        from .outbreak_temporal import resolve_outbreak_closure_date
        resolved = resolve_outbreak_closure_date(record, evidence_index=evidence_index)
        if (resolved and resolved['value'] == record['outbreak_closure_date']
                and resolved == record.get('outbreak_closure_derivation')):
            closure = resolved
        else:
            reasons.append('outbreak_closure_date:unverified_relative_date')
    fields = ['disease'] + [k for k in _GEO + _DATES if _present(record.get(k))] + numeric
    fields += [k for k in _SUBSTANTIVE if _present(record.get(k)) and record.get(k) != []]
    if identity:
        fields.append(identity)
    if record.get('case_span_id') and not identity and not numeric:
        # A span identifies evidence, not necessarily an individual patient.
        reasons.append('case_span_requires_explicit_patient_identity')
    if not any(_present(record.get(k)) for k in _GEO):
        reasons.append('missing_geography')
    if numeric and not any(_present(record.get(k)) for k in _DATES) and not closure:
        reasons.append('missing_observation_period')
    provenance = _provenance(record)
    evidence = []
    for name in fields:
        entry = provenance.get(name) or {}
        if not isinstance(entry, dict):
            entry = {}
        text, digest, locator, error = _local_span(record,entry,evidence_index)
        bound_scope = '\n'.join(s['quote'] for s in locator.get('bound_context_spans') or [] if s.get('role') in {'heading','table_caption','table_note'})
        source_record = {**record, '_validated_bound_scope':bound_scope}
        if not error:
            source_record['_validated_table_grid'] = _source_table_grid(locator,evidence_index)
            chunks = evidence_index.get('evidence_chunks') or {}
            chunk = chunks.get(str(locator.get('chunk_id'))) if isinstance(chunks,dict) else next((c for c in chunks if c.get('chunk_id') == locator.get('chunk_id')),None)
            if chunk and chunk.get('structured_data_kind') == 'json_record' and chunk.get('text') in text:
                from .evidence_chunking import json_fields_for_chunk
                source_record['_validated_json_fields'] = {str(k).casefold():v for k,v in json_fields_for_chunk({},chunk).items()}
        bound_spans = locator.get('bound_context_spans') or []
        headers = [s['quote'] for s in bound_spans if s.get('role') == 'table_header']
        prefix = '\n'.join(s['quote'] for s in bound_spans) + '\n'
        if (locator.get('table_id') and locator.get('row_id') and len(headers) == 1 and
                '|' not in headers[0] and text.startswith(prefix)):
            source_record['_validated_single_column'] = (headers[0],text[len(prefix):])
        supported = not error and _supports(name,record.get(name),text,source_record)
        if supported and name == 'patient_links':
            documents = evidence_index.get('documents') or []
            if isinstance(documents,dict):
                documents = list(documents.values())
            for link in record[name]:
                targets = [d for d in documents if d.get('content_hash') == link.get('document_hash')]
                valid = False
                for target in targets:
                    target_text = str(target.get('clean_text') or '')
                    target_hash = str(target.get('text_hash') or target.get('content_hash') or '').removeprefix('sha256:')
                    url = target.get('final_url') or target.get('url') or target.get('source_url')
                    if (hashlib.sha256(target_text.encode()).hexdigest() == target_hash and
                        _clinical_support('case_id',link['identity'],target_text) and
                        (link['document_hash'] == digest or (url and str(url) in text))):
                        valid = True
                supported = supported and valid
        if supported and name == identity:
            # This exact source-local quote is the patient anchor used by products.
            locator = dict(locator, patient_span_quote=text)
        reason = error or ('' if supported else 'unbound_field_value')
        if not supported and not error and name in _NUMERIC:
            # Distinguish an unsupported number from a correctly cited number
            # whose extracted temporal scope is still unresolved. Neither is
            # promoted; the latter should send recovery to the date evidence.
            without_dates = {key:value for key,value in source_record.items() if key not in _DATES}
            if _supports(name, record.get(name), text, without_dates):
                reason = 'unbound_observation_scope'
        if entry.get('basis') in ('task','search','query','inferred_from_task') or entry.get('source') in ('task','search'):
            supported, reason = False, 'task_or_search_is_not_evidence'
        evidence.append(FieldEvidence(name,record.get(name),digest,locator,text,supported,reason))
        if not supported:
            reasons.append(f'{name}:{reason}')
    if closure:
        evidence.append(FieldEvidence('outbreak_closure_date', closure['value'], closure['document_hash'],
            {**closure['locator'], 'relative_date_derivation':{
                key:closure[key] for key in ('basis','rule','date_role','publication_date','weekday','anchors')}},
            closure['quote'], True, ''))
    record_scope = contract.get('record_scope') if isinstance(contract.get('record_scope'),dict) else contract
    scoped_record = {**record, '_validated_outbreak_closure_date':closure['value'] if closure else None}
    reasons.extend(_constraint_reasons(scoped_record, record_scope))
    status = 'candidate' if reasons else 'qualified'
    if not numeric and not identity and reasons:
        kind = ProductKind.UNRESOLVED
    return RecordQualification(str(record.get('record_id') or ''),status,kind,reasons,evidence)


def prepare_record_evidence(record, *, evidence_index):
    """Correct only demonstrable publication/observation role confusion.

    Original values and citations remain in the audit trail. A missing observed
    period still fails qualification; this never borrows a year from metadata.
    """
    row = dict(record)
    provenance = dict(_provenance(record))
    chunks = evidence_index.get('evidence_chunks') or {}
    if isinstance(chunks, list):
        chunks = {str(c.get('chunk_id') or c.get('evidence_span_id')):c for c in chunks}
    chunk_id = next((row.get(k) for k in ('chunk_id','evidence_chunk_id','supporting_chunk_id') if row.get(k)), None)
    chunk = chunks.get(str(chunk_id)) or {}
    if not chunk.get('text'):
        return row
    text, digest, locator, error = _local_span(row, {'quote':chunk['text']}, evidence_index)
    if error:
        return row
    # Use this record's verified chunk, not another report elsewhere in a page.
    dates = publication_dates(chunk['text'])
    if len({item['value'] for item in dates}) != 1:
        return row
    publication = dates[0]['value']
    if row.get('source_publication_date') not in (None, '', publication):
        return row
    actions = list(row.get('evidence_normalization_actions') or [])
    for name in ('as_of_date','date_reported','report_date','metric_period_end','period_end_date'):
        if row.get(name) != publication:
            continue
        entry = provenance.get(name) or {}
        if not isinstance(entry, dict):
            continue
        if any(key in entry for key in ('char_start','char_end')):
            quoted = next((str(entry[key]) for key in ('quote','evidence_quote','supporting_quote')
                           if entry.get(key)), '')
            offset = chunk.get('char_start')
            positions = normalized_quote_spans(chunk['text'], quoted) if quoted else []
            if (type(offset) is not int or not any(
                    all(type(entry[key]) is int and entry[key] == offset+position
                        for key,position in (('char_start',left),('char_end',right)) if key in entry)
                    for left,right in positions)):
                continue
        local, local_digest, local_locator, local_error = _local_span(row, entry, evidence_index)
        # Correction cannot erase failed evidence integrity checks. The field's
        # own valid citation must actually identify this publication date; a
        # coincidentally equal value elsewhere in the chunk is insufficient.
        if (local_error or local_digest != digest or local_locator.get('chunk_id') != chunk_id
                or entry.get('basis') in ('task','search','query','inferred_from_task')
                or entry.get('source') in ('task','search')
                or publication not in {item['value'] for item in publication_dates(local)}):
            continue
        if _supports(name, row[name], local, row):
            continue
        # An explicit observation occurrence in this chunk remains unresolved
        # when its own citation is ambiguous. Do not erase a real date claim.
        if typed_date_support(name, row[name], text):
            continue
        row.setdefault(name+'_raw', row[name])
        actions.append({'field':name, 'original_value':row[name],
                        'original_provenance':entry,
                        'reason':'publication_date_is_not_observation_date',
                        'document_hash':digest, 'chunk_id':chunk_id,
                        'publication_quote':dates[0]['quote']})
        row[name] = None
        provenance.pop(name, None)
    if len(actions) != len(record.get('evidence_normalization_actions') or []):
        row['source_publication_date'] = publication
        row['field_provenance_json'] = (json.dumps(provenance, ensure_ascii=False)
                                        if isinstance(record.get('field_provenance_json'), str) else provenance)
        row['evidence_normalization_actions'] = actions
    return row


def qualify_records(records, *, contract, evidence_index):
    groups = {'qualified_records':[], 'candidate_records':[], 'rejected_evidence_records':[], 'record_qualifications':[], 'qualified_case_records':[], 'qualified_aggregate_records':[], 'qualified_context_records':[], 'unresolved_records':[]}
    from .outbreak_temporal import recover_closed_outbreak_observation
    pending = list(records)
    existing = {str(record.get('record_id')) for record in records}
    for record in pending:
        record = prepare_record_evidence(record, evidence_index=evidence_index)
        q = assess_record_evidence(record,contract=contract,evidence_index=evidence_index)
        qualification = q.to_dict()
        if any(entry.field == 'outbreak_closure_date' and entry.supported for entry in q.field_evidence):
            qualification['temporal_basis'] = 'outbreak_closure_event'
        from .outbreak_closure import closure_context_evidence
        closure = closure_context_evidence(record, qualification, evidence_index)
        if closure:
            qualification['closure_evidence'] = closure
        row = dict(record, evidence_qualification=qualification, product_kind=q.product_kind.value)
        groups['record_qualifications'].append(qualification)
        key = 'qualified_records' if q.status == 'qualified' else 'candidate_records' if q.status == 'candidate' else 'rejected_evidence_records'
        groups[key].append(row)
        if q.status == 'qualified':
            groups[{ProductKind.CASE_LEVEL:'qualified_case_records',ProductKind.AGGREGATE:'qualified_aggregate_records',ProductKind.CONTEXT:'qualified_context_records',ProductKind.UNRESOLVED:'unresolved_records'}[q.product_kind]].append(row)
        elif q.product_kind == ProductKind.UNRESOLVED:
            groups['unresolved_records'].append(row)
        if q.status == 'candidate' and not record.get('recovered_from_record_id'):
            recovered = recover_closed_outbreak_observation(record, evidence_index=evidence_index)
            if recovered and recovered['record_id'] not in existing:
                # Only a separately qualifying view is emitted. Failed repair
                # attempts do not multiply candidate records on every replay.
                repaired = assess_record_evidence(recovered, contract=contract, evidence_index=evidence_index)
                if repaired.status == 'qualified':
                    pending.append(recovered)
                    existing.add(recovered['record_id'])
    return groups


def qualified_coverage(requirements, records):
    """Evidence coverage counts only qualified, explicitly matching observations.

    Source fetching and arbitrary extracted rows cannot fill an evidence gap.
    """
    rows = []
    for requirement in requirements:
        known_metadata = {'requirement_id','year','week','date_hints','reporting_period_label','period_basis','time_granularity','source_type','official_domains','accepted_source_roles','strict_final_conditions','best_available_conditions','human_review_conditions','official_candidate_urls','agency','title_hints','reason','coverage_status','qualified_record_ids','unresolved_constraint_keys','covered_periods','uncovered_periods'}
        unknown_constraints = [k for k,v in requirement.items() if k not in _CONSTRAINT_KEYS and k not in known_metadata and _present(v)]
        rid = requirement.get("requirement_id")
        matched = []
        intervals = []
        for record in records:
            if unknown_constraints:
                continue
            q = record.get("evidence_qualification") or {}
            if q.get("status") != "qualified" or q.get("product_kind") not in {"case_level", "aggregate"}:
                continue
            if _constraint_reasons(record, requirement) or not any(_present(requirement.get(k)) for k in _CONSTRAINT_KEYS):
                continue
            if requirement.get('accepted_product_kinds') and q.get('product_kind') not in requirement['accepted_product_kinds']:
                continue
            matched.append(record.get("record_id"))
            left,right = _period(record)
            # A source-bound as-of cutoff bounds observed coverage, not admission.
            # Publication/report dates never supply this limit.
            if right and record.get('as_of_date'):
                try:
                    right = min(right,date.fromisoformat(str(record['as_of_date'])))
                except ValueError:
                    right = None
            if left and right:
                intervals.append((left.toordinal(),right.toordinal()))
        period_start = next((requirement.get(k) for k in ('period_start','reporting_period_start','metric_period_start','start_date','task_period_start') if requirement.get(k)),None)
        period_end = next((requirement.get(k) for k in ('period_end','reporting_period_end','metric_period_end','end_date','task_period_end') if requirement.get(k)),None)
        if requirement.get('year') is not None and not period_start and not period_end:
            period_start,period_end = str(requirement['year'])+'-01-01',str(requirement['year'])+'-12-31'
        if not period_start and not period_end:
            label_start,label_end = _period(requirement)
            if label_start and label_end:
                period_start,period_end = label_start.isoformat(),label_end.isoformat()
        complete = bool(matched)
        if period_start or period_end:
            try:
                start,end = date.fromisoformat(str(period_start)).toordinal(),date.fromisoformat(str(period_end)).toordinal()
                cursor = start
                for left,right in sorted(intervals):
                    if left > cursor:
                        break
                    cursor = max(cursor,right+1)
                complete = complete and start <= end and cursor > end
            except ValueError:
                complete = False
        rows.append({**requirement, "coverage_status":"covered" if complete else "evidence_gap", "qualified_record_ids":matched, "unresolved_constraint_keys":unknown_constraints})
    covered = sum(r['coverage_status'] == 'covered' for r in rows)
    return {"coverage_method":"evidence", "requirements":rows, "requirement_count":len(rows), "accepted_requirement_count":covered, "coverage_complete":bool(rows) and covered == len(rows), "coverage_status":"complete" if rows and covered == len(rows) else "not_evaluable" if not rows else "incomplete"}


_CONSTRAINT_KEYS = ('disease','country','subnational_location','locality','location','geography','reporting_period','start_date','end_date','task_period_start','task_period_end','period_start','period_end','reporting_period_start','reporting_period_end','metric_period_start','metric_period_end','accepted_metric_categories','accepted_metric_families','required_metric_fields','accepted_product_kinds')

def _period_label_bounds(label):
    """Compare source precision conservatively without writing inferred dates."""
    from calendar import monthrange
    from .source_assertions import _DATE, _INTERVAL, _date_parts, _fold, date_intervals

    token = str(label).strip()
    try:
        if re.fullmatch(r"\d{4}", token):
            return date(int(token), 1, 1), date(int(token), 12, 31)
        if re.fullmatch(r"\d{4}-\d{2}", token):
            year, month = map(int, token.split("-"))
            return date(year, month, 1), date(year, month, monthrange(year, month)[1])
        folded = _fold(token)
        if _INTERVAL.fullmatch(folded):
            intervals = date_intervals(token)
            if len(intervals) != 1:
                return None, None
            first, last = intervals[0]
        elif _DATE.fullmatch(folded):
            first = last = _date_parts(folded)
        else:
            return None, None
        if not first or not last or not first[0] or not last[0]:
            return None, None
        lower = date(first[0], first[1], first[2] or 1)
        upper = date(last[0], last[1], last[2] or monthrange(last[0], last[1])[1])
        return (lower, upper) if lower <= upper else (None, None)
    except (ValueError, TypeError):
        return None, None


def _period(record):
    start = next((record.get(k) for k in ('metric_period_start','period_start_date','event_start_date') if record.get(k)), None)
    end = next((record.get(k) for k in ('metric_period_end','period_end_date','event_end_date') if record.get(k)), None)
    if start or end:
        try:
            lower, upper = date.fromisoformat(str(start)), date.fromisoformat(str(end or start))
            return (lower, upper) if lower <= upper else (None, None)
        except ValueError:
            return None, None
    period = record.get('reporting_period')
    if _present(period):
        # A stated observation period cannot borrow a different reporting year.
        return _period_label_bounds(period)
    anchor = next((record.get(k) for k in ('as_of_date','date_onset','date_confirmation','date_reported','report_date') if record.get(k)), None)
    try:
        value = date.fromisoformat(str(anchor))
        return value, value
    except ValueError:
        return None, None


def _metric_families(record):
    families = set()
    for field in _NUMERIC:
        if not _present(record.get(field)):
            continue
        if field.startswith('cases_'):
            families.add('case_count')
        elif field == 'deaths':
            families.add('death_count')
        elif field in ('hospitalizations','icu_admissions'):
            families.add('hospitalization_count')
        elif field in ('tests_positive','tests_total'):
            families.add('lab_test_count')
        elif field in ('incidence_rate','positivity_rate'):
            families.add(field)
        elif field in ('cumulative_count','new_count'):
            families.add('public_health_metric')
        elif field == 'metric_value':
            if record.get('metric_category'):
                families.add(str(record['metric_category']))
            canonical = _canonical_metric(record.get('metric_name'), record.get('disease')) or str(record.get('metric_name') or '').casefold()
            if canonical in _NUMERIC and canonical != 'metric_value':
                families.update(_metric_families({canonical:record[field]}))
    return families


def _contract_location_matches(record, target):
    """Translate known territory names without inferring geographic hierarchy."""
    from .geography import explicit_country_key
    # An untyped location can name a country, state or locality. Keep literal
    # matching; explicit country/subnational/locality constraints disambiguate it.
    if str(target).casefold() in {str(record.get(k) or '').casefold() for k in _GEO}:
        return True
    target_key = explicit_country_key(target)
    scope = str(record.get('geographic_scope') or '').casefold()
    local_scope = str(record.get('geographic_scope_type') or '').casefold() in {
        'state','province','county','district','city','municipality','local','locality'}
    for field in ('country','subnational_location','geographic_scope'):
        value = record.get(field)
        if not _present(value):
            continue
        # A standard country name is not an alias for a homonymous local place.
        # A separate, broader subnational field can still identify a territory.
        if field != 'country' and local_scope and (
                field == 'geographic_scope' or not scope or str(value).casefold() == scope):
            continue
        if explicit_country_key(value) == target_key:
            return True
    return False


def _constraint_reasons(record, contract):
    from .geography import explicit_country_key
    reasons = []
    for metric in contract.get('required_metric_fields') or []:
        direct = metric in _NUMERIC and _present(record.get(metric))
        canonical = _canonical_metric(record.get('metric_name'), record.get('disease')) or str(record.get('metric_name') or '').casefold()
        generic = (metric in _NUMERIC and canonical == metric and _present(record.get('metric_value')))
        if not (direct or generic):
            reasons.append(str(metric)+':required_metric_missing')
    for key in ('disease','country','subnational_location','locality','reporting_period'):
        target = contract.get(key)
        matches = same_disease_name(target,record.get(key)) if key == 'disease' else str(target).casefold() == str(record.get(key) or '').casefold()
        if key == 'country':
            matches = explicit_country_key(target) == explicit_country_key(record.get(key))
        if _present(target) and not matches:
            reasons.append(key+':contract_mismatch')
    for key in ('location','geography'):
        target = contract.get(key)
        if _present(target) and not _contract_location_matches(record,target):
            reasons.append(key+':contract_mismatch')
    start,end = _period(record)
    # Closure is an independently verified event anchor for admitting a scoped
    # outbreak result. _period itself remains unchanged: coverage cannot turn
    # this date into a statistical interval or a full calendar-year total.
    if start is None and end is None and record.get('_validated_outbreak_closure_date'):
        try:
            start = end = date.fromisoformat(record['_validated_outbreak_closure_date'])
        except ValueError:
            pass
    if contract.get('year') is not None and (not start or not end or str(start.year) != str(contract['year']) or str(end.year) != str(contract['year'])):
        reasons.append('year:contract_mismatch')
    if contract.get('week') is not None and not any(contract.get(k) for k in ('period_start','reporting_period_start')):
        reasons.append('week:unresolved_constraint')
    for key in ('task','structured_task','scope','task_scope','time_window'):
        if isinstance(contract.get(key),dict):
            reasons.extend(_constraint_reasons(record,contract[key]))
    for key in ('start_date','task_period_start','period_start','reporting_period_start','metric_period_start'):
        if _present(contract.get(key)):
            try:
                if start is None or start < date.fromisoformat(str(contract[key])):
                    reasons.append(key+':contract_mismatch')
            except ValueError:
                reasons.append(key+':unresolved_constraint')
    for key in ('end_date','task_period_end','period_end','reporting_period_end','metric_period_end'):
        if _present(contract.get(key)):
            try:
                if end is None or end > date.fromisoformat(str(contract[key])):
                    reasons.append(key+':contract_mismatch')
            except ValueError:
                reasons.append(key+':unresolved_constraint')
    for key in ('accepted_metric_categories','accepted_metric_families'):
        if contract.get(key):
            if not _metric_families(record).intersection(set(contract[key])):
                reasons.append(key+':contract_mismatch')
    requirements = contract.get('requirements') or []
    if requirements and not any(not _constraint_reasons(record,r) for r in requirements):
        reasons.append('requirements:no_matching_evidence_scope')
    return reasons
