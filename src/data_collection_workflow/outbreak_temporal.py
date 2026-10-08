"""Source-local relative dates for an explicitly declared outbreak closure."""

from datetime import date, timedelta
import hashlib
import json
import re

from .source_assertions import _DATE, _date_parts, count_mentions, normalized_quote_spans, publication_dates, sentence_spans

_WEEKDAYS = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
_RELATIVE_DECLARATION = re.compile(
    r'\b(?:declaration|announcement|declared|announced)\b[^.!?\n]{0,70}?'
    r'\bon\s+(?P<weekday>' + '|'.join(_WEEKDAYS) + r')\b', re.I)
_UNCERTAIN_TIME = re.compile(
    r"\b(?:if|might|may|could|would|will|expects?|expected|plans?|planned|"
    r"previous|earlier|historical|formerly|premature|falsely|rumou?r|not|never|"
    r"exercise|scenario|simulation|hypothetical)\b|\b(?:last|past)\s+(?:year|month|week)\b|"
    r'\b(?:weeks?|months?|years?)\s+ago\b', re.I)


def resolve_outbreak_closure_date(record, *, evidence_index):
    """Derive a declaration date, never a statistical cutoff or start year.

    A news report's unqualified past weekday is interpreted as the most recent
    matching weekday on or before its explicitly labelled publication date.
    Both source anchors and this calendar rule remain available for audit.
    """
    from .evidence_qualification import _DATES, _NUMERIC, _canonical_metric, _local_span, _provenance, _supports
    from .task_result_report import _completed_outbreak_sources

    if not record.get('country') or not record.get('disease'):
        return None
    chunks = evidence_index.get('evidence_chunks') or {}
    if isinstance(chunks, list):
        chunks = {str(c.get('chunk_id')):c for c in chunks}
    chunk_id = next((record.get(k) for k in ('chunk_id','evidence_chunk_id','supporting_chunk_id') if record.get(k)), None)
    chunk = chunks.get(str(chunk_id)) or {}
    text = str(chunk.get('text') or '')
    if not text or chunk.get('table_id') or chunk.get('source_is_heading'):
        return None
    quote, digest, locator, error = _local_span(record, {'quote':text}, evidence_index)
    if error:
        return None
    publications = publication_dates(text)
    if len({p['value'] for p in publications}) != 1:
        return None
    declared = list(_RELATIVE_DECLARATION.finditer(text))
    if len(declared) != 1:
        return None
    declaration = declared[0]
    sentences = list(sentence_spans(text))
    sentence_index = next((i for i,(a,b) in enumerate(sentences)
                           if a <= declaration.start() < declaration.end() <= b), None)
    if sentence_index is None:
        return None
    left, right = sentences[sentence_index]
    declaration_quote = text[left:right]
    if (_UNCERTAIN_TIME.search(declaration_quote) or _DATE.search(declaration_quote)
            or re.search(r'\banniversary\b[^.!?]{0,40}\b(?:declaration|announcement)\b', declaration_quote, re.I)):
        return None
    # An explicit historical year cannot inherit the current publication year.
    published = date.fromisoformat(publications[0]['value'])
    if re.search(r'(?<!\d)(?:19|20)\d{2}(?!\d)', declaration_quote):
        return None
    def closure_statement(statement):
        declaration = re.search(r'\b(?:declared|announced)\b[^.!?\n]{0,100}'
            r'(?:\bend\b[^.!?\n]{0,40}\boutbreak\b|\boutbreak\b[^.!?\n]{0,40}\b(?:over|ended|closed)\b)', statement, re.I)
        return bool(declaration and not _UNCERTAIN_TIME.search(statement)
                    and _supports('disease', record['disease'], statement, record)
                    and _supports('country', record['country'], statement, record))
    paired_start = left
    if not closure_statement(declaration_quote):
        if (not sentence_index or not re.match(r'\s*(?:the|this)\s+(?:declaration|announcement)\s+on\s+', declaration_quote, re.I)
                or not closure_statement(text[sentences[sentence_index-1][0]:sentences[sentence_index-1][1]])):
            return None
        paired_start = sentences[sentence_index-1][0]
    closure_end = left if paired_start != left else right
    # HTML metadata lines are not grammatical subjects of the declaration.
    # Retain the shortest literal suffix that still contains the full matching
    # closure statement; wrapped subjects remain intact when needed.
    starts = [paired_start] + [paired_start+m.end() for m in re.finditer(r'\n', text[paired_start:closure_end])]
    valid_starts = [at for at in starts if closure_statement(text[at:closure_end])]
    if valid_starts:
        # A wrapped historical qualifier is part of the statement, not page
        # metadata. Only standalone copies of the verified publication date
        # (optionally with a clock time) may be removed from this temporal check.
        prefix = text[paired_start:max(valid_starts)]
        for line in prefix.splitlines():
            match = _DATE.search(line)
            if (match and _date_parts(match.group()) == (published.year, published.month, published.day)
                    and re.fullmatch(r'\s*(?:(?:publish|published|publication)\s*[:\-]?\s*)?', line[:match.start()], re.I)
                    and re.fullmatch(r'[\s,.]*(?:\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)?[\s.]*', line[match.end():], re.I)):
                continue
            if (_DATE.search(line) or _UNCERTAIN_TIME.search(line)
                    or re.search(r'(?<!\d)(?:19|20)\d{2}(?!\d)', line)):
                return None
        paired_start = max(valid_starts)
    paired_quote = text[paired_start:right]
    if (_DATE.search(paired_quote) or _UNCERTAIN_TIME.search(paired_quote)
            or re.search(r'\b(?:in|during|since|from|before|after)\s+'
                         r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\b', paired_quote, re.I)):
        return None
    paired_locator = {key:value for key,value in locator.items() if key != 'bound_context_spans'}
    paired_locator.update(char_start=chunk['char_start']+paired_start, char_end=chunk['char_start']+right)
    weekday = next(day for day in _WEEKDAYS if day.casefold() == declaration['weekday'].casefold())
    resolved = published - timedelta(days=(published.weekday()-_WEEKDAYS.index(weekday)) % 7)
    source = {key:value for key,value in record.items() if key not in _DATES}
    provenance = _provenance(record)
    fields = []
    numbers = [name for name in _NUMERIC if record.get(name) is not None]
    closure_fields = [name for name in numbers if name in {'cases_confirmed','cases_probable','cases_suspected','cases_unspecified','deaths','hospitalizations'}]
    if not closure_fields:
        return None
    for name in numbers:
        entry = provenance.get(name) or {}
        if not isinstance(entry, dict) or entry.get('basis') in ('task','search','query','inferred_from_task') or entry.get('source') in ('task','search'):
            return None
        if any(key in entry for key in ('char_start','char_end')):
            quoted = next((str(entry[key]) for key in ('quote','evidence_quote','supporting_quote') if entry.get(key)), '')
            positions = normalized_quote_spans(text, quoted) if quoted else []
            if not any(all(type(entry[key]) is int and entry[key] == chunk['char_start']+position
                           for key,position in (('char_start',a),('char_end',b)) if key in entry)
                       for a,b in positions):
                return None
        local, local_hash, local_locator, local_error = _local_span(record, entry, evidence_index)
        if local_error or local_hash != digest or local_locator.get('chunk_id') != chunk_id:
            return None
        scope = '\n'.join(s['quote'] for s in local_locator.get('bound_context_spans') or []
                          if s.get('role') in {'heading','table_caption','table_note'})
        supported_record = {**source, '_validated_bound_scope':scope}
        if not _supports(name, record[name], local, supported_record):
            return None
        if not _supports('count_semantics', 'cumulative', local, supported_record):
            return None
        body = local[len(scope)+1:] if scope and local.startswith(scope+'\n') else local
        canonical = _canonical_metric(record.get('metric_name'), record.get('disease')) if name == 'metric_value' else name
        candidates = {mention['sentence'] for mention in count_mentions(body)
                      if mention['field'] == canonical and mention['value'] == record[name]}
        if len(candidates) != 1:
            return None
        source_quote = next(iter(candidates))
        local, local_hash, local_locator, local_error = _local_span(record,
            {'quote':source_quote, 'chunk_id':chunk_id, 'document_hash':digest}, evidence_index)
        if local_error:
            return None
        fields.append({'field':name, 'value':record[name], 'document_hash':digest,
                       'locator':local_locator, 'quote':local, 'source_quote':source_quote,
                       'supported':True, 'reason':''})
    closure_field = {'field':'outbreak_closure_date', 'value':resolved.isoformat(),
                     'document_hash':digest, 'locator':paired_locator, 'quote':paired_quote, 'supported':True, 'reason':''}
    provisional = {**source, 'outbreak_closure_date':resolved.isoformat(), 'count_semantics':'cumulative',
                   'evidence_qualification':{'status':'qualified', 'product_kind':'aggregate',
                       'field_evidence':fields+[closure_field],
                       'closure_evidence':[{'document_hash':digest, 'locator':paired_locator, 'quote':paired_quote, 'supported':True}]}}
    task = {'disease':record['disease'], 'location':record['country'],
            'start_date':resolved.isoformat(), 'end_date':resolved.isoformat()}
    window = {'period_start':None, 'period_end':None, 'as_of_date':None}
    for name in closure_fields:
        matched = _completed_outbreak_sources(provisional, task, name, window)
        if not matched:
            return None
        if not any(declaration_quote.strip() in (source.get('closure_statement_quote') or '')
                   or declaration_quote.strip() in (source.get('closure_attribution_quote') or '')
                   for source in matched):
            return None
    offset = chunk['char_start']
    publication = publications[0]
    anchors = [dict(role='publication_date', quote=publication['quote'],
                    char_start=offset+publication['char_start'], char_end=offset+publication['char_end']),
               dict(role='closure_declaration_weekday', quote=text[declaration.start():declaration.end()],
                    char_start=offset+declaration.start(), char_end=offset+declaration.end())]
    return {'field':'outbreak_closure_date', 'value':resolved.isoformat(),
            'basis':'derived_relative_date', 'rule':'previous_weekday_from_publication_date',
            'date_role':'closure_event', 'publication_date':published.isoformat(), 'weekday':weekday,
            'anchors':anchors, 'document_hash':digest, 'locator':paired_locator, 'quote':paired_quote,
            'numeric_fields':numbers, 'numeric_evidence':fields}


def recover_closed_outbreak_observation(record, *, evidence_index):
    """Create a separately auditable observation without rewriting its parent."""
    from .evidence_qualification import _DATES, _provenance

    if record.get('recovered_from_record_id') or not record.get('record_id'):
        return None
    if any(record.get(name) for name in ('case_id','patient_id','case_label','case_identifier','workflow_case_label')):
        return None
    proof = resolve_outbreak_closure_date(record, evidence_index=evidence_index)
    if proof is None:
        return None
    row = dict(record)
    provenance = dict(_provenance(record))
    removed_dates = {key:{'value':record[key], 'provenance':provenance.get(key)}
                     for key in _DATES if record.get(key) is not None}
    for key in _DATES + ('date_anchor','date_anchor_field','date_text_span'):
        if row.get(key) is not None:
            row.setdefault(key+'_raw', row[key])
        row[key] = None
        provenance.pop(key, None)
    for key in ('evidence_qualification','record_inclusion_decision','product_kind'):
        row.pop(key, None)
    anchor = proof['numeric_evidence'][0]
    for numeric in proof['numeric_evidence']:
        provenance[numeric['field']] = {'quote':numeric['source_quote'],
            'chunk_id':numeric['locator']['chunk_id'], 'document_hash':numeric['document_hash']}
    for name in ('count_semantics','statistical_count_type'):
        if not row.get(name):
            row[name] = 'cumulative'
            provenance[name] = {'quote':anchor['source_quote'],
                                'chunk_id':anchor['locator']['chunk_id'], 'document_hash':anchor['document_hash']}
    row['field_provenance_json'] = (json.dumps(provenance, ensure_ascii=False)
                                    if isinstance(record.get('field_provenance_json'), str) else provenance)
    row['outbreak_closure_date'] = proof['value']
    row['outbreak_closure_derivation'] = proof
    row['recovered_from_record_id'] = record['record_id']
    row['evidence_recovery'] = {'method':'source_local_closed_outbreak_observation',
                                'original_record_id':record['record_id'], 'excluded_temporal_claims':removed_dates,
                                'statistical_period_start_known':False, 'statistical_period_end_known':False}
    payload = json.dumps({'parent':record['record_id'], 'proof':proof}, sort_keys=True, ensure_ascii=True)
    row['record_id'] = 'rec_outbreak_' + hashlib.sha256(payload.encode()).hexdigest()[:24]
    return row
