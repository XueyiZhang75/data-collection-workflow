"""Exact source spans and bounded structural context for the evidence extractor."""
from __future__ import annotations

import hashlib
import json
import re

MAX_QUOTE_CHARS = 900


def _span(text, start, end, role, **metadata):
    return dict(role=role, quote=text[start:end], char_start=start, char_end=end, **metadata)


def _valid_span(text, span):
    start, end = span.get('char_start'), span.get('char_end')
    return isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)


def heading_spans(doc):
    text = str(doc.get('clean_text') or '')
    spans = []
    for locator in doc.get('locator_spans') or []:
        if locator.get('role') == 'heading' and _valid_span(text, locator):
            spans.append(_span(text, locator['char_start'], locator['char_end'], 'heading',
                               heading_level=int(locator.get('heading_level') or 1),
                               section_id=locator.get('section_id') or 'heading_' + str(locator['char_start'])))
    for match in re.finditer(r'(?m)^([ \t]*)(#{1,6})[ \t]+[^\r\n]+', text):
        start, end = match.span()
        if not any(s['char_start'] == start and s['char_end'] == end for s in spans):
            spans.append(_span(text, start, end, 'heading', heading_level=len(match.group(2)), section_id='markdown_' + str(start)))
    return sorted(spans, key=lambda s: (s['char_start'], s['char_end']))



def _heading_observation_status(chunk, documents=()):
    """Skip a verified topic heading, retaining it as context for its body.

    This is a routing guard, not a fact qualifier. Unknown phrasing and possible
    observations stay eligible; neither a year nor a figure number is a measure.
    """
    text = str(chunk.get('text') or '')
    heading_text = text if chunk.get('source_is_heading') else None
    for doc in documents:
        if chunk.get('source_id') and doc.get('source_id') != chunk['source_id']:
            continue
        if chunk.get('document_hash') and doc.get('content_hash') != chunk['document_hash']:
            continue
        spans = heading_spans(doc)
        matched = next((s for s in spans if s['char_start'] == chunk.get('char_start')
                        and s['char_end'] == chunk.get('char_end') and s['quote'] == text), None)
        if matched:
            # Inline emphasis splits one heading into several source locators.
            heading_text = ' '.join(s['quote'] for s in spans
                                    if s['section_id'] == matched['section_id'])
            break
    if heading_text is None:
        return None
    from .source_assertions import _DATE, analysis_projection, count_mentions
    if count_mentions(heading_text):
        return 'potential_observation'
    folded = analysis_projection(heading_text, fold=True).text
    # Named/singular patients and finite clinical events may be qualitative.
    if re.search(r'\b(?:a|one|first|index|une?|premier\w*)\b.{0,65}\b(?:patient\w*|case|cas)\b'
                 r'|\b(?:patient|case)\s+(?:no\.?\s*|#\s*)?[A-Z0-9]+\b'
                 r'|\b(?:was|were|has|had|developed|died|recovered|reported|notified)\b'
                 r'|\b(?:a ete|est|sont|ont ete|decede\w*|gueri\w*)\b', folded, re.I):
        return 'potential_observation'
    measurement = re.sub(r'^\s*(?:extended\s+data\s+)?(?:fig(?:ure)?\.?|table|chapter|section|report)'
                         r'\s*#?\s*\d+[a-z]?\s*[:.\-]?\s*', '', folded)
    measurement = _DATE.sub(' ', measurement)
    measurement = re.sub(r'\b(?:19|20)\d{2}\b', ' ', measurement)
    if re.search(r'(?<![\w.\-])\d+(?:[.,]\d+)?(?![\w.\-])', measurement):
        return 'potential_observation'
    # Restrict the negative decision to recognizable topic/section phrasing.
    # Unrecognized languages or wording remain candidates for extraction.
    topic = re.search(r'\b(?:clinical characterization|demographics|symptom data|'
                      r'situation report|surveillance|distribution|intra[- ]action review|'
                      r'outbreak response|disease severity|extended data|overview)\b', folded)
    return 'topic' if topic else 'unknown'

def _metadata_label(value):
    from .source_assertions import analysis_projection
    # Empty heading anchor links add no content; preserve other linked words.
    value = re.sub(r'\[\]\([^\n]*?\)', '', str(value or ''))
    return analysis_projection(value, collapse_whitespace=True, fold=True).text.strip(' #:.')


def _metadata_headings(chunk):
    return [_metadata_label(s.get('quote')) for s in chunk.get('bound_context_spans') or []
            if s.get('role') == 'heading']


def _pure_author_declaration(chunk):
    """Only the complete consent template is known administrative metadata.

    Prefixes, suffixes, unfamiliar declarations and mixed narratives stay at
    their original priority. This routing hint never changes qualification.
    """
    if (chunk.get('source_is_heading') or chunk.get('table_id') is not None
            or chunk.get('row_id') is not None or chunk.get('structured_data_kind')):
        return False
    if not any(h in {'author declarations', 'ethics declarations', 'ethical approval'}
               for h in _metadata_headings(chunk)):
        return False
    text = re.sub(r'\s+', ' ', str(chunk.get('text') or '')).strip()
    consent = (r'I confirm that all necessary patient/participant consent has been obtained and the appropriate '
               r'institutional forms have been archived, and that any patient/participant/sample identifiers '
               r'included were not known to anyone \(e\.g\., hospital staff, patients or participants themselves\) '
               r'outside the research group so cannot be used to identify individuals\.\s*(?:Yes\.?)?')
    return bool(re.fullmatch(consent, text, re.I))


def local_content_skip_reason(chunk):
    """Reject only non-evidence web units; inherited headings are not measurements.

    Tables/JSON and unfamiliar narrative remain eligible. Related content with
    its own numerical or clinical observation is not rejected just by section.
    """
    text = str(chunk.get('row_quote') or chunk.get('text') or '').strip()
    if (chunk.get('row_id') is not None or chunk.get('table_id') is not None
            or chunk.get('structured_data_kind') == 'json_record'
            or chunk.get('chunk_kind') in {'metric_row', 'metric_row_batch', 'table_row'}):
        return None
    encoded = text.strip(' \t\r\n()[]{}<>.,;:')
    if (len(encoded) >= 128 and re.fullmatch(r'[0-9a-fA-F]+', encoded)
            and re.search(r'[a-fA-F]', encoded) and re.search(r'[0-9]', encoded)):
        return 'encoded_payload_without_text'
    from .source_assertions import _DATE, analysis_projection, count_mentions
    def folded(value):
        return analysis_projection(value, collapse_whitespace=True, fold=True).text.strip(' #:.')
    navigation = re.compile(r'(?:post navigation|related (?:posts|articles|stories)|'
        r'recommended (?:reading|articles)|you may also like|latest news|recent posts|'
        r'navigation des articles|articles (?:connexes|associes)|a lire aussi|'
        r'ultimas noticias|articulos relacionados)', re.I)
    headings = [str(s.get('quote') or '') for s in chunk.get('bound_context_spans') or []
                if s.get('role') == 'heading']
    # Source headings may supply the disease context, but cannot turn citation
    # controls or author lookup widgets into observations. Match only explicit
    # interface shapes; unfamiliar narrative remains available to extraction.
    local = folded(text)
    if chunk.get('source_is_heading'):
        heading_unit = _metadata_label(chunk.get('source_heading_text') or text)
        if heading_unit in {'abstract', 'references', 'authors', 'affiliations', 'metrics', 'data availability'}:
            return 'structural_metadata_heading_without_observation'
        if (re.fullmatch(r'total views\s+\d[\d, ]*', heading_unit)
                and any(_metadata_label(quote) == 'metrics' for quote in headings)):
            return 'publication_view_count_without_observation'
    interface_line = re.compile(
        r'(?:copy|cite|collections?|add to collections?|permalink|actions|resources|'
        r'view on publisher site|open in a new tab|find articles by|author links open overlay panel|'
        r'similar articles|cited by other articles|'
        r'links to .+ databases|on this page|format|\.?nbib|'
        r'download\s+\.?nbib|pdf(?:\s*\(\d+(?:[.,]\d+)?\s*[kmgt]?b\))?)', re.I)
    lines = [folded(line) for line in text.splitlines() if line.strip()]
    citation_format = ('format' in lines or any(re.fullmatch(r'cite|citation(?: formats?)?', folded(quote)) for quote in headings))
    if lines and all(interface_line.fullmatch(line) or (citation_format and line in {'ama', 'apa', 'mla', 'nlm'}) for line in lines):
        return 'interface_controls_without_observation'
    administrative = re.compile(
        r'(?:use of artificial intelligence tools|conflicts? of interests?|'
        r'competing interests?|funding|disclosures?)', re.I)
    if (local in {'none declared', 'not applicable'}
            and any(administrative.fullmatch(folded(quote)) for quote in headings)):
        return 'administrative_declaration_without_observation'
    in_navigation = any(navigation.fullmatch(folded(quote)) for quote in headings)
    if not in_navigation:
        return None
    local = folded(text)
    if count_mentions(text) or re.search(
            r'\b(?:patient\w*|hospital\w*|symptom\w*|onset|fever|rash|'
            r'clinical|neurolog\w*|infection\w*|disease\w*|syndrome\w*|outbreak\w*|'
            r'epidem\w*|vaccin\w*|died|death\w*|recovered|discharged|'
            r'cas|deces|fievre|eruption|maladie\w*|casos|obitos)\b', local):
        return None
    # Do not let publication dates look like local measurements, or enforce an
    # English vocabulary on an unfamiliar-language narrative/heading.
    remainder = _DATE.sub(' ', local)
    remainder = re.sub(r'\b(?:19|20)\d{2}\b', ' ', remainder)
    if re.search(r'\d', remainder) or any(c.isalpha() and ord(c) > 127 for c in text):
        return None
    if chunk.get('source_is_heading'):
        return 'navigation_heading_without_observation'
    remainder = re.sub(r'(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/\S*)?', ' ', remainder)
    remainder = re.sub(r'\b(?:read more|no comments?|news|posted|by|continue reading|'
                       r'lire la suite|aucun commentaire|leer mas)\b', ' ', remainder)
    if not any(c.isalnum() for c in remainder):
        return 'navigation_metadata_without_observation'
    return None


def structural_heading_skip_reason(chunk, documents=()):
    if _heading_observation_status(chunk, documents) == 'topic':
        return 'structural_heading_without_observation'
    return None


def _active_headings(doc, start, headings=None):
    active = []
    for span in heading_spans(doc) if headings is None else headings:
        if span['char_start'] > start:
            break
        level = span['heading_level']
        if span['char_end'] > start:
            return [s for s in active if s['heading_level'] < level or s.get('section_id') == span.get('section_id')]
        # Inline pieces of the same HTML heading retain the same section.
        if not active or active[-1].get('section_id') != span.get('section_id'):
            active = [s for s in active if s['heading_level'] < level]
        active.append(span)
    return active


def _table_spans(doc):
    text = str(doc.get('clean_text') or '')
    by_table = {}
    for locator in doc.get('locator_spans') or []:
        if locator.get('table_id') is not None and _valid_span(text,locator):
            by_table.setdefault(str(locator['table_id']),[]).append(locator)
    for index, table in enumerate(doc.get('tables') or []):
        table_id = str(table.get('table_id') or table.get('table_index', index))
        headers = table.get('headers')
        rows = table.get('rows') or []
        if headers is None:
            headers, rows = (rows[0], rows[1:]) if rows else ([], [])
        locators = by_table.get(table_id,[])
        by_row = {str(s['row_id']):s for s in locators if s.get('row_id') is not None}
        header_ids = table.get('header_row_ids')
        header_spans = [by_row[str(row_id)] for row_id in header_ids or [] if str(row_id) in by_row]
        header_quote = ' | '.join(str(v) for v in headers)
        if not header_spans:
            header = next((s for s in locators if text[s['char_start']:s['char_end']] == header_quote), None)
            if header is None and header_quote:
                at = text.find(header_quote)
                if at >= 0 and text.find(header_quote, at + 1) < 0:
                    header = dict(char_start=at, char_end=at + len(header_quote))
            if header is None:
                continue
            header_spans = [header]
        header_quote = '\n'.join(text[s['char_start']:s['char_end']] for s in header_spans)
        context = [_span(text, s['char_start'], s['char_end'], 'table_header', table_id=table_id)
                   for s in header_spans]
        for locator in locators:
            if locator.get('table_part') in {'note', 'caption'}:
                context.append(_span(text, locator['char_start'], locator['char_end'],
                                     'table_' + locator['table_part'], table_id=table_id))
        context.sort(key=lambda span: span['role'] == 'table_header')
        search_start = max(header['char_end'] for header in header_spans)
        physical_ids = table.get('row_ids') or list(range(1, len(rows) + 1))
        spanning_cells = [cell for cell in table.get('cell_spans') or []
                          if not cell.get('is_header') and cell.get('row_end',0) > cell.get('row_id',0) + 1
                          and _valid_span(text,cell)]
        for row_number, row in zip(physical_ids, rows):
            quote = ' | '.join(str(v) for v in row)
            locator = by_row.get(str(row_number))
            if locator is not None and text[locator['char_start']:locator['char_end']] != quote:
                locator = None
            if locator is None:
                at = text.find(quote, search_start)
                if at < 0:
                    continue
                locator = dict(char_start=at, char_end=at + len(quote), row_id=row_number)
            search_start = locator['char_end']
            row_context = list(context)
            for cell in spanning_cells:
                if cell['row_id'] < row_number < cell['row_end']:
                    row_context.append(_span(text,cell['char_start'],cell['char_end'],
                                             'table_row_header',table_id=table_id))
            yield dict(text=quote, char_start=locator['char_start'], char_end=locator['char_end'],
                       table_id=table_id, row_id=str(locator.get('row_id', row_number)),
                       row_quote=quote, table_header=header_quote,
                       source_column_labels=list(table.get('column_labels') or headers),
                       row_context_type='source_table_row', context_spans=row_context)

    # Markdown has an explicit delimiter row; labels need not repeat per data row.
    lines = list(re.finditer(r'[^\r\n]+', text))
    for number, header in enumerate(lines[:-1]):
        if '|' not in header.group():
            continue
        labels = [cell.strip() for cell in header.group().strip('|').split('|')]
        separators = [cell.strip() for cell in lines[number+1].group().strip('|').split('|')]
        if len(labels) != len(separators) or not all(re.fullmatch(r':?-{3,}:?',cell) for cell in separators):
            continue
        if any(s.get('table_id') and s['char_start'] <= header.start() < s['char_end'] for s in doc.get('locator_spans') or []):
            continue
        table_id = 'markdown_table_' + str(header.start())
        context = [_span(text,header.start(),header.end(),'table_header',table_id=table_id)]
        for row_number,row in enumerate(lines[number+2:],1):
            quote = row.group()
            cells = [cell.strip() for cell in quote.strip('|').split('|')]
            if '|' not in quote or len(cells) != len(labels):
                break
            yield dict(text=quote,char_start=row.start(),char_end=row.end(),table_id=table_id,
                       row_id=str(row_number),row_quote=quote,table_header=header.group(),
                       source_column_labels=labels,row_context_type='markdown_table_row',context_spans=context)


def bound_context_for(doc, start, table_id=None):
    json_record = next((r for r in _json_document_records(doc) if r['char_start'] <= start < r['char_end']),None)
    if json_record is not None:
        return json_record['bound_context_spans']
    contexts = _active_headings(doc, start)
    if table_id is not None:
        row = next((r for r in _table_spans(doc) if r['table_id'] == str(table_id) and r['char_start'] <= start < r['char_end']), None)
        if row:
            contexts += row['context_spans']
    seen = set()
    result = []
    for span in contexts:
        key = (span['role'], span['char_start'], span['char_end'])
        if key not in seen:
            seen.add(key)
            result.append(span)
    return result


def validate_bound_context(doc, chunk, spans):
    """Resolve only parser/Markdown structural bindings, never supplied metadata."""
    text = str(doc.get('clean_text') or '')
    anchor = chunk.get('char_start')
    if not isinstance(anchor, int):
        return False
    expected = bound_context_for(doc, anchor, chunk.get('table_id'))
    for span in spans:
        if not isinstance(span, dict) or not _valid_span(text, span):
            return False
        if text[span['char_start']:span['char_end']] != span.get('quote'):
            return False
        if not any(all(span.get(k) == item.get(k) for k in ('role','quote','char_start','char_end','table_id','section_id')) for item in expected):
            return False
    return True


def combine_bound_quote(contexts, quote):
    return '\n'.join([span['quote'] for span in contexts if span['quote'] != quote] + [quote])


def quote_within_chunk(doc, chunk, start, end, contexts):
    text = str(doc.get('clean_text') or '')
    left, right = chunk.get('char_start'), chunk.get('char_end')
    if not (isinstance(left,int) and isinstance(right,int) and 0 <= left < right <= len(text)):
        return False
    if text[left:right] != chunk.get('text'):
        return False
    return left <= start < end <= right or any(s['char_start'] <= start < end <= s['char_end'] for s in contexts)


def source_unit_for_quote(doc, chunk, start, end):
    """Expand a literal field citation within its own sentence/row/object only."""
    text = str(doc.get('clean_text') or '')
    left, right = chunk.get('char_start'), chunk.get('char_end')
    if not (isinstance(left,int) and isinstance(right,int) and left <= start < end <= right):
        return start,end
    if chunk.get('row_id') or chunk.get('structured_data_kind') == 'json_record':
        return left,right
    from .source_assertions import sentence_bounds
    local_start, local_end = sentence_bounds(text[left:right], start-left, end-left, semicolons=True)
    return left+local_start, left+local_end


_JSON_SCOPE_KEYS = {'disease','disease_standard_name','country','subnational_location','locality','geographic_scope',
                    'reporting_period','year','period_start_date','period_end_date','metric_period_start','metric_period_end',
                    'as_of_date','population_scope','case_definition','metric_unit','unit','count_unit','count_semantics','statistical_count_type'}


def _json_records(text):
    decoder = json.JSONDecoder()
    from .evidence_qualification import _NUMERIC, _canonical_metric
    def walk(start, path, inherited):
        while start < len(text) and text[start].isspace():
            start += 1
        value, end = decoder.raw_decode(text,start)
        if isinstance(value,dict):
            position = start + 1
            children, contexts, metrics, descriptors = [], list(inherited), [], []
            for key,child in value.items():
                while text[position].isspace() or text[position] == ',':
                    position += 1
                key_start = position
                _,position = decoder.raw_decode(text,position)
                while text[position].isspace() or text[position] == ':':
                    position += 1
                child_start = position
                _,position = decoder.raw_decode(text,position)
                pointer = path + '/' + str(key).replace('~','~0').replace('/','~1')
                if isinstance(child,(dict,list)):
                    children.append((child_start,pointer))
                else:
                    field = str(key).casefold()
                    if field in _JSON_SCOPE_KEYS:
                        canonical = 'reporting_period' if field == 'year' else field
                        contexts = [s for s in contexts if s.get('json_field') != canonical]
                        contexts.append(_span(text,key_start,position,'json_scope',json_pointer=pointer,json_field=canonical))
                    if field in {'metric_name','metric_category'}:
                        descriptors.append(_span(text,key_start,position,'json_scope',json_pointer=pointer,json_field=field))
                    if (_canonical_metric(field) or field) in _NUMERIC:
                        metrics.append((key_start,position,pointer))
            if value and not children:
                # Only ancestor scope is inherited; a leaf's own fields remain in
                # its exact object, where they override wider enclosing labels.
                yield dict(text=text[start:end],char_start=start,char_end=end,json_pointer=path,
                           structured_data_kind='json_record',bound_context_spans=inherited)
            else:
                # Parent totals remain separate observations; their numeric values
                # are never inherited by nested regional/patient breakdown rows.
                for left,right,pointer in metrics:
                    yield dict(text=text[left:right],char_start=left,char_end=right,json_pointer=pointer,
                               structured_data_kind='json_record',bound_context_spans=contexts + descriptors)
                for position,pointer in children:
                    yield from walk(position,pointer,contexts)
        elif isinstance(value,list):
            position = start + 1
            for index,child in enumerate(value):
                while text[position].isspace() or text[position] == ',':
                    position += 1
                if isinstance(child,(dict,list)):
                    yield from walk(position,path+'/'+str(index),inherited)
                _,position = decoder.raw_decode(text,position)
    try:
        yield from walk(0,'',[])
    except (ValueError,IndexError):
        return


def _json_document_records(doc):
    text = str(doc.get('clean_text') or '')
    roots = [(0,len(text))] if doc.get('document_type') == 'json' else [
        (s['char_start'],s['char_end']) for s in doc.get('locator_spans') or []
        if s.get('structured_data_kind') == 'json_document' and _valid_span(text,s)]
    for start,end in roots:
        for record in _json_records(text[start:end]):
            record['char_start'] += start; record['char_end'] += start
            record['bound_context_spans'] = [{**s,'char_start':s['char_start']+start,'char_end':s['char_end']+start}
                                             for s in record['bound_context_spans']]
            yield record


def json_fields_for_chunk(doc,chunk):
    from .evidence_qualification import _canonical_metric
    def canonical(values):
        return {(_canonical_metric(key) or ('reporting_period' if str(key).casefold() == 'year' else str(key).casefold())):value
                for key,value in values.items()}
    fields = {}
    for span in chunk.get('bound_context_spans') or []:
        if span.get('role') == 'json_scope':
            fields.update(canonical(json.loads('{'+span['quote']+'}')))
    try:
        own = json.loads(chunk['text'])
    except ValueError:
        own = json.loads('{'+chunk['text']+'}')
    return {**fields,**canonical(own)} if isinstance(own,dict) else {}


def exact_chunk_specs(doc):
    """Yield exact main spans with independently resolvable context spans."""
    text = str(doc.get('clean_text') or '')
    json_records = list(_json_document_records(doc))
    for record in json_records:
        yield dict(record,chunk_kind='text')
    if json_records and doc.get('document_type') == 'json':
        return
    headings = heading_spans(doc)
    for heading in headings:
        contexts = _active_headings(doc,heading['char_start'],headings)
        yield dict(text=heading['quote'],char_start=heading['char_start'],char_end=heading['char_end'],
                   chunk_kind='text',bound_context_spans=contexts,source_is_heading=True,
                   source_heading_text=' '.join(s['quote'] for s in headings if s['section_id'] == heading['section_id']),
                   heading_context='\n'.join(s['quote'] for s in contexts) or None)
    table_rows = list(_table_spans(doc))
    for row in table_rows:
        contexts = _active_headings(doc,row['char_start'],headings) + row['context_spans']
        yield {k:v for k,v in row.items() if k != 'context_spans'} | dict(
            chunk_kind='metric_row', bound_context_spans=contexts,
            heading_context='\n'.join(s['quote'] for s in contexts if s['role'] == 'heading') or None)
    # Keep tables out of narrative chunks: another row cannot provide its value.
    occupied = [(s['char_start'],s['char_end']) for s in headings]
    occupied += [(s['char_start'],s['char_end']) for s in doc.get('locator_spans') or [] if s.get('structured_data_kind') == 'json_document']
    occupied += [(s['char_start'],s['char_end']) for s in doc.get('locator_spans') or []
                 if s.get('table_id') and _valid_span(text,s)]
    occupied += [(r['char_start'],r['char_end']) for r in table_rows]
    occupied += [(s['char_start'],s['char_end']) for r in table_rows for s in r['context_spans']]
    cursor = 0
    gaps = []
    for start,end in sorted(occupied):
        if start > cursor:
            gaps.append((cursor,start))
        cursor = max(cursor,end)
    if cursor < len(text):
        gaps.append((cursor,len(text)))
    if doc.get('document_type') == 'pdf':
        pages = [span for span in doc.get('locator_spans') or [] if span.get('page') and _valid_span(text,span)]
        if pages:
            gaps = [(max(start,page['char_start']),min(end,page['char_end'])) for start,end in gaps for page in pages
                    if max(start,page['char_start']) < min(end,page['char_end'])]
    for start,end in gaps:
        while start < end:
            while start < end and text[start].isspace():
                start += 1
            if start >= end:
                break
            contexts = _active_headings(doc,start,headings)
            context_length = sum(len(s['quote']) + 1 for s in contexts)
            limit = max(80, MAX_QUOTE_CHARS - context_length)
            stop = min(end,start + limit)
            if stop < end:
                from .source_assertions import sentence_spans
                # Look just beyond the limit so "Jan. 9" is recognized as one
                # date even when its abbreviation lands at the chunk boundary.
                boundaries = [start+right for _,right in sentence_spans(text[start:min(end,stop+12)])
                              if start+right <= stop]
                boundary = max(boundaries + [text.rfind('\n',start,stop)])
                if boundary > start:
                    stop = boundary
            while stop > start and text[stop-1].isspace():
                stop -= 1
            if stop <= start:
                break
            yield dict(text=text[start:stop], char_start=start, char_end=stop, chunk_kind='text',
                       bound_context_spans=contexts,
                       heading_context='\n'.join(s['quote'] for s in contexts if s['role'] == 'heading') or None)
            start = stop


def build_evidence_chunks(state):
    # Reuse workflow policy/relevance and metadata; change only evidence slicing.
    from .nodes import content_processing as cp
    policy = cp.EvidenceChunkingPolicy(**cp.load_evidence_chunking_policy())
    context = cp.build_disease_relevance_context(state)
    chunks, assessments, skipped = [], [], {}
    documents = state.get('documents') or []
    for doc in documents:
        usable, reason = cp._document_is_chunkable(doc,policy)
        if not usable:
            skipped[reason] = skipped.get(reason,0) + 1
            continue
        for position, spec in enumerate(exact_chunk_specs(doc),1):
            bound = spec.get('bound_context_spans') or []
            evidence_text = '\n'.join([s['quote'] for s in bound] + [spec['text']])
            if cp._is_context_only_document(doc):
                contains, types, context_types, confidence, why, _ = cp._flag_context_only_presence(evidence_text,policy)
            else:
                contains, types, context_types, confidence, why = cp._flag_data_presence(evidence_text,doc,policy)
            if not contains and not cp._is_context_only_document(doc):
                metric_types = cp._generic_metric_row_data_types(evidence_text) if spec.get('row_id') else []
                if metric_types:
                    contains, types, confidence, why = True, metric_types, 0.70, 'explicit source metric row'
                named_patient = re.search(r'\b(?:patient|case)\s+(?:no\.?\s*|#\s*)?[A-Za-z0-9]+\b',spec['text'],re.I)
                clinical = re.search(r'\b(?:aged?\s*[:=]?\s*\d+|\d+[- ]years?[- ]old|hospitali[sz]ed|symptoms?|onset|died|recovered)\b',spec['text'],re.I)
                if named_patient and clinical:
                    contains, types, confidence, why = True, ['case_level_detail'], 0.70, 'explicit source patient and clinical detail'
            heading_status = _heading_observation_status(spec, [doc])
            if heading_status == 'topic':
                contains, types, why = False, [], 'structural_heading_without_observation'
            elif heading_status == 'potential_observation' and not cp._is_context_only_document(doc):
                contains, confidence, why = True, max(confidence, 0.70), 'source heading contains a possible observation'
            if heading_status == 'potential_observation':
                # Presence was assessed above; apply the same disease identity
                # contract without a second, narrower generic signal lexicon.
                from .disease_relevance import assess_text_disease_relevance
                relevance = assess_text_disease_relevance(evidence_text, context,
                    require_data_signal=False, evidence_fields_used=['text', 'bound_context_spans'])
            else:
                relevance = cp.assess_chunk_disease_relevance({'text':evidence_text},context)
            noise_reason = local_content_skip_reason(spec)
            if noise_reason:
                contains, types, why = False, [], noise_reason
            eligible = bool(contains and relevance.get('status') == cp.TARGET_DISEASE_MATCH)
            row = cp._make_evidence_chunk(doc=doc, chunk_index=position, chunk_kind=spec['chunk_kind'],
                text=spec['text'], char_start=spec['char_start'],char_end=spec['char_end'],
                table_id=spec.get('table_id'), row_id=spec.get('row_id'),row_quote=spec.get('row_quote'),
                table_header=spec.get('table_header'),heading_context=spec.get('heading_context'),
                source_column_labels=spec.get('source_column_labels'),row_context_type=spec.get('row_context_type'),
                contains_target_data=eligible,data_types=types if eligible else [],context_types=context_types,
                confidence=confidence,presence_reason=why,disease_assessment=relevance,
                extraction_eligible_for_task_disease=eligible).model_dump()
            row.update(source_is_heading=bool(spec.get('source_is_heading')),
                       bound_context_spans=bound, json_pointer=spec.get('json_pointer'),
                       structured_data_kind=spec.get('structured_data_kind'))
            if spec.get('source_is_heading'):
                row['source_heading_text'] = spec.get('source_heading_text') or spec['text']
            row['normalized_text_hash'] = hashlib.sha256(spec['text'].encode()).hexdigest()
            row['semantic_span_type'] = cp._semantic_span_type(spec['chunk_kind'],spec['text'])
            if _pure_author_declaration(spec):
                row['extraction_priority'] = max(row.get('extraction_priority') or 0, 4)
                row['semantic_span_type'] = 'context'
                row['presence_reason'] += '; lower priority: pure_author_declaration'
            row['span_type'] = row['semantic_span_type']
            chunks.append(row)
            assessments.append(dict(chunk_id=row['chunk_id'],source_id=row['source_id'],
                                    decision_owner='source_bound_chunk_relevance',contains_target_data=eligible,
                                    extraction_eligible_for_task_disease=eligible,disease_relevance_status=relevance.get('status'),
                                    presence_reason=why,data_types=row['data_types'],context_types=context_types))
    summary = dict(input_document_count=len(documents),chunkable_document_count=len(documents)-sum(skipped.values()),
                   skipped_document_count=sum(skipped.values()),skip_reason_counts=skipped,total_chunk_count=len(chunks),
                   text_chunk_count=sum(c['chunk_kind']=='text' for c in chunks),
                   table_chunk_count=sum(c['chunk_kind']=='metric_row' for c in chunks),
                   markdown_metric_row_chunk_count=0,chunking_method='exact_source_spans_evidence')
    presence = dict(total_chunk_count=len(chunks),target_data_chunk_count=sum(c['contains_target_data'] for c in chunks))
    trace = cp.append_trace(state,node_name='evidence_chunking_and_data_presence_flagging',
                            message=f"Created {len(chunks)} exact source chunks.",metadata=summary)
    return dict(evidence_chunks=chunks,evidence_chunking_summary=summary,data_presence_summary=presence,
                chunk_relevance_assessments=assessments,
                disease_relevance_summary=cp.update_disease_relevance_summary({**state,'evidence_chunks':chunks}),
                collection_trace=trace)
