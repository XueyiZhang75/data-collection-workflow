"""Resolve short closure context from the same source unit as a qualified count."""

import re


def closure_context_evidence(record, qualification, evidence_index):
    """Preserve verified context; the task report decides its closure semantics.

    No adjacent document, sibling chunk or free-text title can contribute here.
    The count's prior qualification and the document hash are checked again.
    """
    from .evidence_qualification import _local_span

    if qualification.get('status') != 'qualified':
        return []
    chunks = evidence_index.get('evidence_chunks') or {}
    if isinstance(chunks, list):
        chunks = {c.get('chunk_id'): c for c in chunks}
    result, seen = [], set()
    counts = {'cases_confirmed', 'cases_probable', 'cases_suspected',
              'cases_unspecified', 'deaths', 'hospitalizations'}
    for field in qualification.get('field_evidence') or []:
        if (field.get('field') not in counts or field.get('supported') is not True
                or field.get('value') != record.get(field['field'])):
            continue
        locator = field.get('locator') or {}
        chunk_id = locator.get('chunk_id')
        chunk = chunks.get(chunk_id) or {}
        text = str(chunk.get('text') or '')
        if (chunk_id in seen or chunk.get('table_id') or chunk.get('source_is_heading')
                or not re.search(r'\boutbreak\b', text, re.I)
                or not re.search(r'\b(?:ended|over|closed|declared)\b', text, re.I)):
            continue
        start, end = chunk.get('char_start'), chunk.get('char_end')
        left, right = locator.get('char_start'), locator.get('char_end')
        if not all(isinstance(v, int) for v in (start, end, left, right)):
            continue
        if not start <= left < right <= end:
            continue
        entry = {'chunk_id': chunk_id, 'document_hash': field.get('document_hash'),
                 'quote': text, 'char_start': start}
        quote, digest, resolved, error = _local_span(record, entry, evidence_index)
        if error or digest != field.get('document_hash'):
            continue
        result.append({'document_hash': digest, 'locator': resolved,
                       'quote': quote, 'supported': True})
        seen.add(chunk_id)
    return result
