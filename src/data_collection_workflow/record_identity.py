"""Immutable observation identity from source positions and stated facts.

This is a key builder, not another record model. Workflow annotations and
extraction loop counters never distinguish scientific observations.
"""
from __future__ import annotations
import hashlib
import json
import math
import unicodedata


def _canonical(value):
    if isinstance(value, dict):
        return {str(key):_canonical(item) for key,item in sorted(value.items()) if item not in (None,'',[],{})}
    if isinstance(value, (list,tuple,set)):
        items=[_canonical(item) for item in value]
        return sorted(items,key=lambda item:json.dumps(item,sort_keys=True,ensure_ascii=True))
    if isinstance(value,float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value,str):
        return ' '.join(unicodedata.normalize('NFKC',value).split())
    return value


def record_identity_payload(row, chunk=None):
    """Return only source identity/position, claim scope, and patient facts."""
    from .evidence_qualification import _NUMERIC, _DATES, _GEO, _SUBSTANTIVE, _IDENTITY_FIELDS, _provenance
    from .source_assertions import normalized_quote_spans
    row=row.model_dump() if hasattr(row,'model_dump') else row
    chunk=chunk or {}
    facts={key:row.get(key) for key in ('disease',)+_NUMERIC+_DATES+_GEO+_SUBSTANTIVE+tuple(_IDENTITY_FIELDS)
           if row.get(key) not in (None,'',[],{})}
    quote=row.get('case_span_quote') or row.get('evidence_quote') or ''
    # This workflow field can be a generated display ordinal. It only supplies
    # patient identity when the source quote actually names that label.
    label=facts.get('workflow_case_label')
    if label and str(label).casefold() not in str(quote).casefold():
        facts.pop('workflow_case_label',None)
    provenance=_provenance(row)
    positions={}
    for name,entry in provenance.items():
        if name not in facts or not isinstance(entry,dict):
            continue
        positions[name]={key:entry.get(key) for key in ('document_hash','text_hash','char_start','char_end','page','table_id','row_id','json_pointer','quote','supporting_quote')
                         if entry.get(key) not in (None,'')}
    location={key:chunk.get(key) for key in ('document_hash','text_hash','page','table_id','row_id','json_pointer') if chunk.get(key) not in (None,'')}
    for key in ('case_span_start','case_span_end'):
        if row.get(key) is not None:
            location[key]=row[key]
    text=chunk.get('text') or ''
    start=chunk.get('char_start')
    spans=normalized_quote_spans(text,str(quote)) if text and quote else []
    if isinstance(start,int) and len(spans)==1:
        location['char_start']=start+spans[0][0]
        location['char_end']=start+spans[0][1]
    elif isinstance(start,int):
        location['char_start']=start
        location['char_end']=chunk.get('char_end')
    if not location and not any(positions.values()):
        location['chunk_id']=row.get('supporting_chunk_id') or row.get('chunk_id') or chunk.get('chunk_id')
    # Evidence span labels are only a conservative fallback when source offsets
    # are absent; regenerated display order cannot override actual positions.
    if not any(key in location for key in ('char_start','case_span_start')) and not any('char_start' in p for p in positions.values()):
        span=row.get('case_span_id') or row.get('evidence_span_id')
        if span:
            location['source_span_id']=span
    return _canonical({'version':1,'source_id':row.get('source_id') or chunk.get('source_id'),
        'source_url':None if row.get('source_id') or chunk.get('source_id') else row.get('source_url') or chunk.get('source_url'),
        'location':location,'field_positions':positions,'evidence_quote':quote,'facts':facts})


def stable_record_id(row, chunk=None):
    payload=record_identity_payload(row,chunk)
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=True,separators=(',',':')).encode()).hexdigest()
    return 'rec_evidence_'+digest[:32]
