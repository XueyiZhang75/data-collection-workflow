"""Pure, deterministic products of qualified evidence; no task/profile inference.

Qualification remains the semantic authority. This layer rechecks its field values,
content hashes and locators, and never promotes unqualified input. Unknown scope is
retained but cannot participate in an automatic aggregate derivation.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from urllib.parse import urlparse

from .evidence_qualification import supports_patient_relation, supports_partition_relation

IDENTITIES = ('patient_id', 'case_id', 'case_label', 'case_identifier', 'workflow_case_label')
METRICS = ('cases_confirmed', 'cases_probable', 'cases_suspected', 'cases_unspecified', 'deaths', 'hospitalizations', 'icu_admissions', 'tests_positive', 'tests_total', 'cumulative_count', 'new_count', 'metric_value', 'incidence_rate', 'positivity_rate')
SCOPE = ('disease', 'country', 'subnational_location', 'locality', 'geographic_scope', 'geographic_scope_type', 'reporting_period', 'period_start_date', 'period_end_date', 'event_start_date', 'event_end_date', 'metric_period_start', 'metric_period_end', 'case_definition', 'population_scope', 'unit', 'count_basis', 'statistical_count_type', 'metric_denominator', 'count_semantics', 'count_unit', 'metric_name', 'metric_unit', 'metric_category', 'denominator_scope', 'population_denominator')
EVENT_DATES = ('date_onset', 'onset_date', 'date_confirmation', 'confirmation_date', 'date_death', 'date_reported', 'report_date')


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), default=str)


def _id(prefix, value):
    return prefix + hashlib.sha256(_json(value).encode()).hexdigest()[:24]


def _unique(items):
    return [v for _, v in sorted({_json(v): v for v in items}.items())]


def _documents(index):
    docs = index.get('documents') or []
    return _unique(list(docs.values()) if isinstance(docs, dict) else docs)


def _resolved(evidence, index, *, documents=None):
    if evidence.get('supported') is not True:
        return False
    if documents is None:
        documents = _documents(index)
    matches = [d for d in documents if d.get('content_hash') == evidence.get('document_hash')]
    locator = evidence.get('locator') or {}
    chunks = index.get('evidence_chunks') or index.get('spans') or {}
    if isinstance(chunks, list):
        chunks = {str(c.get('chunk_id') or c.get('evidence_span_id')): c for c in chunks}
    chunk = chunks.get(str(locator.get('chunk_id'))) or {}
    if not chunk or not matches:
        return False
    for doc in matches:
        requested_id = locator.get('document_id') or chunk.get('document_id')
        if requested_id and doc.get('document_id') != requested_id:
            continue
        if chunk.get('document_hash') and doc.get('content_hash') != chunk['document_hash']:
            continue
        text = str(doc.get('clean_text') or '')
        digest = str(doc.get('text_hash') or doc.get('content_hash') or '').removeprefix('sha256:')
        if hashlib.sha256(text.encode()).hexdigest() != digest or chunk.get('source_id') != doc.get('source_id'):
            continue
        start, end = locator.get('char_start'), locator.get('char_end')
        if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(text)):
            continue
        quote = str(evidence.get('quote') or '')
        local = text[start:end]
        if len(quote) > 900:
            continue
        contexts = locator.get('bound_context_spans') or []
        if contexts or isinstance(chunk.get('char_start'), int):
            from .evidence_chunking import validate_bound_context, quote_within_chunk, combine_bound_quote
            if (not validate_bound_context(doc,chunk,contexts) or
                    not quote_within_chunk(doc,chunk,start,end,contexts)):
                continue
            if quote == combine_bound_quote(contexts,local):
                return True
            # Offset-bearing evidence chunks cannot fall back to arbitrary nearby lines.
            if chunk.get('bound_context_spans') is not None:
                continue
        # Table qualifiers may prepend bound headers to the exact row span.
        if local and local in quote and (quote == local or all(part in text[max(0,start-900):end] for part in quote.splitlines())):
            return True
    return False


class _Components:
    def __init__(self):
        self.parent = {}

    def root(self, key):
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.root(self.parent[key])
        return self.parent[key]

    def join(self, a, b):
        a, b = sorted((self.root(a), self.root(b)))
        self.parent[b] = a


def _lineage(sources):
    """Reused upstream reports are not independent, even in citation cycles."""
    nodes, edges, components = {}, [], _Components()
    for source in sorted(sources, key=_json):
        sid = str(source.get('source_id') or '')
        if not sid:
            continue
        components.root(sid)
        nodes.setdefault(sid, []).append(source)
        for upstream in source.get('upstream_source_mentions') or []:
            target = str(upstream.get('source_id') or upstream.get('name') or '') if isinstance(upstream, dict) else str(upstream)
            if target:
                components.join(sid, target)
                edges.append({'source_id': sid, 'upstream_source': target})
    result = []
    for sid, records in sorted(nodes.items()):
        facts = {}
        for key in ('original_publisher', 'republisher'):
            values = _unique([r[key] for r in records if r.get(key)])
            facts[key] = values[0] if len(values) == 1 else None
            if len(values) > 1:
                facts[key + '_candidates'] = values
        hosts = sorted({urlparse(str(r.get('url') or r.get('source_url') or '')).hostname for r in records} - {None})
        result.append(dict(source_id=sid, **facts, hosting_domain=hosts[0] if len(hosts) == 1 else None, hosting_domains=hosts, independence_group=_id('origin_', components.root(sid))))
    return {'sources': result, 'upstream_edges': _unique(edges)}


def _value_kind(obs):
    """Use one conservative precision classification for grouping and arithmetic."""
    kind = obs['fields'].get('value_kind')
    bounded = any(token in e['quote'].lower() for e in obs['field_evidence']
                  for token in ('at least ', 'at most ', 'approximately ', 'estimated ', 'more than ', 'less than '))
    if bounded and kind in (None, '', 'exact'):
        return 'unknown'
    kind = kind or 'exact'
    return kind if kind in ('exact', 'lower_bound', 'upper_bound', 'interval', 'estimate') else 'unknown'


def build_evidence_products(observations, *, evidence_index, sources=(), partitions=()):
    """Return JSON-ready entities, aggregates, temporal observations and lineage.

    ``patient_links`` must itself be a qualified field containing a list of
    {document_hash, identity} targets, with source text explicitly naming the
    identities. Partition declarations are accepted only via qualified structured
    ``disjoint_partition`` fields; see ``_derive`` for the strict schema.
    """
    admitted, excluded = [], []
    for row in _unique(observations):
        q = row.get('evidence_qualification') or row.get('qualification') or {}
        entries = q.get('field_evidence') or []
        reason = None
        if q.get('status') != 'qualified':
            reason = 'not_qualified'
        elif not entries or any(not _resolved(e, evidence_index) or e.get('value') != row.get(e.get('field')) for e in entries) or any(row.get(k) is not None and row.get(k) != '' and k not in {e.get('field') for e in entries} for k in SCOPE + METRICS + ('value_kind',)):
            reason = 'qualification_value_or_locator_mismatch'
        if reason:
            excluded.append({'observation_id': _id('obs_', row), 'reason': reason})
            continue
        fields = {e['field']: e['value'] for e in entries}
        obs = {'observation_id': _id('obs_', [fields, _unique(entries)]), 'fields': fields, 'field_evidence': _unique(entries), 'source_id': row.get('source_id'), 'product_kind': q.get('product_kind'), 'row': row}
        admitted.append(obs)
    admitted = list({o['observation_id']: o for o in sorted(admitted, key=lambda o: _json(o['row']))}.values())
    components, patients = _Components(), []
    for obs in admitted:
        if obs['product_kind'] != 'case_level':
            continue
        row, fields = obs['row'], obs['fields']
        identity = next((k for k in IDENTITIES if fields.get(k)), None)
        identity_entries = [e for e in obs['field_evidence'] if e['field'] == identity]
        # The qualification already resolved an exact source-local identity span.
        # case_span_quote is optional redundant metadata, not a second authority.
        span = str(row.get('case_span_quote') or ((identity_entries[0].get('locator') or {}).get('patient_span_quote') if len(identity_entries) == 1 else ''))
        if not identity or not span or str(fields[identity]) not in span:
            excluded.append({'observation_id': obs['observation_id'], 'reason': 'missing_patient_span_identity'})
            continue
        anchors = [e for e in obs['field_evidence'] if e['field'] == identity and span in e['quote']]
        if len(anchors) != 1:
            excluded.append({'observation_id': obs['observation_id'], 'reason': 'ambiguous_patient_span'})
            continue
        anchor = anchors[0].get('locator') or {}
        key = _json([anchors[0]['document_hash'], str(fields[identity]), anchor.get('char_start'), anchor.get('char_end')])
        components.root(key)
        patients.append((key, obs, span, anchors[0]['document_hash']))
    known = {key for key, *_ in patients}
    identities = defaultdict(set)
    for key,obs,span,digest in patients:
        name = next((obs['fields'].get(k) for k in IDENTITIES if obs['fields'].get(k)),None)
        identities[_json([digest,str(name)])].add(key)
    for key, obs, span, digest in patients:
        for target in obs['fields'].get('patient_links') or []:
            if not isinstance(target, dict):
                continue
            targets = identities.get(_json([target.get('document_hash'), str(target.get('identity'))]),set())
            # Local labels can restart within a document; an unlocated link to
            # such a repeated label does not identify one patient.
            target_key = next(iter(targets)) if len(targets) == 1 else None
            proofs = [e for e in obs['field_evidence'] if e['field'] == 'patient_links']
            target_docs = [d for d in _documents(evidence_index) if d.get('content_hash') == target.get('document_hash')]
            citation_urls = {str(d.get('url') or d.get('source_url') or '') for d in target_docs} - {''}
            same_doc = target.get('document_hash') == digest
            if target_key in known and any(str(target.get('identity')) in e['quote'] and (same_doc or any(url in e['quote'] for url in citation_urls)) and supports_patient_relation(e['quote'],next((obs['fields'].get(k) for k in IDENTITIES if obs['fields'].get(k)),None),target.get('identity')) for e in proofs):
                components.join(key, target_key)
    entities = defaultdict(list)
    for key, obs, span, digest in patients:
        # Every field must remain attached to this patient's own span.
        entries = [e for e in obs['field_evidence'] if e['document_hash'] == digest and e['quote'] in span and e['field'] not in ('patient_links',)]
        if any(e['field'] in IDENTITIES for e in entries):
            entities[components.root(key)].extend(entries)
    case_entities = []
    for key, entries in sorted(entities.items()):
        fields = defaultdict(list)
        for entry in _unique(entries):
            fields[entry['field']].append({'value': entry['value'], 'evidence': entry})
        case_entities.append({'entity_id': _id('patient_', key), 'field_candidates': dict(sorted(fields.items())), 'conflicting_fields': sorted(k for k,v in fields.items() if len({_json(c['value']) for c in v}) > 1)})
    grouped, series = defaultdict(list), []
    for obs in admitted:
        f = obs['fields']
        if obs['product_kind'] != 'aggregate':
            continue
        scope = {k: f.get(k) for k in SCOPE}
        if not any(f.get(k) for k in ('reporting_period', 'period_start_date', 'period_end_date', 'event_start_date', 'event_end_date', 'metric_period_start', 'metric_period_end')):
            scope['event_date_anchors'] = {k: f[k] for k in EVENT_DATES if f.get(k)}
        kind = _value_kind(obs)
        for metric in METRICS:
            value = f.get(metric)
            if value is None:
                continue
            key = _json([scope, metric, kind])
            evidence = [e for e in obs['field_evidence'] if e['field'] == metric]
            candidate = {'observation_id': obs['observation_id'], 'value': value, 'as_of_date': f.get('as_of_date'), 'evidence': evidence}
            grouped[key].append(candidate)
            docs = [d for d in _documents(evidence_index) if d.get('content_hash') in {e['document_hash'] for e in evidence}]
            publication_dates = sorted({str(d.get('publication_date') or d.get('date_published')) for d in docs if d.get('publication_date') or d.get('date_published')})
            event_fields = [name for name in EVENT_DATES if f.get(name)] or [None]
            for date_field in event_fields:
                series.append(dict(candidate, metric=metric, value_kind=kind, scope=scope, event_date=f.get(date_field), event_date_field=date_field, publication_date=publication_dates[0] if len(publication_dates)==1 else None, publication_date_candidates=publication_dates))
    aggregates = []
    for key, values in sorted(grouped.items()):
        scope, metric, kind = json.loads(key)
        values = _unique(values)
        aggregates.append({'aggregate_id': _id('aggregate_', key), 'scope':scope, 'metric':metric, 'value_kind':kind, 'candidates':values, 'conflict':len({_json(v['value']) for v in values}) > 1})
    all_sources = list(sources) + _documents(evidence_index)
    return {'case_entities':case_entities, 'aggregate_groups':aggregates, 'time_series_observations':_unique(series), 'origin_lineage':_lineage(all_sources), 'derivations':_derive(admitted, partitions, evidence_index), 'excluded_observations':_unique(excluded)}


def _derive(observations, partitions, index):
    declarations = list(partitions) + [o['row'] for o in observations if o['fields'].get('disjoint_partition')]
    results = []
    for row in _unique(declarations):
        q = row.get('evidence_qualification') or row.get('qualification') or {}
        declaration = row.get('disjoint_partition')
        if q.get('status') != 'qualified' or not isinstance(declaration, dict):
            continue
        proofs = [e for e in q.get('field_evidence') or [] if e.get('field') == 'disjoint_partition' and e.get('value') == declaration and _resolved(e, index)]
        members = declaration.get('members') or []
        # Only an explicit population partition is currently supported. Geographic
        # hierarchy is never proof of disjointness; arbitrary dates are not split.
        if declaration.get('axis') != 'population_scope' or declaration.get('exhaustive') is not True or declaration.get('mutually_exclusive') is not True or len(members) < 2 or not all(isinstance(m,str) and m for m in members) or len(set(members)) != len(members):
            continue
        target, metric = declaration.get('target'), declaration.get('metric')
        if not isinstance(target, str) or metric not in METRICS or metric.endswith('_rate') or metric == 'metric_value':
            continue
        if not any(supports_partition_relation(e['quote'],members,target) for e in proofs):
            continue
        # A partition declaration cannot be reused outside its qualified scope.
        # Population is the partition axis, so the target is compared separately.
        proof_fields = {e['field']:e['value'] for e in q.get('field_evidence') or []
                        if _resolved(e,index) and e.get('value') == row.get(e.get('field'))}
        proof_scope = {k:row[k] for k in SCOPE + ('as_of_date',)
                       if k != 'population_scope' and row.get(k) not in (None, '')}
        if any(k not in proof_fields for k in proof_scope):
            continue
        groups = defaultdict(list)
        for obs in observations:
            f = obs['fields']
            if obs['product_kind'] != 'aggregate' or f.get('population_scope') not in members or _value_kind(obs) != 'exact':
                continue
            if any(f.get(k) != value for k,value in proof_scope.items()):
                continue
            value = f.get(metric)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or int(value) != value:
                continue
            count_bases = [f.get(k) for k in ('count_basis','count_semantics','statistical_count_type')]
            unknown = {'unknown','unspecified','not reported','n/a'}
            known_count_basis = any(v and str(v).strip().casefold() not in unknown for v in count_bases)
            known_unit = any(f.get(k) and str(f[k]).strip().casefold() not in unknown for k in ('unit','count_unit','metric_unit'))
            if not all(f.get(k) for k in ('disease','country','case_definition')) or not known_unit or not known_count_basis or not (f.get('reporting_period') or (f.get('period_start_date') and f.get('period_end_date'))):
                continue
            scope = {k:f.get(k) for k in SCOPE if k != 'population_scope'}
            # Revisions from different as-of dates must not be added together.
            groups[_json([scope, f.get('as_of_date')])].append(obs)
        for key, parents in sorted(groups.items()):
            by_member = defaultdict(list)
            for parent in parents:
                by_member[parent['fields']['population_scope']].append(parent)
            if set(by_member) != set(members) or any(len({p['fields'][metric] for p in group}) != 1 for group in by_member.values()):
                continue
            # Duplicate corroboration is counted once, with all parent links kept.
            value = sum(int(group[0]['fields'][metric]) for group in by_member.values())
            scope, asof = json.loads(key)
            scope['population_scope'] = target
            parent_ids = sorted({p['observation_id'] for p in parents})
            result = dict(metric=metric, value=value, value_kind='exact', scope=scope, as_of_date=asof, parent_observation_ids=parent_ids, formula='sum(parent values)', conditions=dict(mutually_exclusive=True, exhaustive=True, all_members_present=True, exact_values=True, identical_other_scope=True), partition_evidence=_unique(proofs))
            result['derivation_id'] = _id('derivation_', result)
            results.append(result)
    return _unique(results)
