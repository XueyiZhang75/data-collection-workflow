"""Complete, deterministic source catalogue from the current in-memory package.

This projection does not fetch, reopen files, or reassess evidence. Dataset
membership and the shared source-progress attribution determine contributions.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import PurePosixPath, PureWindowsPath
from urllib.parse import urlsplit

from ..source_progress import build_source_progress, _readable
from ..nodes.source_discovery import canonicalize_url


ROLES = {'qualified': 'final_dataset', 'candidate': 'candidate_records', 'context': 'context_records'}
FIELD_LABELS = {
    'cases_confirmed': 'Confirmed cases', 'cases_probable': 'Probable cases',
    'cases_suspected': 'Suspected cases', 'cases_unspecified': 'Cases (classification unspecified)',
    'deaths': 'Deaths', 'hospitalizations': 'Hospitalizations', 'disease': 'Disease',
    'country': 'Country', 'geographic_scope': 'Geographic scope',
    'reporting_period': 'Statistical period', 'as_of_date': 'Statistics as-of date',
    'metric_period_start': 'Statistical period start', 'metric_period_end': 'Statistical period end',
    'event_start_date': 'Event start date', 'event_end_date': 'Event end date',
    'outbreak_closure_date': 'Outbreak closure announcement date', 'date_reported': 'Record date',
}
SOURCE_TYPES = {
    'international_public_health_agency': 'International public health agency',
    'international_organization_report': 'International organization report',
    'national_public_health_agency': 'National public health agency',
    'official_public_health_agency': 'Public health agency',
    'structured_database': 'Database / data platform',
    'background_fact_sheet': 'Background information / fact sheet',
    'academic_or_peer_reviewed_source': 'Academic literature',
    'peer_reviewed_literature': 'Academic literature', 'news_media': 'News media',
    'news_and_situation_report': 'News / situation report', 'social_media': 'Social media',
    'secondary_aggregator': 'Republishing / aggregation platform',
    'situation_report_aggregator': 'Situation report aggregator', 'commercial_site': 'Commercial website',
}
PROCESSING_LABELS = {
    'records_collected': 'Collected observations recorded',
    'budget_deferred': 'Deferred by budget limit', 'evidence_contributed': 'Read; records extracted',
    'readable': 'Read', 'extracted_without_evidence': 'Read; extraction attempted',
    'acquisition_failed': 'Retrieval failed', 'acquisition_incomplete': 'Retrieval incomplete',
    'screening_excluded': 'Excluded by screening', 'awaiting_extraction': 'Read; awaiting extraction',
    'not_attempted': 'Not yet retrieved', 'acquisition_in_progress': 'Retrieval in progress',
}
PROCESSING_REASONS = {
    'collected_only': 'Collected observations are retained without an evidence-qualified designation.',
    'source_targets': 'The source-processing budget limit was reached.',
    'http_requests': 'The HTTP request budget limit was reached.',
    'browser': 'The browser-call budget limit was reached.',
    'qualified': 'This source contributes currently qualified observations.',
    'candidate_only': 'Candidate records were retained; they do not support the final conclusion.',
    'context_only': 'This source contributes context records.',
    'no_output_observation': 'Extraction produced no observation records.',
    'no_extracted_evidence': 'Readable content was retrieved, but no records were produced.',
    'unprocessed_evidence_chunks': 'Some text passages are awaiting extraction.',
    'acquisition_incomplete': 'Retrieval did not complete.',
    'http_error': 'The HTTP request failed.', 'blocked': 'Access to the page was restricted.',
    'empty_response': 'The retrieval response was empty.',
    'unreadable_content': 'The retrieved content was not readable.',
    'screening_excluded': 'The source was excluded during screening.',
    'geography_mismatch': 'The geographic scope does not match the task.',
    'queued': 'The source is queued for retrieval.', 'not_scheduled': 'Retrieval has not been scheduled.',
    'claimed_target': 'Retrieval has started.',
}
UNKNOWN_PUBLISHERS = {'', 'unknown', 'unknown publisher', 'publisher unknown', 'publisher_unknown',
                      'none', 'null', 'nan', 'not_available', 'not available', 'search', 'tavily'}


def _unique(values):
    result = []
    for value in values:
        if value not in (None, '', [], {}) and value not in result:
            result.append(value)
    return result


def _rows(container, key):
    return [row for row in container.get(key) or [] if isinstance(row, dict)]


def _source_rows(package, state):
    """Augment grouping with inventory/exclusions and otherwise orphaned evidence."""
    rows = []
    for key in ('source_registry', 'source_inventory', 'excluded_sources'):
        for container in (package, state):
            for entry in _rows(container, key):
                row = deepcopy(entry)
                if key == 'excluded_sources':
                    row.setdefault('final_screening_decision', 'exclude')
                rows.append(row)
                # These URLs are explicitly recorded as same-source aliases;
                # connect them through the shared source-progress union rules.
                for alias_url in row.get('official_report_alias_urls') or []:
                    if row.get('source_id') and alias_url:
                        rows.append({'source_id': row['source_id'], 'url': alias_url})
    # Record-only and document-only sources still need an auditable catalogue row.
    known_ids = {str(row.get('source_id')) for row in rows if row.get('source_id')}
    known_urls = {canonicalize_url(str(row.get('canonical_url') or row.get('url') or '')) for row in rows}
    for key in ('documents', 'evidence_chunks', *ROLES.values()):
        containers = (package, state) if key in {'documents', 'evidence_chunks'} else (package,)
        for container in containers:
            for record in _rows(container, key):
                source_id = str(record.get('source_id') or '')
                source_url = record.get('source_url') or record.get('canonical_url') or record.get('url') or ''
                if source_id and source_id not in known_ids:
                    rows.append({'source_id': source_id,
                                 'url': source_url,
                                 'title': record.get('source_title') or record.get('title') or '',
                                 '_catalog_reference_only': key in ROLES.values()})
                    known_ids.add(source_id)
                    known_urls.add(canonicalize_url(str(source_url)))
                elif not source_id and source_url and canonicalize_url(str(source_url)) not in known_urls:
                    rows.append({'url': source_url, 'title': record.get('source_title') or record.get('title') or '',
                                 '_catalog_reference_only': key in ROLES.values()})
                    known_urls.add(canonicalize_url(str(source_url)))
    return sorted(rows, key=lambda row: (str(row.get('source_id') or ''),
                                         str(row.get('canonical_url') or row.get('url') or ''),
                                         str(row.get('title') or '')))


def _saved_statuses(package, state, entries):
    statuses = []
    for container in (state, package):
        statuses.extend(_rows(container, 'source_processing_status'))
        statuses.extend(((container.get('result_manifest') or {}).get('source_progress') or {}).get('sources') or [])
    statuses.extend(row for row in entries if row.get('processing_status'))
    by_id = {}
    for row in statuses:
        ids = row.get('source_ids') or [row.get('source_id')]
        for source_id in ids:
            if source_id:
                by_id.setdefault(str(source_id), row)
    return by_id


def _processing(progress):
    code = progress.get('processing_status') or 'unknown'
    raw_reason = str(progress.get('processing_reason') or '')
    reason = PROCESSING_REASONS.get(raw_reason)
    if raw_reason.startswith('no_extraction_eligible_spans'):
        reason = 'Text screening found no target-data passages eligible for extraction.'
    elif raw_reason.startswith('llm_source_critic_block_fetch'):
        reason = 'Source screening excluded this page from retrieval.'
    if not reason:
        reason = {
            'acquisition_failed': 'Retrieval failed; the original reason is retained in the source catalogue.',
            'budget_deferred': 'Processing was deferred because a budget limit was reached.',
            'screening_excluded': 'The source was excluded during screening.',
        }.get(code, 'The processing status was recorded; the original reason is retained in the source catalogue.')
    acquisition = progress.get('acquisition_status') or 'unknown'
    return {'code': code, 'label': PROCESSING_LABELS.get(code, 'Processing status not recorded'),
            'reason': reason, 'raw_reason': raw_reason, 'acquisition_status': acquisition,
            'readable': acquisition == 'readable',
            'prior_acquisition_failures': progress.get('prior_acquisition_failures', 0)}


def _publisher(entries, url):
    candidates = [(row, row.get('actual_publisher') or row.get('publisher')) for row in entries]
    named = [(row, name) for row, name in candidates if str(name or '').strip().lower() not in UNKNOWN_PUBLISHERS]
    # Verification attaches to the particular publisher label, not an alias's flag.
    chosen = next(((row, name) for row, name in named if row.get('source_identity_unverified') is False),
                  named[0] if named else ({}, None))
    row, name = chosen
    verified = bool(name and row.get('source_identity_unverified') is False)
    try:
        domain = urlsplit(url).hostname
    except ValueError:
        domain = None
    return {'display': name or domain or 'Publisher not recorded', 'verified': verified,
            'basis': ('Source identity check saved during the collection session.' if verified else
                      'Saved publisher label or website domain; publisher identity has not been verified.'),
            'evidence_quotes': _unique(row.get('page_identity_evidence') or []),
            'raw_publisher': name, 'raw_identity_unverified': row.get('source_identity_unverified')}


def _publication(entries, documents):
    values = []
    for row in entries + documents:
        metadata = row.get('metadata') if isinstance(row.get('metadata'), dict) else {}
        for container in (row, metadata):
            for key in ('published_date', 'publication_date', 'published_at'):
                value = container.get(key)
                if isinstance(value, dict):
                    value = value.get('value')
                if value not in (None, ''):
                    values.append(str(value))
    values = sorted(set(values))
    if len(values) == 1:
        return {'value': values[0], 'basis': 'Explicit publication metadata; not a statistical cutoff date.'}
    if values:
        return {'value': None, 'basis': 'Conflicting publication metadata; publication date remains unconfirmed.',
                'recorded_values': values}
    return {'value': None, 'basis': 'No confirmed publication date available.'}


def _record_detail(record, role, ids, chunks):
    qualification = record.get('evidence_qualification') or {}
    fields = []
    for item in qualification.get('field_evidence') or []:
        if not item.get('supported'):
            continue
        locator = item.get('locator') or {}
        bound = str(chunks.get(str(locator.get('chunk_id')), {}).get('source_id') or '')
        if bound not in ids:
            continue
        name = str(item.get('field') or '')
        fields.append({'field': name, 'label': FIELD_LABELS.get(name, name.replace('_', ' ').capitalize()),
                       'value': deepcopy(item.get('value')), 'quote': item.get('quote'),
                       'document_hash': item.get('document_hash'), 'locator': deepcopy(item.get('locator'))})
    return {'record_id': record.get('record_id'), 'source_id': record.get('source_id'),
            'status': 'qualified', 'product_kind': ('context' if role == 'context' else qualification.get('product_kind')),
            'reasons': list(qualification.get('reasons') or []), 'evidence_quote': record.get('evidence_quote') or '',
            'supported_fields': fields, 'closure_evidence': deepcopy(qualification.get('closure_evidence') or [])}


def _periods(details):
    rows, labels = [], []
    period_keys = ('reporting_period', 'metric_period_start', 'metric_period_end', 'as_of_date',
                   'event_start_date', 'event_end_date', 'outbreak_closure_date')
    for detail in details:
        values = {field['field']: field['value'] for field in detail['supported_fields']
                  if field['field'] in period_keys and field['value'] not in (None, '')}
        if not values:
            continue
        rows.append({'record_id': detail['record_id'], **values})
        if values.get('reporting_period'):
            label = str(values['reporting_period'])
        elif values.get('metric_period_start') or values.get('metric_period_end'):
            label = f"{values.get('metric_period_start') or 'Start date unspecified'} to {values.get('metric_period_end') or 'End date unspecified'}"
        elif values.get('as_of_date'):
            label = 'As of ' + str(values['as_of_date'])
        elif values.get('outbreak_closure_date'):
            label = 'Closure announced on ' + str(values['outbreak_closure_date']) + ' (not a statistical cutoff date)'
        else:
            label = f"Event dates: {values.get('event_start_date') or 'Unspecified'} to {values.get('event_end_date') or 'Unspecified'}"
        labels.append(label)
    return {'display': '; '.join(_unique(labels)) or 'No verified statistical period available',
            'supported_record_periods': rows}


def _candidate_reason_labels(reasons):
    labels = {
        'missing_observation_period': 'The observation period could not be confirmed.',
        'missing_geography': 'The geographic scope could not be confirmed.',
        'contract_mismatch': 'The extracted time or location does not match the task requirements.',
        'unbound_observation_scope': 'The numeric value is not reliably linked to its observation scope.',
        'ambiguous_span': 'The location of the supporting text is ambiguous.',
        'unresolvable_or_nonlocal_span': 'The supporting text could not be reliably located.',
        'unbound_field_value': 'Some extracted field values lack support from the corresponding source text.',
        'low_ocr_confidence_candidate': 'OCR confidence is insufficient; the original document requires review.',
    }
    return _unique(labels.get(str(reason).rsplit(':', 1)[-1],
                              'Some evidence requirements remain unmet; see the original decision.') for reason in reasons)


def _unqualified_detail(record, role, decision):
    qualification = record.get('evidence_qualification') or {}
    reasons = _unique([*(qualification.get('reasons') or []), *(record.get('reasons') or []),
                       *(decision.get('reasons') or [])])
    detail = {'record_id': record.get('record_id'), 'source_id': record.get('source_id'), 'status': role,
              'product_kind': qualification.get('product_kind') or record.get('product_kind') or decision.get('product_kind'),
              'evidence_quote': record.get('evidence_quote') or '', 'reasons': reasons,
              'reason_labels': _unique(str(reason) if ' ' in str(reason) and str(reason).isascii()
                                       else _candidate_reason_labels([reason])[0] for reason in reasons),
              'record_final_inclusion_status': record.get('record_final_inclusion_status') or decision.get('record_final_inclusion_status'),
              'requires_human_review': record.get('requires_human_review'),
              'inclusion_decision': deepcopy(decision)}
    if role == 'collected':
        detail['supported_fields'] = []
        detail['observed_fields'] = [{'field': field, 'label': label, 'value': deepcopy(record[field])}
                                     for field, label in FIELD_LABELS.items() if record.get(field) is not None]
    return detail


def _saved_document(document):
    result = {'content_hash': document.get('content_hash'), 'text_hash': document.get('text_hash'),
              'text_char_count': len(document.get('clean_text') or ''), 'retrieved_at': document.get('retrieved_at')}
    value = document.get('artifact_path') or document.get('artifact') or document.get('saved_document_path')
    if value:
        value = str(value).replace('\\', '/')
        # Only a relative evidence locator is exported. No local file is opened.
        path = PurePosixPath(value)
        if not path.is_absolute() and not PureWindowsPath(value).drive and '..' not in path.parts and ':' not in value:
            result['artifact'] = value
    return result


def build_source_catalog(package: dict, state: dict | None = None) -> dict:
    """Return every source and its current evidence contribution, without I/O."""
    state = state or {}
    entries = _source_rows(package, state)
    documents = _unique(_rows(state, 'documents') + _rows(package, 'documents'))
    chunk_rows = _unique(_rows(state, 'evidence_chunks') + _rows(package, 'evidence_chunks'))
    chunks = {str(row.get('chunk_id') or row.get('evidence_span_id')): row for row in reversed(chunk_rows)}
    qualified = [row for row in _rows(package, 'final_dataset')
                 if (row.get('evidence_qualification') or {}).get('status') == 'qualified']
    collected = [row for row in _rows(package, 'final_dataset')
                 if (row.get('evidence_qualification') or {}).get('status') != 'qualified']
    augmented_package = {**package, 'source_registry': entries, 'final_dataset': qualified}
    augmented_state = {**state, 'source_registry': [], 'documents': documents, 'evidence_chunks': list(chunks.values())}
    progress = build_source_progress(augmented_package, augmented_state)
    statuses = _saved_statuses(package, state, entries)
    records = {role: {str(row.get('record_id') or ''): row for row in _rows(augmented_package, key)} for role, key in ROLES.items()}
    if collected:
        records['collected'] = {str(row.get('record_id') or ''): row for row in collected}
    decisions = {str(row.get('record_id') or ''): row for row in _rows(package, 'record_inclusion_decisions')}
    rows = sorted(progress['sources'], key=lambda row: (not bool(row['qualified_record_ids']),
                                                       tuple(row['source_ids']), row['canonical_url']))
    excluded_ids = {str(row.get('source_id')) for owner in (package, state) for row in _rows(owner, 'excluded_sources')}
    sources = []
    for number, progress_row in enumerate(rows, 1):
        ids = progress_row['source_ids']
        group = [row for row in entries if str(row.get('source_id')) in ids
                 or (not row.get('source_id') and canonicalize_url(str(row.get('canonical_url') or row.get('url') or ''))
                     in progress_row['canonical_urls'])]
        if not group:
            group = [{}]
        docs = [row for row in documents if str(row.get('source_id')) in ids]
        status = dict(progress_row)
        saved = next((statuses[source_id] for source_id in ids if source_id in statuses), None)
        # Saved acquisition decisions survive artifact-only reporting; contributions
        # always come from the current datasets, never stale saved record counts.
        if saved and not docs and status['processing_status'] in {'not_attempted', 'screening_excluded', 'evidence_contributed'}:
            for key in ('processing_status', 'processing_reason', 'acquisition_status', 'prior_acquisition_failures'):
                if saved.get(key) is not None:
                    status[key] = saved[key]
        if status['processing_status'] in {'not_attempted', 'readable', 'extracted_without_evidence', 'evidence_contributed'} and progress_row['evidence_status'] != 'none':
            status.update(processing_status='evidence_contributed', processing_reason=progress_row['evidence_status'])
        if status['processing_status'] == 'not_attempted' and not saved and not docs and all(row.get('_catalog_reference_only') for row in group):
            status.update(processing_status='unknown', processing_reason='not_recorded', acquisition_status='unknown')
        roles = {role: [records[role][record_id] for record_id in progress_row[f'{role}_record_ids'] if record_id in records[role]]
                 for role in ROLES}
        if collected:
            roles['collected'] = [record for record in records['collected'].values()
                                  if str(record.get('source_id') or '') in ids or
                                  (not record.get('source_id') and canonicalize_url(str(record.get('source_url') or ''))
                                   in progress_row['canonical_urls'])]
            if roles['collected'] and progress_row['evidence_status'] == 'none' and status['processing_status'] in {
                    'not_attempted', 'unknown', 'readable', 'extracted_without_evidence', 'evidence_contributed'}:
                status.update(processing_status='records_collected', processing_reason='collected_only')
        counts = {role: len(rows) for role, rows in roles.items()}
        url = progress_row['canonical_url']
        pub = _publisher(group, url)
        processing = _processing(status)
        details = [_record_detail(record, role, ids, chunks) for role in ('qualified', 'context') for record in roles[role]]
        candidates = [_unqualified_detail(record, 'candidate', decisions.get(str(record.get('record_id') or ''), {}))
                      for record in roles['candidate']]
        collected_details = [_unqualified_detail(record, 'collected', decisions.get(str(record.get('record_id') or ''), {}))
                             for record in roles.get('collected', [])]
        reason_counts = Counter(reason for record in candidates for reason in record['reasons'])
        readable_docs = [doc for doc in docs if _readable(doc)]
        if details:
            excerpt = next((field['quote'] for detail in details for field in detail['supported_fields'] if field.get('quote')),
                           next((detail['evidence_quote'] for detail in details if detail['evidence_quote']), ''))
            excerpt_basis = 'Original quotation supporting a currently qualified record.'
        elif candidates:
            excerpt = next((row['evidence_quote'] for row in candidates if row['evidence_quote']), '')
            excerpt_basis = 'Candidate quotation; it has not passed evidence checks and does not support the final conclusion.'
        elif collected_details:
            excerpt = next((row['evidence_quote'] for row in collected_details if row['evidence_quote']), '')
            excerpt_basis = 'Quotation saved with a collected observation; evidence qualification has not been established.'
        else:
            excerpt = max((doc.get('clean_text') or '' for doc in readable_docs), key=len, default='')
            excerpt_basis = 'Saved page text; its numeric values have not necessarily passed evidence checks.' if excerpt else 'No usable page body was read.'
        if any(counts.values()):
            summary = (f"This source contributes {counts['qualified']} qualified data records, "
                       f"{counts['candidate']} candidate records, and {counts['context']} context records.")
            if counts.get('collected'):
                summary += f" It also has {counts['collected']} collected observations without evidence qualification."
        elif processing['readable']:
            summary = 'Readable content was saved, but no observation records were included in the results.'
        elif processing['code'] == 'unknown':
            summary = 'No retrieval status was recorded for this reference; source-content verification remains unresolved.'
        else:
            summary = 'No usable page body was read; the title and search metadata do not establish evidence.'
        source_type = next((row.get('source_type_final') or row.get('source_type') for row in group if row.get('source_type_final') or row.get('source_type')), 'unknown')
        type_label = SOURCE_TYPES.get(source_type, 'Source type unverified')
        if not pub['verified'] and source_type in {'international_public_health_agency', 'national_public_health_agency', 'official_public_health_agency'}:
            type_label = 'Unverified source classification: ' + type_label
        upstream = _unique(item for row in group for item in row.get('upstream_source_mentions') or [])
        upstream_verified = bool(upstream) and all(row.get('upstream_mentions_verified') is True for row in group if row.get('upstream_source_mentions'))
        primary = next((row.get('primary_vs_secondary') for row in group if row.get('primary_vs_secondary')), None)
        all_urls = _unique([url, *progress_row['canonical_urls'],
                            *(row.get('url') for row in group),
                            *(value for row in group for value in row.get('official_report_alias_urls') or [])])
        contribution = [label for role, label in [('qualified', 'Qualified data'), ('candidate', 'Candidate records'), ('context', 'Context information')] if counts[role]]
        if counts.get('collected'):
            contribution.append('Collected observations')
        sources.append({
            'report_source_id': f'S{number:03d}', 'source_ids': ids, 'preferred_source_id': ids[0] if ids else None,
            'url': url, 'alias_urls': [value for value in all_urls if value != url],
            'title': next((row.get('title') or row.get('page_title') for row in group if row.get('title') or row.get('page_title')), url or 'Title not recorded'),
            'publisher': pub, 'upstream_mentions': upstream, 'upstream_mentions_verified': upstream_verified,
            'upstream_mentions_basis': ('Verified source attribution saved during collection.' if upstream_verified else
                                        'Saved source annotations; upstream mentions do not verify publisher identity.'),
            'source_type_label': type_label, 'source_type_basis': 'Source classification saved during collection; publisher verification is reported separately.',
            'primary_or_republished': {'primary_or_authoritative': 'Labeled original / authoritative during collection',
                                       'secondary': 'Republished / secondary source', 'social': 'Social content'}.get(primary, 'Original-publication / republication relationship unconfirmed'),
            'publication_date': _publication(group, docs), 'statistics_period': _periods(details),
            'content_summary': summary, 'processing': processing,
            'contribution_labels': contribution or ['No verified data contributed'], 'record_counts': counts,
            'records_by_role': {role: [record.get('record_id') for record in rows] for role, rows in roles.items()},
            'evidence_details': details, 'candidate_details': candidates, 'collected_details': collected_details,
            'candidate_reason_counts': dict(sorted(reason_counts.items())), 'candidate_reason_labels': _candidate_reason_labels(reason_counts),
            'saved_excerpt': {'text': excerpt, 'basis': excerpt_basis},
            'search_metadata_excerpt': {'text': '\n\n'.join(_unique(row.get('snippet') for row in group)),
                                        'basis': 'Search / link-discovery metadata; not equivalent to a retrieved page body.'},
            'saved_documents': [_saved_document(doc) for doc in docs],
            'original_package_excluded_source_ids': sorted(set(ids) & excluded_ids),
            'raw_inventory': deepcopy([row for owner in (package, state) for row in _rows(owner, 'source_inventory') if str(row.get('source_id')) in ids]),
        })
    return {'metadata': {
        'schema_version': 1, 'qualification_basis': 'Current package dataset membership and saved field evidence.',
        'registry_source_ids': len({str(row.get('source_id')) for owner in (package, state) for row in _rows(owner, 'source_registry')}),
        'inventory_source_ids': len({str(row.get('source_id')) for owner in (package, state) for row in _rows(owner, 'source_inventory')}),
        'excluded_source_ids_in_original_package': len(excluded_ids), 'source_rows': len(sources),
        'deduplicated_alias_ids': sum(max(0, len(source['source_ids']) - 1) for source in sources),
        'record_counts': {role: len(rows) for role, rows in records.items()},
        'contribution_counts': {role: sum(source['record_counts'].get(role, 0) for source in sources) for role in records},
        'processing_counts': dict(Counter(source['processing']['label'] for source in sources)),
        'acquisition_counts': dict(Counter(source['processing']['acquisition_status'] for source in sources)),
        'unresolved_record_ids': progress['unresolved_record_ids'],
        'notes': ['Only canonical URL matches and explicit source aliases are merged.',
                  'Qualified data, candidates, and context are separate; candidates do not support the final conclusion.',
                  'Collected observations without a qualified evidence decision are retained separately from qualified data.',
                  'One record can contribute evidence to more than one source; source contributions are not additive record totals.',
                  'Failed, deferred, excluded, and unread sources remain in the catalogue.'],
    }, 'sources': sources}
