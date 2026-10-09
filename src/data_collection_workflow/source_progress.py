"""Projection of current-session source acquisition and evidence contribution.

Discovery counts describe candidates, not relevant-source recall. Qualification
comes from the shared evidence assessor; this module never reclassifies facts.
"""
from collections import Counter, defaultdict


def _readable(doc):
    from .workflow_recovery import _readable_document
    from .page_status import page_failure_reason
    return (_readable_document(doc) and not page_failure_reason(doc)
            and doc.get("fetch_status") not in {"fetch_failed", "failed"}
            and doc.get("parse_status") not in {"parse_failed", "failed"}
            and doc.get("quality_status") != "unusable" and not doc.get("is_shell"))


def _budget_deferred(doc):
    return bool(doc.get("budget_exhausted_kind") or doc.get("acquisition_status") == "budget_exhausted")


def unresolved_source_documents(documents, source_groups):
    """Resolve proven aliases without treating different content versions as equal."""
    from .nodes.source_discovery import canonicalize_url
    from .workflow_recovery import unresolved_acquisition_documents
    unresolved = unresolved_acquisition_documents({'documents': documents})
    groups = {source_id: group for group in source_groups for source_id in group['source_ids']}
    complete = [doc for doc in documents if _readable(doc) and not doc.get('acquisition_incomplete')
                and not _budget_deferred(doc)]
    def same_target(left, right, group):
        known = set(group.get('canonical_urls') or [group.get('canonical_url')]) - {None, ''}
        urls = [canonicalize_url(str(doc.get('canonical_url') or doc.get('url') or '')) for doc in (left,right)]
        if all(urls):
            return urls[0] == urls[1]
        return len(known) == 1 and all(not url or url in known for url in urls)
    result = []
    for doc in unresolved:
        group = groups.get(str(doc.get('source_id')))
        digest = doc.get('content_hash')
        settled = False
        for other in complete:
            if not group or str(other.get('source_id')) not in group['source_ids']:
                continue
            exact_version = digest and digest in {other.get('content_hash'),other.get('response_content_hash')}
            network_denial = doc.get('budget_exhausted_kind') in {'fetch','fetch_ordinary','http_requests','source_targets'} and not _readable(doc)
            if exact_version or ((not digest or network_denial) and same_target(doc,other,group)):
                settled = True
                break
        if not settled:
            result.append(doc)
    return result


def _chunk_skip_reason(chunk):
    from .nodes.extraction import extraction_skip_reason
    return ("no_target_data" if chunk.get("contains_target_data") is False else
            "disease_not_eligible" if chunk.get("extraction_eligible_for_task_disease") is False else
            "disease_relevance_unconfirmed" if chunk.get("disease_relevance_status") not in
            {None, "", "target_disease_match"} else extraction_skip_reason(chunk))


def _recorded_extraction_chunks(package, state, chunks):
    """Attach saved diagnostics without reopening the runtime or changing input."""
    summary = state.get('structured_extraction_summary')
    if summary is None:
        summary = package.get('structured_extraction_summary') or {}
    failed = set(map(str, summary.get('failed_chunk_ids') or []))
    current_failure_inventory = isinstance(summary.get('failed_chunk_ids'), (list, tuple, set))
    empty = {str(row.get('chunk_id')): row for row in summary.get('llm_empty_output_diagnostics') or []
             if isinstance(row, dict) and row.get('chunk_id')}
    observed = {str(row.get('chunk_id') or row.get('supporting_chunk_id') or row.get('evidence_chunk_id'))
                for container in (package, state)
                for name in ('raw_records', 'final_dataset', 'candidate_records', 'context_records')
                for row in container.get(name) or [] if isinstance(row, dict)}
    result = {}
    for key, original in chunks.items():
        row = dict(original)
        if key in failed:
            row.update(extraction_status='failed', extraction_reason='extraction_attempt_failed')
        elif key in observed and (row.get('extraction_status') != 'failed' or current_failure_inventory):
            row.update(extraction_status='completed', extraction_reason='extracted_observations_not_in_output')
        elif key in empty and (row.get('extraction_status') not in {'completed', 'failed'} or
                              (row.get('extraction_status') == 'failed' and current_failure_inventory)):
            row.update(extraction_status='attempted_empty',
                       extraction_reason=empty[key].get('reason') or 'empty_output_recorded')
        result[key] = row
    return result


def _extraction_outcome_counts(chunks, attempted):
    from .evidence_qualification import evidence_qualification_enabled
    evidence_mode = evidence_qualification_enabled()
    counts = Counter()
    for chunk in chunks:
        status = chunk.get('extraction_status')
        if status in {'failed', 'completed', 'attempted_empty'}:
            outcome = {'attempted_empty': 'empty'}.get(status, status)
        elif str(chunk.get('chunk_id')) in attempted:
            # An attempt ID alone does not establish successful empty output.
            outcome = 'attempted_unknown'
        elif status == 'skipped' or (evidence_mode and _chunk_skip_reason(chunk)):
            outcome = 'skipped'
        else:
            outcome = 'pending'
        counts[outcome] += 1
    return dict(counts)


def _processing_projection(group, docs, jobs, chunks, attempted, evidence):
    unresolved = unresolved_source_documents(docs, [{"source_ids": sorted(group["ids"]), "canonical_urls": sorted(group["urls"])}])
    complete = [doc for doc in docs if _readable(doc) and not doc.get("acquisition_incomplete")
                and not _budget_deferred(doc)]
    deferred = [job for job in jobs if job.get("status") == "budget_deferred"]
    denied = [doc for doc in unresolved if _budget_deferred(doc)]
    if deferred or denied:
        reasons = {str(job.get("reason") or "budget_exhausted") for job in deferred}
        reasons.update(str(doc.get("budget_exhausted_kind") or "budget_exhausted") for doc in denied)
        return "budget_deferred", "; ".join(sorted(reasons))
    partial = [doc for doc in unresolved if doc.get("acquisition_incomplete")]
    if partial:
        reasons = {str(doc.get("fetch_error") or doc.get("parse_error") or "acquisition_incomplete") for doc in partial}
        return "acquisition_incomplete", "; ".join(sorted(reasons))
    if evidence != "none":
        return "evidence_contributed", evidence
    if complete:
        from .evidence_qualification import evidence_qualification_enabled
        evidence = evidence_qualification_enabled()
        eligible, skipped = chunks, set()
        if evidence:
            eligible = []
            for chunk in chunks:
                # Match the recovery planner's task flags before the common
                # gate; no document index or full-text reconstruction is needed.
                reason = _chunk_skip_reason(chunk)
                if reason:
                    skipped.add(reason)
                else:
                    eligible.append(chunk)
        pending = [chunk for chunk in eligible if str(chunk.get("chunk_id")) not in attempted
                   and chunk.get("extraction_status") not in {"completed", "skipped", "attempted_empty", "failed"}]
        if pending:
            return "awaiting_extraction", "unprocessed_evidence_chunks"
        failed = [chunk for chunk in chunks if chunk.get('extraction_status') == 'failed']
        if failed:
            return 'extraction_failed', '; '.join(sorted({str(chunk.get('extraction_reason') or
                                                             'extraction_attempt_failed') for chunk in failed}))
        empty = [chunk for chunk in chunks if chunk.get('extraction_status') == 'attempted_empty']
        if empty:
            return 'extracted_without_evidence', '; '.join(sorted({str(chunk.get('extraction_reason') or
                                                                     'empty_output_recorded') for chunk in empty}))
        if any(chunk.get('extraction_reason') == 'extracted_observations_not_in_output' for chunk in chunks):
            return 'extracted_without_evidence', 'extracted_observations_not_in_output'
        if evidence and chunks and not eligible:
            return "readable", "no_extraction_eligible_spans: " + ", ".join(sorted(skipped))
        if evidence and eligible and not any(str(chunk.get("chunk_id")) in attempted or
                chunk.get("extraction_status") in {"completed", "attempted_empty", "failed"} for chunk in eligible):
            return "readable", "extraction_skipped_without_attempt"
        if eligible:
            return "extracted_without_evidence", "no_output_observation"
        return "readable", "no_extracted_evidence"
    excluded = [row for row in group["entries"] if row.get("final_screening_decision") == "exclude"
                or row.get("source_role_final") == "excluded" or row.get("blocked_from_fetch")]
    blocked = [job for job in jobs if job.get("status") == "blocked"]
    if excluded or blocked:
        reasons = {str(job.get("reason") or "screening_excluded") for job in blocked}
        reasons.update(str(row.get("blocked_from_fetch_reason") or row.get("target_fit_status")
                           or "screening_excluded") for row in excluded)
        return "screening_excluded", "; ".join(sorted(reasons))
    failed = [doc for doc in docs if not _budget_deferred(doc) and not _readable(doc)]
    failed_jobs = [job for job in jobs if job.get("status") == "failed"]
    if failed or failed_jobs:
        reasons = {str(doc.get("fetch_error") or doc.get("parse_error") or doc.get("acquisition_status")
                       or "unreadable_content") for doc in failed}
        reasons.update(str(job.get("reason") or "acquisition_failed") for job in failed_jobs)
        return "acquisition_failed", "; ".join(sorted(reasons))
    if any(job.get("status") == "running" for job in jobs):
        return "acquisition_in_progress", "claimed_target"
    return "not_attempted", "queued" if jobs else "not_scheduled"


def build_source_progress(package, state):
    from .nodes.source_discovery import canonicalize_url
    sources = list(state.get('source_registry') or []) + list(package.get('source_registry') or [])
    documents = list(state.get('documents') or package.get('documents') or [])
    chunks = {str(row.get('chunk_id') or row.get('evidence_span_id')): row
              for row in state.get('evidence_chunks') or package.get('evidence_chunks') or []}
    chunks = _recorded_extraction_chunks(package, state, chunks)
    # Collapse canonical URLs and explicit source ID aliases, including chains.
    parents = {}
    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]
    def union(keys):
        roots = sorted({find(key) for key in keys})
        for key in roots[1:]:
            parents[key] = roots[0]
    source_keys = []
    for row in sources:
        url = canonicalize_url(str(row.get('canonical_url') or row.get('url') or ''))
        ids = {str(value) for value in [row.get('source_id'), *(row.get('source_id_aliases') or []),
                                             *(row.get('official_report_alias_source_ids') or [])] if value}
        keys = ['id:' + value for value in sorted(ids)] + (['url:' + url] if url else [])
        if keys:
            union(keys)
            source_keys.append((row, keys[0], ids, url))
    groups = {}
    for row, key, ids, url in source_keys:
        group = groups.setdefault(find(key), {'ids': set(), 'urls': set(), 'screening': set(), 'entries': []})
        group['entries'].append(row)
        group['ids'].update(ids)
        if url:
            group['urls'].add(url)
        group['screening'].update(str(row[key]) for key in ('target_fit_status', 'target_verification_status') if row.get(key))
    by_id = {source_id: key for key, group in groups.items() for source_id in group['ids']}
    by_url = {url: key for key, group in groups.items() for url in group['urls']}
    jobs_by_group = defaultdict(list)
    frontier = state.get('acquisition_frontier') or package.get('acquisition_frontier') or {}
    for job in frontier.get('items') or []:
        key = by_id.get(str(job.get('source_id')))
        if key is None:
            key = by_url.get(canonicalize_url(str(job.get('url') or job.get('target_id') or '')))
        if key is not None:
            jobs_by_group[key].append(job)
    attempted = set(map(str, state.get('extraction_attempted_chunk_ids') or []))
    docs_by_group = defaultdict(list)
    for document in documents:
        key = by_id.get(str(document.get('source_id')))
        if key is not None:
            docs_by_group[key].append(document)
    records = {kind: defaultdict(set) for kind in ('qualified', 'candidate', 'context')}
    unresolved_record_ids = set()
    for kind, rows in [('qualified', package.get('final_dataset') or []),
                       ('candidate', package.get('candidate_records') or []),
                       ('context', package.get('context_records') or [])]:
        for record in rows:
            record_id = str(record.get('record_id') or '')
            fields = (record.get('evidence_qualification') or {}).get('field_evidence') or []
            bound = {str(chunks.get(str((field.get('locator') or {}).get('chunk_id')), {}).get('source_id') or '')
                     for field in fields if field.get('supported')}
            bound.discard('')
            # Older compatible records have no per-field list. When one exists,
            # never attribute qualified facts to an unverified primary source ID.
            ids = bound if fields and kind != 'candidate' else bound | {str(record.get('source_id') or '')}
            matched = {by_id[source_id] for source_id in ids if source_id in by_id}
            if not matched and record_id:
                unresolved_record_ids.add(record_id)
            for key in matched:
                records[kind][key].add(record_id)
    requirements = (package.get('source_coverage_audit') or {}).get('requirements') or []
    rows = []
    mismatch_labels = {'unrelated_disease', 'incompatible_disease', 'temporal_mismatch',
                       'geography_mismatch', 'geographic_mismatch', 'task_fit_mismatch'}
    for key, group in groups.items():
        docs = docs_by_group[key]
        readable = any(_readable(doc) for doc in docs)
        q, c, context = (sorted(records[kind][key]) for kind in ('qualified', 'candidate', 'context'))
        evidence = 'qualified' if q else 'candidate_only' if c else 'context_only' if context else 'none'
        group_chunks = [chunk for chunk in chunks.values() if str(chunk.get('source_id')) in group['ids']]
        outcome_counts = _extraction_outcome_counts(group_chunks, attempted)
        outcome = next((name for name in ('failed', 'pending', 'empty', 'attempted_unknown', 'completed', 'skipped')
                        if outcome_counts.get(name)), 'not_recorded')
        status, reason = _processing_projection(group, docs, jobs_by_group[key], group_chunks, attempted, evidence)
        acquisition = 'readable' if readable else 'failed_or_unreadable' if status == 'acquisition_failed' else 'not_fetched'
        prior_failures = sum(not _readable(doc) and not _budget_deferred(doc) for doc in docs)
        rows.append({'source_ids': sorted(group['ids']), 'canonical_url': min(group['urls'], default=''),
                     'canonical_urls': sorted(group['urls']),
                     'evidence_status': evidence, 'processing_status': status, 'processing_reason': reason,
                     'extraction_outcome': outcome, 'extraction_outcome_counts': outcome_counts,
                     'prior_acquisition_failures': prior_failures,
                     'qualified_record_ids': q, 'candidate_record_ids': c, 'context_record_ids': context,
                     'coverage_requirement_ids': sorted({str(requirement['requirement_id']) for requirement in requirements
                         if requirement.get('requirement_id') and set(q) & set(requirement.get('qualified_record_ids') or [])}),
                     'acquisition_status': acquisition, 'screening_statuses': sorted(group['screening']),
                     'screening_mismatch': bool(group['screening'] & mismatch_labels)})
    rows.sort(key=lambda row: (row['canonical_url'], row['source_ids']))
    return {'discovered_unique_sources': len(rows),
            'readable_sources': sum(row['acquisition_status'] == 'readable' for row in rows),
            'fetch_failed_sources': sum(row['acquisition_status'] == 'failed_or_unreadable' for row in rows),
            'not_fetched_sources': sum(row['acquisition_status'] == 'not_fetched' for row in rows),
            'budget_deferred_sources': sum(row['processing_status'] == 'budget_deferred' for row in rows),
            'screening_excluded_sources': sum(row['processing_status'] == 'screening_excluded' for row in rows),
            'not_attempted_sources': sum(row['processing_status'] == 'not_attempted' for row in rows),
            'awaiting_extraction_sources': sum(row['processing_status'] == 'awaiting_extraction' for row in rows),
            'extraction_empty_sources': sum(bool(row['extraction_outcome_counts'].get('empty')) for row in rows),
            'extraction_failed_sources': sum(bool(row['extraction_outcome_counts'].get('failed')) for row in rows),
            'extraction_skipped_sources': sum(row['extraction_outcome'] == 'skipped' for row in rows),
            'missing_extraction_evidence_sources': sum(row['extraction_outcome'] in {'not_recorded', 'attempted_unknown'}
                                                      and row['acquisition_status'] == 'readable' for row in rows),
            'acquisition_incomplete_sources': sum(row['processing_status'] == 'acquisition_incomplete' for row in rows),
            'qualified_evidence_sources': sum(row['evidence_status'] == 'qualified' for row in rows),
            'candidate_only_sources': sum(row['evidence_status'] == 'candidate_only' for row in rows),
            'task_requirement_contributing_sources': sum(bool(row['coverage_requirement_ids']) for row in rows) if requirements else None,
            'context_only_sources': sum(row['evidence_status'] == 'context_only' for row in rows),
            'without_extracted_evidence_sources': sum(row['evidence_status'] == 'none' for row in rows),
            'screening_mismatch_sources': sum(row['screening_mismatch'] for row in rows),
            'unresolved_record_ids': sorted(unresolved_record_ids), 'source_recall': None,
            'interpretation': 'Discovery and screening counts are candidates, not confirmed coverage. '
                              'Evidence contribution uses the shared record qualification. '
                              'Distinct URLs are not necessarily independent corroboration.',
            'sources': rows}


def source_progress_notice(manifest):
    progress = manifest.get('source_progress') or {}
    if not progress:
        return ''
    contribution = progress.get('task_requirement_contributing_sources')
    detail = '' if contribution is None else (
        f'Sources contributing to requested metric/scope requirements: {contribution}; this does not imply complete coverage. ')
    return (detail + f"Discovered candidate sources after deduplication: {progress['discovered_unique_sources']}; "
            f"readable sources: {progress['readable_sources']}. "
            f"Sources contributing qualified observations: {progress['qualified_evidence_sources']}; "
            f"candidate evidence only: {progress['candidate_only_sources']}; "
            f"not fetched: {progress['not_fetched_sources']}; "
            f"failed or unreadable: {progress['fetch_failed_sources']}; "
            f"budget-deferred sources: {progress.get('budget_deferred_sources', 0)}; "
            f"screening-excluded sources: {progress.get('screening_excluded_sources', 0)}; "
            f"awaiting extraction: {progress.get('awaiting_extraction_sources', 0)}; "
            f"recorded empty extraction results: {progress.get('extraction_empty_sources', 0)}; "
            f"extraction failures: {progress.get('extraction_failed_sources', 0)}; "
            f"extraction skipped: {progress.get('extraction_skipped_sources', 0)}; "
            f"readable sources with unrecorded extraction outcomes: {progress.get('missing_extraction_evidence_sources', 0)}; "
            f"acquisition incomplete: {progress.get('acquisition_incomplete_sources', 0)}. "
            'Discovery counts are not task-match counts or source recall; screening labels and actual evidence contribution are separate.')
