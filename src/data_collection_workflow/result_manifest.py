"""One immutable set of result facts for evidence files and human-facing summaries."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path

from .source_progress import build_source_progress, source_progress_notice


def _counts(package):
    return {'qualified_observations':len(package.get('final_dataset') or []),
            'individual_case_records':len(package.get('final_case_dataset') or []),
            'aggregate_observations':len(package.get('aggregate_dataset') or []),
            'candidate_records':len(package.get('candidate_records') or []),
            'context_records':len(package.get('context_records') or []),
            'discovered_sources':len(package.get('source_registry') or [])}

# Dataset aliases are observations, never inferred patient totals.
OUTPUT_DATASETS = (
    'final_dataset', 'final_case_dataset', 'aggregate_dataset', 'candidate_records',
    'context_records', 'primary_case_dataset', 'task_aware_observation_dataset',
    'reviewable_dataset', 'final_dataset_post_review', 'pending_review_records',
    'quarantined_records', 'non_primary_observations', 'best_available_context_records',
    'case_candidate_dataset', 'reviewable_case_dataset', 'workflow_case_candidate_line_list',
    'workflow_case_bundle_line_list', 'case_evidence_bundles',
    'global_outbreak_event_dataset', 'regional_surveillance_dataset', 'country_year_aggregate_dataset',
    'official_alert_dataset', 'probable_case_dataset', 'suspected_case_dataset',
    'unspecified_case_dataset', 'death_dataset', 'hospitalization_dataset',
    'zero_case_statements', 'exposure_monitoring_records', 'surveillance_summary_records',
    'outbreak_summary_records', 'unclassified_observation_records')
SUMMARY_SECTIONS = ('run_quality_summary', 'final_dataset_quality_summary',
                    'collection_decision_summary', 'direct_collection_summary',
                    'observation_type_dataset_summary', 'finalization_summary')


def normalize_result_views(package):
    """Bind all accepted/reviewable aliases to the final qualified product split."""
    final = list(package.get('final_dataset') or [])
    cases = list(package.get('final_case_dataset') or [])
    aggregates = list(package.get('aggregate_dataset') or [])
    context = list(package.get('context_records') or [])
    candidates = list(package.get('candidate_records') or [])
    package.update(primary_case_dataset=cases, task_aware_observation_dataset=aggregates + context,
                   reviewable_dataset=candidates, final_dataset_post_review=final,
                   pending_review_records=candidates, quarantined_records=candidates,
                   non_primary_observations=aggregates + context, best_available_context_records=context)
    # Legacy patient-promotion products do not establish evidence qualification.
    for key in ('case_candidate_dataset', 'reviewable_case_dataset', 'workflow_case_candidate_line_list',
                'workflow_case_bundle_line_list', 'case_evidence_bundles'):
        package[key] = []
    # Legacy typed views may retain a subset, but never stale/rejected row values.
    accepted = {row.get('record_id'): row for row in final + context if row.get('record_id')}
    for key in OUTPUT_DATASETS[18:]:
        package[key] = [accepted[row['record_id']] for row in package.get(key) or []
                        if row.get('record_id') in accepted]
    return package


def manifest_summary(manifest):
    c = manifest['counts']; datasets = manifest.get('dataset_counts') or {}
    final, cases, aggregates, candidates = (c[k] for k in ('qualified_observations', 'individual_case_records', 'aggregate_observations', 'candidate_records'))
    summary = {key: manifest[key] for key in ('generated_at', 'pipeline_mode', 'technical_completion', 'data_availability',
               'quality_status', 'coverage_status', 'recovery_stop_reason', 'human_review_status', 'release_status')}
    summary['acquisition']=manifest.get('acquisition') or {}
    summary['source_progress']=manifest.get('source_progress') or {}
    summary['collection_status']=manifest.get('collection_status', 'not_assessed')
    summary['provider_stops']=manifest.get('provider_stops') or {}
    summary.update(final_dataset_count=final, final_record_count=final, accepted_record_count=final,
                   qualified_record_count=final, final_case_dataset_count=cases, final_case_record_count=cases,
                   primary_case_dataset_count=cases, primary_case_dataset_eligible_count=cases,
                   accepted_primary_case_record_count=cases, accepted_non_primary_observation_count=aggregates,
                   aggregate_dataset_count=aggregates, aggregate_record_count=aggregates,
                   candidate_record_count=candidates, pending_review_record_count=candidates,
                   quarantined_record_count=candidates, reviewable_record_count=candidates,
                   context_record_count=c['context_records'], non_primary_observation_count=aggregates + c['context_records'],
                   post_review_record_count=final, final_dataset_post_review_count=final,
                   primary_case_dataset_status='available' if cases else 'empty', no_primary_case_dataset_records=not cases,
                   accepted_records_are_not_primary_case_records=bool(final and not cases),
                   run_quality_status=manifest['quality_status'], user_facing_run_status=manifest['data_availability'],
                   recommended_user_message=f'Qualified observations: {final}; individual case records: {cases}; aggregate observations: {aggregates}; candidates: {candidates}. Independent release evaluation is not completed.',
                   dataset_view_counts=dict(datasets), patient_count=None, result_manifest=manifest)
    for alias, key in {'zero_case_statement_count': 'zero_case_statements', 'exposure_monitoring_record_count': 'exposure_monitoring_records',
                       'surveillance_summary_record_count': 'surveillance_summary_records', 'outbreak_summary_record_count': 'outbreak_summary_records',
                       'unclassified_observation_count': 'unclassified_observation_records'}.items():
        summary[alias] = datasets.get(key, 0)
    return summary


def synchronize_package_summary(package):
    normalize_result_views(package)
    summary = manifest_summary(package['result_manifest'])
    for key in SUMMARY_SECTIONS:
        package[key] = dict(summary)
    package['package_metadata'] = {**(package.get('package_metadata') or {}), **summary}
    package['workflow_summaries'] = {key: dict(summary) for key in SUMMARY_SECTIONS}
    package['release_status'] = package['result_manifest']['release_status']
    source_status = {source_id: row for row in package['result_manifest'].get('source_progress', {}).get('sources', [])
                     for source_id in row['source_ids']}
    for view in ('source_registry', 'excluded_sources'):
        for entry in package.get(view) or []:
            row = source_status.get(str(entry.get('source_id')))
            if row:
                entry.update(processing_status=row['processing_status'], processing_reason=row['processing_reason'],
                             acquisition_status=row['acquisition_status'], evidence_contribution_status=row['evidence_status'])
    export_manifest = package.get('export_manifest') or {}
    export_manifest['record_counts'] = {key: len(package.get(key) or []) for key in OUTPUT_DATASETS}
    package['export_manifest'] = export_manifest
    return summary

def _acquisition_summary(state, progress=None):
    from .workflow_recovery import _readable_document
    from .source_progress import _budget_deferred, unresolved_source_documents
    progress = progress if progress is not None else build_source_progress({}, state)
    source_rows = progress.get('sources') or []
    unresolved = unresolved_source_documents(state.get('documents') or [], source_rows)
    deferred = [doc for doc in unresolved if _budget_deferred(doc)]
    deferred_sources = [row for row in source_rows if row['processing_status'] == 'budget_deferred']
    frontier = state.get('acquisition_frontier') or {}
    causes = Counter(str(doc.get('budget_exhausted_kind') or 'unknown') for doc in deferred)
    for job in frontier.get('items') or []:
        if job.get('status') == 'budget_deferred' and not any(doc.get('source_id') == job.get('source_id') for doc in deferred):
            causes[str(job.get('reason') or 'unknown')] += 1
    rows = [{'source_id':doc.get('source_id'), 'document_id':doc.get('document_id'),
             'content_hash':doc.get('content_hash'), 'budget_exhausted_kind':doc.get('budget_exhausted_kind'),
             'content_readable':_readable_document(doc), 'acquisition_incomplete':bool(doc.get('acquisition_incomplete')),
             'unprocessed_pages':sorted(set(doc.get('unprocessed_pages') or []))} for doc in unresolved]
    source_ids = {str(doc['source_id']) for doc in unresolved if doc.get('source_id')}
    for row in source_rows:
        if row['processing_status'] in {'budget_deferred', 'acquisition_incomplete', 'acquisition_failed',
                                         'not_attempted', 'acquisition_in_progress'}:
            source_ids.update(row['source_ids'])
    return {'status':'partial' if source_ids or unresolved else 'no_unresolved_budget_deferrals',
            'budget_deferred_document_count':len(deferred),
            'budget_deferred_source_count':len(deferred_sources),
            'incomplete_document_count':sum(bool(doc.get('acquisition_incomplete')) for doc in unresolved),
            'partially_readable_document_count':sum(_readable_document(doc) for doc in unresolved),
            'budget_exhausted_causes':dict(sorted(causes.items())),
            'unresolved_source_ids':sorted(source_ids),
            'unprocessed_page_count':sum(len(row['unprocessed_pages']) for row in rows),
            'unresolved_documents':rows}


def _acquisition_notice(manifest):
    acquisition=manifest.get('acquisition') or {}
    if acquisition.get('status') != 'partial':
        return ''
    count=acquisition['budget_deferred_document_count']
    source_count=acquisition.get('budget_deferred_source_count',0)
    partial=acquisition['partially_readable_document_count']
    incomplete=acquisition.get('incomplete_document_count',0)
    pages=acquisition['unprocessed_page_count']
    causes=json.dumps(acquisition['budget_exhausted_causes'],ensure_ascii=False,sort_keys=True)
    sources=', '.join(acquisition['unresolved_source_ids'])
    return f'Acquisition incomplete: {source_count} sources and {count} documents deferred by budget; {incomplete} incomplete documents, including {partial} partially readable documents; {pages} pages unprocessed. Budget causes: {causes}. Unresolved sources: {sources}. Available qualified data are retained.'


def build_result_manifest(package,state):
    normalize_result_views(package)
    counts=_counts(package)
    if counts['qualified_observations']:
        availability='qualified_aggregates_without_individual_cases' if not counts['individual_case_records'] else 'qualified_data_available'
    else:
        availability='candidates_only' if counts['candidate_records'] else 'no_usable_evidence'
    coverage=package.get('source_coverage_audit') or {}
    progress=build_source_progress(package,state)
    provider_stops={name: row for name, row in ((state.get('run_budget_ledger') or {}).get('providers') or {}).items()
                    if row.get('status') == 'halted'}
    acquisition=_acquisition_summary(state,progress)
    unfinished = bool(provider_stops or state.get('recovery_gaps')
                      or acquisition.get('status') == 'partial'
                      or 'budget_exhausted' in str(state.get('recovery_stop_reason') or '')
                      or any(row.get('processing_status') == 'awaiting_extraction' for row in progress.get('sources') or []))
    collection_status = 'partial' if unfinished else 'completed' if coverage.get('coverage_complete') is True else 'not_assessed'
    return {'manifest_version':'universal-result/3.2','pipeline_mode':'evidence',
            'generated_at':datetime.now(timezone.utc).isoformat(), 'technical_completion':'completed',
            'data_availability':availability, 'quality_status':'qualified_under_evidence_contract' if counts['qualified_observations'] else 'no_qualified_data',
            'coverage_status':coverage.get('coverage_status') or ('partial' if coverage.get('uncovered_requirement_ids') else 'not_assessed'),
            'recovery_stop_reason':'provider_account_limit' if provider_stops else state.get('recovery_stop_reason') or 'not_run',
            'collection_status':collection_status, 'provider_stops':provider_stops,
            'human_review_status':('pending' if state.get('human_review_enabled') else 'disabled_with_candidates') if counts['candidate_records'] else 'no_pending_candidates',
            'release_status':'not_evaluated','counts':counts,'acquisition':acquisition,
            'source_progress':progress,
            'dataset_counts':{key:len(package.get(key) or []) for key in OUTPUT_DATASETS},'budget':state.get('run_budget_ledger') or {},
            'task':state.get('structured_task') or {},'patient_count':None,
            'limitations':['Record counts are observations, not patient totals.','Qualification is a rule-based evidence decision; independent factual release evaluation is separate.']}

def _provider_notice(manifest):
    if not manifest.get('provider_stops'):
        return ''
    return 'Collection is partial (provider_account_limit): model-provider account allowance is unavailable. New model requests stopped; saved data and pending work are retained. Explicit same-version resume acknowledgement is required after account restoration.'


def _report(manifest):
    c=manifest['counts']
    lines=['# Disease collection results','',f"Generated: {manifest['generated_at']}",'',f"Qualified observations: {c['qualified_observations']}; individual case records: {c['individual_case_records']}; aggregate observations: {c['aggregate_observations']}; candidate evidence: {c['candidate_records']}.",'',f"Execution: {manifest['technical_completion']}. Availability: {manifest['data_availability']}.",f"Evidence qualification: {manifest['quality_status']}. Coverage: {manifest['coverage_status']}.",f"Recovery stop: {manifest['recovery_stop_reason']}. Human review: {manifest['human_review_status']}.",'', 'Observation rows are not patient totals. Independent factual and generalization release evaluation has not been completed.']
    provider_notice=_provider_notice(manifest)
    if provider_notice:lines+=['',provider_notice]
    sources=source_progress_notice(manifest)
    if sources: lines+=['',sources]
    notice=_acquisition_notice(manifest)
    if notice: lines+=['',notice]
    if manifest.get('budget'):
        lines+=['','Budget consumption: '+json.dumps(manifest['budget'].get('used') or {},ensure_ascii=False,sort_keys=True)]
    return '\n'.join(lines)+'\n'


def _english_artifact_links(links):
    """Drop obsolete Chinese report routes from serialized artifact metadata."""
    return {key: _english_artifact_links(value) if isinstance(value, dict) else value
            for key, value in links.items() if 'chinese' not in key.lower()}


def write_universal_outputs(package,output_dir):
    manifest=package.get('result_manifest')
    if not manifest or manifest.get('counts')!=_counts(package):
        raise ValueError('result manifest disagrees with package datasets')
    if manifest['counts']['qualified_observations'] != manifest['counts']['individual_case_records']+manifest['counts']['aggregate_observations']:
        raise ValueError('manifest admitted product views do not partition final observations')
    encode = lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    if Counter(map(encode, package.get('final_dataset') or [])) != Counter(map(encode, (package.get('final_case_dataset') or []) + (package.get('aggregate_dataset') or []))):
        raise ValueError('manifest product views do not partition the same qualified observations')
    synchronize_package_summary(package)
    if manifest.get('dataset_counts') != {key:len(package.get(key) or []) for key in OUTPUT_DATASETS}:
        raise ValueError('manifest dataset alias counts disagree with package')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    from .export import write_json,write_csv_rows
    paths={}
    for key in OUTPUT_DATASETS:
        rows=package.get(key) or []
        paths[key+'_json']=str(write_json(rows,out/(key+'.json')))
        paths[key+'_csv']=str(write_csv_rows(rows,out/(key+'.csv')))
    for key in SUMMARY_SECTIONS:
        paths[key+'_json']=str(write_json(package[key],out/(key+'.json')))
    sources = manifest.get('source_progress', {}).get('sources', [])
    paths['source_processing_status_json'] = str(write_json(sources, out/'source_processing_status.json'))
    paths['source_processing_status_csv'] = str(write_csv_rows(sources, out/'source_processing_status.csv'))
    paths['source_registry_json'] = str(write_json(package.get('source_registry') or [], out/'source_registry.json'))
    paths['result_manifest']=str(write_json(manifest,out/'result_manifest.json'))
    paths['final_package_json']=str(out/'final_package.json')
    (out/'final_report.md').write_text(_report(manifest),encoding='utf-8')
    paths['final_report.md']=str(out/'final_report.md')
    rows=''.join('<tr><th>'+html.escape(k)+'</th><td>'+str(v)+'</td></tr>' for k,v in manifest['counts'].items())
    console = '<!doctype html><meta charset="utf-8"><title>Disease collection results</title><h1>Disease collection results</h1><table>'+rows+'</table><p>'+html.escape(manifest['data_availability'])+'</p><p>Independent release evaluation: '+html.escape(manifest['release_status'])+'</p>'
    provider_notice=_provider_notice(manifest)
    if provider_notice:console+='<p>'+html.escape(provider_notice)+'</p>'
    sources=source_progress_notice(manifest)
    if sources: console+='<p>'+html.escape(sources)+'</p>'
    notice=_acquisition_notice(manifest)
    if notice: console+='<p>'+html.escape(notice)+'</p>'
    (out/'workflow_console.html').write_text(console,encoding='utf-8')
    paths['workflow_console_html'] = str(out/'workflow_console.html')
    paths['workflow_console_summary_json'] = str(write_json(manifest_summary(manifest), out/'workflow_console_summary.json'))
    paths['evidence_products_json'] = str(write_json(package.get('evidence_products') or {},out/'evidence_products.json'))
    artifact_manifest = _english_artifact_links(package.get('artifact_manifest') or {'files':{}, 'section_counts':{}})
    package['artifact_manifest'] = artifact_manifest
    artifact_manifest.setdefault('files', {}).update(paths)
    artifact_manifest.setdefault('section_counts', {}).update(manifest['dataset_counts'])
    write_json(package, out/'final_package.json')
    return paths


def write_universal_run_outputs(package, summary, output_dir):
    """Final runner projection: reports, aliases, facts, console and stdout source."""
    from .export import write_json
    from .task_result_report import build_task_result, write_task_result_artifacts
    out = Path(output_dir)
    collection = (summary.get('artifact_paths') or {}).get('collection_manifest') or {}
    if isinstance(collection, dict) and collection.get('files'):
        package['artifact_manifest'] = {'files':dict(collection['files']), 'section_counts':dict(collection.get('section_counts') or {})}
    task_result = build_task_result(package, package['result_manifest'])
    task_paths = write_task_result_artifacts(task_result, out)
    package.setdefault('artifact_manifest', {'files':{}, 'section_counts':{}}).setdefault('files', {}).update(task_paths)
    paths = write_universal_outputs(package, out/'collection')
    manifest = package['result_manifest']
    projection = manifest_summary(manifest)
    summary.update(projection)
    for key in SUMMARY_SECTIONS:
        summary[key] = dict(projection)
    artifacts = summary.setdefault('artifact_paths', {})
    # Preserve existing console paths until both session and latest aliases are corrected.
    report_keys = ('run_report', 'stable_run_report', 'final_report', 'stable_final_report',
                   'interpretive_report', 'stable_interpretive_report', 'final_report_chinese', 'final_report_english',
                   'stable_final_report_chinese', 'stable_final_report_english', 'interpretive_report_chinese',
                   'interpretive_report_english', 'stable_interpretive_report_chinese', 'stable_interpretive_report_english')
    for key in report_keys:
        if artifacts.get(key):
            Path(artifacts[key]).write_text(_report(manifest), encoding='utf-8')
    for key in list(artifacts):
        if 'chinese' in key.lower():
            del artifacts[key]
        elif isinstance(artifacts[key], dict):
            artifacts[key] = _english_artifact_links(artifacts[key])
    for key in ('final_report_facts', 'stable_final_report_facts', 'interpretive_report_summary', 'stable_interpretive_report_summary',
                'workflow_console_summary_json', 'latest_workflow_console_summary_json'):
        if artifacts.get(key):
            write_json(projection, artifacts[key])
    console = (out/'collection/workflow_console.html').read_text(encoding='utf-8')
    for key in ('workflow_console_html', 'latest_workflow_console_html'):
        if artifacts.get(key):
            Path(artifacts[key]).write_text(console, encoding='utf-8')
    artifacts.update({key:value for key,value in paths.items() if key not in artifacts})
    artifacts.update(task_paths)
    artifacts['universal_result_manifest'] = str(write_json(manifest, out/'result_manifest.json'))
    summary['task_result_summary'] = {
        'headline': task_result['headline'],
        'answer_status': {field: answer['status'] for field, answer in task_result['answers'].items()},
    }
    write_json(summary, out/'workflow_run_summary.json')
    return summary
