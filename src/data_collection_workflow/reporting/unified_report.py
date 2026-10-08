"""One offline English reading report, built only from recorded session evidence."""
from __future__ import annotations

import csv
import html
import json
import re
import zipfile
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

from ..task_result_report import LABELS, RESULT_KIND_LABELS, build_task_result
from .run_settings import build_run_settings, sanitize_configuration
from .source_catalog import build_source_catalog


DATASETS = {
    'final_dataset': ('Qualified observations', 'qualified observation records'),
    'candidate_records': ('Candidate records', 'candidate records requiring evidence checks'),
    'context_records': ('Context', 'context observation records'),
    'final_case_dataset': ('Individual case data', 'qualified individual case records'),
}


def _read(path, default):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _sanitize(value):
    return sanitize_configuration({'value': value})['value']


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def _csv(path, rows, fallback='record_id'):
    keys = list(dict.fromkeys(key for row in rows for key in row)) or [fallback]
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, default=str)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _text(value):
    if value is None:
        return ''
    return str(value) if not isinstance(value, (list, dict)) else json.dumps(value, ensure_ascii=False)


def _escape(value):
    return html.escape(_text(value), quote=True)


def _number(value):
    return f'{value:,}' if isinstance(value, (int, float)) and not isinstance(value, bool) else _text(value)


def _label(field):
    return LABELS.get(field, field.replace('_', ' ')).capitalize()


def _safe_url(value):
    value = _text(value).strip()
    try:
        parts = urlsplit(value)
        if parts.scheme.lower() in {'http', 'https'} and parts.netloc and not re.search(r'[\x00-\x20]', value):
            return value
    except ValueError:
        pass
    return None


def _source_refs(refs, catalog):
    result = []
    for ref in refs or []:
        # A record may contain fields from several sources. An explicit source
        # URL therefore outranks shared record membership in a different source.
        matched = next((source for source in catalog['sources'] if ref.get('url')
                        and ref['url'] in [source.get('url'), *(source.get('alias_urls') or [])]), None)
        found = matched['report_source_id'] if matched else None
        for source in catalog['sources'] if not found else []:
            record_ids = [item for values in (source.get('records_by_role') or {}).values() for item in values]
            record_ids += [item.get('record_id') for key in ('evidence_details', 'candidate_details')
                           for item in source.get(key) or [] if isinstance(item, dict)]
            if ref.get('record_id') and ref['record_id'] in record_ids:
                found = source['report_source_id']
                break
        result.append({**ref, 'report_source_id': found})
    return result


def _citations(refs):
    links = []
    seen = set()
    for ref in refs:
        source_id = ref.get('report_source_id')
        url = _safe_url(ref.get('url'))
        if source_id and source_id not in seen:
            links.append(f'<a class="citation" href="#source-{_escape(source_id)}">[{_escape(source_id)}]</a>')
            seen.add(source_id)
        elif not source_id and url and url not in seen:
            links.append(f'<a href="{_escape(url)}" target="_blank" rel="noopener noreferrer">Source</a>')
            seen.add(url)
    return ' '.join(links)


def _result_scope(result, task):
    start, end = result.get('period_start'), result.get('period_end')
    if result.get('result_kind') == 'full_period_total':
        start, end = task.get('start_date'), task.get('end_date')
    if start and end:
        text = f'Statistical period: {start} to {end}.'
    elif end:
        text = f'Statistical period ends {end}; the start date is not specified.'
    elif start:
        text = f'Statistical period starts {start}; the end date is not specified.'
    else:
        text = 'Exact statistical start and end dates are not specified.'
    if result.get('as_of_date'):
        text += f" Reported as of {result['as_of_date']}."
    if result.get('closure_date'):
        text += f" Outbreak closure event: {result['closure_date']}; this is not a statistical cutoff."
    return text


def _conclusions(task_result, catalog):
    results, cards, gaps = {}, [], []
    task = task_result['task']
    for field, answer in task_result['answers'].items():
        label = _label(field)
        conflict = answer.get('status') == 'conflict' or answer.get('scoped_result_conflict')
        selected = answer.get('selected_result')
        if conflict:
            result = {'status': 'conflict', 'value': None, 'result_kind': None, 'sources': []}
            entries = answer.get('conflicting_values') or answer.get('supported_results') or []
            details = []
            for item in entries:
                refs = _source_refs(item.get('sources'), catalog)
                details.append(f"{_escape(_number(item.get('value')))} {_citations(refs)}")
            gaps.append(f'<p><strong>{_escape(label)}:</strong> qualified evidence conflicts'
                        + (': ' + '; '.join(details) if details else '') + '. No single value is selected.</p>')
            content = '<strong>Conflicting evidence</strong><p class="caption">No single supported total is selected.</p>'
        elif selected or (answer.get('status') == 'confirmed' and answer.get('value') is not None):
            result = dict(selected or {'result_kind': 'full_period_total', 'value': answer['value'],
                                      'sources': answer.get('sources') or []})
            result['status'] = 'supported'
            result['sources'] = _source_refs(result.get('sources'), catalog)
            kind = RESULT_KIND_LABELS.get(result['result_kind'], 'Evidence-supported value')
            qualifier = {'over': 'over ', 'at_least': 'at least ', 'about': 'about ', 'nearly': 'nearly ',
                         'at_most': 'at most ', 'under': 'under '}.get(result.get('qualifier'), '')
            content = (f'<span class="pill">Evidence supported</span><p class="result-kind">{_escape(kind)}</p>'
                       f'<strong class="result-value">{_escape(qualifier + _number(result["value"]))}</strong>'
                       f'<p>{_citations(result["sources"])}</p>'
                       f'<p class="caption">{_escape(_result_scope(result, task))}</p>'
                       f'<p class="caption">{_escape(result.get("boundary_note"))}</p>')
            if result['result_kind'] != 'full_period_total':
                gaps.append(f'<p><strong>{_escape(label)}:</strong> a separate value covering the full requested period '
                            'is not confirmed. The supported result applies to the scope described above.</p>')
        else:
            result = {'status': 'unconfirmed', 'value': None, 'result_kind': None, 'sources': []}
            content = '<strong>Not confirmed</strong><p class="caption">The available qualified evidence does not establish this value. Missing does not mean zero.</p>'
            leads = [lead for lead in answer.get('candidate_leads') or [] if not lead.get('superseded_by_qualified_record_ids')]
            lead_text = []
            for lead in leads:
                lead_text.append(f"{_escape(_number(lead['value']))} {_citations(_source_refs(lead.get('sources'), catalog))}")
            gaps.append(f'<p><strong>{_escape(label)}:</strong> no evidence-supported value is available for the requested scope.'
                        + (' Unverified candidate leads: ' + '; '.join(lead_text) + '. These are not task answers.' if lead_text else '') + '</p>')
        results[field] = result
        cards.append(f'<article class="number" data-field="{_escape(field)}"><h3>{_escape(label)}</h3>{content}</article>')
    if not cards:
        cards.append('<p>No numeric measure is confirmed for this task. Consult the data and source catalogue for available observations.</p>')
    pending = [source for source in catalog['sources'] if (source.get('processing') or {}).get('code') in
               {'budget_deferred', 'acquisition_incomplete', 'acquisition_failed', 'not_attempted',
                'acquisition_in_progress', 'awaiting_extraction', 'extraction_failed'}]
    if pending:
        gaps.append(f'<p>{len(pending)} sources have unresolved acquisition or extraction work. '
                    'Unread or unextracted material may contain further information; inspect its processing status in the catalogue.</p>')
    return results, '<div class="numbers">' + ''.join(cards) + '</div>', gaps


def _first(*values):
    return next((value for value in values if value is not None and value != ''), None)


def _run_information(summary, state, session, manifest, settings):
    status = state.get('run_status') or summary.get('run_status') or _sanitize(_read(session / 'diagnostics/run_status.json', {}))
    ledger = state.get('run_budget_ledger') or summary.get('run_budget_ledger') or summary.get('budget') or manifest.get('budget') or {}
    start = _first(status.get('started_at'), status.get('started_at_utc'), summary.get('started_at_utc'))
    end = _first(status.get('completed_at'), status.get('completed_at_utc'), status.get('finished_at'), summary.get('completed_at_utc'))
    seconds = _first(status.get('duration_seconds'), status.get('elapsed_seconds'), summary.get('elapsed_seconds'))
    if seconds is None and isinstance(status.get('duration_ms'), (int, float)):
        seconds = status['duration_ms'] / 1000
    if seconds is None and start and end:
        try:
            seconds = (datetime.fromisoformat(end.replace('Z', '+00:00')) - datetime.fromisoformat(start.replace('Z', '+00:00'))).total_seconds()
        except (ValueError, TypeError):
            pass
    duration = f'{seconds:,.1f} seconds' if isinstance(seconds, (int, float)) and seconds >= 0 else 'Duration not recorded'
    display = f'{start or "Start not recorded"} to {end or "End not recorded"} · {duration}'
    stop = _first(manifest.get('recovery_stop_reason'), summary.get('recovery_stop_reason'),
                  summary.get('stop_reason'), status.get('stop_reason'), state.get('recovery_stop_reason'))
    used, limits = ledger.get('used') or {}, ledger.get('limits') or {}
    budget_rows = [{'key': key, 'label_en': key.replace('_', ' ').capitalize(), 'limit': limits.get(key),
                    'used': used.get(key), 'unit_en': ''} for key in dict.fromkeys([*limits, *used])]
    if not budget_rows:
        budget_rows = [{'key': 'unknown', 'label_en': 'Resource budget', 'limit': None, 'used': None, 'unit_en': ''}]
    model = _first(summary.get('observed_model'), state.get('observed_model'))
    model_basis = 'Observed execution' if model else None
    if not model:
        model = _first(summary.get('model'), state.get('model'))
        model_basis = 'Recorded run configuration' if model else 'Not recorded'
    if not model:
        model_row = next((row for group in settings.get('groups') or [] for row in group.get('rows') or []
                          if row['key'] == 'llm.model' and row.get('value_status') in {'recorded_effective', 'recorded_configuration'}), {})
        model = model_row.get('effective_value')
        if model:
            model_basis = model_row.get('basis') or 'Recorded session setting'
    cost = _first(ledger.get('cost_usd'), ledger.get('total_cost_usd'), summary.get('total_cost_usd'))
    return {'runtime': {'started_at': start, 'completed_at': end, 'duration_seconds': seconds,
                        'display_en': display, 'stop_reason_en': str(stop).replace('_', ' ') if stop else 'Not recorded'},
            'models': {'display_en': _text(model) if model else 'Not recorded', 'basis': model_basis},
            'budget': {'rows': budget_rows, 'cost_display_en': f'USD {cost:,.4f}' if isinstance(cost, (int, float)) else 'Not recorded'}}


def _evidence_files(session, source_session, package, state, catalog, results):
    """Save source-local quotations and optional selected-document text, never logs."""
    paths = []
    selected_hashes = {ref.get('document_hash') for result in results.values()
                       for ref in result.get('sources') or [] if ref.get('document_hash')}
    documents = list(package.get('documents') or []) + list(state.get('documents') or [])
    for digest in selected_hashes:
        if re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', str(digest)):
            candidate = source_session / 'acquisition' / f'{digest}.json'
            if candidate.is_file() and candidate.resolve().is_relative_to(source_session.resolve()):
                doc = _sanitize(_read(candidate, {}))
                if isinstance(doc, dict):
                    documents.append(doc)
    for source in catalog['sources']:
        details = source.get('evidence_details') or []
        candidates = source.get('candidate_details') or []
        if not details and not candidates:
            continue
        source_id = source['report_source_id']
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', source_id):
            continue
        path = session / 'evidence' / f'{source_id}_evidence.json'
        _json(path, {'report_source_id': source_id, 'url': source.get('url'),
                     'evidence_details': details, 'candidate_details': candidates})
        paths.append(path)
        source['evidence_files'] = [{'label': 'Saved evidence and locators', 'path': path.relative_to(session).as_posix()}]
        hashes = {field.get('document_hash') for item in details for field in item.get('supported_fields') or []}
        for doc in documents:
            digest = doc.get('content_hash') or doc.get('document_hash')
            if digest not in selected_hashes or digest not in hashes:
                continue
            text = doc.get('clean_text') or doc.get('text')
            if not text:
                continue
            # Only selected text and locator metadata are portable; original local paths are not.
            doc_path = session / 'evidence' / f'{source_id}_document.json'
            text_path = session / 'evidence' / f'{source_id}_saved_text.txt'
            text_path.write_text(str(text), encoding='utf-8')
            doc_data = {key: doc[key] for key in ('document_id', 'content_hash', 'document_hash', 'source_id',
                         'url', 'source_url', 'title', 'publication_date', 'pages', 'tables', 'chunks') if key in doc}
            doc_data['text_artifact_path'] = text_path.name
            _json(doc_path, doc_data)
            paths.extend([doc_path, text_path])
            source['evidence_files'].extend([{'label': 'Saved source text', 'path': text_path.relative_to(session).as_posix()},
                                            {'label': 'Document metadata', 'path': doc_path.relative_to(session).as_posix()}])
            break
    return paths


def write_unified_report(session_dir: Path | str, package: dict | None = None,
                         summary: dict | None = None, *, config: dict | None = None,
                         state: dict | None = None, source_session_dir: Path | str | None = None) -> dict:
    """Write one English report and an allowlisted, self-contained evidence bundle.

    Collection exports remain untouched. Rebuilding always derives conclusions and
    record counts from the supplied current package, not old task-result reports.
    """
    session = Path(session_dir)
    session.mkdir(parents=True, exist_ok=True)
    source_session = Path(source_session_dir) if source_session_dir is not None else session
    collection = source_session / 'collection'
    explicit_settings = config is not None or state is not None
    if config is None:
        config = _read(source_session / 'run_config.json', {})
    package = _sanitize(_read(collection / 'final_package.json', {}) if package is None else package)
    summary = _sanitize(_read(source_session / 'workflow_run_summary.json', {}) if summary is None else summary)
    config, state = _sanitize(config or {}), _sanitize(state or {})
    legacy_candidates = 'candidate_records' not in package
    if legacy_candidates:
        candidates, seen_candidates = [], set()
        for key in ('pending_review_records', 'quarantined_records'):
            for row in package.get(key) or []:
                identity = row.get('record_id') or json.dumps(row, sort_keys=True, default=str)
                if identity not in seen_candidates:
                    seen_candidates.add(identity)
                    candidates.append(row)
        package['candidate_records'] = candidates
    for key in ('evidence_chunks', 'source_inventory', 'source_registry', 'source_processing_status', 'excluded_sources'):
        if key not in package and key not in state:
            saved_path = collection / f'{key}.json'
            if saved_path.is_file():
                state[key] = _sanitize(_read(saved_path, []))
    manifest = dict(package.get('result_manifest') or {})
    qualified_count = sum((row.get('evidence_qualification') or {}).get('status') == 'qualified'
                          for row in package.get('final_dataset') or [])
    manifest['counts'] = {
        'qualified_observations': qualified_count,
        'individual_case_records': len(package.get('final_case_dataset') or []),
        'aggregate_observations': len(package.get('aggregate_dataset') or []),
        'candidate_records': len(package.get('candidate_records') or []),
        'context_records': len(package.get('context_records') or []),
        'discovered_sources': len(package.get('source_registry') or []),
    }
    manifest['dataset_counts'] = {key: len(package.get(key) or [])
                                  for key in dict.fromkeys([*(manifest.get('dataset_counts') or {}), *DATASETS])}
    manifest['task'] = (manifest.get('task') or state.get('structured_task') or package.get('structured_task')
                        or config.get('structured_task') or config.get('task')
                        or _sanitize(_read(collection / 'structured_task.json', {}))
                        or _sanitize(_read(source_session / 'diagnostics/structured_task.json', {})))
    task_result = build_task_result(package, manifest)
    catalog = build_source_catalog(package, state=state)
    # This is the previously redacted inventory, not a raw configuration. Its
    # credential_redacted field is a count, so it must not be masked as a secret.
    saved_settings = _read(source_session / 'data/run_settings.json', {}) if not explicit_settings else {}
    settings = saved_settings if saved_settings.get('groups') else build_run_settings(config=config, summary=summary, state=state)
    results, conclusion, gaps = _conclusions(task_result, catalog)
    data = session / 'data'
    data.mkdir(parents=True, exist_ok=True)
    bundle_files = []
    download_rows = []
    datasets = dict(DATASETS)
    if qualified_count < len(package.get('final_dataset') or []):
        datasets['final_dataset'] = ('Collected observations', f'observation records; {qualified_count:,} passed evidence qualification')
    if any((row.get('evidence_qualification') or {}).get('status') != 'qualified' for row in package.get('final_case_dataset') or []):
        datasets['final_case_dataset'] = ('Individual case data', 'individual case records retained by the run; evidence qualification is recorded separately')
    if legacy_candidates and package['candidate_records']:
        datasets['candidate_records'] = ('Candidate and excluded records', 'records pending review or excluded by the run checks')
        for key, label in [('pending_review_records', 'Pending review'), ('quarantined_records', 'Excluded records')]:
            if key in package:
                datasets[key] = (label, 'records with their original review or exclusion reasons')
    for key, (label, description) in datasets.items():
        rows = list(package.get(key) or [])
        for extension in ('json', 'csv'):
            path = data / f'{key}.{extension}'
            (_json if extension == 'json' else _csv)(path, rows)
            bundle_files.append(path)
        download_rows.append(f'<tr><td>{label}</td><td><strong>{len(rows):,}</strong> {description}.</td>'
                             f'<td><a href="data/{key}.csv">CSV</a> · <a href="data/{key}.json">JSON</a></td></tr>')
    all_rows = [row for key in DATASETS for row in package.get(key) or []]
    saved_decisions = {row.get('record_id'): row for row in package.get('record_inclusion_decisions') or [] if isinstance(row, dict)}
    decisions, seen = [], set()
    for row in all_rows:
        record_id = row.get('record_id')
        if record_id and record_id in seen:
            continue
        if record_id:
            seen.add(record_id)
        decision = dict(saved_decisions.get(record_id) or {})
        decision.update(record_id=record_id, product_kind=row.get('product_kind'))
        if row.get('evidence_qualification'):
            decision.update(row['evidence_qualification'])
        elif not decision.get('status'):
            decision['status'] = row.get('record_final_inclusion_status') or decision.get('record_final_inclusion_status') or 'not_recorded'
        decisions.append(decision)
    chunks = package.get('evidence_chunks')
    if chunks is None:
        chunks = state.get('evidence_chunks')
    if chunks is None:
        chunks = _sanitize(_read(collection / 'evidence_chunks.json', []))
    for key, rows in [('evidence_chunks', chunks), ('record_inclusion_decisions', decisions)]:
        _json(data / f'{key}.json', rows)
        bundle_files.append(data / f'{key}.json')
        if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
            _csv(data / f'{key}.csv', rows)
            bundle_files.append(data / f'{key}.csv')
    evidence_paths = _evidence_files(session, source_session, package, state, catalog, results)
    bundle_files.extend(evidence_paths)
    snapshot = {'schema_version': 'unified-session-report/1',
                'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'task': task_result['task'], 'results': results, 'task_result': task_result,
                'result_manifest': manifest,
                'record_counts': {**{key: len(package.get(key) or []) for key in DATASETS},
                                  'qualified_observations': qualified_count,
                                  'individual_cases': len(package.get('final_case_dataset') or [])},
                'source_counts': {**(catalog.get('metadata') or {}), 'total_sources': len(catalog['sources'])},
                'run_information': _run_information(summary, state, source_session, manifest, settings),
                'source_catalog_file': 'source_catalog.json', 'run_settings_file': 'run_settings.json'}
    for key, value in [('report_snapshot', snapshot), ('source_catalog', catalog), ('run_settings', settings), ('task_result', task_result)]:
        _json(data / f'{key}.json', value)
        bundle_files.append(data / f'{key}.json')
    settings_rows = [{**row, 'group_id': group['id'], 'group_label': group['label']}
                     for group in settings.get('groups') or [] for row in group.get('rows') or []]
    _csv(data / 'source_catalog.csv', catalog['sources'], 'report_source_id')
    _csv(data / 'run_settings.csv', settings_rows, 'key')
    bundle_files.extend([data / 'source_catalog.csv', data / 'run_settings.csv'])
    download_rows += [f'<tr><td>Complete source catalogue</td><td>{len(catalog["sources"]):,} sources, with processing statuses and evidence contributions.</td><td><a href="data/source_catalog.csv">CSV</a> · <a href="data/source_catalog.json">JSON</a></td></tr>',
                      f'<tr><td>Evidence and inclusion decisions</td><td>Saved passages and assessments for {len(decisions):,} records.</td><td><a href="data/evidence_chunks.json">Evidence</a> · <a href="data/record_inclusion_decisions.json">Decisions</a></td></tr>',
                      '<tr><td>Report snapshot</td><td>Task answers, record counts and recorded run information.</td><td><a href="data/report_snapshot.json">JSON</a></td></tr>']
    task = task_result['task']
    title = ' in '.join(str(value) for value in (task.get('disease'), task.get('location')) if value) or 'Data collection report'
    scope = ' · '.join(str(value) for value in [
        f"{task.get('start_date') or 'Start not specified'} to {task.get('end_date') or 'End not specified'}",
        task.get('location'), task.get('disease')] if value)
    request = _first(config.get('user_request'), summary.get('user_request'), state.get('user_request'), task.get('user_request'))
    if not request:
        fields = ', '.join(_label(field).lower() for field in task.get('target_fields') or [])
        request = 'Collect ' + (fields or 'available observations') + ' with source references and supporting evidence.'
    gap_html = '<section id="gaps"><h2>Unresolved questions</h2><div class="notice">' + ''.join(gaps) + '</div></section>' if gaps else ''
    payload = json.dumps({'snapshot': snapshot, 'catalog': catalog, 'settings': settings},
                         ensure_ascii=False, separators=(',', ':'), default=str)
    payload = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    template = files('data_collection_workflow').joinpath('resources/unified_report.html').read_text(encoding='utf-8')
    replacements = {'TITLE': _escape(title), 'SCOPE': _escape(scope), 'REQUEST': _escape(request),
                    'CONCLUSIONS': conclusion, 'GAPS': gap_html, 'GAP_NAV': '<a href="#gaps">Unresolved questions</a>' if gaps else '',
                    'DOWNLOADS': ''.join(download_rows), 'GENERATED': _escape(snapshot['generated_at']), 'PAYLOAD': payload}
    # One substitution pass prevents user text containing a template marker from being interpreted.
    rendered = re.sub(r'@@([A-Z_]+)@@', lambda match: replacements[match.group(1)], template)
    report_path = session / 'final_report.html'
    report_path.write_text(rendered, encoding='utf-8')
    bundle_path = session / 'session_report.zip'
    with zipfile.ZipFile(bundle_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in dict.fromkeys([report_path, *bundle_files]):
            if not path.resolve().is_relative_to(session.resolve()):
                raise ValueError('Report bundle member is outside the session directory')
            archive.write(path, path.relative_to(session).as_posix())
    from .output_contract import retire_legacy_reading_files
    retire_legacy_reading_files(session)
    return {'final_report_english': str(report_path), 'final_report': str(report_path),
            'report_bundle': str(bundle_path), 'report_snapshot_json': str(data / 'report_snapshot.json'),
            'final_report_facts': str(data / 'report_snapshot.json'),
            'source_catalog_json': str(data / 'source_catalog.json'), 'source_catalog_csv': str(data / 'source_catalog.csv'),
            'run_settings_json': str(data / 'run_settings.json'), 'run_settings_csv': str(data / 'run_settings.csv')}
