"""Adaptive acquisition ordering and session-local result references.

Priority is a retrieval estimate; it never qualifies a source or a fact.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from urllib.parse import urlsplit


def adaptive_runtime():
    from .session_runtime import get_runtime
    from .acquisition_budget import is_adaptive_budget
    runtime = get_runtime()
    return runtime if runtime and is_adaptive_budget(runtime.config) else None


def source_family(entry):
    return str(entry.get('source_independence_group') or
               urlsplit(entry.get('canonical_url') or entry.get('url') or '').hostname or '')


def acquisition_priority(entry, *, family_counts=None, state=None):
    """Favor task products and unmet families, never a PDF suffix or official badge."""
    fit = str(entry.get('target_verification_status') or entry.get('target_fit_status') or '')
    if (state is not None and entry.get('task_fit_evidence_origin') == 'discovery_metadata'
            and fit in {'temporal_mismatch', 'task_fit_mismatch'}
            and entry.get('disease_fit') not in {'mismatch', 'wrong_disease'}
            and entry.get('geography_fit') not in {'mismatch', 'wrong_geography'}):
        # Reuse the source-statement contract for ranking stale metadata. A
        # publication year cannot permanently demote an overlapping report;
        # explicit/received-content contradictions remain untouched.
        from .nodes.source_screening import _source_positive_target_verification
        current = _source_positive_target_verification(entry, state)
        fit = current['target_verification_status']
    score = 60 if fit.startswith('verified') else 30
    if fit in {'unrelated_disease','temporal_mismatch','task_fit_mismatch'}:
        score -= 100
    product = str(entry.get('data_product_type') or '')
    if product in {'structured_dataset','downloadable_dataset','surveillance_dataset','line_list'}:
        score += 45
    elif any(word in product for word in ('surveillance','statistical','report','dashboard','database')):
        score += 25
    elif product in {'research_article','case_report_article'}:
        score += 15
    if entry.get('source_role_final') == 'context_only':
        score -= 30
    score += min(20, 4 * len(entry.get('coverage_requirement_ids') or []))
    # Scores based on page-wide text or host equality carry no link evidence.
    reasons = set(entry.get('resource_link_selection_reasons') or [])
    if reasons & {'link_task_match', 'scoped_data_availability_link',
                  'scoped_task_attachment', 'scoped_task_product'}:
        score += min(60, float(entry.get('resource_link_selection_score') or 0))
    counts = family_counts or {}
    score += 15 if not counts.get(source_family(entry), 0) else -min(30, 5 * counts[source_family(entry)])
    return score


def save_frontier_result(runtime, target_id, *, document, entry, request):
    relative = '.universal/frontier-results/' + hashlib.sha256(target_id.encode()).hexdigest() + '.json'
    destination = runtime.session_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps({'document':document,'entry':entry,'request':request},
                                    ensure_ascii=False), encoding='utf-8')
    temporary.replace(destination)
    return relative


def load_frontier_results(runtime):
    result = []
    root = runtime.session_dir.resolve()
    for row in runtime.frontier.snapshot()['items']:
        reference = row.get('result_ref')
        if not reference:
            continue
        target = (root / reference).resolve()
        if not target.is_relative_to(root):
            raise ValueError('frontier artifact escapes current session')
        if not target.is_file():
            from .session_runtime import ResumeMismatch
            raise ResumeMismatch('missing persisted acquisition result: ' + reference)
        result.append(json.loads(target.read_text(encoding='utf-8')))
    return result


def rerank_frontier(runtime, registry, documents, *, state=None):
    by_id = {row.get('source_id'):row for row in registry}
    counts = Counter(source_family(by_id.get(doc.get('source_id'), {}))
                     for doc in documents if doc.get('content_readable'))
    priorities = {}
    for job in runtime.frontier.snapshot()['items']:
        if job['status'] == 'pending':
            entry = by_id.get(job['source_id']) or job['payload'].get('entry') or {}
            priorities[job['target_id']] = acquisition_priority(entry, family_counts=counts, state=state)
    runtime.frontier.reprioritize(priorities)
    return {'pending_targets': len(priorities), 'readable_families':len(counts),
            'used':runtime.ledger.snapshot()['used']}
