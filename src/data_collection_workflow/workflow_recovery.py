"""Bounded incremental recovery; no task re-intake or historical evidence access."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
import json
import hashlib
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .session_runtime import BudgetExceeded, ProviderAccountLimit, fingerprint, get_runtime

@dataclass(frozen=True)
class RecoveryGap:
    kind: str
    target_id: str
    reason: str
    source_id: str | None = None
    budget_kind: str | None = None
    document_hash: str | None = None

@dataclass(frozen=True)
class RecoveryAction:
    kind: str
    target_id: str
    action_id: str
    expected_gain: str
    document_hash: str | None = None
    strategy: str = "native"
    search_direction: str | None = None
    search_scope: str | None = None
    query: dict | None = None
    budget_revision: int = 0
    source_version: str | None = None

@dataclass
class RecoveryPlan:
    actions: list[RecoveryAction] = field(default_factory=list)
    stop_reason: str | None = None
    budget_deferred_gaps: list[dict] = field(default_factory=list)
    provider_deferred_gaps: list[dict] = field(default_factory=list)

@dataclass
class RecoveryDelta:
    sources: list[dict] = field(default_factory=list)
    documents: list[dict] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)
    record_updates: list[dict] = field(default_factory=list)
    attempted_chunk_ids: list[str] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)

def _readable_document(doc):
    return bool(doc.get('content_readable')) if doc.get('content_readable') is not None else bool(doc.get('clean_text'))



def _adaptive(budget=None):
    runtime = get_runtime()
    return bool(getattr(budget, "adaptive", False) or
                runtime and getattr(runtime.ledger, "adaptive", False))


def _parse_eligible(doc):
    if doc.get("quality_status") in {"not_task_relevant", "unsupported_format", "error_page"}:
        return False
    if doc.get("parse_eligible") is not None:
        return bool(doc["parse_eligible"])
    if doc.get("request_success") is False or doc.get("raw_content_complete") is False:
        return False
    status = doc.get("http_status_code")
    if status is not None and not 200 <= int(status) < 300:
        return False
    if doc.get("acquisition_status") in {"unsupported_format", "request_error", "http_error", "blocked", "error_page", "attempts_exhausted"}:
        return False
    if not doc.get("raw_artifact_path"):
        return _readable_document(doc)
    runtime = get_runtime()
    if runtime:
        root = runtime.session_dir.resolve()
        path = (root / doc["raw_artifact_path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or not path.stat().st_size:
            return False
    return bool(doc.get("raw_artifact_path") or _readable_document(doc))


def _frontier_jobs():
    runtime = get_runtime()
    return runtime.frontier.snapshot()["items"] if runtime and _adaptive(runtime.ledger) else []


def _frontier_source(job):
    payload = job.get("payload") or {}
    return {**(payload.get("entry") or payload), "source_id": job["source_id"],
            "url": job["url"], "canonical_url": job["target_id"]}


def _search_scope(state):
    return fingerprint({"task":state.get("structured_task") or {},
                        "contract":state.get("task_evidence_contract") or {}})


def _search_stalled(state):
    directions = set()
    scope = _search_scope(state)
    history=state.get("recovery_action_history") or []
    for action in reversed(history[int(state.get("recovery_search_streak_start") or 0):]):
        if action.get("kind") != "search":
            continue
        if action.get("search_scope") not in {None, scope}:
            continue
        if action.get("status") != "completed" or action.get("new_independent_source_ids"):
            break
        direction = action.get("search_direction")
        if not direction:
            break
        directions.add(direction)
        if len(directions) >= 2:
            return True
    return False


def unresolved_acquisition_documents(state):
    """Retain unresolved versions while excluding superseded denied dispatches."""
    documents=state.get('documents') or []
    complete=[d for d in documents if _readable_document(d) and not d.get('acquisition_incomplete') and d.get('acquisition_status')!='budget_exhausted']
    result={}
    for doc in documents:
        if not doc.get('acquisition_incomplete') and doc.get('acquisition_status')!='budget_exhausted':
            continue
        digest=doc.get('content_hash')
        pending=set(doc.get('unprocessed_pages') or [])
        if doc.get('budget_exhausted_kind')=='ocr' and pending and any(
                other is not doc and other.get('source_id')==doc.get('source_id')
                and other.get('content_hash')==digest and other.get('budget_exhausted_kind')=='ocr'
                and not other.get('parse_error') and set(other.get('unprocessed_pages') or []) < pending
                for other in documents):
            # Keep older text/locators, but resume only the most advanced page set.
            continue
        # A later fetch can settle the network deferral while leaving its own
        # OCR/browser work outstanding. Keep only the latter unresolved step.
        if doc.get('budget_exhausted_kind') in {'fetch','fetch_ordinary','http_requests','source_targets'} and not doc.get('is_shell'):
            acquired=[other for other in documents if other.get('source_id')==doc.get('source_id')
                      and other is not doc and other.get('content_hash') and other.get('request_success')
                      and other.get('budget_exhausted_kind') not in {'fetch','fetch_ordinary','http_requests','source_targets'}
                      and 200<=int(other.get('http_status_code') or 0)<300
                      and (not doc.get('url') or other.get('url')==doc.get('url'))]
            if acquired and not _readable_document(doc):
                continue
        if any(other.get('source_id')==doc.get('source_id') and
               (not digest or digest in {other.get('content_hash'),other.get('response_content_hash')}) for other in complete):
            continue
        key=fingerprint([doc.get('source_id'),digest,doc.get('budget_exhausted_kind'),sorted(set(doc.get('unprocessed_pages') or []))])
        result.setdefault(key,doc)
    return [result[key] for key in sorted(result)]


def _recovery_fetch_skip_reason(source, state):
    """Use the same permissions as the fetch consumer before reserving a slot."""
    from .nodes import content_processing as content
    from .models import ContentFetchPolicy
    from .config import load_content_fetch_policy, load_source_role_policy
    policy = ContentFetchPolicy(**load_content_fetch_policy())
    config = content._fetch_config_from_env(policy)
    mode = str((state.get('structured_task') or {}).get('collection_mode') or
               (state.get('collection_spec') or {}).get('collection_mode') or state.get('collection_mode') or 'standard')
    return content._classify_skip_reason(source,policy,content._parse_source_id_allowlist(),mode,
                                         load_source_role_policy(),config)


def _saved_pending_discoveries(state, context, *, action=None):
    """Resume only current-session discovery or provider-deferred identity work."""
    if context is None or not state.get('structured_task'):
        return []
    discovery=importlib.import_module('.nodes.source_discovery',__package__)
    rows=[]
    for source in state.get('source_registry') or []:
        relation=source.get('recovery_discovery') or {}
        provider_deferred=source.get('source_identity_status')=='provider_deferred'
        if (source.get('source_identity_status') not in {'pending_assessment','provider_deferred'}
                or not provider_deferred and (not relation.get('action_id')
                    or relation.get('session_id')!=context.session_dir.name
                    or relation.get('task_fingerprint')!=fingerprint(state.get('structured_task') or {}))
                or source.get('blocked_from_fetch')
                or source.get('source_excluded_by_human_review')
                or source.get('source_role_final')=='excluded'
                or source.get('final_screening_decision') in {'exclude','reserved_for_validation'}
                # Discovery advice cannot overwrite an existing body decision;
                # merge_recovery_delta preserves that same evidence boundary.
                or source.get('task_fit_evidence_origin')=='fetched_content' and not provider_deferred):
            continue
        if action and action.kind=='search' and (
                relation.get('action_id')!=action.action_id
                or relation.get('query_fingerprint')!=fingerprint(discovery._search_request_identity(action.query or {}))):
            continue
        if action and action.kind=='assess_source' and str(source.get('source_id'))!=action.target_id:
            continue
        rows.append(source)
    return rows


def _pending_source_version(source, state):
    fields=('source_id','canonical_url','url','domain','title','snippet','publisher','published_date',
            'source_type','search_provider','search_result_source_raw','search_provider_result_source',
            'result_source','search_rank','query_used','discovery_method','role_hint','source_purpose',
            'content_hash','text_hash','recovery_discovery')
    return fingerprint({'source':{key:source.get(key) for key in fields},
                        'task':state.get('structured_task') or {},'collection_spec':state.get('collection_spec') or {}})


def _pending_assessment_blockers(sources, state, budget):
    snapshot=budget.snapshot() if hasattr(budget,'snapshot') else budget
    remaining=snapshot.get('remaining') or {}
    blockers=[]
    if remaining.get('extraction',0)<=0:
        blockers.append('extraction')
    readable={d.get('source_id') for d in state.get('documents') or [] if _readable_document(d)}
    for source in sources:
        if source.get('source_id') in readable or _fetch_affordable(source,state,remaining,budget):
            continue
        candidates=('source_targets','http_requests') if _adaptive(budget) else ('fetch','fetch_ordinary')
        blockers.extend(kind for kind in candidates if remaining.get(kind,0)<=0)
    return sorted(set(blockers))


def assess_collection_gaps(state):
    gaps=[]
    documents=state.get('documents') or []
    chunks=state.get('evidence_chunks') or []
    unresolved=unresolved_acquisition_documents(state)
    incomplete_sources={d.get('source_id') for d in unresolved}
    doc_sources={d.get('source_id') for d in documents if _readable_document(d)}
    pending={str(source.get('source_id')) for source in _saved_pending_discoveries(state,get_runtime())}
    for source in state.get('source_registry') or []:
        if str(source.get('source_id')) in pending:
            gaps.append(RecoveryGap('source_assessment_pending',str(source.get('source_id')),'saved discovery identity assessment is incomplete'))
            continue
        if source.get('source_id') not in doc_sources|incomplete_sources and not _recovery_fetch_skip_reason(source,state):
            gaps.append(RecoveryGap('fetch_failed',str(source.get('source_id')),'no readable source content'))
    for doc in unresolved:
        pages=sorted(set(doc.get('unprocessed_pages') or []))
        reason=('budget deferred: '+str(doc['budget_exhausted_kind'])) if doc.get('budget_exhausted_kind') else ('acquisition incomplete: '+str(doc.get('parse_error') or doc.get('fetch_error') or doc.get('acquisition_status') or 'unknown'))
        if pages: reason+='; unprocessed pages: '+', '.join(map(str,pages))
        gaps.append(RecoveryGap('acquisition_incomplete',str(doc.get('document_id') or doc.get('source_id')),reason,
                                str(doc.get('source_id')),doc.get('budget_exhausted_kind'),doc.get('content_hash')))
    for doc in documents:
        # An app shell is not parsed evidence. Partial documents are tracked above;
        # their already usable spans remain eligible for extraction below.
        if (not _parse_eligible(doc) or doc.get('acquisition_incomplete') or doc.get('is_shell')
                or doc.get('acquisition_status') in {'budget_exhausted','http_error','blocked','error_page'}):
            continue
        if doc.get('raw_artifact_path') and (doc.get('acquisition_status')=='parse_error' or doc.get('parse_status')=='parsed_partial'):
            # Usable spans do not imply the remaining saved content was parsed.
            settled=any(other is not doc and other.get('source_id')==doc.get('source_id')
                        and doc.get('content_hash') and other.get('content_hash')==doc.get('content_hash')
                        and _readable_document(other) and not other.get('acquisition_incomplete')
                        and other.get('acquisition_status') not in {'parse_error','budget_exhausted'}
                        and other.get('parse_status')!='parsed_partial' for other in documents)
            if not settled:
                gaps.append(RecoveryGap('parse_failed',str(doc.get('document_id') or doc.get('source_id')),
                                        'saved content parsing failed or is incomplete',document_hash=doc.get('content_hash')))
            continue
        if (doc.get('raw_artifact_path') or _readable_document(doc)) and not any(c.get('source_id')==doc.get('source_id') and (not doc.get('content_hash') or (c.get('document_hash') or c.get('content_hash'))==doc.get('content_hash')) for c in chunks):
            gaps.append(RecoveryGap('parse_missing',str(doc.get('document_id') or doc.get('source_id')),'content has no spans',document_hash=doc.get('content_hash')))
    attempted=set(state.get('extraction_attempted_chunk_ids') or [])
    runtime=get_runtime()
    failed_spans=set(runtime.ledger.failed_chunk_ids(retryable_only=True)) if runtime else set()
    from .nodes.extraction import extraction_skip_reason
    for chunk in chunks:
        if extraction_skip_reason(chunk):
            continue
        # Background and explicitly excluded spans are not recovery work.
        if (chunk.get('contains_target_data') is False
                or chunk.get('extraction_eligible_for_task_disease') is False
                or chunk.get('disease_relevance_status') not in {None, '', 'target_disease_match'}):
            continue
        key=chunk.get('chunk_id')
        if key in failed_spans:
            gaps.append(RecoveryGap('extraction_failed',str(key),'latest extraction failed and one explicit retry remains'))
        elif key not in attempted:
            gaps.append(RecoveryGap('unprocessed_span',str(key),'span has no extraction attempt'))
    for requirement in (state.get('source_coverage_audit') or {}).get('requirements') or []:
        if requirement.get('coverage_status') != 'covered' or not requirement.get('qualified_record_ids'):
            gaps.append(RecoveryGap('source_missing',str(requirement.get('requirement_id')),'qualified coverage missing'))
    if not state.get('source_registry'):
        gaps.append(RecoveryGap('source_missing','task','no discovered source'))
    for row in state.get('candidate_records') or []:
        reasons = (row.get('evidence_qualification') or {}).get('reasons') or []
        if any(reason in {'missing_geography','missing_observation_period'} or 'unbound_field_value' in reason for reason in reasons):
            gaps.append(RecoveryGap('candidate_fields_missing',str(row.get('record_id') or ''),'; '.join(reasons),source_id=row.get('source_id')))
    for job in _frontier_jobs():
        if job["status"] == "pending":
            gaps.append(RecoveryGap("frontier_pending", job["source_id"], "persisted acquisition target remains pending"))
    for conflict in state.get('conflicts') or []:
        gaps.append(RecoveryGap('conflict',str(conflict.get('conflict_id')),'requires independent supporting evidence'))
    temporal = _temporal_recovery_query(state)
    if temporal:
        gaps.append(RecoveryGap('report_timeline_gap', temporal['temporal_probe']['probe_id'],
                                'bounded date-gap retrieval opportunity; not missing observation coverage'))
    return gaps


def _recovery_acquisition_config(runtime, *, reparse=False):
    from .document_acquisition import _paths
    config=((runtime.config.get('universal') or {}).get('acquisition') or {}) if runtime else {}
    if not reparse:
        from .nodes.content_processing import _fetch_config_from_env
        from .models import ContentFetchPolicy
        from .config import load_content_fetch_policy
        config={**(runtime.config if runtime else {}),**config,**_fetch_config_from_env(ContentFetchPolicy(**load_content_fetch_policy()))}
    return _paths(config)


def _cached_acquisition(runtime, budget, kind, payload):
    from .document_acquisition import PARSER_VERSION
    return bool(runtime and budget is runtime.ledger and runtime.has_cached(kind,{'parser_version':PARSER_VERSION,**payload}))


def _fetch_affordable(source,state,remaining,budget,*,needs_browser=False):
    from .models import ContentFetchRequest
    from .nodes.content_processing import _universal_priority_context
    runtime=get_runtime()
    url=source.get('canonical_url') or source.get('url') or ''
    source_id=str(source.get('source_id') or '')
    request=ContentFetchRequest(source_id=source_id,url=url,canonical_url=url,final_screening_decision=source.get('final_screening_decision') or '',fetch_purpose='recovery')
    priority=bool(_universal_priority_context(request,source))
    kind='fetch_priority' if priority else 'fetch_ordinary'
    config=_recovery_acquisition_config(runtime)
    cached=_cached_acquisition(runtime,budget,kind,{'url':url,'max_bytes':config.get('max_bytes',20_000_000)})
    if _adaptive(budget):
        browser_cached = needs_browser and _cached_acquisition(runtime,budget,'browser_navigation',{'url':url,'render_wait_ms':config.get('render_wait_ms',300)})
        if cached and not needs_browser or browser_cached:
            return True
        if remaining.get('http_requests',0)<=0:
            return False
        charged = bool(url and hasattr(budget,'has_source_target') and budget.has_source_target(url))
        if not charged and remaining.get('source_targets',0)<=0:
            return False
        return not needs_browser or remaining.get('browser',0)>0
    if not cached and (remaining.get('fetch',0)<=0 or not priority and remaining.get('fetch_ordinary',0)<=0):
        return False
    if needs_browser:
        cached_browser=_cached_acquisition(runtime,budget,'browser_fetch',{'url':url,'render_wait_ms':config.get('render_wait_ms',300)})
        if not cached_browser and (remaining.get('browser',0)<=0 or remaining.get('fetch',0)<(1 if cached else 2)):
            return False
    return True


def _repair_action_input(row, state):
    """Only source facts and evidence can reopen a completed local repair."""
    from .evidence_qualification import _NUMERIC, _DATES, _GEO, _SUBSTANTIVE, _IDENTITY_FIELDS, _provenance

    # The repair consumer reads raw_records; qualification/consistency annotations
    # on its candidate projection are derived output, not a new action input.
    row = next((raw for raw in state.get('raw_records') or []
                if raw.get('record_id') == row.get('record_id')), row)
    fields = ('record_id', 'source_id', 'disease', 'supporting_chunk_id',
              'chunk_id', 'evidence_chunk_id', 'case_span_id', 'evidence_span_id',
              'case_span_quote', 'evidence_quote') + _NUMERIC + _DATES + _GEO + _SUBSTANTIVE + tuple(_IDENTITY_FIELDS)
    chunk = next((c for c in state.get('evidence_chunks') or []
                  if c.get('chunk_id') == row.get('supporting_chunk_id')
                  and c.get('source_id') == row.get('source_id')), {})
    document = next((d for d in state.get('documents') or []
                     if d.get('source_id') == row.get('source_id')
                     and (not chunk.get('document_id') or d.get('document_id') == chunk.get('document_id'))
                     and (not chunk.get('document_hash') or d.get('content_hash') == chunk.get('document_hash'))), {})
    return {
        'record': {key: row[key] for key in fields if key in row},
        'provenance': _provenance(row),
        'span': {key: chunk.get(key) for key in ('chunk_id', 'document_id', 'document_hash',
            'char_start', 'char_end', 'table_id', 'row_id', 'page_number', 'text', 'bound_context_spans')},
        'document': {key: document.get(key) for key in ('content_hash', 'text_hash', 'parser_version')},
        'task': state.get('structured_task') or {},
        'contract': state.get('task_evidence_contract') or {},
    }


def _new_source_search_blockers(remaining):
    """A new source search needs a payable acquisition and extraction path."""
    return [kind for kind in ('search', 'search_results', 'source_targets', 'http_requests', 'extraction')
            if remaining.get(kind, 0) <= 0]


def _provider_stop(snapshot):
    if not snapshot.get('providers'):
        return None
    from .llm_clients import get_llm_settings
    provider=str(get_llm_settings().get('provider') or '').lower()
    status=(snapshot.get('providers') or {}).get(provider) or {}
    return status if status.get('status')=='halted' else None


def plan_recovery(gaps,*,state,budget):
    snapshot=budget.snapshot() if hasattr(budget,'snapshot') else budget
    provider_stop=_provider_stop(snapshot)
    provider_deferred_gaps=[]
    remaining=snapshot.get('remaining') or {}
    revision=int(snapshot.get('budget_revision') or 0)
    kinds={'parse_missing':('reparse',0),'parse_failed':('reparse',0),
           'candidate_fields_missing':('repair_fields',1),'unprocessed_span':('extract',2),
           'extraction_failed':('extract',2),'fetch_failed':('fetch',3),
           'frontier_pending':('fetch',3),'source_assessment_pending':('assess_source',3),'source_missing':('search',4),
           'conflict':('search',4),'report_timeline_gap':('search',4),'acquisition_incomplete':('fetch',3)}
    history=state.get('recovery_action_history') or []
    actions=[]; budget_blocked=False; budget_deferred_gaps=[]
    jobs=_frontier_jobs()
    for gap in sorted(gaps,key=lambda g:(0 if g.kind=='acquisition_incomplete' and g.budget_kind=='ocr' else kinds.get(g.kind,('',9))[1],
                                      0 if g.kind in {'extraction_failed','report_timeline_gap'} else 1,g.target_id)):
        if gap.kind not in kinds:
            continue
        kind=kinds[gap.kind][0]; target=gap.target_id; strategy='native'; query=None
        if provider_stop and kind in {'extract','assess_source','search'}:
            provider_deferred_gaps.append({**asdict(gap),'status':'provider_deferred','reason':'provider_account_limit','provider_stop':provider_stop})
            continue
        source_version=None
        doc=next((d for d in state.get('documents') or [] if str(d.get('document_id') or d.get('source_id'))==target and (not gap.document_hash or d.get('content_hash')==gap.document_hash)),{})
        if gap.kind=='acquisition_incomplete':
            if gap.budget_kind=='action_attempts':
                continue
            if gap.budget_kind=='ocr':
                kind='reparse'
            else:
                target=gap.source_id or target
        if kind=='reparse':
            if not _parse_eligible(doc):
                continue
            if gap.budget_kind=='ocr':
                runtime=get_runtime(); config=_recovery_acquisition_config(runtime,reparse=True)
                pages=doc.get('unprocessed_pages') or []
                missing=sum(not _cached_acquisition(runtime,budget,'ocr',{
                    'document_hash':doc.get('content_hash'),'page':page,
                    'languages':config.get('ocr_languages','eng'),'render_scale':config.get('ocr_render_scale',3)}) for page in pages) if pages else 1
                required_pages=min(1,missing) if _adaptive(budget) else missing
                if remaining.get('ocr',0)<required_pages:
                    budget_blocked=True; continue
        elif kind=='repair_fields':
            row=next((r for r in [*(state.get('candidate_records') or []),*(state.get('raw_records') or [])]
                      if str(r.get('record_id') or '')==target),None)
            if not row or not any(c.get('chunk_id')==row.get('supporting_chunk_id') for c in state.get('evidence_chunks') or []):
                continue
        elif kind=='assess_source':
            source=next((row for row in _saved_pending_discoveries(state,get_runtime()) if str(row.get('source_id'))==target),None)
            if source is None:
                continue
            blockers=_pending_assessment_blockers([source],state,budget)
            if blockers:
                budget_blocked=True
                budget_deferred_gaps.append({**asdict(gap),'status':'budget_deferred','blocked_budget_kinds':blockers})
                continue
            strategy='source_identity_v1'; source_version=_pending_source_version(source,state)
        elif kind=='search':
            if gap.kind=='report_timeline_gap' and any((a.query or {}).get('temporal_probe') for a in actions):
                continue
            blockers=_new_source_search_blockers(remaining) if _adaptive(budget) else []
            if blockers:
                budget_blocked=True
                budget_deferred_gaps.append({**asdict(gap), 'status':'budget_deferred',
                    'blocked_budget_kinds':blockers,
                    'reason':gap.reason+'; new source discovery needs downstream budget: '+', '.join(blockers)})
                continue
            if remaining.get('search',0)<=0:
                budget_blocked=True; continue
            query=_recovery_query(state,target)
            if not query:
                continue
        elif kind=='fetch':
            source=next((row for row in state.get('source_registry') or [] if str(row.get('source_id'))==target),None)
            job=next((job for job in jobs if str(job['source_id'])==target),None)
            source=source or (_frontier_source(job) if job else {})
            if not source or _recovery_fetch_skip_reason(source,state):
                continue
            failures=[d for d in state.get('documents') or [] if str(d.get('source_id'))==target and
                      d.get('acquisition_status') in {'request_error','attempts_exhausted','browser_error','blocked','shell'}]
            strategy='browser' if gap.budget_kind=='browser' or doc.get('is_shell') or doc.get('fetch_provider')=='chromium' or 'browser_resources_incomplete' in (doc.get('quality_issues') or []) or failures else 'native'
            if not _fetch_affordable(source,state,remaining,budget,needs_browser=strategy=='browser'):
                # A still-available native branch can serve strict legacy retries.
                if strategy=='browser' and gap.budget_kind!='browser' and not doc.get('is_shell') and not any(d.get('acquisition_status')=='attempts_exhausted' for d in failures) and _fetch_affordable(source,state,remaining,budget):
                    strategy='native'
                else:
                    budget_blocked=True; continue
            if job and job['status']=='failed' and not get_runtime().frontier.can_retry(job['target_id'],strategy=strategy):
                continue
        required={'extract':('extraction',), 'search':('search','search_results') if _adaptive(budget) else ('search','fetch','fetch_ordinary','extraction')}.get(kind,())
        if any(remaining.get(key,0)<=0 for key in required):
            budget_blocked=True; continue
        identity={'kind':kind,'target':target}
        if kind in {'reparse','fetch'}:
            identity['strategy']=strategy
        if kind=='reparse':
            from .document_acquisition import PARSER_VERSION
            config=_recovery_acquisition_config(get_runtime(),reparse=True)
            identity.update(parser_version=PARSER_VERSION,
                ocr_languages=config.get('ocr_languages','eng'),
                ocr_render_scale=config.get('ocr_render_scale',3),
                force_ocr=bool(config.get('force_ocr')))
            if gap.document_hash: identity['document_hash']=gap.document_hash
        if kind=='assess_source':
            identity.update(strategy=strategy,source_version=source_version)
        if kind=='repair_fields':
            identity['strategy']=strategy
            identity['input']=_repair_action_input(row,state)
        if query:
            identity['query']=query['query']; identity['search_direction']=query['search_direction']
        action_id=fingerprint(identity)
        prior=[h for h in history if h.get('action_id')==action_id]
        # A denial has no attempt. Only a new budget revision reopens that denial.
        if any(h.get('status') in {'budget_exhausted','budget_deferred'} and int(h.get('budget_revision') or 0)>=revision for h in prior):
            budget_blocked=True; continue
        actual=[h for h in prior if h.get('status') in {'completed','completed_empty','failed','skipped'}]
        if len(actual)>=2 or any(h.get('status') in {'completed','completed_empty','skipped'} for h in actual):
            continue
        if any(a.action_id==action_id for a in actions):
            continue
        actions.append(RecoveryAction(kind,target,action_id,'supported fields or independent evidence',
            document_hash=gap.document_hash if kind=='reparse' else None,strategy=strategy,
            query=query,search_direction=query.get('search_direction') if query else None,
            search_scope=(fingerprint([_search_scope(state),query['temporal_probe']['probe_id']])
                          if query and query.get('temporal_probe') else _search_scope(state) if query else None),
            budget_revision=revision,source_version=source_version))
    if len(actions)>8:
        representatives={}
        for action in actions:
            representatives.setdefault(action.kind,action.action_id)
        selected=set(representatives.values())
        for action in actions:
            if len(selected)>=8: break
            selected.add(action.action_id)
        actions=[action for action in actions if action.action_id in selected]
    return RecoveryPlan(actions,None if actions else 'coverage_satisfied' if not gaps else 'provider_account_limit' if provider_deferred_gaps else 'budget_exhausted' if budget_blocked else 'no_executable_action',budget_deferred_gaps,provider_deferred_gaps)


def _url(url):
    split=urlsplit(str(url or ''))
    return urlunsplit((split.scheme.lower(),split.netloc.lower(),split.path.rstrip('/'),split.query,''))

def merge_recovery_delta(state,delta):
    if isinstance(delta,dict): delta=RecoveryDelta(**delta)
    sources=[dict(source) for source in state.get('source_registry') or []]
    url_index={_url(s.get('canonical_url') or s.get('url')):s for s in sources}
    aliases={}
    for source in delta.sources:
        key=_url(source.get('canonical_url') or source.get('url'))
        if key and key in url_index:
            retained=url_index[key]
            aliases[source.get('source_id')]=retained.get('source_id')
            # Only an unfinished discovery assessment may be replaced by its
            # later screened result. Existing source/field evidence wins.
            if (retained.get('source_id')==source.get('source_id')
                    and retained.get('source_identity_status')=='pending_assessment'
                    and source.get('source_identity_status') not in {None,'pending_assessment'}
                    and retained.get('task_fit_evidence_origin')!='fetched_content'
                    and not retained.get('blocked_from_fetch')
                    and retained.get('final_screening_decision') not in {'exclude','reserved_for_validation'}):
                retained.update(source)
        else:
            sources.append(source); url_index[key]=source
    document_aliases={}
    def remap(row):
        out=dict(row)
        if out.get('source_id') in aliases: out['source_id']=aliases[out['source_id']]
        if out.get('document_id') in document_aliases: out['document_id']=document_aliases[out['document_id']]
        return out
    def merge(existing,extra,key):
        index={key(r):r for r in existing}
        for row in extra:
            item=remap(row); index.setdefault(key(item),item)
        return [index[k] for k in sorted(index)]
    def document_key(d):
        return fingerprint([d.get('source_id'),d.get('content_hash') or d.get('document_id'),d.get('text_hash'),d.get('parser_version')])
    document_index={document_key(d):dict(d) for d in state.get('documents') or []}
    for row in delta.documents:
        item=remap(row); key=document_key(item)
        retained=document_index.setdefault(key,item)
        # A blank OCR page can complete without changing the evidence text/hash.
        # Advance its acquisition metadata while preserving existing references.
        old_pages=set(retained.get('unprocessed_pages') or [])
        new_pages=set(item.get('unprocessed_pages') or [])
        completed=(not item.get('acquisition_incomplete') and item.get('acquisition_status') not in {'budget_exhausted','parse_error'}
                   and item.get('parse_status')!='parsed_partial')
        incomplete=(retained.get('acquisition_incomplete') or retained.get('acquisition_status')=='parse_error'
                    or retained.get('parse_status')=='parsed_partial')
        if incomplete and _readable_document(item) and (completed or new_pages<old_pages):
            retained_id=retained.get('document_id')
            retained.update(item)
            if retained_id: retained['document_id']=retained_id
        incoming_id=row.get('document_id')
        if incoming_id:
            if not retained.get('document_id'): retained['document_id']=incoming_id
            document_aliases[incoming_id]=retained['document_id']
    documents=[document_index[k] for k in sorted(document_index)]
    chunk_aliases={}
    chunk_index={}
    def chunk_key(c):
        return fingerprint([c.get('source_id'),c.get('document_hash') or c.get('content_hash'),c.get('char_start'),c.get('char_end'),c.get('table_id'),c.get('row_id'),c.get('text')])
    used_ids={}
    for row in [*(state.get('evidence_chunks') or []),*delta.chunks]:
        item=remap(row); key=chunk_key(item); old_id=item.get('chunk_id')
        if key in chunk_index:
            chunk_aliases[old_id]=chunk_index[key].get('chunk_id')
            continue
        if old_id in used_ids and used_ids[old_id]!=key:
            item['chunk_id']=str(old_id)+'_'+key[:20]
            chunk_aliases[old_id]=item['chunk_id']
        used_ids[item.get('chunk_id')]=key
        chunk_index[key]=item
    chunks=[chunk_index[k] for k in sorted(chunk_index)]
    def references(value):
        if isinstance(value,list): return [references(v) for v in value]
        if not isinstance(value,dict): return value
        out={}
        for k,v in value.items():
            if k=='source_id': out[k]=aliases.get(v,v)
            elif k=='document_id': out[k]=document_aliases.get(v,v)
            elif k in {'chunk_id','supporting_chunk_id','evidence_chunk_id','evidence_span_id'}: out[k]=chunk_aliases.get(v,v)
            elif k=='field_provenance_json' and isinstance(v,str):
                try: out[k]=json.dumps(references(json.loads(v)),sort_keys=True)
                except (ValueError,TypeError): out[k]=v
            else: out[k]=references(v)
        return out
    ignored={'record_id','extraction_confidence','normalization_status','evidence_qualification','product_kind'}
    # Existing references keep their original version; aliases only describe delta.
    existing=[dict(row) for row in state.get('raw_records') or []]
    from .evidence_qualification import evidence_qualification_enabled
    from .record_identity import record_identity_payload, stable_record_id
    universal = evidence_qualification_enabled()
    chunk_by_id={chunk.get('chunk_id'):chunk for chunk in chunks}
    def record_chunk(row):
        return chunk_by_id.get(row.get('supporting_chunk_id')) or {}
    def record_key(row):
        if universal:
            return fingerprint(record_identity_payload(row,record_chunk(row)))
        canonical={k:v for k,v in row.items() if k not in ignored}
        value=canonical.get('field_provenance_json')
        if isinstance(value,str):
            try: canonical['field_provenance_json']=json.loads(value)
            except ValueError: pass
        return fingerprint(canonical)
    incoming=[references(r) for r in delta.records]
    if universal:
        keys_by_id={row.get('record_id'):record_key(row) for row in existing}
        for row in incoming:
            key=record_key(row)
            if row.get('record_id') in keys_by_id and keys_by_id[row['record_id']]!=key:
                row['record_id']=stable_record_id(row,record_chunk(row))
            keys_by_id[row.get('record_id')]=key
    records=merge(existing,incoming,record_key)
    for update in delta.record_updates:
        current=next((row for row in records if row.get("record_id")==update.get("record_id")
                      and (not universal or fingerprint(row)==update.get("before_fingerprint"))),None)
        if current is None or fingerprint(current)!=update.get("before_fingerprint"):
            continue
        current.update(update.get("fields") or {})
        provenance=current.get("field_provenance_json") or {}
        if isinstance(provenance,str):
            try: provenance=json.loads(provenance)
            except ValueError: provenance={}
        provenance={**provenance,**(update.get("field_provenance") or {})}
        current["field_provenance_json"]=json.dumps(provenance,ensure_ascii=False,sort_keys=True)
        current.pop("evidence_qualification",None)
    return {'source_registry':sources,'documents':documents,'evidence_chunks':chunks,'raw_records':records,'extraction_attempted_chunk_ids':sorted(set(state.get('extraction_attempted_chunk_ids') or [])|{chunk_aliases.get(c,c) for c in delta.attempted_chunk_ids})}


def _gain(state):
    from .evidence_qualification import build_evidence_index
    from .evidence_products import _documents, _resolved, IDENTITIES, SCOPE
    fields=set()
    index=build_evidence_index(state)
    documents=None
    from .evidence_products import _lineage
    sources={str(item.get("source_id")):item for item in state.get("source_registry") or []}
    lineages={item["source_id"]:item for item in _lineage(list(sources.values()))["sources"]}
    for row in [*(state.get('qualified_records') or []),*(state.get('candidate_records') or [])]:
        q=row.get('evidence_qualification') or {}
        if q.get('status') not in {'qualified','candidate'} or any('contract_mismatch' in reason for reason in q.get('reasons') or []):
            continue
        evidence=q.get('field_evidence') or []
        supported={}
        for e in evidence:
            if e.get('value') != row.get(e.get('field')) or e.get('supported') is not True:
                continue
            if documents is None:
                # Preparing the full document set is shared only within this call;
                # every field still verifies the current original hash and locator.
                documents=_documents(index)
            if _resolved(e,index,documents=documents):
                supported[e.get('field')]=e
        scope={k:supported[k]['value'] for k in SCOPE if k in supported}
        identity=next((supported[k] for k in IDENTITIES if k in supported),None)
        # Patient labels are local to their document until explicit linkage.
        patient=[identity['document_hash'],identity['value']] if identity else None
        for name,e in supported.items():
            fact=[patient,scope,name,e['value']]
            fields.add(fingerprint(fact))
            source=sources.get(str(row.get("source_id")), {})
            origin=lineages.get(str(row.get("source_id")), {})
            if source.get("source_identity_unverified") is False and origin.get("original_publisher"):
                fields.add(fingerprint(["independent_support",origin["independence_group"],fact]))
    return sorted(fields)


def recovery_control(state):
    from .evidence_qualification import qualified_coverage, qualify_records, build_evidence_index
    # Finalization has not run yet. Assess the CURRENT normalized evidence at
    # every loop boundary instead of reading absent/stale final views.
    groups={}
    if 'normalized_records' in state:
        contract={**(state.get('structured_task') or {}),**(state.get('task_evidence_contract') or {})}
        groups=qualify_records(state.get('normalized_records') or [],contract=contract,evidence_index=build_evidence_index(state))
        state={**state,**groups}
    requirements=state.get('source_coverage_requirements')
    if requirements is None:
        requirements=(state.get('source_coverage_audit') or {}).get('requirements') or []
    coverage=qualified_coverage(requirements,state.get('qualified_records') or [])
    from .report_timeline import build_report_timeline_inventory
    coverage['reporting_timeline']=build_report_timeline_inventory(state)
    state={**state,'source_coverage_audit':coverage}
    round_number=int(state.get('recovery_round') or 0)
    runtime=get_runtime()
    limits=(runtime.config.get('universal') or {}) if runtime else {}
    gain=_gain(state)
    streak_start=int(state.get('recovery_search_streak_start') or 0)
    if set(gain)-set(state.get('recovery_previous_gain') or []):
        streak_start=len(state.get('recovery_action_history') or [])
        state={**state,'recovery_search_streak_start':streak_start}
    gaps=assess_collection_gaps(state)
    adaptive=bool(runtime and _adaptive(runtime.ledger))
    if adaptive and not any(gap.kind in {'source_missing','conflict'} for gap in gaps):
        gaps.append(RecoveryGap('source_missing','corroboration','continue independent source discovery'))
    plan=plan_recovery(gaps,state=state,budget=runtime.ledger if runtime else {'remaining':{}})
    work=[action for action in plan.actions if action.kind!='search']
    novel_temporal=any((action.query or {}).get('temporal_probe') for action in plan.actions)
    if adaptive:
        if not work and not novel_temporal and plan.stop_reason not in {'budget_exhausted','provider_account_limit'} and _search_stalled(state) and not (runtime and runtime.budget_continuation and any(job['status']=='pending' for job in _frontier_jobs())):
            plan=RecoveryPlan(stop_reason='no_progress')
    elif plan.stop_reason!='provider_account_limit':
        maximum=min(2,int(limits.get('recovery_rounds',2)))
        if round_number>=maximum:
            plan=RecoveryPlan(stop_reason='round_limit')
        elif round_number and not work and not novel_temporal and not (set(gain)-set(state.get('recovery_previous_gain') or [])):
            plan=RecoveryPlan(stop_reason='no_progress')
    return {**groups,'source_coverage_audit':coverage,'recovery_gaps':[asdict(g) for g in gaps], 'recovery_plan':asdict(plan), 'recovery_previous_gain':gain,'recovery_search_streak_start':streak_start,'recovery_stop_reason':plan.stop_reason,'run_budget_ledger':runtime.ledger.snapshot() if runtime else {}}


def _temporal_recovery_query(state):
    from .nodes.source_discovery import _source_search_settings_from_env
    from .query_policy import assess_query_task_fit
    from .temporal_search_queries import temporal_search_queries
    settings = _source_search_settings_from_env()
    if not settings.search_enabled:
        return None
    runtime = get_runtime()
    return next((query for query in temporal_search_queries(state, channels=settings.provider_channel_allowlist,
                 operation_history=runtime.ledger.operation_audit() if runtime else ())
                 if assess_query_task_fit(query,state)['accepted']), None)


def _recovery_query(state,target_id):
    if str(target_id).startswith('temporal_'):
        query = _temporal_recovery_query(state)
        return query if query and query['temporal_probe']['probe_id']==target_id else None
    from .query_policy import generic_retry_specs, _retry_domain
    task=dict(state.get('structured_task') or {})
    requirement=next((r for r in (state.get('source_coverage_audit') or {}).get('requirements') or [] if str(r.get('requirement_id'))==target_id),{})
    task['location']=requirement.get('location') or requirement.get('geography') or task.get('location')
    start=requirement.get('period_start') or requirement.get('reporting_period_start') or task.get('start_date')
    end=requirement.get('period_end') or requirement.get('reporting_period_end') or task.get('end_date')
    period=' '.join(str(v) for v in (start,end) if v)
    domains=[domain for value in requirement.get('official_domains') or [] if (domain:=_retry_domain(value))]
    candidates=[row for row in state.get('source_registry') or []
                if _retry_domain(row.get('domain') or row.get('canonical_url') or row.get('url')) in domains]
    candidates.extend({'domain':domain} for domain in domains)
    scoped={**state,'structured_task':task}
    specs=generic_retry_specs(scoped,candidates,period)
    specs+=generic_retry_specs(scoped,candidates,period,literature=True)
    if domains:
        specs=[spec for spec in specs if spec.get('official_domain_hint') in domains]
    if not domains and not any('study' in spec['query'] for spec in specs):
        disease=str(task.get('disease') or '').strip()
        location=str(task.get('location') or '').strip()
        if disease:
            specs.append({'query':f'"{disease}" "{location}" research study outbreak cases {period}',
                          'source_type':'academic_or_peer_reviewed_source','provider_channel':'web_search',
                          'role_hint':'collection_support','official_domain_hint':None})
    from .temporal_search_queries import attempted_query_keys, query_key
    used_keys=attempted_query_keys(state,retry_failed=True)
    used=set(); failures={}
    for row in state.get('recovery_action_history') or []:
        if row.get('kind')!='search':
            continue
        query=row.get('query') if isinstance(row.get('query'),str) else (row.get('query') or {}).get('query')
        if row.get('status') in {'completed','skipped'}:
            used.add(query)
        elif row.get('status')=='failed':
            failures[query]=failures.get(query,0)+1
    used.update(query for query,attempts in failures.items() if attempts>=2)
    for spec in specs:
        if spec['query'] in used or query_key(spec) in used_keys:
            continue
        text=spec['query']
        direction=('research' if 'study' in text else 'dataset' if 'data table' in text else
                   'archive' if 'archive' in text else 'report_download' if 'download' in text else 'official_surveillance')
        return {**spec,'search_direction':direction}
    return None


def _reparse_saved_document(doc,context,*,strategy="native"):
    """Reparse verified raw bytes from this session, never discover external files."""
    from .document_acquisition import parse_response
    from .session_runtime import ResumeMismatch
    raw=doc.get('raw_artifact_path')
    if not raw:
        if not _parse_eligible(doc): raise ValueError('document is not eligible for parsing')
        return dict(doc)
    session=context.session_dir.resolve()
    target=(session/Path(raw)).resolve()
    if not target.is_relative_to(session):
        raise ResumeMismatch('raw artifact must remain within the current session')
    body=target.read_bytes()
    if not body or not _parse_eligible(doc):
        raise ValueError('saved response is empty or not eligible for parsing')
    digest=hashlib.sha256(body).hexdigest()
    if digest!=doc.get('content_hash'):
        raise ResumeMismatch('saved raw artifact content hash mismatch')
    config=dict((context.config.get('universal') or {}).get('acquisition') or {})
    if strategy=='force_ocr': config['force_ocr']=True
    parsed=parse_response(body,url=doc.get('url') or doc.get('canonical_url') or '',source_id=doc.get('source_id'),session_dir=session,content_type=doc.get('content_type') or 'text/plain',status_code=doc.get('http_status_code') or 200,final_url=doc.get('final_url'),config=config,recovery=True)
    result={**doc,**parsed}
    for key,default in [('acquisition_incomplete',False),('budget_exhausted_kind',None),('unprocessed_pages',[])]:
        result[key]=parsed.get(key,default)
    result['document_id']='doc_'+fingerprint([doc.get('source_id'),digest,parsed.get('parser_version'),parsed.get('text_hash')])[:32]
    result['retrieved_at']=doc.get('retrieved_at') or parsed['retrieved_at']
    result['fetch_error']=parsed.get('fetch_error')
    return result


def _repair_candidate_fields(action,state):
    from .nodes.extraction import _build_extraction_context, _official_outbreak_record_from_chunk, extraction_skip_reason
    from .config import load_structured_extraction_policy
    from .models import StructuredExtractionPolicy
    from .evidence_qualification import assess_record_evidence, build_evidence_index, _NUMERIC, _DATES, _GEO, _local_span, _provenance
    from .disease_identity import same_disease_name
    matches=[r for r in state.get('raw_records') or [] if str(r.get('record_id'))==action.target_id]
    if len(matches)>1:
        return None,'ambiguous_record_identity'
    row=matches[0] if matches else None
    if row is None:
        return None,'candidate_not_found'
    chunk=next((c for c in state.get('evidence_chunks') or [] if c.get('chunk_id')==row.get('supporting_chunk_id') and c.get('source_id')==row.get('source_id')),None)
    if chunk is None or extraction_skip_reason(chunk):
        return None,'no_eligible_source_span'
    index=build_evidence_index(state)
    counts=[name for name in _NUMERIC if row.get(name) is not None]
    provenance=_provenance(row)
    entry=next((provenance[name] for name in counts if isinstance(provenance.get(name),dict)),{})
    if not entry:
        quote=row.get('case_span_quote') or row.get('evidence_quote')
        entry={'quote':quote} if quote else {}
    quote,digest,locator,error=_local_span(row,entry,index)
    if error:
        return None,'unresolved_original_observation_locator'
    document=next(doc for doc in index['documents'] if doc.get('document_id')==locator.get('document_id') and doc.get('content_hash')==digest)
    chunk={**chunk,'text':document['clean_text'][locator['char_start']:locator['char_end']],
           'char_start':locator['char_start'],'char_end':locator['char_end']}
    policy=StructuredExtractionPolicy(**load_structured_extraction_policy())
    producer,_=_official_outbreak_record_from_chunk(chunk,1,policy,_build_extraction_context(state,policy))
    if producer is None:
        return None,'source_producer_returned_empty'
    proposal=producer.model_dump()
    if not same_disease_name(row.get('disease'),proposal.get('disease')):
        return None,'different_observation_disease'
    # A generic metric is another representation of the same typed count,
    # not a different observation merely because this producer uses count fields.
    identity_counts={name:row[name] for name in counts if name!='metric_value'}
    if row.get('metric_value') is not None:
        from .evidence_qualification import _canonical_metric, _COUNT_LABELS, _supports
        metric=_canonical_metric(row.get('metric_name'))
        if (metric not in _COUNT_LABELS or
                metric in identity_counts and identity_counts[metric]!=row['metric_value']):
            return None,'different_or_ambiguous_observation_count'
        # A mixed assertion may expose cases and deaths together; its first
        # count's unit does not describe every metric in the candidate.
        metric_count_unit='cases' if metric.startswith('cases_') else metric
        for field,allowed in (
                ('metric_unit',{'count'}),
                ('count_unit',{'count',metric_count_unit})):
            unit=row.get(field)
            if unit not in (None,'') and str(unit).strip().casefold() not in allowed:
                return None,'different_or_ambiguous_observation_count'
        if row.get('case_definition') and not _supports(
                'case_definition',row['case_definition'],quote,proposal):
            return None,'different_or_ambiguous_observation_count'
        identity_counts[metric]=row['metric_value']
    if not identity_counts or any(proposal.get(name)!=value for name,value in identity_counts.items()):
        return None,'different_or_ambiguous_observation_count'
    index=build_evidence_index(state)
    contract={**(state.get('structured_task') or {}),**(state.get('task_evidence_contract') or {})}
    original=assess_record_evidence(row,contract=contract,evidence_index=index)
    supported={field.field for field in original.field_evidence if field.supported}
    proposed=assess_record_evidence(proposal,contract=contract,evidence_index=index)
    fields={}; provenance={}
    for field in proposed.field_evidence:
        if field.field not in _GEO+_DATES or not field.supported or field.field in supported:
            continue
        if row.get(field.field)==field.value:
            continue
        if field.field=='geographic_scope' and any((fields.get(key) or row.get(key)) for key in ('country','subnational_location','locality')):
            continue
        # FieldEvidence.quote may combine distant, validated headings with the
        # local assertion. Persist its original slice, keeping context locators
        # separate so the next qualification can resolve the same source again.
        left,right=field.locator.get('char_start'),field.locator.get('char_end')
        source_text=document.get('clean_text') or ''
        if (field.document_hash!=document.get('content_hash') or
                not isinstance(left,int) or not isinstance(right,int) or
                not 0<=left<right<=len(source_text)):
            continue
        fields[field.field]=field.value
        provenance[field.field]={'quote':source_text[left:right],'document_hash':field.document_hash,
            'chunk_id':chunk['chunk_id'],**field.locator}
    if not fields:
        return None,'no_supported_field_gain'
    updated={**row,**fields}
    prior=row.get('field_provenance_json') or {}
    if isinstance(prior,str):
        try: prior=json.loads(prior)
        except ValueError: prior={}
    updated['field_provenance_json']={**prior,**provenance}
    qualification=assess_record_evidence(updated,contract=contract,evidence_index=index)
    new_supported={field.field for field in qualification.field_evidence if field.supported}
    if not supported<=new_supported or not set(fields)<=new_supported:
        return None,'repair_would_break_existing_binding'
    return {'record_id':row['record_id'],'before_fingerprint':fingerprint(row),
            'fields':fields,'field_provenance':provenance},None


def _new_independent_sources(before,after,documents):
    from .evidence_products import _lineage
    old_ids={str(source.get('source_id')) for source in before}
    old_urls={_url(source.get('canonical_url') or source.get('url')) for source in before}
    origins={source['source_id']:source['independence_group'] for source in _lineage(before+after)['sources']}
    old_origins={origins.get(sid) for sid in old_ids}
    hashes={doc.get('content_hash') for doc in documents if str(doc.get('source_id')) in old_ids and doc.get('content_hash')}
    result=[]
    seen=set()
    for source in after:
        sid=str(source.get('source_id') or '')
        if not sid or sid in old_ids or _url(source.get('canonical_url') or source.get('url')) in old_urls:
            continue
        if source.get('source_role_final') in {'excluded','context_only'} or source.get('blocked_from_fetch'):
            continue
        negative={"mismatch","wrong_disease","wrong_geography","wrong_period","unrelated",
                  "unrelated_disease","incompatible_disease","task_fit_mismatch","temporal_mismatch",
                  "geography_mismatch","not_task_relevant"}
        if any(str(source.get(key) or "").lower() in negative for key in (
                "target_fit_status","target_verification_status","source_disease_relevance_status",
                "disease_fit","geography_fit","date_fit","period_fit")):
            continue
        if origins.get(sid) in old_origins or origins.get(sid) in seen:
            continue
        if any(doc.get('content_hash') in hashes for doc in documents if str(doc.get('source_id'))==sid):
            continue
        # A linked representation of an already located report is not a new source.
        if any(source.get(key) in old_ids for key in ('parent_source_id','resource_parent_source_id')):
            continue
        result.append(sid); seen.add(origins.get(sid))
    return sorted(result)


def execute_recovery(plan,*,context,artifacts,budget):
    if isinstance(plan,dict):
        plan=RecoveryPlan([RecoveryAction(**a) for a in plan.get('actions') or []],plan.get('stop_reason'))
    from .nodes import content_processing as content
    from .nodes.extraction import structured_extraction, extraction_skip_reason
    discovery=importlib.import_module('.nodes.source_discovery',__package__)
    delta=RecoveryDelta(); previous=context.recovery; context.recovery=True
    working=dict(artifacts)
    try:
        for action in plan.actions:
            local={**working,'documents':[],'evidence_chunks':[],'raw_records':[],'extraction_attempted_chunk_ids':[]}
            status='completed'; error=None; outcome=None; search_details={}
            discovered_slice=None
            pending=_saved_pending_discoveries(working,context,action=action) if action.kind in {'search','assess_source'} else []
            reassessment=bool(pending)
            if action.kind=='assess_source' and (not pending or action.source_version!=_pending_source_version(pending[0],working)):
                delta.actions.append({**asdict(action),'status':'skipped','error':'pending_source_or_version_changed','outcome':'not_executed'})
                continue
            if reassessment:
                blockers=_pending_assessment_blockers(pending,working,budget)
                if blockers:
                    delta.actions.append({**asdict(action),'status':'budget_deferred','error':'source assessment needs downstream budget',
                        'outcome':'not_executed','search_executed':False,'new_independent_source_ids':[],'blocked_budget_kinds':blockers})
                    continue
            if action.kind=='search' and not reassessment and _adaptive(budget):
                snapshot=budget.snapshot() if hasattr(budget,'snapshot') else budget
                blockers=_new_source_search_blockers(snapshot.get('remaining') or {})
                if blockers:
                    delta.actions.append({**asdict(action), 'status':'budget_deferred',
                        'error':'new source discovery needs downstream budget: '+', '.join(blockers),
                        'outcome':'not_executed', 'search_executed':False,
                        'new_independent_source_ids':[], 'blocked_budget_kinds':blockers})
                    continue
            try:
                if reassessment:
                    local['source_registry']=[dict(source) for source in pending]
                    discovered_slice=(len(delta.sources),len(delta.sources)+len(pending))
                    delta.sources.extend(local['source_registry'])
                    search_details={'search_executed':False,'new_independent_source_ids':[],
                                    'assessment_source_ids':[source['source_id'] for source in pending]}
                    from .nodes.source_screening import source_screening, source_critic_and_uncertainty_routing
                    if action.kind=='assess_source' and all(source.get('source_identity_status')=='provider_deferred' for source in pending):
                        from .source_identity import apply_source_identity_to_registry
                        from .llm_clients import llm_source_identity_enabled
                        screening=importlib.import_module('.nodes.source_screening',__package__)
                        sources,assessments,summary=apply_source_identity_to_registry(
                            local['source_registry'],collection_spec=local.get('collection_spec'),
                            llm_enabled=llm_source_identity_enabled(),
                            max_sources=screening._parse_positive_int_env('LLM_SOURCE_IDENTITY_MAX_SOURCES'),
                            require_llm=screening._env_flag('LLM_SOURCE_IDENTITY_REQUIRE_LLM'),
                            allow_deterministic_fallback=screening._env_flag('LLM_SOURCE_IDENTITY_ALLOW_DETERMINISTIC_FALLBACK',default=True))
                        # Identity review cannot replace a decision already grounded in the body.
                        body_fields={'source_role_final','final_screening_decision','status','ready_for_content_fetch',
                                     'blocked_from_fetch','blocked_from_fetch_reason','target_verification_status',
                                     'extraction_ready','extraction_skip_reason'}
                        originals={str(source.get('source_id')):source for source in pending}
                        for source in sources:
                            original=originals.get(str(source.get('source_id')), {})
                            if original.get('task_fit_evidence_origin')=='fetched_content':
                                source.update({key:value for key,value in original.items()
                                    if key in body_fields or key.endswith('_fit') or key.startswith('task_fit_')})
                        local.update(source_registry=sources,source_identity_assessments=assessments,source_identity_summary=summary)
                    else:
                        local.update(source_screening(local))
                        local.update(source_critic_and_uncertainty_routing(local))
                    assessed={source.get('source_id') for source in local.get('source_registry') or []
                              if source.get('source_identity_status')=='assessed'}
                    incomplete={source['source_id'] for source in pending}-assessed
                    if incomplete:
                        status=('provider_deferred' if any(source.get('source_identity_status')=='provider_deferred' for source in local.get('source_registry') or []) else 'budget_deferred' if any(source.get('source_identity_status')=='budget_deferred'
                                for source in local.get('source_registry') or []) else 'failed')
                        error='source_assessment_incomplete'; outcome='assessment_incomplete'
                    else:
                        outcome='source_assessment'
                elif action.kind=='repair_fields':
                    update,reason=_repair_candidate_fields(action,working)
                    if update:
                        delta.record_updates.append(update)
                        outcome='supported_field_gain'
                    else:
                        status='completed_empty'; error=reason; outcome='no_field_gain'
                elif action.kind=='reparse':
                    selected=[d for d in working.get('documents') or [] if str(d.get('document_id') or d.get('source_id'))==action.target_id and (not action.document_hash or d.get('content_hash')==action.document_hash)]
                    if not selected:
                        status='skipped'; error='document_not_found'
                    else:
                        local['documents']=[_reparse_saved_document(d,context,strategy=action.strategy) for d in selected]
                elif action.kind=='extract':
                    local['documents']=working.get('documents') or []
                    selected=[c for c in working.get('evidence_chunks') or [] if str(c.get('chunk_id'))==action.target_id]
                    local['evidence_chunks']=[c for c in selected if not extraction_skip_reason(c)]
                    if not local['evidence_chunks']:
                        status='skipped'; error=next((extraction_skip_reason(c) for c in selected),'span_not_found')
                else:
                    sources=[s for s in working.get('source_registry') or [] if str(s.get('source_id'))==action.target_id]
                    if action.kind=='search':
                        query=dict(action.query or _recovery_query(working,action.target_id) or {})
                        if not query:
                            status='skipped'; error='no_unused_search_direction'
                        else:
                            query.update(query_id='recovery_'+action.action_id[:12],execution_status='planned_not_executed')
                            settings=discovery._source_search_settings_from_env()
                            totals={'selected_query_count':0,'executed_query_count':0,'raw_result_count':0,'deduped_result_count':0,'provider_error_count':0}
                            found,_,query_records,*_=discovery._execute_iterative_query_batch(query_batch=[query],provider=discovery._provider_for_settings(settings),settings=settings,seen_canonical_urls={_url(s.get('canonical_url') or s.get('url')) for s in working.get('source_registry') or []},totals=totals,task=working.get('structured_task') or {},evidence_state=working,max_queries_override=1)
                            search_details={'query':query['query'],'search_direction':query.get('search_direction'),
                                            'provider_channel':query.get('provider_channel'),
                                            'search_dispatched':any(row.get('selected_for_execution') and row.get('error')!='provider_unavailable' for row in query_records),
                                            **({'temporal_probe':query['temporal_probe']} if query.get('temporal_probe') else {}),
                                            'new_independent_source_ids':[],'search_executed':totals['executed_query_count']>0}
                            if totals.get('search_budget_exhausted') or totals.get('search_result_budget_exhausted'):
                                status='budget_exhausted'; error='shared_search_budget_exhausted'
                            elif not totals['executed_query_count']:
                                status='skipped'; error='search_not_executed'
                            elif totals['provider_error_count']:
                                status='failed'; error='search_provider_failed'
                            local['source_candidates']=[s.model_dump() for s in found]
                            local.update(discovery.source_dedup_and_registry(local))
                            # Cached result aliases must not reopen an existing
                            # screened/body-verified source under a new source ID.
                            known={_url(source.get('canonical_url') or source.get('url')):source
                                for source in working.get('source_registry') or []}
                            local['source_registry']=[{**source,'source_id':
                                known.get(_url(source.get('canonical_url') or source.get('url')),source)['source_id']}
                                for source in local.get('source_registry') or []
                                if _url(source.get('canonical_url') or source.get('url')) not in known
                                or known[_url(source.get('canonical_url') or source.get('url'))].get('source_identity_status')=='pending_assessment']
                            # A paid discovery survives downstream advisory failure.
                            # Pending identity carries no official/verified assertion.
                            local['source_registry']=[{**source,
                                'source_identity_unverified':True,
                                'source_identity_status':'pending_assessment',
                                'recovery_discovery':{'action_id':action.action_id,'session_id':context.session_dir.name,
                                    'task_fingerprint':fingerprint(working.get('structured_task') or {}),
                                    'query_fingerprint':fingerprint(discovery._search_request_identity(query)),
                                    'query':query.get('query')}}
                                for source in local.get('source_registry') or []]
                            discovered_slice=(len(delta.sources),len(delta.sources)+len(local['source_registry']))
                            delta.sources.extend(local['source_registry'])
                            from .nodes.source_screening import source_screening, source_critic_and_uncertainty_routing
                            local.update(source_screening(local))
                            local.update(source_critic_and_uncertainty_routing(local))
                            sources=local.get('source_registry') or []
                    if action.kind=='fetch' and _adaptive(budget):
                        jobs=context.frontier.snapshot()['items']
                        job=next((job for job in jobs if str(job['source_id'])==action.target_id),None)
                        if job and not sources:
                            sources=[_frontier_source(job)]
                        if job and job['status']=='completed':
                            status='skipped'; error='already_handled_by_frontier_batch'
                        if job and job['status']=='failed':
                            if not context.frontier.retry(job['target_id'],strategy=action.strategy):
                                status='skipped'; error='frontier_attempts_exhausted'
                    if status=='completed':
                        sources=[{**source,'acquisition_strategy':action.strategy} if str(source.get('source_id'))==action.target_id else source for source in sources]
                        if _adaptive(budget):
                            registry={str(s.get('source_id')):s for s in working.get('source_registry') or []}
                            registry.update({str(s.get('source_id')):s for s in sources})
                            local['source_registry']=list(registry.values())
                            local['documents']=working.get('documents') or []
                        else:
                            local['source_registry']=sources
                        if sources or _adaptive(budget) and any(job['status']=='pending' for job in context.frontier.snapshot()['items']):
                            try:
                                local.update(content.content_fetch_and_parse(local))
                            finally:
                                delta.sources.extend(local.get('source_registry') or [])
                        elif action.kind=='fetch':
                            status='skipped'; error='source_not_found'
                    elif action.kind=='search':
                        delta.sources.extend(sources)
                if action.kind!='repair_fields' and not reassessment and status=='completed':
                    if action.kind!='extract':
                        local.update(content.document_quality_check(local))
                        local.update(content.evidence_chunking_and_data_presence_flagging(local))
                    if action.kind!='extract':
                        attempted=set(working.get('extraction_attempted_chunk_ids') or [])
                        local['evidence_chunks']=[c for c in local.get('evidence_chunks') or [] if c.get('chunk_id') not in attempted]
                    if local.get('evidence_chunks'):
                        local.update(structured_extraction(local))
                    selected_ids={str(c.get('chunk_id')) for c in local.get('evidence_chunks') or []}
                    provider_deferred_ids=set((local.get('llm_extraction_summary') or {}).get('provider_deferred_chunk_ids') or [])&selected_ids
                    actual_ids=(set(local.get('extraction_attempted_chunk_ids') or [])&selected_ids)-provider_deferred_ids
                    if provider_deferred_ids:
                        status='provider_deferred'; error='provider_account_limit'
                    delta.attempted_chunk_ids.extend(sorted(actual_ids))
                    if action.kind=='extract' and action.target_id not in actual_ids and not local.get('raw_records'):
                        if _provider_stop(context.ledger.snapshot()):
                            status='provider_deferred'; error='provider_account_limit'
                        else:
                            status='skipped'; error='consumer_did_not_attempt_span'
                    def representation(doc):
                        return (doc.get('source_id'),doc.get('content_hash'),doc.get('text_hash'),doc.get('acquisition_status'))
                    old_documents={representation(doc) for doc in working.get('documents') or []}
                    action_documents=[doc for doc in local.get('documents') or []
                                      if action.kind=='reparse' or representation(doc) not in old_documents]
                    deferred=unresolved_acquisition_documents({'documents':action_documents}) if action.kind!='extract' else []
                    if deferred:
                        kinds={str(d.get('budget_exhausted_kind')) for d in deferred if d.get('budget_exhausted_kind')}
                        status='budget_exhausted' if kinds else 'failed'
                        error=('budget deferred: '+', '.join(sorted(kinds))) if kinds else 'acquisition_incomplete'
                    else:
                        failed=[d for d in action_documents if action.kind!='extract' and
                                d.get('acquisition_status') in {'parse_error','browser_error','request_error','blocked','error_page','http_error','attempts_exhausted'}]
                        if failed:
                            status='failed'; error='; '.join(sorted({str(d.get('fetch_error') or d.get('parse_error') or d['acquisition_status']) for d in failed}))
                        elif action.kind=='reparse' and not any(_readable_document(d) for d in local.get('documents') or []):
                            status='completed_empty'; error='parser_returned_no_readable_content'
                    failed_spans=set(context.ledger.failed_chunk_ids())&selected_ids
                    if failed_spans and status!='provider_deferred':
                        status='failed'; error='; '.join(filter(None,[error,'extraction_failed']))
                    outcome=outcome or ('records' if local.get('raw_records') else 'empty' if actual_ids else 'content' if local.get('documents') else 'empty_search')
                delta.documents.extend(local.get('documents') or [])
                delta.chunks.extend(local.get('evidence_chunks') or [])
                delta.records.extend(local.get('raw_records') or [])
                if action.kind=='fetch' and _adaptive(budget) and status not in {'skipped','failed'}:
                    target_job=next((job for job in context.frontier.snapshot()['items']
                                     if str(job['source_id'])==action.target_id),None)
                    if target_job and target_job['status']!='completed':
                        status={'budget_deferred':'budget_exhausted','failed':'failed',
                                'blocked':'skipped','pending':'pending','running':'pending'}.get(target_job['status'],'failed')
                        error=target_job.get('reason') or 'frontier_not_dispatched'
                if action.kind=='search' and not reassessment:
                    search_details['new_independent_source_ids']=_new_independent_sources(
                        working.get('source_registry') or [],local.get('source_registry') or [],
                        [*(working.get('documents') or []),*(local.get('documents') or [])])
            except ProviderAccountLimit as exc:
                status='provider_deferred'; error='provider_account_limit'
            except BudgetExceeded as exc:
                status='budget_exhausted' if exc.kind!='action_attempts' else 'failed'; error=str(exc)
            except Exception as exc:
                status='failed'; error=type(exc).__name__+': '+str(exc)
            finally:
                if discovered_slice is not None:
                    start,end=discovered_slice
                    latest={source.get('source_id'):source for source in local.get('source_registry') or []}
                    delta.sources[start:end]=[latest.get(source.get('source_id'),source)
                        for source in delta.sources[start:end]]
                    if error:
                        for source in delta.sources[start:end]:
                            if source.get('source_identity_status')=='pending_assessment':
                                source['source_identity_errors']=list(dict.fromkeys(
                                    [*(source.get('source_identity_errors') or []),error]))
                                source['source_identity_warnings']=list(dict.fromkeys(
                                    [*(source.get('source_identity_warnings') or []),'source_assessment_incomplete']))
            delta.actions.append({**asdict(action),**search_details,'status':status,'error':error,'outcome':outcome})
            working={**working,**merge_recovery_delta(working,delta)}
    finally:
        context.recovery=previous
    return delta


def recovery_execute(state):
    runtime=get_runtime()
    if runtime is None: raise RuntimeError('recovery requires session runtime')
    runtime.recovery_round=int(state.get('recovery_round') or 0)+1
    delta=execute_recovery(state.get('recovery_plan') or {},context=runtime,artifacts=state,budget=runtime.ledger)
    update=merge_recovery_delta(state,delta)
    update.update(recovery_round=int(state.get('recovery_round') or 0)+1,recovery_action_history=[*(state.get('recovery_action_history') or []),*delta.actions],run_budget_ledger=runtime.ledger.snapshot())
    return update
