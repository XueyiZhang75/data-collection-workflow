"""Session-local operation accounting, caching, and checkpoints."""
from __future__ import annotations

from data_collection_workflow.environment import get_env
from .acquisition_budget import ADAPTIVE_DEFAULTS, canonical_source_target, is_adaptive_budget, nonnegative_integer

import base64
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from .provider_failures import ProviderAccountLimit, provider_account_limit

REVISION = 'evidence'

class BudgetExceeded(RuntimeError):
    def __init__(self, message, *, kind=None):
        super().__init__(message)
        self.kind = kind

class ResumeMismatch(RuntimeError):
    pass

def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), default=str)

def fingerprint(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()

def policy_fingerprint():
    root = Path(__file__).resolve().parent
    files = [('src/data_collection_workflow/' + p.relative_to(root).as_posix(),
              hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in {'.py','.json','.csv'}]
    # Only executable workflow entry points belong here, not generated artifacts
    # or independent evaluation/report scripts. Missing packaged runners are explicit.
    for name in ('run_workflow.py', 'collect.py'):
        path = root.parents[1] / 'scripts' / name
        files.append(('scripts/' + name, hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None))
    return fingerprint(files)

def _validate_resume_manifest(session_dir, config):
    """Read-only validation must precede any preflight or ledger mutation."""
    manifest = Path(session_dir) / '.universal/session.json'
    if not manifest.exists():
        raise ResumeMismatch('cannot resume a missing session')
    expected = fingerprint({'config': config, 'policy': policy_fingerprint()})
    saved = json.loads(manifest.read_text(encoding='utf-8'))
    if saved.get('fingerprint') != expected:
        raise ResumeMismatch('task/config/policy fingerprint mismatch')
    return expected


def derive_budget_limits(config):
    adaptive = is_adaptive_budget(config)
    search = config.get('source_search') or {}
    iterative = search.get('iterative') or {}
    authority = search.get('authority_gap_retry') or {}
    extraction = (config.get('llm') or {}).get('extraction') or {}
    scheduler = extraction.get('scheduler') or {}
    fetch = config.get('content_fetch') or {}
    base_queries = int(iterative.get('max_total_queries',20)) if iterative.get('enabled',True) else int(search.get('max_queries',8))
    extra = sum(int(authority.get(k,d)) for k,d in [('max_queries',24),('known_domain_max_queries',10),('jurisdiction_max_queries',12),('official_page_family_max_queries',10)]) if authority.get('enabled',True) else 0
    base_results = min(int(search.get('max_total_results',64)), int(iterative.get('max_total_results',120))) if iterative.get('enabled',True) else int(search.get('max_total_results',64))
    limits = {'search': base_queries+extra, 'search_results': base_results+(int(authority.get('result_budget',80)) if authority.get('enabled',True) else 0),
              'fetch': 200, 'fetch_ordinary': min(int(fetch.get('max_total_sources',18)),int(fetch.get('max_search_derived_sources',18))),
              'browser': 20, 'ocr': 100, 'extraction': min(2400,int(scheduler.get('safety_max_calls',2400)))}
    if adaptive:
        limits.pop('fetch'); limits.pop('fetch_ordinary')
        limits.update(ADAPTIVE_DEFAULTS)
    if not search.get('enabled',True):
        limits['search'] = limits['search_results'] = 0
    stage_aliases = {'SourceIdentityAssessment': 'SourceIdentityAgentOutput',
                     'SourceCredibilityAssessment': 'LLMSourceCredibilitySuggestion'}
    def restrict(kind, value):
        if kind.startswith('model:'):
            stage = kind.removeprefix('model:')
            kind = 'model:' + stage_aliases.get(stage, stage)
        if adaptive and kind in {'fetch', 'fetch_ordinary'}:
            raise ValueError('adaptive policy uses source_targets/http_requests, not '+kind)
        cap = nonnegative_integer(value, kind) if adaptive else max(0, int(value))
        limits[kind] = cap if adaptive and kind in ADAPTIVE_DEFAULTS else min(limits.get(kind, cap), cap)
    for kind, value in ((config.get('universal') or {}).get('budget_limits') or {}).items():
        restrict(kind, value)
    for stage,section in [('SourceIdentityAgentOutput','source_identity'),('LLMSourceCredibilitySuggestion','source_credibility'),('SourceCriticAgentOutput','source_critic')]:
        cap=((config.get('llm') or {}).get(section) or {}).get('max_sources')
        if cap is not None: restrict('model:'+stage, cap)
    for kind, value in ((config.get('universal') or {}).get('model_limits') or {}).items():
        restrict('model:'+kind, value)
    return limits

def _encode(value):
    if isinstance(value,bytes):
        return {'__bytes__':base64.b64encode(value).decode('ascii')}
    if hasattr(value,'model_dump'):
        module = type(value).__module__
        if not module.startswith(('data_collection_workflow.models','langchain_core.messages')):
            raise TypeError('unsupported cached model type: '+module)
        return {'__model__':module+':'+type(value).__name__, 'value':_encode(value.model_dump())}
    if isinstance(value,dict):
        return {str(k):_encode(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_encode(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    raise TypeError('unsupported cached response: '+type(value).__name__)

def _decode(value):
    if isinstance(value,list):
        return [_decode(v) for v in value]
    if isinstance(value,dict):
        if set(value)=={'__bytes__'}:
            return base64.b64decode(value['__bytes__'])
        if set(value)=={'__model__','value'}:
            module,name=value['__model__'].split(':',1)
            if not module.startswith(('data_collection_workflow.models','langchain_core.messages')):
                raise ValueError('untrusted response type')
            return getattr(importlib.import_module(module),name).model_validate(_decode(value['value']))
        return {k:_decode(v) for k,v in value.items()}
    return value

def provider_token_usage(value):
    """Normalize provider-reported counts; absence is unknown, never zero."""
    usage=getattr(value,'usage_metadata',None)
    normalized=bool(usage)
    metadata=getattr(value,'response_metadata',None) or {}
    if isinstance(value,dict):
        normalized=bool(value.get('usage_metadata'))
        usage=value.get('usage_metadata') or value.get('token_usage') or value.get('usage')
        metadata=value.get('response_metadata') or metadata
    usage=usage or metadata.get('token_usage') or metadata.get('usage')
    if not isinstance(usage,dict): return None
    result={}
    for key,alias in [('input_tokens','prompt_tokens'),('output_tokens','completion_tokens'),('total_tokens','total_tokens')]:
        count=usage.get(key,usage.get(alias))
        result[key]=count if isinstance(count,int) and not isinstance(count,bool) and count>=0 else None
    # Raw Anthropic usage excludes cached input. LangChain usage_metadata already
    # includes it, so only add the provider's separate counters to raw usage.
    cache_keys=('cache_read_input_tokens','cache_creation_input_tokens')
    if not normalized and any(key in usage for key in cache_keys):
        cache_counts=[usage[key] for key in cache_keys if key in usage]
        if result['input_tokens'] is not None and all(
            isinstance(count,int) and not isinstance(count,bool) and count>=0
            for count in cache_counts
        ):
            result['input_tokens']+=sum(cache_counts)
        else:
            result['input_tokens']=None
    if result['total_tokens'] is None and result['input_tokens'] is not None and result['output_tokens'] is not None:
        result['total_tokens']=result['input_tokens']+result['output_tokens']
    return result if any(v is not None for v in result.values()) else None


class RunBudgetLedger:
    def __init__(self,path,limits,*,extraction_reserve=40,adaptive=False):
        self.path=Path(path); self.limits=dict(limits); self.base_limits=dict(limits)
        self.adaptive=bool(adaptive)
        self.reserve=min(max(0,int(extraction_reserve)),self.limits.get('extraction',2400))
        with self._db() as db:
            db.executescript('CREATE TABLE IF NOT EXISTS operations (id INTEGER PRIMARY KEY, opkey TEXT, kind TEXT, status TEXT, response TEXT, error TEXT, started REAL); CREATE TABLE IF NOT EXISTS charges (operation INTEGER, kind TEXT, amount INTEGER); CREATE INDEX IF NOT EXISTS operation_key ON operations(opkey); CREATE TABLE IF NOT EXISTS extraction_attempts (chunk_id TEXT PRIMARY KEY); CREATE TABLE IF NOT EXISTS extraction_operation_chunks (operation INTEGER, chunk_id TEXT, PRIMARY KEY(operation,chunk_id)); CREATE INDEX IF NOT EXISTS extraction_chunk_operation ON extraction_operation_chunks(chunk_id);')
            db.execute('CREATE TABLE IF NOT EXISTS provider_state (provider TEXT PRIMARY KEY, status TEXT NOT NULL, event TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS provider_events (event_id TEXT PRIMARY KEY, event TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS acquisition_targets (target_id TEXT PRIMARY KEY, first_operation INTEGER NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS budget_amendments (revision INTEGER PRIMARY KEY, amendment_id TEXT UNIQUE NOT NULL, event TEXT NOT NULL)')
            columns={r['name'] for r in db.execute('PRAGMA table_info(operations)')}
            for column in ('token_usage','metadata'):
                if column not in columns: db.execute('ALTER TABLE operations ADD COLUMN '+column+' TEXT')
            if 'response_ready' not in columns:
                db.execute('ALTER TABLE operations ADD COLUMN response_ready INTEGER NOT NULL DEFAULT 0')
            self._effective_budget(db)

    @contextmanager
    def _db(self):
        db=sqlite3.connect(self.path,timeout=60)
        db.execute('PRAGMA busy_timeout=60000')
        db.row_factory=sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def _costs(self,kind):
        if self.adaptive:
            if kind in {'fetch_priority','fetch_ordinary','http_request'}: return {'http_requests':1}
            if kind in {'browser_fetch','browser_navigation'}: return {'browser':1}
        if kind=='fetch_priority': return {'fetch':1}
        if kind=='fetch_ordinary': return {'fetch':1,'fetch_ordinary':1}
        if kind=='browser_fetch': return {'browser':1,'fetch':1}
        return {kind:1}

    def _effective_budget(self, db):
        limits = dict(self.base_limits)
        events = [json.loads(row['event']) for row in db.execute('SELECT event FROM budget_amendments ORDER BY revision')]
        for event in events:
            limits.update(event['increases'])
        self.limits = limits
        return limits, events

    @property
    def budget_revision(self):
        with self._db() as db:
            return int(db.execute('SELECT COALESCE(MAX(revision),0) FROM budget_amendments').fetchone()[0])

    def amend_budget(self, *, amendment_id, increases, reason):
        if not self.adaptive:
            raise ValueError('budget amendments require adaptive policy version 2')
        if not isinstance(amendment_id,str) or not amendment_id.strip() or not isinstance(reason,str) or not reason.strip():
            raise ValueError('amendment_id and reason are required')
        if not isinstance(increases,dict) or not increases:
            raise ValueError('an amendment must name budgets to increase')
        normalized = {key: nonnegative_integer(value,key) for key,value in increases.items()}
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            prior = db.execute('SELECT event FROM budget_amendments WHERE amendment_id=?',(amendment_id,)).fetchone()
            if prior:
                event = json.loads(prior['event'])
                if event['increases'] != normalized or event['reason'] != reason:
                    raise ValueError('amendment_id already used for different changes')
                self._effective_budget(db)
                return event
            limits, events = self._effective_budget(db)
            used = {row['kind']:row['total'] for row in db.execute('SELECT kind,SUM(amount) total FROM charges GROUP BY kind')}
            known = set(limits) | {key for key in used if key.startswith('model:')}
            for key,value in normalized.items():
                if key not in known or key not in {'source_targets','http_requests','search','search_results','extraction','browser','ocr'} and not key.startswith('model:'):
                    raise ValueError('unknown budget dimension: '+key)
                if value <= limits.get(key,30):
                    raise ValueError('budget amendments may only increase limits')
            updated = {**limits, **normalized}
            event = {'amendment_id':amendment_id,'budget_revision':len(events)+1,
                     'previous_limits':limits,'limits':updated,'increases':normalized,
                     'reason':reason,'timestamp':time.time(),'used_at_amendment':used}
            db.execute('INSERT INTO budget_amendments VALUES (?,?,?)',(event['budget_revision'],amendment_id,_json(event)))
            self.limits = updated
            return event

    def provider_status(self, provider):
        with self._db() as db:
            row=db.execute('SELECT event,status FROM provider_state WHERE provider=?',(str(provider).lower(),)).fetchone()
            return {**json.loads(row['event']), 'status':row['status']} if row else None

    def resume_provider(self, *, provider, event_id, reason):
        if not all(isinstance(v,str) and v.strip() for v in (provider,event_id,reason)):
            raise ValueError('provider, event_id and reason are required')
        if event_id.startswith('stop:'):
            raise ValueError('stop: event IDs are reserved for provider refusals')
        provider=provider.strip().lower()
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            prior=db.execute('SELECT event FROM provider_events WHERE event_id=?',(event_id,)).fetchone()
            if prior:
                event=json.loads(prior['event'])
                if any(event.get(k)!=v for k,v in {'provider':provider,'reason':reason,'status':'resumed'}.items()):
                    raise ValueError('provider event_id already used for different changes')
                return event
            row=db.execute('SELECT status,event FROM provider_state WHERE provider=?',(provider,)).fetchone()
            if not row or row['status']!='halted':
                raise ValueError('provider is not halted in this session')
            event={'event_id':event_id,'provider':provider,'status':'resumed','reason':reason,
                   'timestamp':time.time(),'previous_stop':json.loads(row['event'])}
            db.execute('INSERT INTO provider_events VALUES (?,?)',(event_id,_json(event)))
            db.execute('UPDATE provider_state SET status=?,event=? WHERE provider=?',('resumed',_json(event),provider))
            return event

    def has_source_target(self, url):
        """Inspect an already charged canonical target without reserving budget."""
        target = canonical_source_target(url)
        with self._db() as db:
            return db.execute('SELECT 1 FROM acquisition_targets WHERE target_id=?', (target,)).fetchone() is not None

    def has_cached(self,kind,payload):
        """Inspect the exact latest response without reserving or spending budget."""
        key=fingerprint({'kind':kind,'payload':payload})
        with self._db() as db:
            row=db.execute('SELECT status FROM operations WHERE opkey=? ORDER BY id DESC LIMIT 1',(key,)).fetchone()
            return row is not None and row['status']=='completed'

    def begin(self,kind,payload,*,recovery=False,metadata=None,source_target=None,provider=None):
        key=fingerprint({'kind':kind,'payload':payload})
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            rows=db.execute('SELECT * FROM operations WHERE opkey=? ORDER BY id DESC',(key,)).fetchall()
            extraction_input=(payload.get('input') or {}) if kind=='extraction' else {}
            chunk_ids=sorted({str(key) for key in [extraction_input.get('chunk_id'),*(extraction_input.get('chunk_ids') or [])] if key})
            if rows and rows[0]['status']=='completed':
                db.executemany('INSERT OR IGNORE INTO extraction_attempts VALUES (?)',[(key,) for key in chunk_ids])
                return {'cached':True,'value':_decode(json.loads(rows[0]['response']))}
            if provider:
                stopped=db.execute("SELECT event FROM provider_state WHERE provider=? AND status='halted'",(provider,)).fetchone()
                if stopped:
                    raise ProviderAccountLimit(provider,json.loads(stopped['event']),dispatched=False)
            if rows and rows[0]['status'] in {'running','in_doubt'}:
                raise ResumeMismatch('in_doubt operation cannot be dispatched again: '+key)
            if len(rows)>=2:
                raise BudgetExceeded('same action attempt limit reached', kind='action_attempts')
            for chunk_id in chunk_ids:
                attempts=db.execute('SELECT o.status FROM operations o JOIN extraction_operation_chunks c ON c.operation=o.id WHERE c.chunk_id=? ORDER BY o.id DESC',(chunk_id,)).fetchall()
                if any(attempt['status'] in {'running','in_doubt'} for attempt in attempts):
                    raise ResumeMismatch('in_doubt extraction span cannot be dispatched again: '+chunk_id)
                if len(attempts)>=2:
                    raise BudgetExceeded('extraction span attempt limit reached', kind='action_attempts')
            costs=self._costs(kind)
            target = None
            if self.adaptive and 'http_requests' in costs:
                target = canonical_source_target(source_target or (payload.get('input') or payload).get('url'))
                if db.execute('SELECT 1 FROM acquisition_targets WHERE target_id=?',(target,)).fetchone() is None:
                    costs['source_targets']=1
            limits, amendments = self._effective_budget(db)
            used={r['kind']:r['total'] for r in db.execute('SELECT kind,SUM(amount) total FROM charges GROUP BY kind')}
            for k,n in costs.items():
                limit=limits.get(k,30 if k.startswith('model:') else 0)
                if k=='extraction' and not recovery: limit-=self.reserve
                if used.get(k,0)+n>limit:
                    raise BudgetExceeded(k+' budget exhausted', kind=k)
            cur=db.execute('INSERT INTO operations(opkey,kind,status,started) VALUES (?,?,?,?)',(key,kind,'running',time.time()))
            ident=cur.lastrowid
            if target:
                db.execute('INSERT OR IGNORE INTO acquisition_targets VALUES (?,?)',(target,ident))
            audit={**(metadata or {}),**({'provider':provider} if provider else {}),'recovery':bool(recovery),'budget_before':used,'budget_changes':costs}
            if self.adaptive:
                audit.update(source_target=target or source_target,budget_revision=len(amendments))
            db.execute('UPDATE operations SET metadata=? WHERE id=?',(_json(audit),ident))
            db.executemany('INSERT OR IGNORE INTO extraction_attempts VALUES (?)',[(key,) for key in chunk_ids])
            db.executemany('INSERT INTO extraction_operation_chunks VALUES (?,?)',[(ident,key) for key in chunk_ids])
            db.executemany('INSERT INTO charges VALUES (?,?,?)',[(ident,k,n) for k,n in costs.items()])
            return {'cached':False,'id':ident,'key':key}

    def stage_response(self, ticket, value):
        """Durably mark a complete response before finalizing its operation status."""
        response=json.dumps(_encode(value),ensure_ascii=False)
        with self._db() as db:
            db.execute("UPDATE operations SET response=?,response_ready=1 WHERE id=? AND status='running'",
                       (response,ticket['id']))

    def finish(self,ticket,value=None,error=None,usage=None,provider_stop=None):
        # Failed raw responses remain audit artifacts; only completed rows are cached.
        response=json.dumps(_encode(value),ensure_ascii=False) if value is not None or error is None else None
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            counts=usage or provider_token_usage(value)
            db.execute('UPDATE operations SET status=?,response=?,error=?,token_usage=? WHERE id=?',('failed' if error else 'completed',response,error,_json(counts) if counts is not None else None,ticket['id']))
            if provider_stop:
                provider=provider_stop['provider']
                row=db.execute('SELECT status FROM provider_state WHERE provider=?',(provider,)).fetchone()
                if not row or row['status']!='halted':
                    event={**provider_stop,'status':'halted','first_operation':ticket['id'],
                           'timestamp':time.time(),'event_id':'stop:'+str(ticket['id'])}
                    db.execute('INSERT OR REPLACE INTO provider_state VALUES (?,?,?)',(provider,'halted',_json(event)))
                    db.execute('INSERT INTO provider_events VALUES (?,?)',(event['event_id'],_json(event)))

    def attempted_chunk_ids(self):
        with self._db() as db:
            return [r['chunk_id'] for r in db.execute('SELECT chunk_id FROM extraction_attempts ORDER BY chunk_id')]

    def failed_chunk_ids(self, *, retryable_only=False):
        """Return spans whose latest real extraction failed, never valid empty output."""
        with self._db() as db:
            rows=db.execute('SELECT c.chunk_id,o.status FROM extraction_operation_chunks c JOIN operations o ON o.id=c.operation ORDER BY o.id').fetchall()
        latest={}; counts={}
        for row in rows:
            key=row['chunk_id']; latest[key]=row['status']; counts[key]=counts.get(key,0)+1
        return sorted(key for key,status in latest.items() if status=='failed' and (not retryable_only or counts[key]<2))

    def mark_interrupted(self):
        with self._db() as db:
            if self.adaptive:
                db.execute("UPDATE operations SET status='completed',error=NULL WHERE status IN ('running','in_doubt') AND response_ready=1")
                db.execute("""UPDATE operations SET status='failed',error='Interrupted: transport outcome unknown'
                    WHERE status IN ('running','in_doubt') AND response_ready=0
                    AND kind IN ('fetch_ordinary','fetch_priority','http_request','browser_fetch','browser_navigation')""")
            db.execute("UPDATE operations SET status='in_doubt' WHERE status='running'")

    def operation_audit(self):
        with self._db() as db:
            return [{'operation_id':r['id'],'action_id':r['opkey'],'kind':r['kind'],'status':r['status'],'started':r['started'],'error':r['error'],'token_usage':json.loads(r['token_usage']) if r['token_usage'] else None,**(json.loads(r['metadata']) if r['metadata'] else {})} for r in db.execute('SELECT * FROM operations ORDER BY id')]

    def snapshot(self):
        with self._db() as db:
            used={r['kind']:r['total'] for r in db.execute('SELECT kind,SUM(amount) total FROM charges GROUP BY kind')}
            statuses={r['status']:r['n'] for r in db.execute('SELECT status,COUNT(*) n FROM operations GROUP BY status')}
            model_rows=db.execute("SELECT token_usage FROM operations WHERE kind='extraction' OR kind LIKE 'model:%'").fetchall()
            limits, amendments = self._effective_budget(db)
            providers={row['provider']:{**json.loads(row['event']),'status':row['status']} for row in db.execute('SELECT * FROM provider_state')}
            provider_events=[json.loads(row['event']) for row in db.execute('SELECT event FROM provider_events ORDER BY rowid')]
        for kind in used:
            if kind.startswith('model:'): limits.setdefault(kind,30)
        known=[json.loads(r['token_usage']) for r in model_rows if r['token_usage']]
        counts={k:sum(row[k] for row in known if row.get(k) is not None) if any(row.get(k) is not None for row in known) else None for k in ('input_tokens','output_tokens','total_tokens')}
        complete=len(known)==len(model_rows) and bool(known) and all(row.get('availability')!='partial' and all(row.get(k) is not None for k in counts) for row in known)
        counts.update(availability='complete' if complete else 'partial' if known else 'unavailable',known_operations=len(known),model_operations=len(model_rows))
        return {'providers':providers,'provider_events':provider_events,'limits':limits,'budget_revision':len(amendments),'budget_amendments':amendments,'model_default_limit':30,'used':used,'remaining':{k:max(0,v-used.get(k,0)) for k,v in limits.items()},'operations':statuses,'extraction_reserve':self.reserve,'token_usage':counts,'cost_usd':None,'cost_availability':'unavailable'}

_MODEL_DISPATCH=threading.local()

@contextmanager
def capture_model_dispatch():
    """Observe this worker's actual ledger admission, including cache-only calls."""
    previous=getattr(_MODEL_DISPATCH,'receipt',None)
    receipt={'dispatched':None}
    _MODEL_DISPATCH.receipt=receipt
    try:
        yield receipt
    finally:
        _MODEL_DISPATCH.receipt=previous

_ACTIVE_CONTEXT=None
_CONTEXT_LOCK=threading.RLock()

class RunContext:
    def __init__(self,session_dir,config,*,resume=False):
        self.session_dir=Path(session_dir).resolve(); self.config=config
        self.fingerprint=(_validate_resume_manifest(self.session_dir,config) if resume
                          else fingerprint({'config':config,'policy':policy_fingerprint()}))
        store=self.session_dir/'.universal'
        store.mkdir(parents=True,exist_ok=True)
        manifest=store/'session.json'
        if manifest.exists():
            if not resume:
                raise ResumeMismatch('existing session requires explicit resume')
            saved=json.loads(manifest.read_text(encoding='utf-8'))
            if saved.get('fingerprint')!=self.fingerprint:
                raise ResumeMismatch('task/config/policy fingerprint mismatch')
        elif resume:
            raise ResumeMismatch('cannot resume a missing session')
        else:
            with manifest.open('x',encoding='utf-8') as stream:
                json.dump({'revision':REVISION,'fingerprint':self.fingerprint},stream)
        universal=config.get('universal') or {}
        extraction=(config.get('llm') or {}).get('extraction') or {}
        self.ledger=RunBudgetLedger(store/'operations.sqlite',derive_budget_limits(config),extraction_reserve=universal.get('extraction_reserve',extraction.get('focused_recovery_reserved_calls',40)),adaptive=is_adaptive_budget(config))
        from .acquisition_frontier import AcquisitionFrontier
        self.frontier=AcquisitionFrontier(store/'operations.sqlite') if is_adaptive_budget(config) else None
        self.budget_continuation=False
        self.provider_continuation=False
        if resume:
            self.ledger.mark_interrupted()
            if self.frontier:
                transports={'fetch_ordinary','fetch_priority','http_request','browser_fetch','browser_navigation'}
                audit=self.ledger.operation_audit()
                outcomes={}
                for item in self.frontier.snapshot()['items']:
                    if item['status']!='running': continue
                    target=canonical_source_target(item['target_id'])
                    rows=[row for row in audit if row['kind'] in transports and row.get('source_target')
                          and canonical_source_target(row['source_target'])==target and row['started']>=item['updated']]
                    if any(row['status'] in {'running','in_doubt'} for row in rows): continue
                    started=bool(rows)
                    unknown=any(row.get('error')=='Interrupted: transport outcome unknown' for row in rows)
                    terminal=unknown and self.frontier.attempts_for_strategy(item)+int(started)>=2
                    outcomes[item['target_id']]={'status':'failed' if terminal else 'pending',
                        'operation_started':started,'reason':'interrupted_transport_attempts_exhausted' if terminal else 'reconciled_interrupted_claim'}
                self.frontier.reconcile_running(outcomes)
        self.execution_instance=uuid.uuid4().hex
        self.recovery_round=None
        self.recovery=False

    def has_cached(self,kind,payload):
        return self.ledger.has_cached(kind,{'fingerprint':self.fingerprint,'input':payload})

    def call(self,kind,payload,fn,*,recovery=False,usage_supplier=None,operation_metadata=None,source_target=None):
        provider=str((payload.get('settings') or {}).get('provider') or (self.config.get('llm') or {}).get('provider') or '').lower() if kind=='extraction' or kind.startswith('model:') else None
        ticket=self.ledger.begin(kind,{'fingerprint':self.fingerprint,'input':payload},recovery=recovery or self.recovery,metadata={'execution_instance':self.execution_instance,'recovery_round':self.recovery_round,**(operation_metadata or {})},source_target=source_target,provider=provider)
        receipt=getattr(_MODEL_DISPATCH,'receipt',None)
        if provider and receipt is not None:
            receipt['dispatched']=not ticket['cached']
        if ticket['cached']: return ticket['value']
        try:
            value=fn()
            if self.ledger.adaptive: self.ledger.stage_response(ticket,value)
            self.ledger.finish(ticket,value=value,usage=usage_supplier() if usage_supplier else None)
            return value
        except BaseException as exc:
            if isinstance(exc,Exception):
                limit=provider_account_limit(exc) if provider else None
                stop={**limit,'provider':provider} if limit else None
                self.ledger.finish(ticket,value=getattr(exc,'response_payload',getattr(exc,'llm_raw_response',None)),error=type(exc).__name__+': '+str(exc),usage=usage_supplier() if usage_supplier else None,provider_stop=stop)
                if stop:
                    raise ProviderAccountLimit(provider,self.ledger.provider_status(provider),dispatched=True) from exc
            raise

    def amend_budget(self, *, amendment_id, increases, reason):
        event=self.ledger.amend_budget(amendment_id=amendment_id,increases=increases,reason=reason)
        self.frontier.resume_budget_deferred(budget_revision=event['budget_revision'])
        self.budget_continuation=True
        return event

    @contextmanager
    def activate(self):
        # The runner owns process-wide configuration; reject overlap, including
        # nested activation, while allowing its worker threads to share this run.
        global _ACTIVE_CONTEXT
        with _CONTEXT_LOCK:
            if _ACTIVE_CONTEXT is not None:
                raise ResumeMismatch('only one active session is supported per process')
            _ACTIVE_CONTEXT=self
            previous=os.environ.get('UNIVERSAL_SESSION_DIR')
            os.environ['UNIVERSAL_SESSION_DIR']=str(self.session_dir)
        try: yield self
        finally:
            with _CONTEXT_LOCK:
                _ACTIVE_CONTEXT=None
                if previous is None: os.environ.pop('UNIVERSAL_SESSION_DIR',None)
                else: os.environ['UNIVERSAL_SESSION_DIR']=previous

def get_runtime():
    with _CONTEXT_LOCK:
        return _ACTIVE_CONTEXT


def external_call(kind,payload,fn,*,recovery=False,usage_supplier=None,operation_metadata=None):
    if get_env('PIPELINE_MODE') != REVISION:
        return fn()
    context=get_runtime()
    if context is None:
        raise ResumeMismatch('evidence external calls require an initialized session runtime')
    return context.call(kind,payload,fn,recovery=recovery,usage_supplier=usage_supplier,operation_metadata=operation_metadata)


class SessionArtifactSerializer:
    """Checkpoints store session-local content references for large values."""
    def __init__(self,directory):
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        self.directory=Path(directory).resolve(); self.directory.mkdir(parents=True,exist_ok=True)
        self.delegate=JsonPlusSerializer()

    def dumps_typed(self,value):
        kind,body=self.delegate.dumps_typed(value)
        if len(body)<4096: return kind,body
        digest=hashlib.sha256(body).hexdigest()
        target=self.directory/(digest+'.bin')
        try:
            with target.open('xb') as stream: stream.write(body)
        except FileExistsError:
            if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
                raise ResumeMismatch('checkpoint artifact hash mismatch')
        return 'session_artifact_v1',json.dumps({'kind':kind,'sha256':digest}).encode()

    def loads_typed(self,value):
        kind,body=value
        if kind!='session_artifact_v1': return self.delegate.loads_typed(value)
        reference=json.loads(body)
        digest=reference.get('sha256') or ''
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ResumeMismatch('invalid checkpoint reference')
        payload=(self.directory/(digest+'.bin')).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=digest:
            raise ResumeMismatch('checkpoint artifact hash mismatch')
        return self.delegate.loads_typed((reference['kind'],payload))

def prepare_universal_config(config):
    from copy import deepcopy
    result=deepcopy(config)
    if result.get('pipeline_mode')!=REVISION: return result
    workflow=result.setdefault('workflow',{})
    workflow.update(seed_source_overlay_path=None,source_role_policy_overlay_path=None,use_fixture_documents=False,validation_ground_truth_records_path=None)
    result.setdefault('source_search',{})['combine_with_seed_catalog']=False
    result['source_sets']={'source_id_allowlist_enabled':False,'collection_source_ids':[],'context_source_ids':[],'validation_reserved_source_ids':[],'workflow_source_ids':[],'llm_source_critic_source_ids':[]}
    result.setdefault('validation',{}).update(held_out_records_path=None,validation_records_path=None)
    result.setdefault('output',{})['write_latest_alias']=False
    result['output']['write_latest_console_alias']=False
    return result

def initialize_universal_run(config,session_dir,*,resume_session=None):
    from .document_acquisition import preflight_acquisition
    directory=Path(session_dir).resolve()
    if resume_session and str(resume_session)!=directory.name:
        raise ResumeMismatch('resume must explicitly name this session')
    if not resume_session and directory.exists() and any(directory.iterdir()):
        raise ResumeMismatch('new run requires an empty output directory')
    if resume_session:
        _validate_resume_manifest(directory,config)
    # Checks execute locally before any budgeted provider/model call.
    readiness=preflight_acquisition((config.get('universal') or {}).get('acquisition') or {},directory)
    context=RunContext(directory,config,resume=bool(resume_session))
    context.resume=bool(resume_session)
    context.readiness=readiness
    return context

def require_ready_runtime(state):
    context=get_runtime()
    if context is None or not getattr(context,'readiness',{}).get('ready'):
        raise ResumeMismatch('evidence requires mandatory browser/OCR preflight before graph execution')
    return {'pipeline_mode':REVISION,'session_artifact_root':str(context.session_dir),'run_budget_ledger':context.ledger.snapshot()}

@contextmanager
def checkpoint_graph(context):
    from langgraph.checkpoint.sqlite import SqliteSaver
    from .graph import build_graph
    connection=sqlite3.connect(context.session_dir/'.universal/checkpoints.sqlite',check_same_thread=False)
    try:
        saver=SqliteSaver(connection,serde=SessionArtifactSerializer(context.session_dir/'.universal/checkpoint_artifacts'))
        yield build_graph(checkpointer=saver)
    finally:
        connection.close()
