"""Task-grounded query policy for the opt-in universal pipeline."""
from __future__ import annotations

from data_collection_workflow.environment import get_env

import hashlib
import re
import unicodedata
from urllib.parse import urlsplit


def universal_queries_enabled():
    return get_env('PIPELINE_MODE') == 'evidence'


def _query_words(value):
    # Search spelling varies in accent/case and Unicode composition; this is
    # normalization only, not stemming or permission to invent event aliases.
    text = unicodedata.normalize('NFKD', str(value).casefold())
    text = ''.join(char for char in text if not unicodedata.combining(char))
    return set(re.findall(r'[^\W\d_]+', text))


# Retrieval methods, evidence formats and reporting vocabulary are not event
# facts. Keep this finite and disease/location independent; unfamiliar names
# still need the task or a bound fetched span below.
_RETRIEVAL_INTENT_WORDS = _query_words("""
    collect collection data case cases death deaths confirmed probable suspected
    report reports reporting reported situation surveillance incidence prevalence
    mortality morbidity cumulative new total annual monthly weekly daily historical
    archive archived archives dashboard table tables download downloadable export
    csv tsv json pdf xls xlsx xml spreadsheet spreadsheets dataset datasets database
    open outbreak risk assessment public health ministry department government
    scientific literature study studies statistics population hospital
    hospitalization hospitalized age sex by of the and or in on at for from to
    with as latest update updates local national state province provincial county
    region regional year years official site source sources count counts number
    numbers clinical individual patient patients line list time series sequence
    genomic retrieval evidence epidemiology epidemiological epidemiologic
    laboratory laboratories infection infections bulletin bulletins supplement
    supplementary appendix appendices attachment attachments index registry
    jurisdiction subnational locality location country global international
    tracker tracking external multi contact tracing
    cas donnees epidemiologique epidemiologiques rapport rapports hebdomadaire
    hebdomadaires telechargement telecharger tableau tableaux laboratoire
    laboratoires archives historique historiques mensuel quotidien annuel
    donnees ouvertes sante publique
    brote casos informe informes datos epidemiologia epidemiologica
    epidemiologico epidemiologicas epidemiologicos boletin boletines semanal
    semanales descarga descargar archivo archivos tabla tablas laboratorio
    laboratorios infecciones datos abiertos
    dados mortes boletim boletins relatorio relatorios historico historicos
    arquivo arquivos planilha planilhas tabela tabelas baixar transferencia
    infeccoes dados abertos
""")


# Publication names identify a reporting product, not an outbreak or place.
_REPORTING_PUBLICATIONS = ("MMWR", "Morbidity and Mortality Weekly Report")


def _reporting_word_forms(words):
    """Inflect reporting/target-field words, never disease/place/event names."""
    forms = set(words)
    for word in words:
        if word.endswith('ies') and len(word) > 4:
            forms.add(word[:-3] + 'y')
        elif word.endswith('s') and not word.endswith(('ss', 'us', 'is')):
            forms.add(word[:-1])
        elif word.endswith('y') and len(word) > 2 and word[-2] not in 'aeiou':
            forms.add(word[:-1] + 'ies')
        elif word.endswith(('s', 'x', 'ch', 'sh')):
            forms.add(word + 'es')
        else:
            forms.add(word + 's')
    return forms


def _contains_term(text, term):
    return bool(term and re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', text, re.I))


def supported_event_terms(state):
    task = state.get('structured_task') or {}
    request = str(task.get('user_request') or state.get('user_request') or '').casefold()
    terms = []
    documents = state.get('documents') or []
    for item in state.get('supported_event_terms') or []:
        if not isinstance(item, dict):
            continue
        term = str(item.get('term') or '').strip()
        if not term:
            continue
        if _contains_term(request, term):
            terms.append(term)
            continue
        # Real Document records use source_id; legacy records may use
        # document_id. Every supplied ID must bind to the same unique record.
        identities = {key: str(item[key]) for key in ('document_id', 'source_id')
                      if item.get(key)}
        matches = [doc for doc in documents if identities and all(
            str(doc.get(key) or '') == value for key, value in identities.items())]
        if len(matches) != 1:
            continue
        doc = matches[0]
        content_hash = str(doc.get('content_hash') or '')
        if not content_hash or any(
            str(item.get(key) or '') != content_hash
            for key in ('document_hash', 'content_hash') if key in item
        ):
            continue
        # Never fall back to old text or metadata when current clean_text is
        # present. content_hash identifies response bytes; text_hash binds text.
        content = doc.get('clean_text')
        if content is None:
            content = doc.get('text') or doc.get('parsed_text') or doc.get('content') or ''
        content = str(content)
        if doc.get('text_hash') and hashlib.sha256(content.encode('utf-8')).hexdigest() != doc['text_hash']:
            continue
        quote = str(item.get('quote') or '')
        if quote and quote in content and _contains_term(quote, term):
            terms.append(term)
    return list(dict.fromkeys(terms))


def assess_query_task_fit(query, state):
    task = state.get('structured_task') or {}
    disease = str(task.get('disease') or '').strip()
    text = str(query.get('query') or '')
    from .disease_identity import disease_names
    aliases = list(dict.fromkeys([disease, *sorted(disease_names(disease)), *(task.get('disease_aliases') or [])]))
    reasons = []
    if not disease or not any(re.search(r'(?<!\w)' + re.escape(str(t)) + r'(?!\w)', text, re.I) for t in aliases if t):
        reasons.append('missing_task_disease')
    request = str(task.get('user_request') or state.get('user_request') or '').casefold()
    supported = {t.casefold() for t in supported_event_terms(state)}
    for term in query.get('event_terms') or []:
        if not _contains_term(request, str(term)) and str(term).casefold() not in supported:
            reasons.append('unsupported_event_term')
    for term in query.get('disease_terms_used') or []:
        if str(term).casefold() not in {str(t).casefold() for t in aliases}:
            reasons.append('unsupported_disease_term')
    # Unknown content words cannot silently introduce unannotated events.
    task_words = ' '.join(str(v) for v in task.values() if isinstance(v, (str, int))) + ' ' + request
    target_fields = task.get('target_fields') or (state.get('collection_spec') or {}).get('required_fields') or []
    field_words = _query_words(' '.join(str(field) for field in target_fields))
    allowed = _reporting_word_forms(_RETRIEVAL_INTENT_WORDS | field_words) | _query_words(task_words)
    allowed.update(word for term in aliases for word in _query_words(term))
    allowed.update(word for term in supported for word in _query_words(term))
    from .source_identity import load_source_identity_registry
    stripped = re.sub(r'(?:site|filetype):[^\s]+', '', text, flags=re.I)
    # A known publisher/domain names an authority; its individual words must
    # not license an unrelated country or event.
    labels = list(_REPORTING_PUBLICATIONS)
    for entry in load_source_identity_registry():
        labels.extend([entry.get('publisher_name') or entry.get('publisher'),
                       entry.get('domain'), *(entry.get('publisher_aliases') or [])])
    for label in sorted({str(label) for label in labels if label}, key=len, reverse=True):
        stripped = re.sub(r'(?<!\w)' + re.escape(label) + r'(?!\w)', ' ', stripped, flags=re.I)
    unknown = _query_words(stripped) - allowed
    # A vocabulary list can describe retrieval language, but cannot enumerate
    # every ministry, surveillance programme or reporting term. Reject concrete
    # contradictory scope and unbound event names, not unfamiliar common words.
    from .disease_relevance import build_disease_relevance_context
    incompatible = build_disease_relevance_context(state).get('incompatible_disease_terms') or []
    if any(_contains_term(stripped, term) and not _contains_term(task_words, term)
           for term in incompatible):
        reasons.append('unsupported_disease_term')
    from .geography import explicit_country_key, explicit_country_matches
    location = str(task.get('location') or '')
    task_countries = {key for _, key in explicit_country_matches(task_words)}
    task_countries.add(explicit_country_key(location))
    foreign = [name for name, key in explicit_country_matches(stripped) if key not in task_countries]
    if foreign:
        reasons.append('unsupported_geography_term')
    # Event nouns bind unfamiliar names even when the query uses lower case.
    event_context = re.search(r'\b(?:vessel|ship|cruise|aboard|wedding|festival|hotel|farm|prison)\b', stripped, re.I)
    named_unknown = {word.casefold() for word in re.findall(r'\b[A-Z][A-Za-z]+\b', stripped)
                     if word.casefold() in unknown}
    anchored = bool(location and _contains_term(text, location))
    institution_intent = (query.get('provider_channel') == 'official_site_search'
                          and anchored and not event_context)
    unbound = set(unknown) if event_context else named_unknown
    if institution_intent:
        # Permit institutional phrases/programme acronyms, not an arbitrary
        # standalone event name merely because the selected channel is official.
        institutional_words = set()
        for phrase in re.findall(r'\b(?:Ministry|Department|Agency|Institute|University|Secretariat)\s+(?:of\s+)?(?:[A-Z][A-Za-z]+(?:\s+(?:and|of|for))?\s*)+', text):
            institutional_words.update(_query_words(phrase))
        institutional_words.update(word.casefold() for word in re.findall(r'\b[A-Z]{2,8}\b', stripped))
        # CLDR macro-regions beside a maintained institutional label describe
        # an office, not an inferred observation country (e.g. a regional CDC).
        from babel import Locale
        regions = {str(name) for code, name in Locale.parse('en').territories.items() if code.isdigit()}
        if any(_contains_term(text, str(label)) for label in labels if label):
            institutional_words.update(word for region in regions if _contains_term(text, region)
                                       for word in _query_words(region))
        unbound -= institutional_words
    if unbound:
        reasons.append('unbound_query_terms')
    return {'accepted': not reasons, 'reasons': list(dict.fromkeys(reasons)),
            'unbound_terms': sorted(unbound), 'retrieval_terms': sorted(unknown - unbound)}



def _retry_domain(value):
    parsed = urlsplit(str(value or '') if '://' in str(value or '') else '//' + str(value or ''))
    return (parsed.hostname or '').casefold().removeprefix('www.')


def _retry_channel(source_type):
    if 'literature' in source_type or 'academic' in source_type:
        return 'literature_api'
    if 'database' in source_type:
        return 'database_search'
    if 'news' in source_type or source_type == 'secondary_aggregator':
        return 'news_search'
    if 'public_health_agency' in source_type or source_type in {'government_report', 'international_organization_report'}:
        return 'official_site_search'
    return None


def _bound_retry_identity(state, candidate, domain):
    """An unregistered institution needs identity evidence in the current page."""
    value = candidate.get if isinstance(candidate, dict) else lambda key, default=None: getattr(candidate, key, default)
    source_id = value('source_id')
    task = state.get('structured_task') or {}
    aliases = [task.get('disease'), *(task.get('disease_aliases') or [])]
    for assessment in state.get('source_identity_assessments') or []:
        if (not source_id or assessment.get('source_id') != source_id
                or _retry_domain(assessment.get('domain')) != domain
                or assessment.get('post_fetch_identity_assessed') is not True
                or assessment.get('metadata_only_identity') is not False
                or assessment.get('source_identity_unverified') is not False
                or assessment.get('publisher_domain_mismatch')
                or not assessment.get('actual_publisher')):
            continue
        for doc in state.get('documents') or []:
            text = str(doc.get('clean_text') or '')
            if (doc.get('source_id') == source_id and doc.get('content_readable')
                    and _retry_domain(doc.get('final_url') or doc.get('url')) == domain
                    and any(_contains_term(text, str(alias)) for alias in aliases if alias)
                    and (not task.get('location') or _contains_term(text, str(task['location'])))
                    and any(str(quote).strip() and str(quote) in text
                            for quote in assessment.get('page_identity_evidence') or [])):
                return {'source_type': assessment.get('source_type_final')}
    return None


def generic_retry_specs(state, candidates, year_suffix, *, literature=False):
    from .source_identity import authority_domain_hints_for_jurisdictions, lookup_source_identity_registry
    task = state.get('structured_task') or {}
    disease = str(task.get('disease') or '').strip()
    if not disease:
        return []
    location = str(task.get('location') or '').strip()
    event_terms = supported_event_terms(state)
    anchor = ' '.join(event_terms)
    # Search query families are desired sources, not independently established
    # identities. Never use a candidate's inherited source_type as authority.
    by_domain = {}
    for hint in authority_domain_hints_for_jurisdictions([location]):
        domain = _retry_domain(hint.get('domain'))
        if domain and _retry_channel(str(hint.get('source_type') or '')):
            by_domain.setdefault(domain, hint)
    for candidate in candidates:
        value = candidate.get if isinstance(candidate, dict) else lambda key, default=None: getattr(candidate, key, default)
        domain = _retry_domain(value('domain') or value('canonical_url') or value('url'))
        identity = lookup_source_identity_registry(domain) or _bound_retry_identity(state, candidate, domain)
        if domain and identity and _retry_channel(str(identity.get('source_type') or '')):
            by_domain.setdefault(domain, identity)
    terms = ['case report surveillance study', 'data table incidence'] if literature else ['surveillance cases deaths report', 'situation report data download', 'historical reports archive']
    specs = []
    # Round-robin intents retain domain breadth when the caller's cap is small.
    for suffix in terms:
        for domain, identity in by_domain.items():
            source_type = str(identity['source_type'])
            specs.append({'query': f'"{disease}" "{location}" {anchor} {suffix} site:{domain} {year_suffix}'.strip(), 'source_type': source_type, 'provider_channel': _retry_channel(source_type), 'role_hint': 'collection_support', 'official_domain_hint': domain, 'event_terms': event_terms, 'disease_terms_used': [disease]})
    if not specs:
        specs.append({'query': f'"{disease}" "{location}" surveillance cases data {year_suffix}', 'source_type': 'news_and_situation_report', 'provider_channel': 'web_search', 'role_hint': 'collection_support', 'official_domain_hint': None})
    return specs
