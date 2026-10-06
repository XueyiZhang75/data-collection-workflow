"""Bounded discovery of resources explicitly linked by current-session pages."""
from __future__ import annotations

import hashlib
import re
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit


def canonical_resource_url(value):
    try:
        parsed = urlsplit(str(value or ''))
        if parsed.scheme.lower() not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
            return ''
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or '/', parsed.query, ''))
    except ValueError:
        return ''


def _anchor_sentence(anchor, context):
    """A sentence around this anchor, with other anchors kept out of its claim."""
    from bs4 import NavigableString
    if context.name in {'tr', 'li', 'dd', 'figure'}:
        # A row/list item can bind the article title to its adjacent PDF/data
        # representation. Admission still requires the destination's own
        # product/attachment purpose, so row authors do not inherit it.
        return context.get_text(' ', strip=True)[:700]
    marker = "__collection_anchor__"
    parts = []
    for node in context.descendants:
        if isinstance(node, NavigableString):
            owner = node.find_parent('a')
            if owner is anchor:
                continue
            parts.append(str(node) if owner is None else ' ')
        elif node is anchor:
            parts.append(marker)
    if context is anchor:
        return anchor.get_text(' ', strip=True)
    text = ' '.join(parts)
    for sentence in re.split(r'(?<=[.!?;])\s+|[\r\n]+', text):
        if marker in sentence:
            return re.sub(r'\s+', ' ', sentence.replace(marker, anchor.get_text(' ', strip=True))).strip()[:700]
    return anchor.get_text(' ', strip=True)


def html_resource_links(soup, *, source_url, content_hash):
    """Keep raw provenance and bounded link scope before HTML tags are stripped."""
    result = []
    def resolve_href(raw_href, node, locator, base_url):
        try:
            return canonical_resource_url(urljoin(base_url, str(raw_href or '')))
        except ValueError as exc:
            result.append({'href': str(raw_href or ''), 'raw_href': str(raw_href or ''),
                'resource_link_error': {'reason': 'invalid_resource_url',
                                        'error_type': type(exc).__name__, 'message': str(exc)},
                'source_url': source_url, 'source_content_hash': content_hash,
                'locator': {**locator, 'source_line': getattr(node, 'sourceline', None),
                            'source_position': getattr(node, 'sourcepos', None)}})
            return ''
    base = soup.find('base', href=True)
    base_url = source_url
    if base:
        base_url = resolve_href(base.get('href'), base, {'base_index': 0}, source_url) or source_url
    declarations = {}
    for declaration_index, node in enumerate(soup.find_all(['meta', 'link'])):
        kind = None
        if node.name == 'meta' and str(node.get('name') or '').lower() == 'citation_pdf_url':
            kind, declared_url = 'citation_pdf_url', node.get('content')
        elif (node.name == 'link' and 'alternate' in (node.get('rel') or [])
              and str(node.get('type') or '').lower().split(';')[0].strip() == 'application/pdf'):
            kind, declared_url = 'alternate_pdf', node.get('href')
        if kind and str(declared_url or '').strip():
            destination = resolve_href(declared_url, node, {'metadata_index': declaration_index}, base_url)
            if destination:
                declarations.setdefault(destination, []).append({
                    'kind': kind, 'url': destination, 'tag_index': declaration_index,
                    'source_line': getattr(node, 'sourceline', None),
                    'source_position': getattr(node, 'sourcepos', None)})
    for index, anchor in enumerate(soup.find_all('a', href=True)):
        raw_href = str(anchor.get('href') or '').strip()
        if not raw_href or raw_href.startswith('#'):
            continue
        href = resolve_href(raw_href, anchor, {'anchor_index': index}, base_url)
        if not href:
            continue
        context = anchor.find_parent(['p', 'li', 'tr', 'dd', 'figure'])
        if context is None:
            context = anchor
        ancestors = list(anchor.parents)
        heading_context, level = [], 7
        for previous in anchor.find_all_previous(re.compile(r'^h[1-6]$')):
            boundary = previous.find_parent(['article', 'section'])
            if boundary is not None and not any(boundary is node for node in ancestors):
                continue
            previous_level = int(previous.name[1:])
            if previous_level < level:
                heading_context.insert(0, previous.get_text(' ', strip=True)[:300])
                level = previous_level
            if level == 1:
                break
        result.append({
            'href': href, 'raw_href': raw_href, 'text': anchor.get_text(' ', strip=True),
            'title': str(anchor.get('title') or anchor.get('aria-label') or ''),
            'aria_label': str(anchor.get('aria-label') or ''),
            'rel': list(anchor.get('rel') or []),
            'context': context.get_text(' ', strip=True)[:700],
            'anchor_context': _anchor_sentence(anchor, context),
            'heading': heading_context[-1] if heading_context else '',
            'heading_context': heading_context,
            'page_topic': soup.title.get_text(' ', strip=True)[:300] if soup.title else '',
            'scope_version': 2,
            'representation_declarations': declarations.get(href, []),
            'anchor_present': True,
            'type': str(anchor.get('type') or ''), 'download': anchor.has_attr('download'),
            'navigation': anchor.find_parent(['nav', 'header', 'footer']) is not None,
            'source_url': source_url, 'source_content_hash': content_hash,
            'locator': {'anchor_index': index, 'source_line': getattr(anchor, 'sourceline', None),
                        'source_position': getattr(anchor, 'sourcepos', None)},
        })
    anchored = {link['href'] for link in result}
    for href, declared in declarations.items():
        if href in anchored:
            continue
        first = declared[0]
        result.append({
            'href': href, 'raw_href': href, 'text': 'PDF', 'title': '', 'aria_label': '',
            'rel': [], 'context': '', 'anchor_context': '', 'heading': '', 'heading_context': [],
            'page_topic': soup.title.get_text(' ', strip=True)[:300] if soup.title else '',
            'scope_version': 2, 'representation_declarations': declared, 'anchor_present': False,
            'type': 'application/pdf', 'download': True, 'navigation': False,
            'source_url': source_url, 'source_content_hash': content_hash,
            'locator': {'metadata_index': first['tag_index'], 'source_line': first['source_line'],
                        'source_position': first['source_position']},
        })
    return result


_PRODUCT = re.compile(r'\b(?:data(?:set)?s?|dashboards?|download|table|counts?|surveillance|reports?|archives?|historical|history|series|bulletin|rapport|donnees|donn\u00e9es|telecharger|t\u00e9l\u00e9charger|boletin|boletim|informe|datos|relat\u00f3rio)\b', re.I)
_NAVIGATION = re.compile(r'\b(?:privacy|cookies?|terms of use|careers?|login|sign in|subscribe|donate|accessibility|about us)\b', re.I)
_EDITORIAL_FILE = re.compile(r'\b(?:peer[- ]review|reviewer(?:s)?(?: report| comments?)?|author response|editorial decision)\b', re.I)
_RECOMMENDATION = re.compile(r'\b(?:recommended reading|related articles)\b', re.I)
_REFERENCES = re.compile(r'\b(?:references|bibliography|literature cited)\b', re.I)
_DATA_SECTION = re.compile(r'\b(?:data availability|availability of data|underlying data|source data|data access|disponibilit[ée] des donn[ée]es|disponibilidad de (?:los )?datos)\b', re.I)

_ATTACHMENT = re.compile(r'\b(?:pdf|supplement(?:ary|al)?(?: material| information)?|appendi(?:x|ces)|table\s+\d+)\b', re.I)
_SUPPLEMENT = re.compile(r'\b(?:supplement(?:ary|al)?|appendi(?:x|ces))\b', re.I)
_DEICTIC = re.compile(r'^(?:here|this link|access|view|open|ici|ici les donn\u00e9es)?$', re.I)

_FORMAT = re.compile(r'\.(csv|tsv|json|xlsx?|pdf)(?:$|[?&])|\b(?:format|type)=(csv|tsv|json|xlsx?|pdf)\b', re.I)

_UTILITY_ACTION = re.compile(
    r'\b(?:(?:give|send|submit|provide)\s+(?:us\s+)?feedback|tell us what you think|'
    r'(?:take|complete|answer)\s+(?:our |the |a )?survey|share (?:this|the|on|via)|'
    r'(?:find|search|look up)\s+(?:this |the |a )?(?:reference|citation)|google scholar)\b', re.I)
_UTILITY_PATH = re.compile(
    r'(?:^|/)(?:feedback|surveys?|share|sharing|login|sign[-_]?in|privacy|cookies?|'
    r'subscribe|contact|scholar_lookup|citation[-_]?lookup)(?:/|$)', re.I)


def _utility_link(url, link, kind):
    # Nearby report prose describes the parent content, not the action of a
    # feedback/share/search widget. Check the anchor itself and endpoint path.
    labels = [str(link.get(key) or '').strip() for key in ('text', 'title', 'aria_label')]
    label = ' '.join(labels)
    french_share = any(re.match(r'partager(?:$|\s+(?:sur|via|ce|ces|cet|cette)\b)', value, re.I)
                       for value in labels)
    # Match the complete action label of a publishing widget, not nearby
    # prose or the identity of its destination. Unknown access labels remain
    # candidates under the existing scoped-data relationship.
    publication_action = any(re.fullmatch(
        r'publish\s+(?:this|the)\s+(?:post|article|page)\s+(?:to|on|via)\s+[^.!?;\r\n]+',
        value, re.I) for value in labels)
    contact_action = re.match(r'contact(?:$|\s+(?:us|support|our|the)\b)', label.strip(), re.I)
    if _NAVIGATION.search(label) or _UTILITY_ACTION.search(label) or contact_action or french_share or publication_action:
        return True
    path = unquote(urlsplit(url).path)
    if not kind and _UTILITY_PATH.search(path):
        # A survey can publish observations; its data/results landing page is
        # a product, while the participation form remains a utility action.
        survey_product = (re.search(r'(?:^|/)surveys?/(?:data|results|reports?)(?:/|$)', path, re.I)
                          and _PRODUCT.search(label))
        # Repository access tokens can use a share path for an actual dataset.
        # Explicit social-action labels were already rejected above.
        utility_parts = [part.lower() for part in path.split('/') if _UTILITY_PATH.fullmatch(part)]
        shared_product = (utility_parts and all(part in {'share', 'sharing'} for part in utility_parts)
                          and _PRODUCT.search(label))
        return not (survey_product or shared_product)
    return False


def task_resource_candidates(document, entry, state, *, diagnostics=None):
    """Admit scoped products or target articles; page/host identity is not a link claim."""
    from .disease_relevance import build_disease_relevance_context, _find_terms
    from .geography import explicit_country_matches
    task = state.get('structured_task') or state.get('collection_spec') or {}
    disease_context = build_disease_relevance_context(state)
    aliases = [*disease_context['target_disease_identity_terms'], *(task.get('disease_aliases') or [])]
    if not aliases or not document.get('content_readable'):
        return []
    incompatible = disease_context['incompatible_disease_terms']
    location = str(task.get('location') or task.get('geography') or '')
    target_countries = {key for _, key in explicit_country_matches(location)}
    if len(target_countries) != 1:
        target_countries = set()
    def matches(text):
        return bool(_find_terms(text, aliases))
    def contradicts(text):
        if not matches(text) and _find_terms(text, incompatible):
            return True
        countries = {key for _, key in explicit_country_matches(text)}
        return bool(target_countries and countries and not countries & target_countries)
    parent_url = document.get('final_url') or document.get('url') or entry.get('url') or ''
    parent_text = str(document.get('clean_text') or '')
    parent_host = urlsplit(parent_url).hostname
    task_years = {str(task.get(key) or '')[:4] for key in ('start_date', 'end_date')}
    first_year, last_year = str(task.get('start_date') or '')[:4], str(task.get('end_date') or '')[:4]
    if first_year.isdigit() and last_year.isdigit():
        task_years = {str(year) for year in range(int(first_year), int(last_year) + 1)}
    selected = {}
    def invalid_link(index, link, reason, **detail):
        if diagnostics is not None:
            diagnostics.append({'link_index': index, 'status': 'rejected', 'reason': reason,
                'href': str(link.get('href') or '') if isinstance(link, dict) else None, **detail})
    metadata = document.get('metadata')
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        invalid_link(None, None, 'invalid_resource_metadata', field='metadata', value_type=type(metadata).__name__)
        return []
    links = metadata.get('outbound_links')
    if links is None:
        links = []
    if not isinstance(links, (list, tuple)):
        invalid_link(None, None, 'invalid_resource_metadata', field='outbound_links', value_type=type(links).__name__)
        return []
    for index, link in enumerate(links):
        if not isinstance(link, dict):
            invalid_link(index, link, 'invalid_resource_metadata', field='link', value_type=type(link).__name__)
            continue
        if 'resource_link_error' in link:
            error = link['resource_link_error']
            if not isinstance(error, dict) or error.get('reason') != 'invalid_resource_url':
                invalid_link(index, link, 'invalid_resource_metadata', field='resource_link_error',
                             value_type=type(error).__name__)
            else:
                invalid_link(index, link, error['reason'], error_type=error.get('error_type'), message=error.get('message'))
            continue
        malformed = next((key for key, types in (
            ('rel', (list, tuple, str)), ('heading_context', (list, tuple)),
            ('representation_declarations', (list, tuple)))
            if link.get(key) is not None and not isinstance(link[key], types)), None)
        if not malformed:
            malformed = next((key for key in ('rel', 'heading_context')
                if isinstance(link.get(key), (list, tuple))
                and any(not isinstance(value, str) for value in link[key])), None)
        if malformed:
            invalid_link(index, link, 'invalid_resource_metadata', field=malformed, value_type=type(link[malformed]).__name__)
            continue
        try:
            url = canonical_resource_url(urljoin(parent_url, str(link.get('href') or '')))
        except ValueError as exc:
            invalid_link(index, link, 'invalid_resource_url', error_type=type(exc).__name__, message=str(exc))
            continue
        if not url or url == canonical_resource_url(parent_url):
            continue
        # Query strings often contain the parent title (share/citation widgets).
        # They cannot describe the destination's evidence purpose or task scope.
        explicit_labels = [str(link.get(key) or '').strip()
                           for key in ('text', 'title', 'aria_label')]
        own = ' '.join((*explicit_labels,
                        unquote(urlsplit(url).path).replace('-', ' ').replace('_', ' ')))
        local = str(link.get('anchor_context') if 'anchor_context' in link else link.get('context') or '')
        headings = [str(value) for value in link.get('heading_context') or []]
        if not headings and link.get('heading'):
            headings = [str(link['heading'])]
        section = ' '.join(headings)
        fmt = _FORMAT.search(url)
        kind = next((part for part in fmt.groups() if part), '') if fmt else ''
        mime = str(link.get('type') or '').lower().split(';')[0].strip()
        kind = kind or {'application/pdf':'pdf','text/csv':'csv','text/tab-separated-values':'tsv',
                        'application/json':'json','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':'xlsx'}.get(mime,'')
        if not kind and (link.get('download') or re.search(r'[?&]download=(?:true|1|yes)(?:&|$)',url,re.I)):
            kind = 'download'
        if _utility_link(url, link, kind) or 'author' in (link.get('rel') or []):
            continue
        # Editorial files are not observations. A relevant related article is
        # still a candidate; its own scope is assessed independently below.
        if _EDITORIAL_FILE.search(own + ' ' + str(link.get('heading') or '')):
            continue
        own_match = matches(own)
        if contradicts(own):
            continue
        # Use the nearest topic first. A sibling article/section cannot donate
        # its heading; own-title matches do not depend on page topic at all.
        topic = next((value for value in reversed(headings) if matches(value) or contradicts(value)), '')
        declared_representation = any(
            item.get('url') == url and item.get('kind') in {'citation_pdf_url', 'alternate_pdf'}
            for item in link.get('representation_declarations') or [] if isinstance(item, dict))
        # A named supplement declares a work relationship, unlike a file suffix.
        supplement = not link.get('navigation') and (
            _SUPPLEMENT.search(str(link.get('text') or '')) or
            ((kind or _ATTACHMENT.search(own)) and _SUPPLEMENT.search(str(link.get('heading') or ''))))
        if not topic and (not link.get('scope_version') or declared_representation or supplement):
            topic = str(link.get('page_topic') or '')
        if not topic and not link.get('scope_version'):
            topic = parent_text  # older parsers retained only a page context
        if not own_match and (contradicts(local) or contradicts(topic)):
            continue
        topic_match = matches(topic) and not contradicts(topic)
        local_match = matches(local) and not contradicts(local)
        if (_REFERENCES.search(str(link.get('heading') or ''))
                and not own_match and not local_match and not declared_representation):
            continue
        product = bool(_PRODUCT.search(own))
        attachment = bool(kind or _ATTACHMENT.search(own))
        data_purpose = bool(product or attachment or _DEICTIC.fullmatch(str(link.get('text') or '').strip()))
        # A site policy label cannot itself establish a data statement.
        label = str(link.get('text') or '').strip()
        local_data_statement = bool(local.strip() != label and _DATA_SECTION.search(local))
        data_statement = bool(_DATA_SECTION.search(section) or local_data_statement)
        if link.get('navigation') and link.get('scope_version'):
            data_statement = local_data_statement
        page_representation = declared_representation and topic_match
        data_relationship = bool(data_purpose and (topic_match or local_match)
                                 and data_statement)
        scoped_attachment = bool(attachment and (topic_match or local_match))
        scoped_product = bool(product and (topic_match or local_match))
        if (link.get('navigation') and not data_relationship and not page_representation
                and not (own_match and product) and not (local_match and attachment)):
            continue
        if not (own_match or data_relationship or scoped_attachment or scoped_product):
            continue
        if not own_match and _RECOMMENDATION.search(str(link.get('heading') or '')) and not data_relationship:
            continue
        # An admitted unknown remains a retrieval candidate, never a source fact.
        reason = ('scoped_data_availability_link' if data_relationship else
                  'link_task_match' if own_match else
                  'scoped_task_attachment' if scoped_attachment else 'scoped_task_product')
        years = set(re.findall(r'\b(?:19|20)\d{2}\b', own + ' ' + local))
        in_period = bool(years & task_years)
        # Format chooses the adapter; a bounded relationship gives priority.
        score = 40 * data_relationship + 20 * ((local_match and not own_match) or page_representation) + 12 * product + 8 * own_match + 4 * in_period
        if years and task_years - {''} and not in_period:
            score -= 4
        candidate = {'url': url, 'score': score, 'resource_type': kind or 'html',
                     'link_text': str(link.get('text') or ''), 'same_domain': urlsplit(url).hostname == parent_host,
                     'provenance': dict(link), 'selection_reasons': [
                         'explicit_resource_format' if kind else 'linked_report_or_archive', reason]}
        previous = selected.get(url)
        if previous is None or candidate['score'] > previous['score']:
            selected[url] = candidate
    return sorted(selected.values(), key=lambda row: -row['score'])


def resource_source_entry(candidate, *, parent, depth, state=None):
    """Build a child independently; link provenance grants neither trust nor period."""
    from .source_identity import lookup_source_identity_registry
    url = candidate['url']
    host = urlsplit(url).hostname or ''
    identity = lookup_source_identity_registry(host) or {}
    provenance = candidate.get('provenance') or {}
    # The nearest heading and this anchor's own row/sentence are discovery
    # evidence. Broader parent titles, publisher identity and URLs are not.
    snippets = [str(provenance.get('heading') or '').strip(),
                str(provenance.get('anchor_context') if 'anchor_context' in provenance
                    else provenance.get('context') or '').strip()]
    result = {
        'source_id': 'resource_' + hashlib.sha256(url.encode()).hexdigest()[:16],
        'url': url, 'canonical_url': url, 'domain': host,
        'title': (candidate.get('link_text') or provenance.get('title')
                  or provenance.get('aria_label') or None),
        'snippet': '\n'.join(dict.fromkeys(value for value in snippets if value)),
        'publisher': identity.get('publisher_name') or identity.get('publisher') or host,
        'source_type': identity.get('source_type') or 'unknown',
        'source_identity_unverified': not bool(identity), 'metadata_only_identity': not bool(identity),
        'discovery_method': 'task_resource_link', 'parent_source_id': parent.get('source_id'),
        'parent_canonical_url': (candidate.get('provenance') or {}).get('source_url'),
        'resource_link_depth': depth, 'resource_type': candidate['resource_type'],
        'resource_link_provenance': candidate.get('provenance') or {},
        'resource_link_selection_score': candidate['score'],
        'resource_link_selection_reasons': candidate['selection_reasons'],
        'source_role': 'candidate_data_source', 'source_role_final': 'collection',
        'target_fit_status': 'task_record_collection_candidate',
        'target_verification_status': 'candidate_task_record_source',
        'disease_fit': 'candidate', 'geography_fit': 'candidate', 'date_fit': 'candidate',
        'must_fetch': False, 'coverage_requirement_ids': [],
        'final_screening_decision': 'include_for_content_fetch',
        'status': 'ready_for_content_fetch', 'ready_for_content_fetch': True,
    }

    if state is not None:
        from .nodes.source_screening import assess_source_task_fit
        assessment = assess_source_task_fit(result, state)
        # This link evidence only adds period triage. It neither qualifies an
        # observation nor grants the parent's disease/geography to the child.
        for key in ('date_fit', 'task_fit_evidence_origin', 'task_fit_content_hash',
                    'task_fit_assessment_version'):
            result[key] = assessment[key]
        if assessment['date_fit'] == 'mismatch':
            for key in ('target_verification_status', 'target_verification_reason',
                        'triage_role', 'target_fit_status'):
                result[key] = assessment[key]
    return result
