"""Source-response failure classification shared by acquisition and consumers.

Numbers in tables or narratives do not identify an HTTP error page. Only an
HTTP failure, an anchored page title/banner, or an existing failure status does.
"""
from __future__ import annotations

import re


_ERROR = re.compile(
    r'^(?:page not found|the page you requested (?:was|could) not (?:be )?found|'
    r"not found(?=\s*[|:-]|\s*$)|we (?:could not|couldn't|cannot|can't) find (?:that |the |this )?(?:web ?)?page|"
    r'technical difficulties(?=\s*[|:-]|\s*$)|error page|access denied|forbidden|service unavailable)\b', re.I)
_ERROR_CODE = re.compile(
    r'^(?:(?:http|error|http error)\s*)?(?:400|401|403|404|410|429|500|502|503|504)'
    r'(?:\s*[-:|]?\s*(?:not found|forbidden|bad request|unauthorized|access denied|'
    r'too many requests|service unavailable|internal server error|gateway timeout|error)'
    r'(?:\b.*)?)?\s*$', re.I)
_CHALLENGE = re.compile(
    r'^(?:checking (?:your )?browser\b|just a moment(?:\s|[.!-]|$)|'
    r'verify (?:that )?you are (?:a )?human\b|security verification\b|'
    r'please wait while (?:we |your browser )|performing security verification\b|'
    r'(?:re)?captcha(?:\s*[-:|]|\s*$)|robot check\b)', re.I)


def _get(document, key, default=None):
    return document.get(key,default) if isinstance(document,dict) else getattr(document,key,default)


def page_failure_reason(document):
    try:
        status = int(_get(document,'http_status_code') or 0)
        if status >= 400 or 0 < status < 200:
            return 'http_error'
    except (TypeError,ValueError):
        pass
    existing = _get(document,'acquisition_status')
    if existing in {'http_error','error_page','blocked'}:
        return existing
    metadata = _get(document,'metadata') or {}
    title = str(metadata.get('page_title') or _get(document,'title') or '').strip()
    # The DOM's primary content heading remains a page banner even when a long
    # site navigation precedes it. It is not an arbitrary body substring.
    for banner in (title, str(metadata.get('page_heading') or '').strip()):
        if banner and _CHALLENGE.search(banner):
            return 'blocked'
        if banner and (_ERROR.search(banner) or _ERROR_CODE.fullmatch(banner)):
            return 'error_page'
    text = next((str(_get(document,key)) for key in ('clean_text','text','raw_text','excerpt') if _get(document,key)), '')
    # DOM parsers can preserve a doctype token and a benign site title before the
    # actual failure banner. Inspect the beginning, not arbitrary body matches.
    lines = [line.strip() for line in text[:1200].splitlines() if line.strip() and line.strip().lower() not in {'html','<!doctype html>'}][:8]
    for line in lines:
        if len(line) > 300:
            continue
        if _CHALLENGE.search(line):
            return 'blocked'
        if _ERROR.search(line):
            return 'error_page'
        # A table's cell may be exactly 404; it is still a reported number.
        code_only = bool(re.fullmatch(r'\d{3}',line))
        if (not _get(document,'tables') and _ERROR_CODE.fullmatch(line) and
                (not code_only or text.strip() == line)):
            return 'error_page'
    return None


_LOADING_PLACEHOLDER = re.compile(
    r"^(?:loading(?:\s+(?:real[- ]time\s+)?(?:data|results|content|dashboard|charts?|page|application))?"
    r"[.\s\u2026]*|please enable javascript(?:\s+to\s+(?:view|use|load|continue)(?:\s+(?:this|the)\s+(?:page|site|application))?)?[.\s]*)$",
    re.I)


def is_dynamic_shell(document, *, has_script):
    """Recognize an unrendered UI without discarding a short report beside it.

    A loading marker must account for the body, apart from its repeated page
    title/heading. Other prose, numeric headings and tables remain content.
    """
    if not has_script or _get(document, 'tables'):
        return False
    text = str(_get(document, 'clean_text') or '')
    lines = [line.strip() for line in text.splitlines()
             if line.strip() and line.strip().lower() not in {'html', '<!doctype html>'}]
    if not lines:
        return True
    if len(text) > 600 or len(lines) > 8:
        return False
    if not any(_LOADING_PLACEHOLDER.fullmatch(line) for line in lines):
        return False
    metadata = _get(document, 'metadata') or {}
    titles = [str(metadata.get(key) or '').strip().casefold()
              for key in ('page_title', 'page_heading')]
    for line in lines:
        if _LOADING_PLACEHOLDER.fullmatch(line):
            continue
        # Years can label a dashboard; other numbers may be observations even
        # when they occur only in the title. Keep those for the evidence gate.
        from .source_assertions import count_mentions
        if count_mentions(line):
            return False
        # Written zero counts and qualitative patient events are still source
        # content when the entire observation happens to be a heading.
        if re.search(r'\b(?:no|zero)\s+(?:(?:confirmed|suspected|probable|reported)\s+)?'
                     r'(?:cases?|deaths?|hospitali[sz]ations?)\b'
                     r'|\b(?:patients?|case)\b.{0,80}\b(?:developed|hospitali[sz]ed|died|'
                     r'recovered|presented|diagnosed|experienced|was|were|had)\b', line, re.I):
            return False
        numbers = re.findall(r'\d+(?:[.,]\d+)?', line)
        if any(not re.fullmatch(r'(?:19|20)\d{2}', value) for value in numbers):
            return False
        if not any(line.casefold() in title for title in titles if title):
            return False
    return True
