"""Retire generated reading duplicates without touching stage diagnostics."""
from __future__ import annotations

import html
import os
from pathlib import Path


_OBSOLETE_KEYS = {
    'task_result_english', 'task_result_chinese', 'run_report', 'stable_run_report',
    'final_report.md', 'final_report', 'stable_final_report', 'final_report_english',
    'final_report_chinese', 'stable_final_report_english', 'stable_final_report_chinese',
    'interpretive_report', 'stable_interpretive_report', 'interpretive_report_english',
    'interpretive_report_chinese', 'stable_interpretive_report_english', 'stable_interpretive_report_chinese',
}


def without_legacy_reading_links(value):
    if isinstance(value, dict):
        return {key: without_legacy_reading_links(item) for key, item in value.items()
                if key not in _OBSOLETE_KEYS}
    if isinstance(value, list):
        return [without_legacy_reading_links(item) for item in value]
    return value


def retire_legacy_reading_files(session_dir):
    """Remove only known generated report names in this output's owned folders."""
    root = Path(session_dir).resolve()
    for folder in (root, root / 'collection'):
        if not folder.is_dir() or folder.is_symlink():
            continue
        for path in folder.iterdir():
            if not path.is_file() or path.is_symlink():
                continue
            name = path.name
            obsolete = (name in {'task_result.md', 'final_report.md', 'final_report_chinese.md'} or
                        (name.endswith('.md') and name.startswith(('workflow_run_report', 'workflow_interpretive_report'))))
            if obsolete and path.resolve().is_relative_to(root):
                path.unlink()


def report_href(target, report_path):
    """A URL relative to the linking file, with a cross-drive Windows fallback."""
    target, report_path = Path(target), Path(report_path)
    from urllib.parse import quote
    try:
        relative = Path(os.path.relpath(report_path, target.parent)).as_posix()
        url = quote(relative, safe='/.')
    except ValueError:
        # Windows cannot express a relative path between different drives.
        url = report_path.resolve().as_uri()
    return url


def write_report_shortcut(target, report_path):
    """Write a redirect so a latest alias cannot break report data links."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    url = html.escape(report_href(target, report_path), quote=True)
    target.write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
                      f'<meta http-equiv="refresh" content="0;url={url}">'
                      f'<title>Session report</title><p><a href="{url}">Open the session report</a></p></html>', encoding='utf-8')
    return str(target)
