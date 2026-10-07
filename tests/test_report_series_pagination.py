"""Pagination must belong to the fetched task-scoped report series."""
import pytest

from data_collection_workflow.document_acquisition import parse_response
from data_collection_workflow.resource_discovery import task_resource_candidates
from test_evidence_resource_discovery import _node, _source


def _candidates(tmp_path, html, url="https://example.invalid/reports"):
    document = parse_response(html.encode(), url=url, source_id="index",
                              session_dir=tmp_path, content_type="text/html")
    return task_resource_candidates(document, _source(url), {
        "structured_task": {"disease": "Example fever", "location": "Canada",
                            "start_date": "2025-01-01", "end_date": "2025-12-31"}})


@pytest.mark.parametrize("link", [
    '<nav><a rel="next" href="?page=2">Next</a></nav>',
    '<head><link rel="next" href="?page=2"></head>',
    '<head><link rel="NEXT" href="?page=2"></head>',
    '<nav><a aria-label="Next page" href="/reports/page/2">2</a></nav>',
    '<nav><a href="?page=2">Next &raquo;</a></nav>',
])
def test_explicit_next_page_keeps_html_provenance(tmp_path, link):
    rows = _candidates(tmp_path, '<title>Example fever surveillance reports, Canada 2025</title>' + link)
    assert len(rows) == 1
    row = rows[0]
    assert row["is_pagination"] is True
    assert "same_report_series_pagination" in row["selection_reasons"]
    assert row["provenance"]["source_url"] == "https://example.invalid/reports"
    assert row["provenance"]["source_content_hash"]
    assert row["provenance"]["locator"]


@pytest.mark.parametrize("title,href", [
    ("Example fever reports", "https://other.invalid/reports?page=2"),
    ("Example fever reports", "/unrelated?page=2"),
    ("Example fever reports", "?page=2&disease=influenza"),
    ("Example fever reports", "?page=2&region=France"),
    ("Example fever reports", "?sort=date"),
    ("General ministry procurement reports", "?page=2"),
    ("Example fever FAQ", "?page=2"),
])
def test_next_link_cannot_change_series_or_borrow_search_title(tmp_path, title, href):
    rows = _candidates(tmp_path, f'<title>{title}</title><nav><a rel="next" href="{href}">Next</a></nav>')
    assert rows == []


def test_nonpagination_filters_are_preserved(tmp_path):
    rows = _candidates(tmp_path,
        '<h1>Example fever report archive</h1><nav><a rel="next" href="?topic=example&page=3">Next</a></nav>',
        url="https://example.invalid/reports?topic=example&page=2")
    assert len(rows) == 1 and rows[0]["is_pagination"]


def test_numeric_start_date_is_not_a_pagination_offset(tmp_path):
    assert _candidates(tmp_path,
        '<h1>Example fever reports</h1><nav><a rel="next" href="?start=20240101&amp;end=20250331">Next</a></nav>',
        url="https://example.invalid/reports?start=20250101&end=20250331") == []


@pytest.mark.parametrize("next_first", [False, True])
def test_duplicate_report_anchor_retains_validated_pagination(next_first, tmp_path):
    next_link = '<a rel="next" href="?page=2">Next</a>'
    report_link = '<a href="?page=2">Example fever reports page 2</a>'
    html = next_link + report_link if next_first else report_link + next_link
    rows = _candidates(tmp_path, '<h1>Example fever surveillance reports</h1>' + html)
    assert len(rows) == 1 and rows[0]["is_pagination"]
    assert rows[0]["provenance"]["rel"] == ["next"]


def test_pagination_can_reach_later_reports_without_consuming_directory_depth(monkeypatch, tmp_path):
    base = "https://example.invalid/reports"
    pages = {}
    for number in range(1, 5):
        url = base if number == 1 else f"{base}?page={number}"
        href = f"?page={number + 1}" if number < 4 else "?page=2"
        report = b'<a href="/late.csv">Download case counts CSV</a>' if number == 4 else b""
        pages[url] = ("text/html", b"<h1>Example fever surveillance reports Canada 2025</h1>" +
                      f'<nav><a rel="next" href="{href}">Next</a></nav>'.encode() + report)
    pages["https://example.invalid/late.csv"] = ("text/csv", b"year,cases\n2025,12\n")
    result, visits, ledger = _node(monkeypatch, tmp_path, pages, sources=[_source(base)])
    assert visits == list(pages)
    assert len(visits) == len(set(visits))
    assert ledger["used"]["fetch"] == 5
    pagination = [row for row in result["source_registry"]
                  if "same_report_series_pagination" in row.get("resource_link_selection_reasons", [])]
    assert len(pagination) == 3
    assert {row["resource_link_depth"] for row in pagination} == {0}
    assert all(not row.get("must_fetch") and not row.get("reporting_period_start") for row in pagination)
