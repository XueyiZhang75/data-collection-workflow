"""Observed report versions retain their own scope, provenance, and fetch limits."""
import pytest

from data_collection_workflow.document_acquisition import parse_response
from data_collection_workflow.resource_discovery import task_resource_candidates
from test_evidence_resource_discovery import _node, _source


BASE = "https://example.invalid/items/current"
HEADING = "<h1>Example fever epidemiological summary, Canada 2025</h1>"
STATE = {"structured_task": {"disease": "Example fever", "location": "Canada",
                             "start_date": "2025-01-01", "end_date": "2025-12-31"}}


def _parse(tmp_path, html, url=BASE):
    return parse_response(html.encode(), url=url, source_id="current",
                          session_dir=tmp_path, content_type="text/html")


def _candidates(tmp_path, html, url=BASE):
    document = _parse(tmp_path, html, url)
    return task_resource_candidates(document, _source(url), STATE)


@pytest.mark.parametrize("relation", ["prev", "next", "PREV", "NEXT"])
def test_metadata_report_version_retains_actual_title_and_relation(tmp_path, relation):
    title = "Example fever epidemiological summary, February 27, 2025"
    html = f'<title>Example fever surveillance reports Canada</title><link rel="{relation}" title="{title}" href="/items/prior">'
    document = _parse(tmp_path, html)
    links = document["metadata"]["outbound_links"]
    assert len(links) == 1
    assert links[0]["title"] == title
    assert links[0]["rel"] == [relation]
    rows = task_resource_candidates(document, _source(BASE), STATE)
    assert len(rows) == 1 and rows[0]["is_report_version"] is True
    assert not rows[0].get("is_pagination")
    assert rows[0]["provenance"]["source_content_hash"] == document["content_hash"]
    assert rows[0]["provenance"]["locator"]["metadata_index"] == 0


@pytest.mark.parametrize("relation", ["prev", "next"])
def test_unscoped_metadata_relation_is_retained_but_not_admitted(tmp_path, relation):
    document = _parse(tmp_path, HEADING + f'<link rel="{relation}" href="/items/other">')
    assert len(document["metadata"]["outbound_links"]) == 1
    assert task_resource_candidates(document, _source(BASE), STATE) == []


@pytest.mark.parametrize("link", [
    '<nav><a rel="prev" href="/items/prior">Previous Example fever report</a></nav>',
    '<nav><a rel="next" href="/items/prior">Next Example fever report</a></nav>',
    '<nav><a href="/items/prior">Previous epidemiological summary</a></nav>',
    '<nav><a aria-label="Next epidemiological summary" href="/items/prior">Next</a></nav>',
    '<p>See the previous epidemiological summary of <a href="/items/prior">February 27, 2025</a>.</p>',
    '<p>The next report is available <a href="/items/prior">here</a>.</p>',
])
def test_scoped_version_relationship_can_bind_a_date_only_anchor(tmp_path, link):
    rows = _candidates(tmp_path, HEADING + link)
    assert len(rows) == 1 and rows[0]["is_report_version"] is True
    assert rows[0]["url"] == "https://example.invalid/items/prior"
    assert "explicit_report_series_version" in rows[0]["selection_reasons"]
    assert rows[0]["provenance"]["source_url"] == BASE


@pytest.mark.parametrize("heading,link", [
    ("General ministry procurement reports", '<a rel="prev" href="/items/prior">Previous report</a>'),
    ("Example fever FAQ", '<a rel="prev" href="/items/prior">Previous</a>'),
    ("Example fever report", '<nav><a rel="prev" href="/items/prior">Previous</a></nav>'),
    ("Example fever report", '<nav><a rel="next" href="/items/prior">Next</a></nav>'),
    ("Example fever report", '<a rel="prev" href="/items/prior">Previous influenza report</a>'),
    ("Example fever report", '<a rel="prev" href="/items/prior">Previous report for France</a>'),
    ("Example fever report", '<a rel="prev" href="/feedback">Give feedback about the previous report</a>'),
    ("Example fever report", '<h2>Influenza reports</h2><a rel="prev" href="/items/prior">Previous report</a>'),
    ("Example fever report", '<p>The previous epidemiological summary was updated.</p><p><a href="/items/prior">February 27, 2025</a></p>'),
    ("Example fever report", '<p>Previous report <a href="/report">Example fever report</a>.</p><p><a rel="prev" href="/items/prior">February 27, 2025</a></p>'),
])
def test_unrelated_navigation_cannot_borrow_report_scope(tmp_path, heading, link):
    rows = _candidates(tmp_path, f"<h1>{heading}</h1>{link}")
    assert not any(row["url"].endswith("/items/prior") for row in rows)


@pytest.mark.parametrize("href", [
    "https://other.invalid/items/prior?topic=example&region=Canada&start=20250101",
    "/items/prior?topic=influenza&region=Canada&start=20250101",
    "/items/prior?topic=example&region=France&start=20250101",
    "/items/prior?topic=example&region=Canada&start=20240101",
    "/items/prior",
])
def test_version_relation_cannot_change_origin_or_nonpage_filters(tmp_path, href):
    url = BASE + "?topic=example&region=Canada&start=20250101"
    html = HEADING + f'<a rel="prev" href="{href}">Previous Example fever report</a>'
    assert _candidates(tmp_path, html, url) == []


def test_version_relation_preserves_query_filters_without_guessed_urls(tmp_path):
    url = BASE + "?topic=example&region=Canada&page=4"
    html = HEADING + '<a rel="prev" href="/opaque/key?region=Canada&topic=example&page=3">Previous report</a>'
    rows = _candidates(tmp_path, html, url)
    assert len(rows) == 1 and rows[0]["is_report_version"]
    assert rows[0]["url"] == "https://example.invalid/opaque/key?region=Canada&topic=example&page=3"


def test_next_summary_with_page_parameter_still_has_explicit_version_route(tmp_path):
    rows = _candidates(tmp_path, HEADING + '<a rel="next" href="?page=2">Next epidemiological summary</a>')
    assert len(rows) == 1 and rows[0]["is_report_version"]
    assert rows[0]["url"] == BASE + "?page=2"


@pytest.mark.parametrize("version_first", [False, True])
def test_duplicate_ordinary_anchor_keeps_version_provenance_in_either_order(tmp_path, version_first):
    version = '<a rel="prev" href="/items/prior">Previous Example fever report</a>'
    ordinary = '<a href="/items/prior">Example fever report 2025</a>'
    rows = _candidates(tmp_path, HEADING + (version + ordinary if version_first else ordinary + version))
    assert len(rows) == 1 and rows[0]["is_report_version"]
    assert rows[0]["provenance"]["rel"] == ["prev"]


def test_metadata_version_keeps_title_when_an_ordinary_anchor_has_same_destination(tmp_path):
    html = (HEADING + '<link rel="prev" href="/items/prior" title="Example fever previous report">'
            '<a href="/items/prior">Example fever report 2025</a>')
    rows = _candidates(tmp_path, html)
    assert len(rows) == 1 and rows[0]["is_report_version"]
    assert rows[0]["provenance"]["title"] == "Example fever previous report"
    assert rows[0]["provenance"]["anchor_present"] is False


def test_prior_report_attribution_cannot_admit_an_adjacent_author_anchor(tmp_path):
    html = (HEADING + '<ul><li>Previous epidemiological summary: '
            '<a href="/items/prior">February 27, 2025</a> '
            '<a href="/people/researcher">Author profile</a></li></ul>')
    rows = _candidates(tmp_path, html)
    assert [row["url"] for row in rows] == ["https://example.invalid/items/prior"]
    assert rows[0]["is_report_version"]


def test_incidental_prior_report_reference_does_not_block_ordinary_data_link(tmp_path):
    html = (HEADING + '<p>For comparison with the previous report, '
            '<a href="https://data.invalid/cases.csv">Download case counts CSV</a>.</p>')
    rows = _candidates(tmp_path, html)
    assert len(rows) == 1 and rows[0]["url"] == "https://data.invalid/cases.csv"
    assert not rows[0].get("is_report_version")


def _version_pages():
    pages = {}
    for number in range(4, 0, -1):
        href = f"/items/v{number - 1}" if number > 1 else "/items/v4"
        html = HEADING + f'<p>See the previous epidemiological summary of <a href="{href}">February {number}, 2025</a>.</p>'
        pages[f"https://example.invalid/items/v{number}"] = ("text/html", html.encode())
    return pages


@pytest.mark.parametrize("adaptive", [False, True])
def test_report_chain_exceeds_directory_depth_without_repeated_fetches(monkeypatch, tmp_path, adaptive):
    pages = _version_pages()
    parent = _source(next(iter(pages)))
    parent.update(must_fetch=True, reporting_period_start="2025-01-01", date_fit="match")
    result, visits, ledger = _node(monkeypatch, tmp_path, pages, sources=[parent], adaptive=adaptive)
    assert visits == list(pages)
    children = [row for row in result["source_registry"] if row.get("parent_source_id")]
    assert len(children) == 3
    assert {row["resource_link_depth"] for row in children} == {0}
    assert all("explicit_report_series_version" in row["resource_link_selection_reasons"] for row in children)
    assert all(not row.get("must_fetch") and not row.get("reporting_period_start") for row in children)
    assert all(row["source_identity_unverified"] and row["publisher"] != parent["publisher"] for row in children)
    if adaptive:
        assert ledger["used"]["source_targets"] == ledger["used"]["http_requests"] == 4
    else:
        assert ledger["used"] == {"fetch": 4, "fetch_ordinary": 4}


@pytest.mark.parametrize("limit", [0, 1])
def test_report_versions_respect_explicit_resource_limits(monkeypatch, tmp_path, limit):
    pages = _version_pages()
    result, visits, _ = _node(monkeypatch, tmp_path, pages, sources=[_source(next(iter(pages)))], limit=limit)
    assert visits == list(pages)[:limit + 1]
    assert len(result["source_registry"]) == limit + 1
    assert result["documents"][0]["metadata"]["outbound_links"]


def test_report_versions_stop_at_shared_acquisition_budget(monkeypatch, tmp_path):
    pages = _version_pages()
    _, visits, ledger = _node(monkeypatch, tmp_path, pages, sources=[_source(next(iter(pages)))], budget=2)
    assert visits == list(pages)[:2]
    assert ledger["used"] == {"fetch": 2, "fetch_ordinary": 2}


@pytest.mark.parametrize("exclusion", [
    {"requires_human_review": True},
    {"source_excluded_by_human_review": True},
    {"source_role_final": "excluded", "final_screening_decision": "exclude"},
])
def test_version_route_does_not_remove_independent_exclusions(monkeypatch, tmp_path, exclusion):
    parent = _source(BASE)
    child = _source("https://example.invalid/items/prior", 2)
    child.update(discovery_method="task_resource_link", resource_link_depth=3,
                 blocked_from_fetch=True, blocked_from_fetch_reason="resource_depth_limit",
                 ready_for_content_fetch=False, processing_status="deferred",
                 processing_reason="resource_depth_limit", acquisition_status="not_started", **exclusion)
    pages = {BASE: ("text/html", (HEADING + '<a rel="prev" href="/items/prior">Previous report</a>').encode())}
    result, visits, _ = _node(monkeypatch, tmp_path, pages, sources=[parent, child])
    assert visits == [BASE]
    retained = next(row for row in result["source_registry"] if row["source_id"] == child["source_id"])
    assert retained["blocked_from_fetch"] and retained["resource_link_depth"] == 3
