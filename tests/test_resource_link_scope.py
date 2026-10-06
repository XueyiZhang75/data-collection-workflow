"""Fresh-run failures expressed without case-specific production inputs."""
from bs4 import BeautifulSoup
import pytest
from data_collection_workflow.resource_discovery import html_resource_links, task_resource_candidates


def candidates(html, disease="dengue", place="Brazil"):
    soup = BeautifulSoup(html, "html.parser")
    url = "https://journal.example/article/42"
    doc = {"source_id": "parent", "url": url, "content_readable": True,
           "clean_text": soup.get_text(" ", strip=True),
           "metadata": {"outbound_links": html_resource_links(soup, source_url=url, content_hash="body")}}
    return task_resource_candidates(doc, {}, {"structured_task": {"disease": disease,
        "location": place, "start_date": "2025-01-01", "end_date": "2025-12-31"}})


@pytest.mark.parametrize("disease,place", [("dengue", "Brazil"), ("measles", "Canada"), ("Example fever", "Example Region")])
@pytest.mark.parametrize("utility", [
    '<footer><h2>Agencies</h2><a href="/archives-act-2017.pdf">Public Archives</a></footer>',
    '<div><a href="/app">Download App</a></div>',
    '<nav><a href="/policies/data-availability">Data Availability</a></nav>',
    '<nav><a href="/journal-2024.pdf">Journal Report 2024</a></nav>',
])
def test_site_utilities_cannot_borrow_article_topic(disease, place, utility):
    html = f'<title>{disease} in {place} 2025</title><article><h1>{disease} in {place} 2025</h1></article>' + utility
    assert candidates(html, disease, place) == []


def test_navigation_policy_cannot_self_assert_data_relationship_inside_article():
    html = '<article><h1>Dengue Brazil 2025</h1><nav><a href="/policies/data">Data Availability</a></nav></article>'
    assert candidates(html) == []


def test_references_do_not_make_unrelated_pdf_a_study_attachment():
    html = '<article><h1>Dengue Brazil 2025</h1><h2>References</h2><p><a href="/archives-act.pdf">Public archives legislation</a></p></article>'
    assert candidates(html) == []


@pytest.mark.parametrize("label", ["Dengue Brazil surveillance report", "Dengue Brazil case counts"])
def test_file_format_does_not_grant_extra_priority(label):
    html = '<h1>Publications</h1>' + ''.join(f'<p><a href="/release.{fmt}">{label}</a></p>' for fmt in ('html', 'pdf', 'csv'))
    rows = candidates(html)
    assert len(rows) == 3
    assert len({row['score'] for row in rows}) == 1


@pytest.mark.parametrize("html,expected", [
    ('<article><h1>Dengue Brazil 2025</h1><h2>Data availability</h2><p>The underlying observations are <a href="https://data.example/42" download type="text/csv">here</a>.</p></article>', "https://data.example/42"),
    ('<nav><p>Dengue Brazil surveillance report 2025 <a href="/report.pdf">PDF</a></p></nav>', "https://journal.example/report.pdf"),
    ('<nav><a href="/surveillance">Dengue Brazil report archive</a></nav>', "https://journal.example/surveillance"),
    ('<article><h1>Dengue Brazil 2025</h1><a href="/paper.pdf">PDF</a></article>', "https://journal.example/paper.pdf"),
    ('<article><h1>Dengue Brazil 2025</h1><a href="/table/1">Table 1</a></article>', "https://journal.example/table/1"),
])
def test_explicit_report_data_and_article_relationships_are_preserved(html, expected):
    assert [row['url'] for row in candidates(html)] == [expected]


@pytest.mark.parametrize("declaration", [
    '<meta name="citation_pdf_url" content="https://files.example/12345.pdf">',
    '<link rel="alternate" type="application/pdf" href="https://files.example/12345.pdf">',
])
def test_explicit_page_representation_supports_toolbar_before_article(declaration):
    html = '<head><title>Dengue in Brazil 2025</title>' + declaration + '</head>'
    html += '<nav><a href="https://files.example/12345.pdf">PDF</a></nav><article><h1>Dengue in Brazil 2025</h1></article>'
    assert [row['url'] for row in candidates(html)] == ['https://files.example/12345.pdf']


def test_declared_representation_cannot_authorize_another_pdf_toolbar_link():
    html = '<head><title>Dengue Brazil 2025</title><meta name="citation_pdf_url" content="/actual.pdf"></head>'
    html += '<nav><a href="/journal.pdf">Journal PDF</a></nav>'
    assert [row['url'] for row in candidates(html)] == ['https://journal.example/actual.pdf']


def test_explicit_supplement_keeps_same_work_relation_with_flat_heading_markup():
    html='<title>Dengue Brazil 2025 study</title><h3>Dengue Brazil 2025 study</h3>'
    html+='<h3>Authors contributions</h3><p>Supplementary File <a href="/files/42">Supplementary Table S1</a></p>'
    assert [row['url'] for row in candidates(html)]==['https://journal.example/files/42']


def test_flat_heading_supplement_cannot_borrow_another_article_topic():
    html='<title>Dengue Brazil 2025 study</title><h3>Measles Canada 2025</h3>'
    html+='<p><a href="/files/42">Supplementary Table S1</a></p>'
    assert candidates(html)==[]


def test_flat_heading_supplement_section_keeps_non_editorial_attachment():
    html='<title>Dengue Brazil 2025 study</title><h3>Supplementary information</h3>'
    html+='<section><a href="/files/reporting.pdf">Reporting Summary</a>'
    html+='<a href="/files/review.pdf">Peer Review File</a></section>'
    assert [row['url'] for row in candidates(html)]==['https://journal.example/files/reporting.pdf']


@pytest.mark.parametrize("declaration", [
    '<meta name="citation_pdf_url" content="../files/42.pdf">',
    '<link rel="alternate" type="application/pdf" href="../files/42.pdf">',
])
def test_metadata_only_relative_representation_is_discovered(declaration):
    html='<head><title>Dengue Brazil 2025</title>'+declaration+'</head><article></article>'
    rows=candidates(html)
    assert [row['url'] for row in rows]==['https://journal.example/files/42.pdf']
    assert rows[0]['provenance']['anchor_present'] is False
    assert rows[0]['provenance']['source_content_hash']=='body'
    assert rows[0]['provenance']['locator']['metadata_index'] >= 0


@pytest.mark.parametrize("topic",["Measles Canada 2025","Dengue Peru 2025"])
def test_wrong_scope_metadata_declaration_is_not_a_task_resource(topic):
    html=f'<head><title>{topic}</title><meta name="citation_pdf_url" content="/paper.pdf"></head>'
    assert candidates(html)==[]


def test_duplicate_declarations_and_anchors_discover_one_target():
    html='<head><title>Dengue Brazil 2025</title><meta name="citation_pdf_url" content="/paper.pdf">'
    html+='<link rel="alternate" type="application/pdf" href="/paper.pdf"></head>'
    html+='<nav><a href="/paper.pdf">PDF</a><a href="/paper.pdf#page=1">PDF</a></nav>'
    rows=candidates(html)
    assert [row['url'] for row in rows]==['https://journal.example/paper.pdf']
    assert rows[0]['provenance']['anchor_present'] is True
    assert len(rows[0]['provenance']['representation_declarations'])==2
