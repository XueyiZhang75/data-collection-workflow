"""Independent page-scope and target-swap controls for linked resources."""
from bs4 import BeautifulSoup
from data_collection_workflow.resource_discovery import html_resource_links, task_resource_candidates


def _links(html, disease='dengue', location='Brazil'):
    url='https://catalog.example/publications'
    soup=BeautifulSoup(html,'html.parser')
    doc={'source_id':'catalog','url':url,'content_readable':True,
         'clean_text':soup.get_text(' ',strip=True),
         'metadata':{'outbound_links':html_resource_links(soup,source_url=url,content_hash='raw')}}
    state={'structured_task':{'disease':disease,'location':location,
          'start_date':'2025-01-01','end_date':'2025-12-31'}}
    return task_resource_candidates(doc,{},state)


def test_independent_report_cards_cannot_borrow_data_availability_scope():
    rows=_links('<h1>Publications</h1><article><h2>Dengue in Brazil, 2025</h2>'
        '<h3>Data availability</h3><p><a href="/dengue.csv">Download data</a></p></article>'
        '<article><h2>Measles in Canada, 2025</h2><h3>Data availability</h3>'
        '<p><a href="/measles.csv">Download data</a></p></article>')
    urls={r['url'] for r in rows}
    assert 'https://catalog.example/dengue.csv' in urls
    assert 'https://catalog.example/measles.csv' not in urls


def test_switching_task_switches_which_report_card_is_relevant():
    html=('<h1>Publications</h1><article><h2>Dengue in Brazil, 2025</h2>'
        '<p><a href="/dengue.csv">Download data</a></p></article>'
        '<article><h2>Measles in Canada, 2025</h2>'
        '<p><a href="/measles.csv">Download data</a></p></article>')
    urls={r['url'] for r in _links(html,'measles','Canada')}
    assert 'https://catalog.example/measles.csv' in urls
    assert 'https://catalog.example/dengue.csv' not in urls


def test_author_link_does_not_borrow_neighboring_dataset_product():
    rows=_links('<h1>Dengue in Brazil, 2025</h1>'
        '<p>Data analysis by <a href="/people/jane" rel="author">Jane Doe</a>. '
        'Download <a href="/data.csv">dengue case data</a> for Brazil.</p>')
    urls={r['url'] for r in rows}
    assert 'https://catalog.example/data.csv' in urls
    assert 'https://catalog.example/people/jane' not in urls


def test_task_specific_research_title_does_not_require_data_keyword():
    rows=_links('<h1>Latest publications</h1><p>'
        '<a href="/item/123">Dengue transmission in Brazil, 2025</a></p>')
    assert [r['url'] for r in rows] == ['https://catalog.example/item/123']


def test_french_alias_in_article_title_is_sufficient_for_discovery():
    soup=BeautifulSoup('<h1>Publications</h1><p><a href="/item/7">'
        'Transmission de la rougeole au Canada en 2025</a></p>','html.parser')
    url='https://bibliotheque.example/index'
    doc={'url':url,'content_readable':True,'clean_text':soup.get_text(' ',strip=True),
         'metadata':{'outbound_links':html_resource_links(soup,source_url=url,content_hash='raw')}}
    rows=task_resource_candidates(doc,{}, {'structured_task':{'disease':'measles',
        'disease_aliases':['rougeole'],'location':'Canada','start_date':'2025-01-01',
        'end_date':'2025-12-31'}})
    assert [r['url'] for r in rows] == ['https://bibliotheque.example/item/7']


def test_new_disease_name_does_not_require_a_hardcoded_disease_dictionary():
    rows=_links('<h1>Publications</h1><p><a href="/item/88">'
        'Example fever transmission in Example Region, 2025</a></p>',
        disease='Example fever',location='Example Region')
    assert [r['url'] for r in rows] == ['https://catalog.example/item/88']


def test_earlier_weak_duplicate_cannot_demote_a_later_bound_data_link():
    bound=('<article><h2>Dengue in Brazil, 2025</h2><h3>Data availability</h3>'
        '<p><a href="/shared.pdf">Underlying dengue case data</a></p></article>')
    strong=_links('<h1>Publications</h1>'+bound)
    mixed=_links('<h1>Publications</h1><p><a href="/shared.pdf">PDF</a></p>'+bound)
    assert len(strong)==len(mixed)==1
    assert mixed[0]['url']==strong[0]['url']
    assert mixed[0]['score'] >= strong[0]['score']


def test_related_section_keeps_task_matched_paper_below_explicit_data_resource():
    rows=_links('<h1>Dengue in Brazil, 2025</h1>'
        '<section><h2>Data availability</h2><p><a href="/data.csv">Underlying dengue case data</a></p></section>'
        '<section><h2>Related articles</h2><p><a href="/study/123">Dengue transmission in Brazil, 2025</a></p></section>')
    by_url={r['url']:r for r in rows}
    assert 'https://catalog.example/study/123' in by_url
    assert 'https://catalog.example/data.csv' in by_url
    assert by_url['https://catalog.example/data.csv']['score'] > by_url['https://catalog.example/study/123']['score']


def test_peer_review_file_stays_editorial_even_when_url_names_task_disease():
    rows=_links('<h1>Dengue in Brazil, 2025</h1><h2>Peer review reports</h2>'
        '<p><a href="/dengue-peer-review.pdf">PDF</a></p>')
    assert rows == []
