"""Unicode CLDR lookup and bad-link isolation share the real acquisition consumer."""
import unicodedata
import pytest
from data_collection_workflow.geography import explicit_country_key, explicit_country_matches
from data_collection_workflow.resource_discovery import task_resource_candidates, html_resource_links
from bs4 import BeautifulSoup

@pytest.mark.parametrize('base,variant',[
 ('Türkiye','TÜRKİYE'),('Türkiye','TÜRKıYE'),('Türkiye','TU\u0308RKI\u0307YE'),
 ('Chile','CHİLE'),('India','ındıa'),('Sierra Leone','SİERRA LEONE'),
 ('Réunion','RE\u0301UNION'),('Curaçao','CURAC\u0327AO'),
 ('Åland Islands','A\u030aLAND ISLANDS'),('Côte d’Ivoire','CO\u0302TE D’IVOIRE'),
])
def test_country_unicode_case_and_canonical_forms_retain_original_quote(base,variant):
 text='Observed in '+variant+' during 2025.'
 assert list(explicit_country_matches(text))==[(variant,explicit_country_key(base))]
 assert explicit_country_key(variant)==explicit_country_key(base)

@pytest.mark.parametrize('text',['state of Georgia','province of Georgia','city of Georgia','us','in French Polynesian waters unknownplace'])
def test_no_widened_homonym_or_short_alias_admission(text):
 if text.startswith(('state','province','city')) or text=='us':
  assert list(explicit_country_matches(text))==[]
 else:
  assert not any(name=='French' for name,_ in explicit_country_matches(text))


def _doc(links):
 return {'content_readable':True,'final_url':'https://report.example/index','clean_text':'Measles Canada 2025 report',
 'metadata':{'outbound_links':links}}
STATE={'structured_task':{'disease':'measles','location':'Canada','start_date':'2025-01-01','end_date':'2025-12-31'}}
GOOD={'href':'/cases.csv','text':'Measles Canada 2025 case data','scope_version':2}

@pytest.mark.parametrize('bad',[
 {'href':'https://[broken','text':'Measles Canada report'},
 {'href':'/bad','text':'Measles Canada report','rel':7},
 {'href':'/bad','text':'Measles Canada report','heading_context':7},
])
def test_single_invalid_link_keeps_good_sibling_with_reason(bad):
 issues=[]
 rows=task_resource_candidates(_doc([bad,GOOD]),{},STATE,diagnostics=issues)
 assert [r['url'] for r in rows]==['https://report.example/cases.csv']
 assert len(issues)==1 and issues[0]['link_index']==0
 assert issues[0]['status']=='rejected' and issues[0]['reason'].startswith('invalid_resource_')


def test_country_lookup_failure_is_not_silently_hidden(monkeypatch):
 from data_collection_workflow import geography
 def broken(text):raise RuntimeError('deliberate internal bug')
 monkeypatch.setattr(geography,'explicit_country_matches',broken)
 with pytest.raises(RuntimeError,match='deliberate internal bug'):
  task_resource_candidates(_doc([GOOD]),{},STATE)


def test_parser_preserves_bad_url_diagnostic_and_good_link():
 soup=BeautifulSoup('<h1>Measles Canada 2025</h1><a href="https://[broken">Bad report</a><a href="/cases.csv">Measles Canada case data</a>','html.parser')
 links=html_resource_links(soup,source_url='https://report.example/index',content_hash='hash')
 issues=[]
 rows=task_resource_candidates(_doc(links),{},STATE,diagnostics=issues)
 assert [r['url'] for r in rows]==['https://report.example/cases.csv']
 assert any(link.get('resource_link_error') for link in links)
 assert len(issues)==1 and issues[0]['reason']=='invalid_resource_url'


def test_real_queue_continues_after_unicode_country_and_bad_url(tmp_path,monkeypatch):
 import requests
 from test_acquisition_queue import setup_node
 from data_collection_workflow.nodes.content_processing import content_fetch_and_parse
 runtime,state,visits=setup_node(monkeypatch,tmp_path,targets=3)
 state['source_registry']=state['source_registry'][:1]
 parent=state['source_registry'][0]['url'];child='https://data.example/cases.csv'
 html='<h1>Example fever Canada 2025</h1><a href="https://other.example/world">TÜRKİYE news</a><a href="https://[broken">Invalid link</a><a href="/cases.csv">Example fever Canada case data</a>'
 class Response:
  status_code=200
  def __init__(self,url):self.url=url;self.headers={'content-type':'text/html' if url==parent else 'text/csv'}
  def __enter__(self):return self
  def __exit__(self,*args):pass
  def iter_content(self,size):yield (html.encode() if self.url==parent else b'disease,country,year,cases\nExample fever,Canada,2025,12\n')
 def get(url,**kwargs):visits.append(url);return Response(url)
 monkeypatch.setattr(requests,'get',get)
 with runtime.activate():result=content_fetch_and_parse(state)
 assert visits==[parent,child]
 assert any(row['url']==child for row in result['source_registry'])
 doc=next(d for d in result['documents'] if d['url']==parent)
 assert doc['content_readable']
 assert any(i['reason']=='invalid_resource_url' for i in doc['metadata']['resource_discovery_issues'])
 assert runtime.ledger.snapshot()['used']['source_targets']==2

@pytest.mark.parametrize('text',["İn Peru there were 12 cases", "ın Peru there were 12 cases", "uſ outbreak report", "uS outbreak report"])
def test_unicode_short_aliases_do_not_bypass_original_uppercase_guard(text):
 forbidden={explicit_country_key('India'),explicit_country_key('United States')}
 assert not any(key in forbidden for _,key in explicit_country_matches(text))

@pytest.mark.parametrize('text,base',[("US",'United States'),("UK",'United Kingdom')])
def test_explicit_uppercase_country_aliases_remain_supported(text,base):
 assert list(explicit_country_matches(text))==[(text,explicit_country_key(base))]

@pytest.mark.parametrize('bad',[
 '<meta name="citation_pdf_url" content="https://[broken">',
 '<link rel="alternate" type="application/pdf" href="https://[broken">',
 '<base href="https://[broken">',
])
def test_invalid_declaration_or_base_keeps_valid_anchor_and_diagnostic(bad):
 soup=BeautifulSoup(bad+'<h1>Measles Canada 2025</h1><a href="/cases.csv">Measles Canada case data</a>','html.parser')
 links=html_resource_links(soup,source_url='https://report.example/index',content_hash='hash')
 issues=[]
 rows=task_resource_candidates(_doc(links),{},STATE,diagnostics=issues)
 assert [r['url'] for r in rows]==['https://report.example/cases.csv']
 assert len(issues)==1 and issues[0]['reason']=='invalid_resource_url'


def test_unconfigured_uppercase_iso_code_is_not_invented():
 assert list(explicit_country_matches('İN'))==[]

@pytest.mark.parametrize('bad',[
 {'href':'/bad','text':'Measles Canada','resource_link_error':7},
 {'href':'/bad','text':'Measles Canada','resource_link_error':{}},
 {'href':'/bad','text':'Measles Canada','heading_context':[{'invalid':'Measles Canada'}]},
])
def test_malformed_nested_link_metadata_is_diagnostic_and_sibling_survives(bad):
 issues=[]
 rows=task_resource_candidates(_doc([bad,GOOD]),{},STATE,diagnostics=issues)
 assert [r['url'] for r in rows]==['https://report.example/cases.csv']
 assert len(issues)==1 and issues[0]['reason']=='invalid_resource_metadata'

@pytest.mark.parametrize('metadata',[[],{'outbound_links':7},{'outbound_links':{'href':'/cases.csv'}}])
def test_malformed_document_link_container_is_diagnostic(metadata):
 doc=_doc([]);doc['metadata']=metadata;issues=[]
 assert task_resource_candidates(doc,{},STATE,diagnostics=issues)==[]
 assert len(issues)==1 and issues[0]['reason']=='invalid_resource_metadata'


def test_supported_cldr_country_names_keep_identity_under_case_and_canonical_variants():
 from babel import Locale
 tested=0
 for language in ('en','fr','es','pt','zh_Hans'):
  for code,name in Locale.parse(language).territories.items():
   if len(code)!=2 or not code.isalpha():continue
   baseline=list(explicit_country_matches(name))
   if len(baseline)!=1 or baseline[0][0]!=name:continue
   if name.isascii() and len(name.replace('.',''))<=2:continue
   for form in ('NFC','NFD'):
    for variant in (name.upper(),name.lower(),name.title()):
     variant=unicodedata.normalize(form,variant)
     assert list(explicit_country_matches(variant))==[(variant,baseline[0][1])],(language,name,variant)
     tested+=1
 assert tested>1000


def test_country_matches_preserve_exact_original_decomposed_quote_locations():
 variants=['RE\u0301UNION','TU\u0308RKI\u0307YE','CURAC\u0327AO']
 text='Observed in '+', then '.join(variants)+' during 2025.'
 matches=list(explicit_country_matches(text))
 assert [quote for quote,_ in matches]==variants
 for quote,_ in matches:
  start=text.index(quote)
  assert text[start:start+len(quote)]==quote
