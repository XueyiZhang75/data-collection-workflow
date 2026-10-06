"""Source grouping glyphs cannot turn a suffix into a different qualified count."""
import pytest
from test_acquisition_evidence import assess
from data_collection_workflow.source_assertions import count_mentions, number_value
from data_collection_workflow.nodes.extraction import _official_best_count_mentions

@pytest.fixture(autouse=True)
def evidence(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')

SEPARATORS=["'",'\u2019','\u02bc','\u02b9','\u2032']
@pytest.mark.parametrize('separator',SEPARATORS)
@pytest.mark.parametrize('disease,country',[('measles','Canada'),('dengue','Brazil')])
def test_complete_grouped_count_has_original_span_and_qualifies(separator,disease,country):
    expression=f'100{separator}123'
    text=f'{country} reported {expression} confirmed {disease} cases during 2025.'
    mention,_,_=_official_best_count_mentions(text)
    assert mention is not None and mention['value']==100123
    assert expression in text[mention['char_start']:mention['char_end']]
    qualification=assess(text,disease=disease,country=country,reporting_period='2025',cases_confirmed=100123)
    assert qualification.status=='qualified',qualification.reasons
    assert number_value(expression)==100123

@pytest.mark.parametrize('separator',SEPARATORS)
@pytest.mark.parametrize('value',[100,123])
def test_group_fragments_never_qualify(separator,value):
    text=f'Canada reported 100{separator}123 confirmed measles cases during 2025.'
    q=assess(text,disease='measles',country='Canada',reporting_period='2025',cases_confirmed=value)
    assert not next(f for f in q.field_evidence if f.field=='cases_confirmed').supported
    assert q.status!='qualified'

@pytest.mark.parametrize('expression,fragment',[("1'23",23),("12’34",34),("1234′567",567),("1'234'56",56)])
def test_malformed_group_does_not_offer_a_tail_count(expression,fragment):
    text=f'Canada reported {expression} confirmed measles cases during 2025.'
    assert all(item['value']!=fragment for item in count_mentions(text))
    q=assess(text,disease='measles',country='Canada',reporting_period='2025',cases_confirmed=fragment)
    assert not next(f for f in q.field_evidence if f.field=='cases_confirmed').supported

@pytest.mark.parametrize('prefix',['more than ','approximately ','at least '])
def test_grouping_does_not_erase_quantity_qualifier(prefix):
    text=f'Canada reported {prefix}100’123 confirmed measles cases during 2025.'
    assert count_mentions(text)[0]['bounded']
    assert assess(text,disease='measles',country='Canada',reporting_period='2025',cases_confirmed=100123).status!='qualified'

@pytest.mark.parametrize('text',[
    "Canada did not report 100’123 confirmed measles cases during 2025.",
    "France reported 100’123 confirmed measles cases during 2025.",
    "Canada reported 100’123 confirmed dengue cases during 2025.",
    "Canada reported 100’123 confirmed measles cases during 2024.",
])
def test_grouping_does_not_override_scope_or_polarity(text):
    assert assess(text,disease='measles',country='Canada',reporting_period='2025',cases_confirmed=100123).status!='qualified'

@pytest.mark.parametrize('separator',["''",'’’','′′',"’'","' ","'"*40," ' "*16])
def test_arbitrary_malformed_separator_runs_never_offer_partial_count(separator):
    text=f'Canada reported 100{separator}123 confirmed measles cases during 2025.'
    assert not count_mentions(text)
    q=assess(text,disease='measles',country='Canada',reporting_period='2025',cases_confirmed=123)
    assert not next(f for f in q.field_evidence if f.field=='cases_confirmed').supported

@pytest.mark.parametrize('separator',SEPARATORS)
def test_rate_uses_the_whole_explicit_grouped_denominator(separator):
    text=f'Canada reported a measles incidence rate of 12.5 per 100{separator}000 population during 2025.'
    q=assess(text,disease='measles',country='Canada',reporting_period='2025',metric_name='incidence_rate',metric_value=12.5,metric_unit='per 100000 population')
    assert q.status=='qualified',q.reasons


@pytest.mark.parametrize("expression", ["100''123", "100\u2019\u2019123"])
def test_date_punctuation_does_not_legalize_bad_group_inside_quote(expression):
    text = f"Canada reported measles: During 2025, '{expression} confirmed measles cases.'"
    assert not any(item["value"] == 123 for item in count_mentions(text))
    result = assess(text, disease="measles", country="Canada", reporting_period="2025", cases_confirmed=123)
    assert result.status != "qualified"
