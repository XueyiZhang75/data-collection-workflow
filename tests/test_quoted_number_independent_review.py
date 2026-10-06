"""Independent grouping/proof boundary review, without production edits."""
import re
import pytest
from data_collection_workflow.source_assertions import count_mentions,number_pattern,number_value
from test_acquisition_evidence import assess

@pytest.fixture(autouse=True)
def evidence(monkeypatch):monkeypatch.setenv('PIPELINE_MODE','evidence')

@pytest.mark.parametrize('token,value',[
 ('100,123',100123),('100 123',100123),('100\u00a0123',100123),('100\u202f123',100123),
 ('123.5',123.5),('1,234.5',1234.5),('0',0),('000',0),('1’234.5',1234.5),
])
def test_scalar_preserves_supported_decimal_and_grouping(token,value):
 assert number_value(token)==value

@pytest.mark.parametrize('token',['-12','+12','1e3','NaN','inf','1,23','1’23','1’234’56'])
def test_scalar_does_not_promote_outside_number_token_grammar(token):
 assert number_value(token) is None

@pytest.mark.parametrize('expression',["100''123",'100’’123','100′′123',"100’'123","100' 123"])
def test_malformed_group_cannot_promote_suffix_to_fact(expression):
 text=f'Canada reported {expression} confirmed measles cases during 2025.'
 qualification=assess(text,disease='measles',country='Canada',reporting_period='2025',cases_confirmed=123)
 assert not next(f for f in qualification.field_evidence if f.field=='cases_confirmed').supported
 assert qualification.status!='qualified'

@pytest.mark.parametrize('expression',['100,123','100 123','100\u00a0123','100\u202f123'])
def test_existing_integer_group_proof_unchanged(expression):
 text=f'Canada reported {expression} confirmed measles cases during 2025.'
 assert assess(text,disease='measles',country='Canada',reporting_period='2025',cases_confirmed=100123).status=='qualified'
 assert assess(text,disease='measles',country='Canada',reporting_period='2025',cases_confirmed=123).status!='qualified'

@pytest.mark.parametrize('expression',['100’123%','100’123 percent','100’123 years old','Page 100’123'])
def test_units_and_context_do_not_become_counts(expression):
 text=f'Canada reported {expression} confirmed measles cases during 2025.'
 assert assess(text,disease='measles',country='Canada',reporting_period='2025',cases_confirmed=100123).status!='qualified'

@pytest.mark.parametrize('text',[
 "Canada stated: '100’123 confirmed measles cases during 2025.'",
 'Canada stated: "100’123 confirmed measles cases during 2025."',
])
def test_ordinary_sentence_quotation_does_not_truncate_group(text):
 assert count_mentions(text)[0]['value']==100123
 assert assess(text,disease='measles',country='Canada',reporting_period='2025',cases_confirmed=100123).status=='qualified'


@pytest.mark.parametrize('period,facts',[
 ('During 2025',{'reporting_period':'2025'}),
 ('As of 9 October 2025',{'metric_period_end':'2025-10-09'}),
])
@pytest.mark.parametrize('expression,value',[('59',59),('25 420',25420)])
def test_sentence_date_comma_is_not_numeric_group(period,facts,expression,value):
 text=f'{period}, Canada reported {expression} confirmed measles cases.'
 # Keep the count immediately after the comma, as in real surveillance prose.
 text=f'Canada reported measles: {period}, {expression} confirmed measles cases.'
 assert count_mentions(text)[0]['value']==value
 q=assess(text,disease='measles',country='Canada',cases_confirmed=value,**facts)
 assert next(f for f in q.field_evidence if f.field=='cases_confirmed').supported
 assert q.status=='qualified',q.reasons

@pytest.mark.parametrize('expression',["100'000",'100’000',"100''000"])
def test_rate_denominator_cannot_use_truncated_prefix(expression):
 text=f'Canada reported a measles incidence rate of 12 per {expression} population during 2025.'
 q=assess(text,disease='measles',country='Canada',reporting_period='2025',metric_name='incidence_rate',metric_value=12,metric_unit='per 100')
 assert q.status!='qualified'
 assert not next(f for f in q.field_evidence if f.field=='metric_value').supported


@pytest.mark.parametrize('expression',["2025''123",'2025’’123','2025,59'])
def test_plain_numeric_prefix_is_not_assumed_to_be_a_year(expression):
 text=f'Canada reported {expression} confirmed measles cases during 2025.'
 q=assess(text,disease='measles',country='Canada',reporting_period='2025',cases_confirmed=123 if expression.endswith('123') else 59)
 assert q.status!='qualified'

@pytest.mark.parametrize('quotation',["'",'"'])
def test_date_comma_preserves_an_ordinary_quoted_statement(quotation):
 text=f'Canada reported measles: During 2025, {quotation}59 confirmed measles cases.{quotation}'
 assert count_mentions(text)[0]['value']==59
 q=assess(text,disease='measles',country='Canada',reporting_period='2025',cases_confirmed=59)
 assert q.status=='qualified',q.reasons
