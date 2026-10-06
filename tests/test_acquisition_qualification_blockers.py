"""Diagnostic REDs: evidence identity and source-supported canonical semantics."""
import pytest

from data_collection_workflow.evidence_qualification import assess_record_evidence
from test_acquisition_evidence import evidence


def qualify(text, **fields):
    row, doc, chunk = evidence(text, **fields)
    return assess_record_evidence(row, contract={},
        evidence_index={'documents': [doc], 'evidence_chunks': {'span': chunk}})


def test_evidence_span_does_not_require_a_patient_identity_for_supported_aggregate():
    result = qualify('Canada reported 43 confirmed measles cases during 2025.',
        disease='measles', country='Canada', reporting_period='2025', cases_confirmed=43,
        case_span_id='source_evidence_span')
    assert all(field.supported for field in result.field_evidence), result.reasons
    assert result.product_kind == 'aggregate'
    assert result.status == 'qualified', result.reasons


@pytest.mark.parametrize('text,semantics,extra', [
    ('During 2025, Quebec in Canada reported 43 confirmed measles cases out of 187 national cases.', 'subset', {'subnational_location': 'Quebec'}),
    ('In 2025, Canada reported a total of 43 confirmed measles cases since January.', 'cumulative', {}),
    ('Canada reported 43 confirmed measles cases during 2025, all newly reported.', 'newly_reported', {}),
])
def test_canonical_semantics_has_source_meaning_without_literal_enum_token(text, semantics, extra):
    result = qualify(text, disease='measles', country='Canada', reporting_period='2025',
        cases_confirmed=43, count_semantics=semantics, statistical_count_type=semantics, **extra)
    assert all(field.supported for field in result.field_evidence
               if field.field not in {'count_semantics', 'statistical_count_type'}), result.reasons
    assert result.status == 'qualified', result.reasons


@pytest.mark.parametrize('semantics', ['cumulative', 'subset'])
def test_canonical_semantics_is_not_accepted_without_matching_source_meaning(semantics):
    result = qualify('Canada reported 43 confirmed measles cases during 2025.',
        disease='measles', country='Canada', reporting_period='2025', cases_confirmed=43,
        count_semantics=semantics, statistical_count_type=semantics)
    assert result.status == 'candidate'
    assert any(not field.supported for field in result.field_evidence
               if field.field in {'count_semantics', 'statistical_count_type'})


@pytest.mark.parametrize('text,semantics', [
    ('Canada reported 43 confirmed measles cases during 2025. France reported a cumulative total of 100 confirmed measles cases since January 2025.', 'cumulative'),
    ('Canada reported 43 confirmed measles cases during 2025 and a cumulative total of 100 tests since January.', 'cumulative'),
    ('Canada reported 5 cases out of 43 confirmed measles cases during 2025.', 'subset'),
    ('Canada reported 43 confirmed measles cases and 5 new confirmed measles cases during 2025.', 'newly_reported'),
    ('Canada reported a total of 43 confirmed measles cases during 2025.', 'cumulative'),
    ('Canada reported 43 confirmed measles cases during 2025.', 'future_semantics'),
])
def test_semantic_cue_cannot_move_from_another_count_country_or_role(text, semantics):
    result = qualify(text, disease='measles', country='Canada', reporting_period='2025',
        cases_confirmed=43, count_semantics=semantics, statistical_count_type=semantics)
    assert result.status == 'candidate', result.reasons


def test_explicit_included_count_inherits_parent_scope_but_parent_cannot_borrow_child_place():
    text = 'Canada reported 43 confirmed measles cases during 2025, including 5 confirmed measles cases in Quebec.'
    child = qualify(text, disease='measles', country='Canada', subnational_location='Quebec', reporting_period='2025', cases_confirmed=5, count_semantics='subset')
    parent_as_child = qualify(text, disease='measles', country='Canada', subnational_location='Quebec', reporting_period='2025', cases_confirmed=43)
    assert child.status == 'qualified', child.reasons
    assert parent_as_child.status == 'candidate'


@pytest.mark.parametrize('text,fields', [
    ('Canada reported 43 confirmed measles cases during 2025, including 5 confirmed measles cases in France.', {'country': 'Canada', 'reporting_period': '2025'}),
    ('Canada reported 43 confirmed measles cases during 2025, including 5 confirmed measles cases during 2024.', {'country': 'Canada', 'reporting_period': '2025'}),
    ('Canada reported 43 confirmed measles cases during 2025, including a statement that France reported 5 confirmed measles cases.', {'country': 'France', 'reporting_period': '2025'}),
])
def test_included_count_does_not_override_own_scope_or_cross_new_reporting_predicate(text, fields):
    result = qualify(text, disease='measles', cases_confirmed=5, count_semantics='subset', **fields)
    assert result.status == 'candidate', result.reasons


@pytest.mark.parametrize('location,count', [('Quebec', 11), ('Ontario', 7)])
def test_coordinated_included_members_keep_explicit_subset_relation_with_verified_heading(location, count):
    from data_collection_workflow.evidence_qualification import _supports
    scope = 'Measles in Canada during 2025'
    quote = '25.7%), including Quebec (11 cases) and Ontario (7 cases).'
    row = dict(disease='measles', country='Canada', subnational_location=location,
        reporting_period='2025', cases_unspecified=count, _validated_bound_scope=scope)
    assert _supports('cases_unspecified', count, scope+'\n'+quote, row)
    assert _supports('count_semantics', 'subset', scope+'\n'+quote, row)


def test_included_count_cannot_inherit_unmeasured_ordinary_sentence_scope():
    result = qualify('Canada monitored measles during 2025, including a list of 5 confirmed cases.',
        disease='measles', country='Canada', reporting_period='2025', cases_confirmed=5,
        count_semantics='subset')
    assert result.status == 'candidate'


@pytest.mark.parametrize('tail,expected', [('persons; cumulative.', 'qualified'), ('persons; France reported a cumulative total of 12 hospitalizations.', 'candidate')])
def test_explicit_metric_qualifier_tail_binds_only_its_own_assertion(tail, expected):
    result = qualify('Pertussis in Canada during 2025: 12 hospitalizations; '+tail,
        disease='Pertussis', country='Canada', reporting_period='2025',
        metric_name='hospitalizations', metric_value=12, metric_unit='persons', count_semantics='cumulative')
    assert result.status == expected, result.reasons
