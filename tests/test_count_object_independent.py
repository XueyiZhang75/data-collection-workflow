"""Independent source-local metric object controls; no runtime/network."""
import pytest
from data_collection_workflow.source_assertions import count_mentions,case_count_label_role

@pytest.mark.parametrize('text,kept',[
 ('Canada reported 12 confirmed measles cases in 2025; 9 cases of adverse events were reported after vaccination.',12),
 ('9 cases of adverse reactions were reported. Brazil reported 12 confirmed dengue cases in 2025.',12),
 ('12 cases of adverse events were reported; Canada reported 12 confirmed measles cases in 2025.',12),
 ('En France, 12 cas confirmés ont été signalés; 9 cas d’effets indésirables ont été analysés.',12),
 ('Cas d’effets indésirables: 9; 12 cas confirmés ont été signalés au Canada.',12),
 ('Cases of adverse events: 9; 12 confirmed cases were reported in Canada.',12),
])
def test_non_disease_object_is_local_to_its_own_number(text,kept):
    cases=[m for m in count_mentions(text) if m['field'].startswith('cases_')]
    assert [m['value'] for m in cases]==[kept]
    assert 'adverse' not in text[cases[0]['char_start']:cases[0]['char_end']].lower()

@pytest.mark.parametrize('label',[
 'Number of cases of adverse events (n)',
 'Nombre de cas d’effets indésirables (nombre)',
 'Cases of adverse reactions (count)',
 'Cas d’effets indésirables (n)',
])
def test_explicit_wrapped_column_object_is_preserved(label):
    assert case_count_label_role(label)=='adverse_event'

@pytest.mark.parametrize('label',[
 'Number of disease cases with adverse events (n)',
 'Disease cases without adverse reactions',
 'Adverse event assessment among confirmed cases',
 'Cases of dengue with adverse events',
 'Cases of measles after vaccination',
])
def test_mentioning_vaccination_or_adverse_events_does_not_change_disease_case_object(label):
    assert case_count_label_role(label) is None
