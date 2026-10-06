import pytest
from data_collection_workflow.source_assertions import count_mentions
from data_collection_workflow.evidence_qualification import _supports
from data_collection_workflow.nodes.extraction import _chunk_has_strong_record_signal

@pytest.mark.parametrize('prefix',['According to the report, ','The report notes that ','For surveillance, '])
@pytest.mark.parametrize('disease',['measles','dengue'])
def test_introductory_words_do_not_turn_observation_year_into_case_count(monkeypatch,prefix,disease):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    text=prefix+f'in 2025 {disease} cases were monitored in Canada.'
    assert count_mentions(text)==[]
    assert not _supports('cases_unspecified',2025,text,{'disease':disease,'country':'Canada','reporting_period':'2025','cases_unspecified':2025})
    assert not _chunk_has_strong_record_signal({'text':text})

@pytest.mark.parametrize('text,value',[
    ('The epidemic resulted in 2025 measles cases in Canada.',2025),
    ('Canada reported 2025 dengue cases during 2024.',2025),
    ('The report notes that Canada recorded 1999 dengue cases.',1999),
    ('En 2025, le Canada a signale 2100 cas de rougeole.',2100),
])
def test_real_four_digit_counts_are_not_blanket_year_filtered(monkeypatch,text,value):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    assert [m['value'] for m in count_mentions(text)] == [value]
    assert _chunk_has_strong_record_signal({'text':text})
