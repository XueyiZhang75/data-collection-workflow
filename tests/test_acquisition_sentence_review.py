"""Independent run-on boundaries must not mistake abbreviations for identifiers."""
import pytest
from data_collection_workflow.disease_relevance import (
    _evidence_sentences, assess_record_disease_compatibility, build_disease_relevance_context,
)

@pytest.mark.parametrize('ending', ['WHO', 'PAHO', 'US', 'HANTAVIRUS'])
def test_uppercase_sentence_end_before_count_preserves_wrong_disease_isolation(ending):
    first = f'Hantavirus was monitored by {ending}.'
    second = '12 influenza cases and 4 deaths were confirmed.'
    text = first + second
    assert _evidence_sentences(text) == [first, second]
    result = assess_record_disease_compatibility(
        {'disease': 'hantavirus', 'cases_unspecified': 12, 'deaths': 4, 'evidence_quote': text},
        build_disease_relevance_context({'structured_task': {'disease': 'hantavirus'}}))
    assert result['reject_record'], result
    assert 'local_evidence_disease_mismatch' in result['reason']

@pytest.mark.parametrize('label,identifier', [('lineage','B.1'), ('variant','BA.2.86'), ('strain','WHO.12')])
def test_explicit_identifier_label_preserves_dots_without_prefix_whitelist(label, identifier):
    text = f'Mpox {label} {identifier} was associated with 12 cases in Canada during 2025.'
    assert _evidence_sentences(text) == [text]
