"""Local contextual number roles do not consume nearby measured counts."""
import pytest
from data_collection_workflow import source_assertions as assertions


@pytest.mark.parametrize('prefix,role', [
    ('Page ', 'page'), ('p. ', 'page'), ('page: ', 'page'),
    ('Line ', 'line'), ('ligne ', 'line'),
    ('Section ', 'section'), ('chapter ', 'section'), ('chapitre ', 'section'),
    ('Reference ', 'reference'), ('Ref. ', 'reference'),
    ('Age ', 'age'), ('aged ', 'age'), ('\u00e2ge : ', 'age'),
])
def test_explicit_adjacent_context_number_is_not_a_measured_count(prefix, role):
    text = prefix + '124\nNew cases of measles in Canada during 2025 are discussed.'
    assert assertions.count_mentions(text) == []
    helper = getattr(assertions, 'contextual_number_role', None)
    assert callable(helper)
    start = len(prefix)
    assert helper(text, start, start + 3) == role


@pytest.mark.parametrize('text', [
    'Canada reported 4156\nnew measles cases in 2025.',
    'France a signal\u00e9 4156\nnouveaux cas de chikungunya en 2025.',
    'Page 12 describes how Canada reported 4156 new measles cases in 2025.',
    'Canada reported 4156 new measles cases on this page during 2025.',
])
def test_actual_count_survives_wrapping_and_other_nonadjacent_role_words(text):
    mentions = assertions.count_mentions(text)
    assert [item['value'] for item in mentions] == [4156]
    helper = getattr(assertions, 'contextual_number_role', None)
    assert callable(helper)
    start = text.index('4156')
    assert helper(text, start, start + 4) is None


def test_bracketed_reference_number_is_not_a_case_quantity():
    helper = getattr(assertions, 'contextual_number_role', None)
    assert callable(helper)
    assert helper('See [124] for new cases.', 5, 8) == 'reference'
