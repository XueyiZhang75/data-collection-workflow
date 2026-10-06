"""Independent French modality controls for newly supported count labels."""
import pytest
from test_acquisition_count_semantics_review import _assess, _semantic_support


@pytest.mark.parametrize("semantics,text,value", [
    ("cumulative", "En 2025, dans un scénario hypothétique, la France pourrait signaler un total cumulé de 43 cas confirmés de chikungunya depuis janvier.", 43),
    ("newly_reported", "En 2025, dans un scénario hypothétique, la France pourrait signaler 43 nouveaux cas confirmés de chikungunya.", 43),
    ("subset", "En 2025, dans un scénario hypothétique, la France pourrait signaler 43 cas confirmés de chikungunya, dont 5 cas confirmés chez les enfants.", 5),
])
def test_french_hypothetical_counts_are_not_observed_canonical_semantics(semantics, text, value):
    result = _assess(text, value=value, semantics=semantics, disease="chikungunya", country="France")
    assert _semantic_support(result) == [False, False]
    assert result.status == "candidate"
