"""Do not trade false historical exclusions for ungrounded period matches."""
import pytest
from data_collection_workflow.nodes.source_screening import _source_reporting_date_fit

def fit(text):
    return _source_reporting_date_fit([text], {"structured_task": {
        "disease": "measles", "location": "Canada",
        "start_date": "2025-01-01", "end_date": "2025-12-31",
    }})

@pytest.mark.parametrize("text,expected", [
    ("In 2024, the authority declared a measles emergency after reporting 12 cases.", "mismatch"),
    ("In 2025, the authority declared a measles emergency after reporting 12 cases.", "match"),
    ("In 2024, the authority declared 12 measles deaths.", "mismatch"),
    ("On 14 August 2024, the measles outbreak was declared a public health emergency.", "candidate"),
    ("On 14 August 2025, the authority declared a measles emergency.", "candidate"),
    ("The measles outbreak began in August 2023, but cases increased in recent weeks of 2024.", "mismatch"),
    ("The measles outbreak began in August 2024, but cases increased in recent weeks of 2025.", "match"),
    ("The measles outbreak began in August 2026, but cases increased in recent weeks.", "mismatch"),
    ("The measles outbreak began in August 2024 and ended in December 2024, after cases increased in recent weeks.", "mismatch"),
    ("The measles outbreak began in August 2024, but Canada now reports 2024 cases.", "candidate"),
    ("The measles outbreak began in August 2024.", "mismatch"),
    ("The authority declared a measles emergency in 2024. Canada reported 12 measles cases in 2025.", "match"),
    ("The authority declared a measles emergency in 2025. Canada reported 12 measles cases in 2024.", "mismatch"),
    ("Canada reported 12 measles cases in 2024. Latest reports and emergency information.", "mismatch"),
])
def test_period_role_boundaries(text, expected):
    assert fit(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("Le 14 août 2024, l'autorité a déclaré l'épidémie de rougeole une urgence de santé publique.", "candidate"),
    ("En 2025, l'autorité a déclaré une urgence de santé publique liée à la rougeole.", "candidate"),
    ("En 2024, las autoridades declararon el brote de sarampión una emergencia sanitaria.", "candidate"),
    ("En 2024, l'autorité a déclaré une urgence après avoir signalé 12 cas de rougeole.", "mismatch"),
    ("En 2025, las autoridades declararon una emergencia después de registrar 12 casos de sarampión.", "match"),
    ("L'épidémie de rougeole a commencé en août 2024, mais les cas augmentent ces dernières semaines.", "candidate"),
    ("L'épidémie de rougeole a commencé en août 2024, mais les cas augmentent ces dernières semaines de 2024.", "mismatch"),
    ("El brote de sarampión comenzó en 2024, pero los casos aumentaron en las últimas semanas.", "candidate"),
    ("El brote de sarampión comenzó en 2024, pero los casos aumentaron en las últimas semanas de 2025.", "match"),
    ("The measles outbreak started in August 2024, but 12 cases were recently reported.", "candidate"),
    ("The measles outbreak began in August 2024 and ended in December 2024. Latest reports are available.", "mismatch"),
    ("The measles outbreak began in 2024, but the agency now provides advice.", "mismatch"),
    ("In 2025, a measles emergency was declared. Surveillance period: 2024-01-01 to 2024-12-31.", "mismatch"),
    ("In 2024, a measles emergency was declared and 12 deaths were reported.", "mismatch"),
    ("In 2024, a measles emergency was declared and cases were reported.", "mismatch"),
])
def test_multilingual_event_and_observation_periods(text, expected):
    assert fit(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("In 2025, a measles emergency was declared, but cases were reported during 2024.", "mismatch"),
    ("In 2024, a measles emergency was declared, but cases were reported during 2025.", "match"),
    ("In 2024, the authority declared a measles emergency after reporting 12 measles cases.", "mismatch"),
    ("The measles outbreak began on 2025-12-01, but cases increased in recent weeks.", "mismatch"),
])
def test_independent_clauses_and_future_onset_preserve_date_precision(text, expected):
    state = {"structured_task": {"disease": "measles", "location": "Canada",
        "start_date": "2025-01-01", "end_date": "2025-06-30"}}
    assert _source_reporting_date_fit([text], state) == expected


@pytest.mark.parametrize("text,expected", [
    ("The measles emergency was declared in 2024, and the outbreak ended in December 2024.", "mismatch"),
    ("The measles emergency was declared in 2025, and the outbreak ended in December 2024.", "mismatch"),
    ("The measles emergency was declared in 2024, and the outbreak ended in December 2025.", "match"),
    ("Une urgence a été déclarée en 2025, et l'épidémie s'est terminée en décembre 2024.", "mismatch"),
    ("The measles emergency was declared in 2025, and the outbreak began in December 2024.", "mismatch"),
    ("After declaring a measles emergency in 2024, cases were recorded during 2023.", "mismatch"),
])
def test_administrative_date_cannot_erase_a_separate_outbreak_endpoint(text, expected):
    assert fit(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("In 2024, a measles emergency was declared, and 12 cases were reported.", "mismatch"),
    ("In 2024, a measles emergency was declared, and 2025 measles cases were reported.", "mismatch"),
    ("In 2025, a measles emergency was declared, and 12 cases were reported during 2024.", "mismatch"),
    ("In 2024, a measles emergency was declared, and 12 cases were reported during 2025.", "match"),
])
def test_shared_and_separate_observation_dates(text, expected):
    assert fit(text) == expected
