"""A matching report week cannot manufacture disease or geographic target fit."""
import pytest

from data_collection_workflow.nodes.source_screening import source_screening


@pytest.fixture(autouse=True)
def universal_pipeline(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    monkeypatch.setenv('ENABLE_LLM_SOURCE_CRITIC', 'false')
    monkeypatch.setenv('ENABLE_LLM_SOURCE_IDENTITY', 'false')


def screen(title, *, snippet='', disease='mpox', country='United States', **metadata):
    task = {'disease': disease, 'location': country, 'start_date': '2025-01-01',
            'end_date': '2025-12-31', 'collection_mode': 'direct_collection'}
    entry = {'source_id': 'fixture_source', 'canonical_url': 'https://health.example/report',
             'title': title, 'snippet': snippet, 'publisher': 'Public Health Agency',
             'source_type': 'official_public_health_agency', 'source_type_final': 'national_public_health_agency',
             'status': 'registered', 'discovery_method': 'fixture_search_result', 'priority': 1,
             'expected_fields': ['cases_confirmed', 'deaths'], **metadata}
    result = source_screening({'structured_task': task,
        'collection_spec': {**task, 'geography': country}, 'source_registry': [entry], 'collection_trace': []})
    return result['source_registry'][0], result['source_triage_results'][0]


@pytest.mark.parametrize('title,snippet', [
    ('South Sudan IDSR epidemiological bulletin, week 40, 2025', 'Mpox surveillance in South Sudan.'),
    ('Communicable disease threats report, week 21, 2025', 'Global mpox surveillance update.'),
])
def test_source_week_overlap_does_not_verify_missing_target_geography(title, snippet):
    registry, triage = screen(title, snippet=snippet)
    assert registry['target_fit_status'] != 'verified_target'
    assert triage['target_verification_status'] != 'verified_target'
    assert registry['geography_fit'] != 'match'
    assert triage['date_fit'] == 'match'  # Preserve the dimension actually supported.


def test_query_task_and_classifier_metadata_do_not_supply_missing_source_facts():
    registry, triage = screen('Weekly surveillance bulletin, week 40, 2025',
        query_used='mpox United States 2025 surveillance', matched_terms=['mpox', 'United States'],
        source_disease_relevance_status='target_disease_match',
        source_target_disease_terms_found=['mpox'], jurisdiction_hint='United States',
        disease_fit='match', geography_fit='match')
    assert triage['target_verification_status'] != 'verified_target'
    assert registry['disease_fit'] != 'match'
    assert registry['geography_fit'] != 'match'


def test_positive_location_and_date_still_require_source_disease_evidence():
    registry, triage = screen('United States health surveillance, week 40, 2025',
        query_used='mpox United States 2025', source_disease_relevance_status='target_disease_match')
    assert triage['target_verification_status'] != 'verified_target'
    assert registry['geography_fit'] == 'match'
    assert registry['disease_fit'] != 'match'


def test_publisher_identity_does_not_prove_observation_geography():
    registry, triage = screen('Mpox surveillance report, week 40, 2025',
        publisher='United States Centers for Disease Control and Prevention')
    assert triage['target_verification_status'] != 'verified_target'
    assert registry['geography_fit'] != 'match'


@pytest.mark.parametrize('disease,country,label', [
    ('mpox', 'United States', 'Mpox'), ('mpox', 'United States', 'Monkeypox'),
    ('dengue', 'Brazil', 'Dengue'), ('measles', 'Canada', 'Measles'),
])
def test_explicit_disease_place_and_reporting_year_range_verify_without_a_week(disease, country, label):
    registry, triage = screen(f'{label} surveillance — {country}, 2024–2025', disease=disease, country=country,
        published_date='2026-08-20')
    assert triage['target_verification_status'] == 'verified_target'
    assert registry['disease_fit'] == 'match'
    assert registry['geography_fit'] == 'match'
    assert registry['date_fit'] == 'match'


def test_publication_year_cannot_be_attached_to_an_undated_reporting_week():
    registry, triage = screen('Mpox in United States: reporting week 40', published_date='2025-10-15')
    assert triage['target_verification_status'] != 'verified_target'
    assert registry['date_fit'] != 'match'


def test_explicit_reporting_year_outside_task_is_not_rescued_by_publication_date():
    registry, triage = screen('Mpox in United States: annual surveillance 2024', published_date='2025-03-01')
    assert triage['target_verification_status'] != 'verified_target'
    assert registry['date_fit'] == 'mismatch'


def test_existing_explicit_wrong_disease_guard_stays_in_force():
    registry, triage = screen('Dengue in United States, week 40, 2025',
        source_disease_relevance_status='unrelated_disease')
    assert triage['target_verification_status'] == 'unrelated_disease'
    assert registry['disease_fit'] == 'mismatch'

@pytest.mark.parametrize('disease', ['mpox', 'dengue', 'measles'])
@pytest.mark.parametrize('role', ['context_source', 'irrelevant_source'])
def test_critic_and_final_routing_explanations_do_not_inject_another_disease(disease, role):
    from data_collection_workflow.config import load_source_screening_policy
    from data_collection_workflow.models import SourceScreeningPolicy
    from data_collection_workflow.nodes.source_screening import _critic_review_entry, _final_route_entry
    policy = SourceScreeningPolicy(**load_source_screening_policy())
    entry = {'source_id': 'fixture', 'title': f'{disease} source', 'source_role': role,
             'screening_decision': 'include' if role == 'context_source' else 'exclude',
             'screening_confidence': 0.8}
    critic = _critic_review_entry(entry, policy)
    routing = _final_route_entry(entry, critic, policy)
    assert 'hantavirus' not in critic.critic_reason.casefold()
    assert 'hantavirus' not in routing.final_reason.casefold()


def test_publication_year_inside_snippet_is_not_a_reporting_period():
    registry, triage = screen('Mpox in United States: annual surveillance 2024', snippet='Published March 2025.')
    assert triage['target_verification_status'] != 'verified_target'
    assert registry['date_fit'] == 'mismatch'


def test_complete_reporting_date_range_can_overlap_between_its_endpoints():
    registry, triage = screen('Mpox in United States: reporting period 2024-01-01 to 2026-12-31')
    assert triage['target_verification_status'] == 'verified_target'
    assert registry['date_fit'] == 'match'


def test_year_in_disease_name_is_not_reporting_year():
    from data_collection_workflow.nodes.source_screening import _direct_target_verification
    result = _direct_target_verification({'title': 'Coronavirus disease 2019 in Canada'},
        {'structured_task': {'disease': 'COVID-19', 'location': 'Canada', 'start_date': '2019-01-01', 'end_date': '2019-12-31'}})
    assert result['date_fit'] == 'candidate'
    assert result['target_verification_status'] != 'verified_target'


@pytest.mark.parametrize('title,expected', [('Mpox in US, 2025', 'match'), ('Mpox information: contact us, 2025', 'candidate')])
def test_us_country_abbreviation_does_not_treat_pronoun_as_geography(title, expected):
    registry, _ = screen(title)
    assert registry['geography_fit'] == expected


@pytest.mark.parametrize("dash", ["\u2013", "\u2014", "\u2212"])
@pytest.mark.parametrize("period", ["2024{dash}2026", "2024-01-01{dash}2026-12-31"])
def test_interior_target_year_overlaps_typographic_date_ranges(dash, period):
    registry, triage = screen("Dengue in Brazil: reporting period " + period.format(dash=dash),
        disease="dengue", country="Brazil")
    assert triage["target_verification_status"] == "verified_target"
    assert registry["date_fit"] == "match"


@pytest.mark.parametrize("title,snippet", [
    ("Mpox in Brazil during 2024. Measles in United States during 2025.", ""),
    ("Mpox in Brazil during 2024; measles in United States during 2025.", ""),
    ("Mpox in Brazil during 2024", "Measles in United States during 2025."),
    ("Measles in United States during 2025. Mpox in Brazil during 2024.", ""),
])
def test_unrelated_source_statements_cannot_combine_into_verified_target(title, snippet):
    registry, triage = screen(title, snippet=snippet)
    assert triage["target_verification_status"] != "verified_target"
    assert registry["triage_role"] != "verified_target_collection"


@pytest.mark.parametrize("country_label", ["United States", "U.S.", "U.S.A."])
def test_one_complete_target_statement_can_verify_with_other_source_statements(country_label):
    registry, triage = screen(
        f"Measles in Brazil during 2024. Mpox in {country_label} during 2025.")
    assert triage["target_verification_status"] == "verified_target"
    assert registry["triage_role"] == "verified_target_collection"
