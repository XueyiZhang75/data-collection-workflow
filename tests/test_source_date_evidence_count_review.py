"""Independent review: measured four-digit counts are not observation years."""
import pytest
from data_collection_workflow.nodes.source_screening import _source_positive_target_verification

@pytest.mark.parametrize("disease,place", [("dengue","Brazil"),("measles","Canada")])
@pytest.mark.parametrize("count", [850,1899,1900,1999,2025,2099,2100,3000])
def test_current_undated_assertion_is_stable_across_count_magnitude(disease, place, count):
    state={"structured_task":{"disease":disease,"location":place,"start_date":"2025-01-01","end_date":"2025-12-31"}}
    row={"title":f"{disease} in {place}",
         "snippet":f"In 2005-2006, an outbreak caused 120 cases. The current outbreak has {count} cases."}
    fit=_source_positive_target_verification(row,state)
    assert fit["date_fit"]=="candidate", fit
    assert fit["target_verification_status"]=="unverified_candidate"

@pytest.mark.parametrize("count", [850,2025,3000])
def test_genuine_current_assertion_year_remains_outside(count):
    state={"structured_task":{"disease":"dengue","location":"Brazil","start_date":"2025-01-01","end_date":"2025-12-31"}}
    row={"title":"Dengue in Brazil",
         "snippet":f"In 2005-2006, an outbreak caused 120 cases. The current outbreak in 2024 has {count} cases."}
    assert _source_positive_target_verification(row,state)["date_fit"]=="mismatch"
