import pytest
from data_collection_workflow.nodes.source_screening import _source_reporting_date_fit, _source_positive_target_verification

STATE={'structured_task':{'disease':'dengue','location':'Brazil','start_date':'2025-01-01','end_date':'2025-12-31'}}

@pytest.mark.parametrize('texts,expected',[
 (['Dengue Brazil surveillance', 'Published 2026-03-01'],'candidate'),
 (['Dengue Brazil surveillance in 2024. Published 2025-02-03.'],'mismatch'),
 (['Dengue Brazil surveillance in 2025. Published 2026-02-03.'],'match'),
 (['Dengue surveillance week 21, 2024; cases from 2025-01-01 to 2025-01-31.'],'match'),
 (['Dengue surveillance 2024-W21 and cases in 2025-01-01 to 2025-01-31.'],'match'),
 (['An estimated 2023-2025 cases were reported in Brazil.'],'candidate'),
 (['In 2024, 2025 dengue cases were reported in Brazil.'],'mismatch'),
 (['Dengue case reports cover 2023-2025.'],'match'),
 (['Dengue cases have been reported since 2024.'],'candidate'),
 (['Dengue cases have been reported since 2026.'],'mismatch'),
])
def test_retrieval_dates_need_their_own_temporal_binding(texts,expected):
    assert _source_reporting_date_fit(texts,STATE)==expected

@pytest.mark.parametrize('publisher',['Independent source','Official health agency'])
def test_publisher_identity_cannot_make_an_open_period_exact(publisher):
    row={'title':'Dengue Brazil surveillance','snippet':'Cases have been reported since 2024.','publisher':publisher}
    result=_source_positive_target_verification(row,STATE)
    assert result['date_fit']=='candidate'
    assert result['target_verification_status']=='unverified_candidate'
