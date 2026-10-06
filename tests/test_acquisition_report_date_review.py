"""Independent report-date role and precision boundary checks."""
import pytest
from data_collection_workflow.source_assertions import typed_date_support


@pytest.mark.parametrize('text',[
    'On 16 January 2025, symptoms began; Canada reported four cases on 20 January 2025.',
    'On 16 January 2025, symptoms began and Canada reported four cases on 20 January 2025.',
    'Canada reported cases diagnosed on 16 January 2025.',
    'Canada reported deaths that occurred on 16 January 2025.',
    'Published on 16 January 2025, Canada reported four cases in 2024.',
    'Report date: January 2025.',
])
def test_report_date_does_not_borrow_clinical_or_publication_date(text):
    assert typed_date_support('date_reported','2025-01-16',text) is False


@pytest.mark.parametrize('text',[
    'On Thursday 16 January 2025, Canada reported four measles cases.',
    'Canada notified four cases on 16 January 2025.',
    'Date de notification: 16 janvier 2025.',
    'Published on 16 January 2025. Canada reported four cases on 16 January 2025.',
])
def test_local_report_date_remains_supported(text):
    assert typed_date_support('report_date','2025-01-16',text) is True
