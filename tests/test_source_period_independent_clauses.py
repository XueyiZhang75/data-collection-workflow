"""Independent period-role controls using unrelated task contexts."""
import pytest
from data_collection_workflow.nodes.source_screening import _source_reporting_date_fit

TASK={'structured_task':{'disease':'cholera','location':'Ghana','start_date':'2025-01-01','end_date':'2025-12-31'}}
@pytest.mark.parametrize('text,expected',[
 ('On 1 May 2025, authorities declared a cholera emergency.','candidate'),
 ('The cholera outbreak began in 2024 and cases are now increasing.','candidate'),
 ('The cholera outbreak began in 2024 and 12 cases were reported in 2024.','mismatch'),
 ('The cholera emergency was declared in 2025 after the outbreak ended in December 2024.','mismatch'),
 ('The cholera emergency was declared in 2024 after the outbreak ended in December 2024.','mismatch'),
 ('An emergency was declared in 2025 for a cholera surveillance period in 2024.','mismatch'),
 ('The cholera outbreak ended in December 2024 before an emergency was declared in 2025.','mismatch'),
 ('The cholera emergency was declared in 2025. An outbreak ended in December 2024.','mismatch'),
 ('The cholera emergency was declared in 2024. Twelve cases were documented in 2025.','match'),
 ('The cholera outbreak began in 2026 and 12 cases are now reported.','mismatch'),
 ('Le choléra a été déclaré une urgence en 2025 après une épidémie terminée en décembre 2024.','mismatch'),
 ('En 2025, une urgence a été déclarée. Période de surveillance du choléra: 2024.','mismatch'),
])
def test_administrative_date_cannot_conceal_closed_observations(text,expected):
 assert _source_reporting_date_fit([text],TASK)==expected
