"""Independent preservation of zero-count and clinical heading content."""
import hashlib
import pytest
from data_collection_workflow import document_acquisition as acquisition

@pytest.mark.parametrize('heading', [
    'No cases reported in Canada during 2025',
    'No deaths reported',
    'Zero confirmed cases in 2025',
    'One patient developed fever and acute renal failure after rodent exposure',
    'The patient developed fever and acute renal failure after rodent exposure',
])
def test_substantive_zero_count_or_clinical_heading_is_not_discardable_ui(tmp_path, heading):
    raw = ('<title>Clinical report</title><main><h1>' + heading +
           '</h1><div>Loading...</div></main><script>analytics()</script>').encode()
    doc = acquisition.parse_response(raw, url='https://neutral.example/report', source_id='s',
                                     session_dir=tmp_path, content_type='text/html')
    assert doc['content_readable'] and doc['parse_eligible']
    assert doc['acquisition_status'] == 'readable' and not doc['is_shell']
    assert heading in doc['clean_text']
    assert doc['content_hash'] == hashlib.sha256(raw).hexdigest()
    assert (tmp_path/doc['raw_artifact_path']).read_bytes() == raw
    assert any(doc['clean_text'][s['char_start']:s['char_end']] == heading for s in doc['locator_spans'])

@pytest.mark.parametrize('heading', ['Example Tracker', 'Example Tracker 2025'])
def test_ui_title_year_is_still_a_shell_when_only_placeholder_follows(tmp_path, heading):
    raw = ('<title>' + heading + '</title><h1>' + heading +
           '</h1><div>Loading real-time data...</div><script src="app.js"></script>').encode()
    doc = acquisition.parse_response(raw, url='https://neutral.example/dashboard', source_id='s',
                                     session_dir=tmp_path, content_type='text/html')
    assert doc['acquisition_status'] == 'shell' and not doc['parse_eligible']
    assert doc['content_hash'] == hashlib.sha256(raw).hexdigest()
