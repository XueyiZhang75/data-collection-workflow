import pytest
import copy
import hashlib
from data_collection_workflow.evidence_products import build_evidence_products


def fixture(fields, *, source='s', text=None, kind='aggregate', **extra):
    text = text or '; '.join(f'{k}: {v}' for k, v in fields.items())
    digest = hashlib.sha256(text.encode()).hexdigest()
    chunk = source + digest[:8]
    evidence = [dict(field=k, value=v, supported=True, document_hash=digest,
                     quote=text, locator=dict(chunk_id=chunk, char_start=0, char_end=len(text)))
                for k, v in fields.items()]
    row = dict(fields, source_id=source, evidence_qualification=dict(status='qualified', product_kind=kind, field_evidence=evidence), **extra)
    index = dict(documents=[dict(source_id=source, content_hash=digest, clean_text=text)],
                 evidence_chunks={chunk: dict(source_id=source, text=text)})
    return row, index


def combine(*indexes):
    return dict(documents=[d for i in indexes for d in i['documents']], evidence_chunks={k:v for i in indexes for k,v in i['evidence_chunks'].items()})


def aggregate(**changes):
    return dict(disease='Disease X', country='Country X', reporting_period='2025', cases_confirmed=10, case_definition='confirmed', population_scope='all residents', unit='persons', count_basis='cumulative', **changes)


def test_count_does_not_create_patients_and_unqualified_fails_closed():
    row, index = fixture(aggregate())
    bad = dict(row, evidence_qualification={'status':'candidate'})
    result = build_evidence_products([row, bad], evidence_index=index)
    assert result['case_entities'] == []
    assert len(result['aggregate_groups']) == 1
    assert len(result['excluded_observations']) == 1


def test_patient_identity_scoped_to_document_and_fields_conflict():
    a, ai = fixture(dict(case_label='Case 1', age=20), kind='case_level', case_span_id='span', case_span_quote='case_label: Case 1; age: 20')
    b, bi = fixture(dict(case_label='Case 1', age=30), source='other', kind='case_level', case_span_id='span', case_span_quote='case_label: Case 1; age: 30')
    assert len(build_evidence_products([a,b], evidence_index=combine(ai,bi))['case_entities']) == 2
    missing = dict(a, case_span_quote=None)
    assert not build_evidence_products([missing], evidence_index=ai)['case_entities']


def test_mutated_and_unresolved_evidence_cannot_enter_products():
    a, index = fixture(aggregate())
    a['cases_confirmed'] = 99
    assert not build_evidence_products([a], evidence_index=index)['aggregate_groups']
    a, index = fixture(aggregate())
    index['documents'][0]['clean_text'] += ' tamper'
    assert not build_evidence_products([a], evidence_index=index)['aggregate_groups']


def test_exact_comparability_no_geographic_rollup_or_definition_mix():
    rows, indexes = [], []
    for changes in ({}, {'subnational_location':'State A'}, {'locality':'County B'}, {'case_definition':'probable'}, {'population_scope':'children'}, {'unit':'reports'}, {'count_basis':'incident'}):
        fields = aggregate(); fields.update(changes)
        row, idx = fixture(fields); rows.append(row); indexes.append(idx)
    result = build_evidence_products(rows, evidence_index=combine(*indexes))
    assert len(result['aggregate_groups']) == 7
    assert result['derivations'] == []


def test_order_and_duplicate_invariance():
    a, ai = fixture(aggregate())
    b, bi = fixture(dict(case_label='Case 1', age=20), kind='case_level', case_span_id='span', case_span_quote='case_label: Case 1; age: 20')
    index = combine(ai,bi)
    assert build_evidence_products([a,b], evidence_index=index) == build_evidence_products([b,a,a,b], evidence_index=index)


def test_revisions_keep_publication_separate_and_downward_values():
    rows, indexes = [], []
    for n, pub in ((10,'2025-02-01'),(8,'2025-02-03')):
        fields=aggregate(); fields.update(cases_confirmed=n, date_reported='2025-01-20', as_of_date=pub)
        row, idx=fixture(fields)
        idx['documents'][0]['publication_date']=pub
        rows.append(row); indexes.append(idx)
    result=build_evidence_products(rows, evidence_index=combine(*indexes))
    series=result['time_series_observations']
    assert {s['value'] for s in series} == {10,8}
    assert {s['event_date'] for s in series} == {'2025-01-20'}
    assert {s['publication_date'] for s in series} == {'2025-02-01','2025-02-03'}
    assert len(result['aggregate_groups']) == 1
    assert result['aggregate_groups'][0]['conflict']


def test_origin_lineage_cycles_and_hosting_are_separate():
    sources=[dict(source_id='a', original_publisher='Agency', url='https://mirror.example/a', upstream_source_mentions=['b']), dict(source_id='b', republisher='News', upstream_source_mentions=['a']), dict(source_id='c', url='https://mirror.example/c')]
    result=build_evidence_products([], evidence_index={}, sources=sources)['origin_lineage']
    nodes={n['source_id']:n for n in result['sources']}
    assert nodes['a']['independence_group']==nodes['b']['independence_group']
    assert nodes['a']['independence_group']!=nodes['c']['independence_group']
    assert nodes['a']['hosting_domain']=='mirror.example'
    assert nodes['a']['original_publisher']=='Agency'


def test_bounds_estimates_and_missing_values_are_never_exact_or_zero():
    rows,indexes=[],[]
    for kind in ('exact','lower_bound','estimate'):
        row,idx=fixture(dict(aggregate(), value_kind=kind)); rows.append(row); indexes.append(idx)
    out=build_evidence_products(rows,evidence_index=combine(*indexes))
    assert len(out['aggregate_groups'])==3
    assert {g['value_kind'] for g in out['aggregate_groups']}=={'exact','lower_bound','estimate'}
    assert not build_evidence_products([],evidence_index={})['time_series_observations']


def test_patient_field_conflicts_and_evidenced_alias_merge():
    text='Case 1 age 20; Case 1 age 30; Case A is the same patient as Case 1, cited at https://other.test/report.'
    a, ai=fixture(dict(case_label='Case 1',age=20), text=text, kind='case_level',case_span_quote=text)
    b, bi=fixture(dict(case_label='Case 1',age=30), text=text, kind='case_level',case_span_quote=text)
    out=build_evidence_products([a,b],evidence_index=combine(ai,bi))
    assert out['case_entities'][0]['conflicting_fields']==['age']
    other='Case A age 20.'
    c,ci=fixture(dict(case_label='Case A',age=20),source='other',text=other,kind='case_level',case_span_quote=other)
    ci['documents'][0]['url']='https://other.test/report'
    links=[dict(document_hash=ci['documents'][0]['content_hash'],identity='Case A')]
    a,ai=fixture(dict(case_label='Case 1',age=20,patient_links=links),text=text,kind='case_level',case_span_quote=text)
    assert len(build_evidence_products([a,c],evidence_index=combine(ai,ci))['case_entities'])==1
    a['evidence_qualification']['field_evidence']=[e for e in a['evidence_qualification']['field_evidence'] if e['field']!='patient_links']
    assert len(build_evidence_products([a,c],evidence_index=combine(ai,ci))['case_entities'])==2


def partition_fixture():
    rows,indexes=[],[]
    for label,n in (('children',3),('adults',7)):
        fields=aggregate(); fields.update(population_scope=label,cases_confirmed=n)
        row,idx=fixture(fields); rows.append(row); indexes.append(idx)
    declaration=dict(partition_id='age-groups',axis='population_scope',members=['children','adults'],target='all residents',metric='cases_confirmed',exhaustive=True,mutually_exclusive=True)
    p,pi=fixture(dict(disjoint_partition=declaration),text='children and adults are mutually exclusive and exhaustive partitions of all residents.',kind='context')
    return rows,combine(*indexes,pi),p


def test_validated_disjoint_sum_has_parents_formula_conditions():
    rows,index,p=partition_fixture()
    out=build_evidence_products(rows,evidence_index=index,partitions=[p])
    assert len(out['derivations'])==1
    total=out['derivations'][0]
    assert total['value']==10
    assert len(total['parent_observation_ids'])==2
    assert total['formula']=='sum(parent values)'
    assert total['conditions']['mutually_exclusive'] is True
    assert out==build_evidence_products(rows[::-1]+rows,evidence_index=index,partitions=[p,p])


def test_unproved_missing_or_overlapping_partitions_never_sum():
    rows,index,p=partition_fixture()
    assert not build_evidence_products(rows[:1],evidence_index=index,partitions=[p])['derivations']
    p['disjoint_partition']['mutually_exclusive']=False
    assert not build_evidence_products(rows,evidence_index=index,partitions=[p])['derivations']
    rows,index,p=partition_fixture()
    rows[0]['case_definition']='suspected'
    assert not build_evidence_products(rows,evidence_index=index,partitions=[p])['derivations']


def test_unqualified_bound_modifier_and_patient_field_from_other_span_fail_close():
    row,idx=fixture(aggregate())
    row['value_kind']='lower_bound'
    assert not build_evidence_products([row],evidence_index=idx)['aggregate_groups']
    text='Case 1 age 20. Case 2 age 50.'
    row,idx=fixture(dict(case_label='Case 1',age=50),text=text,kind='case_level',case_span_quote='Case 1 age 20.')
    out=build_evidence_products([row],evidence_index=idx)
    assert not out['case_entities'] or 'age' not in out['case_entities'][0]['field_candidates']


def test_partition_rejects_geography_unknown_scope_estimates_and_revisions():
    for field,value in (('axis','country'),('members',['children','children'])):
        rows,index,p=partition_fixture(); p['disjoint_partition'][field]=value
        assert not build_evidence_products(rows,evidence_index=index,partitions=[p])['derivations']
    rows,index,p=partition_fixture()
    fields=aggregate(); fields.update(population_scope='children',cases_confirmed=3,value_kind='estimate')
    r,idx=fixture(fields)
    assert not build_evidence_products([r,rows[1]],evidence_index=combine(index,idx),partitions=[p])['derivations']
    fields.pop('value_kind'); fields['as_of_date']='2025-03-01'
    r,idx=fixture(fields)
    assert not build_evidence_products([r,rows[1]],evidence_index=combine(index,idx),partitions=[p])['derivations']


def test_date_candidates_preserve_multiple_events_without_averaging():
    fields=aggregate(); fields.update(date_onset='2025-01-01',date_confirmation='2025-01-09',date_death='2025-01-15')
    row,idx=fixture(fields)
    series=build_evidence_products([row],evidence_index=idx)['time_series_observations']
    assert {s['event_date'] for s in series}=={'2025-01-01','2025-01-09','2025-01-15'}
    assert all(s['publication_date'] is None for s in series)


def test_plain_alias_claim_without_cross_document_citation_does_not_merge():
    text='Case 1 is the same patient as Case A.'
    b,bi=fixture(dict(case_label='Case A'),source='other',kind='case_level',case_span_quote='case_label: Case A')
    links=[dict(document_hash=bi['documents'][0]['content_hash'],identity='Case A')]
    a,ai=fixture(dict(case_label='Case 1',patient_links=links),text=text,kind='case_level',case_span_quote=text)
    assert len(build_evidence_products([a,b],evidence_index=combine(ai,bi))['case_entities'])==2


def test_report_dates_are_observation_periods_when_no_period_is_given():
    rows,indexes=[],[]
    for day in ('2025-01-01','2025-01-02'):
        fields=aggregate(); fields.pop('reporting_period'); fields['date_reported']=day
        row,idx=fixture(fields);rows.append(row);indexes.append(idx)
    assert len(build_evidence_products(rows,evidence_index=combine(*indexes))['aggregate_groups'])==2


def test_implicit_bound_partition_parent_never_produces_exact_sum():
    rows,index,p=partition_fixture()
    fields=aggregate(); fields.update(population_scope='children', cases_confirmed=3)
    bounded,bi=fixture(fields,text='At least 3 cases among children in Country X in 2025.')
    out=build_evidence_products([bounded,rows[1]],evidence_index=combine(index,bi),partitions=[p])
    assert not out['derivations']
    assert next(g for g in out['aggregate_groups'] if g['scope']['population_scope']=='children')['value_kind'] != 'exact'


@pytest.mark.parametrize('modifier',['not ', 'possibly ', 'never '])
def test_negated_or_uncertain_partition_never_derives(modifier):
    rows,index,p=partition_fixture()
    bad,bi=fixture(dict(disjoint_partition=p['disjoint_partition']),text='children and adults are '+modifier+'mutually exclusive and exhaustive partitions of all residents.',kind='context')
    assert not build_evidence_products(rows,evidence_index=combine(index,bi),partitions=[bad])['derivations']


def test_negated_cited_patient_alias_never_merges():
    target='Patient B age 20.'
    b,bi=fixture(dict(case_label='Patient B',age=20),source='other',text=target,kind='case_level',case_span_quote=target)
    bi['documents'][0]['url']='https://other.test/report'
    text='Patient A age 20; Patient A is not the same patient as Patient B, cited at https://other.test/report.'
    links=[dict(document_hash=bi['documents'][0]['content_hash'],identity='Patient B')]
    a,ai=fixture(dict(case_label='Patient A',age=20,patient_links=links),text=text,kind='case_level',case_span_quote=text)
    assert len(build_evidence_products([a,b],evidence_index=combine(ai,bi))['case_entities'])==2
