"""Exact disease-name equivalences, separate from broad discovery relevance.

This vocabulary identifies names, never patient facts or subtype equivalence.
Unknown names require exact equality. Add aliases only as full disease names;
pathogens, syndromes, clades and task-generated search terms do not belong here.
"""
from __future__ import annotations

# Curated terminology, not task- or case-study-specific source hints.
_NAME_GROUPS = (
    ('mpox', 'monkeypox', 'mpox (monkeypox)'),
    ('pertussis', 'whooping cough'),
    ('measles', 'rubeola'),
    ('covid-19', 'coronavirus disease 2019'),
    ('hantavirus', 'hantavirus disease'),
)


def _key(value):
    return ' '.join(str(value or '').split()).casefold()


_NAMES = {_key(name): frozenset(_key(alias) for alias in group)
          for group in _NAME_GROUPS for name in group}


def disease_names(value):
    key = _key(value)
    return _NAMES.get(key, frozenset([key]) if key else frozenset())


def same_disease_name(left, right):
    return bool(_key(left) and _key(right) in disease_names(left))
