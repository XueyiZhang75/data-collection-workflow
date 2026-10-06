"""Synthetic source-role overlays shared by offline routing tests."""
from __future__ import annotations

import json
from pathlib import Path

COLLECTION_SOURCE_ID = "src_synthetic_collection"
CONTEXT_SOURCE_ID = "src_synthetic_context"
VALIDATION_SOURCE_ID = "src_synthetic_validation"


def write_source_role_overlays(directory: Path) -> tuple[Path, Path]:
    source_ids = [COLLECTION_SOURCE_ID, CONTEXT_SOURCE_ID, VALIDATION_SOURCE_ID]
    seeds = {
        "catalog_name": "synthetic_source_roles",
        "seed_sources": [
            {
                "source_id": source_id,
                "seed_source_id": source_id.replace("src_", "seed_", 1),
                "title": "Hantavirus surveillance report" if source_id != CONTEXT_SOURCE_ID else "Hantavirus background and context",
                "url": f"https://example.org/{source_id}",
                "publisher": "Example Public Health Agency",
                "source_type": "official_public_health_agency",
                "priority": 1,
                "source_purpose": "Synthetic fixture for source-role routing.",
                "expected_fields": ["cases", "date", "location", "source_url", "evidence_quote"],
                "match_terms": ["hantavirus", "cases", "surveillance"],
            }
            for source_id in source_ids
        ],
    }
    policy = {
        "policy_name": "synthetic_source_roles",
        "domain_masking_enabled": False,
        "collection_allowed_source_ids": [COLLECTION_SOURCE_ID],
        "context_only_source_ids": [CONTEXT_SOURCE_ID],
        "validation_reserved_source_ids": [VALIDATION_SOURCE_ID],
        "validation_reserved_domains": [],
    }
    seed_path = directory / "source_roles_seeds.json"
    policy_path = directory / "source_roles_policy.json"
    seed_path.write_text(json.dumps(seeds), encoding="utf-8")
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return seed_path, policy_path
