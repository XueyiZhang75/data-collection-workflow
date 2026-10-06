"""Tests for centralized Data Collection Workflow runtime configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(autouse=True)
def isolated_model_selection(monkeypatch):
    from data_collection_workflow import runtime_profile

    monkeypatch.setattr(runtime_profile, "load_project_env", lambda: False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)


def test_default_workflow_run_config_drives_env_and_studio_input(tmp_path):
    from data_collection_workflow.workflow_run_config import (
        DEFAULT_WORKFLOW_RUN_CONFIG_PATH,
        load_workflow_run_config,
        validation_records_path_from_config,
        workflow_initial_state_from_config,
        workflow_output_dir_from_config,
        workflow_run_config_with_overrides,
        workflow_run_env_from_config,
    )

    assert DEFAULT_WORKFLOW_RUN_CONFIG_PATH.exists()
    assert DEFAULT_WORKFLOW_RUN_CONFIG_PATH.suffix == ".jsonc"
    config = load_workflow_run_config()

    assert config["workflow"]["graph_name"] == "data_collection_workflow"
    assert config["pipeline_mode"] == "standard"
    assert config["workflow"]["collection_mode"] == "standard"
    assert config["profile_name"] == "data_collection"
    assert config["user_request"] == ""
    assert config["structured_task"] == {}
    assert config["live_web"]["enabled"] is True
    assert config["llm"]["provider"] == "anthropic"
    assert config["llm"]["model"] == ""
    assert config["llm"]["source_planning_enabled"] is True
    assert config["llm"]["source_critic_enabled"] is True
    assert config["llm"]["structured_extraction_enabled"] is True
    assert config["llm"]["source_identity"]["enabled"] is True
    assert config["llm"]["source_identity"]["require_llm"] is True
    assert config["llm"]["source_identity"]["allow_deterministic_fallback"] is False
    assert config["disease_intelligence"]["llm_enabled"] is True
    assert config["disease_intelligence"]["force_llm"] is True
    assert config["disease_intelligence"]["fallback_to_curated"] is True
    assert config["output"]["sessionized"] is True
    assert config["output"]["auto_build_console"] is True
    assert config["validation"]["allow_incompatible_validation_records"] is False

    env = workflow_run_env_from_config(config)
    assert env["COLLECTION_MODE"] == "standard"
    assert env["PIPELINE_MODE"] == "standard"
    assert env["USE_FIXTURE_DOCUMENTS"] == "false"
    assert env["ENABLE_LIVE_FETCH"] == "true"
    assert env["ENABLE_LLM_DISEASE_INTELLIGENCE"] == "true"
    assert env["DISEASE_INTELLIGENCE_FORCE_LLM"] == "true"
    assert env["DISEASE_INTELLIGENCE_FALLBACK_TO_CURATED"] == "true"
    assert env["ENABLE_LLM_SOURCE_PLANNING"] == "true"
    assert env["ENABLE_LLM_SOURCE_CRITIC"] == "true"
    assert env["ENABLE_LLM_SOURCE_IDENTITY"] == "true"
    assert env["LLM_SOURCE_IDENTITY_REQUIRE_LLM"] == "true"
    assert env["LLM_SOURCE_IDENTITY_ALLOW_DETERMINISTIC_FALLBACK"] == "false"
    assert env["ENABLE_LLM_EXTRACTION"] == "true"
    assert env["ALLOW_INCOMPATIBLE_VALIDATION_RECORDS"] == "false"
    assert env["HUMAN_REVIEW_ENABLED"] == "false"
    assert env["FETCH_ALLOW_NEEDS_REVIEW"] == "true"
    assert env["VALIDATION_MODE"] == "live_cross_source"
    assert env["EXTERNAL_FETCH_ENABLED"] == "true"
    assert env["LLM_PROVIDER"] == "anthropic"
    assert env["LLM_MODEL"] == ""
    assert env["LLM_MUST_FETCH_MIN_CHUNKS_PER_SOURCE"] == "6"
    assert env["LLM_MAX_CHUNKS"] == "30"
    assert env["LLM_OFFICIAL_EXTRACTION_MAX_CHUNKS"] == "30"
    assert env["DIRECT_COLLECTION_ENABLE_AUDIT_VALIDATION"] == "false"
    assert env["SOURCE_ID_ALLOWLIST"] == ""
    assert config["source_sets"]["workflow_source_ids"] == []
    assert config["source_sets"]["source_id_allowlist_enabled"] is False
    assert env["SEED_SOURCE_OVERLAY_PATH"] == ""
    assert env["SOURCE_ROLE_POLICY_OVERLAY_PATH"] == ""
    assert validation_records_path_from_config(config) is None

    state = workflow_initial_state_from_config(config, include_empty_fields=False)
    assert state["user_request"] == config["user_request"]
    assert state["structured_task"] == config["structured_task"]
    assert state["structured_task"] == {}
    assert state["human_review_enabled"] is False

    override_request = "Collect dengue data for Florida in 2025."
    overridden = workflow_run_config_with_overrides(
        config,
        user_request=override_request,
    )
    assert overridden["user_request"] == override_request
    # Text-only overrides preserve the empty task; runnable inputs require explicit fields.
    assert overridden["structured_task"] == {}

    output_dir = workflow_output_dir_from_config(config, session_id="test_session")
    assert output_dir == (
        _PROJECT_ROOT
        / "outputs"
        / "sessions"
        / "test_session"
    )

    validation_csv = tmp_path / "validation.csv"
    validation_csv.write_text("record_id,disease,cases_confirmed\nsynthetic_case,dengue,2\n", encoding="utf-8")
    validation_override = dict(config)
    validation_override["validation"] = {
        "allow_incompatible_validation_records": True,
        "mode": "held_out_file",
        "held_out_records_path": str(validation_csv),
    }
    env_override = workflow_run_env_from_config(validation_override)
    assert env_override["ALLOW_INCOMPATIBLE_VALIDATION_RECORDS"] == "true"
    assert env_override["VALIDATION_MODE"] == "held_out_file"
    assert validation_records_path_from_config(validation_override).exists()


def test_direct_collection_config_maps_protected_extraction_budget_to_env():
    from data_collection_workflow.workflow_run_config import workflow_run_env_from_config

    config = {
        "workflow": {
            "collection_mode": "direct_collection",
            "use_fixture_documents": False,
        },
        "llm": {
            "max_chunks": 20,
            "max_tokens": 8192,
            "extraction": {
                "soft_primary_calls": 20,
                "hard_primary_calls": 40,
                "focused_recovery_reserved_calls": 8,
                "max_domain_call_share": 0.25,
                "high_value_source_min_spans": 2,
                "event_source_min_spans": 1,
            },
            "must_fetch_min_chunks_per_source": 4,
            "official_extraction_max_chunks": 12,
            "focused_recovery": {
                "max_calls": 120,
                "max_sources": 80,
            },
        },
        "validation": {
            "mode": "diagnostic_only",
            "direct_collection_enable_audit_validation": False,
        },
    }

    env = workflow_run_env_from_config(config)

    assert env["COLLECTION_MODE"] == "direct_collection"
    assert env["LLM_MAX_CHUNKS"] == "20"
    assert env["LLM_EXTRACTION_SOFT_CHECKPOINT_CALLS"] == "20"
    assert env["LLM_EXTRACTION_SAFETY_MAX_CALLS"] == "40"
    assert env["LLM_SOFT_PRIMARY_CALLS"] == "20"
    assert env["LLM_HARD_PRIMARY_CALLS"] == "40"
    assert env["LLM_FOCUSED_RECOVERY_RESERVED_CALLS"] == "8"
    assert env["LLM_MAX_DOMAIN_CALL_SHARE"] == "0.25"
    assert env["LLM_HIGH_VALUE_SOURCE_MIN_SPANS"] == "2"
    assert env["LLM_EVENT_SOURCE_MIN_SPANS"] == "1"
    assert env["LLM_MAX_TOKENS"] == "8192"
    assert env["LLM_MUST_FETCH_MIN_CHUNKS_PER_SOURCE"] == "4"
    assert env["LLM_OFFICIAL_EXTRACTION_MAX_CHUNKS"] == "12"
    assert env["LLM_FOCUSED_RECOVERY_MAX_CALLS"] == "120"
    assert env["LLM_FOCUSED_RECOVERY_MAX_SOURCES"] == "80"
    assert env["VALIDATION_MODE"] == "diagnostic_only"
    assert env["DIRECT_COLLECTION_ENABLE_AUDIT_VALIDATION"] == "false"


def test_loaded_legacy_scheduler_fields_override_merged_scheduler_defaults(tmp_path):
    from data_collection_workflow.workflow_run_config import (
        load_workflow_run_config,
        workflow_run_env_from_config,
    )

    config_path = tmp_path / "legacy_scheduler.json"
    config_path.write_text(
        """
        {
          "llm": {
            "extraction": {
              "soft_primary_calls": 9,
              "hard_primary_calls": 17
            }
          }
        }
        """,
        encoding="utf-8",
    )

    config = load_workflow_run_config(config_path)
    env = workflow_run_env_from_config(config)

    assert config["llm"]["extraction"]["scheduler"]["soft_checkpoint_calls"] == 9
    assert config["llm"]["extraction"]["scheduler"]["safety_max_calls"] == 17
    assert env["LLM_EXTRACTION_SOFT_CHECKPOINT_CALLS"] == "9"
    assert env["LLM_EXTRACTION_SAFETY_MAX_CALLS"] == "17"
    assert env["LLM_SOFT_PRIMARY_CALLS"] == "9"
    assert env["LLM_HARD_PRIMARY_CALLS"] == "17"


def test_canonical_scheduler_values_drive_legacy_env_aliases():
    from data_collection_workflow.workflow_run_config import workflow_run_env_from_config

    env = workflow_run_env_from_config(
        {
            "llm": {
                "extraction": {
                    "soft_primary_calls": 3,
                    "hard_primary_calls": 5,
                    "scheduler": {
                        "soft_checkpoint_calls": 11,
                        "safety_max_calls": 19,
                    },
                }
            }
        }
    )

    assert env["LLM_EXTRACTION_SOFT_CHECKPOINT_CALLS"] == "11"
    assert env["LLM_EXTRACTION_SAFETY_MAX_CALLS"] == "19"
    assert env["LLM_SOFT_PRIMARY_CALLS"] == "11"
    assert env["LLM_HARD_PRIMARY_CALLS"] == "19"


def test_generated_default_config_uses_only_canonical_scheduler_fields():
    from data_collection_workflow.runtime_profile import default_workflow_run_config

    config = default_workflow_run_config()
    extraction_config = config["llm"]["extraction"]

    assert config["pipeline_mode"] == "standard"
    assert extraction_config["scheduler"]["safety_max_calls"] == 2400
    assert "soft_primary_calls" not in extraction_config
    assert "hard_primary_calls" not in extraction_config


def test_source_search_authority_gap_retry_config_maps_to_env():
    from data_collection_workflow.workflow_run_config import workflow_run_env_from_config

    config = {
        "source_search": {
            "enabled": True,
            "mode": "live",
            "authority_gap_retry": {
                "enabled": True,
                "max_queries": 4,
                "known_domain_max_queries": 3,
                "jurisdiction_max_queries": 5,
                "official_page_family_max_queries": 2,
                "result_budget": 7,
                "max_iterations": 1,
            },
        }
    }

    env = workflow_run_env_from_config(config)

    assert env["AUTHORITY_GAP_RETRY_ENABLED"] == "true"
    assert env["AUTHORITY_GAP_RETRY_MAX_QUERIES"] == "4"
    assert env["AUTHORITY_GAP_KNOWN_DOMAIN_RETRY_MAX_QUERIES"] == "3"
    assert env["AUTHORITY_GAP_JURISDICTION_RETRY_MAX_QUERIES"] == "5"
    assert env["AUTHORITY_GAP_OFFICIAL_PAGE_FAMILY_RETRY_MAX_QUERIES"] == "2"
    assert env["AUTHORITY_GAP_RETRY_RESULT_BUDGET"] == "7"
    assert env["AUTHORITY_GAP_RETRY_MAX_ITERATIONS"] == "1"


def test_output_dir_override_replaces_non_sessionized_run_output_dir(tmp_path):
    from data_collection_workflow.workflow_run_config import (
        workflow_output_dir_from_config,
        workflow_run_config_with_overrides,
    )

    target_dir = tmp_path / "rerun_after_source_hint"
    config = {
        "output": {
            "sessionized": False,
            "run_output_root": "outputs/old_root",
            "run_output_dir": "outputs/old_target",
        }
    }

    overridden = workflow_run_config_with_overrides(config, output_dir=target_dir)

    assert overridden["output"]["run_output_root"] == str(target_dir)
    assert overridden["output"]["run_output_dir"] == str(target_dir)
    assert workflow_output_dir_from_config(overridden) == target_dir


def test_low_level_workflow_env_defaults_disease_intelligence_to_resilient_fallback():
    from data_collection_workflow.runtime_profile import workflow_run_env

    env = workflow_run_env()

    assert env["ENABLE_LLM_DISEASE_INTELLIGENCE"] == "true"
    assert env["DISEASE_INTELLIGENCE_FORCE_LLM"] == "true"
    assert env["DISEASE_INTELLIGENCE_FALLBACK_TO_CURATED"] == "true"


def test_jsonc_loader_preserves_urls_and_strips_comments(tmp_path):
    from data_collection_workflow.workflow_run_config import load_workflow_run_config

    config_path = tmp_path / "profile.jsonc"
    config_path.write_text(
        """
        {
          // Comments outside strings are ignored.
          "profile_name": "jsonc_test",
          "workflow": {
            "seed_source_overlay_path": "synthetic_seed_sources.json"
          },
          "source_sets": {
            "workflow_source_ids": ["src_synthetic_official_cases"]
          },
          "example_url": "https://example.org/path//still-string"
        }
        """,
        encoding="utf-8",
    )

    config = load_workflow_run_config(config_path)
    assert config["profile_name"] == "jsonc_test"
    assert config["example_url"] == "https://example.org/path//still-string"
    assert config["source_sets"]["workflow_source_ids"] == [
        "src_synthetic_official_cases"
    ]


def test_custom_live_config_does_not_inherit_new_mexico_sources(tmp_path):
    from data_collection_workflow.workflow_run_config import (
        load_workflow_run_config,
        workflow_run_env_from_config,
    )

    config_path = tmp_path / "new_york_live.jsonc"
    config_path.write_text(
        """
        {
          "profile_name": "new_york_live",
          "workflow": {
            "collection_mode": "standard",
            "use_fixture_documents": false
          },
          "structured_task": {
            "disease": "hantavirus",
            "location": "New York",
            "start_date": "2024",
            "end_date": "2026"
          },
          "source_search": {
            "enabled": true,
            "mode": "live",
            "combine_with_seed_catalog": false
          },
          "source_sets": {
            "source_id_allowlist_enabled": false
          },
          "validation": {
            "mode": "live_cross_source",
            "held_out_records_path": null
          }
        }
        """,
        encoding="utf-8",
    )

    config = load_workflow_run_config(config_path)
    env = workflow_run_env_from_config(config)

    assert env["SEED_SOURCE_OVERLAY_PATH"] == ""
    assert env["SOURCE_ROLE_POLICY_OVERLAY_PATH"] == ""
    assert env["SOURCE_ID_ALLOWLIST"] == ""
    assert "new_mexico" not in str(env).lower()


def test_live_cross_source_config_does_not_default_to_new_mexico_validation():
    from data_collection_workflow.validation_source_compatibility import (
        resolve_task_compatible_validation_records,
    )
    from data_collection_workflow.workflow_run_config import (
        validation_records_path_from_config,
        workflow_initial_state_from_config,
    )

    config = {
        "workflow": {"collection_mode": "standard", "use_fixture_documents": False},
        "structured_task": {
            "disease": "hantavirus",
            "location": "New York",
            "start_date": "2024",
            "end_date": "2026",
        },
        "validation": {"mode": "live_cross_source", "held_out_records_path": None},
    }

    assert validation_records_path_from_config(config) is None
    state = workflow_initial_state_from_config(config)
    resolved = resolve_task_compatible_validation_records(
        validation_records=[],
        state_or_task_context=state,
        validation_records_path=None,
        validation_records_path_requested=None,
        validation_records_explicit=False,
        validation_mode="live_cross_source",
    )

    summary = resolved["validation_source_compatibility_summary"]
    assert summary["validation_mode"] == "live_cross_source"
    assert summary["validation_records_path"] is None
    assert summary["validation_records_source"] == "none"
    assert summary["compatibility_status"] == "live_validation_pending"
    assert "New Mexico" not in str(summary)


def test_disable_all_llm_also_disables_explicit_source_credibility():
    from data_collection_workflow.runtime_profile import (
        default_workflow_run_config, workflow_run_config_with_overrides,
        workflow_run_env_from_config,
    )
    config = default_workflow_run_config()
    config['llm']['source_credibility']['enabled'] = True
    updated = workflow_run_config_with_overrides(config, all_llm=False)
    env = workflow_run_env_from_config(updated)
    assert env['ENABLE_LLM_SOURCE_CREDIBILITY'] == 'false'
    assert config['llm']['source_credibility']['enabled'] is True
