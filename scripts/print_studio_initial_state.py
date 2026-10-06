"""Print an Data Collection Workflow initial state as JSON for LangGraph Studio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_collection_workflow.workflow_run_config import (  # noqa: E402
    DEFAULT_WORKFLOW_RUN_CONFIG_PATH,
    load_workflow_run_config,
    workflow_initial_state_from_config,
    workflow_run_config_with_overrides,
)


def build_initial_state(
    user_request: str | None = None,
    *,
    config_path: str | None = None,
    minimal: bool = False,
) -> dict:
    config = workflow_run_config_with_overrides(
        load_workflow_run_config(config_path),
        user_request=user_request,
    )
    return workflow_initial_state_from_config(
        config,
        include_empty_fields=not minimal,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the LangGraph Studio input payload for a workflow run."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_WORKFLOW_RUN_CONFIG_PATH),
        help="Path to the workflow runtime JSON config file.",
    )
    parser.add_argument(
        "--user-request",
        default=None,
        help="Task text to place in the Studio User Request field.",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Print only user_request; useful when Studio exposes state fields directly.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_initial_state(
                args.user_request,
                config_path=args.config,
                minimal=args.minimal,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
