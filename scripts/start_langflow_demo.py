"""Start the local Langflow demo shell for the Data Collection Workflow."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
import webbrowser
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_collection_workflow.runtime_profile import load_project_env  # noqa: E402

COMPONENTS_PATH = PROJECT_ROOT / "integrations" / "langflow" / "components"
FLOW_PATH = PROJECT_ROOT / "integrations" / "langflow" / "flows" / "workflow.json"
VISUAL_CONTRACT_VERSION = "workflow-ui/2"
VISUAL_FLOW_ID = "f4f7b3e7-8c7e-5a12-98fb-5a6e3d82f0f2"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the Data Collection Workflow Langflow visual demo."
    )
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8010)
    parser.add_argument("--langflow-host", default="127.0.0.1")
    parser.add_argument("--langflow-port", type=int, default=7860)
    parser.add_argument("--flow-id", default="")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--runner-user-request", default="")
    parser.add_argument("--runner-disease", default="")
    parser.add_argument("--runner-location", default="")
    parser.add_argument("--runner-start-date", default="")
    parser.add_argument("--runner-end-date", default="")
    parser.add_argument("--runner-session-id", default="")
    parser.add_argument("--runner-provider", choices=("anthropic", "openai"), default="")
    parser.add_argument("--runner-model", default="", help="Your provider's exact model ID; blank inherits LLM_MODEL at run time.")
    parser.add_argument("--runner-no-llm", action="store_true")
    parser.add_argument("--runner-quick-test-mode", action="store_true")
    return parser


def _find_available_port(host: str, preferred_port: int) -> int:
    for port in range(int(preferred_port), int(preferred_port) + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) != 0:
                return port
    return int(preferred_port)


def _env(api_base_url: str) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(SRC)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["LANGFLOW_COMPONENTS_PATH"] = str(COMPONENTS_PATH)
    env["LANGFLOW_DEMO_API_BASE"] = api_base_url
    return env


def _require_optional_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        print(
            f"Missing optional dependency: {name}. "
            "Install the demo extra with `python -m pip install -e .[langflow-demo]`.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _langflow_command(host: str, port: int) -> list[str]:
    langflow_bin = shutil.which("langflow")
    if langflow_bin:
        return [langflow_bin, "run", "--host", host, "--port", str(port)]
    return [sys.executable, "-m", "langflow", "run", "--host", host, "--port", str(port)]


def _start_process(command: list[str], env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
    )


def _wait_for_http_ready(
    url: str,
    *,
    timeout_seconds: float = 180.0,
    poll_interval_seconds: float = 1.0,
    request_get=requests.get,
    sleep=time.sleep,
) -> bool:
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while time.monotonic() < deadline:
        try:
            response = request_get(url, timeout=5)
            if getattr(response, "status_code", 500) < 500:
                return True
        except Exception:
            pass
        sleep(max(0.001, float(poll_interval_seconds)))
    return False


def _patch_template_value(node: dict, field_name: str, value: object) -> None:
    template = (((node.get("data") or {}).get("node") or {}).get("template") or {})
    field = template.get(field_name)
    if isinstance(field, dict):
        field["value"] = value
    elif isinstance(template, dict):
        template[field_name] = {
            "name": field_name,
            "display_name": field_name.replace("_", " ").title(),
            "value": value,
            "show": True,
            "advanced": False,
        }


def _flow_id_from_runner_defaults(runner_defaults: dict[str, object] | None = None) -> str:
    session_id = str((runner_defaults or {}).get("session_id") or "").strip()
    if not session_id:
        return VISUAL_FLOW_ID
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"workflow-langflow:{session_id}"))


def _flow_name_from_runner_defaults(runner_defaults: dict[str, object] | None = None) -> str:
    session_id = str((runner_defaults or {}).get("session_id") or "").strip()
    if session_id:
        return f"Data Collection Workflow - {session_id}"
    return "Data Collection Workflow"


def _patched_flow_payload(
    *,
    api_base_url: str,
    runner_defaults: dict[str, object] | None = None,
    flow_id: str | None = None,
    flow_path: Path = FLOW_PATH,
) -> dict:
    payload = json.loads(flow_path.read_text(encoding="utf-8"))
    runner_defaults = runner_defaults or {}
    payload["id"] = str(flow_id or _flow_id_from_runner_defaults(runner_defaults))
    payload["name"] = _flow_name_from_runner_defaults(runner_defaults)

    for node in ((payload.get("data") or {}).get("nodes") or []):
        _patch_template_value(node, "api_base_url", api_base_url)
        if (node.get("data") or {}).get("type") == "workflow_runner":
            for key, value in runner_defaults.items():
                _patch_template_value(node, key, value)
    return payload


def _upload_flow_to_langflow(
    langflow_base_url: str,
    flow_payload: dict,
    *,
    request_session_factory=requests.Session,
) -> dict[str, object]:
    base = langflow_base_url.rstrip("/")
    session = request_session_factory()
    login_response = session.get(f"{base}/api/v1/auto_login", timeout=30)
    login_response.raise_for_status()
    content = json.dumps(flow_payload, ensure_ascii=False).encode("utf-8")
    response = session.post(
        f"{base}/api/v1/flows/upload/",
        files={"file": ("workflow.json", content, "application/json")},
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    first = body[0] if isinstance(body, list) and body else body
    flow_id = str(first.get("id")) if isinstance(first, dict) and first.get("id") else ""
    return {
        "flow_id": flow_id,
        "flow_url": f"{base}/flow/{flow_id}" if flow_id else base,
        "response": body,
    }


def _runner_defaults_from_args(args: argparse.Namespace) -> dict[str, object]:
    defaults: dict[str, object] = {}
    for attr, field in (
        ("runner_user_request", "user_request"),
        ("runner_disease", "disease"),
        ("runner_location", "location"),
        ("runner_start_date", "start_date"),
        ("runner_end_date", "end_date"),
        ("runner_session_id", "session_id"),
        ("runner_provider", "provider"),
        ("runner_model", "model"),
    ):
        value = str(getattr(args, attr, "") or "").strip()
        if value:
            defaults[field] = value
    if getattr(args, "runner_no_llm", False):
        defaults["no_llm"] = True
    if getattr(args, "runner_quick_test_mode", False):
        defaults["quick_test_mode"] = True
    return defaults


def _ensure_processes_alive(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is not None:
            raise RuntimeError(f"Process exited with code {process.returncode}.")


def _terminate(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    load_project_env()
    _require_optional_module("fastapi")
    _require_optional_module("uvicorn")
    _require_optional_module("langflow")

    api_port = _find_available_port(args.api_host, args.api_port)
    langflow_port = _find_available_port(args.langflow_host, args.langflow_port)
    api_base_url = f"http://{args.api_host}:{api_port}"
    env = _env(api_base_url)

    api_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "data_collection_workflow.langflow_demo:create_app",
        "--factory",
        "--host",
        args.api_host,
        "--port",
        str(api_port),
    ]
    langflow_command = _langflow_command(args.langflow_host, langflow_port)

    print("Starting Data Collection Langflow demo.")
    print(f"components_path: {COMPONENTS_PATH}")

    processes: list[subprocess.Popen] = []
    try:
        print(f"Starting demo API on {api_base_url} ...")
        api_process = _start_process(api_command, env)
        processes.append(api_process)
        if not _wait_for_http_ready(f"{api_base_url}/health", timeout_seconds=60):
            raise RuntimeError(f"Demo API did not become ready: {api_base_url}")
        _ensure_processes_alive(processes)
        print(f"demo_api ready: {api_base_url}")

        langflow_base_url = f"http://{args.langflow_host}:{langflow_port}"
        print(f"Starting Langflow on {langflow_base_url} ...")
        langflow_process = _start_process(langflow_command, env)
        processes.append(langflow_process)
        if not _wait_for_http_ready(f"{langflow_base_url}/health_check", timeout_seconds=240):
            raise RuntimeError(f"Langflow did not become ready: {langflow_base_url}")
        _ensure_processes_alive(processes)
        print(f"langflow ready: {langflow_base_url}")

        flow_result: dict[str, object] | None = None
        try:
            runner_defaults = _runner_defaults_from_args(args)
            flow_id = str(args.flow_id or "").strip() or _flow_id_from_runner_defaults(runner_defaults)
            flow_payload = _patched_flow_payload(
                api_base_url=api_base_url,
                runner_defaults=runner_defaults,
                flow_id=flow_id,
            )
            flow_result = _upload_flow_to_langflow(langflow_base_url, flow_payload)
            print(f"flow_url: {flow_result['flow_url']}")
            print("Flow imported and prefilled.")
            print("Use the newly opened flow URL; old Langflow tabs may show stale values.")
            print("Workflow state: NOT_STARTED. Click Play on Data Collection Workflow Runner once to begin.")
            if not args.no_browser:
                webbrowser.open_new_tab(str(flow_result["flow_url"]))
        except Exception as exc:
            print(
                "Could not auto-import the Langflow flow. "
                f"Import this file manually: {FLOW_PATH} ({exc.__class__.__name__}: {exc})",
                file=sys.stderr,
            )
            print(f"langflow: {langflow_base_url}")
            if not args.no_browser:
                webbrowser.open(langflow_base_url)

        while True:
            for process in processes:
                if process.poll() is not None:
                    print(f"Process exited with code {process.returncode}.", file=sys.stderr)
                    return int(process.returncode or 0)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping Data Collection Langflow demo.")
        return 0
    finally:
        _terminate(processes)


if __name__ == "__main__":
    raise SystemExit(main())
