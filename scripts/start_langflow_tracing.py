"""Start the deep Langflow + LangGraph Studio + LangSmith demo shell."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_collection_workflow.runtime_profile import load_project_env  # noqa: E402

COMPONENTS_PATH = PROJECT_ROOT / "integrations" / "langflow" / "components"
FLOW_PATH = PROJECT_ROOT / "integrations" / "langflow" / "flows" / "workflow.json"
DEFAULT_LANGSMITH_PROJECT = "data-collection-workflow-demo"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the Data Collection deep visual demo with Langflow, Studio, and LangSmith."
    )
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8010)
    parser.add_argument("--langflow-host", default="127.0.0.1")
    parser.add_argument("--langflow-port", type=int, default=7860)
    parser.add_argument("--studio-port", type=int, default=2024)
    parser.add_argument("--langsmith-project", default=DEFAULT_LANGSMITH_PROJECT)
    return parser


def _find_available_port(host: str, preferred_port: int) -> int:
    for port in range(int(preferred_port), int(preferred_port) + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) != 0:
                return port
    return int(preferred_port)


def _require_optional_module(name: str, extra: str) -> None:
    if importlib.util.find_spec(name) is None:
        print(
            f"Missing optional dependency: {name}. "
            f"Install with `python -m pip install -e .[{extra}]`.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _require_langsmith_key() -> None:
    if not (os.environ.get("LANGSMITH_API_KEY") or "").strip():
        print(
            "Missing LANGSMITH_API_KEY. Deep visual demo requires LangSmith tracing.",
            file=sys.stderr,
        )
        raise SystemExit(3)


def _langflow_command(host: str, port: int) -> list[str]:
    langflow_bin = shutil.which("langflow")
    if langflow_bin:
        return [langflow_bin, "run", "--host", host, "--port", str(port)]
    return [sys.executable, "-m", "langflow", "run", "--host", host, "--port", str(port)]


def _langgraph_command(port: int) -> list[str]:
    langgraph_bin = shutil.which("langgraph")
    if langgraph_bin:
        return [langgraph_bin, "dev", "--no-reload", "--port", str(port)]
    return [
        sys.executable,
        "-m",
        "langgraph_cli.cli",
        "dev",
        "--no-reload",
        "--port",
        str(port),
    ]


def _env(api_base_url: str, studio_url: str, project_name: str) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(SRC)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["LANGFLOW_COMPONENTS_PATH"] = str(COMPONENTS_PATH)
    env["LANGFLOW_DEMO_API_BASE"] = api_base_url
    env["LANGGRAPH_STUDIO_URL"] = studio_url
    env["LANGSMITH_TRACING"] = "true"
    env["LANGSMITH_PROJECT"] = project_name
    env["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"
    return env


def _start_process(command: list[str], env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
    )


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
    _require_optional_module("fastapi", "langflow-deep-demo")
    _require_optional_module("uvicorn", "langflow-deep-demo")
    _require_optional_module("langflow", "langflow-deep-demo")
    _require_optional_module("langsmith", "langflow-deep-demo")
    _require_optional_module("langgraph_cli", "langflow-deep-demo")
    _require_langsmith_key()

    api_port = _find_available_port(args.api_host, args.api_port)
    langflow_port = _find_available_port(args.langflow_host, args.langflow_port)
    studio_port = _find_available_port("127.0.0.1", args.studio_port)
    api_base_url = f"http://{args.api_host}:{api_port}"
    studio_url = f"http://127.0.0.1:{studio_port}"
    env = _env(api_base_url, studio_url, args.langsmith_project)

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
    processes = [
        _start_process(api_command, env),
        _start_process(_langflow_command(args.langflow_host, langflow_port), env),
        _start_process(_langgraph_command(studio_port), env),
    ]

    print("Starting Data Collection deep visual demo.")
    print(f"demo_api: {api_base_url}")
    print(f"langflow: http://{args.langflow_host}:{langflow_port}")
    print(f"langgraph_studio: {studio_url}")
    print(f"langsmith_project: {args.langsmith_project}")
    print(f"components_path: {COMPONENTS_PATH}")
    print(f"deep_flow_blueprint: {FLOW_PATH}")

    try:
        while True:
            for process in processes:
                if process.poll() is not None:
                    print(f"Process exited with code {process.returncode}.", file=sys.stderr)
                    return int(process.returncode or 0)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping Data Collection deep visual demo.")
        return 0
    finally:
        _terminate(processes)


if __name__ == "__main__":
    raise SystemExit(main())
