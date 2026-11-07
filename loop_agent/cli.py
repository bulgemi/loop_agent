# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI entrypoints for running `harness2` demos and benchmark sweeps with Google ADK `LoopAgent`."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from harness2.cli import _common as C
from harness2.cli import run_sweep as RS
from harness2.core import schedule
from harness2.core.proposer import Proposer
from harness2.demo import DemoAdapter, DemoProposer

from .bridge import make_adk_adapter, wrap_adapter
from .runtime import ROOT_DIR, adk_session_service_uri, load_dotenv


def _run_demo(args: argparse.Namespace) -> int:
    if args.k < 1:
        raise SystemExit("--k must be positive")
    root = args.output.expanduser().resolve()
    adapter = wrap_adapter(DemoAdapter(root), max_iterations=args.max_iterations)
    proposer = DemoProposer(adapter.spec.channels["proposer"])
    modes = ("agg", "seq") if args.mode == "both" else (schedule.mode_token(args.mode),)
    result = schedule.run_domain(
        "accounting",
        adapter.tasks(),
        args.k,
        adapter=adapter,
        prop=proposer,
        runs_root=adapter.spec.runs_root,
        library_root=root / "library" / f"k{args.k}",
        modes=modes,
    )
    result["execution_count"] = adapter.execution_count
    result["editor_calls"] = len(proposer._calls)
    report = root / "demo.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"ADK LoopAgent demo: {result['done']} tasks completed; "
        f"{result['failed']} failed. {report}"
    )
    return int(result["failed"] > 0)


def _run_sweep(args: argparse.Namespace) -> int:
    if args.k < 1 or args.workers < 1:
        raise SystemExit("--k and --workers must be positive")
    modes = C.parse_modes(args.modes)
    adapter = make_adk_adapter(
        args.bench,
        args.subset,
        split_file=args.split_file,
        model=args.model,
        max_iterations=args.max_iterations,
    )
    if args.dry_run:
        return RS._print_stream(adapter, args, modes)

    adapter.preflight()
    runs_root = C.runs_root_for(adapter)
    library_root = C.library_root_for(adapter, args.k, seed=args.seed)
    summary_ns = C.summary_namespace(adapter, args.k, seed=args.seed)
    prop = Proposer(
        adapter.spec.channels["proposer"], audit_dir=runs_root / "_audit" / summary_ns
    )
    only = RS._csv(args.only_domains)
    rep = schedule.run_sweep(
        args.k,
        adapter=adapter,
        prop=prop,
        runs_root=runs_root,
        library_root=library_root,
        split=args.split,
        modes=modes,
        seed=args.seed,
        tag="",
        domain_parallel=args.workers,
        only_domains=only,
        max_tasks=args.max_tasks,
        pool_policy="finals",
    )
    done = sum(int(d.get("done", 0)) for d in rep["domains"])
    failed = sum(int(d.get("failed", 0)) for d in rep["domains"])
    print(f"ADK sweep {summary_ns}: {done} tasks ok, {failed} failed")
    return 2 if failed else 0


def _ensure_telemetry_consent() -> None:
    try:
        from google.adk.utils import _telemetry_config

        if _telemetry_config.read_telemetry_consent() is None:
            _telemetry_config.write_telemetry_consent(False)
    except Exception:
        pass


def _invoke_adk_cli(adk_args: list[str], env: dict[str, str]) -> int:
    os.environ.update(env)
    _ensure_telemetry_consent()
    try:
        from google.adk.cli import main as adk_main

        rc = adk_main(args=adk_args, prog_name="adk", standalone_mode=False)
        return int(rc) if isinstance(rc, int) else 0
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 0
    except ImportError:
        adk_bin = shutil.which("adk") or str(Path(sys.executable).parent / "adk")
        return subprocess.call([adk_bin, *adk_args], env=env)


def _run_serve(args: argparse.Namespace) -> int:
    adk_bin = shutil.which("adk") or str(Path(sys.executable).parent / "adk")
    subcmd = "web" if args.web else args.server_mode
    default_db = ROOT_DIR / "runs" / "adk_interactive" / "sessions.db"
    default_db.parent.mkdir(parents=True, exist_ok=True)
    session_uri = (
        args.session_service_uri
        or adk_session_service_uri()
        or f"sqlite:///{default_db.resolve()}"
    )

    env = dict(os.environ)
    env["HARNESS2_ADK_ISOLATE_SESSIONS"] = "FALSE" if args.no_isolate_sessions else "TRUE"
    if args.harness_dir is not None:
        env["HARNESS2_ADK_HARNESS_DIR"] = str(args.harness_dir.expanduser().resolve())
    if args.workspace is not None:
        env["HARNESS2_ADK_WORKSPACE"] = str(args.workspace.expanduser().resolve())

    adk_args = [
        subcmd,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--reload" if args.reload else "--no-reload",
        f"--session_service_uri={session_uri}",
    ]
    if args.reload_agents:
        adk_args.append("--reload_agents")
    adk_args.append(str(ROOT_DIR / "loop_agent"))

    argv = [adk_bin, *adk_args]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "argv": argv,
                    "isolate_sessions": env["HARNESS2_ADK_ISOLATE_SESSIONS"] == "TRUE",
                    "session_service_uri": session_uri,
                }
            )
        )
        return 0
    return _invoke_adk_cli(adk_args, env)


def _run_interactive(args: argparse.Namespace) -> int:
    adk_bin = shutil.which("adk") or str(Path(sys.executable).parent / "adk")
    env = dict(os.environ)
    env["HARNESS2_ADK_ISOLATE_SESSIONS"] = "FALSE" if args.no_isolate_sessions else "TRUE"
    if args.harness_dir is not None:
        env["HARNESS2_ADK_HARNESS_DIR"] = str(args.harness_dir.expanduser().resolve())
    if args.workspace is not None:
        env["HARNESS2_ADK_WORKSPACE"] = str(args.workspace.expanduser().resolve())

    adk_args = ["run"]
    if args.save_session:
        adk_args.append("--save_session")
    adk_args.append(str(ROOT_DIR / "loop_agent"))
    argv = [adk_bin, *adk_args]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "argv": argv,
                    "isolate_sessions": env["HARNESS2_ADK_ISOLATE_SESSIONS"] == "TRUE",
                }
            )
        )
        return 0
    return _invoke_adk_cli(adk_args, env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loop_agent.cli", description=__doc__)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional path to a .env file to load before running",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="Run the offline ADK LoopAgent adaptation demo")
    p_demo.add_argument("--output", type=Path, default=Path("runs/demo_adk"))
    p_demo.add_argument("--k", type=int, default=2)
    p_demo.add_argument("--mode", choices=("parallel", "sequential", "both"), default="both")
    p_demo.add_argument("--max-iterations", type=int, default=None)

    p_sweep = sub.add_parser("sweep", help="Run a benchmark sweep with AdkBenchmarkAdapter")
    p_sweep.add_argument("--bench", required=True, choices=C.BENCHES)
    p_sweep.add_argument("--subset", default=None, choices=C.WB_SUBSETS)
    C.add_run_identity_args(p_sweep)
    p_sweep.add_argument("--split", choices=("all", "train", "test"), default="all")
    p_sweep.add_argument("--split-file", type=Path, default=None)
    p_sweep.add_argument("--workers", type=int, default=1)
    p_sweep.add_argument("--only-domains", default="")
    p_sweep.add_argument("--max-tasks", type=int, default=None)
    p_sweep.add_argument("--model", default="")
    p_sweep.add_argument("--max-iterations", type=int, default=None)
    p_sweep.add_argument("--dry-run", action="store_true")

    p_serve = sub.add_parser(
        "serve",
        help="Start multi-user ADK Web UI or FastAPI server with per-session isolation & DB storage",
    )
    p_serve.add_argument(
        "--server-mode",
        choices=("api_server", "web"),
        default="api_server",
        help="ADK server mode (`api_server` for REST/SSE or `web` for Web UI)",
    )
    p_serve.add_argument("--web", action="store_true", help="Shortcut for `--server-mode web`")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument(
        "--session-service-uri",
        default="",
        help="Persistent session DB URI (default: sqlite:///runs/adk_interactive/sessions.db)",
    )
    p_serve.add_argument("--harness-dir", type=Path, default=None)
    p_serve.add_argument("--workspace", type=Path, default=None)
    p_serve.add_argument("--no-isolate-sessions", action="store_true")
    p_serve.add_argument("--reload", action="store_true", help="Enable uvicorn reload")
    p_serve.add_argument(
        "--reload-agents",
        action="store_true",
        help="Live-reload agent definitions on file change",
    )
    p_serve.add_argument("--dry-run", action="store_true")

    p_run = sub.add_parser(
        "run",
        help="Start an interactive terminal chat session (`adk run`) in-process",
    )
    p_run.add_argument("--harness-dir", type=Path, default=None)
    p_run.add_argument("--workspace", type=Path, default=None)
    p_run.add_argument("--no-isolate-sessions", action="store_true")
    p_run.add_argument("--save-session", action="store_true")
    p_run.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.env_file is not None:
        load_dotenv(args.env_file, override=True)
        config_load = getattr(C.config, "load_dotenv", None)
        if callable(config_load):
            config_load(args.env_file)
    else:
        load_dotenv()

    if args.cmd == "demo":
        return _run_demo(args)
    if args.cmd == "serve":
        return _run_serve(args)
    if args.cmd == "run":
        return _run_interactive(args)
    return _run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
