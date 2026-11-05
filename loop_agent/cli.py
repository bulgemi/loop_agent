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
from pathlib import Path

from harness2.cli import _common as C
from harness2.cli import run_sweep as RS
from harness2.core import schedule
from harness2.core.proposer import Proposer
from harness2.demo import DemoAdapter, DemoProposer

from .bridge import make_adk_adapter, wrap_adapter
from .runtime import load_dotenv


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
    return _run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
