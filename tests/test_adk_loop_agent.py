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

"""End-to-end tests for the decoupled `loop_agent` package (`runtime`, `bridge`, and `cli`)."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from typing import AsyncGenerator
from unittest.mock import patch

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.tools import AgentTool
from google.genai import types

from harness2.core import prompts as HP
from harness2.core import schedule
from harness2.core import tree as HT
from harness2.core.spec import BenchmarkAdapter, SubstrateLike
from loop_agent import (
    build_loop_agent,
    make_workspace_tools,
    render_events,
    run_loop_rollout,
)
from loop_agent.bridge import (
    ADK_SURFACE,
    DEMO_STOP_HOOK,
    AdkBenchmarkAdapter,
    AdkSubstrate,
    make_adk_adapter,
    wrap_adapter,
)
from loop_agent.cli import main as loop_cli_main

REPO_ROOT = Path(__file__).resolve().parents[1]


class _TwoIterationHookTestLlm(BaseLlm):
    """Custom ADK BaseLlm that attempts premature exit on iteration 1 and fixes it on iteration 2."""

    model: str = "test/two-iteration-hook"
    solver_turns: int = 0
    verifier_turns: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        sys_inst = str(getattr(llm_request.config, "system_instruction", "") or "")
        last_content = llm_request.contents[-1] if llm_request.contents else None
        last_parts = (last_content.parts or []) if last_content else []
        fn_responses = [
            p.function_response
            for p in last_parts
            if getattr(p, "function_response", None) is not None
        ]
        is_verifier = "role: verifier" in sys_inst.lower()

        if not is_verifier:
            if not fn_responses:
                self.solver_turns += 1
                if self.solver_turns == 1:
                    yield LlmResponse(
                        content=types.Content(
                            role="model",
                            parts=[
                                types.Part.from_text(
                                    text="First solver turn: skipping deliverable creation."
                                )
                            ],
                        )
                    )
                    return
                yield LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part.from_function_call(
                                name="write_file",
                                args={"path": "answer.txt", "content": "Total: 85.15\n"},
                            )
                        ],
                    )
                )
                return
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="Second solver turn wrote answer.txt.")],
                )
            )
            return

        if not fn_responses:
            self.verifier_turns += 1
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_function_call(
                            name="exit_loop",
                            args={"summary": f"Verifier attempt #{self.verifier_turns}"},
                        )
                    ],
                )
            )
            return

        last_fn = fn_responses[-1]
        resp = last_fn.response if isinstance(last_fn.response, dict) else {}
        if resp.get("decision") == "block":
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=f"Blocked by Stop hook: {resp.get('reason')}; retrying solver."
                        )
                    ],
                )
            )
            return
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="Stop hook passed; escalating out of LoopAgent.")],
            )
        )


class AdkLoopAgentTest(unittest.TestCase):
    def test_runtime_has_zero_harness2_dependency_and_upstream_is_unmodified(self) -> None:
        for rel in ("loop_agent/__init__.py", "loop_agent/runtime.py"):
            src = (REPO_ROOT / rel).read_text(encoding="utf-8")
            mod = ast.parse(src, filename=rel)
            imported: list[str] = []
            for node in ast.walk(mod):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module)
            self.assertFalse(
                any(name == "harness2" or name.startswith("harness2.") for name in imported),
                f"{rel} must not import harness2, found: {imported}",
            )

        diff = subprocess.run(
            ["git", "diff", "origin/main", "--", "harness2/", "pyproject.toml", ".env.example"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(diff.stdout.strip(), "")

    def test_adk_surface_validation_and_autotrigger(self) -> None:
        tree: HT.Tree = {
            "systemprompt.md": "You are a financial ledger assistant.\n",
            "guardrails.md": "",
            "LongTermMEMORY.md": "",
            "tool_descriptions/tool_guidance.md": "",
            "skills/ledger_audit/SKILL.md": (
                "---\nname: ledger_audit\ndescription: auditing CSV ledgers\n---\n\n"
                "Run `python3 scripts/summarize.py ledger.csv`.\n"
            ),
            "sub_agents/solver/AGENT.md": (
                "---\ndescription: Solver stage override\n---\n\nCompute totals.\n"
            ),
            "sub_agents/verifier/AGENT.md": (
                "---\ndescription: Verifier stage override\n---\n\nVerify answer.txt.\n"
            ),
            "sub_agents/tax_specialist/AGENT.md": (
                "---\ndescription: Delegated tax specialist subagent\n---\n\nCheck tax lines.\n"
            ),
            "scripts/summarize.py": "print('Total: 85.15')\n",
            ".adk/hooks.json": json.dumps(DEMO_STOP_HOOK),
        }
        HT.autotrigger(ADK_SURFACE, tree)
        self.assertIn("load the `ledger_audit` skill", tree["guardrails.md"])
        stats = HT.validate(ADK_SURFACE, tree, allow_subagents=True)
        self.assertEqual(stats["skills"], 1)
        self.assertEqual(stats["sub_agents"], 3)
        self.assertEqual(stats["scripts"], 1)
        self.assertEqual(stats["hooks"], 1)

        bad_hooks = dict(tree)
        bad_hooks[".adk/hooks.json"] = json.dumps({"hooks": {"InvalidHookEvent": []}})
        with self.assertRaises(ValueError) as ctx:
            HT.validate(ADK_SURFACE, bad_hooks, allow_subagents=True)
        self.assertIn("unknown event", str(ctx.exception))

    def test_build_loop_agent_topology_and_delegated_subagents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staged = root / "staged"
            ws = root / "workspace"
            tree: HT.Tree = {
                "systemprompt.md": "Always verify numeric formatting.\n",
                "guardrails.md": "- When auditing ledgers, load the `ledger_audit` skill.\n",
                "LongTermMEMORY.md": "Currency values use two decimal places.\n",
                "tool_descriptions/tool_guidance.md": "Prefer scripts over mental arithmetic.\n",
                "skills/ledger_audit/SKILL.md": (
                    "---\nname: ledger_audit\ndescription: auditing ledgers\n---\n\n"
                    "Use `scripts/summarize.py`.\n"
                ),
                "sub_agents/solver/AGENT.md": (
                    "---\ndescription: Custom solver stage\n---\n\nCUSTOM_SOLVER_MARKER\n"
                ),
                "sub_agents/verifier/AGENT.md": (
                    "---\ndescription: Custom verifier stage\n---\n\nCUSTOM_VERIFIER_MARKER\n"
                ),
                "sub_agents/auditor/AGENT.md": (
                    "---\ndescription: Independent auditor subagent\n---\n\nAUDITOR_SUBAGENT_BODY\n"
                ),
                "scripts/summarize.py": "print('Total: 85.15')\n",
            }
            HT.write_tree(ADK_SURFACE, tree, staged)

            loop_agent = build_loop_agent(
                staged, ws, model="local/deterministic", max_iterations=4
            )
            self.assertIsInstance(loop_agent, LoopAgent)
            self.assertEqual(loop_agent.max_iterations, 4)
            self.assertEqual(len(loop_agent.sub_agents), 2)

            solver, verifier = loop_agent.sub_agents
            self.assertIsInstance(solver, LlmAgent)
            self.assertIsInstance(verifier, LlmAgent)
            self.assertEqual(solver.name, "solver")
            self.assertEqual(verifier.name, "verifier")
            self.assertIn("CUSTOM_SOLVER_MARKER", str(solver.instruction))
            self.assertIn("CUSTOM_VERIFIER_MARKER", str(verifier.instruction))
            self.assertIn("ledger_audit", str(solver.instruction))

            agent_tools = [t for t in solver.tools if isinstance(t, AgentTool)]
            self.assertEqual(len(agent_tools), 1)
            self.assertEqual(agent_tools[0].agent.name, "auditor")

            ws_tools = make_workspace_tools(staged, ws)
            skill_res = ws_tools["load_skill"]("ledger_audit")
            self.assertEqual(skill_res["status"], "ok")
            self.assertIn("scripts/summarize.py", skill_res["content"])

    def test_stop_hook_blocks_premature_exit_loop_and_escalates_on_second_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staged = root / "staged"
            ws = root / "workspace"
            events_path = root / "adk_events.jsonl"
            tree: HT.Tree = {
                "systemprompt.md": "Compute the ledger total and write answer.txt.\n",
                "guardrails.md": "",
                "LongTermMEMORY.md": "",
                "tool_descriptions/tool_guidance.md": "",
                ".adk/hooks.json": json.dumps(DEMO_STOP_HOOK),
            }
            HT.write_tree(ADK_SURFACE, tree, staged)

            llm = _TwoIterationHookTestLlm()
            result = run_loop_rollout(
                staged,
                ws,
                "Sum ledger.csv and write answer.txt.",
                events_path,
                model=llm,
                max_iterations=3,
            )
            self.assertTrue(result["escalated"])
            self.assertEqual(llm.solver_turns, 2)
            self.assertEqual(llm.verifier_turns, 2)
            self.assertEqual((ws / "answer.txt").read_text(encoding="utf-8").strip(), "Total: 85.15")

            rendered = render_events(events_path, obs_trunc=5000, max_chars=50_000)
            self.assertIn("[ACTION] (verifier) exit_loop", rendered)
            self.assertIn("[OBSERVATION ERROR]", rendered)
            self.assertIn("answer.txt missing or invalid", rendered)
            self.assertIn("[ACTION] (solver) write_file", rendered)
            self.assertIn("[HARNESS] loop escalated by verifier", rendered)

    def test_bridge_wraps_demo_and_benchmark_adapters(self) -> None:
        self.assertIn("adk", HP._SURFACE_TEXTS)
        with tempfile.TemporaryDirectory() as tmp:
            demo_adk = make_adk_adapter("demo", root=Path(tmp))
            self.assertIsInstance(demo_adk, AdkBenchmarkAdapter)
            self.assertIsInstance(demo_adk, BenchmarkAdapter)
            self.assertIsInstance(demo_adk.substrate, AdkSubstrate)
            self.assertIsInstance(demo_adk.substrate, SubstrateLike)
            self.assertEqual(demo_adk.substrate.name, "adk")
            self.assertEqual(demo_adk.spec.install_mode, "workspace_config")

        jb_adk = make_adk_adapter("jb")
        self.assertEqual(jb_adk.spec.name, "jb")
        self.assertEqual(jb_adk.substrate.name, "adk")

        lab_adk = make_adk_adapter("lab")
        self.assertEqual(lab_adk.spec.name, "lab")
        self.assertEqual(lab_adk.substrate.name, "adk")

        wb_adk = make_adk_adapter("wb", "office")
        self.assertEqual(wb_adk.spec.name, "wb-office")
        self.assertEqual(wb_adk.substrate.name, "adk")

        with patch.object(
            AdkBenchmarkAdapter,
            "tasks",
            return_value=["finance__task_1"],
        ), patch.object(
            AdkBenchmarkAdapter,
            "domain_of",
            return_value="finance",
        ), patch.object(
            AdkBenchmarkAdapter,
            "reference_domain_order",
            return_value=["finance"],
        ):
            rc = loop_cli_main(["sweep", "--bench", "jb", "--k", "1", "--dry-run"])
        self.assertEqual(rc, 0)

    def test_cli_demo_runs_end_to_end_with_adk_loop_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "demo_adk"
            rc = loop_cli_main(
                ["demo", "--output", str(out), "--k", "1", "--mode", "both", "--max-iterations", "2"]
            )
            self.assertEqual(rc, 0)

            report = json.loads((out / "demo.json").read_text(encoding="utf-8"))
            self.assertEqual(report["done"], 2)
            self.assertEqual(report["failed"], 0)

            ns = schedule.ns("demo", "adk", "agg", 1)
            for task_id, expected in (("ledger-1", "Total: 85.15"), ("ledger-2", "Total: 20.25")):
                summary = json.loads(
                    (out / "runs" / ns / task_id / "summary.json").read_text(encoding="utf-8")
                )
                for mode in ("agg", "seq"):
                    run_dir = Path(summary["selections"][mode]["chosen_run_dir"])
                    self.assertEqual(
                        (run_dir / "answer.txt").read_text(encoding="utf-8").strip(),
                        expected,
                    )
                    self.assertTrue((run_dir / "adk_events.jsonl").is_file())
                    traj = (run_dir / "trajectory.md").read_text(encoding="utf-8")
                    self.assertIn("[HARNESS] loop escalated by verifier", traj)


if __name__ == "__main__":
    unittest.main()
