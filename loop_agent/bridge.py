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

"""Layer 2: Non-invasive bridge connecting `loop_agent.runtime` to `harness2`.

Requires ZERO modifications to the upstream `harness2` package:
  - Registers `ADK_SURFACE` into `harness2.core.prompts` at runtime.
  - Implements `AdkSubstrate` (`SubstrateLike` / `Substrate` protocol) using standard
    `harness2.core.spec.INSTALL_MODES`.
  - Provides `AdkBenchmarkAdapter` (Decorator/Proxy pattern) that wraps any existing
    `BenchmarkAdapter` (`DemoAdapter`, `JobBenchAdapter`, `LabAdapter`, `WbAdapter`),
    delegating task discovery, deliverable extraction, and judging to the wrapped adapter
    while executing rollouts via `loop_agent.runtime` (`google.adk.agents.LoopAgent`).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from harness2 import config
from harness2.core import prompts as HP
from harness2.core import runmark as RM
from harness2.core import tree as HT
from harness2.core.spec import BenchPrompts, BenchSpec
from harness2.core.tree import SurfaceSpec
from harness2.substrate.base import Install, Launcher, RunRequest

from . import runtime as RT

ADK_SURFACE = SurfaceSpec(
    name="adk",
    always_on=RT.ALWAYS_ON_FILES,
    always_on_max=24_000,
    always_on_cap_reason=(
        "every token here is injected into every LoopAgent iteration and competes "
        "with the agent's own output budget"
    ),
    skill_dir="skills",
    skill_name_rule="must_match",
    trigger_hint=(
        "Fix it by adding a trigger line naming `{name}` to systemprompt.md or "
        'guardrails.md, e.g. "When <the situation this skill handles>, load the '
        '`{name}` skill."'
    ),
    autotrigger_file="guardrails.md",
    trigger_line="- When {desc}, load the `{name}` skill.\n",
    subagent_requires_mode=False,
    plugin_dir=None,
    hooks_file=".adk/hooks.json",
    hook_events=frozenset({"PreToolUse", "PostToolUse", "Stop", "SessionStart"}),
    compile_skill_scripts=True,
    stubs=(
        (
            "scripts/README.md",
            "<!-- c7: deterministic helpers, run as `python3 scripts/<name>.py`. "
            "Add one by emitting scripts/<name>.py. -->\n",
        ),
        (
            "sub_agents/README.md",
            "<!-- c5: LoopAgent stage overrides (`sub_agents/solver/AGENT.md`, "
            "`sub_agents/verifier/AGENT.md`) or delegated subagents (`sub_agents/<name>/AGENT.md`) "
            "invoked via AgentTool in their OWN context. Frontmatter requires `description:`. -->\n",
        ),
        (
            ".adk/README.md",
            "<!-- c8: ADK LoopAgent lifecycle hooks -- acts INSIDE the tool and escalation loop, "
            "so the model cannot skip it. Add ONE file, .adk/hooks.json, with schema: "
            '{"hooks": {"PostToolUse": [{"matcher": "run_bash", "hooks": [{"type": '
            '"command", "command": "..."}]}], "Stop": [{"matcher": "exit_loop", "hooks": '
            '[{"type": "command", "command": "..."}]}]}}. Printing '
            '{"decision":"block","reason":"..."} blocks a tool call or prevents exit_loop from '
            "escalating out of the LoopAgent. -->\n",
        ),
        (
            "skills/README.md",
            "<!-- c6: reusable workflows loaded on demand via the `load_skill` tool. Add one as "
            "skills/<name>/SKILL.md with frontmatter `name:` matching the directory and a "
            "`description:`. -->\n",
        ),
    ),
)

_SURFACE_ADK_MD = """\
# WHAT YOU CAN CHANGE

The harness is a tree of components for a Google ADK `LoopAgent` (`solver` -> `verifier` loop).
Each has a different runtime cost and a different reach:

| path | what it is | when it is in context |
|---|---|---|
| `systemprompt.md` | role and procedure | ALWAYS ON -- injected into every LoopAgent stage; every token competes with the agent's own output budget |
| `guardrails.md` | standing rules | ALWAYS ON -- same cost |
| `LongTermMEMORY.md` | durable domain facts | ALWAYS ON -- same cost |
| `tool_descriptions/tool_guidance.md` | how to use the tools | ALWAYS ON -- same cost |
| `skills/<name>/SKILL.md` | a reusable workflow | loaded ON DEMAND via `load_skill`, only when invoked |
| `sub_agents/<name>/AGENT.md` | LoopAgent stage override (`solver`, `verifier`) or delegated `AgentTool` | stage override customizes `solver`/`verifier`; other names run in their OWN context |
| `scripts/<name>.py` | a deterministic helper | run on demand via `run_bash` |
| `.adk/hooks.json` | lifecycle and escalation hooks | ALWAYS REGISTERED, and acts INSIDE the ADK tool and `exit_loop` loop |

The three that are easy to get wrong:

`skills/` -- frontmatter needs `name:` matching the directory and a `description:`, or it is
dropped silently at load. Two further rules below decide whether it survives validation at all:
it must be named in the always-on text, and any helper it calls must travel with it.

`sub_agents/` -- frontmatter needs `description:`. `sub_agents/solver/AGENT.md` and
`sub_agents/verifier/AGENT.md` override the instructions of the two LoopAgent stages directly;
any other `sub_agents/<name>/AGENT.md` is bound as a delegated `AgentTool` running in its OWN
context and returning text.

`.adk/hooks.json` -- ONE JSON file, Claude-Code hook schema: a top-level "hooks" object maps
an event (`PreToolUse`, `PostToolUse`, `Stop`, `SessionStart`) to matcher entries. A `Stop`
hook matching `exit_loop` or a `PostToolUse` hook printing `{"decision":"block","reason":"..."}`
acts inside the tool/escalation loop and prevents premature loop termination when a deliverable
is missing or malformed.

THREE RULES THAT DISCARD THE WHOLE PROPOSAL. Validation runs before any rollout, and a harness
that fails it is thrown away entire.

1. EVERY SKILL NEEDS A TRIGGER LINE -- if you create or keep `skills/<name>/SKILL.md`, its
   directory name must appear VERBATIM in `systemprompt.md` or `guardrails.md`, in the SAME edit:

       When <the situation this skill handles>, load the `<name>` skill.

2. NO TEMPLATE BRACES IN ALWAYS-ON TEXT. Use a single brace, or words.

3. EVERY SCRIPT YOU NAME MUST EXIST. A `scripts/<file>.py` invoked from a SKILL.md or from
   always-on text must ship in the same proposal, or at `skills/<skill-name>/scripts/<file>.py`.

THE DELIVERABLE'S SHAPE BELONGS TO THE REQUEST, NOT TO YOU.

# HOW TO MAKE A CHANGE

The harness is in `context/harness/`, and it is YOURS TO EDIT.
- Files you do not touch are inherited unchanged.
- Delete a file to remove that component.
- If the evidence does not argue for any change, leave everything as it is and say KEEP_BASE
  in your notes.

Then reply with ONE `<notes>` block and nothing else."""

_TTS_SURFACE_HEAD_ADK = """\
# EDITABLE HARNESS SURFACE

You may write ONLY these paths. Anything else is off the editable surface: it is dropped before
your work is used, however good it is.

| path | component | when it is in context |
|---|---|---|
| `systemprompt.md` | c1 persona/procedure | ALWAYS ON -- every token competes with the output budget |
| `guardrails.md` | c6 always-on rules | ALWAYS ON -- same cost |
| `LongTermMEMORY.md` | c3 durable domain facts | ALWAYS ON -- same cost |
| `tool_descriptions/tool_guidance.md` | c4 tool prose | ALWAYS ON -- same cost |
| `skills/<name>/SKILL.md` | c6 reusable workflow | loaded ON DEMAND via `load_skill` |
| `sub_agents/<name>/AGENT.md` | c5 LoopAgent stage override (`solver`, `verifier`) or delegated `AgentTool` | customizes LoopAgent stages or runs ON DEMAND in its OWN context |
| `scripts/<name>.py` | c7 deterministic helper | run ON DEMAND via `run_bash` as `python3 scripts/<name>.py` |
| `.adk/hooks.json` | c8 lifecycle/escalation hooks | fire INSIDE the ADK tool and `exit_loop` loop |"""


def register_adk_surface() -> None:
    """Register `ADK_SURFACE` into `harness2.core.prompts` at runtime without modifying `harness2`."""
    if "adk" in HP._SURFACE_TEXTS:
        return
    adk_tts = "\n\n".join(
        [
            _TTS_SURFACE_HEAD_ADK,
            HP._surface_part(
                _SURFACE_ADK_MD, "THREE RULES THAT DISCARD", "THE DELIVERABLE'S SHAPE"
            ),
            HP._surface_part(_SURFACE_ADK_MD, "# HOW TO MAKE A CHANGE"),
        ]
    )
    HP._SURFACE_TEXTS["adk"] = HP._SurfaceText(
        surface_md=_SURFACE_ADK_MD,
        inv=HP._INV_TEMPLATE.replace(
            "__LOOP_LEVER__",
            "hook in `.adk/hooks.json` -- for example, blocking `exit_loop` in `Stop` "
            "before a {deliverable} exists, with a corrective reason",
        ),
        dom=HP._DOM_TEMPLATE.replace(
            "__TRIGGER_SENTENCE__",
            "A skill exists only once its name appears in the always-on text: write that "
            "trigger line in the same edit.",
        ),
        tts_surface=adk_tts,
    )


register_adk_surface()

_SUPPORTED_INSTALL_MODES = ("workspace_config", "workspace_agents_md", "home_dir_jinja")


class AdkSubstrate:
    """Google ADK `LoopAgent` substrate satisfying `harness2.substrate.base.Substrate`."""

    name = "adk"

    def __init__(self, launcher: Launcher | None = None):
        register_adk_surface()
        self._launcher = launcher

    def surface(self) -> SurfaceSpec:
        return ADK_SURFACE

    def materialise(self, tree_dir: Path, spec: BenchSpec, ctx: Mapping[str, str]) -> Install:
        mode = spec.install_mode
        if mode not in _SUPPORTED_INSTALL_MODES:
            raise ValueError(
                f"install_mode {mode!r} is not supported by AdkSubstrate; expected one of "
                f"{_SUPPORTED_INSTALL_MODES}"
            )
        tree_dir = Path(tree_dir)
        if not HT.is_tree(ADK_SURFACE, tree_dir):
            raise ValueError(
                f"{tree_dir} is not an ADK harness tree (no always-on file present)"
            )
        staged = Path(tempfile.mkdtemp(prefix=f"adapt-adk-{mode}-"))
        try:
            shutil.copytree(tree_dir, staged, dirs_exist_ok=True)
            for key, body in ctx.items():
                if key.startswith("file:"):
                    rel = HT.normalize_rel(key[len("file:") :])
                    p = staged / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(body, encoding="utf-8")

            composed = RT.compose_global_instruction(
                staged, {k: v for k, v in ctx.items() if not k.startswith("file:")}
            )
            out = staged / "_composed" / "global_instruction.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(composed, encoding="utf-8")
        except BaseException:
            shutil.rmtree(staged, ignore_errors=True)
            raise

        return Install(
            staging_dir=staged,
            fingerprint=RM.fingerprint(tree_dir),
            extras={
                "install_mode": mode,
                **{k: v for k, v in ctx.items() if not k.startswith("file:")},
            },
        )

    def run_argv(
        self,
        prompt_file: Path,
        workspace: Path,
        staged_dir: Path,
        events_out: Path,
        *,
        model: str = "",
        max_iterations: int | None = None,
    ) -> list[str]:
        """Reference CLI argv to execute `loop_agent.runtime` in an isolated subprocess."""
        return [
            sys.executable,
            "-m",
            "loop_agent.runtime",
            "--harness-dir",
            str(staged_dir),
            "--workspace",
            str(workspace),
            "--prompt-file",
            str(prompt_file),
            "--events-out",
            str(events_out),
            "--model",
            model or RT.adk_model(),
            "--max-iterations",
            str(
                int(
                    max_iterations
                    if max_iterations is not None
                    else RT.adk_max_iterations()
                )
            ),
        ]

    def run_env(
        self, base_env: Mapping[str, str] | None = None, *, model: str = ""
    ) -> dict[str, str]:
        RT.load_dotenv()
        keep = (
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_GENERATIVE_AI_API_KEY",
            "ADK_API_KEY",
            "HARNESS2_ADK_API_KEY",
        ) + config.provider_keys(model or config.SOLVER_MODEL_OC)
        env = config.scrub_env(base_env, keep=keep)
        if config.VERTEX_PROJECT or os.environ.get("GOOGLE_CLOUD_PROJECT"):
            proj = os.environ.get("GOOGLE_CLOUD_PROJECT") or config.VERTEX_PROJECT
            loc = os.environ.get("GOOGLE_CLOUD_LOCATION") or config.VERTEX_REGION
            env.setdefault("GOOGLE_CLOUD_PROJECT", proj)
            env.setdefault("GOOGLE_CLOUD_LOCATION", loc)
            if not env.get("GOOGLE_API_KEY") and not env.get("GEMINI_API_KEY"):
                env.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
        return env

    def execute(
        self, task_ref: str, install: Install, run_ids: list[str], timeout: int
    ) -> list[Path]:
        if self._launcher is None:
            raise RuntimeError(
                "AdkSubstrate was constructed without a launcher body; supply a launcher or "
                "use AdkBenchmarkAdapter"
            )
        return self._launcher(
            RunRequest(task_ref=task_ref, run_ids=tuple(run_ids), timeout=int(timeout)), install
        )

    def read_trajectory(
        self, stream_path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 1000
    ) -> str:
        return RT.render_events(
            Path(stream_path), obs_trunc=obs_trunc, max_chars=max_chars, args_trunc=args_trunc
        )

    def preflight(self) -> None:
        RT.load_adk_classes()


DEMO_STOP_HOOK = {
    "hooks": {
        "Stop": [
            {
                "matcher": "exit_loop",
                "hooks": [
                    {
                        "type": "command",
                        "command": (
                            'python3 -c "import json, pathlib, sys; '
                            "p = pathlib.Path('answer.txt'); "
                            "ok = p.is_file() and p.read_text().strip().startswith('Total:'); "
                            "sys.stdout.write('' if ok else json.dumps("
                            "{'decision': 'block', 'reason': 'answer.txt missing or invalid'}))"
                            '"'
                        ),
                    }
                ],
            }
        ]
    }
}


class AdkBenchmarkAdapter:
    """Decorator/Proxy wrapping any `harness2` `BenchmarkAdapter` to run on `AdkSubstrate`.

    Task discovery, domain grouping, prompt overlays, deliverable extraction, and benchmark
    scoring (`score()`) are delegated to `inner` without touching `harness2` source files.
    """

    def __init__(
        self,
        inner: Any,
        *,
        model: str | Any = "",
        max_iterations: int | None = None,
    ):
        register_adk_surface()
        self.inner = inner
        self.model = model or (
            "local/deterministic" if inner.spec.name == "demo" else RT.adk_model()
        )
        self.max_iterations = (
            int(max_iterations) if max_iterations is not None else RT.adk_max_iterations()
        )
        self.substrate = AdkSubstrate(launcher=self._launch)
        self.spec: BenchSpec = replace(
            inner.spec,
            install_mode="workspace_config",
            allow_subagents=True,
            reserved_scripts=frozenset(),
        )
        self.execution_count = 0
        self._base_dir: Path | None = None

    def tasks(self, split: str = "all") -> list[str]:
        return self.inner.tasks(split)

    def domain_of(self, task_id: str) -> str:
        return self.inner.domain_of(task_id)

    def reference_domain_order(self) -> list[str]:
        fn = getattr(self.inner, "reference_domain_order", None)
        return fn() if callable(fn) else sorted({self.domain_of(t) for t in self.tasks()})

    def prompts_for(self, task_id: str) -> BenchPrompts:
        fn = getattr(self.inner, "prompts_for", None)
        return fn(task_id) if callable(fn) else self.spec.prompts

    def task_prompt(self, task_id: str) -> str:
        return self.inner.task_prompt(task_id)

    def task_files(self, task_id: str) -> Path | None:
        return self.inner.task_files(task_id)

    def base_harness(self) -> Path:
        if self._base_dir is not None and HT.is_tree(ADK_SURFACE, self._base_dir):
            return self._base_dir
        target = self.spec.runs_root / f"_adk_base_harness_{os.getpid()}"
        base_tree: HT.Tree = {name: "" for name in ADK_SURFACE.always_on}
        try:
            inner_base = self.inner.base_harness()
            inner_surface = self.inner.substrate.surface()
            raw_tree = HT.read_tree(inner_surface, inner_base)
            for rel, content in raw_tree.items():
                if rel == "AGENTS.md":
                    base_tree["systemprompt.md"] = content
                elif HT.is_allowed(ADK_SURFACE, rel):
                    base_tree[rel] = content
        except Exception:
            if self.spec.name == "lab":
                from harness2.benchmarks.lab import CODEX_BASE_BODY

                base_tree["systemprompt.md"] = CODEX_BASE_BODY

        if self.spec.name == "demo":
            base_tree["sub_agents/solver/AGENT.md"] = (
                "---\ndescription: Ledger solver stage inside LoopAgent\n---\n\n"
                "Run `python3 scripts/summarize.py ledger.csv > answer.txt` to compute the total.\n"
            )
            base_tree["sub_agents/verifier/AGENT.md"] = (
                "---\ndescription: Deliverable verifier stage inside LoopAgent\n---\n\n"
                "Read `answer.txt` and call `exit_loop` once `Total:` is present.\n"
            )
            base_tree[".adk/hooks.json"] = json.dumps(DEMO_STOP_HOOK, indent=2) + "\n"

        HT.write_tree(ADK_SURFACE, base_tree, target, stubs=(self.spec.name != "demo"))
        self._base_dir = target
        return target

    def validate_tree(self, harness: HT.Tree) -> None:
        if self.spec.name == "lab":
            body = HT.always_on_text(ADK_SURFACE, harness)
            errs = []
            if "output/" not in body:
                errs.append(
                    "always-on text must say deliverables go to `output/<filename>` explicitly"
                )
            if "read_doc.py" not in body:
                errs.append(
                    "always-on text must say .docx/.pdf/.pptx/.xlsx are readable via "
                    "`python3 scripts/read_doc.py <path>`"
                )
            if errs:
                raise ValueError(" | ".join(errs))
            return
        if self.spec.name.startswith("wb-"):
            return
        self.inner.validate_tree(harness)

    def _launch(self, req: RunRequest, install: Install) -> list[Path]:
        dirs: list[Path] = []
        for run_id in req.run_ids:
            if self.spec.name == "demo":
                run_dir = self.inner.root / "executions" / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                src = self.task_files(req.task_ref)
                if src and src.is_dir():
                    for f in src.iterdir():
                        if f.is_file():
                            shutil.copy2(f, run_dir / f.name)
                events_path = run_dir / "adk_events.jsonl"
                RT.run_loop_rollout(
                    install.staging_dir,
                    run_dir,
                    self.task_prompt(req.task_ref),
                    events_path,
                    model=self.model,
                    max_iterations=self.max_iterations,
                )
                rendered = self.substrate.read_trajectory(
                    events_path, obs_trunc=50_000, max_chars=200_000
                )
                (run_dir / "trajectory.md").write_text(rendered + "\n", encoding="utf-8")
                dirs.append(run_dir)
            elif self.spec.name == "jb":
                label = run_id.replace("/", "__")
                out_dir = self.inner._out_dir(req.task_ref, label)
                traj_dir = self.inner._traj_dir(req.task_ref, label)
                out_dir.mkdir(parents=True, exist_ok=True)
                traj_dir.mkdir(parents=True, exist_ok=True)
                ws = Path(tempfile.mkdtemp(prefix=f"jb_adk_{label}_"))
                try:
                    shutil.copytree(
                        self.inner._ensure_store_task(req.task_ref), ws / "task_folder"
                    )
                    (ws / "output").mkdir(parents=True, exist_ok=True)
                    events_path = traj_dir / "adk_events.jsonl"
                    RT.run_loop_rollout(
                        install.staging_dir,
                        ws,
                        self.task_prompt(req.task_ref),
                        events_path,
                        model=self.model,
                        max_iterations=self.max_iterations,
                    )
                    if (ws / "output").is_dir():
                        shutil.copytree(ws / "output", out_dir, dirs_exist_ok=True)
                finally:
                    shutil.rmtree(ws, ignore_errors=True)
                dirs.append(out_dir)
            else:
                run_dir = self.spec.runs_root / "adk_runs" / run_id.replace("/", "__")
                ws = run_dir / "workspace"
                (ws / "output").mkdir(parents=True, exist_ok=True)
                src = self.task_files(req.task_ref)
                if src and src.is_dir():
                    shutil.copytree(src, ws / "documents", dirs_exist_ok=True)
                events_path = run_dir / "adk_events.jsonl"
                RT.run_loop_rollout(
                    install.staging_dir,
                    ws,
                    self.task_prompt(req.task_ref),
                    events_path,
                    model=self.model,
                    max_iterations=self.max_iterations,
                )
                if (ws / "output").is_dir():
                    shutil.copytree(ws / "output", run_dir / "output", dirs_exist_ok=True)
                dirs.append(run_dir)
        return dirs

    def execute(
        self, task_id: str, harness_dir: Path | None, run_id: str, *, timeout: int | None = None
    ) -> Path:
        self.execution_count += 1
        if hasattr(self.inner, "execution_count"):
            self.inner.execution_count += 1
        src = Path(harness_dir) if harness_dir is not None else self.base_harness()
        install = self.substrate.materialise(src, self.spec, {})
        try:
            return self.substrate.execute(task_id, install, [run_id], int(timeout or 60))[0]
        finally:
            shutil.rmtree(install.staging_dir, ignore_errors=True)

    def delivered(self, run_dir: Path) -> bool:
        return self.inner.delivered(run_dir)

    def trajectory(self, run_dir: Path, *, obs_trunc: int) -> str:
        events = Path(run_dir) / "adk_events.jsonl"
        if events.is_file():
            return self.substrate.read_trajectory(
                events, obs_trunc=obs_trunc, max_chars=2_000_000
            )
        return self.inner.trajectory(run_dir, obs_trunc=obs_trunc)

    def deliverable_files(self, run_dir: Path) -> dict[str, str]:
        return self.inner.deliverable_files(run_dir)

    def deliverable_text(self, run_dir: Path) -> str:
        return self.inner.deliverable_text(run_dir)

    def score(self, run_dir: Path, task_id: str) -> dict | None:
        return self.inner.score(run_dir, task_id)

    def preflight(self) -> None:
        self.substrate.preflight()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def wrap_adapter(
    adapter: Any, *, model: str | Any = "", max_iterations: int | None = None
) -> AdkBenchmarkAdapter:
    """Wrap any `harness2` `BenchmarkAdapter` instance with `AdkBenchmarkAdapter`."""
    return AdkBenchmarkAdapter(adapter, model=model, max_iterations=max_iterations)


def make_adk_adapter(
    bench: str,
    subset: str | None = None,
    *,
    split_file: Path | None = None,
    root: Path | None = None,
    model: str | Any = "",
    max_iterations: int | None = None,
) -> AdkBenchmarkAdapter:
    """Construct a `harness2` benchmark or demo adapter wrapped with `AdkBenchmarkAdapter`."""
    if bench == "demo":
        from harness2.demo import DemoAdapter

        inner = DemoAdapter(root or Path("runs/demo_adk"))
    else:
        from harness2.cli._common import make_adapter

        inner = make_adapter(bench, subset, "oc", split_file=split_file)
    return wrap_adapter(inner, model=model, max_iterations=max_iterations)


__all__ = [
    "ADK_SURFACE",
    "AdkBenchmarkAdapter",
    "AdkSubstrate",
    "DEMO_STOP_HOOK",
    "make_adk_adapter",
    "register_adk_surface",
    "wrap_adapter",
]
