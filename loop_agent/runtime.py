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

"""Layer 1: Pure Google ADK `LoopAgent` runtime (zero imports from `harness2`).

Operates directly on a harness directory on disk:
  - Always-on files (`systemprompt.md`, `guardrails.md`, `LongTermMEMORY.md`,
    `tool_descriptions/tool_guidance.md`) compose the shared instruction body.
  - `skills/<name>/SKILL.md` are loaded on demand via the `load_skill` tool.
  - `sub_agents/solver/AGENT.md` (or `drafter`) and `sub_agents/verifier/AGENT.md`
    override the stage instructions of the 2-stage `LoopAgent` (`solver` -> `verifier`),
    while any other `sub_agents/<name>/AGENT.md` is bound as a delegated `AgentTool`.
  - `scripts/<name>.py` are synced into the rollout workspace and invoked via `run_bash`.
  - `.adk/hooks.json` dispatches `PreToolUse`, `PostToolUse`, `Stop` (inside `exit_loop`
    before escalating), and `SessionStart` inside the ADK tool loop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import warnings
from collections.abc import AsyncGenerator, Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]

ALWAYS_ON_FILES = (
    "systemprompt.md",
    "guardrails.md",
    "LongTermMEMORY.md",
    "tool_descriptions/tool_guidance.md",
)

_SCRUB_KEYS = frozenset(
    {
        "HARNESS2_JUDGE_API_KEY",
        "HARNESS2_BRIDGE_KEY",
        "HARNESS2_CODEX_API_KEY",
        "HARNESS2_ADK_API_KEY",
        "ADK_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_BEDROCK_BASE_URL",
        "CLAUDE_CODE_USE_BEDROCK",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "SLACK_BOT_TOKEN",
        "SLACK_WEBHOOK_URL",
        "JIRA_API_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_BEARER_TOKEN_BEDROCK",
    }
)

_TOKEN = re.compile(r"\{\{\s*(\w+)\s*\}\}")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)

DEFAULT_SOLVER_STAGE_INSTRUCTION = (
    "You are the `solver` stage inside an iterative Google ADK LoopAgent.\n"
    "Execute the requested work in the workspace using your tools (`load_skill`, `run_bash`, "
    "`read_file`, `write_file`). If a previous `verifier` turn reported issues or if a hook "
    "blocked completion, address every reported defect before finishing your turn."
)

DEFAULT_VERIFIER_STAGE_INSTRUCTION = (
    "You are the `verifier` stage inside an iterative Google ADK LoopAgent.\n"
    "Inspect the deliverables and workspace state produced by `solver` using your tools. "
    "When the requested deliverables exist, are non-empty, and satisfy the task instructions, "
    "call `exit_loop(summary=...)` to finish the loop. If anything is missing or defective, "
    "DO NOT call `exit_loop`; state the exact defects so `solver` can fix them on the next "
    "iteration."
)


warnings.filterwarnings(
    "ignore",
    message=".*GOOGLE_GENAI_USE_VERTEXAI is deprecated.*",
    category=DeprecationWarning,
)

_DEFAULT_DOTENV_LOADED = False


def _sync_genai_env(
    *, override: bool = False, loaded: Mapping[str, str] | None = None
) -> None:
    """Normalize API key and Vertex AI environment variables for `google-adk` / `google-genai`."""
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "":
        os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)

    if override and loaded is not None:
        explicit_key = (
            loaded.get("GOOGLE_API_KEY")
            or loaded.get("GEMINI_API_KEY")
            or loaded.get("ADK_API_KEY")
            or loaded.get("HARNESS2_ADK_API_KEY")
            or loaded.get("GOOGLE_GENERATIVE_AI_API_KEY")
            or ""
        ).strip()
        if explicit_key:
            os.environ["GOOGLE_API_KEY"] = explicit_key
            os.environ.pop("GEMINI_API_KEY", None)
        elif any(
            k in loaded
            for k in (
                "GOOGLE_API_KEY",
                "GEMINI_API_KEY",
                "ADK_API_KEY",
                "HARNESS2_ADK_API_KEY",
            )
        ):
            os.environ["GOOGLE_API_KEY"] = ""
            os.environ.pop("GEMINI_API_KEY", None)

    api_key = (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("ADK_API_KEY")
        or os.environ.get("HARNESS2_ADK_API_KEY")
        or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
        or ""
    ).strip()
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        if "GEMINI_API_KEY" in os.environ:
            os.environ.pop("GEMINI_API_KEY", None)

    vertex_project = (
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("HARNESS2_VERTEX_PROJECT")
        or ""
    ).strip()
    if vertex_project:
        if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
            os.environ["GOOGLE_CLOUD_PROJECT"] = vertex_project
        if not os.environ.get("GOOGLE_CLOUD_LOCATION"):
            os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
        if not api_key and not os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"):
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"


def load_dotenv(
    path: Path | str | None = None, *, override: bool = False
) -> dict[str, str]:
    """Load `KEY=VALUE` settings (LLM API keys, ADK model, limits) from `.env` into `os.environ`.

    Shell environment variables take precedence unless `override=True`.
    """
    global _DEFAULT_DOTENV_LOADED
    if path is None and not override and _DEFAULT_DOTENV_LOADED:
        _sync_genai_env()
        return {}

    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        env_file = os.environ.get("LOOP_AGENT_ENV_FILE", "").strip()
        if env_file:
            candidates.append(Path(env_file))
        cwd_env = Path.cwd() / ".env"
        root_env = ROOT_DIR / ".env"
        candidates.append(cwd_env)
        if root_env != cwd_env:
            candidates.append(root_env)

    loaded: dict[str, str] = {}
    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            key, _, value = line.partition("=")
            key = key.strip()
            if not key or not all(c.isalnum() or c == "_" for c in key):
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            loaded[key] = value
            if key == "GOOGLE_GENAI_USE_VERTEXAI" and not value:
                if override:
                    os.environ.pop(key, None)
                continue
            if override:
                os.environ[key] = value
            elif not os.environ.get(key) and (value or key not in os.environ):
                os.environ[key] = value
        break

    if path is None and not override:
        _DEFAULT_DOTENV_LOADED = True
    _sync_genai_env(override=override, loaded=loaded)
    return loaded


load_dotenv()


def adk_model(default: str = "gemini-3.1-pro-preview") -> str:
    """Resolve the ADK model name from `.env` / environment variables without importing `harness2`."""
    load_dotenv()
    raw = (
        os.environ.get("HARNESS2_ADK_MODEL")
        or os.environ.get("ADK_MODEL")
        or os.environ.get("HARNESS2_SOLVER_MODEL")
        or default
    ).strip()
    if "/" in raw and raw != "local/deterministic":
        return raw.split("/", 1)[-1]
    return raw or default


def adk_max_iterations(default: int = 3) -> int:
    """Resolve the maximum LoopAgent iterations from `.env` / environment variables."""
    load_dotenv()
    raw = (
        os.environ.get("HARNESS2_ADK_MAX_ITERATIONS")
        or os.environ.get("ADK_MAX_ITERATIONS")
        or ""
    ).strip()
    if not raw:
        return int(default)
    return int(raw)


def adk_command_timeout(default: int = 120) -> int:
    """Resolve the per-command `run_bash` timeout in seconds from `.env` / environment variables."""
    load_dotenv()
    raw = (
        os.environ.get("HARNESS2_ADK_COMMAND_TIMEOUT")
        or os.environ.get("ADK_COMMAND_TIMEOUT")
        or ""
    ).strip()
    if not raw:
        return int(default)
    return int(raw)


def adk_max_concurrent_commands(default: int = 8) -> int:
    """Resolve the maximum number of concurrent `run_bash`/hook subprocesses across sessions."""
    load_dotenv()
    raw = (
        os.environ.get("HARNESS2_ADK_MAX_CONCURRENT_COMMANDS")
        or os.environ.get("ADK_MAX_CONCURRENT_COMMANDS")
        or ""
    ).strip()
    if not raw:
        return max(1, int(default))
    return max(1, int(raw))


def adk_isolate_sessions(default: bool = False) -> bool:
    """Resolve whether per-user/per-session workspace isolation is enabled."""
    load_dotenv()
    raw = (
        os.environ.get("HARNESS2_ADK_ISOLATE_SESSIONS")
        or os.environ.get("ADK_ISOLATE_SESSIONS")
        or ""
    ).strip().lower()
    if not raw:
        return bool(default)
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return bool(default)


def adk_session_service_uri(default: str = "") -> str:
    """Resolve the persistent ADK `session_service_uri` from `.env` / environment variables."""
    load_dotenv()
    return (
        os.environ.get("HARNESS2_ADK_SESSION_SERVICE_URI")
        or os.environ.get("ADK_SESSION_SERVICE_URI")
        or default
    ).strip()


_SUBPROCESS_SEM_LOCK = threading.Lock()
_SUBPROCESS_SEMAPHORES: dict[int, threading.BoundedSemaphore] = {}


def _get_subprocess_semaphore(limit: int | None = None) -> threading.BoundedSemaphore:
    cap = int(limit) if limit is not None else adk_max_concurrent_commands()
    cap = max(1, cap)
    with _SUBPROCESS_SEM_LOCK:
        sem = _SUBPROCESS_SEMAPHORES.get(cap)
        if sem is None:
            sem = threading.BoundedSemaphore(cap)
            _SUBPROCESS_SEMAPHORES[cap] = sem
        return sem


def resolve_session_workspace(
    base_workspace: Path,
    tool_context: Any = None,
    *,
    isolate_sessions: bool = False,
) -> Path:
    """Resolve the workspace directory for the current tool/callback invocation.

    When `isolate_sessions=True` and `tool_context` provides `user_id`/`session.id`,
    returns `<base_workspace>/<user_id>/<session_id>` so concurrent multi-user sessions
    never collide on files or temporary directories.
    """
    base = Path(base_workspace).resolve()
    if not isolate_sessions or tool_context is None:
        return base
    user_id = str(getattr(tool_context, "user_id", "") or "").strip()
    session_obj = getattr(tool_context, "session", None)
    session_id = str(getattr(session_obj, "id", "") or "").strip()
    if not user_id and not session_id:
        return base
    safe_user = re.sub(r"[^a-zA-Z0-9_.-]", "_", user_id or "anonymous")
    safe_session = re.sub(r"[^a-zA-Z0-9_.-]", "_", session_id or "default")
    target = (base / safe_user / safe_session).resolve()
    if target != base and base not in target.parents:
        return base
    return target


def scrub_runtime_env(
    base_env: Mapping[str, str] | None = None,
    *,
    keep: Iterable[str] = (),
    workspace: Path | None = None,
) -> dict[str, str]:
    """Return a copy of `base_env` with host/judge secrets removed and optional per-session TMPDIR."""
    src = os.environ if base_env is None else base_env
    drop = _SCRUB_KEYS - set(keep)
    env = {k: v for k, v in src.items() if k not in drop}
    if workspace is not None:
        tmp_dir = Path(workspace).resolve() / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(tmp_dir)
        env["TMP"] = str(tmp_dir)
        env["TEMP"] = str(tmp_dir)
    return env


def _interp(text: str, ctx: Mapping[str, str] | None) -> str:
    if not ctx:
        return text
    return _TOKEN.sub(lambda m: str(ctx.get(m.group(1), "")), text)


def _strip_comments(text: str) -> str:
    return _COMMENT.sub("", text).strip()


def _parse_markdown_component(raw: str) -> tuple[str, str]:
    """Return (description, body_without_frontmatter)."""
    desc = "(no description)"
    body = raw
    m = _FRONTMATTER_RE.match(raw)
    if m:
        fm, body = m.group(1), m.group(2)
        dm = re.search(r"^description:\s*(.+)$", fm, re.M)
        if dm:
            desc = dm.group(1).strip()
    else:
        dm = re.search(r"^description:\s*(.+)$", raw[:600], re.M)
        if dm:
            desc = dm.group(1).strip()
    return desc, body.strip()


def compose_global_instruction(staged: Path, ctx: Mapping[str, str] | None = None) -> str:
    """Compose the always-on c1/c3/c4/c6 files and component index for an ADK LoopAgent run."""
    staged = Path(staged)
    parts: list[str] = []

    def _read(rel: str) -> str:
        p = staged / rel
        if not p.is_file():
            return ""
        return _strip_comments(_interp(p.read_text(encoding="utf-8", errors="replace"), ctx))

    sp = _read("systemprompt.md")
    if sp:
        parts.append(sp)
    gr = _read("guardrails.md")
    if gr:
        parts.append("# Operating Guardrails\n\n" + gr)
    tg = _read("tool_descriptions/tool_guidance.md")
    if tg:
        parts.append(tg)
    mem = _read("LongTermMEMORY.md")
    if mem:
        parts.append("# Long-Term Memory\n\n" + mem)

    skill_lines: list[str] = []
    for p in sorted(staged.glob("skills/*/SKILL.md"), key=lambda x: x.parent.name):
        desc, _ = _parse_markdown_component(p.read_text(encoding="utf-8", errors="replace"))
        skill_lines.append(f"- `{p.parent.name}`: {desc}")
    if skill_lines:
        parts.append(
            "# AVAILABLE SKILLS\nLoad on demand with `load_skill(name=...)` when triggered:\n"
            + "\n".join(skill_lines)
        )

    sub_lines: list[str] = []
    for p in sorted(staged.glob("sub_agents/*/AGENT.md"), key=lambda x: x.parent.name):
        if p.parent.name in ("solver", "drafter", "verifier"):
            continue
        desc, _ = _parse_markdown_component(p.read_text(encoding="utf-8", errors="replace"))
        sub_lines.append(f"- `{p.parent.name}`: {desc}")
    if sub_lines:
        parts.append(
            "# AVAILABLE DELEGATED SUBAGENTS\nInvoke as a tool when the situation matches:\n"
            + "\n".join(sub_lines)
        )

    scripts = sorted(
        p.name for p in staged.glob("scripts/*.py") if not p.name.startswith("_")
    )
    if scripts:
        parts.append(
            "# HELPER SCRIPTS\nAvailable in `./scripts/`: "
            + ", ".join(f"`scripts/{s}`" for s in scripts)
            + "\nRun with `run_bash(command='python3 scripts/<name>.py ...')`."
        )

    return "\n\n".join(p for p in parts if p.strip()) + "\n"


def _matcher_matches(matcher: str | None, tool_name: str) -> bool:
    if not matcher or matcher == "*":
        return True
    if matcher.lower() == tool_name.lower():
        return True
    try:
        return bool(re.search(matcher, tool_name, re.I))
    except re.error:
        return matcher in tool_name


def run_hook_event(
    staged: Path,
    workspace: Path,
    event: str,
    *,
    tool_name: str = "",
    tool_input: Any = None,
    tool_response: Any = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Execute matching `.adk/hooks.json` commands for `event`.

    Returns `{"blocked": bool, "reason": str, "outputs": list[str]}`.
    """
    hooks_path = Path(staged) / ".adk" / "hooks.json"
    if not hooks_path.is_file():
        return {"blocked": False, "reason": "", "outputs": []}
    try:
        doc = json.loads(hooks_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"blocked": False, "reason": "", "outputs": []}

    entries = (doc.get("hooks") or {}).get(event)
    if not isinstance(entries, list):
        return {"blocked": False, "reason": "", "outputs": []}

    payload = json.dumps(
        {
            "event": event,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": tool_response,
            "workspace": str(workspace),
        },
        default=str,
    )
    outputs: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if tool_name and not _matcher_matches(entry.get("matcher"), tool_name):
            continue
        for hook in entry.get("hooks") or ():
            if not isinstance(hook, dict) or hook.get("type") != "command":
                continue
            cmd = str(hook.get("command", "")).strip()
            if not cmd:
                continue
            try:
                with _get_subprocess_semaphore():
                    res = subprocess.run(
                        ["bash", "-c", cmd],
                        input=payload,
                        cwd=workspace,
                        env=scrub_runtime_env(workspace=workspace),
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
            except (OSError, subprocess.TimeoutExpired) as exc:
                outputs.append(f"hook error: {exc}")
                continue
            out = (res.stdout or "").strip()
            if out:
                outputs.append(out)
                for line in reversed(out.splitlines()):
                    line = line.strip()
                    if not line.startswith("{"):
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict) and parsed.get("decision") == "block":
                        reason = str(parsed.get("reason") or "blocked by .adk/hooks.json")
                        return {"blocked": True, "reason": reason, "outputs": outputs}
    return {"blocked": False, "reason": "", "outputs": outputs}


def sync_workspace_assets(staged: Path, workspace: Path) -> None:
    """Copy `scripts/*.py` and `skills/*/scripts/*` from `staged` into `workspace`."""
    staged, workspace = Path(staged), Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    scripts_src = staged / "scripts"
    if scripts_src.is_dir():
        for py in sorted(scripts_src.glob("*.py")):
            if py.name.startswith("_"):
                continue
            dst = workspace / "scripts" / py.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(py, dst)
    skills_src = staged / "skills"
    if skills_src.is_dir():
        for helper in sorted(skills_src.glob("*/scripts/*")):
            if not helper.is_file():
                continue
            rel = helper.relative_to(staged)
            dst = workspace / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(helper, dst)


def make_workspace_tools(
    staged: Path,
    workspace: Path,
    *,
    command_timeout: int | None = None,
    isolate_sessions: bool | None = None,
) -> dict[str, Callable[..., Any]]:
    """Build the Python functions exposed as ADK tools inside `LoopAgent` stages."""
    staged = Path(staged).resolve()
    workspace = Path(workspace).resolve()
    effective_timeout = (
        int(command_timeout) if command_timeout is not None else adk_command_timeout()
    )
    effective_isolate = (
        bool(isolate_sessions)
        if isolate_sessions is not None
        else adk_isolate_sessions(default=False)
    )
    if not effective_isolate:
        sync_workspace_assets(staged, workspace)

    def _ws_for(tool_context: Any = None) -> Path:
        ws = resolve_session_workspace(
            workspace, tool_context, isolate_sessions=effective_isolate
        )
        sync_workspace_assets(staged, ws)
        return ws

    def _safe_ws_path(rel_or_abs: str, ws: Path) -> Path:
        p = Path(rel_or_abs)
        target = p.resolve() if p.is_absolute() else (ws / p).resolve()
        if target != ws and ws not in target.parents:
            raise ValueError(f"path {rel_or_abs!r} is outside workspace {ws}")
        return target

    def load_skill(name: str, tool_context: Any = None) -> dict[str, Any]:
        """Load the on-demand SKILL.md instructions for skill `name`."""
        clean = Path(name).name
        skill_file = staged / "skills" / clean / "SKILL.md"
        if not skill_file.is_file():
            available = sorted(p.parent.name for p in staged.glob("skills/*/SKILL.md"))
            return {
                "status": "error",
                "error": f"skill {name!r} not found; available skills: {available}",
            }
        _ws_for(tool_context)
        return {
            "status": "ok",
            "skill": clean,
            "content": skill_file.read_text(encoding="utf-8", errors="replace"),
        }

    def _run_bash_sync(command: str, ws: Path) -> dict[str, Any]:
        try:
            with _get_subprocess_semaphore():
                res = subprocess.run(
                    ["bash", "-c", command],
                    cwd=ws,
                    env=scrub_runtime_env(workspace=ws),
                    capture_output=True,
                    text=True,
                    timeout=effective_timeout,
                )
            return {
                "status": "ok" if res.returncode == 0 else "error",
                "exit_code": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "status": "error",
                "exit_code": 124,
                "stdout": exc.stdout or "",
                "stderr": f"Command timed out after {effective_timeout}s",
            }
        except OSError as exc:
            return {"status": "error", "exit_code": 1, "stdout": "", "stderr": str(exc)}

    async def run_bash(command: str, tool_context: Any = None) -> dict[str, Any]:
        """Execute a bash command asynchronously in the session's workspace directory."""
        ws = _ws_for(tool_context)
        return await asyncio.to_thread(_run_bash_sync, command, ws)

    def read_file(path: str, tool_context: Any = None) -> dict[str, Any]:
        """Read a UTF-8 text file from the workspace."""
        ws = _ws_for(tool_context)
        try:
            target = _safe_ws_path(path, ws)
            if not target.is_file():
                return {"status": "error", "error": f"file not found: {path}"}
            return {
                "status": "ok",
                "path": path,
                "content": target.read_text(encoding="utf-8", errors="replace"),
            }
        except (OSError, ValueError) as exc:
            return {"status": "error", "error": str(exc)}

    def write_file(path: str, content: str, tool_context: Any = None) -> dict[str, Any]:
        """Write UTF-8 text `content` to `path` inside the workspace."""
        ws = _ws_for(tool_context)
        try:
            target = _safe_ws_path(path, ws)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return {"status": "ok", "path": path, "bytes_written": len(content.encode("utf-8"))}
        except (OSError, ValueError) as exc:
            return {"status": "error", "error": str(exc)}

    async def exit_loop(summary: str, tool_context: Any = None) -> dict[str, Any]:
        """Signal that verification passed and escalate out of the ADK LoopAgent."""
        ws = _ws_for(tool_context)
        hook_res = await asyncio.to_thread(
            run_hook_event,
            staged,
            ws,
            "Stop",
            tool_name="exit_loop",
            tool_input={"summary": summary},
        )
        if hook_res["blocked"]:
            return {
                "status": "blocked",
                "decision": "block",
                "reason": hook_res["reason"],
            }
        if tool_context is not None and getattr(tool_context, "actions", None) is not None:
            tool_context.actions.escalate = True
        return {"status": "ok", "escalated": True, "summary": summary}

    return {
        "load_skill": load_skill,
        "run_bash": run_bash,
        "read_file": read_file,
        "write_file": write_file,
        "exit_loop": exit_loop,
    }


def make_tool_callbacks(
    staged: Path,
    workspace: Path,
    *,
    isolate_sessions: bool | None = None,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Build async ADK `before_tool_callback` and `after_tool_callback` backed by `.adk/hooks.json`."""
    staged = Path(staged).resolve()
    workspace = Path(workspace).resolve()
    effective_isolate = (
        bool(isolate_sessions)
        if isolate_sessions is not None
        else adk_isolate_sessions(default=False)
    )

    def _ws_for(tool_context: Any = None) -> Path:
        ws = resolve_session_workspace(
            workspace, tool_context, isolate_sessions=effective_isolate
        )
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    async def before_tool_callback(
        tool: Any, args: dict[str, Any], tool_context: Any
    ) -> dict[str, Any] | None:
        tname = getattr(tool, "name", str(tool))
        ws = _ws_for(tool_context)
        res = await asyncio.to_thread(
            run_hook_event, staged, ws, "PreToolUse", tool_name=tname, tool_input=args
        )
        if res["blocked"]:
            return {
                "status": "error",
                "decision": "block",
                "error": f"Blocked by PreToolUse hook: {res['reason']}",
            }
        return None

    async def after_tool_callback(
        tool: Any,
        args: dict[str, Any],
        tool_context: Any,
        tool_response: dict[str, Any],
    ) -> dict[str, Any] | None:
        tname = getattr(tool, "name", str(tool))
        ws = _ws_for(tool_context)
        res = await asyncio.to_thread(
            run_hook_event,
            staged,
            ws,
            "PostToolUse",
            tool_name=tname,
            tool_input=args,
            tool_response=tool_response,
        )
        if res["blocked"]:
            base = (
                dict(tool_response)
                if isinstance(tool_response, dict)
                else {"output": tool_response}
            )
            base["decision"] = "block"
            base["hook_feedback"] = f"Blocked by PostToolUse hook: {res['reason']}"
            return base
        return None

    return before_tool_callback, after_tool_callback


def load_adk_classes() -> dict[str, Any]:
    """Lazy-import `google.adk` and `google.genai.types` with a clear installation message."""
    try:
        logging.getLogger("google_adk").setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from google.adk.agents import LlmAgent, LoopAgent
            from google.adk.models import BaseLlm, LlmRequest, LlmResponse
            from google.adk.runners import InMemoryRunner
            from google.adk.tools import AgentTool, FunctionTool, ToolContext
            from google.genai import types
        return {
            "LlmAgent": LlmAgent,
            "LoopAgent": LoopAgent,
            "BaseLlm": BaseLlm,
            "LlmRequest": LlmRequest,
            "LlmResponse": LlmResponse,
            "InMemoryRunner": InMemoryRunner,
            "AgentTool": AgentTool,
            "FunctionTool": FunctionTool,
            "ToolContext": ToolContext,
            "types": types,
        }
    except ImportError as exc:
        raise RuntimeError(
            "google-adk is required for loop_agent; install it with `pip install google-adk`"
        ) from exc


def make_deterministic_loop_llm(workspace: Path, *, staged: Path | None = None):
    """Create a deterministic `google.adk.models.BaseLlm` instance for offline `LoopAgent` runs."""
    adk = load_adk_classes()
    BaseLlm = adk["BaseLlm"]
    LlmRequest = adk["LlmRequest"]
    LlmResponse = adk["LlmResponse"]
    types = adk["types"]

    class DeterministicLoopLlm(BaseLlm):
        model: str = "local/deterministic"
        workspace_dir: str = str(workspace)
        staged_dir: str = str(staged or workspace)

        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[Any, None]:
            usage = types.GenerateContentResponseUsageMetadata(
                prompt_token_count=32,
                candidates_token_count=16,
                total_token_count=48,
            )
            sys_inst = str(getattr(llm_request.config, "system_instruction", "") or "")
            last_content = llm_request.contents[-1] if llm_request.contents else None
            last_parts = (last_content.parts or []) if last_content else []
            fn_responses = [
                p.function_response
                for p in last_parts
                if getattr(p, "function_response", None) is not None
            ]

            is_verifier = (
                "you are the `verifier` stage" in sys_inst.lower()
                or "role: verifier" in sys_inst.lower()
            )

            if is_verifier:
                if not fn_responses:
                    yield LlmResponse(
                        content=types.Content(
                            role="model",
                            parts=[
                                types.Part.from_function_call(
                                    name="read_file", args={"path": "answer.txt"}
                                )
                            ],
                        ),
                        usage_metadata=usage,
                    )
                    return
                last_fn = fn_responses[-1]
                if last_fn.name == "read_file":
                    resp_dict = last_fn.response if isinstance(last_fn.response, dict) else {}
                    if resp_dict.get("status") == "ok" and resp_dict.get("content", "").strip():
                        yield LlmResponse(
                            content=types.Content(
                                role="model",
                                parts=[
                                    types.Part.from_function_call(
                                        name="exit_loop",
                                        args={
                                            "summary": "Verified answer.txt: "
                                            + resp_dict["content"].strip()
                                        },
                                    )
                                ],
                            ),
                            usage_metadata=usage,
                        )
                        return
                    yield LlmResponse(
                        content=types.Content(
                            role="model",
                            parts=[
                                types.Part.from_text(
                                    text="Deliverable answer.txt is missing or empty; "
                                    "requesting another solver iteration."
                                )
                            ],
                        ),
                        usage_metadata=usage,
                    )
                    return
                resp_dict = last_fn.response if isinstance(last_fn.response, dict) else {}
                if resp_dict.get("decision") == "block":
                    yield LlmResponse(
                        content=types.Content(
                            role="model",
                            parts=[
                                types.Part.from_text(
                                    text=f"Stop hook blocked exit_loop: {resp_dict.get('reason')}"
                                )
                            ],
                        ),
                        usage_metadata=usage,
                    )
                else:
                    yield LlmResponse(
                        content=types.Content(
                            role="model",
                            parts=[
                                types.Part.from_text(
                                    text="Verification passed; exiting LoopAgent."
                                )
                            ],
                        ),
                        usage_metadata=usage,
                    )
                return

            if not fn_responses:
                ws = Path(self.workspace_dir)
                cmd = (
                    f"{sys.executable} scripts/summarize.py ledger.csv > answer.txt "
                    "&& cat answer.txt"
                )
                if not (ws / "ledger.csv").is_file():
                    csvs = sorted(ws.rglob("*.csv"))
                    if csvs:
                        rel_csv = csvs[0].relative_to(ws)
                        cmd = (
                            f"{sys.executable} scripts/summarize.py {rel_csv} > answer.txt "
                            "&& cat answer.txt"
                        )
                yield LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part.from_function_call(
                                name="run_bash",
                                args={"command": cmd},
                            )
                        ],
                    ),
                    usage_metadata=usage,
                )
                return

            last_fn = fn_responses[-1]
            resp_dict = last_fn.response if isinstance(last_fn.response, dict) else {}
            stdout = str(resp_dict.get("stdout") or "").strip()
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=f"Completed solver pass. Output: {stdout or '(none)'}"
                        )
                    ],
                ),
                usage_metadata=usage,
            )

    return DeterministicLoopLlm()


def build_loop_agent(
    staged: Path,
    workspace: Path,
    *,
    model: str | Any = "",
    max_iterations: int | None = None,
    ctx: Mapping[str, str] | None = None,
    isolate_sessions: bool | None = None,
):
    """Construct the Google ADK `LoopAgent` from a harness directory on disk."""
    adk = load_adk_classes()
    LlmAgent = adk["LlmAgent"]
    LoopAgent = adk["LoopAgent"]
    AgentTool = adk["AgentTool"]

    staged = Path(staged).resolve()
    workspace = Path(workspace).resolve()
    resolved_model: Any = model or adk_model()
    if isinstance(resolved_model, str) and resolved_model == "local/deterministic":
        resolved_model = make_deterministic_loop_llm(workspace, staged=staged)
    elif isinstance(resolved_model, str) and "/" in resolved_model:
        resolved_model = resolved_model.split("/", 1)[-1]

    max_iters = int(max_iterations if max_iterations is not None else adk_max_iterations())
    global_body = compose_global_instruction(staged, ctx)

    ws_tools = make_workspace_tools(
        staged, workspace, isolate_sessions=isolate_sessions
    )
    before_cb, after_cb = make_tool_callbacks(
        staged, workspace, isolate_sessions=isolate_sessions
    )

    solver_override = ""
    verifier_override = ""
    delegated_tools: list[Any] = []

    sub_dir = staged / "sub_agents"
    if sub_dir.is_dir():
        for agent_md in sorted(sub_dir.glob("*/AGENT.md"), key=lambda p: p.parent.name):
            role_name = agent_md.parent.name
            raw = agent_md.read_text(encoding="utf-8", errors="replace")
            desc, body = _parse_markdown_component(raw)
            body = _strip_comments(_interp(body, ctx))
            if role_name in ("solver", "drafter"):
                solver_override = body
            elif role_name == "verifier":
                verifier_override = body
            else:
                safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", role_name)
                sub_agent = LlmAgent(
                    name=safe_name,
                    model=resolved_model,
                    description=desc,
                    instruction=body or f"You are the `{role_name}` delegated subagent.",
                    tools=[
                        ws_tools["load_skill"],
                        ws_tools["run_bash"],
                        ws_tools["read_file"],
                        ws_tools["write_file"],
                    ],
                    before_tool_callback=before_cb,
                    after_tool_callback=after_cb,
                )
                delegated_tools.append(AgentTool(agent=sub_agent))

    solver_stage = solver_override or DEFAULT_SOLVER_STAGE_INSTRUCTION
    verifier_stage = verifier_override or DEFAULT_VERIFIER_STAGE_INSTRUCTION

    solver_instruction = (
        f"{global_body}\n\n# LOOP STAGE: SOLVER\n\n{solver_stage}".strip()
        if global_body.strip()
        else f"# LOOP STAGE: SOLVER\n\n{solver_stage}"
    )
    verifier_instruction = (
        f"{global_body}\n\n# LOOP STAGE: VERIFIER (ROLE: VERIFIER)\n\n{verifier_stage}".strip()
        if global_body.strip()
        else f"# LOOP STAGE: VERIFIER (ROLE: VERIFIER)\n\n{verifier_stage}"
    )

    solver_agent = LlmAgent(
        name="solver",
        model=resolved_model,
        description="Primary task solver stage inside the LoopAgent.",
        instruction=solver_instruction,
        tools=[
            ws_tools["load_skill"],
            ws_tools["run_bash"],
            ws_tools["read_file"],
            ws_tools["write_file"],
            *delegated_tools,
        ],
        before_tool_callback=before_cb,
        after_tool_callback=after_cb,
    )

    verifier_agent = LlmAgent(
        name="verifier",
        model=resolved_model,
        description="Verification stage that checks deliverables and calls exit_loop when complete.",
        instruction=verifier_instruction,
        tools=[
            ws_tools["load_skill"],
            ws_tools["run_bash"],
            ws_tools["read_file"],
            ws_tools["write_file"],
            ws_tools["exit_loop"],
            *delegated_tools,
        ],
        before_tool_callback=before_cb,
        after_tool_callback=after_cb,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return LoopAgent(
            name="harness2_loop_agent",
            sub_agents=[solver_agent, verifier_agent],
            max_iterations=max_iters,
        )


async def _run_loop_rollout_async(
    staged: Path,
    workspace: Path,
    prompt: str,
    events_path: Path,
    *,
    model: str | Any = "",
    max_iterations: int | None = None,
    ctx: Mapping[str, str] | None = None,
    isolate_sessions: bool = False,
) -> dict[str, Any]:
    adk = load_adk_classes()
    InMemoryRunner = adk["InMemoryRunner"]
    types = adk["types"]

    staged = Path(staged).resolve()
    workspace = Path(workspace).resolve()
    events_path = Path(events_path).resolve()
    events_path.parent.mkdir(parents=True, exist_ok=True)

    run_hook_event(staged, workspace, "SessionStart")
    loop_agent = build_loop_agent(
        staged,
        workspace,
        model=model,
        max_iterations=max_iterations,
        ctx=ctx,
        isolate_sessions=isolate_sessions,
    )
    runner = InMemoryRunner(agent=loop_agent, app_name="loop_agent_runtime")
    session = await runner.session_service.create_session(
        app_name="loop_agent_runtime", user_id="solver"
    )
    new_message = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])

    event_count = 0
    escalated = False
    with events_path.open("w", encoding="utf-8") as out:
        try:
            async for event in runner.run_async(
                user_id="solver", session_id=session.id, new_message=new_message
            ):
                payload = event.model_dump(mode="json", by_alias=True, exclude_none=True)
                out.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
                out.flush()
                event_count += 1
                if (payload.get("actions") or {}).get("escalate"):
                    escalated = True
        except Exception as exc:
            err_ev = {"type": "error", "error": {"message": f"{type(exc).__name__}: {exc}"}}
            out.write(json.dumps(err_ev, ensure_ascii=False) + "\n")
            out.flush()
            raise
        summary_ev = {
            "type": "turn.completed",
            "events": event_count,
            "escalated": escalated,
        }
        out.write(json.dumps(summary_ev, ensure_ascii=False) + "\n")
        out.flush()

    return {"events": event_count, "escalated": escalated, "events_path": events_path}


def run_loop_rollout(
    staged: Path,
    workspace: Path,
    prompt: str,
    events_path: Path,
    *,
    model: str | Any = "",
    max_iterations: int | None = None,
    ctx: Mapping[str, str] | None = None,
    isolate_sessions: bool = False,
) -> dict[str, Any]:
    """Run one synchronous `LoopAgent` rollout and write its JSONL stream to `events_path`."""
    return asyncio.run(
        _run_loop_rollout_async(
            staged,
            workspace,
            prompt,
            events_path,
            model=model,
            max_iterations=max_iterations,
            ctx=ctx,
            isolate_sessions=isolate_sessions,
        )
    )


_TRUNC_FMT = "…(+{n} chars — render truncation)"


def _clip(s: Any, n: int) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\x00", "")
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + _TRUNC_FMT.format(n=len(s) - n)


def render_events(
    path: Path, *, obs_trunc: int = 50_000, max_chars: int = 200_000, args_trunc: int = 1000
) -> str:
    """Render an ADK `LoopAgent` JSONL event stream into the `[ACTION]/[OBSERVATION]` grammar."""
    args_cap = int(args_trunc)
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"(failed to read trajectory: {e})"

    lines: list[str] = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            ev = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue

        err = ev.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or json.dumps(err)[:300]
            lines.append(f"[ERROR] {_clip(msg, 500)}")
            continue
        if ev.get("errorMessage") or ev.get("error_message"):
            msg = ev.get("errorMessage") or ev.get("error_message")
            lines.append(f"[ERROR] {_clip(msg, 500)}")
            continue

        if ev.get("type") == "turn.completed":
            lines.append(
                f"[HARNESS] loop completed | events={ev.get('events')} "
                f"escalated={ev.get('escalated')}"
            )
            continue

        author = str(ev.get("author") or "").strip()
        author_tag = f" ({author})" if author and author != "user" else ""

        content = ev.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict):
                    continue
                fc = part.get("functionCall") or part.get("function_call")
                if isinstance(fc, dict):
                    name = fc.get("name") or "tool"
                    args = fc.get("args")
                    args_str = (
                        json.dumps(args, default=str, ensure_ascii=False)
                        if args is not None
                        else ""
                    )
                    lines.append(f"[ACTION]{author_tag} {name}({_clip(args_str, args_cap)})")
                    continue

                fr = part.get("functionResponse") or part.get("function_response")
                if isinstance(fr, dict):
                    resp = fr.get("response")
                    is_err = isinstance(resp, dict) and (
                        resp.get("status") == "error"
                        or resp.get("decision") == "block"
                        or (resp.get("exit_code") not in (None, 0))
                    )
                    if isinstance(resp, dict):
                        if "stdout" in resp or "stderr" in resp:
                            out_txt = (
                                (str(resp.get("stdout") or "") + str(resp.get("stderr") or ""))
                                .strip()
                            )
                            if not out_txt:
                                out_txt = json.dumps(resp, ensure_ascii=False, default=str)
                        else:
                            out_txt = json.dumps(resp, ensure_ascii=False, default=str)
                    else:
                        out_txt = str(resp)
                    tag = "OBSERVATION ERROR" if is_err else "OBSERVATION"
                    lines.append(f"[{tag}] {_clip(out_txt, obs_trunc)}")
                    continue

                txt = part.get("text")
                if isinstance(txt, str) and txt.strip():
                    if part.get("thought"):
                        lines.append(f"[THINKING{author_tag}] {_clip(txt, obs_trunc)}")
                    else:
                        lines.append(f"[ASSISTANT{author_tag}] {_clip(txt, obs_trunc)}")

        actions = ev.get("actions")
        if isinstance(actions, dict) and actions.get("escalate"):
            lines.append(f"[HARNESS] loop escalated by {author or 'agent'}")

    rendered = "\n".join(lines).strip()
    if not rendered:
        return "(no parseable events in trajectory)"
    if len(rendered) > max_chars:
        dropped = len(rendered) - max_chars
        rendered = rendered[:max_chars] + "\n" + _TRUNC_FMT.format(n=dropped)
    return rendered


def main(argv: list[str] | None = None) -> int:
    """Standalone CLI entrypoint (`python -m loop_agent.runtime`)."""
    p = argparse.ArgumentParser(prog="loop_agent.runtime", description=__doc__)
    p.add_argument("--harness-dir", type=Path, required=True)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--prompt-file", type=Path, required=True)
    p.add_argument("--events-out", type=Path, required=True)
    p.add_argument("--model", default="")
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional path to a .env file to load before running the LoopAgent",
    )
    args = p.parse_args(argv)
    if args.env_file is not None:
        load_dotenv(args.env_file, override=True)
    else:
        load_dotenv()

    prompt = args.prompt_file.read_text(encoding="utf-8", errors="replace")
    try:
        run_loop_rollout(
            args.harness_dir,
            args.workspace,
            prompt,
            args.events_out,
            model=args.model,
            max_iterations=args.max_iterations,
        )
    except Exception as exc:
        print(f"loop_agent.runtime error: {exc}", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "ALWAYS_ON_FILES",
    "DEFAULT_SOLVER_STAGE_INSTRUCTION",
    "DEFAULT_VERIFIER_STAGE_INSTRUCTION",
    "ROOT_DIR",
    "adk_command_timeout",
    "adk_isolate_sessions",
    "adk_max_concurrent_commands",
    "adk_max_iterations",
    "adk_model",
    "adk_session_service_uri",
    "build_loop_agent",
    "compose_global_instruction",
    "load_adk_classes",
    "load_dotenv",
    "make_deterministic_loop_llm",
    "make_tool_callbacks",
    "make_workspace_tools",
    "render_events",
    "resolve_session_workspace",
    "run_hook_event",
    "run_loop_rollout",
    "scrub_runtime_env",
    "sync_workspace_assets",
]


if __name__ == "__main__":
    raise SystemExit(main())
