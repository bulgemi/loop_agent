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

"""Google ADK entrypoint (`root_agent`) for `adk web`, `adk run`, and `adk api_server`.

Usage:
  - Interactive Web UI:   `.venv/bin/adk web loop_agent --port 8000`
  - Terminal CLI Chat:    `.venv/bin/adk run loop_agent`
  - FastAPI REST Server:  `.venv/bin/adk api_server loop_agent --port 8000`
"""

from __future__ import annotations

import os
from pathlib import Path

from .runtime import (
    ALWAYS_ON_FILES,
    ROOT_DIR,
    adk_isolate_sessions,
    build_loop_agent,
    load_dotenv,
)

load_dotenv()


def _resolve_interactive_dirs() -> tuple[Path, Path]:
    """Resolve (harness_dir, workspace_dir) for interactive ADK sessions (`adk web` / `adk run`)."""
    default_root = ROOT_DIR / "runs" / "adk_interactive"
    harness_raw = (
        os.environ.get("HARNESS2_ADK_HARNESS_DIR")
        or os.environ.get("ADK_HARNESS_DIR")
        or ""
    ).strip()
    workspace_raw = (
        os.environ.get("HARNESS2_ADK_WORKSPACE")
        or os.environ.get("ADK_WORKSPACE")
        or ""
    ).strip()

    harness_dir = Path(harness_raw).expanduser().resolve() if harness_raw else (default_root / "harness")
    workspace_dir = Path(workspace_raw).expanduser().resolve() if workspace_raw else (default_root / "workspace")

    harness_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    if not any((harness_dir / rel).is_file() for rel in ALWAYS_ON_FILES):
        (harness_dir / "systemprompt.md").write_text(
            "You are an iterative Google ADK LoopAgent (`solver` -> `verifier`).\n"
            "Solve the user's request in the workspace and verify all outputs before calling `exit_loop`.\n",
            encoding="utf-8",
        )
        for rel in ALWAYS_ON_FILES[1:]:
            p = harness_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.write_text("", encoding="utf-8")

    return harness_dir, workspace_dir


def create_root_agent(*, isolate_sessions: bool | None = None):
    """Build the `LoopAgent` instance exposed as `root_agent` to the Google ADK CLI/Web server.

    By default, `isolate_sessions=True` in interactive/server mode so each `(user_id, session_id)`
    operates in its own `<workspace>/<user_id>/<session_id>/` directory without file collisions.
    """
    harness_dir, workspace_dir = _resolve_interactive_dirs()
    effective_isolate = (
        adk_isolate_sessions(default=True)
        if isolate_sessions is None
        else bool(isolate_sessions)
    )
    return build_loop_agent(
        harness_dir, workspace_dir, isolate_sessions=effective_isolate
    )


root_agent = create_root_agent()

__all__ = ["create_root_agent", "root_agent"]
