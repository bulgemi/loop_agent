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

"""Decoupled Google ADK `LoopAgent` runtime and `harness2` bridge package.

- `loop_agent.runtime`: Layer 1 pure Google ADK `LoopAgent` builder, tool/hook middleware,
  and trajectory renderer with zero imports from `harness2`.
- `loop_agent.bridge`: Layer 2 non-invasive `AdkSubstrate` and `AdkBenchmarkAdapter`
  decorator connecting `loop_agent.runtime` to `harness2` without modifying `harness2`.
"""

from .runtime import (
    ALWAYS_ON_FILES,
    build_loop_agent,
    compose_global_instruction,
    make_deterministic_loop_llm,
    make_tool_callbacks,
    make_workspace_tools,
    render_events,
    run_hook_event,
    run_loop_rollout,
    sync_workspace_assets,
)

__all__ = [
    "ALWAYS_ON_FILES",
    "build_loop_agent",
    "compose_global_instruction",
    "make_deterministic_loop_llm",
    "make_tool_callbacks",
    "make_workspace_tools",
    "render_events",
    "run_hook_event",
    "run_loop_rollout",
    "sync_workspace_assets",
]
