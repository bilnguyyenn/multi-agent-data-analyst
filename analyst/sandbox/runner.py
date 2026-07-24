"""Sandbox contract + backend selection.

Contract with generated code (kept brutally simple so small local models can follow it):

- Dataset CSVs are at /data (read-only). Code may use pandas or duckdb.
- Code MUST write its final answer as JSON to ./result.json (cwd is writable).
- stdout/stderr are captured but truncated; result.json > 256 KB is a contract
  violation (an "answer" that big is a data dump, not an answer).

Failure taxonomy (each maps to different handling upstream — ARCHITECTURE.md §4):
  ok | timeout | oom_or_killed | error | contract_violation
Deterministic failures (error/timeout) bounce back to the EXECUTOR with the traceback
(that's a bug-fix loop); only clean results ever reach the CRITIC (that's a review loop).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

from analyst.graph.state import ExecutionOutcome

RESULT_MAX_BYTES = 256 * 1024
STREAM_MAX_CHARS = 10_000


class SandboxRunner(Protocol):
    def run(
        self,
        code: str,
        dataset_dir: Path,
        timeout_s: int = 60,
        deps: dict[str, Any] | None = None,
    ) -> ExecutionOutcome:
        """deps: dependency task results, materialized as read-only JSON files
        at $DEPS_DIR/<task_id>.json inside the sandbox. Added after a live run
        where the model ASSUMED file-based handoff existed and wrote code to
        read a dependency from disk — the fix was to make its assumption true."""
        ...


def write_deps(work: Path, deps: dict[str, Any] | None) -> Path:
    """Shared by all backends: materialize dependency results as files."""
    deps_dir = work / "deps"
    deps_dir.mkdir()
    deps_dir.chmod(0o755)
    for tid, payload in (deps or {}).items():
        f = deps_dir / f"{tid}.json"
        f.write_text(json.dumps(payload))
        f.chmod(0o644)
    return deps_dir


def make_runner(backend: str | None = None) -> SandboxRunner:
    backend = backend or os.environ.get("ANALYST_SANDBOX", "docker")
    if backend == "docker":
        from analyst.sandbox.docker_backend import DockerRunner

        return DockerRunner()
    if backend == "subprocess":
        from analyst.sandbox.subprocess_backend import SubprocessRunner

        return SubprocessRunner()
    raise ValueError(f"Unknown sandbox backend {backend!r}")


def collect_result(out_dir: Path, stdout: str, stderr: str, duration: float) -> ExecutionOutcome:
    """Shared post-run contract enforcement for all backends."""
    stdout, stderr = stdout[:STREAM_MAX_CHARS], stderr[:STREAM_MAX_CHARS]
    f = out_dir / "result.json"
    if not f.exists():
        return ExecutionOutcome(
            status="contract_violation",
            stdout=stdout,
            stderr=stderr + "\n[contract] result.json was not written",
            duration_s=duration,
        )
    if f.stat().st_size > RESULT_MAX_BYTES:
        return ExecutionOutcome(
            status="contract_violation",
            stdout=stdout,
            stderr=stderr + f"\n[contract] result.json exceeds {RESULT_MAX_BYTES} bytes",
            duration_s=duration,
        )
    try:
        result = json.loads(f.read_text())
    except json.JSONDecodeError as e:
        return ExecutionOutcome(
            status="contract_violation",
            stdout=stdout,
            stderr=stderr + f"\n[contract] result.json is not valid JSON: {e}",
            duration_s=duration,
        )
    return ExecutionOutcome(
        status="ok", stdout=stdout, stderr=stderr, result=result, duration_s=duration
    )
