"""DEV-ONLY subprocess backend — NOT a security boundary.

Exists for exactly two reasons:
  1. CI environments without a Docker daemon (our replay-mode eval on GitHub Actions
     uses handwritten/recorded code that we've already reviewed).
  2. Fast local iteration on graph logic when you're not testing isolation.

What it does enforce: wall-clock timeout, a CPU rlimit, a best-effort address-space
rlimit, and the same result.json contract as the Docker backend — so
failure-handling code paths are identical across backends. macOS does not reliably
support lowering RLIMIT_AS in a Python pre-exec hook, so memory limiting is skipped
there; the Docker backend remains the security boundary. What this backend does NOT
enforce: network isolation or filesystem isolation. It refuses to run unless
ANALYST_SANDBOX=subprocess is set explicitly, so nobody ends up here by accident.
"""
from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from analyst.graph.state import ExecutionOutcome
from analyst.sandbox.runner import collect_result, write_deps

MEM_LIMIT_BYTES = 1024**3  # 1 GiB address space


def _rlimits():
    """Apply portable, best-effort limits without making process startup fail.

    Darwin exposes RLIMIT_AS but rejects lowering it from ``preexec_fn`` with
    ``EINVAL``. That previously made every subprocess-backed test fail before the
    generated code even started. CPU limiting works on both Linux and macOS; the
    address-space cap is kept on platforms where the kernel supports it.
    """
    if sys.platform != "darwin":
        try:
            resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))
        except (OSError, ValueError):
            # This is a developer convenience backend, not the security boundary.
            # A wall-clock timeout and the result contract still apply.
            pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    except (OSError, ValueError):
        pass


class SubprocessRunner:
    def __init__(self):
        if os.environ.get("ANALYST_SANDBOX") != "subprocess":
            raise RuntimeError(
                "SubprocessRunner is dev-only and must be opted into explicitly "
                "with ANALYST_SANDBOX=subprocess. Use the Docker backend for any "
                "code you haven't read."
            )

    def run(self, code, dataset_dir, timeout_s=60, deps=None):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            # Same filesystem layout as the Docker backend: data at /data-equivalent,
            # cwd writable, result.json in cwd.
            (work / "code.py").write_text(code)
            deps_dir = write_deps(work, deps)
            env = {
                "DATA_DIR": str(dataset_dir.resolve()),
                "DEPS_DIR": str(deps_dir),
                "PATH": os.environ.get("PATH", ""),
                "HOME": td,
                # HOME is redirected to the tempdir, which silently breaks Python's
                # user-site package resolution (~/.local/...). Hand the child our
                # already-resolved sys.path instead of letting it re-derive one.
                "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
            }
            start = time.monotonic()
            try:
                proc = subprocess.run(
                    [sys.executable, "code.py"],
                    cwd=work, env=env, capture_output=True, text=True,
                    timeout=timeout_s, preexec_fn=_rlimits,
                )
            except subprocess.TimeoutExpired:
                return ExecutionOutcome(
                    status="timeout", stderr=f"[sandbox] killed after {timeout_s}s",
                    duration_s=time.monotonic() - start,
                )
            duration = time.monotonic() - start
            if proc.returncode != 0:
                status = "oom_or_killed" if proc.returncode in (-9, 137) else "error"
                return ExecutionOutcome(
                    status=status, stdout=proc.stdout, stderr=proc.stderr,
                    duration_s=duration,
                )
            return collect_result(work, proc.stdout, proc.stderr, duration)
