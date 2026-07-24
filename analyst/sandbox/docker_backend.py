"""Docker sandbox: one throwaway container per execution.

Every flag maps to a threat (ARCHITECTURE.md §4 table):

  --network none                  exfiltration / phoning home
  --read-only + --tmpfs /tmp      filesystem damage, persistence
  -v dataset:/data:ro             source data mutation
  --cpus/--memory/--pids-limit    resource exhaustion, fork bombs
  --user 65534 --cap-drop ALL     privilege escalation
  --security-opt no-new-privileges
  wall-clock kill                 infinite loops

The image (docker/sandbox.Dockerfile) preinstalls pandas/duckdb/scipy/statsmodels so
per-run cost is container START (~100-300ms), not pip install.

Upgrade path, documented not built: swap runtime for gVisor (--runtime=runsc, same
images) for syscall-level isolation; E2B/Firecracker for hosted multi-tenant.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from analyst.graph.state import ExecutionOutcome
from analyst.sandbox.runner import collect_result, write_deps

IMAGE = "analyst-sandbox:latest"


class DockerRunner:
    def __init__(self, image: str = IMAGE):
        self.image = image
        self._preflight()

    def _preflight(self) -> None:
        """Fail at startup with a human message, not mid-run with a socket trace."""
        try:
            probe = subprocess.run(
                ["docker", "info"], capture_output=True, text=True, timeout=10
            )
        except FileNotFoundError:
            raise RuntimeError(
                "docker CLI not found. Install Docker Desktop or `brew install colima docker`."
            ) from None
        except subprocess.TimeoutExpired:
            raise RuntimeError("docker daemon not responding (is the VM still starting?)")
        if probe.returncode != 0:
            raise RuntimeError(
                "Docker daemon is not running. Start it first:\n"
                "  colima start --memory 2 --cpu 2   (or launch Docker Desktop)\n"
                f"[docker info said: {probe.stderr.strip()[:200]}]"
            )
        img = subprocess.run(
            ["docker", "image", "inspect", self.image], capture_output=True, timeout=10
        )
        if img.returncode != 0:
            raise RuntimeError(
                f"Sandbox image {self.image!r} not built yet. Run:\n"
                "  docker build -t analyst-sandbox:latest -f docker/sandbox.Dockerfile ."
            )

    @staticmethod
    def _exec_base() -> Path:
        """Host<->container exchange dir. MUST be under a path the Docker VM shares.

        On macOS the daemon lives in a Linux VM (Colima shares ~, Docker Desktop
        shares /Users); the default tempdir (/var/folders/...) is NOT shared, and
        Docker silently 'mounts' a nonexistent path as an empty directory — which
        surfaced as `can't find '__main__' module in '/sandbox/code.py'` on the
        first live run. Home is the portable safe choice; override with
        ANALYST_EXEC_TMP if your setup shares somewhere else."""
        base = Path(
            os.environ.get("ANALYST_EXEC_TMP", Path.home() / ".cache" / "analyst-exec")
        )
        base.mkdir(parents=True, exist_ok=True)
        return base

    def run(self, code, dataset_dir, timeout_s=60, deps=None):
        name = f"analyst-exec-{uuid.uuid4().hex[:10]}"
        td = tempfile.mkdtemp(dir=self._exec_base())
        try:
            work = Path(td)
            deps_dir = write_deps(work, deps)
            # The container runs as uid 65534 (nobody), but tempdirs are created
            # 0700 for *our* uid — nobody couldn't even read its own code file.
            # Found the hard way on the first live Colima run (3 failures @0.4s).
            work.chmod(0o755)
            (work / "code.py").write_text(code)
            (work / "code.py").chmod(0o644)
            out_dir = work / "out"
            out_dir.mkdir()
            out_dir.chmod(0o777)  # nobody must WRITE result.json here
            cmd = [
                "docker", "run", "--rm", "--name", name,
                "--network", "none",
                "--read-only",
                "--tmpfs", "/tmp:size=64m",
                "-v", f"{dataset_dir.resolve()}:/data:ro",
                "-v", f"{(work / 'code.py').resolve()}:/sandbox/code.py:ro",
                "-v", f"{out_dir.resolve()}:/out",
                "-v", f"{deps_dir.resolve()}:/deps:ro",
                "-w", "/out",
                "-e", "DATA_DIR=/data",  # same env contract as the dev backend
                "-e", "DEPS_DIR=/deps",
                "--cpus", "1", "--memory", "1g", "--pids-limit", "128",
                "--user", "65534:65534",
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges",
                self.image,
                "python", "/sandbox/code.py",
            ]
            start = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout_s
                )
            except subprocess.TimeoutExpired as e:
                subprocess.run(["docker", "kill", name], capture_output=True)
                return ExecutionOutcome(
                    status="timeout",
                    stdout=(e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
                    stderr=f"[sandbox] killed after {timeout_s}s wall clock",
                    duration_s=time.monotonic() - start,
                )
            duration = time.monotonic() - start
            if proc.returncode == 137:  # SIGKILL: cgroup OOM or external kill
                return ExecutionOutcome(
                    status="oom_or_killed",
                    stdout=proc.stdout, stderr=proc.stderr, duration_s=duration,
                )
            if proc.returncode != 0:
                return ExecutionOutcome(
                    status="error",
                    stdout=proc.stdout, stderr=proc.stderr, duration_s=duration,
                )
            return collect_result(out_dir, proc.stdout, proc.stderr, duration)
        finally:
            shutil.rmtree(td, ignore_errors=True)
