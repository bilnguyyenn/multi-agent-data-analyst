"""Run store: persistence for run records (question, status, events, final state).

v1 is JSON files under .analyst_runs/ — deliberately boring. The interface is the
architecture; Postgres replaces the backend in phase (g) without touching callers.
(Same argument as the sandbox backends: swap the implementation, keep the contract.)
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_DIR = ".analyst_runs"


class RunStore:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or os.environ.get("ANALYST_RUNS_DIR", DEFAULT_DIR))
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # worker thread + API thread share this

    def _path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.json"

    def create(self, run_id: str, question: str, data_dir: str, options: dict) -> None:
        self._write(run_id, {
            "run_id": run_id, "question": question, "data_dir": data_dir,
            "options": options, "status": "queued", "created_at": time.time(),
            "events": [], "answer": None, "dag": {}, "claims": [],
            "statistical_evidence": [],
            "predictive_evidence": [],
            "data_quality_report": None,
            "council_findings": [], "council_decisions": [],
            "council_routing": None,
        })

    def read(self, run_id: str) -> dict | None:
        p = self._path(run_id)
        if not p.exists():
            return None
        with self._lock:
            return json.loads(p.read_text())

    def list_runs(self, limit: int = 50) -> list[dict]:
        runs = []
        for p in sorted(self.root.glob("*.json"), key=os.path.getmtime, reverse=True)[:limit]:
            try:
                r = json.loads(p.read_text())
                runs.append({k: r.get(k) for k in
                             ("run_id", "question", "status", "created_at", "answer")})
            except (json.JSONDecodeError, OSError):
                continue
        return runs

    def update(self, run_id: str, **fields: Any) -> None:
        with self._lock:
            data = json.loads(self._path(run_id).read_text())
            data.update(fields)
            self._path(run_id).write_text(json.dumps(data, indent=2, default=str))

    def append_event(self, run_id: str, event: dict) -> None:
        with self._lock:
            data = json.loads(self._path(run_id).read_text())
            data["events"].append(event)
            self._path(run_id).write_text(json.dumps(data, indent=2, default=str))

    def _write(self, run_id: str, data: dict) -> None:
        with self._lock:
            self._path(run_id).write_text(json.dumps(data, indent=2, default=str))
