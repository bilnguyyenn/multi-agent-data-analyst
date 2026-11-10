"""Local durable operations reviews. SQLite transactions own job/evidence state."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from analyst.operations.metrics import (
    VERSION,
    COLUMNS,
    MetricRequest,
    compute,
    contract_hash,
    connect,
    fingerprint,
    prepare,
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Playbook(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    metrics: list[MetricRequest] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def reusable(self):
        if any(request.start for request in self.metrics):
            raise ValueError("set dates when refreshing, not in the saved playbook")
        if len({fingerprint(m.model_dump(mode="json")) for m in self.metrics}) != len(self.metrics):
            raise ValueError("duplicate metric requests")
        return self


class Refresh(BaseModel):
    model_config = ConfigDict(extra="forbid")
    playbook_id: str
    snapshot_id: str
    start: date
    end: date
    timeout_s: int = Field(default=60, ge=1, le=300)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=120)

    @model_validator(mode="after")
    def window(self):
        if self.start >= self.end:
            raise ValueError("start must precede exclusive end")
        return self


class Operations:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "snapshots").mkdir(exist_ok=True)
        with self.db() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS snapshots(id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS playbooks(id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(
              id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL, spec TEXT NOT NULL,
              status TEXT NOT NULL, owner TEXT, lease_until REAL, cancel_requested INTEGER DEFAULT 0,
              data TEXT NOT NULL);
            """)

    def db(self):
        return sqlite3.connect(self.root / "operations.sqlite", timeout=30)

    def list(self, table: str) -> list[dict]:
        if table not in {"snapshots", "playbooks", "jobs"}:
            raise ValueError("invalid collection")
        with self.db() as db:
            return [
                json.loads(row[0])
                for row in db.execute(f"SELECT data FROM {table} ORDER BY rowid DESC LIMIT 100")
            ]

    def get(self, table: str, identity: str) -> dict:
        if table not in {"snapshots", "playbooks", "jobs", "evidence"}:
            raise ValueError("invalid collection")
        with self.db() as db:
            row = db.execute(f"SELECT data FROM {table} WHERE id=?", [identity]).fetchone()
        if row is None:
            raise KeyError(f"unknown {table}: {identity}")
        result = json.loads(row[0])
        if (
            table == "evidence"
            and fingerprint({k: v for k, v in result.items() if k != "computation_id"}) != identity
        ):
            raise ValueError("evidence fingerprint mismatch")
        return result

    def ingest(self, source: str | Path, metadata: dict | None = None) -> dict:
        source = Path(source)
        with tempfile.TemporaryDirectory(dir=self.root, prefix="ingest-") as directory:
            temporary = Path(directory) / "transactions.csv"
            shutil.copyfile(source, temporary)
            identity = file_hash(temporary)
            c = connect()
            try:
                prepare(c, temporary)
                count, low, high = c.execute(
                    "SELECT count(*), min(occurred), max(occurred) FROM lines"
                ).fetchone()
            finally:
                c.close()
            if not count:
                raise ValueError("snapshot has no source rows")
            destination = self.root / "snapshots" / f"{identity}.csv"
            if destination.exists():
                if file_hash(destination) != identity:
                    raise ValueError("stored snapshot was modified")
                try:
                    return self.get("snapshots", identity)
                except KeyError:
                    pass  # Recover a crash after file publication but before DB insertion.
            else:
                os.replace(temporary, destination)
                destination.chmod(0o444)
            result = {
                "snapshot_id": identity,
                "content_hash": identity,
                "rows": count,
                "observed_start": str(low) if low else None,
                "observed_end": str(high) if high else None,
                "metadata": metadata or {},
                "created_at": time.time(),
            }
            with self.db() as db:
                db.execute(
                    "INSERT OR IGNORE INTO snapshots VALUES (?,?)", [identity, json.dumps(result)]
                )
            return self.get("snapshots", identity)

    def save_playbook(self, raw: dict) -> dict:
        spec = Playbook.model_validate(raw).model_dump(mode="json")
        definition_hash = contract_hash()
        identity = fingerprint(
            {"metric_version": VERSION, "metric_contract_hash": definition_hash, **spec}
        )
        result = {
            "playbook_id": identity,
            "metric_version": VERSION,
            "metric_contract_hash": definition_hash,
            **spec,
        }
        with self.db() as db:
            db.execute(
                "INSERT OR IGNORE INTO playbooks VALUES (?,?)", [identity, json.dumps(result)]
            )
        return result

    def submit(self, raw: dict) -> dict:
        request = Refresh.model_validate(raw)
        playbook = self.get("playbooks", request.playbook_id)
        self.get("snapshots", request.snapshot_id)
        if (
            playbook["metric_version"] != VERSION
            or playbook.get("metric_contract_hash") != contract_hash()
        ):
            raise ValueError("playbook metric version is incompatible with this engine")
        spec = request.model_dump(mode="json", exclude={"idempotency_key"})
        spec["metric_version"] = VERSION
        spec["metric_contract_hash"] = playbook["metric_contract_hash"]
        key = request.idempotency_key or fingerprint(spec)
        result = {
            "job_id": uuid.uuid4().hex,
            "status": "queued",
            "spec": spec,
            "created_at": time.time(),
            "evidence_ids": [],
        }
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT spec,data FROM jobs WHERE request_key=?", [key]).fetchone()
            if prior:
                if json.loads(prior[0]) != spec:
                    raise ValueError("idempotency key already belongs to a different refresh")
                return json.loads(prior[1])
            db.execute(
                "INSERT INTO jobs(id,request_key,spec,status,data) VALUES (?,?,?,'queued',?)",
                [result["job_id"], key, json.dumps(spec), json.dumps(result)],
            )
        return result

    def pending(self) -> list[str]:
        with self.db() as db:
            return [
                r[0]
                for r in db.execute(
                    "SELECT id FROM jobs WHERE status='queued' OR (status='running' AND lease_until < ?)",
                    [time.time()],
                )
            ]

    def cancel(self, identity: str) -> dict:
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status,data FROM jobs WHERE id=?", [identity]).fetchone()
            if row is None:
                raise KeyError(identity)
            status, data = row[0], json.loads(row[1])
            if status == "queued":
                status = data["status"] = "cancelled"
                data["finished_at"] = time.time()
            elif status == "running":
                data["cancel_requested"] = True
            db.execute(
                "UPDATE jobs SET status=?, cancel_requested=1,data=? WHERE id=?",
                [status, json.dumps(data), identity],
            )
        return data

    def _cancelled(self, identity: str) -> bool:
        with self.db() as db:
            return bool(
                db.execute("SELECT cancel_requested FROM jobs WHERE id=?", [identity]).fetchone()[0]
            )

    def retry(self, identity: str) -> dict:
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status,data FROM jobs WHERE id=?", [identity]).fetchone()
            if row is None:
                raise KeyError(identity)
            result = json.loads(row[1])
            if row[0] not in {"error", "cancelled"}:
                raise ValueError("only failed or cancelled reviews can be explicitly retried")
            history = result.get("attempt_history", [])
            if len(history) >= 5:
                raise ValueError(
                    "retry budget exhausted; inspect the failure before submitting a new refresh"
                )
            history.append(
                {key: result.get(key) for key in ["status", "error", "started_at", "finished_at"]}
            )
            result.update(status="queued", evidence_ids=[], attempt_history=history)
            for key in ["error", "finished_at", "started_at", "cancel_requested"]:
                result.pop(key, None)
            db.execute(
                "UPDATE jobs SET status='queued',owner=NULL,lease_until=NULL,cancel_requested=0,data=? WHERE id=?",
                [json.dumps(result), identity],
            )
            return result

    def execute(self, identity: str) -> dict:
        owner = uuid.uuid4().hex
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status,lease_until,data FROM jobs WHERE id=?", [identity]
            ).fetchone()
            if row is None:
                raise KeyError(identity)
            result = json.loads(row[2])
            if row[0] in {"done", "error", "cancelled"} or (
                row[0] == "running" and row[1] >= time.time()
            ):
                return result
            result.update(status="running", started_at=time.time())
            db.execute(
                "UPDATE jobs SET status='running',owner=?,lease_until=?,data=? WHERE id=?",
                [
                    owner,
                    time.time() + result["spec"]["timeout_s"] + 30,
                    json.dumps(result),
                    identity,
                ],
            )
        started, packets = time.monotonic(), []
        spec, connection, timer = result["spec"], None, None
        try:
            if spec.get("metric_contract_hash") != contract_hash():
                raise ValueError("metric definitions changed while the job was queued")
            path = self.root / "snapshots" / f"{spec['snapshot_id']}.csv"
            if file_hash(path) != spec["snapshot_id"]:
                raise ValueError("snapshot content changed")
            connection = connect()
            timer = threading.Timer(spec["timeout_s"], connection.interrupt)
            timer.daemon = True
            timer.start()
            prepare(connection, path)
            for raw in self.get("playbooks", spec["playbook_id"])["metrics"]:
                if self._cancelled(identity):
                    break
                if time.monotonic() - started > spec["timeout_s"]:
                    raise TimeoutError("refresh time budget exhausted")
                request = MetricRequest.model_validate(
                    {**raw, "start": spec["start"], "end": spec["end"]}
                )
                packets.append(compute(connection, request, spec["snapshot_id"]))
            if file_hash(path) != spec["snapshot_id"]:
                raise ValueError("snapshot changed during execution")
            if time.monotonic() - started > spec["timeout_s"]:
                raise TimeoutError("refresh time budget exhausted")
            result.update(status="done", evidence_ids=[p["computation_id"] for p in packets])
        except Exception as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}", evidence_ids=[])
        finally:
            if timer:
                timer.cancel()
                timer.join()
            if connection:
                connection.close()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            1024**2 if sys.platform == "darwin" else 1024
        )
        result.update(
            finished_at=time.time(),
            duration_s=round(time.monotonic() - started, 4),
            process_peak_rss_mb=round(peak, 2),
            memory_scope="process lifetime high-water mark",
        )
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT owner,cancel_requested FROM jobs WHERE id=?", [identity]
            ).fetchone()
            if row[0] != owner:
                return self.get("jobs", identity)
            if row[1]:
                result.update(status="cancelled", evidence_ids=[])
            if result["status"] == "done":
                for packet in packets:
                    db.execute(
                        "INSERT OR IGNORE INTO evidence VALUES (?,?)",
                        [packet["computation_id"], json.dumps(packet)],
                    )
            db.execute(
                "UPDATE jobs SET status=?,data=? WHERE id=?",
                [result["status"], json.dumps(result), identity],
            )
        return result

    def report_record(self, identity: str) -> dict:
        job = self.get("jobs", identity)
        if job["status"] != "done":
            raise ValueError("review is not completed")
        evidence = [self.get("evidence", reference) for reference in job["evidence_ids"]]
        snapshot = self.get("snapshots", job["spec"]["snapshot_id"])
        diagnostics = evidence[0]["quality"] if evidence else {}
        return {
            "run_id": identity,
            "status": "done",
            "question": "Retail operations review",
            "answer": "\n".join(
                f"{p['metric']}: {p['values']} [{p['computation_id']}]" for p in evidence
            ),
            "operations_evidence": evidence,
            "created_at": job["created_at"],
            "dataset_provenance": {
                "content_hash": snapshot["content_hash"],
                "tables": [
                    {"name": "transactions", "row_count": snapshot["rows"], "columns": COLUMNS}
                ],
            },
            "data_quality_report": {
                "summary": "Snapshot diagnostics (not a readiness score): "
                + "; ".join(f"{key}={value}" for key, value in diagnostics.items()),
                "score_method": "diagnostics only; no readiness score computed",
                "table_count": 1,
                "total_rows": snapshot["rows"],
                "issues": [
                    {
                        "code": key,
                        "severity": "warning",
                        "table": "transactions",
                        "column": None,
                        "message": f"{value} source rows flagged for {key}.",
                        "recommendation": "Inspect the source and metric eligibility rules; counts can overlap. Source rows remain unchanged.",
                    }
                    for key, value in diagnostics.items()
                    if key != "source_rows" and value
                ],
            },
            "claims": [],
            "demo_note": "Deterministic metric-contract review. No LLM approval or causal inference is claimed.",
        }

    def compare(self, previous: str, current: str) -> dict:
        old, new = self.get("jobs", previous), self.get("jobs", current)
        if any(job["status"] != "done" for job in [old, new]):
            raise ValueError("both reviews must be completed")
        a, b = old["spec"], new["spec"]
        if not a.get("metric_contract_hash") or a.get("metric_contract_hash") != b.get(
            "metric_contract_hash"
        ):
            raise ValueError("incompatible or missing metric definition fingerprints")
        if (a["playbook_id"], a["metric_version"]) != (b["playbook_id"], b["metric_version"]):
            raise ValueError("incompatible playbook or metric definitions")

        def days(spec):
            return (date.fromisoformat(spec["end"]) - date.fromisoformat(spec["start"])).days

        if days(a) != days(b) or not (a["end"] <= b["start"] or b["end"] <= a["start"]):
            raise ValueError("comparison requires equal-length, nonoverlapping periods")
        rows = []
        for old_id, new_id in zip(old["evidence_ids"], new["evidence_ids"]):
            before, after = self.get("evidence", old_id), self.get("evidence", new_id)
            left = {r["segment"]: r["value"] for r in before["values"]}
            right = {r["segment"]: r["value"] for r in after["values"]}
            for key in sorted(left.keys() | right.keys()):
                absent = None if after["unit"] == "fraction" else 0
                x, y = left.get(key, absent), right.get(key, absent)
                rows.append(
                    {
                        "metric": after["metric"],
                        "segment": key,
                        "previous": x,
                        "current": y,
                        "delta": round(y - x, 10) if x is not None and y is not None else None,
                        "evidence_refs": [old_id, new_id],
                    }
                )
        return {
            "previous_job": previous,
            "current_job": current,
            "rows": rows,
            "warnings": [
                "Descriptive changes, not causal attribution. Check snapshot coverage and truncated group lists."
            ],
        }
