"""PostgresRunStore: the production run ledger (phase g).

One table, JSONB payload — deliberately minimal. Runs are read/written as whole
documents (the same shape the JSON-file store used), so both stores satisfy the
same interface and the API/worker code doesn't know which one it has. Relational
decomposition (events table, traces table) is the documented next step if
query patterns ever demand it; starting there would be schema speculation.

Selected automatically when ANALYST_DB_URL is set (see make_run_store).
"""
from __future__ import annotations

import json
import time
from typing import Any

DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      text PRIMARY KEY,
    created_at  double precision NOT NULL,
    data        jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_created_idx ON runs (created_at DESC);
"""


class PostgresRunStore:
    def __init__(self, dsn: str):
        import psycopg

        self._psycopg = psycopg
        self.dsn = dsn
        with self._conn() as conn:
            conn.execute(DDL)

    def _conn(self):
        return self._psycopg.connect(self.dsn, autocommit=True)

    def create(self, run_id: str, question: str, data_dir: str, options: dict) -> None:
        doc = {
            "run_id": run_id, "question": question, "data_dir": data_dir,
            "options": options, "status": "queued", "created_at": time.time(),
            "events": [], "answer": None, "dag": {}, "claims": [],
            "statistical_evidence": [],
            "predictive_evidence": [],
            "data_quality_report": None,
            "council_findings": [], "council_decisions": [],
            "council_routing": None,
        }
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, created_at, data) VALUES (%s, %s, %s) "
                "ON CONFLICT (run_id) DO NOTHING",
                (run_id, doc["created_at"], json.dumps(doc, default=str)),
            )

    def read(self, run_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT data FROM runs WHERE run_id = %s", (run_id,)
            ).fetchone()
        return row[0] if row else None

    def list_runs(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT data FROM runs ORDER BY created_at DESC LIMIT %s", (limit,)
            ).fetchall()
        return [{k: r[0].get(k) for k in
                 ("run_id", "question", "status", "created_at", "answer")}
                for r in rows]

    def update(self, run_id: str, **fields: Any) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET data = data || %s::jsonb WHERE run_id = %s",
                (json.dumps(fields, default=str), run_id),
            )

    def append_event(self, run_id: str, event: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET data = jsonb_set(data, '{events}', "
                "(data->'events') || %s::jsonb) WHERE run_id = %s",
                (json.dumps(event, default=str), run_id),
            )


def make_run_store():
    """Env-selected backend, same pattern as LLM/sandbox/event-bus."""
    import os

    dsn = os.environ.get("ANALYST_DB_URL")
    if dsn:
        return PostgresRunStore(dsn)
    from analyst.api.runstore import RunStore

    return RunStore()
