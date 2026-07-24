"""FastAPI application — async throughout, and the API process NEVER runs agent
graphs (ARCHITECTURE.md §5): it enqueues work and streams progress.

Two execution paths, chosen by env:
- default: enqueue onto arq (Redis). The worker (worker.py) executes.
- ANALYST_DEV_INPROCESS=1: run in a background thread of this process, publish
  over in-memory queues. Zero infrastructure — this is `make dev` and CI.

WebSocket contract: on connect, the client first receives REPLAYED history from the
run store (late joiners and reconnects see the whole story), then live events until
"stream_end". Streaming is UX; the store is truth.

Run it:  ANALYST_DEV_INPROCESS=1 uvicorn analyst.api.app:app --reload
"""
from __future__ import annotations

import asyncio
import csv
import os
import shutil
import uuid
from io import StringIO
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from analyst.api.events import make_event_bus
from analyst.api.pgstore import make_run_store
from analyst.api.runner import execute_run, new_run_id
from analyst.data import build_manifest

app = FastAPI(title="analyst", version="0.1.0")
app.add_middleware(  # dev: vite serves the UI from :5173
    CORSMiddleware, allow_origins=["http://localhost:5173"],
    allow_methods=["*"], allow_headers=["*"],
)

store = make_run_store()  # Postgres if ANALYST_DB_URL set, JSON files otherwise
bus = make_event_bus()

MAX_DATASET_FILES = 12
MAX_DATASET_BYTES = 20 * 1024 * 1024


class CreateRun(BaseModel):
    question: str
    data_dir: str = "data/sample"
    provider: str | None = None
    sandbox: str | None = None
    timeout: int = 60
    # The end-to-end ablation found that the critic added no accuracy and produced
    # false alarms on half of sound analyses. Keep it available for experiments,
    # but ship the evidence-backed path by default.
    no_critic: bool = True
    no_validator: bool = False
    no_council: bool = False
    council_mode: Literal["off", "adaptive", "always"] = "off"
    no_memory: bool = False


class DatasetFile(BaseModel):
    name: str
    content: str


class CreateDataset(BaseModel):
    files: list[DatasetFile]


@app.post("/datasets", status_code=201)
async def create_dataset(req: CreateDataset):
    """Persist a small collection of browser-uploaded CSVs for one local run.

    JSON text upload avoids a multipart dependency and keeps the single-user demo
    self-contained. Filenames, count, total size, headers, and parseability are
    validated before the directory becomes available to the analysis worker.
    """
    if not req.files:
        raise HTTPException(400, "select at least one CSV file")
    if len(req.files) > MAX_DATASET_FILES:
        raise HTTPException(400, f"at most {MAX_DATASET_FILES} CSV files per dataset")
    total_bytes = sum(len(f.content.encode("utf-8")) for f in req.files)
    if total_bytes > MAX_DATASET_BYTES:
        raise HTTPException(413, "dataset exceeds the 20 MB upload limit")

    names: set[str] = set()
    for uploaded in req.files:
        safe_name = Path(uploaded.name).name
        if safe_name != uploaded.name or not safe_name.lower().endswith(".csv"):
            raise HTTPException(400, f"invalid CSV filename: {uploaded.name!r}")
        normalized = safe_name.casefold()
        if normalized in names:
            raise HTTPException(400, f"duplicate filename: {safe_name}")
        names.add(normalized)
        try:
            rows = csv.reader(StringIO(uploaded.content))
            header = next(rows)
            if not header or any(not column.strip() for column in header):
                raise ValueError("header contains an empty column name")
            next(rows, None)  # force the parser to consume at least one data row
        except (csv.Error, StopIteration, ValueError) as exc:
            raise HTTPException(400, f"invalid CSV {safe_name}: {exc}") from exc

    dataset_id = uuid.uuid4().hex[:12]
    upload_root = Path(os.environ.get("ANALYST_UPLOAD_DIR", ".analyst_uploads"))
    dataset_dir = (upload_root / dataset_id).resolve()
    dataset_dir.mkdir(parents=True, exist_ok=False)
    try:
        for uploaded in req.files:
            (dataset_dir / uploaded.name).write_text(uploaded.content)
        manifest = build_manifest(dataset_dir)
    except Exception as exc:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise HTTPException(400, f"could not load dataset: {exc}") from exc

    return {
        "dataset_id": dataset_id,
        "data_dir": str(dataset_dir),
        "tables": [
            {"name": table.name, "rows": table.row_count, "columns": list(table.columns)}
            for table in manifest.tables
        ],
        "content_hash": manifest.content_hash,
        "data_quality_report": manifest.quality_report.model_dump()
        if manifest.quality_report is not None
        else None,
    }


@app.post("/runs", status_code=202)
async def create_run(req: CreateRun):
    run_id = new_run_id()
    options = req.model_dump(exclude={"question", "data_dir"})
    store.create(run_id, req.question, req.data_dir, options)

    if os.environ.get("ANALYST_DEV_INPROCESS"):
        loop = asyncio.get_running_loop()

        def publish(event: dict) -> None:  # worker thread -> event loop, safely
            asyncio.run_coroutine_threadsafe(bus.publish(run_id, event), loop)

        loop.run_in_executor(
            None, execute_run, run_id, req.question, req.data_dir, options,
            store, publish,
        )
    else:
        from arq import create_pool
        from arq.connections import RedisSettings

        pool = await create_pool(
            RedisSettings.from_dsn(os.environ.get("REDIS_URL", "redis://localhost:6379"))
        )
        await pool.enqueue_job("run_analysis", run_id, req.question, req.data_dir, options)
        await pool.aclose()

    return {"run_id": run_id, "status": "queued"}


@app.get("/runs")
async def list_runs():
    return store.list_runs()


@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    run = store.read(run_id)
    if run is None:
        raise HTTPException(404, f"no run {run_id}")
    return run


@app.websocket("/runs/{run_id}/stream")
async def stream_run(ws: WebSocket, run_id: str):
    run = store.read(run_id)
    if run is None:
        await ws.close(code=4404)
        return
    await ws.accept()
    try:
        # 1) replay: everything that already happened
        await ws.send_json({"type": "replay", "run": run})
        if run["status"] in ("done", "error"):
            await ws.send_json({"type": "stream_end"})
            return
        # 2) live: until the runner says stream_end
        async for event in bus.subscribe(run_id):
            await ws.send_json(event)
    except WebSocketDisconnect:
        pass
