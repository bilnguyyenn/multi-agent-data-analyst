"""Single-user operations API and recoverable local worker lifecycle."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager, suppress
from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from analyst.operations.metrics import REGISTRY, VERSION, MetricRequest, resolve_question
from analyst.operations.service import Operations, Playbook, Refresh
from analyst.reporting import render_report, report_payload

router = APIRouter(prefix="/operations", tags=["operations"])


@lru_cache(maxsize=4)
def _store(root: str) -> Operations:
    return Operations(root)


def service() -> Operations:
    return _store(os.environ.get("ANALYST_OPERATIONS_DIR", ".operations"))


@asynccontextmanager
async def lifespan(app):
    async def worker():
        while True:
            for identity in service().pending():
                await asyncio.to_thread(service().execute, identity)
            await asyncio.sleep(0.25)

    task = asyncio.create_task(worker())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def invoke(method, *args):
    try:
        return method(*args)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(409, str(exc)) from exc


class SnapshotInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_path: str = Field(min_length=1, max_length=4096)
    label: str = Field(default="Uploaded retail data", max_length=200)


class ResolveInput(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    metric: str | None = None


class InterpretInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    policy: Literal["single", "standard", "high"] = "standard"
    prior_request: MetricRequest | None = None


@router.post("/interpret")
async def interpret(request: InterpretInput):
    from analyst.config import load_routing
    from analyst.llm.client import make_client
    from analyst.operations.agent import propose
    from analyst.operations.preflight import local_preflight

    check = await asyncio.to_thread(local_preflight)
    if not check["allowed"]:
        raise HTTPException(409, "; ".join(check["reasons"]))
    prior = request.prior_request.model_dump(mode="json") if request.prior_request else None
    routing = load_routing()
    routing.limits.update(max_output_tokens=256, request_timeout_s=90)
    result = await asyncio.to_thread(
        invoke, propose, make_client("local", routing), request.question, request.policy, prior
    )
    return {key: value for key, value in result.items() if key != "calls"}


@router.get("/metrics")
def metric_registry():
    return {"version": VERSION, "metrics": REGISTRY}


@router.post("/resolve")
def resolve(request: ResolveInput):
    return invoke(resolve_question, request.question, request.metric)


@router.get("/snapshots")
def snapshots():
    return service().list("snapshots")


@router.post("/snapshots", status_code=201)
async def ingest(request: SnapshotInput):
    return await asyncio.to_thread(
        invoke, service().ingest, request.source_path, {"label": request.label}
    )


@router.get("/playbooks")
def playbooks():
    return service().list("playbooks")


@router.post("/playbooks", status_code=201)
def save_playbook(request: Playbook):
    return invoke(service().save_playbook, request.model_dump(mode="json"))


@router.get("/jobs")
def jobs():
    return service().list("jobs")


@router.post("/jobs", status_code=202)
def refresh(request: Refresh):
    return invoke(service().submit, request.model_dump(mode="json"))


@router.get("/jobs/{identity}")
def job(identity: str):
    result = invoke(service().get, "jobs", identity)
    return {
        **result,
        "evidence": [invoke(service().get, "evidence", ref) for ref in result["evidence_ids"]],
    }


@router.post("/jobs/{identity}/cancel")
def cancel(identity: str):
    return invoke(service().cancel, identity)


@router.post("/jobs/{identity}/retry")
def retry(identity: str):
    return invoke(service().retry, identity)


@router.get("/evidence/{identity}")
def evidence(identity: str):
    return invoke(service().get, "evidence", identity)


@router.get("/compare")
def compare(previous: str, current: str):
    return invoke(service().compare, previous, current)


@router.get("/jobs/{identity}/report")
def report(identity: str, format: str = "html"):
    if format not in {"html", "json"}:
        raise HTTPException(422, "format must be html or json")
    payload = report_payload(invoke(service().report_record, identity))
    content = (
        render_report(payload)
        if format == "html"
        else json.dumps(payload, indent=2, allow_nan=False)
    )
    return Response(
        content,
        media_type="text/html" if format == "html" else "application/json",
        headers={
            "Content-Disposition": f'attachment; filename="retail-review.{format}"',
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'",
        },
    )
