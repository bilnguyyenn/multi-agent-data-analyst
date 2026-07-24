"""arq worker: the production execution path (phase g wires this into compose).

Why a queue at all: agent runs take minutes; HTTP requests should take milliseconds.
The API enqueues and returns; this process does the heavy lifting and narrates over
Redis pub/sub. Kill the worker mid-run and the API stays healthy — that isolation
is the point.

Run it:  arq analyst.api.worker.WorkerSettings
"""
from __future__ import annotations

import asyncio
import os

from arq.connections import RedisSettings

from analyst.api.events import RedisEventBus
from analyst.api.pgstore import make_run_store
from analyst.api.runner import execute_run


async def run_analysis(ctx: dict, run_id: str, question: str, data_dir: str,
                        options: dict) -> str:
    store = make_run_store()
    bus: RedisEventBus = ctx["bus"]
    loop = asyncio.get_running_loop()

    def publish(event: dict) -> None:
        asyncio.run_coroutine_threadsafe(bus.publish(run_id, event), loop)

    # execute_run is synchronous (sandbox subprocesses, sync LLM calls) — run it in
    # a thread so this worker can heartbeat and accept concurrent jobs.
    await asyncio.to_thread(
        execute_run, run_id, question, data_dir, options, store, publish
    )
    return run_id


async def startup(ctx: dict) -> None:
    ctx["bus"] = RedisEventBus()


class WorkerSettings:
    functions = [run_analysis]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(
        os.environ.get("REDIS_URL", "redis://localhost:6379")
    )
    max_jobs = 2          # agent runs are heavy; don't stampede the sandbox
    job_timeout = 1800    # 30 min hard cap per run
