"""Run driver: executes the agent graph and narrates it as a stream of events.

Shared by both execution paths (dev-mode background thread and the arq worker) so
the event protocol the UI sees is identical either way.

Uses graph.stream(stream_mode="values"): LangGraph yields the FULL merged state
after every superstep, so each publish carries a consistent snapshot of task
statuses — the UI never has to reassemble partial updates.

Event protocol (one JSON object per message):
  {"type": "status",      "status": "running|done|error"}
  {"type": "node_update", "tasks": [...], "new_events": [...]}
  {"type": "answer",      "answer": str, "claims": [...]}
  {"type": "stream_end"}
"""
from __future__ import annotations

import traceback
import time
import uuid
from typing import Callable

from analyst.api.runstore import RunStore
from analyst.data import build_manifest
from analyst.followups import FollowupContext
from analyst.graph.control_loop import build_graph
from analyst.graph.state import Budgets
from analyst.llm.client import make_client
from analyst.reporting import dataset_provenance
from analyst.sandbox.runner import make_runner


def _tasks_snapshot(dag: dict) -> list[dict]:
    return [
        {
            "id": t.id, "goal": t.goal, "kind": t.kind, "status": t.status,
            "depends_on": t.depends_on, "attempts": t.attempts,
            "revisions": t.revisions, "reuse_of": t.reuse_of,
        }
        for t in dag.values()
    ]


def _council_snapshot(state: dict) -> tuple[list[dict], list[dict], dict | None]:
    round_number = state.get("council_round", 0)
    findings = [
        finding.model_dump()
        for key, finding in state.get("council_findings", {}).items()
        if key.startswith(f"{round_number}:")
    ]
    decisions = [decision.model_dump() for decision in state.get("council_decisions", [])]
    routing = state.get("council_routing")
    return findings, decisions, routing.model_dump() if routing is not None else None


def _statistics_snapshot(state: dict) -> list[dict]:
    return [item.model_dump() for item in state.get("statistical_evidence", {}).values()]


def _predictive_snapshot(state: dict) -> list[dict]:
    return [item.model_dump() for item in state.get("predictive_evidence", {}).values()]


def _quality_snapshot(state: dict) -> dict | None:
    dataset = state.get("dataset")
    report = dataset.quality_report if dataset is not None else None
    return report.model_dump() if report is not None else None


def execute_run(
    run_id: str,
    question: str,
    data_dir: str,
    options: dict,
    store: RunStore,
    publish: Callable[[dict], None],
) -> None:
    """Synchronous; callers wrap it in a thread (dev mode) or arq job (prod).
    `publish` must be thread-safe for the caller's event loop."""
    try:
        store.update(run_id, status="running", started_at=time.time())
        publish({"type": "status", "status": "running"})

        dataset = build_manifest(data_dir)
        expected_hash = options.get("expected_dataset_hash")
        context = (FollowupContext.model_validate(options["followup_context"])
                   if options.get("followup_context") else None)
        if context is not None and expected_hash != context.dataset_hash:
            raise ValueError("follow-up context does not match the expected dataset fingerprint")
        if expected_hash and dataset.content_hash != expected_hash:
            raise ValueError("dataset changed while this follow-up was queued; start a new analysis")
        store.update(run_id, dataset_provenance=dataset_provenance(dataset))

        llm = make_client(provider=options.get("provider"))
        sandbox = make_runner(options.get("sandbox"))
        memory = None
        if not options.get("no_memory"):
            from analyst.memory.store import MemoryStore

            memory = MemoryStore()
        graph = build_graph(
            llm, sandbox,
            exec_timeout_s=options.get("timeout", 60),
            critic_enabled=not options.get("no_critic", True),
            validator_enabled=not options.get("no_validator"),
            council_mode=(
                "off" if options.get("no_council")
                else options.get("council_mode", "off")
            ),
            memory=memory,
        )

        initial = {
            "run_id": run_id,
            "question": question,
            "dataset": dataset,
            "budgets": Budgets(),
        }
        if context is not None:
            initial["followup_context"] = context
        seen_events = 0
        final = None
        for state in graph.stream(initial, stream_mode="values",
                                  config={"recursion_limit": 120}):
            final = state
            events = state.get("events", [])
            new = events[seen_events:]
            seen_events = len(events)
            for e in new:
                store.append_event(run_id, e)
            council_findings, council_decisions, council_routing = _council_snapshot(state)
            publish({
                "type": "node_update",
                "tasks": _tasks_snapshot(state.get("dag", {})),
                "new_events": new,
                "council_findings": council_findings,
                "council_decisions": council_decisions,
                "council_routing": council_routing,
                "statistical_evidence": _statistics_snapshot(state),
                "predictive_evidence": _predictive_snapshot(state),
                "data_quality_report": _quality_snapshot(state),
            })

        answer = (final or {}).get("answer") or "(no answer produced)"
        claims = [c.model_dump() for c in (final or {}).get("claims", [])]
        council_findings, council_decisions, council_routing = _council_snapshot(final or {})
        statistical_evidence = _statistics_snapshot(final or {})
        predictive_evidence = _predictive_snapshot(final or {})
        data_quality_report = _quality_snapshot(final or {})
        store.update(
            run_id, status="done", finished_at=time.time(), answer=answer, claims=claims,
            dag={t["id"]: t for t in _tasks_snapshot((final or {}).get("dag", {}))},
            council_findings=council_findings,
            council_decisions=council_decisions,
            council_routing=council_routing,
            statistical_evidence=statistical_evidence,
            predictive_evidence=predictive_evidence,
            data_quality_report=data_quality_report,
        )
        publish({
            "type": "answer",
            "answer": answer,
            "claims": claims,
            "council_findings": council_findings,
            "council_decisions": council_decisions,
            "council_routing": council_routing,
            "statistical_evidence": statistical_evidence,
            "predictive_evidence": predictive_evidence,
            "data_quality_report": data_quality_report,
        })
        publish({"type": "status", "status": "done"})
    except Exception as e:  # a crashed run must still tell the client it died
        store.update(run_id, status="error", finished_at=time.time(), answer=f"run crashed: {e}")
        publish({"type": "status", "status": "error",
                 "detail": f"{e}\n{traceback.format_exc()[-1500:]}"})
    finally:
        publish({"type": "stream_end"})


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]
