"""Planner agent: question + schema -> task DAG (as data, validated).

Validation matters: a small model can emit cycles, dangling deps, or 12 tasks.
We validate structurally (cycle check via topological sort) BEFORE anything
executes — a malformed plan is a planner failure, not an executor mystery."""
from __future__ import annotations

from graphlib import CycleError, TopologicalSorter

from analyst.agents import prompts
from analyst.graph.state import DatasetManifest, TaskSpec
from analyst.llm.client import LLMClient
from analyst.llm.parsing import ModelOutputError, extract_json

MAX_TASKS = 5


class PlanError(RuntimeError):
    pass


def make_plan(
    llm: LLMClient,
    question: str,
    dataset: DatasetManifest,
    memory_hits: list | None = None,
) -> tuple[dict[str, TaskSpec], dict]:
    memory_section = ""
    if memory_hits:
        hits_txt = "\n".join(
            f'- memory_id={h.id} (similarity {h.similarity}): goal="{h.goal}"'
            for h in memory_hits
        )
        memory_section = prompts.PLANNER_MEMORY.format(hits=hits_txt)
    resp = llm.complete(
        role="planner",
        system=prompts.PLANNER_SYSTEM,
        user=prompts.PLANNER_USER.format(
            question=question, schema=dataset.schema_text(),
            memory_section=memory_section,
        ),
    )
    usage = {
        "model": resp.model,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
    }
    try:
        raw = extract_json(resp.text)
    except ModelOutputError as e:
        raise PlanError(str(e)) from e

    tasks = raw.get("tasks")
    if not tasks or not isinstance(tasks, list):
        raise PlanError(f"Planner emitted no tasks: {resp.text[:300]}")
    if len(tasks) > MAX_TASKS:
        tasks = tasks[:MAX_TASKS]  # trim rather than fail: budgets over perfection

    valid_hit_ids = {h.id for h in (memory_hits or [])}
    dag: dict[str, TaskSpec] = {}
    for t in tasks:
        kind = t.get("kind", "python")
        reuse_of = t.get("reuse_of")
        if kind == "reuse" and reuse_of not in valid_hit_ids:
            kind, reuse_of = "python", None  # hallucinated memory id -> recompute
        if kind not in ("python", "sql", "reuse"):
            kind = "python"
        spec = TaskSpec(
            id=str(t["id"]),
            goal=str(t["goal"]),
            depends_on=[str(d) for d in t.get("depends_on", [])],
            kind=kind,
            reuse_of=reuse_of,
        )
        dag[spec.id] = spec

    # Structural validation: no dangling deps, no cycles.
    for spec in dag.values():
        missing = [d for d in spec.depends_on if d not in dag]
        if missing:
            raise PlanError(f"Task {spec.id} depends on unknown task(s) {missing}")
    try:
        # list() matters: static_order() is lazy and only raises when consumed
        list(TopologicalSorter({k: set(v.depends_on) for k, v in dag.items()}).static_order())
    except CycleError as e:
        raise PlanError(f"Planner produced a cyclic DAG: {e}") from e

    return dag, usage


def make_replan(
    llm: LLMClient,
    question: str,
    dataset: DatasetManifest,
    rejected: dict[str, TaskSpec],
    reason: str,
    known_ids: set[str],
) -> tuple[dict[str, TaskSpec], dict]:
    """Critic-triggered subtree replan: rewrite the rejected tasks IN PLACE.

    Ids are pinned so edges from the rest of the DAG stay valid — the planner
    changes what the tasks *do*, not the graph's shape (a deliberately bounded
    form of replanning; full-DAG restructuring costs more than it buys here)."""
    tasks_txt = "\n".join(
        f'- id={t.id}, depends_on={t.depends_on}, goal="{t.goal}"' for t in rejected.values()
    )
    resp = llm.complete(
        role="planner",
        system=prompts.REPLAN_SYSTEM,
        user=prompts.REPLAN_USER.format(
            question=question,
            schema=dataset.schema_text(),
            reason=reason,
            tasks=tasks_txt,
        ),
    )
    usage = {
        "model": resp.model,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
    }
    try:
        raw = extract_json(resp.text)
    except ModelOutputError as e:
        raise PlanError(str(e)) from e

    out: dict[str, TaskSpec] = {}
    for t in raw.get("tasks", []):
        tid = str(t.get("id"))
        if tid not in rejected:
            continue  # planner may not invent new ids in replan mode
        old = rejected[tid]
        deps = [str(d) for d in t.get("depends_on", old.depends_on)]
        if any(d not in known_ids for d in deps):
            deps = old.depends_on  # keep valid edges over hallucinated ones
        out[tid] = TaskSpec(
            id=tid,
            goal=str(t.get("goal", old.goal)),
            depends_on=deps,
            kind=t.get("kind", old.kind) if t.get("kind") in ("python", "sql") else old.kind,
            status="pending",
        )
    missing = set(rejected) - set(out)
    if missing:
        raise PlanError(f"Replan did not cover rejected task(s): {sorted(missing)}")
    return out, usage
