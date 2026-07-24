"""Pure DAG-scheduling logic, extracted from the graph node so it unit-tests
without LLMs, sandboxes, or LangGraph. The scheduler is deterministic code —
one of the deliberate 'no LLM where code will do' decisions."""
from __future__ import annotations

from analyst.graph.state import TaskSpec


def cascade_failures(dag: dict[str, TaskSpec]) -> dict[str, TaskSpec]:
    """Tasks whose dependencies failed can never run -> mark failed (transitively)."""
    updates: dict[str, TaskSpec] = {}
    changed = True
    while changed:
        changed = False
        merged = {**dag, **updates}
        for t in merged.values():
            if t.status == "pending" and any(
                merged[d].status == "failed" for d in t.depends_on if d in merged
            ):
                updates[t.id] = t.model_copy(update={"status": "failed"})
                changed = True
    return updates


def ready_set(dag: dict[str, TaskSpec]) -> list[str]:
    """Pending tasks whose dependencies are all done. Deterministic order."""
    return sorted(
        t.id
        for t in dag.values()
        if t.status == "pending"
        and all(dag[d].status == "done" for d in t.depends_on if d in dag)
    )
