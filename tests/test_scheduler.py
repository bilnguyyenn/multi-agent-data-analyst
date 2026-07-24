from analyst.graph.scheduler import cascade_failures, ready_set
from analyst.graph.state import TaskSpec


def dag_of(*specs):
    return {s.id: s for s in specs}


def test_ready_set_respects_dependencies():
    dag = dag_of(
        TaskSpec(id="a", goal="", status="done"),
        TaskSpec(id="b", goal="", depends_on=["a"]),
        TaskSpec(id="c", goal="", depends_on=["b"]),
        TaskSpec(id="d", goal=""),  # independent -> ready in parallel with b
    )
    assert ready_set(dag) == ["b", "d"]


def test_ready_set_empty_when_all_done():
    dag = dag_of(TaskSpec(id="a", goal="", status="done"))
    assert ready_set(dag) == []


def test_cascade_failures_transitive():
    dag = dag_of(
        TaskSpec(id="a", goal="", status="failed"),
        TaskSpec(id="b", goal="", depends_on=["a"]),
        TaskSpec(id="c", goal="", depends_on=["b"]),
        TaskSpec(id="d", goal=""),  # unaffected
    )
    updates = cascade_failures(dag)
    assert set(updates) == {"b", "c"}
    assert all(t.status == "failed" for t in updates.values())


def test_running_tasks_not_redispatched():
    dag = dag_of(TaskSpec(id="a", goal="", status="running"))
    assert ready_set(dag) == []
