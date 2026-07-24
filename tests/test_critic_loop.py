"""Phase (b): every critic verdict path, in replay mode.

approve -> done | revise -> executor retry with instructions | replan -> planner
rewrites the task spec | budget exhaustion -> done-with-objection (caveat in answer).
"""
import uuid

import pytest

from analyst.data import build_manifest
from analyst.eval.data_gen import generate
from analyst.graph.control_loop import build_graph
from analyst.graph.state import Budgets
from analyst.llm.replay import ReplayClient
from analyst.sandbox.runner import make_runner

PLAN_ONE = """{"tasks": [
  {"id": "t1", "goal": "average downtime hours per machine", "depends_on": [], "kind": "python"}
]}"""

REPLAN_ONE = """{"tasks": [
  {"id": "t1", "goal": "average downtime hours per machine INCLUDING decommissioned machines", "depends_on": [], "kind": "python"}
]}"""

CODE_OK = '''```python
import os, json
import pandas as pd
df = pd.read_csv(os.path.join(os.environ["DATA_DIR"], "downtime_events.csv"))
avg = df.groupby("machine_id")["downtime_hours"].mean().round(2)
json.dump({"answer": "Computed average downtime per machine.",
           "data": avg.to_dict()}, open("result.json", "w"))
```'''

APPROVE = '{"verdict": "approve", "reason": "method is sound"}'
REVISE = '{"verdict": "revise", "reason": "uses mean over raw events, goal implies per-machine totals first", "feedback": "aggregate to per-machine totals before averaging"}'
REPLAN = '{"verdict": "replan", "reason": "task filters to active machines; survivorship bias against the question"}'


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("data") / "sample"
    generate(d, seed=7)
    return build_manifest(d)


@pytest.fixture()
def sandbox(monkeypatch):
    monkeypatch.setenv("ANALYST_SANDBOX", "subprocess")
    return make_runner("subprocess")


def run(script, dataset, sandbox):
    llm = ReplayClient(cassette_dir=None, script=script)
    # validator off: these tests isolate the CRITIC's loop
    graph = build_graph(
        llm, sandbox, exec_timeout_s=60, critic_enabled=True, validator_enabled=False
    )
    return graph.invoke(
        {"run_id": uuid.uuid4().hex[:8], "question": "q", "dataset": dataset,
         "budgets": Budgets()},
        config={"recursion_limit": 60},
    )


def test_approve_promotes_to_done(dataset, sandbox):
    state = run(
        {"planner": [{"text": PLAN_ONE}], "executor": [{"text": CODE_OK}],
         "critic": [{"text": APPROVE}]},
        dataset, sandbox,
    )
    assert state["dag"]["t1"].status == "done"
    assert state["verdicts"]["t1"].verdict == "approve"
    assert "Caveats" not in state["answer"]


def test_revise_sends_back_to_executor_with_instructions(dataset, sandbox):
    state = run(
        {"planner": [{"text": PLAN_ONE}],
         "executor": [{"text": CODE_OK}, {"text": CODE_OK}],
         "critic": [{"text": REVISE}, {"text": APPROVE}]},
        dataset, sandbox,
    )
    t1 = state["dag"]["t1"]
    assert t1.status == "done"
    assert t1.revisions == 1
    assert state["verdicts"]["t1"].verdict == "approve"  # latest verdict wins


def test_replan_rewrites_the_task_spec(dataset, sandbox):
    state = run(
        {"planner": [{"text": PLAN_ONE}, {"text": REPLAN_ONE}],
         "executor": [{"text": CODE_OK}, {"text": CODE_OK}],
         "critic": [{"text": REPLAN}, {"text": APPROVE}]},
        dataset, sandbox,
    )
    t1 = state["dag"]["t1"]
    assert t1.status == "done"
    assert "decommissioned" in t1.goal          # planner actually changed the spec
    assert state["budgets"].replans_left == 1   # budget consumed


def test_exhausted_revisions_become_a_caveat(dataset, sandbox):
    # max_revisions=2 -> third "revise" verdict accepts the result WITH an objection
    state = run(
        {"planner": [{"text": PLAN_ONE}],
         "executor": [{"text": CODE_OK}] * 3,
         "critic": [{"text": REVISE}] * 3},
        dataset, sandbox,
    )
    t1 = state["dag"]["t1"]
    assert t1.status == "done"          # degrades to honesty, not deadlock
    assert t1.revisions == 2
    assert "Caveats" in state["answer"]
    assert "mean over raw events" in state["answer"]


def test_unparseable_critic_fails_open(dataset, sandbox):
    state = run(
        {"planner": [{"text": PLAN_ONE}], "executor": [{"text": CODE_OK}],
         "critic": [{"text": "hmm, looks fine to me I guess?"}]},
        dataset, sandbox,
    )
    assert state["dag"]["t1"].status == "done"
    assert "unparseable" in state["verdicts"]["t1"].reason
