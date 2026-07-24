"""End-to-end: full LangGraph loop with SCRIPTED model responses and the
subprocess sandbox. Zero LLM calls, zero Docker — this is what CI runs.

The scripts are 'what the model would have said' — code included. Parallel
fan-out is covered by scheduler unit tests; these use a linear DAG so the
per-role call order (and thus the script) is deterministic.
"""
import uuid

import pytest

from analyst.data import build_manifest
from analyst.eval.data_gen import generate
from analyst.graph.control_loop import build_graph
from analyst.graph.state import Budgets
from analyst.llm.replay import ReplayClient
from analyst.sandbox.runner import make_runner

PLAN_TWO_STEP = """{"tasks": [
  {"id": "t1", "goal": "total downtime hours per machine", "depends_on": [], "kind": "python"},
  {"id": "t2", "goal": "identify the machine with the most downtime", "depends_on": ["t1"], "kind": "python"}
]}"""

CODE_T1 = '''```python
import os, json
import pandas as pd
df = pd.read_csv(os.path.join(os.environ["DATA_DIR"], "downtime_events.csv"))
per = df.groupby("machine_id")["downtime_hours"].sum().round(2)
json.dump({"answer": "Computed downtime hours per machine.",
           "data": per.to_dict()}, open("result.json", "w"))
```'''

# t2 consumes t1's result VIA THE DEPS FILE — exercising the file-based handoff
# (added after a live run where the model assumed files existed; now they do).
CODE_T2 = '''```python
import os, json
t1 = json.load(open(os.path.join(os.environ["DEPS_DIR"], "t1.json")))
per = t1["data"]
top = max(per, key=per.get)
json.dump({"answer": f"{top} had the most downtime: {per[top]:.1f} hours.",
           "data": {"machine": top, "hours": round(float(per[top]), 2)}},
          open("result.json", "w"))
```'''

PLAN_ONE_STEP = """{"tasks": [
  {"id": "t1", "goal": "count work orders", "depends_on": [], "kind": "python"}
]}"""

CODE_BROKEN = '''```python
import os, json
import pandas as pd
df = pd.read_csv(os.path.join(os.environ["DATA_DIR"], "work_orders.csv"))
raise ValueError("simulated bad first attempt")
```'''

CODE_FIXED = '''```python
import os, json
import pandas as pd
df = pd.read_csv(os.path.join(os.environ["DATA_DIR"], "work_orders.csv"))
json.dump({"answer": f"There are {len(df)} work orders.",
           "data": {"count": len(df)}}, open("result.json", "w"))
```'''


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("data") / "sample"
    generate(d, seed=7)
    return build_manifest(d)


@pytest.fixture()
def sandbox(monkeypatch):
    monkeypatch.setenv("ANALYST_SANDBOX", "subprocess")
    return make_runner("subprocess")


def invoke(script, dataset, sandbox):
    llm = ReplayClient(cassette_dir=None, script=script)
    # Both gates off: these tests cover the phase-(a) core loop, which is exactly
    # the fully-ablated configuration the eval harness compares against.
    graph = build_graph(
        llm, sandbox, exec_timeout_s=60, critic_enabled=False, validator_enabled=False
    )
    return graph.invoke(
        {"run_id": uuid.uuid4().hex[:8], "question": "q", "dataset": dataset,
         "budgets": Budgets()},
        config={"recursion_limit": 60},
    )


def test_two_step_dag_end_to_end(dataset, sandbox):
    script = {
        "planner": [{"text": PLAN_TWO_STEP}],
        "executor": [{"text": CODE_T1}, {"text": CODE_T2}],
    }
    state = invoke(script, dataset, sandbox)
    assert state["dag"]["t1"].status == "done"
    assert state["dag"]["t2"].status == "done"
    assert "most downtime" in state["answer"]
    assert "hours" in state["answer"]
    # t2's artifact carries the actual number, sourced from real sandbox execution
    top = state["artifacts"]["t2"].outcome.result["data"]
    assert top["hours"] > 0
    # audit trail recorded every node
    nodes = {e["node"] for e in state["events"]}
    assert {"plan", "schedule", "execute", "finalize"} <= nodes


def test_executor_retry_loop(dataset, sandbox):
    script = {
        "planner": [{"text": PLAN_ONE_STEP}],
        "executor": [{"text": CODE_BROKEN}, {"text": CODE_FIXED}],
    }
    state = invoke(script, dataset, sandbox)
    t1 = state["dag"]["t1"]
    assert t1.status == "done"
    assert t1.attempts == 1  # one failed attempt, then success
    assert state["artifacts"]["t1"].outcome.status == "ok"
    assert "work orders" in state["answer"]


def test_budget_exhaustion_reports_failure(dataset, sandbox):
    script = {
        "planner": [{"text": PLAN_ONE_STEP}],
        "executor": [{"text": CODE_BROKEN}] * 5,  # never fixes it
    }
    state = invoke(script, dataset, sandbox)
    assert state["dag"]["t1"].status == "failed"
    assert "Unresolved" in state["answer"]  # degrades to honesty, not silence
