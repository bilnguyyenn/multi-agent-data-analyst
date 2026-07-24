"""Parallel council branches, deterministic agreement, and bounded adjudication."""
import json
import threading
import uuid

import pytest

from analyst.data import build_manifest
from analyst.eval.data_gen import generate
from analyst.graph.control_loop import build_graph
from analyst.graph.state import Budgets, CouncilFinding
from analyst.llm.replay import ReplayClient
from analyst.sandbox.runner import make_runner

PLAN = json.dumps({
    "tasks": [
        {"id": "t1", "goal": "count work orders", "depends_on": [], "kind": "python"}
    ]
})

CODE_RAW = '''```python
df = pd.read_csv(os.path.join(os.environ["DATA_DIR"], "work_orders.csv"))
json.dump({"answer": f"There are {len(df)} work orders.", "data": {"n": len(df)}},
          open("result.json", "w"))
```'''

CODE_DEDUPED = '''```python
df = pd.read_csv(os.path.join(os.environ["DATA_DIR"], "work_orders.csv"))
df = df.drop_duplicates(subset=["wo_id"])
json.dump({"answer": f"There are {len(df)} unique work orders.",
           "data": {"n": len(df)}}, open("result.json", "w"))
```'''

CODE_WITH_MODEL = '''```python
import json
json.dump({
    "answer": "Requested an out-of-sample quantity prediction model.",
    "data": {"target": "qty_produced"},
    "modeling_requests": [{
        "id": "quantity_model",
        "task_type": "regression",
        "table": "work_orders",
        "target": "qty_produced",
        "features": ["shift"],
        "split_strategy": "temporal",
        "time_column": "wo_date",
        "models": ["ridge"],
        "primary_metric": "rmse",
        "bootstrap_iterations": 100
    }]
}, open("result.json", "w"))
```'''


def finding(verdict="support", specialist="metric", *, recompute=False):
    return json.dumps({
        "verdict": verdict,
        "conclusion": "The dataset contains 3000 work orders.",
        "answer_key": "3000_work_orders",
        "evidence_refs": ["t1"],
        "risks": ["Duplicate ids were not checked"] if recompute else [],
        "confidence": 0.9 if verdict == "support" else 0.6,
        "recommended_action": "recompute" if recompute else "accept",
        "target_task_id": "t1" if recompute else None,
        "instruction": "Deduplicate wo_id before counting." if recompute else "",
    })


ADJUDICATE_RECOMPUTE = json.dumps({
    "action": "recompute",
    "conclusion": "The count needs a uniqueness check.",
    "rationale": "The executed artifact counted rows without testing the entity key.",
    "dissent": ["Metric analyst accepted the raw row count."],
    "task_id": "t1",
    "instruction": "Deduplicate wo_id before counting and report the unique count.",
})

VALIDATE_PASS = json.dumps({
    "claims": [
        {"task_id": "t1", "claim": "3000 work orders", "verdict": "pass", "note": ""}
    ]
})


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("council-data") / "sample"
    generate(data_dir, seed=7)
    return build_manifest(data_dir)


@pytest.fixture()
def sandbox(monkeypatch):
    monkeypatch.setenv("ANALYST_SANDBOX", "subprocess")
    return make_runner("subprocess")


def invoke(script, dataset, sandbox):
    llm = ReplayClient(cassette_dir=None, script=script)
    graph = build_graph(
        llm,
        sandbox,
        critic_enabled=False,
        validator_enabled=True,
        council_enabled=True,
    )
    state = graph.invoke(
        {
            "run_id": uuid.uuid4().hex[:8],
            "question": "how many work orders are there?",
            "dataset": dataset,
            "budgets": Budgets(),
        },
        config={"recursion_limit": 120},
    )
    return state, llm


class ParallelProbeClient:
    """A barrier makes the test fail if council branches run sequentially."""

    def __init__(self, script):
        self.inner = ReplayClient(cassette_dir=None, script=script)
        self.barrier = threading.Barrier(3, timeout=3)
        self.council_users = []
        self.lock = threading.Lock()

    @property
    def _script_pos(self):
        return self.inner._script_pos

    def complete(self, *, role, system, user):
        if role.startswith("council_"):
            with self.lock:
                self.council_users.append(user)
            self.barrier.wait()
        return self.inner.complete(role=role, system=system, user=user)


def invoke_with_client(llm, dataset, sandbox):
    graph = build_graph(
        llm,
        sandbox,
        critic_enabled=False,
        validator_enabled=True,
        council_enabled=True,
    )
    state = graph.invoke(
        {
            "run_id": uuid.uuid4().hex[:8],
            "question": "how many work orders are there?",
            "dataset": dataset,
            "budgets": Budgets(),
        },
        config={"recursion_limit": 120},
    )
    return state, llm


def test_three_supporting_specialists_form_consensus(dataset, sandbox):
    script = {
        "planner": [{"text": PLAN}],
        "executor": [{"text": CODE_RAW}],
        "council_metric": [{"text": finding(specialist="metric")}],
        "council_statistical": [{"text": finding(specialist="statistical")}],
        "council_data_quality": [{"text": finding(specialist="data_quality")}],
        "validator": [{"text": VALIDATE_PASS}],
    }

    state, llm = invoke_with_client(ParallelProbeClient(script), dataset, sandbox)

    assert len(state["council_findings"]) == 3
    assert state["council_decisions"][-1].status == "consensus"
    assert state["council_complete"] is True
    assert "[council: consensus]" in state["answer"]
    assert "adjudicator" not in (llm._script_pos or {})


def test_council_reviews_computed_predictive_evidence_once(
    dataset, sandbox, monkeypatch
):
    from analyst.graph import control_loop

    real_evaluator = control_loop.evaluate_modeling_artifacts
    calls = 0

    def counted_evaluator(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_evaluator(*args, **kwargs)

    monkeypatch.setattr(control_loop, "evaluate_modeling_artifacts", counted_evaluator)
    script = {
        "planner": [{"text": PLAN}],
        "executor": [{"text": CODE_WITH_MODEL}],
        "council_metric": [{"text": finding(specialist="metric")}],
        "council_statistical": [{"text": finding(specialist="statistical")}],
        "council_data_quality": [{"text": finding(specialist="data_quality")}],
        "validator": [{"text": VALIDATE_PASS}],
    }
    client = ParallelProbeClient(script)

    state, llm = invoke_with_client(client, dataset, sandbox)

    assert calls == 1
    assert state["predictive_evidence"]["t1:quantity_model"].status == "computed"
    assert len(llm.council_users) == 3
    assert all("Deterministic predictive evidence" in user for user in llm.council_users)
    assert all('"primary_metric": "rmse"' in user for user in llm.council_users)


def test_adaptive_mode_skips_council_for_simple_lookup(dataset, sandbox):
    llm = ReplayClient(
        cassette_dir=None,
        script={
            "planner": [{"text": PLAN}],
            "executor": [{"text": CODE_RAW}],
            "validator": [{"text": VALIDATE_PASS}],
        },
    )
    graph = build_graph(
        llm,
        sandbox,
        critic_enabled=False,
        validator_enabled=True,
        council_mode="adaptive",
    )
    state = graph.invoke(
        {
            "run_id": uuid.uuid4().hex[:8],
            "question": "How many work orders are in the dataset?",
            "dataset": dataset,
            "budgets": Budgets(),
        },
        config={"recursion_limit": 120},
    )

    assert state["council_routing"].invoked is False
    assert state["council_complete"] is True
    assert not state.get("council_findings")
    assert not any(role.startswith("council_") for role in (llm._script_pos or {}))


def test_disagreement_triggers_one_recompute_then_new_council_round(dataset, sandbox):
    script = {
        "planner": [{"text": PLAN}],
        "executor": [{"text": CODE_RAW}, {"text": CODE_DEDUPED}],
        "council_metric": [
            {"text": finding(specialist="metric")},
            {"text": finding(specialist="metric")},
        ],
        "council_statistical": [
            {"text": finding("challenge", "statistical", recompute=True)},
            {"text": finding(specialist="statistical")},
        ],
        "council_data_quality": [
            {"text": finding(specialist="data_quality")},
            {"text": finding(specialist="data_quality")},
        ],
        "adjudicator": [{"text": ADJUDICATE_RECOMPUTE}],
        "validator": [{"text": VALIDATE_PASS}],
    }

    state, llm = invoke(script, dataset, sandbox)

    assert [d.action for d in state["council_decisions"]] == ["recompute", "accept"]
    assert state["council_decisions"][-1].status == "consensus"
    assert state["council_round"] == 1
    assert len(state["council_findings"]) == 6
    assert (llm._script_pos or {})["executor"] == 2
    assert state["budgets"].council_recomputes_left == 0


def test_answer_key_disagreement_requires_adjudication():
    findings = [
        CouncilFinding(
            specialist="metric", verdict="support", conclusion="Day wins",
            answer_key="day", confidence=0.8,
        ),
        CouncilFinding(
            specialist="statistical", verdict="support", conclusion="Night wins",
            answer_key="night", confidence=0.8,
        ),
        CouncilFinding(
            specialist="data_quality", verdict="support", conclusion="Day wins",
            answer_key="day", confidence=0.8,
        ),
    ]
    from analyst.agents.council import requires_adjudication

    assert requires_adjudication(findings) is True
