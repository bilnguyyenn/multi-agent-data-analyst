"""Phase (c): statistical validator paths — pass / caveat / fail-recompute /
fail-open — with the critic disabled to isolate the validator's loop."""
import uuid

import pytest

from analyst.data import build_manifest
from analyst.eval.data_gen import generate
from analyst.graph.control_loop import _validator_retry_feedback, build_graph
from analyst.graph.state import Budgets
from analyst.llm.replay import ReplayClient
from analyst.sandbox.runner import make_runner

PLAN_ONE = """{"tasks": [
  {"id": "t1", "goal": "defect rate per shift", "depends_on": [], "kind": "python"}
]}"""

CODE_OK = '''```python
import os, json
import pandas as pd
wo = pd.read_csv(os.path.join(os.environ["DATA_DIR"], "work_orders.csv"))
counts = wo.groupby("shift")["qty_produced"].sum()
json.dump({"answer": "Computed production per shift.",
           "data": counts.to_dict()}, open("result.json", "w"))
```'''

CODE_WITH_STATS = '''```python
import json
json.dump({
    "answer": "Day and evening defect rates were compared.",
    "data": {"day_rate": 0.10, "evening_rate": 0.15},
    "statistical_tests": [{
        "id": "day_vs_evening",
        "kind": "two_proportion",
        "groups": [
            {"label": "day", "successes": 20, "n": 200},
            {"label": "evening", "successes": 30, "n": 200}
        ]
    }]
}, open("result.json", "w"))
```'''

CODE_WITH_STRATIFIED_STATS = '''```python
import json
json.dump({
    "answer": "Crude and complexity-adjusted shift rates were compared.",
    "data": {"crude_winner": "night"},
    "statistical_tests": [{
        "id": "shift_mix_adjusted",
        "kind": "stratified_proportion",
        "direction": "lower",
        "standardization": "pooled",
        "cells": [
            {"group": "day", "stratum": "low", "successes": 6, "n": 1000},
            {"group": "day", "stratum": "high", "successes": 40, "n": 1000},
            {"group": "night", "stratum": "low", "successes": 24, "n": 3000},
            {"group": "night", "stratum": "high", "successes": 6, "n": 100}
        ]
    }]
}, open("result.json", "w"))
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

V_PASS = '{"claims": [{"task_id": "t1", "claim": "production per shift computed", "verdict": "pass", "note": ""}]}'
V_CAVEAT = '{"claims": [{"task_id": "t1", "claim": "night shift has lowest defect rate", "verdict": "caveat", "note": "shifts run different product mixes; rate comparison is confounded by job complexity"}]}'
V_FAIL = '{"claims": [{"task_id": "t1", "claim": "night is best", "verdict": "fail", "note": "must compare within complexity classes, not raw rates"}]}'


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
    graph = build_graph(
        llm, sandbox, exec_timeout_s=60, critic_enabled=False, validator_enabled=True
    )
    return graph.invoke(
        {"run_id": uuid.uuid4().hex[:8], "question": "q", "dataset": dataset,
         "budgets": Budgets()},
        config={"recursion_limit": 60},
    )


def test_pass_is_silent(dataset, sandbox):
    state = run(
        {"planner": [{"text": PLAN_ONE}], "executor": [{"text": CODE_OK}],
         "validator": [{"text": V_PASS}]},
        dataset, sandbox,
    )
    assert state["dag"]["t1"].status == "done"
    assert state["claims"][0].verdict == "pass"
    assert "caveat" not in state["answer"].lower()


def test_caveat_rides_into_the_answer(dataset, sandbox):
    state = run(
        {"planner": [{"text": PLAN_ONE}], "executor": [{"text": CODE_OK}],
         "validator": [{"text": V_CAVEAT}]},
        dataset, sandbox,
    )
    assert "Statistical caveats" in state["answer"]
    assert "confounded" in state["answer"]


def test_fail_triggers_recompute_then_revalidation(dataset, sandbox):
    state = run(
        {"planner": [{"text": PLAN_ONE}],
         "executor": [{"text": CODE_OK}, {"text": CODE_OK}],
         "validator": [{"text": V_FAIL}, {"text": V_PASS}]},
        dataset, sandbox,
    )
    assert state["dag"]["t1"].status == "done"
    assert state["budgets"].validations_left == 0   # two passes consumed
    assert state["claims"][-1].verdict == "pass"    # second pass published
    assert "caveat" not in state["answer"].lower()


def test_fail_on_last_pass_becomes_caveat(dataset, sandbox):
    state = run(
        {"planner": [{"text": PLAN_ONE}],
         "executor": [{"text": CODE_OK}, {"text": CODE_OK}],
         "validator": [{"text": V_FAIL}, {"text": V_FAIL}]},
        dataset, sandbox,
    )
    # budget exhausted with an unfixed fail -> objection caveat, not silence
    assert "[stats]" in state["answer"]
    assert "complexity classes" in state["answer"]


def test_unparseable_validator_fails_open(dataset, sandbox):
    state = run(
        {"planner": [{"text": PLAN_ONE}], "executor": [{"text": CODE_OK}],
         "validator": [{"text": "all good probably"}]},
        dataset, sandbox,
    )
    assert state["dag"]["t1"].status == "done"
    assert any("defaulted to pass" in c.note for c in state["claims"])


def test_deterministic_statistics_reach_validator_state_and_answer(dataset, sandbox):
    state = run(
        {"planner": [{"text": PLAN_ONE}], "executor": [{"text": CODE_WITH_STATS}],
         "validator": [{"text": V_PASS}]},
        dataset, sandbox,
    )

    evidence = state["statistical_evidence"]["t1:day_vs_evening"]
    assert evidence.status == "computed"
    assert evidence.test.name == "two-proportion z-test"
    assert "Deterministic statistical evidence" in state["answer"]
    assert "Cohen's h" in state["answer"]
    validator_event = next(event for event in state["events"] if event["node"] == "validator")
    assert validator_event["deterministic_tests"] == 1
    assert validator_event["invalid_tests"] == 0


def test_predictive_modeling_reaches_validator_state_and_answer(dataset, sandbox):
    state = run(
        {"planner": [{"text": PLAN_ONE}], "executor": [{"text": CODE_WITH_MODEL}],
         "validator": [{"text": V_PASS}]},
        dataset, sandbox,
    )

    evidence = state["predictive_evidence"]["t1:quantity_model"]
    assert evidence.status == "computed"
    assert evidence.selected_model == "ridge"
    assert evidence.split_strategy == "temporal"
    assert any(metric.name == "rmse" for metric in evidence.metrics)
    assert "Predictive modeling evidence" in state["answer"]
    assert "baseline=" in state["answer"]
    validator_event = next(event for event in state["events"] if event["node"] == "validator")
    assert validator_event["predictive_models"] == 1
    assert validator_event["invalid_models"] == 0


def test_confounding_failure_recomputes_with_stratified_evidence(dataset, sandbox):
    state = run(
        {"planner": [{"text": PLAN_ONE}],
         "executor": [{"text": CODE_OK}, {"text": CODE_WITH_STRATIFIED_STATS}],
         "validator": [{"text": V_FAIL}, {"text": V_PASS}]},
        dataset, sandbox,
    )

    evidence = state["statistical_evidence"]["t1:shift_mix_adjusted"]
    assert evidence.status == "computed"
    assert evidence.diagnostics["crude_ranking"][0] == "night"
    assert evidence.diagnostics["adjusted_ranking"][0] == "day"
    assert evidence.diagnostics["simpson_reversal"] is True
    assert "winner reversed after adjustment" in state["answer"]


def test_confounding_retry_feedback_requires_stratified_contract():
    feedback = _validator_retry_feedback(
        "Raw rates are confounded because shifts run different complexity mixes"
    )

    assert "stratified_proportion" in feedback
    assert "group × confounder stratum" in feedback
    assert "complete cells grid" in feedback
