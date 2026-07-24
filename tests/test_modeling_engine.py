from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyst.data import build_manifest
from analyst.graph.state import Artifact, ExecutionOutcome
from analyst.modeling.engine import evaluate_modeling_artifacts, evaluate_modeling_request


@pytest.fixture(scope="module")
def modeling_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("modeling")
    rng = np.random.default_rng(17)
    n = 800
    entity = np.arange(n)
    x = rng.normal(size=n)
    segment = rng.choice(["north", "south", "west"], n)
    logits = 1.8 * x + 0.9 * (segment == "north") - 0.4
    probability = 1 / (1 + np.exp(-logits))
    target = rng.binomial(1, probability)
    value = 4.0 * x + 2.5 * (segment == "south") + rng.normal(scale=1.2, size=n)
    observations = pd.DataFrame(
        {
            "row_id": [f"R{item:04d}" for item in entity],
            "account_group": [f"G{item // 4:03d}" for item in entity],
            "event_date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "x": x,
            "segment": segment,
            "target": target,
            "value": value,
        }
    )
    attributes = pd.DataFrame(
        {
            "row_id": observations.row_id,
            "tenure": rng.integers(1, 20, n),
        }
    )
    events = pd.DataFrame(
        {
            "event_id": [f"E{item:05d}" for item in range(n * 2)],
            "row_id": np.repeat(observations.row_id.to_numpy(), 2),
        }
    )
    observations.to_csv(root / "observations.csv", index=False)
    attributes.to_csv(root / "attributes.csv", index=False)
    events.to_csv(root / "events.csv", index=False)
    return build_manifest(root)


def test_binary_classification_selects_on_train_and_scores_untouched_holdout(
    modeling_dataset,
):
    evidence = evaluate_modeling_request(
        "t1",
        {
            "id": "risk_model",
            "task_type": "classification",
            "table": "observations",
            "target": "target",
            "features": ["x", "segment", "tenure"],
            "joins": [
                {"table": "attributes", "on": ["row_id"], "validate": "one_to_one"}
            ],
            "models": ["logistic_regression", "random_forest"],
            "primary_metric": "roc_auc",
            "slice_by": ["segment"],
            "bootstrap_iterations": 100,
        },
        modeling_dataset,
    )

    assert evidence.status == "computed"
    assert evidence.selected_model in {"logistic_regression", "random_forest"}
    assert evidence.train_size == 640
    assert evidence.test_size == 160
    assert len(evidence.candidates) == 2
    assert all(candidate.folds == 5 for candidate in evidence.candidates)
    metrics = {metric.name: metric for metric in evidence.metrics}
    assert metrics["roc_auc"].value > metrics["roc_auc"].baseline_value
    assert metrics["roc_auc"].lower < metrics["roc_auc"].value < metrics["roc_auc"].upper
    assert evidence.feature_importance[0].feature == "x"
    assert {item.value for item in evidence.slices} == {"north", "south", "west"}
    assert evidence.diagnostics["baseline_improvement"] > 0


def test_regression_reports_error_metrics_baseline_and_importance(modeling_dataset):
    evidence = evaluate_modeling_request(
        "t2",
        {
            "id": "value_model",
            "task_type": "regression",
            "table": "observations",
            "target": "value",
            "features": ["x", "segment"],
            "models": ["ridge"],
            "primary_metric": "rmse",
            "bootstrap_iterations": 100,
        },
        modeling_dataset,
    )

    assert evidence.status == "computed"
    assert evidence.selected_model == "ridge"
    metrics = {metric.name: metric for metric in evidence.metrics}
    assert set(metrics) == {"rmse", "mae", "r2"}
    assert metrics["rmse"].value < metrics["rmse"].baseline_value
    assert metrics["r2"].value > 0.8
    assert evidence.feature_importance[0].feature == "x"


@pytest.mark.parametrize(
    ("split_fields", "diagnostic"),
    [
        ({"split_strategy": "temporal", "time_column": "event_date"}, "train_time_max"),
        ({"split_strategy": "group", "group_column": "account_group"}, "group_overlap"),
    ],
)
def test_temporal_and_group_splits_are_auditable(
    modeling_dataset, split_fields, diagnostic
):
    evidence = evaluate_modeling_request(
        "t3",
        {
            "task_type": "regression",
            "table": "observations",
            "target": "value",
            "features": ["x", "segment"],
            "models": ["ridge"],
            "bootstrap_iterations": 100,
            **split_fields,
        },
        modeling_dataset,
    )

    assert evidence.status == "computed"
    assert diagnostic in evidence.diagnostics
    if diagnostic == "group_overlap":
        assert evidence.diagnostics[diagnostic] == 0


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"features": ["target", "x"]}, "target cannot also be a feature"),
        ({"features": ["row_id", "x"]}, "identifier-like features are rejected"),
        ({"split_strategy": "temporal"}, "temporal split requires time_column"),
        ({"models": ["made_up_model"]}, "unsupported classification model"),
    ],
)
def test_invalid_or_leaky_model_specs_fail_visibly(modeling_dataset, update, message):
    request = {
        "task_type": "classification",
        "table": "observations",
        "target": "target",
        "features": ["x", "segment"],
        "models": ["logistic_regression"],
        "bootstrap_iterations": 100,
        **update,
    }

    evidence = evaluate_modeling_request("t4", request, modeling_dataset)

    assert evidence.status == "invalid"
    assert message in evidence.warnings[0]


def test_one_to_many_join_is_rejected_before_modeling(modeling_dataset):
    evidence = evaluate_modeling_request(
        "t5",
        {
            "task_type": "classification",
            "table": "observations",
            "target": "target",
            "features": ["x", "segment"],
            "joins": [
                {"table": "events", "on": ["row_id"], "validate": "one_to_one"}
            ],
            "models": ["logistic_regression"],
        },
        modeling_dataset,
    )

    assert evidence.status == "invalid"
    assert "join validation failed" in evidence.warnings[0]


def test_artifact_requests_are_collected_and_invalid_shape_is_visible(modeling_dataset):
    valid = Artifact(
        task_id="t1",
        code="",
        outcome=ExecutionOutcome(
            status="ok",
            result={
                "modeling_requests": {
                    "id": "value",
                    "task_type": "regression",
                    "table": "observations",
                    "target": "value",
                    "features": ["x"],
                    "models": ["ridge"],
                    "bootstrap_iterations": 100,
                }
            },
        ),
    )
    invalid = Artifact(
        task_id="t2",
        code="",
        outcome=ExecutionOutcome(
            status="ok", result={"modeling_requests": "not a request"}
        ),
    )

    evidence = evaluate_modeling_artifacts(
        {"t1": valid, "t2": invalid}, modeling_dataset
    )

    assert evidence["t1:value"].status == "computed"
    assert evidence["t2:model_1"].status == "invalid"
    assert evidence["t2:model_1"].warnings == [
        "modeling_requests must be a list or object"
    ]
