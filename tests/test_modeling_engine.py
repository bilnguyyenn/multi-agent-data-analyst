from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyst.data import build_manifest
from analyst.graph.state import Artifact, ExecutionOutcome
from analyst.modeling.engine import (
    ModelingRequest,
    _cv_splitter,
    _split,
    evaluate_modeling_artifacts,
    evaluate_modeling_request,
)


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


def test_temporal_holdout_keeps_simultaneous_observations_together():
    # A row-count cut would put part of the final 30-row cohort in training.
    frame = pd.DataFrame({"when": np.repeat(pd.date_range("2025-01-01", periods=4), 30)})
    target = pd.Series(np.arange(len(frame), dtype=float))
    request = ModelingRequest(
        task_type="regression", table="observations", target="value", features=["x"],
        split_strategy="temporal", time_column="when",
    )

    train, test, _, _, diagnostics = _split(request, frame, target)

    assert train["when"].max() < test["when"].min()
    assert (len(train), len(test)) == (90, 30)
    assert diagnostics["train_time_max"] < diagnostics["test_time_min"]


def test_temporal_cv_never_splits_timestamp_cohorts():
    frame = pd.DataFrame({"when": np.repeat(pd.date_range("2025-01-01", periods=9), 17)})
    target = pd.Series(np.arange(len(frame), dtype=float))
    request = ModelingRequest(
        task_type="regression", table="observations", target="value", features=["x"],
        split_strategy="temporal", time_column="when",
    )
    cv, _, folds = _cv_splitter(request, target, frame)
    splits = cv.split(frame) if hasattr(cv, "split") else cv

    boundaries = [(frame.iloc[a]["when"].max(), frame.iloc[b]["when"].min())
                  for a, b in splits]
    assert len(boundaries) == folds
    assert all(train_end < validation_start for train_end, validation_start in boundaries)


def test_temporal_split_rejects_a_single_timestamp():
    frame = pd.DataFrame({"when": ["2025-01-01"] * 100})
    request = ModelingRequest(
        task_type="regression", table="observations", target="value", features=["x"],
        split_strategy="temporal", time_column="when",
    )
    with pytest.raises(ValueError, match="distinct timestamps"):
        _split(request, frame, pd.Series(np.arange(100)))


def test_temporal_cv_rejects_insufficient_distinct_training_times():
    frame = pd.DataFrame({"when": ["2025-01-01"] * 30 + ["2025-01-02"] * 30})
    request = ModelingRequest(
        task_type="regression", table="observations", target="value", features=["x"],
        split_strategy="temporal", time_column="when",
    )
    with pytest.raises(ValueError, match="three distinct training timestamps"):
        _cv_splitter(request, pd.Series(np.arange(len(frame))), frame)


def test_nonfinite_cv_scores_fail_visibly(modeling_dataset, monkeypatch):
    monkeypatch.setattr(
        "analyst.modeling.engine.cross_val_score", lambda *args, **kwargs: np.full(5, np.nan)
    )
    evidence = evaluate_modeling_request("cv", {
        "task_type": "regression", "table": "observations", "target": "value",
        "features": ["x"], "models": ["ridge"],
    }, modeling_dataset)

    assert evidence.status == "invalid"
    assert "non-finite cross-validation" in evidence.warnings[0]


def test_changing_only_holdout_labels_cannot_change_model_selection(modeling_dataset, tmp_path):
    frame = pd.read_csv(next(t.path for t in modeling_dataset.tables if t.name == "observations"))
    frame["shifted_value"] = frame["value"].copy()
    frame.loc[frame.index[-160:], "shifted_value"] *= -10
    frame.to_csv(tmp_path / "observations.csv", index=False)
    dataset = build_manifest(tmp_path)
    request = {
        "task_type": "regression", "table": "observations", "features": ["x", "segment"],
        "models": ["ridge", "random_forest"], "split_strategy": "temporal",
        "time_column": "event_date", "cv_folds": 3, "bootstrap_iterations": 100,
    }

    original = evaluate_modeling_request("original", {**request, "target": "value"}, dataset)
    changed = evaluate_modeling_request("changed", {**request, "target": "shifted_value"}, dataset)

    assert original.status == changed.status == "computed"
    assert original.candidates == changed.candidates
    assert original.selected_model == changed.selected_model
    assert original.diagnostics["train_primary_metric"] == changed.diagnostics["train_primary_metric"]
    assert original.diagnostics["test_primary_metric"] < changed.diagnostics["test_primary_metric"]
