"""Leakage-aware, deterministic predictive modeling over manifest CSV tables.

Executors request a governed model evaluation; they do not fit and grade arbitrary
models themselves. This layer owns splitting, training-only cross-validation,
baseline comparison, untouched-holdout metrics, bootstrap uncertainty, permutation
importance, and optional slice diagnostics.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analyst.graph.state import (
    Artifact,
    DatasetManifest,
    FeatureImportance,
    ModelCandidate,
    PredictiveEvidence,
    PredictiveMetric,
    SliceMetric,
)

TaskType = Literal["classification", "regression"]
SplitStrategy = Literal["random", "temporal", "group"]
MAX_MODEL_ROWS = 100_000
MAX_FEATURES = 50
_IDENTIFIER_RE = re.compile(r"(^id$|_id$)", re.IGNORECASE)


class JoinSpec(BaseModel):
    table: str
    on: list[str] = Field(min_length=1)
    how: Literal["left", "inner"] = "left"
    validation: Literal["one_to_one", "many_to_one"] = Field(
        default="many_to_one", alias="validate"
    )


class ModelingRequest(BaseModel):
    id: str = "model"
    task_type: TaskType
    table: str
    target: str
    features: list[str] = Field(min_length=1, max_length=MAX_FEATURES)
    joins: list[JoinSpec] = Field(default_factory=list)
    split_strategy: SplitStrategy = "random"
    time_column: str | None = None
    group_column: str | None = None
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    models: list[str] = Field(default_factory=list)
    primary_metric: str | None = None
    positive_label: Any = None
    slice_by: list[str] = Field(default_factory=list, max_length=3)
    random_seed: int = 42
    cv_folds: int = Field(default=5, ge=3, le=10)
    bootstrap_iterations: int = Field(default=300, ge=100, le=2000)
    allow_identifier_features: bool = False


def _round(value: float) -> float:
    if value == 0:
        return 0.0
    return float(f"{float(value):.10g}")


def _table_map(dataset: DatasetManifest) -> dict[str, Path]:
    return {table.name: Path(table.path) for table in dataset.tables}


def _load_frame(request: ModelingRequest, dataset: DatasetManifest) -> pd.DataFrame:
    tables = _table_map(dataset)
    if request.table not in tables:
        raise ValueError(f"unknown source table {request.table!r}")
    frame = pd.read_csv(tables[request.table])
    for join in request.joins:
        if join.table not in tables:
            raise ValueError(f"unknown join table {join.table!r}")
        right = pd.read_csv(tables[join.table])
        missing_left = sorted(set(join.on) - set(frame.columns))
        missing_right = sorted(set(join.on) - set(right.columns))
        if missing_left or missing_right:
            raise ValueError(
                f"join {join.table!r} missing keys; left={missing_left}, right={missing_right}"
            )
        overlaps = sorted((set(frame.columns) & set(right.columns)) - set(join.on))
        if overlaps:
            raise ValueError(
                f"join {join.table!r} has ambiguous non-key columns: {overlaps}"
            )
        before = len(frame)
        try:
            frame = frame.merge(
                right, on=join.on, how=join.how, validate=join.validation
            )
        except pd.errors.MergeError as exc:
            raise ValueError(f"join validation failed for {join.table!r}: {exc}") from exc
        if len(frame) > before:
            raise ValueError(
                f"join {join.table!r} increased row count from {before} to {len(frame)}"
            )
    if len(frame) > MAX_MODEL_ROWS:
        raise ValueError(
            f"modeling is capped at {MAX_MODEL_ROWS:,} rows; found {len(frame):,}"
        )
    return frame


def _validate_columns(request: ModelingRequest, frame: pd.DataFrame) -> None:
    if len(set(request.features)) != len(request.features):
        raise ValueError("features must be unique")
    required = set(request.features) | {request.target} | set(request.slice_by)
    if request.time_column:
        required.add(request.time_column)
    if request.group_column:
        required.add(request.group_column)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"modeling columns not found: {missing}")
    if request.target in request.features:
        raise ValueError("target cannot also be a feature")
    identifiers = [feature for feature in request.features if _IDENTIFIER_RE.search(feature)]
    if identifiers and not request.allow_identifier_features:
        raise ValueError(
            "identifier-like features are rejected by default because they can memorize rows: "
            + ", ".join(identifiers)
        )
    all_missing = [feature for feature in request.features if frame[feature].isna().all()]
    if all_missing:
        raise ValueError(f"features are entirely missing: {all_missing}")
    if request.split_strategy == "temporal" and not request.time_column:
        raise ValueError("temporal split requires time_column")
    if request.split_strategy == "group" and not request.group_column:
        raise ValueError("group split requires group_column")


def _prepare_target(
    request: ModelingRequest, frame: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any], list[str]]:
    warnings: list[str] = []
    if request.task_type == "regression":
        target = pd.to_numeric(frame[request.target], errors="coerce")
    else:
        target = frame[request.target]
    missing = int(target.isna().sum())
    if missing:
        warnings.append(
            f"dropped {missing} rows ({missing / len(frame):.1%}) with missing target"
        )
    keep = target.notna()
    frame = frame.loc[keep].copy()
    target = target.loc[keep]
    diagnostics: dict[str, Any] = {"target_rows_dropped": missing}
    if len(frame) < 50:
        raise ValueError(f"at least 50 complete target rows are required; found {len(frame)}")
    if request.task_type == "classification":
        classes = list(pd.unique(target))
        if len(classes) != 2:
            raise ValueError(
                f"classification currently requires exactly two target classes; found {classes}"
            )
        positive = request.positive_label
        if positive is None:
            positive = sorted(classes, key=lambda value: str(value))[-1]
        if positive not in classes:
            raise ValueError(f"positive_label {positive!r} is not in target classes {classes}")
        negative = next(value for value in classes if value != positive)
        target = (target == positive).astype(int)
        rate = float(target.mean())
        diagnostics.update(
            {
                "positive_label": str(positive),
                "negative_label": str(negative),
                "positive_rate": _round(rate),
            }
        )
        if min(rate, 1 - rate) < 0.1:
            warnings.append(
                f"class imbalance: minority class is {min(rate, 1 - rate):.1%} of rows"
            )
    return frame, target, diagnostics, warnings


def _split(
    request: ModelingRequest, frame: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    diagnostics: dict[str, Any] = {}
    if request.split_strategy == "temporal":
        times = pd.to_datetime(frame[request.time_column], errors="coerce", utc=True)
        if times.isna().any():
            raise ValueError(
                f"time_column {request.time_column!r} contains {int(times.isna().sum())} invalid values"
            )
        order = np.argsort(times.to_numpy(), kind="stable")
        ordered_frame = frame.iloc[order]
        ordered_target = target.iloc[order]
        cut = int(len(frame) * (1 - request.test_size))
        # An observation's timestamp is an availability boundary. Keep the entire
        # boundary cohort in holdout rather than learning from simultaneous rows.
        ordered_times = times.iloc[order]
        cut = int(ordered_times.searchsorted(ordered_times.iloc[cut], side="left"))
        if cut == 0:
            raise ValueError("temporal split requires distinct timestamps before the holdout")
        train_frame, test_frame = ordered_frame.iloc[:cut], ordered_frame.iloc[cut:]
        y_train, y_test = ordered_target.iloc[:cut], ordered_target.iloc[cut:]
        diagnostics["train_time_max"] = times.loc[train_frame.index].max().isoformat()
        diagnostics["test_time_min"] = times.loc[test_frame.index].min().isoformat()
        diagnostics["actual_test_fraction"] = _round(len(test_frame) / len(frame))
    elif request.split_strategy == "group":
        groups = frame[request.group_column]
        if groups.isna().any():
            raise ValueError("group_column cannot contain missing values")
        if groups.nunique() < 3:
            raise ValueError("group split requires at least three distinct groups")
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=request.test_size, random_state=request.random_seed
        )
        train_pos, test_pos = next(splitter.split(frame, target, groups=groups))
        train_frame, test_frame = frame.iloc[train_pos], frame.iloc[test_pos]
        y_train, y_test = target.iloc[train_pos], target.iloc[test_pos]
        overlap = set(groups.iloc[train_pos]) & set(groups.iloc[test_pos])
        diagnostics["group_overlap"] = len(overlap)
        if overlap:
            raise ValueError("group split leaked groups across train and test")
    else:
        stratify = target if request.task_type == "classification" else None
        train_frame, test_frame, y_train, y_test = train_test_split(
            frame,
            target,
            test_size=request.test_size,
            random_state=request.random_seed,
            stratify=stratify,
        )
    if request.task_type == "classification" and (
        y_train.nunique() < 2 or y_test.nunique() < 2
    ):
        raise ValueError("both train and holdout partitions must contain both classes")
    return train_frame, test_frame, y_train, y_test, diagnostics


def _preprocessor(frame: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    numeric = [feature for feature in features if pd.api.types.is_numeric_dtype(frame[feature])]
    categorical = [feature for feature in features if feature not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _supported_models(task_type: TaskType) -> tuple[str, ...]:
    if task_type == "classification":
        return ("logistic_regression", "random_forest")
    return ("ridge", "random_forest")


def _pipeline(
    request: ModelingRequest, train_frame: pd.DataFrame, model_name: str
) -> Pipeline:
    if request.task_type == "classification":
        estimators = {
            "logistic_regression": LogisticRegression(max_iter=2000),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=5,
                random_state=request.random_seed,
                n_jobs=1,
            ),
        }
    else:
        estimators = {
            "ridge": Ridge(alpha=1.0),
            "random_forest": RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=5,
                random_state=request.random_seed,
                n_jobs=1,
            ),
        }
    if model_name not in estimators:
        raise ValueError(
            f"unsupported {request.task_type} model {model_name!r}; "
            f"expected one of {sorted(estimators)}"
        )
    return Pipeline(
        [
            ("preprocess", _preprocessor(train_frame, request.features)),
            ("model", estimators[model_name]),
        ]
    )


def _primary_metric(request: ModelingRequest) -> tuple[str, str, bool]:
    if request.task_type == "classification":
        supported = {
            "roc_auc": ("roc_auc", True),
            "f1": ("f1", True),
            "balanced_accuracy": ("balanced_accuracy", True),
        }
        name = request.primary_metric or "roc_auc"
    else:
        supported = {
            "rmse": ("neg_root_mean_squared_error", False),
            "mae": ("neg_mean_absolute_error", False),
            "r2": ("r2", True),
        }
        name = request.primary_metric or "rmse"
    if name not in supported:
        raise ValueError(
            f"unsupported primary_metric {name!r}; expected one of {sorted(supported)}"
        )
    scoring, higher_is_better = supported[name]
    return name, scoring, higher_is_better


def _cv_splitter(
    request: ModelingRequest,
    y_train: pd.Series,
    train_frame: pd.DataFrame,
):
    folds = request.cv_folds
    groups = None
    if request.split_strategy == "temporal":
        times = pd.to_datetime(train_frame[request.time_column], utc=True)
        unique_times = np.sort(times.unique())
        folds = min(folds, len(unique_times) - 1, max(2, len(train_frame) // 20))
        if folds < 2:
            raise ValueError("temporal CV requires at least three distinct training timestamps")
        # Split time cohorts first, then map them to row positions. Row-based
        # TimeSeriesSplit can otherwise leak the same timestamp across a fold.
        codes = np.searchsorted(unique_times, times.to_numpy())
        splits = [
            (np.flatnonzero(np.isin(codes, train)), np.flatnonzero(np.isin(codes, valid)))
            for train, valid in TimeSeriesSplit(n_splits=folds).split(unique_times)
        ]
        return splits, groups, folds
    if request.split_strategy == "group":
        groups = train_frame[request.group_column]
        folds = min(folds, groups.nunique())
        if folds < 2:
            raise ValueError("not enough training groups for cross-validation")
        return GroupKFold(n_splits=folds), groups, folds
    if request.task_type == "classification":
        folds = min(folds, int(y_train.value_counts().min()))
        if folds < 2:
            raise ValueError("not enough minority-class rows for cross-validation")
        return (
            StratifiedKFold(n_splits=folds, shuffle=True, random_state=request.random_seed),
            groups,
            folds,
        )
    folds = min(folds, len(train_frame))
    return KFold(n_splits=folds, shuffle=True, random_state=request.random_seed), groups, folds


def _classification_values(
    y_true: pd.Series | np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
    }


def _regression_values(
    y_true: pd.Series | np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, prediction))),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)),
    }


def _bootstrap_interval(
    request: ModelingRequest,
    y_true: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray | None,
    metric: str,
) -> tuple[float, float] | None:
    rng = np.random.default_rng(request.random_seed)
    values = []
    for _ in range(request.bootstrap_iterations):
        indices = rng.integers(0, len(y_true), len(y_true))
        sampled_y = y_true[indices]
        if request.task_type == "classification":
            if metric == "roc_auc" and len(np.unique(sampled_y)) < 2:
                continue
            scores = _classification_values(
                sampled_y, prediction[indices], probability[indices]
            )
        else:
            scores = _regression_values(sampled_y, prediction[indices])
        values.append(scores[metric])
    if len(values) < 50:
        return None
    lower, upper = np.quantile(values, [0.025, 0.975])
    return _round(lower), _round(upper)


def _metric_models(
    task_type: TaskType,
    model_values: dict[str, float],
    baseline_values: dict[str, float],
    primary: str,
    interval: tuple[float, float] | None,
) -> list[PredictiveMetric]:
    if task_type == "classification":
        directions = {
            "roc_auc": True,
            "accuracy": True,
            "balanced_accuracy": True,
            "precision": True,
            "recall": True,
            "f1": True,
            "log_loss": False,
        }
    else:
        directions = {"rmse": False, "mae": False, "r2": True}
    output = []
    for name, value in model_values.items():
        item = PredictiveMetric(
            name=name,
            value=_round(value),
            higher_is_better=directions[name],
            baseline_value=_round(baseline_values[name]),
        )
        if name == primary and interval is not None:
            item.lower, item.upper, item.confidence = interval[0], interval[1], 0.95
        output.append(item)
    return output


def _slice_metrics(
    request: ModelingRequest,
    test_frame: pd.DataFrame,
    y_test: pd.Series,
    prediction: np.ndarray,
    probability: np.ndarray | None,
    primary: str,
) -> list[SliceMetric]:
    output = []
    for column in request.slice_by:
        values = test_frame[column].fillna("(missing)").astype(str)
        for value in sorted(values.unique()):
            positions = np.flatnonzero(values.to_numpy() == value)
            warning = ""
            score = None
            minimum = 20 if request.task_type == "classification" else 10
            if len(positions) < minimum:
                warning = f"too few holdout rows for a stable slice metric (n={len(positions)})"
            else:
                sliced_y = y_test.iloc[positions].to_numpy()
                if request.task_type == "classification":
                    if primary == "roc_auc" and len(np.unique(sliced_y)) < 2:
                        warning = "ROC AUC is undefined because this slice has one outcome class"
                    else:
                        score = _classification_values(
                            sliced_y, prediction[positions], probability[positions]
                        )[primary]
                else:
                    score = _regression_values(sliced_y, prediction[positions])[primary]
            output.append(
                SliceMetric(
                    column=column,
                    value=value,
                    n=len(positions),
                    metric=primary,
                    score=_round(score) if score is not None else None,
                    warning=warning,
                )
            )
    return output


def _compute(
    task_id: str, request: ModelingRequest, dataset: DatasetManifest
) -> PredictiveEvidence:
    frame = _load_frame(request, dataset)
    _validate_columns(request, frame)
    frame, target, diagnostics, warnings = _prepare_target(request, frame)
    train_frame, test_frame, y_train, y_test, split_diagnostics = _split(
        request, frame, target
    )
    diagnostics.update(split_diagnostics)
    x_train = train_frame[request.features]
    x_test = test_frame[request.features]
    primary, scoring, higher_is_better = _primary_metric(request)
    cv, cv_groups, folds = _cv_splitter(request, y_train, train_frame)
    if request.split_strategy == "temporal":
        times = pd.to_datetime(train_frame[request.time_column], utc=True)
        diagnostics["cv_time_boundaries"] = [
            {"train_time_max": times.iloc[train].max().isoformat(),
             "validation_time_min": times.iloc[valid].min().isoformat()}
            for train, valid in cv
        ]

    model_names = request.models or list(_supported_models(request.task_type))
    if len(set(model_names)) != len(model_names):
        raise ValueError("models must be unique")
    candidates: list[ModelCandidate] = []
    fitted_candidates: dict[str, Pipeline] = {}
    for model_name in model_names:
        pipeline = _pipeline(request, train_frame, model_name)
        try:
            scores = cross_val_score(
                pipeline,
                x_train,
                y_train,
                cv=cv,
                groups=cv_groups,
                scoring=scoring,
                error_score="raise",
                n_jobs=1,
            )
        except ValueError as exc:
            raise ValueError(f"cross-validation failed for {model_name}: {exc}") from exc
        if not np.isfinite(scores).all():
            raise ValueError(f"non-finite cross-validation scores for {model_name}")
        natural_scores = scores if higher_is_better else -scores
        candidates.append(
            ModelCandidate(
                name=model_name,
                cv_mean=_round(float(natural_scores.mean())),
                cv_std=_round(float(natural_scores.std(ddof=1))) if folds > 1 else 0.0,
                folds=folds,
            )
        )
        fitted_candidates[model_name] = pipeline
    selected = (
        max(candidates, key=lambda candidate: candidate.cv_mean)
        if higher_is_better
        else min(candidates, key=lambda candidate: candidate.cv_mean)
    )
    model = fitted_candidates[selected.name]
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    train_prediction = model.predict(x_train)

    if request.task_type == "classification":
        probability = model.predict_proba(x_test)[:, 1]
        train_probability = model.predict_proba(x_train)[:, 1]
        model_values = _classification_values(y_test, prediction, probability)
        train_values = _classification_values(y_train, train_prediction, train_probability)
        if len(np.unique(prediction)) < 2:
            warnings.append(
                "the default 0.5 threshold predicts only one class on the holdout; "
                "threshold selection must use training/validation data, not this holdout"
            )
        baseline = DummyClassifier(strategy="prior").fit(x_train, y_train)
        baseline_prediction = baseline.predict(x_test)
        baseline_probability = baseline.predict_proba(x_test)[:, 1]
        baseline_values = _classification_values(
            y_test, baseline_prediction, baseline_probability
        )
    else:
        probability = None
        model_values = _regression_values(y_test, prediction)
        train_values = _regression_values(y_train, train_prediction)
        baseline = DummyRegressor(strategy="mean").fit(x_train, y_train)
        baseline_values = _regression_values(y_test, baseline.predict(x_test))

    interval = _bootstrap_interval(
        request,
        y_test.to_numpy(),
        np.asarray(prediction),
        np.asarray(probability) if probability is not None else None,
        primary,
    )
    metric_models = _metric_models(
        request.task_type, model_values, baseline_values, primary, interval
    )
    delta = (
        model_values[primary] - baseline_values[primary]
        if higher_is_better
        else baseline_values[primary] - model_values[primary]
    )
    diagnostics.update(
        {
            "train_primary_metric": _round(train_values[primary]),
            "test_primary_metric": _round(model_values[primary]),
            "baseline_primary_metric": _round(baseline_values[primary]),
            "baseline_improvement": _round(delta),
            "random_seed": request.random_seed,
            "bootstrap_iterations": request.bootstrap_iterations,
        }
    )
    if delta <= 0:
        warnings.append(
            f"selected model does not beat the naive baseline on holdout {primary}"
        )
    if higher_is_better and train_values[primary] - model_values[primary] > 0.1:
        warnings.append(
            f"possible overfitting: train {primary} exceeds holdout by "
            f"{train_values[primary] - model_values[primary]:.3f}"
        )
    if not higher_is_better and model_values[primary] > train_values[primary] * 1.5:
        warnings.append(
            f"possible overfitting: holdout {primary} is more than 1.5x training"
        )

    importance_result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring=scoring,
        n_repeats=5,
        random_state=request.random_seed,
        n_jobs=1,
    )
    feature_importance = sorted(
        [
            FeatureImportance(
                feature=feature,
                importance=_round(float(importance)),
                std=_round(float(std)),
            )
            for feature, importance, std in zip(
                request.features,
                importance_result.importances_mean,
                importance_result.importances_std,
            )
        ],
        key=lambda item: abs(item.importance),
        reverse=True,
    )
    slices = _slice_metrics(
        request,
        test_frame,
        y_test,
        np.asarray(prediction),
        np.asarray(probability) if probability is not None else None,
        primary,
    )
    ci_text = (
        f" (95% bootstrap CI {interval[0]:.3g}–{interval[1]:.3g})"
        if interval is not None
        else ""
    )
    summary = (
        f"{selected.name} selected by {folds}-fold training CV; holdout "
        f"{primary}={model_values[primary]:.4g}{ci_text}, "
        f"baseline={baseline_values[primary]:.4g}, n_train={len(train_frame)}, "
        f"n_test={len(test_frame)}"
    )
    return PredictiveEvidence(
        task_id=task_id,
        request_id=request.id,
        task_type=request.task_type,
        status="computed",
        target=request.target,
        features=request.features,
        selected_model=selected.name,
        primary_metric=primary,
        train_size=len(train_frame),
        test_size=len(test_frame),
        split_strategy=request.split_strategy,
        metrics=metric_models,
        candidates=candidates,
        feature_importance=feature_importance,
        slices=slices,
        summary=summary,
        warnings=warnings,
        diagnostics=diagnostics,
    )


def evaluate_modeling_request(
    task_id: str,
    raw: dict[str, Any],
    dataset: DatasetManifest,
    index: int = 0,
) -> PredictiveEvidence:
    request_id = (
        str(raw.get("id") or f"model_{index + 1}")
        if isinstance(raw, dict)
        else f"model_{index + 1}"
    )
    task_type = str(raw.get("task_type") or "unknown") if isinstance(raw, dict) else "unknown"
    try:
        request = ModelingRequest.model_validate(raw)
        request.id = request_id
        return _compute(task_id, request, dataset)
    except (ValidationError, ValueError, TypeError, ArithmeticError) as exc:
        return PredictiveEvidence(
            task_id=task_id,
            request_id=request_id,
            task_type=task_type,
            status="invalid",
            summary="predictive modeling request could not be computed",
            warnings=[str(exc)],
        )


def evaluate_modeling_artifacts(
    artifacts: dict[str, Artifact], dataset: DatasetManifest
) -> dict[str, PredictiveEvidence]:
    evidence: dict[str, PredictiveEvidence] = {}
    for task_id, artifact in artifacts.items():
        result = artifact.outcome.result
        if not isinstance(result, dict) or "modeling_requests" not in result:
            continue
        requests = result.get("modeling_requests")
        if isinstance(requests, dict):
            requests = [requests]
        if not isinstance(requests, list):
            item = PredictiveEvidence(
                task_id=task_id,
                request_id="model_1",
                task_type="unknown",
                status="invalid",
                summary="predictive modeling request could not be computed",
                warnings=["modeling_requests must be a list or object"],
            )
            evidence[f"{task_id}:{item.request_id}"] = item
            continue
        for index, raw in enumerate(requests):
            item = evaluate_modeling_request(task_id, raw, dataset, index)
            key = f"{task_id}:{item.request_id}"
            if key in evidence:
                item.warnings.append(
                    "duplicate request id; later result replaced an earlier one"
                )
            evidence[key] = item
    return evidence
