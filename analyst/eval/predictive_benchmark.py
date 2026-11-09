"""Versioned, API-free reliability checks of the governed modeling engine.

This is a synthetic development benchmark of deterministic specifications, not
an evaluation of natural-language planning or an external held-out dataset.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from analyst.data import build_manifest
from analyst.graph.state import PredictiveEvidence
from analyst.modeling import engine

SUITE = "predictive-reliability-v1"
SEEDS = (20261109, 20261110, 20261111)
ROWS = 960


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    request: dict[str, Any]
    expected_status: str = "computed"
    expected_warning: str = ""
    minimum_metric: float | None = None
    maximum_metric: float | None = None
    minimum_improvement: float | None = None
    maximum_improvement: float | None = None


def cases() -> tuple[Case, ...]:
    """Specs and tolerances are fixed before running the three published seeds."""
    regression = {
        "task_type": "regression", "table": "observations", "target": "value",
        "features": ["x", "segment"], "models": ["ridge"], "primary_metric": "rmse",
        "cv_folds": 3, "bootstrap_iterations": 100,
    }
    classification = {
        **regression, "task_type": "classification", "target": "converted",
        "models": ["logistic_regression", "random_forest"], "primary_metric": "roc_auc",
        "slice_by": ["segment"],
    }
    temporal = {**regression, "split_strategy": "temporal", "time_column": "observed_at"}
    return (
        Case("classification_signal", "signal", classification,
             minimum_metric=0.75, minimum_improvement=0.25),
        Case("regression_signal", "signal", regression,
             maximum_metric=0.7, minimum_improvement=1.8),
        Case("temporal_cohorts", "split", temporal,
             maximum_metric=0.7, minimum_improvement=1.8),
        Case("group_holdout", "split", {
            **regression, "split_strategy": "group", "group_column": "account_group",
        }, maximum_metric=0.7, minimum_improvement=1.8),
        Case("no_signal", "negative_control", {
            **regression, "target": "noise", "primary_metric": "r2",
        }, minimum_metric=-0.2, maximum_metric=0.1),
        Case("temporal_shift", "negative_control", {**temporal, "target": "shifted"},
             expected_warning="does not beat the naive baseline", minimum_metric=4.0,
             maximum_improvement=-0.5),
        Case("target_leakage", "rejection", {
            **regression, "features": ["x", "value"],
        }, expected_status="invalid", expected_warning="target cannot also be a feature"),
        Case("identifier_leakage", "rejection", {
            **regression, "features": ["x", "row_id"],
        }, expected_status="invalid", expected_warning="identifier-like features are rejected"),
        Case("join_fanout", "rejection", {
            **regression, "joins": [{"table": "events", "on": ["row_id"]}],
        }, expected_status="invalid", expected_warning="join validation failed"),
        Case("single_timestamp", "rejection", {**temporal, "time_column": "same_time"},
             expected_status="invalid", expected_warning="distinct timestamps"),
    )


def contract() -> dict[str, Any]:
    return {"suite": SUITE, "seeds": list(SEEDS), "rows": ROWS,
            "cases": [asdict(case) for case in cases()]}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def generate(root: Path, seed: int) -> None:
    """Known signal, independent noise, a late sign reversal, and repeated cohorts."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=ROWS)
    segment = rng.choice(["north", "south", "west"], ROWS)
    signal = 3 * x + 0.8 * (segment == "north")
    value = signal + rng.normal(scale=0.35, size=ROWS)
    logits = 2.5 * x + 0.5 * (segment == "north")
    frame = pd.DataFrame({
        "row_id": [f"R{i:04d}" for i in range(ROWS)],
        "account_group": [f"G{i // 4:03d}" for i in range(ROWS)],
        "observed_at": np.repeat(pd.date_range("2025-01-01", periods=12), ROWS // 12),
        "same_time": "2025-01-01",
        "x": x, "segment": segment, "value": value,
        "converted": rng.binomial(1, 1 / (1 + np.exp(-logits))),
        "noise": rng.normal(size=ROWS),
        # The temporal policy holds out the final three complete date cohorts.
        "shifted": np.where(np.arange(ROWS) < 720, signal, -signal)
        + rng.normal(scale=0.35, size=ROWS),
    })
    frame.to_csv(root / "observations.csv", index=False, float_format="%.12g")
    pd.DataFrame({
        "event_id": [f"E{i:04d}" for i in range(ROWS * 2)],
        "row_id": np.repeat(frame.row_id.to_numpy(), 2),
    }).to_csv(root / "events.csv", index=False)


def check_evidence(case: Case, evidence: PredictiveEvidence) -> dict[str, bool]:
    """Recompute verdicts from typed evidence; never trust a saved pass flag."""
    checks = {"expected_status": evidence.status == case.expected_status}
    if case.expected_warning:
        checks["expected_warning"] = any(case.expected_warning in w for w in evidence.warnings)
    if case.expected_status == "invalid":
        checks["no_model_or_metrics"] = not evidence.selected_model and not evidence.metrics
        return checks
    if evidence.status != "computed":
        return checks
    checks["request_specification"] = (
        evidence.task_type == case.request["task_type"]
        and evidence.target == case.request["target"]
        and evidence.features == case.request["features"]
        and evidence.primary_metric == case.request["primary_metric"]
    )
    checks["all_rows_accounted_for"] = evidence.train_size + evidence.test_size == ROWS
    checks["requested_split"] = evidence.split_strategy == case.request.get("split_strategy", "random")
    metrics = {metric.name: metric for metric in evidence.metrics}
    primary = metrics.get(evidence.primary_metric)
    checks["primary_metric_present"] = primary is not None
    checks["finite_metrics_and_baselines"] = bool(metrics) and all(
        math.isfinite(m.value) and m.baseline_value is not None and math.isfinite(m.baseline_value)
        for m in metrics.values()
    )
    checks["finite_training_cv"] = bool(evidence.candidates) and all(
        math.isfinite(candidate.cv_mean) and math.isfinite(candidate.cv_std)
        and candidate.folds == case.request["cv_folds"]
        for candidate in evidence.candidates
    )
    checks["requested_candidates"] = [c.name for c in evidence.candidates] == case.request["models"]
    if primary is not None:
        checks["bootstrap_interval"] = (
            primary.lower is not None and primary.upper is not None
            and math.isfinite(primary.lower) and math.isfinite(primary.upper)
            and primary.lower <= primary.upper and primary.confidence == 0.95
        )
        if evidence.candidates:
            winner = (max if primary.higher_is_better else min)(
                evidence.candidates, key=lambda item: item.cv_mean
            )
            checks["selected_by_training_cv"] = evidence.selected_model == winner.name
        if case.minimum_metric is not None:
            checks["minimum_metric"] = primary.value >= case.minimum_metric
        if case.maximum_metric is not None:
            checks["maximum_metric"] = primary.value <= case.maximum_metric
        if primary.baseline_value is not None:
            delta = primary.value - primary.baseline_value
            if not primary.higher_is_better:
                delta = -delta
            if case.minimum_improvement is not None:
                checks["minimum_baseline_improvement"] = delta >= case.minimum_improvement
            if case.maximum_improvement is not None:
                checks["maximum_baseline_improvement"] = delta <= case.maximum_improvement
    if evidence.split_strategy == "temporal":
        diag = evidence.diagnostics
        checks["strict_holdout_time_boundary"] = (
            bool(diag.get("train_time_max")) and bool(diag.get("test_time_min"))
            and diag["train_time_max"] < diag["test_time_min"]
        )
        boundaries = diag.get("cv_time_boundaries", [])
        checks["strict_cv_time_boundaries"] = len(boundaries) == case.request["cv_folds"] and all(
            boundary["train_time_max"] < boundary["validation_time_min"] for boundary in boundaries
        )
    if evidence.split_strategy == "group":
        checks["no_entity_overlap"] = evidence.diagnostics.get("group_overlap") == 0
    if case.id == "classification_signal":
        checks["probability_loss_beats_baseline"] = (
            "log_loss" in metrics and metrics["log_loss"].baseline_value is not None
            and metrics["log_loss"].value < metrics["log_loss"].baseline_value
        )
        checks["subgroup_coverage"] = {item.value for item in evidence.slices} == {
            "north", "south", "west"
        } and all(item.score is not None for item in evidence.slices)
    return checks


def summarize(run: dict[str, Any]) -> dict[str, Any]:
    """Reject partial/duplicate/mislabeled runs and score every expected case."""
    errors = []
    if run.get("contract") != contract() or run.get("contract_sha256") != _digest(contract()):
        errors.append("suite contract does not match the current version")
    results = run.get("results")
    if not isinstance(results, list):
        results = []
        errors.append("results must be a list")
    expected = {(seed, case.id) for seed in SEEDS for case in cases()}
    keys = [(row.get("seed"), row.get("id")) for row in results if isinstance(row, dict)]
    if len(keys) != len(results) or len(keys) != len(expected) or set(keys) != expected:
        errors.append("expected exactly one result for every fixed seed and case")
    datasets = run.get("dataset_hashes", {})
    if set(datasets) != {str(seed) for seed in SEEDS} or any(
        not isinstance(value, str) or len(value) != 64 for value in datasets.values()
    ):
        errors.append("missing dataset content hashes")
    specs = {case.id: case for case in cases()}
    scored = []
    for row in results:
        if not isinstance(row, dict) or row.get("id") not in specs:
            errors.append("unknown or malformed case")
            continue
        case = specs[row["id"]]
        checks = {}
        if row.get("error"):
            errors.append(f"{row.get('seed')}/{case.id}: execution error: {row['error']}")
        else:
            try:
                evidence = PredictiveEvidence.model_validate(row.get("evidence"))
                checks = check_evidence(case, evidence)
                checks["request_identity"] = evidence.request_id == case.id
            except (ValueError, TypeError, KeyError) as exc:
                errors.append(f"{row.get('seed')}/{case.id}: malformed evidence: {exc}")
        scored.append({"seed": row.get("seed"), "id": case.id, "category": case.category,
                       "checks": checks, "passed": bool(checks) and all(checks.values())})
    passed = sum(row["passed"] for row in scored)
    return {"integrity_errors": errors, "passed": not errors and passed == len(expected),
            "cases_passed": passed, "cases_total": len(expected), "results": scored}


def run_benchmark() -> dict[str, Any]:
    run: dict[str, Any] = {
        "contract": contract(), "contract_sha256": _digest(contract()),
        "environment": {"python": platform.python_version(), **{
            name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")
        }},
        "source_sha256": {
            "engine": hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
            "benchmark": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "dataset_hashes": {}, "results": [],
    }
    with TemporaryDirectory(prefix="analyst-predictive-") as temporary:
        for seed in SEEDS:
            root = Path(temporary) / str(seed)
            generate(root, seed)
            dataset = build_manifest(root)
            run["dataset_hashes"][str(seed)] = dataset.content_hash
            for case in cases():
                row: dict[str, Any] = {"seed": seed, "id": case.id, "error": None}
                try:
                    row["evidence"] = engine.evaluate_modeling_request(
                        "predictive_benchmark", {**case.request, "id": case.id}, dataset
                    ).model_dump(mode="json")
                except Exception as exc:
                    # Preserve diagnostics for all cases; any exception fails the gate.
                    row["error"] = f"{type(exc).__name__}: {exc}"
                run["results"].append(row)
    return run


def write_report(run: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path, bool]:
    summary = summarize(run)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "results.json"
    report = output / "REPORT.md"
    lines = [
        "# Predictive reliability benchmark", "",
        f"Suite `{SUITE}` · **{summary['cases_passed']}/{summary['cases_total']} case instances passed**",
        "", f"Integrity gate: **{'PASS' if summary['passed'] else 'FAIL'}**.", "",
        "Synthetic development benchmark of fixed modeling specifications. This does not",
        "measure LLM planning accuracy, real-world generalization, or a production guarantee.", "",
        "| case | category | passed seeds |", "|---|---|---:|",
    ]
    for case in cases():
        count = sum(row["passed"] for row in summary["results"] if row["id"] == case.id)
        lines.append(f"| {case.id} | {case.category} | {count}/{len(SEEDS)} |")
    lines += ["", "## Holdout metrics by seed", "",
              "These metrics are separate from behavioral pass counts. Lower RMSE is better;",
              "higher ROC AUC and R² are better. A negative-control model may correctly lose to baseline.", "",
              "| seed | case | metric | model | baseline | 95% bootstrap interval |",
              "|---|---|---|---:|---:|---|" ]
    for row in run.get("results", []):
        evidence = row.get("evidence", {}) if isinstance(row, dict) else {}
        for metric in evidence.get("metrics", []):
            if metric["name"] == evidence.get("primary_metric"):
                interval = (f"[{metric['lower']:.4f}, {metric['upper']:.4f}]"
                            if metric.get("lower") is not None and metric.get("upper") is not None
                            else "unavailable")
                lines.append(f"| {row['seed']} | {row['id']} | {metric['name']} | "
                             f"{metric['value']:.4f} | {metric['baseline_value']:.4f} | {interval} |")
    lines += ["", "## Reproduction and provenance", "", "```bash",
              "python -m analyst predictive-benchmark --out-dir predictive_eval_results", "```", "",
              "No API keys, Docker, downloads, or paid services are used. Raw evidence, fixed",
              "requests/tolerances, runtime versions, source hashes, and per-seed CSV content",
              "hashes are in [results.json](results.json). Verdicts are recomputed from evidence;",
              "missing/duplicate cases or execution errors fail the gate and CLI exit status.", "",
              f"Contract SHA-256: `{run.get('contract_sha256')}`.", "",
              "The three seeds are development seeds. Tolerances test known planted behavior;",
              "they are not confidence bounds or estimates of external predictive accuracy.", "",
              "Bootstrap intervals condition on the fitted model and resample holdout rows.",
              "They do not account for temporal/entity dependence or model-selection uncertainty.", ""]
    failures = [f"{row['seed']}/{row['id']}: {', '.join(k for k, v in row['checks'].items() if not v)}"
                for row in summary["results"] if not row["passed"]]
    if summary["integrity_errors"] or failures:
        lines += ["## Failures", ""] + [f"- {error}" for error in summary["integrity_errors"] + failures]
    # Replace each artifact only after serialization succeeds. Raw evidence owns the verdict.
    payload = json.dumps({**run, "summary": summary}, indent=2, allow_nan=False) + "\n"
    raw.with_suffix(".json.tmp").write_text(payload)
    raw.with_suffix(".json.tmp").replace(raw)
    report.with_suffix(".md.tmp").write_text("\n".join(lines))
    report.with_suffix(".md.tmp").replace(report)
    return raw, report, summary["passed"]
