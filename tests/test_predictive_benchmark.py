import copy
import json
from pathlib import Path

import pytest

from analyst.cli import main
from analyst.data import build_manifest
from analyst.eval.predictive_benchmark import SEEDS, generate, summarize, write_report


@pytest.fixture
def published_run():
    path = Path(__file__).resolve().parents[1] / "predictive_eval_results" / "results.json"
    return json.loads(path.read_text())


def test_published_evidence_is_complete_and_rescores_without_saved_verdicts(published_run):
    published_run.pop("summary")
    summary = summarize(published_run)

    assert summary["integrity_errors"] == []
    assert summary["passed"]
    assert summary["cases_passed"] == summary["cases_total"] == 30


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "seed", "contract", "error", "hash"])
def test_integrity_gate_rejects_broken_runs(published_run, corruption):
    if corruption == "missing":
        published_run["results"].pop()
    elif corruption == "duplicate":
        published_run["results"][-1] = copy.deepcopy(published_run["results"][0])
    elif corruption == "seed":
        published_run["results"][0]["seed"] = 0
    elif corruption == "contract":
        published_run["contract"]["cases"][0]["minimum_metric"] = 0.0
    elif corruption == "error":
        published_run["results"][0]["error"] = "model crashed"
    else:
        published_run["dataset_hashes"].pop(str(SEEDS[0]))

    summary = summarize(published_run)

    assert not summary["passed"]
    assert summary["integrity_errors"]


@pytest.mark.parametrize("corruption", ["metric", "temporal_boundary", "rejection", "cv_selection"])
def test_saved_success_cannot_override_failed_evidence(published_run, corruption):
    case_id = {"metric": "regression_signal", "temporal_boundary": "temporal_cohorts",
               "rejection": "target_leakage", "cv_selection": "classification_signal"}[corruption]
    row = next(row for row in published_run["results"] if row["id"] == case_id)
    evidence = row["evidence"]
    if corruption == "metric":
        next(m for m in evidence["metrics"] if m["name"] == "rmse")["value"] = 50.0
    elif corruption == "temporal_boundary":
        boundary = evidence["diagnostics"]["cv_time_boundaries"][0]
        boundary["validation_time_min"] = boundary["train_time_max"]
    elif corruption == "rejection":
        evidence["warnings"] = ["an unrelated failure is not a successful leakage rejection"]
    else:
        evidence["selected_model"] = "made_up_model"

    summary = summarize(published_run)

    assert published_run["summary"]["passed"]  # stale verdict from the original artifact
    assert not summary["passed"]


def test_generated_data_are_byte_reproducible_and_seed_specific(tmp_path, published_run):
    for seed in SEEDS:
        root = tmp_path / str(seed)
        generate(root, seed)
        assert build_manifest(root).content_hash == published_run["dataset_hashes"][str(seed)]
    assert len(set(published_run["dataset_hashes"].values())) == len(SEEDS)


@pytest.mark.parametrize("fails", [False, True])
def test_cli_writes_reviewable_evidence_and_returns_failure_status(
    tmp_path, published_run, monkeypatch, capsys, fails,
):
    if fails:
        published_run["results"].pop()
    monkeypatch.setattr("analyst.eval.predictive_benchmark.run_benchmark", lambda: published_run)

    code = main(["predictive-benchmark", "--out-dir", str(tmp_path)])

    assert code == int(fails)
    assert (tmp_path / "REPORT.md").exists()
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["summary"]["passed"] is not fails
    assert ("FAIL" if fails else "PASS") in capsys.readouterr().out


def test_report_keeps_expected_rejections_separate_from_execution_errors(tmp_path, published_run):
    _, report, passed = write_report(published_run, tmp_path)

    assert passed
    text = report.read_text()
    assert "30/30" in text
    assert "target_leakage | rejection | 3/3" in text
    assert "does not" in text and "LLM planning accuracy" in text
