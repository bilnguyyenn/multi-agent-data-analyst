"""Numerical contract for the deterministic statistics layer."""
import pandas as pd
import pytest

from analyst.eval.data_gen import generate
from analyst.graph.state import Artifact, ExecutionOutcome
from analyst.stats.engine import evaluate_artifacts, evaluate_request


def test_one_proportion_wilson_interval_and_exact_test():
    evidence = evaluate_request("t1", {
        "id": "conversion_rate",
        "kind": "one_proportion",
        "groups": [{"label": "converted", "successes": 60, "n": 100}],
        "null_value": 0.5,
    })

    assert evidence.status == "computed"
    assert evidence.intervals[0].method == "Wilson score"
    assert evidence.intervals[0].estimate == pytest.approx(0.6)
    assert evidence.intervals[0].lower == pytest.approx(0.5020, abs=1e-3)
    assert evidence.intervals[0].upper == pytest.approx(0.6906, abs=1e-3)
    assert evidence.test.name == "exact binomial test"
    assert evidence.test.p_value == pytest.approx(0.05689, rel=1e-3)
    assert evidence.test.significant is False
    assert evidence.effects[0].name == "Cohen's h"


def test_two_proportion_newcombe_interval_effects_and_z_test():
    evidence = evaluate_request("t1", {
        "id": "defect_rates",
        "kind": "two_proportion",
        "groups": [
            {"label": "A", "successes": 30, "n": 100},
            {"label": "B", "successes": 20, "n": 100},
        ],
    })

    assert evidence.status == "computed"
    difference = next(i for i in evidence.intervals if i.parameter.startswith("rate_difference"))
    assert difference.method == "Newcombe-Wilson"
    assert difference.estimate == pytest.approx(0.1)
    assert difference.lower < 0 < difference.upper
    assert evidence.test.name == "two-proportion z-test"
    assert evidence.test.statistic == pytest.approx(1.633, rel=1e-3)
    assert evidence.test.p_value == pytest.approx(0.1025, rel=1e-3)
    effects = {effect.name: effect for effect in evidence.effects}
    assert effects["odds ratio"].value == pytest.approx(12 / 7, rel=1e-6)
    assert effects["Cohen's h"].interpretation == "small"


def test_two_proportion_uses_fisher_for_sparse_counts():
    evidence = evaluate_request("t1", {
        "id": "rare_event",
        "kind": "two_proportion",
        "groups": [
            {"label": "A", "successes": 1, "n": 20},
            {"label": "B", "successes": 5, "n": 20},
        ],
    })

    assert evidence.status == "computed"
    assert evidence.test.name == "Fisher exact test"
    assert any("expected cell is below 5" in warning for warning in evidence.warnings)


def test_two_mean_welch_interval_and_hedges_g():
    evidence = evaluate_request("t2", {
        "id": "duration_difference",
        "kind": "two_mean",
        "groups": [
            {"label": "new", "mean": 5.0, "std": 2.0, "n": 40},
            {"label": "old", "mean": 4.0, "std": 2.0, "n": 50},
        ],
    })

    assert evidence.status == "computed"
    difference = next(i for i in evidence.intervals if i.parameter.startswith("mean_difference"))
    assert difference.estimate == pytest.approx(1.0)
    assert difference.lower > 0
    assert evidence.test.name == "Welch's t-test"
    assert evidence.test.statistic == pytest.approx(2.357, rel=1e-3)
    assert evidence.test.p_value == pytest.approx(0.0208, rel=2e-2)
    hedges = next(effect for effect in evidence.effects if effect.name == "Hedges' g")
    assert hedges.value == pytest.approx(0.496, rel=1e-2)
    assert hedges.interpretation == "small"


def test_one_mean_t_interval_test_and_cohen_d():
    evidence = evaluate_request("t2", {
        "id": "mean_against_target",
        "kind": "one_mean",
        "groups": [{"label": "cycle_time", "mean": 12.0, "std": 4.0, "n": 64}],
        "null_value": 10.0,
    })

    assert evidence.status == "computed"
    assert evidence.intervals[0].method == "Student t"
    assert evidence.test.name == "one-sample t-test"
    assert evidence.test.statistic == pytest.approx(4.0)
    assert evidence.effects[0].name == "Cohen's d"
    assert evidence.effects[0].value == pytest.approx(0.5)
    assert "p=" in evidence.summary


def test_chi_square_and_cramers_v_with_assumption_warning():
    evidence = evaluate_request("t3", {
        "id": "operator_by_outcome",
        "kind": "chi_square",
        "rows": ["A", "B"],
        "columns": ["defect", "ok"],
        "counts": [[30, 20], [15, 35]],
    })

    assert evidence.status == "computed"
    assert evidence.test.name.startswith("Pearson chi-square")
    assert evidence.test.p_value < 0.01
    effect = evidence.effects[0]
    assert effect.name == "Cramer's V"
    assert 0.2 < effect.value < 0.4

    sparse = evaluate_request("t3", {
        "id": "sparse",
        "kind": "chi_square",
        "rows": ["A", "B"],
        "columns": ["yes", "no"],
        "counts": [[1, 9], [0, 10]],
    })
    assert any("expected cells are below 5" in warning for warning in sparse.warnings)


def test_invalid_request_is_visible_and_does_not_raise():
    evidence = evaluate_request("t1", {
        "id": "impossible_rate",
        "kind": "one_proportion",
        "groups": [{"label": "A", "successes": 11, "n": 10}],
    })

    assert evidence.status == "invalid"
    assert "successes cannot exceed n" in evidence.warnings[0]


def test_artifact_scanner_accepts_one_or_many_requests():
    artifact = Artifact(
        task_id="t1",
        code="",
        outcome=ExecutionOutcome(status="ok", result={
            "answer": "done",
            "statistical_tests": {
                "id": "mean_ci",
                "kind": "one_mean",
                "groups": [{"label": "duration", "mean": 4.2, "std": 1.1, "n": 80}],
            },
        }),
    )

    evidence = evaluate_artifacts({"t1": artifact})
    assert list(evidence) == ["t1:mean_ci"]
    assert evidence["t1:mean_ci"].intervals[0].method == "Student t"


def test_stratified_standardization_detects_simpson_reversal_and_runs_cmh():
    evidence = evaluate_request("t4", {
        "id": "simpson_adjustment",
        "kind": "stratified_proportion",
        "direction": "lower",
        "standardization": "pooled",
        "cells": [
            {"group": "A", "stratum": "low", "successes": 1, "n": 10},
            {"group": "A", "stratum": "high", "successes": 36, "n": 100},
            {"group": "B", "stratum": "low", "successes": 18, "n": 100},
            {"group": "B", "stratum": "high", "successes": 4, "n": 10},
        ],
    })

    assert evidence.status == "computed"
    assert evidence.diagnostics["crude_ranking"] == ["B", "A"]
    assert evidence.diagnostics["adjusted_ranking"] == ["A", "B"]
    assert evidence.diagnostics["simpson_reversal"] is True
    assert evidence.diagnostics["adjusted_rates"] == pytest.approx({"A": 0.23, "B": 0.29})
    assert evidence.test.name.startswith("Cochran-Mantel-Haenszel")
    effects = {effect.name: effect.value for effect in evidence.effects}
    assert effects["crude risk difference:A-B"] > 0
    assert effects["adjusted risk difference:A-B"] < 0
    assert effects["Mantel-Haenszel common odds ratio"] < 1
    assert "winner reversed" in evidence.summary
    assert any("Simpson's paradox" in warning for warning in evidence.warnings)


def test_stratified_request_rejects_missing_overlap_cell():
    evidence = evaluate_request("t4", {
        "id": "missing_support",
        "kind": "stratified_proportion",
        "cells": [
            {"group": "A", "stratum": "low", "successes": 1, "n": 10},
            {"group": "A", "stratum": "high", "successes": 4, "n": 10},
            {"group": "B", "stratum": "low", "successes": 2, "n": 10},
        ],
    })

    assert evidence.status == "invalid"
    assert "complete 2x2 grid" in evidence.warnings[0]


def test_stratified_custom_weights_are_normalized_and_direction_is_respected():
    evidence = evaluate_request("t4", {
        "id": "custom_target",
        "kind": "stratified_proportion",
        "direction": "higher",
        "standardization": "custom",
        "reference_weights": {"low": 8, "high": 2},
        "cells": [
            {"group": "A", "stratum": "low", "successes": 10, "n": 100},
            {"group": "A", "stratum": "high", "successes": 36, "n": 100},
            {"group": "B", "stratum": "low", "successes": 18, "n": 100},
            {"group": "B", "stratum": "high", "successes": 40, "n": 100},
        ],
    })

    assert evidence.status == "computed"
    assert evidence.diagnostics["reference_weights"] == {"low": 0.8, "high": 0.2}
    assert evidence.diagnostics["adjusted_rates"] == pytest.approx({"A": 0.152, "B": 0.224})
    assert evidence.diagnostics["adjusted_ranking"] == ["B", "A"]


def test_planted_shift_and_supplier_traps_are_corrected(tmp_path):
    data_dir = tmp_path / "sample"
    generate(data_dir, seed=7)

    work_orders = pd.read_csv(data_dir / "work_orders.csv")
    products = pd.read_csv(data_dir / "products.csv")
    defects = (
        pd.read_csv(data_dir / "defects.csv")
        .groupby("wo_id", as_index=False).defect_count.sum()
    )
    shift_rows = (
        work_orders
        .merge(products[["product_id", "complexity"]], on="product_id")
        .merge(defects, on="wo_id", how="left")
        .fillna({"defect_count": 0})
        .groupby(["shift", "complexity"], as_index=False)
        .agg(successes=("defect_count", "sum"), n=("qty_produced", "sum"))
    )
    shift = evaluate_request("shift", {
        "id": "shift_mix_adjusted",
        "kind": "stratified_proportion",
        "direction": "lower",
        "cells": [
            {
                "group": row.shift,
                "stratum": row.complexity,
                "successes": int(row.successes),
                "n": int(row.n),
            }
            for row in shift_rows.itertuples()
        ],
    })
    assert shift.diagnostics["crude_ranking"][0] == "night"
    assert shift.diagnostics["adjusted_ranking"] == ["day", "evening", "night"]
    assert shift.diagnostics["simpson_reversal"] is True

    lots = pd.read_csv(data_dir / "material_lots.csv")
    supplier_rows = (
        lots.groupby(["supplier_id", "material"], as_index=False)
        .agg(successes=("defective_units", "sum"), n=("units", "sum"))
    )
    supplier = evaluate_request("supplier", {
        "id": "supplier_mix_adjusted",
        "kind": "stratified_proportion",
        "direction": "lower",
        "cells": [
            {
                "group": row.supplier_id,
                "stratum": row.material,
                "successes": int(row.successes),
                "n": int(row.n),
            }
            for row in supplier_rows.itertuples()
        ],
    })
    assert supplier.diagnostics["crude_ranking"][0] == "S2"
    assert supplier.diagnostics["adjusted_ranking"][0] == "S1"
    assert supplier.diagnostics["simpson_reversal"] is True
