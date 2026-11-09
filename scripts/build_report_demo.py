"""Build the key-free report example from published evidence and regenerated CSVs.

Run from the repository root: python scripts/build_report_demo.py
No LLM, sandbox, or model training is used. Narrative/task labels are illustrative.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from analyst.data import build_manifest
from analyst.eval.predictive_benchmark import generate
from analyst.reporting import dataset_provenance, render_report, report_payload
from analyst.stats.engine import evaluate_request


def build(root: Path) -> None:
    published = json.loads((root / "predictive_eval_results/results.json").read_text())
    seed = 20261109
    selected = [row["evidence"] for row in published["results"]
                if row["seed"] == seed and row["id"] in {"classification_signal", "temporal_shift"}]
    if len(selected) != 2:
        raise ValueError("expected two published model evaluations")
    with TemporaryDirectory(prefix="analyst-report-demo-") as temporary:
        data = Path(temporary)
        generate(data, seed)
        manifest = build_manifest(data)
        if manifest.content_hash != published["dataset_hashes"][str(seed)]:
            raise ValueError("regenerated CSVs do not match the published evidence")
        observations = pd.read_csv(data / "observations.csv")
        interval = evaluate_request("descriptive", {
            "id": "observed_conversion", "kind": "one_proportion",
            "groups": [{"label": "converted", "successes": int(observations.converted.sum()),
                        "n": len(observations)}],
        }).model_dump()
        quality = manifest.quality_report.model_dump()
        # Avoid sample frequencies/column profiles in the portable sample record.
        quality.pop("tables")
        quality.pop("relationships")
        metrics = [{m["name"]: m for m in evidence["metrics"]} for evidence in selected]
        auc = metrics[0]["roc_auc"]
        rmse = metrics[1]["rmse"]
        record = {
            "run_id": "synthetic-report-demo", "status": "done",
            "question": "Can we predict conversion, and does a model remain useful after a shift?",
            "demo_note": "Deterministic synthetic example built from published benchmark evidence. "
                         "Narrative and task labels are illustrative. No LLM or council review was run.",
            "answer": (
                f"On this generated dataset, the conversion model achieves holdout ROC AUC "
                f"{auc['value']:.3f}, compared with {auc['baseline_value']:.3f} for the naive baseline.\n\n"
                f"A separate regression stress test reverses the target relationship in the final "
                f"three time cohorts. Holdout RMSE rises to {rmse['value']:.3f}, worse than the "
                f"baseline's {rmse['baseline_value']:.3f}. The engine surfaces that failure rather "
                f"than presenting the model as useful.\n\n"
                "These are synthetic demonstrations of evaluation behavior, not evidence of "
                "real-world predictive accuracy or causal effects."
            ),
            "dataset_provenance": dataset_provenance(manifest),
            "data_quality_report": quality,
            "dag": {
                "descriptive": {"id": "descriptive", "kind": "python", "status": "done",
                                "goal": "Compute observed conversion and Wilson interval", "depends_on": []},
                "predictive_benchmark": {"id": "predictive_benchmark", "kind": "python", "status": "done",
                                         "goal": "Evaluate baseline-relative prediction and a temporal-shift control",
                                         "depends_on": []},
            },
            "claims": [], "statistical_evidence": [interval], "predictive_evidence": selected,
            "council_findings": [], "council_decisions": [],
        }
    output = root / "examples"
    output.mkdir(exist_ok=True)
    (output / "report_run.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    (output / "analysis-report.html").write_text(render_report(report_payload(record)))
    print(f"Report example: {output / 'analysis-report.html'}")


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1])
