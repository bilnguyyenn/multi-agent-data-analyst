"""Validate every aggregate group against the independent reference; publish $0 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from analyst.operations.changes import SOURCE_SHA256, grouped_values, render_bridge
from analyst.operations.metrics import fingerprint
from analyst.operations.reference import reference_values
from analyst.operations.service import Operations, file_hash

METRICS = ["gross_sales", "cancellation_value", "net_sales", "units_sold"]


def build(root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    with TemporaryDirectory(prefix="retail-change-evidence-") as temporary:
        service = Operations(temporary)
        book = service.save_playbook(
            {"name": "Contribution review", "metrics": [{"metric": metric} for metric in METRICS]}
        )
        jobs, sources = [], []
        for label, start, end in [
            ("development", "2011-11-01", "2011-11-08"),
            ("refresh", "2011-11-08", "2011-11-15"),
        ]:
            source = root / "data/retail" / label / "transactions.csv"
            snapshot = service.ingest(source)
            job = service.execute(
                service.submit(
                    {
                        "snapshot_id": snapshot["snapshot_id"],
                        "playbook_id": book["playbook_id"],
                        "start": start,
                        "end": end,
                    }
                )["job_id"]
            )
            if job["status"] != "done":
                raise ValueError(job.get("error", job["status"]))
            jobs.append(job)
            sources.append(source)
        for metric in METRICS:
            for dimension in ["country", "product"]:
                result = service.explain_change(
                    *(job["job_id"] for job in jobs),
                    {"metric": metric, "group_by": dimension, "limit": 10},
                )
                checks = []
                for source, ref in zip(sources, result["evidence_refs"]):
                    packet = service.get("evidence", ref)
                    actual = grouped_values(source, packet, dimension)["values"]
                    expected = {
                        r["segment"]: (r["numerator"], r["source_rows"])
                        for r in reference_values(
                            source, {**packet["request"], "group_by": dimension, "limit": 100_000}
                        )
                    }
                    checks.append(
                        {
                            "source_sha256": file_hash(source),
                            "groups": len(expected),
                            "aggregate_sha256": fingerprint(expected),
                            "passed": actual == expected,
                        }
                    )
                passed = (
                    all(check["passed"] for check in checks)
                    and result["reconciliation_residual_base_units"] == 0
                    and sum(row["delta_base_units"] for row in result["rows"])
                    + result["remainder"]["delta_base_units"]
                    == result["delta_base_units"]
                )
                cases.append(
                    {
                        "metric": metric,
                        "group_by": dimension,
                        "passed": passed,
                        "period_checks": checks,
                        "union_groups": result["group_count"],
                        "delta_base_units": result["delta_base_units"],
                        "residual_base_units": result["reconciliation_residual_base_units"],
                    }
                )
                if not passed:
                    raise ValueError(f"independent reconciliation failed: {metric}/{dimension}")
                if metric == "net_sales":
                    # Make the offline artifact fully self-contained after the temporary store closes.
                    evidence = {
                        "change": result,
                        "overall_evidence": [
                            service.get("evidence", ref) for ref in result["evidence_refs"]
                        ],
                    }
                    (output / f"change-{dimension}.json").write_text(
                        json.dumps(evidence, indent=2) + "\n"
                    )
                    (output / f"change-{dimension}.html").write_text(render_bridge(result))
        summary = {
            "version": "retail-change-validation-v1",
            "source_sha256": SOURCE_SHA256,
            "passed": all(case["passed"] for case in cases),
            "cases": cases,
            "scope": "8 deterministic real-data reconciliations: 4 additive metrics × 2 dimensions. Every aggregate group is checked against independent row-wise Decimal logic. Not model accuracy or causal evidence.",
        }
        (output / "change-validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Change checks: {sum(case['passed'] for case in cases)}/{len(cases)}; {output}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("retail_results"))
    args = parser.parse_args()
    build(Path(__file__).resolve().parents[1], args.out_dir)
