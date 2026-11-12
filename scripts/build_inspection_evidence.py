"""Reproduce source-inspection examples and independent real-data checks without a model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from tempfile import TemporaryDirectory

from analyst.operations.inspection import SOURCE_SHA256, render_inspection
from analyst.operations.metrics import COLUMNS
from analyst.operations.reference import reference_values
from analyst.operations.service import Operations, file_hash

METRICS = {
    "gross_sales": "gross_pence",
    "cancellation_value": "cancellation_pence",
    "net_sales": "net_pence",
    "units_sold": "units_sold",
}
CASES = [
    ("product", "AMAZONFEE"),
    ("product", "DOT"),
    ("product", "21915"),
    ("country", "United Kingdom"),
    ("country", "Switzerland"),
    ("country", "France"),
]


def build(root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with TemporaryDirectory(prefix="retail-source-inspection-") as temporary:
        service = Operations(temporary)
        book = service.save_playbook(
            {"name": "Source inspection example", "metrics": [{"metric": name} for name in METRICS]}
        )
        jobs, sources, raw_rows = [], [], []
        for label, start, end in [
            ("development", "2011-11-01", "2011-11-08"),
            ("refresh", "2011-11-08", "2011-11-15"),
        ]:
            source = root / "data/retail" / label / "transactions.csv"
            snap = service.ingest(source)
            job = service.execute(
                service.submit(
                    {
                        "snapshot_id": snap["snapshot_id"],
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
            with source.open(newline="") as stream:
                raw_rows.append(
                    Counter(
                        tuple(value or None for value in row)
                        for row in list(csv.reader(stream))[1:]
                    )
                )
        for dimension in ["product", "country"]:
            parent = service.explain_change(
                *(job["job_id"] for job in jobs), {"group_by": dimension}
            )
            reference = {}
            for index, (source, period) in enumerate(zip(sources, parent["periods"])):
                for metric in METRICS:
                    reference[index, metric] = {
                        row["segment"]: row
                        for row in reference_values(
                            source,
                            {**period, "metric": metric, "group_by": dimension, "limit": 100_000},
                        )
                    }
            for group_by, segment in CASES:
                if group_by != dimension:
                    continue
                packet = service.inspect_contributor(parent["computation_id"], {"segment": segment})
                checks = []
                for index, period in enumerate(packet["periods"]):
                    totals = period["totals"]
                    checks.extend(
                        totals[field]
                        == reference[index, metric].get(segment, {"numerator": 0})["numerator"]
                        for metric, field in METRICS.items()
                    )
                    expected_rows = reference[index, "net_sales"].get(segment, {"source_rows": 0})[
                        "source_rows"
                    ]
                    checks.append(totals["eligible_rows"] == expected_rows)
                    sample_counts = Counter(
                        tuple(row["source"][c] for c in COLUMNS) for row in period["sample"]
                    )
                    checks.append(
                        all(
                            count <= raw_rows[index][record]
                            for record, count in sample_counts.items()
                        )
                    )
                    for row in period["sample"]:
                        price = Decimal(row["source"]["UnitPrice"])
                        quantity = Decimal(row["source"]["Quantity"])
                        expected = int(
                            (price * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
                        ) * int(quantity)
                        checks.append(row["signed_pence"] == expected)
                if not all(checks):
                    raise ValueError(f"independent source check failed: {dimension}/{segment}")
                results.append(
                    {
                        "group_by": dimension,
                        "segment": segment,
                        "passed": True,
                        "source_sha256": [file_hash(path) for path in sources],
                        "eligible_rows": [p["totals"]["eligible_rows"] for p in packet["periods"]],
                        "sampled_rows": [p["sampled_rows"] for p in packet["periods"]],
                        "delta": packet["contributor"]["delta"],
                        "residual_base_units": packet["reconciliation_residual_base_units"],
                    }
                )
                if segment == "AMAZONFEE":
                    (output / "inspection-amazonfee.json").write_text(
                        json.dumps({"inspection": packet, "parent": parent}, indent=2) + "\n"
                    )
                    (output / "inspection-amazonfee.html").write_text(render_inspection(packet))
        summary = {
            "version": "retail-inspection-validation-v1",
            "source_sha256": SOURCE_SHA256,
            "passed": all(row["passed"] for row in results),
            "results": results,
            "scope": "Six deterministic real-data inspections. All four component totals and source-row counts match the independent Decimal reference for both periods. Every sampled record is present in its source, duplicate multiplicity is bounded by the source, and signed pence are independently recalculated. Not model accuracy or representative sampling.",
        }
        (output / "inspection-validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Source inspections: {len(results)}/{len(CASES)} passed; {output}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("retail_results"))
    args = parser.parse_args()
    build(Path(__file__).resolve().parents[1], args.out_dir)
