"""Rebuild the operations evidence package without a model or paid service."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from analyst.eval.retail import build_contract, contract_checks
from analyst.operations.metrics import REGISTRY
from analyst.operations.preflight import local_preflight
from analyst.operations.service import Operations
from analyst.reporting import render_report, report_payload


def build(root: Path):
    output = root / "retail_results"
    output.mkdir(exist_ok=True)
    data = root / "data/retail"
    contract = build_contract(data)
    if (output / "contract.json").exists() and json.loads(
        (output / "contract.json").read_text()
    ) != contract:
        raise ValueError("frozen contract changed; version the suite before replacing it")
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    checked = contract_checks(contract, data)
    if not checked["passed"]:
        raise ValueError("metric engine disagrees with independent Decimal reference")
    (output / "contract-results.json").write_text(json.dumps(checked, indent=2) + "\n")
    runtime, jobs = [], []
    with TemporaryDirectory(prefix="operations-evidence-") as temporary:
        service = Operations(temporary)
        book = service.save_playbook(
            {
                "name": "Weekly retail review",
                "metrics": [{"metric": metric} for metric in REGISTRY]
                + [{"metric": "net_sales", "group_by": "country", "limit": 100}],
            }
        )
        for label, start, end, path in [
            ("development", "2011-11-01", "2011-11-08", data / "development/transactions.csv"),
            ("refresh", "2011-11-08", "2011-11-15", data / "refresh/transactions.csv"),
            ("full", "2010-12-01", "2011-12-10", root / ".retail_source/full/transactions.csv"),
        ]:
            if not path.exists():
                continue
            snapshot = service.ingest(path, {"label": label})
            job = service.submit(
                {
                    "playbook_id": book["playbook_id"],
                    "snapshot_id": snapshot["snapshot_id"],
                    "start": start,
                    "end": end,
                }
            )
            result = service.execute(job["job_id"])
            if result["status"] != "done":
                raise ValueError(result.get("error", result["status"]))
            packet = report_payload(service.report_record(job["job_id"]))
            (output / f"{label}.json").write_text(json.dumps(packet, indent=2) + "\n")
            (output / f"{label}.html").write_text(render_report(packet))
            runtime.append(
                {
                    "case": label,
                    "source_rows": snapshot["rows"],
                    "csv_bytes": path.stat().st_size,
                    **{
                        key: result[key]
                        for key in ["duration_s", "process_peak_rss_mb", "memory_scope"]
                    },
                }
            )
            jobs.append(result["job_id"])
        comparison = service.compare(jobs[0], jobs[1])
        (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (output / "runtime.json").write_text(
        json.dumps(
            {
                "measurements": runtime,
                "scope": "One local process, sequential warm runs; not an agent/API latency benchmark. RSS is the process-lifetime high-water mark, not per-job allocation.",
            },
            indent=2,
        )
        + "\n"
    )
    preflight = local_preflight()
    (output / "local-preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    (output / "README.md").write_text(f"""# Retail operations evidence

**{sum(row["passed"] for row in checked["results"])}/{len(checked["results"])} deterministic checks passed** on public UCI retail slices:
100 fixed questions and 20 two-turn scenarios. This is metric-engine/independent
Decimal-oracle agreement and definition-guard behavior, **not live-model accuracy**.

- [Frozen task contract](contract.json), [all check outcomes](contract-results.json)
- [First review](development.html), [refresh](refresh.html), [comparison](comparison.json)
- [Recorded runtime and process memory](runtime.json)
- [Local-model preflight](local-preflight.json)

The reports link every displayed metric to its stored query, transformations,
parameters, definition fingerprint, source hash, and aggregate rows. Raw JSON
packets sit beside each HTML file. The full source conversion is reproduced from
the attributed, hash-pinned UCI download; public weekly slices are in data/retail.

Local-agent evaluation status: **pending**. No model accuracy is inferred from
the deterministic checks. Last preflight: {json.dumps(preflight.get("reasons", []))}.
User pilot status: **participants deferred by the user**; no time-saving claim.

Rebuild: `python scripts/build_operations_evidence.py`.
Run the guarded free live evaluation later: `python -m analyst retail-eval --live-local`.
Do not tune on the final task outcomes or describe this public temporal split as
sealed or guaranteed absent from model pretraining.
""")
    print(f"Evidence package: {output}")


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1])
