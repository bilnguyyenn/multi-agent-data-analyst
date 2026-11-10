import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from analyst.operations.metrics import (
    REGISTRY,
    MetricRequest,
    compute,
    connect,
    prepare,
    resolve_question,
)
from analyst.operations.reference import reference_values
from analyst.operations.service import Operations
from analyst.reporting import render_report, report_payload


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "transactions.csv"
    rows = [
        ["1", "A", "quoted <script>x</script>", "2", "2011-11-01 12:00:00", "1.25", "001", "UK"],
        ["1", "A", "quoted <script>x</script>", "2", "2011-11-01 12:00:00", "1.25", "001", "UK"],
        ["C1", "A", "cancel", "-1", "2011-11-02 12:00:00", "1.25", "001", "UK"],
        ["2", "B", "missing id", "3", "2011-11-03 12:00:00", "2.00", "", "France"],
        ["3", "C", "zero", "1", "2011-11-03 12:00:00", "0", "003", "France"],
        ["4", "D", "unexpected", "-1", "2011-11-03 12:00:00", "2.00", "004", "UK"],
        ["5", "E", "invalid", "x", "2011-11-03 12:00:00", "2.00", "004", "UK"],
        ["6", "E", "fractional units", "1.5", "2011-11-03 12:00:00", "2.00", "004", "UK"],
        ["7", "E", "tiny fractional units", "1.00001", "2011-11-03 12:00:00", "2.00", "004", "UK"],
        ["8", "E", "excess price precision", "1", "2011-11-03 12:00:00", "1.004999", "004", "UK"],
    ]
    from analyst.operations.metrics import COLUMNS

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("metric", list(REGISTRY))
@pytest.mark.parametrize("group", [None, "country", "customer"])
def test_contract_matches_independent_decimal_oracle(csv_path, metric, group):
    c = connect()
    try:
        prepare(c, csv_path)
        request = MetricRequest(metric=metric, group_by=group)
        packet = compute(c, request, "a" * 64)
        assert packet["values"] == reference_values(csv_path, request.model_dump(mode="json"))
        assert packet["duplicate_rows_retained"] == 1
        assert packet["quality"]["unexpected_quantity_sign"] == 1
    finally:
        c.close()


def test_real_public_slice_matches_independent_oracle():
    path = Path(__file__).resolve().parents[1] / "data/retail/development/transactions.csv"
    c = connect()
    try:
        prepare(c, path)
        for metric in REGISTRY:
            request = MetricRequest(metric=metric, group_by="country", country="United Kingdom")
            assert compute(c, request, "a" * 64)["values"] == reference_values(
                path, request.model_dump(mode="json")
            )
    finally:
        c.close()


def test_ambiguity_and_unsupported_outcomes_are_explicit():
    assert resolve_question("What is revenue?")["status"] == "clarification_required"
    assert resolve_question("What is net sales?")["metric"] == "net_sales"
    assert resolve_question("What caused profit to rise?")["status"] == "unsupported"
    with pytest.raises(ValueError):
        MetricRequest(metric="net_sales", group_by="country; DROP TABLE raw")


def test_empty_period_keeps_rate_undefined(csv_path):
    c = connect()
    prepare(c, csv_path)
    try:
        p = compute(
            c,
            MetricRequest(metric="cancellation_rate", start="2020-01-01", end="2020-01-08"),
            "a" * 64,
        )
        assert p["values"][0]["value"] is None
        assert p["values"][0]["denominator"] == 0
    finally:
        c.close()


def setup_review(tmp_path, csv_path):
    service = Operations(tmp_path / "operations")
    snapshot = service.ingest(csv_path)
    book = service.save_playbook(
        {"name": "weekly", "metrics": [{"metric": "net_sales"}, {"metric": "cancellation_rate"}]}
    )
    spec = {
        "playbook_id": book["playbook_id"],
        "snapshot_id": snapshot["snapshot_id"],
        "start": "2011-11-01",
        "end": "2011-11-08",
    }
    return service, snapshot, book, spec


def test_snapshots_are_frozen_jobs_idempotent_and_evidence_resolves(tmp_path, csv_path):
    service, snapshot, book, spec = setup_review(tmp_path, csv_path)
    first = service.submit(spec)
    assert service.submit(spec)["job_id"] == first["job_id"]
    csv_path.write_text("source changed")
    result = service.execute(first["job_id"])
    assert result["status"] == "done"
    assert service.execute(first["job_id"]) == result
    packet = service.get("evidence", result["evidence_ids"][0])
    assert packet["values"][0]["value"] == 9.75
    assert packet["snapshot_hash"] == snapshot["snapshot_id"]
    report = render_report(report_payload(service.report_record(first["job_id"])))
    assert f'href="#computation-{packet["computation_id"]}"' in report
    assert f'id="computation-{packet["computation_id"]}"' in report
    assert "SELECT" in report
    with service.db() as db:
        assert db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 2


def test_restart_expired_lease_and_concurrent_execution_have_one_completion(tmp_path, csv_path):
    service, _, _, spec = setup_review(tmp_path, csv_path)
    job = service.submit(spec)
    with service.db() as db:
        db.execute(
            "UPDATE jobs SET status='running',owner='dead',lease_until=0 WHERE id=?",
            [job["job_id"]],
        )
    restarted = Operations(service.root)
    assert restarted.pending() == [job["job_id"]]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(restarted.execute, [job["job_id"]] * 2))
    assert any(r["status"] == "done" for r in results)
    assert restarted.get("jobs", job["job_id"])["status"] == "done"
    with service.db() as db:
        assert db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 2


def test_cancellation_and_corrupt_snapshot_cannot_publish_results(tmp_path, csv_path):
    service, snapshot, _, spec = setup_review(tmp_path, csv_path)
    job = service.submit(spec)
    assert service.cancel(job["job_id"])["status"] == "cancelled"
    assert service.execute(job["job_id"])["status"] == "cancelled"
    other = service.submit({**spec, "idempotency_key": "second"})
    path = service.root / "snapshots" / f"{snapshot['snapshot_id']}.csv"
    path.chmod(0o644)
    path.write_text("tampered")
    result = service.execute(other["job_id"])
    assert result["status"] == "error" and result["evidence_ids"] == []


def test_explicit_retry_retains_failure_history_and_cannot_duplicate_success(tmp_path, csv_path):
    service, _, _, spec = setup_review(tmp_path, csv_path)
    job = service.submit(spec)
    service.cancel(job["job_id"])
    retried = service.retry(job["job_id"])
    assert retried["status"] == "queued"
    assert retried["attempt_history"][0]["status"] == "cancelled"
    assert service.execute(job["job_id"])["status"] == "done"
    with pytest.raises(ValueError, match="explicitly retried"):
        service.retry(job["job_id"])


def test_currency_cells_do_not_round_away_pence_at_large_magnitudes(tmp_path, csv_path):
    with csv_path.open("a", newline="") as source:
        csv.writer(source).writerow(
            ["9", "Z", "large", "1", "2011-11-03 12:00:00", "330410.49", "007", "UK"]
        )
    service, _, _, spec = setup_review(tmp_path, csv_path)
    job = service.execute(service.submit(spec)["job_id"])
    html = render_report(report_payload(service.report_record(job["job_id"])))
    assert "<td>330420.24</td>" in html


def test_definition_and_period_changes_block_comparison(tmp_path, csv_path):
    service, _, _, spec = setup_review(tmp_path, csv_path)
    first = service.execute(service.submit(spec)["job_id"])
    second = service.execute(
        service.submit({**spec, "start": "2011-11-08", "end": "2011-11-15"})["job_id"]
    )
    diff = service.compare(first["job_id"], second["job_id"])
    assert diff["rows"][0]["delta"] == -9.75
    assert diff["rows"][1]["delta"] is None
    with pytest.raises(ValueError, match="nonoverlapping"):
        service.compare(first["job_id"], first["job_id"])
    other = service.save_playbook(
        {"name": "new definitions", "metrics": [{"metric": "gross_sales"}]}
    )
    incompatible = service.execute(
        service.submit({**spec, "playbook_id": other["playbook_id"]})["job_id"]
    )
    with pytest.raises(ValueError, match="incompatible"):
        service.compare(first["job_id"], incompatible["job_id"])


def test_orphaned_snapshot_file_is_recoverable_after_ingestion_crash(tmp_path, csv_path):
    service, snapshot, _, _ = setup_review(tmp_path, csv_path)
    with service.db() as db:
        db.execute("DELETE FROM snapshots WHERE id=?", [snapshot["snapshot_id"]])
    recovered = service.ingest(csv_path)
    assert recovered["snapshot_id"] == snapshot["snapshot_id"]
    assert service.get("snapshots", snapshot["snapshot_id"])["rows"] == recovered["rows"]


def test_query_time_budget_interrupts_actual_duckdb_work(tmp_path, csv_path, monkeypatch):
    service, _, _, spec = setup_review(tmp_path, csv_path)

    def expensive(connection, *args):
        return connection.execute("SELECT sum(i) FROM range(1000000000000) t(i)").fetchall()

    monkeypatch.setattr("analyst.operations.service.compute", expensive)
    job = service.submit({**spec, "timeout_s": 1})
    result = service.execute(job["job_id"])
    assert result["status"] == "error" and result["evidence_ids"] == []
    assert result["duration_s"] < 5


def test_definition_drift_and_unknown_export_fields_are_rejected(tmp_path, csv_path, monkeypatch):
    from analyst.operations.metrics import fingerprint

    service, _, _, spec = setup_review(tmp_path, csv_path)
    job = service.execute(service.submit(spec)["job_id"])
    record = service.report_record(job["job_id"])
    packet = record["operations_evidence"][0]
    packet["internal_log"] = "must not be exported"
    packet["computation_id"] = fingerprint(
        {k: v for k, v in packet.items() if k != "computation_id"}
    )
    with pytest.raises(ValueError, match="not exportable"):
        report_payload(record)
    monkeypatch.setattr("analyst.operations.service.contract_hash", lambda: "changed")
    with pytest.raises(ValueError, match="incompatible"):
        service.submit(spec)


def test_api_worker_completes_and_downloads_evidence(tmp_path, csv_path, monkeypatch):
    monkeypatch.setenv("ANALYST_OPERATIONS_DIR", str(tmp_path / "api-operations"))
    from analyst.api.app import app

    with TestClient(app) as client:
        snapshot = client.post("/operations/snapshots", json={"source_path": str(csv_path)}).json()
        book = client.post(
            "/operations/playbooks", json={"name": "weekly", "metrics": [{"metric": "net_sales"}]}
        ).json()
        response = client.post(
            "/operations/jobs",
            json={
                "snapshot_id": snapshot["snapshot_id"],
                "playbook_id": book["playbook_id"],
                "start": "2011-11-01",
                "end": "2011-11-08",
            },
        )
        assert response.status_code == 202
        identity = response.json()["job_id"]
        for _ in range(100):
            result = client.get(f"/operations/jobs/{identity}").json()
            if result["status"] == "done":
                break
            time.sleep(0.03)
        assert result["status"] == "done"
        evidence = result["evidence"][0]
        assert client.get(f"/operations/evidence/{evidence['computation_id']}").json() == evidence
        assert client.get(f"/operations/jobs/{identity}/report").status_code == 200
        assert (
            client.get(f"/operations/jobs/{identity}/report?format=json").json()[
                "operations_evidence"
            ][0]
            == evidence
        )
