import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from analyst.operations.changes import render_bridge
from analyst.operations.metrics import COLUMNS, REGISTRY
from analyst.operations.reference import reference_values
from analyst.operations.service import Operations


def reviews(root, *, metric_requests=None):
    service = Operations(root / "state")
    book = service.save_playbook(
        {
            "name": "change review",
            "metrics": metric_requests or [{"metric": name} for name in REGISTRY],
        }
    )
    jobs = []
    for label, start, end, rows in [
        (
            "old",
            "2011-11-01",
            "2011-11-08",
            [
                ["1", "A", "A", 1, "2011-11-02", 100, "1", "UK"],
                ["2", "B", "B", 1, "2011-11-02", 80, "1", "UK"],
                ["3", "<script>C</script>", "C", 1, "2011-11-02", 10, "", "France"],
                ["C4", "A", "A", -1, "2011-11-02", 20, "1", "UK"],
            ],
        ),
        (
            "new",
            "2011-11-08",
            "2011-11-15",
            [
                ["5", "A", "A", 1, "2011-11-09", 90, "1", "UK"],
                ["6", "B", "B", 1, "2011-11-09", 120, "1", "UK"],
                ["7", "D", "D", 1, "2011-11-09", 25, "", "France"],
                ["C8", "B", "B", -1, "2011-11-09", 5, "1", "UK"],
            ],
        ),
    ]:
        path = root / f"{label}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        snapshot = service.ingest(path)
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
        assert job["status"] == "done"
        jobs.append(job["job_id"])
    return service, jobs


def test_exact_bridge_preserves_remainder_cohorts_and_ledger(tmp_path):
    service, jobs = reviews(tmp_path)
    packet = service.explain_change(*jobs, {"limit": 1})
    assert packet["delta"] == "60.00"
    assert packet["rows"][0]["segment"] == "B"
    assert packet["rows"][0]["delta"] == "35.00"
    assert packet["remainder"]["delta"] == "25.00"
    assert packet["remainder"]["group_count"] == 3
    assert packet["reconciliation_residual_base_units"] == 0
    assert {r["cohort"]: r["delta"] for r in packet["cohorts"]} == {
        "continuing": "45.00",
        "newly_observed": "25.00",
        "no_longer_observed": "-10.00",
    }
    assert service.get("evidence", packet["computation_id"]) == packet
    assert Operations(service.root).explain_change(*jobs, {"limit": 1}) == packet
    assert packet["queries"][0]["parameters"] == ["2011-11-01", "2011-11-08"]
    assert "LIMIT" not in packet["queries"][0]["sql"]
    html = render_bridge(service.explain_change(*jobs))
    assert "<script>C</script>" not in html
    assert "&lt;script&gt;C&lt;/script&gt;" in html
    assert "All other groups (0)" in html
    with pytest.raises(ValueError, match="fingerprint"):
        render_bridge({**packet, "delta": "invented"})


def test_truncated_comparison_absence_is_unknown_and_bridge_is_complete(tmp_path):
    service, jobs = reviews(
        tmp_path,
        metric_requests=[
            {"metric": "net_sales"},
            {"metric": "net_sales", "group_by": "product", "limit": 1},
        ],
    )
    comparison = service.compare(*jobs)
    grouped = [r for r in comparison["rows"] if r["segment"] != "all"]
    assert {r["segment"] for r in grouped} == {"A", "B"}
    assert all(r["delta"] is None for r in grouped)
    bridge = service.explain_change(*jobs)
    assert bridge["group_count"] == 4
    assert sum(r["delta_base_units"] for r in bridge["rows"]) == 6000


@pytest.mark.parametrize(
    "proposal",
    [
        {"metric": "active_customers"},
        {"metric": "order_count"},
        {"metric": "cancellation_rate"},
        {"group_by": "product; DROP TABLE raw"},
        {"limit": 0},
        {"limit": 101},
    ],
)
def test_unsafe_or_nonadditive_breakdowns_are_rejected(tmp_path, proposal):
    service = Operations(tmp_path / "state")
    with pytest.raises(ValueError):
        service.explain_change("missing", "missing", proposal)


def test_changed_source_engine_and_order_cannot_publish_bridge(tmp_path, monkeypatch):
    service, jobs = reviews(tmp_path)
    with pytest.raises(ValueError, match="earlier"):
        service.explain_change(*reversed(jobs))
    monkeypatch.setattr("analyst.operations.metrics.ENGINE_SHA256", "changed")
    with pytest.raises(ValueError, match="current metric engine"):
        service.explain_change(*jobs)
    monkeypatch.undo()
    job = service.get("jobs", jobs[0])
    path = service.root / "snapshots" / f"{job['spec']['snapshot_id']}.csv"
    path.chmod(0o644)
    path.write_text("changed")
    with pytest.raises(ValueError, match="snapshot content changed"):
        service.explain_change(*jobs)


def test_country_filter_is_applied_and_nonoverall_playbooks_rejected(tmp_path):
    service, jobs = reviews(tmp_path, metric_requests=[{"metric": "net_sales", "country": "UK"}])
    packet = service.explain_change(*jobs, {"group_by": "country"})
    assert packet["delta"] == "45.00" and packet["group_count"] == 1
    assert packet["queries"][0]["parameters"][-1] == "UK"
    other = tmp_path / "other"
    other.mkdir()
    service, jobs = reviews(other, metric_requests=[{"metric": "net_sales", "group_by": "country"}])
    with pytest.raises(ValueError, match="overall"):
        service.explain_change(*jobs)


def test_change_api_report_and_schema_validation(tmp_path, monkeypatch):
    from analyst.api.app import app
    from analyst.api.operations import _store

    service, jobs = reviews(tmp_path)
    monkeypatch.setenv("ANALYST_OPERATIONS_DIR", str(service.root))
    _store.cache_clear()
    with TestClient(app) as client:
        params = {"previous": jobs[0], "current": jobs[1], "group_by": "product"}
        response = client.get("/operations/changes", params=params)
        assert response.status_code == 200
        packet = response.json()
        report = client.get(f"/operations/changes/{packet['computation_id']}/report")
        assert report.status_code == 200 and "60.00" in report.text
        assert "default-src 'none'" in report.headers["content-security-policy"]
        assert (
            client.get(
                "/operations/changes", params={**params, "metric": "order_count"}
            ).status_code
            == 409
        )
        assert client.get(f"/operations/evidence/{packet['computation_id']}").json() == packet


@pytest.mark.parametrize("metric", ["gross_sales", "cancellation_value", "net_sales", "units_sold"])
@pytest.mark.parametrize("group_by", ["country", "product"])
def test_real_data_bridge_matches_independent_decimal_reference(tmp_path, metric, group_by):
    root = Path(__file__).resolve().parents[1]
    service = Operations(tmp_path / "state")
    book = service.save_playbook({"name": "real", "metrics": [{"metric": metric}]})
    jobs, expected = [], []
    for label, start, end in [
        ("development", "2011-11-01", "2011-11-08"),
        ("refresh", "2011-11-08", "2011-11-15"),
    ]:
        path = root / "data/retail" / label / "transactions.csv"
        snap = service.ingest(path)
        result = service.execute(
            service.submit(
                {
                    "snapshot_id": snap["snapshot_id"],
                    "playbook_id": book["playbook_id"],
                    "start": start,
                    "end": end,
                }
            )["job_id"]
        )
        assert result["status"] == "done"
        jobs.append(result["job_id"])
        expected.append(
            {
                r["segment"]: r["numerator"]
                for r in reference_values(
                    path,
                    {
                        "metric": metric,
                        "group_by": group_by,
                        "start": start,
                        "end": end,
                        "limit": 100_000,
                    },
                )
            }
        )
    packet = service.explain_change(*jobs, {"metric": metric, "group_by": group_by, "limit": 100})
    left, right = expected
    differences = {key: right.get(key, 0) - left.get(key, 0) for key in left.keys() | right.keys()}
    assert packet["delta_base_units"] == sum(differences.values())
    for row in packet["rows"]:
        assert row["delta_base_units"] == differences.pop(row["segment"])
    assert packet["remainder"]["delta_base_units"] == sum(differences.values())
    assert packet["remainder"]["group_count"] == len(differences)


def test_zero_net_change_retains_offsetting_movement_and_exact_large_amounts():
    from analyst.operations.changes import ChangeRequest, bridge

    large = 2**54 + 1
    common = {
        "metric": "net_sales",
        "unit": "GBP",
        "snapshot_hash": "a" * 64,
        "metric_contract_hash": "contract",
        "engine_sha256": "engine",
        "request": {"start": "2011-11-01", "end": "2011-11-08", "country": None},
    }
    before = {**common, "values": [{"numerator": large}], "computation_id": "before"}
    after = {**common, "values": [{"numerator": large}], "computation_id": "after"}
    left = {"values": {"A": (large, 1), "B": (0, 1)}, "sql": "SELECT", "parameters": []}
    right = {"values": {"A": (large - 123, 1), "B": (123, 1)}, "sql": "SELECT", "parameters": []}
    result = bridge(before, after, ChangeRequest(limit=1), left, right)
    assert result["delta"] == "0.00"
    assert result["absolute_movement_base_units"] == 246
    assert result["rows"][0]["delta"] == "-1.23"
    assert result["remainder"]["delta"] == "1.23"
    assert result["previous"] == "180143985094819.85"
    with pytest.raises(ValueError, match="reconcile"):
        bridge(before, after, ChangeRequest(), left, {**right, "values": {}})
