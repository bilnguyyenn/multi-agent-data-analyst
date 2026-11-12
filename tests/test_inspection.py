import csv

import pytest
from fastapi.testclient import TestClient

from analyst.operations.inspection import render_inspection
from analyst.operations.metrics import COLUMNS
from analyst.operations.service import Operations


@pytest.fixture
def investigation(tmp_path):
    service = Operations(tmp_path / "state")
    book = service.save_playbook(
        {
            "name": "investigate",
            "metrics": [
                {"metric": metric}
                for metric in ["net_sales", "gross_sales", "cancellation_value", "units_sold"]
            ],
        }
    )
    jobs = []
    for label, start, end, rows in [
        (
            "old",
            "2011-11-01",
            "2011-11-08",
            [
                [
                    "1",
                    "P",
                    "widget <script>unsafe</script>",
                    "2",
                    "2011-11-02",
                    "1.255",
                    "001",
                    "UK",
                ],
                [
                    "1",
                    "P",
                    "widget <script>unsafe</script>",
                    "2",
                    "2011-11-02",
                    "1.255",
                    "001",
                    "UK",
                ],
                ["C2", "P", "cancel", "-1", "2011-11-03", "1.255", "001", "UK"],
                ["3", "P", "widget France", "1", "2011-11-04", "2.00", "", "France"],
                ["4", "P", "excluded zero price", "9", "2011-11-04", "0", "", "UK"],
                ["5", "Q", "other", "1", "2011-11-04", "10", "", "UK"],
                ["6", "X' OR 1=1--", "literal code", "1", "2011-11-04", "1", "", "UK"],
            ],
        ),
        (
            "new",
            "2011-11-08",
            "2011-11-15",
            [
                ["C7", "P", "correction", "-2", "2011-11-09", "2.005", "", "UK"],
                ["8", "P", "excluded fractional unit", "1.5", "2011-11-09", "100", "", "UK"],
                ["9", "Q", "other", "1", "2011-11-09", "5", "", "UK"],
            ],
        ),
    ]:
        path = tmp_path / f"{label}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        snapshot = service.ingest(path)
        result = service.execute(
            service.submit(
                {
                    "snapshot_id": snapshot["snapshot_id"],
                    "playbook_id": book["playbook_id"],
                    "start": start,
                    "end": end,
                }
            )["job_id"]
        )
        assert result["status"] == "done"
        jobs.append(result["job_id"])
    return service, jobs


def test_source_inspection_retains_duplicates_rounding_and_omission_counts(investigation):
    service, jobs = investigation
    parent = service.explain_change(*jobs)
    packet = service.inspect_contributor(
        parent["computation_id"], {"segment": "P", "sample_limit": 1}
    )
    old, new = packet["periods"]
    assert old["totals"] == {
        "eligible_rows": 4,
        "gross_pence": 704,
        "cancellation_pence": 126,
        "net_pence": 578,
        "units_sold": 5,
        "units_sold_display": "5",
        "gross": "7.04",
        "cancellation": "1.26",
        "net": "5.78",
    }
    assert new["totals"]["net"] == "-4.02"
    assert old["sampled_rows"] == 1 and old["omitted_rows"] == 3
    assert old["sample"][0]["source"]["CustomerID"] == "001"
    assert old["sample"][0]["signed_value"] == "2.52"
    assert packet["contributor"]["delta"] == "-9.80"
    assert packet["reconciliation_residual_base_units"] == 0
    for period in packet["periods"]:
        assert sum(d["net_pence"] for d in period["daily"]) == period["totals"]["net_pence"]
        assert sum(d["eligible_rows"] for d in period["daily"]) == period["totals"]["eligible_rows"]
    assert (
        Operations(service.root).inspect_contributor(
            parent["computation_id"], {"segment": "P", "sample_limit": 1}
        )
        == packet
    )
    html = render_inspection(packet)
    assert "<script>unsafe</script>" not in html and "&lt;script&gt;unsafe&lt;/script&gt;" in html
    assert "Showing 1 of 4 eligible rows; 3 omitted" in html
    with pytest.raises(ValueError, match="fingerprint"):
        render_inspection({**packet, "metric": "invented"})


@pytest.mark.parametrize("metric", ["gross_sales", "cancellation_value", "units_sold"])
def test_each_additive_parent_metric_reconciles(investigation, metric):
    service, jobs = investigation
    parent = service.explain_change(*jobs, {"metric": metric})
    packet = service.inspect_contributor(parent["computation_id"], {"segment": "P"})
    assert packet["metric"] == metric
    assert packet["reconciliation_residual_base_units"] == 0
    assert len(packet["periods"][0]["sample"]) == 4
    assert packet["periods"][0]["sample"][0] == packet["periods"][0]["sample"][1]


def test_country_inspection_handles_absent_period_and_literal_product_codes(investigation):
    service, jobs = investigation
    parent = service.explain_change(*jobs, {"group_by": "country"})
    packet = service.inspect_contributor(parent["computation_id"], {"segment": "France"})
    assert packet["periods"][0]["totals"]["net"] == "2.00"
    assert packet["periods"][1]["totals"]["eligible_rows"] == 0
    assert packet["periods"][1]["sample"] == packet["periods"][1]["daily"] == []
    parent = service.explain_change(*jobs)
    packet = service.inspect_contributor(parent["computation_id"], {"segment": "X' OR 1=1--"})
    assert packet["periods"][0]["totals"]["eligible_rows"] == 1
    assert packet["periods"][0]["queries"][0]["parameters"][-1] == "X' OR 1=1--"
    assert "X' OR 1=1--" not in packet["periods"][0]["queries"][0]["sql"]


def test_unlisted_segment_bad_limits_and_noncontribution_parent_rejected(investigation):
    service, jobs = investigation
    parent = service.explain_change(*jobs, {"limit": 1})
    with pytest.raises(ValueError, match="displayed"):
        service.inspect_contributor(parent["computation_id"], {"segment": "nonexistent"})
    for size in [0, 26]:
        with pytest.raises(ValueError):
            service.inspect_contributor(
                parent["computation_id"], {"segment": "P", "sample_limit": size}
            )
    with pytest.raises(ValueError, match="contribution evidence"):
        service.inspect_contributor(parent["evidence_refs"][0], {"segment": "P"})


def test_corruption_size_limit_and_contract_drift_do_not_publish(investigation, monkeypatch):
    import analyst.operations.inspection as module

    service, jobs = investigation
    parent = service.explain_change(*jobs)
    with service.db() as db:
        before = db.execute("SELECT count(*) FROM evidence").fetchone()[0]
    monkeypatch.setattr(module, "MAX_PACKET_BYTES", 50)
    with pytest.raises(ValueError, match="512 KiB"):
        service.inspect_contributor(parent["computation_id"], {"segment": "P"})
    monkeypatch.undo()
    monkeypatch.setattr(module, "ENGINE_SHA256", "changed")
    with pytest.raises(ValueError, match="current metric engine"):
        service.inspect_contributor(parent["computation_id"], {"segment": "P"})
    monkeypatch.undo()
    real = module.inspect_period

    def changed_source(path, *args):
        result = real(path, *args)
        path.chmod(0o644)
        path.write_text("changed after reading")
        return result

    monkeypatch.setattr(module, "inspect_period", changed_source)
    with pytest.raises(ValueError, match="snapshot changed"):
        service.inspect_contributor(parent["computation_id"], {"segment": "P"})
    with service.db() as db:
        assert db.execute("SELECT count(*) FROM evidence").fetchone()[0] == before


def test_parent_reconciliation_is_checked_before_publication(investigation, monkeypatch):
    import analyst.operations.inspection as module

    service, jobs = investigation
    parent = service.explain_change(*jobs)
    real = module.inspect_period

    def wrong_total(*args):
        result = real(*args)
        result["totals"]["net_pence"] += 1
        return result

    monkeypatch.setattr(module, "inspect_period", wrong_total)
    with pytest.raises(ValueError, match="source totals"):
        service.inspect_contributor(parent["computation_id"], {"segment": "P"})


def test_api_and_cli_export_recorded_source_inspection(
    investigation, monkeypatch, tmp_path, capsys
):
    from analyst.api.app import app
    from analyst.api.operations import _store
    from analyst.cli import main

    service, jobs = investigation
    parent = service.explain_change(*jobs)
    monkeypatch.setenv("ANALYST_OPERATIONS_DIR", str(service.root))
    _store.cache_clear()
    with TestClient(app) as client:
        result = client.get(
            f"/operations/changes/{parent['computation_id']}/inspect", params={"segment": "P"}
        )
        assert result.status_code == 200
        packet = result.json()
        report = client.get(f"/operations/inspections/{packet['computation_id']}/report")
        assert (
            report.status_code == 200
            and "default-src 'none'" in report.headers["content-security-policy"]
        )
        assert "Source inspection: P" in report.text
        assert client.get(f"/operations/evidence/{packet['computation_id']}").json() == packet
        assert (
            client.get(
                f"/operations/changes/{parent['computation_id']}/inspect",
                params={"segment": "P", "sample_limit": 100},
            ).status_code
            == 409
        )
    output = tmp_path / "inspection.html"
    assert (
        main(
            [
                "retail-inspect",
                parent["computation_id"],
                "P",
                "--root",
                str(service.root),
                "--out",
                str(output),
            ]
        )
        == 0
    )
    assert output.is_file()
    assert packet["computation_id"] in capsys.readouterr().out


def test_explicit_country_filter_is_preserved(investigation):
    service, original_jobs = investigation
    book = service.save_playbook(
        {"name": "UK only", "metrics": [{"metric": "net_sales", "country": "UK"}]}
    )
    jobs = []
    for identity in original_jobs:
        spec = service.get("jobs", identity)["spec"]
        job = service.execute(
            service.submit(
                {
                    "playbook_id": book["playbook_id"],
                    **{k: spec[k] for k in ["snapshot_id", "start", "end"]},
                }
            )["job_id"]
        )
        jobs.append(job["job_id"])
    parent = service.explain_change(*jobs)
    packet = service.inspect_contributor(parent["computation_id"], {"segment": "P"})
    assert packet["periods"][0]["totals"]["net"] == "3.78"
    assert all(p["period"]["country"] == "UK" for p in packet["periods"])
    assert all(r["source"]["Country"] == "UK" for p in packet["periods"] for r in p["sample"])


def test_daily_and_sample_currency_preserve_pence_above_float_precision(tmp_path):
    from analyst.operations.inspection import InspectionRequest, inspect_period

    path = tmp_path / "large.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        writer.writerow(["1", "BIG", "large", "9007199254740993", "2011-11-02", "0.01", "1", "UK"])
    parent = {
        "group_by": "product",
        "periods": [{"start": "2011-11-01", "end": "2011-11-08", "country": None}],
        "snapshot_hashes": ["source"],
    }
    result = inspect_period(path, parent, 0, InspectionRequest(segment="BIG"))
    assert result["totals"]["net"] == "90071992547409.93"
    assert result["totals"]["units_sold_display"] == "9007199254740993"
    assert result["daily"][0]["net"] == result["sample"][0]["signed_value"] == "90071992547409.93"
