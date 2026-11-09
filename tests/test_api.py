"""Phase (e): API e2e in dev-inprocess mode — create run, stream WebSocket to
completion, verify persisted state. Zero Redis, zero Docker, zero LLM calls."""
import importlib
import json

import pytest
from fastapi.testclient import TestClient

from analyst.eval.data_gen import generate

PLAN = """{"tasks": [
  {"id": "t1", "goal": "count work orders", "depends_on": [], "kind": "python"}
]}"""
CODE = ('```python\nimport os, json\nimport pandas as pd\n'
        'df = pd.read_csv(os.path.join(os.environ["DATA_DIR"], "work_orders.csv"))\n'
        'json.dump({"answer": f"There are {len(df)} work orders.", '
        '"data": {"n": len(df)}}, open("result.json", "w"))\n```')
APPROVE = '{"verdict": "approve", "reason": "fine"}'
V_PASS = '{"claims": [{"task_id": "t1", "claim": "count computed", "verdict": "pass", "note": ""}]}'
COUNCIL_SUPPORT = json.dumps({
    "verdict": "support", "conclusion": "There are 3000 work orders.",
    "answer_key": "3000_work_orders", "evidence_refs": ["t1"], "risks": [],
    "confidence": 0.9, "recommended_action": "accept",
    "target_task_id": None, "instruction": "",
})


@pytest.fixture()
def client(tmp_path, monkeypatch):
    data_dir = tmp_path / "sample"
    generate(data_dir, seed=7)

    script = {"planner": [{"text": PLAN}], "executor": [{"text": CODE}],
              "critic": [{"text": APPROVE}],
              "council_metric": [{"text": COUNCIL_SUPPORT}],
              "council_statistical": [{"text": COUNCIL_SUPPORT}],
              "council_data_quality": [{"text": COUNCIL_SUPPORT}],
              "validator": [{"text": V_PASS}]}
    script_file = tmp_path / "script.json"
    script_file.write_text(json.dumps(script))

    monkeypatch.setenv("ANALYST_DEV_INPROCESS", "1")
    monkeypatch.setenv("ANALYST_LLM_PROVIDER", "replay")
    monkeypatch.setenv("ANALYST_CASSETTE_SCRIPT", str(script_file))
    monkeypatch.setenv("ANALYST_SANDBOX", "subprocess")
    monkeypatch.setenv("ANALYST_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ANALYST_MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.setenv("ANALYST_UPLOAD_DIR", str(tmp_path / "uploads"))

    import analyst.api.app as app_module
    importlib.reload(app_module)  # rebuild store/bus with the patched env
    return TestClient(app_module.app), str(data_dir)


def test_create_stream_and_persist(client):
    api, data_dir = client
    r = api.post("/runs", json={"question": "how many work orders?",
                                "data_dir": data_dir, "no_memory": True,
                                "no_critic": False,
                                "council_mode": "always"})
    assert r.status_code == 202
    run_id = r.json()["run_id"]

    # The story must arrive EITHER live (node_update/answer events) or via the
    # replay message (if the run finished before we connected) — the client
    # protocol treats both as equivalent, so the test does too.
    saw_tasks = saw_answer = False
    with api.websocket_connect(f"/runs/{run_id}/stream") as ws:
        while True:
            msg = ws.receive_json()
            if msg["type"] == "replay":
                run = msg["run"]
                if run.get("dag"):
                    saw_tasks = True
                if run.get("answer"):
                    saw_answer = True
                    assert "3000 work orders" in run["answer"]
            if msg["type"] == "node_update" and msg["tasks"]:
                saw_tasks = True
            if msg["type"] == "answer":
                saw_answer = True
                assert "3000 work orders" in msg["answer"]
            if msg["type"] == "stream_end":
                break
    assert saw_tasks and saw_answer

    run = api.get(f"/runs/{run_id}").json()
    assert run["status"] == "done"
    assert "3000" in run["answer"]
    assert any(e["node"] == "critic" for e in run["events"])  # full audit persisted
    assert len(run["council_findings"]) == 3
    assert run["council_decisions"][-1]["status"] == "consensus"
    assert run["council_routing"]["mode"] == "always"
    assert run["council_routing"]["invoked"] is True
    assert run["statistical_evidence"] == []
    assert run["predictive_evidence"] == []
    assert run["data_quality_report"]["table_count"] == 7
    assert run["data_quality_report"]["total_rows"] > 0
    assert api.get("/runs").json()[0]["run_id"] == run_id

    report = api.get(f"/runs/{run_id}/report")
    assert report.status_code == 200
    assert "3000 work orders" in report.text
    assert "attachment" in report.headers["content-disposition"]
    assert report.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'none'" in report.headers["content-security-policy"]
    evidence = api.get(f"/runs/{run_id}/report?format=json").json()
    assert evidence["dataset"]["content_hash"] == run["dataset_provenance"]["content_hash"]
    assert len(evidence["dataset"]["tables"]) == 7
    assert len(evidence["council_findings"]) == 3
    assert "events" not in evidence and "options" not in evidence
    assert run["created_at"] <= run["started_at"] <= run["finished_at"]

    # Export describes the data used during execution, even if source files change.
    from pathlib import Path

    (Path(data_dir) / "work_orders.csv").write_text("changed,data\n1,2\n")
    assert api.get(f"/runs/{run_id}/report?format=json").json()["dataset"] == evidence["dataset"]


def test_late_joiner_gets_full_replay(client):
    api, data_dir = client
    r = api.post("/runs", json={"question": "how many work orders?",
                                "data_dir": data_dir, "no_memory": True,
                                "council_mode": "always"})
    run_id = r.json()["run_id"]

    # wait for completion via polling (as a reconnecting client would)
    import time
    for _ in range(100):
        if api.get(f"/runs/{run_id}").json()["status"] == "done":
            break
        time.sleep(0.1)

    with api.websocket_connect(f"/runs/{run_id}/stream") as ws:
        replay = ws.receive_json()
        assert replay["type"] == "replay"
        assert replay["run"]["status"] == "done"
        assert ws.receive_json()["type"] == "stream_end"


def test_evidence_backed_product_defaults():
    from analyst.api.app import CreateRun

    request = CreateRun(question="how many work orders?")
    assert request.no_critic is True
    assert request.no_validator is False
    assert request.council_mode == "off"


def test_unknown_run_404(client):
    api, _ = client
    assert api.get("/runs/nope").status_code == 404
    assert api.get("/runs/nope/report").status_code == 404


@pytest.mark.parametrize("status", ["queued", "running", "error"])
def test_report_requires_a_completed_run(client, status):
    api, _ = client
    import analyst.api.app as app_module

    app_module.store.create("incomplete", "a question", "data/sample", {})
    app_module.store.update("incomplete", status=status)
    assert api.get("/runs/incomplete/report").status_code == 409
    assert api.get("/runs/incomplete/report?format=exe").status_code == 422


def test_browser_csv_upload_builds_dataset_manifest(client):
    api, _ = client
    response = api.post(
        "/datasets",
        json={
            "files": [
                {"name": "orders.csv", "content": "order_id,amount\n1,12.5\n2,8.0\n"},
                {"name": "customers.csv", "content": "customer_id,segment\n1,A\n"},
            ]
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert {table["name"] for table in payload["tables"]} == {"orders", "customers"}
    assert payload["content_hash"]
    assert payload["data_dir"].endswith(payload["dataset_id"])
    assert payload["data_quality_report"]["table_count"] == 2
    assert payload["data_quality_report"]["total_rows"] == 3
    assert "quality_score" in payload["data_quality_report"]


def test_browser_csv_upload_rejects_path_traversal(client):
    api, _ = client
    response = api.post(
        "/datasets",
        json={"files": [{"name": "../secret.csv", "content": "id\n1\n"}]},
    )

    assert response.status_code == 400
    assert "invalid CSV filename" in response.json()["detail"]
