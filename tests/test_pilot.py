import json
from pathlib import Path

import pytest

from analyst.operations.pilot import analyze, prepare_pilot

ROOT = Path(__file__).resolve().parents[1]


def test_prepared_pilot_does_not_invent_participant_outcomes(tmp_path):
    protocol = prepare_pilot(ROOT / "data/retail", tmp_path)
    observations = json.loads((tmp_path / "observations.json").read_text())
    result = analyze(protocol, observations)
    assert result["status"] == "pending_participants"
    assert result["participants"] == 0
    assert result["conditions"]["manual"]["median_seconds_correct_only"] is None
    handout = json.loads((tmp_path / "participant-tasks.json").read_text())
    assert all("expected" not in task for task in handout["tasks"])


def test_scorer_rejects_simulation_and_accounts_for_wrong_answers(tmp_path):
    # These are temporary unit-test fixtures, never published as participant evidence.
    protocol = prepare_pilot(ROOT / "data/retail", tmp_path)
    observations = {
        "protocol_hash": protocol["protocol_hash"],
        "participants": [{"id": "fixture", "consent": True, "sequence_group": "AB"}],
        "trials": [
            {
                "participant": "fixture",
                "task_id": "pair1A",
                "condition": "manual",
                "answer": None,
                "elapsed_seconds": 20,
                "synthetic": True,
            }
        ],
    }
    with pytest.raises(ValueError, match="synthetic"):
        analyze(protocol, observations)
    observations["trials"][0]["synthetic"] = False
    result = analyze(protocol, observations)
    assert result["status"] == "incomplete"
    assert result["observed_trials"] == 1 and result["expected_trials"] == 6
    assert result["conditions"]["manual"]["correct"] == 0
    observations["trials"].append(dict(observations["trials"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        analyze(protocol, observations)


def test_resource_preflight_blocks_before_model_construction(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from analyst.api.app import app

    monkeypatch.setenv("ANALYST_OPERATIONS_DIR", str(tmp_path / "operations"))
    monkeypatch.setattr(
        "analyst.operations.preflight.local_preflight",
        lambda: {"allowed": False, "reasons": ["fixture memory shortage"]},
    )
    monkeypatch.setattr(
        "analyst.llm.client.make_client", lambda *a, **kw: pytest.fail("model access")
    )
    with TestClient(app) as client:
        response = client.post("/operations/interpret", json={"question": "What is net sales?"})
        assert response.status_code == 409
        assert "memory shortage" in response.json()["detail"]
        assert (
            client.post(
                "/operations/interpret",
                json={
                    "question": "test",
                    "prior_request": {"metric": "net_sales", "sql": "not allowed"},
                },
            ).status_code
            == 422
        )
