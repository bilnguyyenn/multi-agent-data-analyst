import json
import shutil
from pathlib import Path

import pytest

from analyst.eval.reproduce import reproduce
from analyst.eval.retail import build_contract, evaluate_local
from analyst.llm.client import LLMResponse
from analyst.operations.agent import propose, validate_proposal
from analyst.operations.metrics import MetricRequest


ROOT = Path(__file__).resolve().parents[1]


def test_published_scorecards_reproduce_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "analyst.llm.client.make_client", lambda *a, **k: pytest.fail("model access")
    )
    result = reproduce(ROOT, tmp_path)
    assert result["development_runs"] == 7 and result["heldout_runs"] == 2
    assert all((ROOT / path).is_file() for path in result["provenance_files"])
    assert (tmp_path / "DEVELOPMENT.md").exists()


def test_missing_published_provenance_fails(tmp_path):
    for folder in ["eval_results", "heldout_eval_results"]:
        (tmp_path / folder).mkdir()
        shutil.copyfile(ROOT / folder / "SCORECARD.md", tmp_path / folder / "SCORECARD.md")
    with pytest.raises(FileNotFoundError):
        reproduce(tmp_path, tmp_path / "out")


def test_contract_has_exact_task_and_conversation_set():
    contract = build_contract(ROOT / "data/retail")
    assert len(contract["tasks"]) == 100
    assert len(contract["chains"]) == 20
    assert len({t["id"] for t in contract["tasks"]}) == 100
    assert {t["expected_status"] for t in contract["tasks"]} == {
        "compute",
        "clarify",
        "unsupported",
    }
    assert len({c["turns"][1]["question"] for c in contract["chains"]}) == 5


def test_published_real_data_contract_has_complete_check_accounting():
    contract = build_contract(ROOT / "data/retail")
    published = json.loads((ROOT / "retail_results/contract.json").read_text())
    assert contract == published
    checked = json.loads((ROOT / "retail_results/contract-results.json").read_text())
    expected_ids = {t["id"] for t in contract["tasks"]} | {
        f"{c['id']}/{i}" for c in contract["chains"] for i in range(len(c["turns"]))
    }
    assert len(checked["results"]) == len(expected_ids) == 140
    assert {r["id"] for r in checked["results"]} == expected_ids
    assert all(r["passed"] for r in checked["results"])


class FakeClient:
    def __init__(self):
        self.calls = []

    def complete(self, *, role, system, user):
        self.calls.append(role)
        return LLMResponse(
            text='{"status":"compute","request":{"metric":"net_sales"}}',
            model="scripted",
            input_tokens=20,
            output_tokens=10,
        )


def test_review_modes_are_explicit_and_use_the_same_tool_contract():
    for policy, calls in [("single", 1), ("standard", 2), ("high", 4)]:
        client = FakeClient()
        result = propose(client, "What is net sales?", policy)
        assert result["request"]["metric"] == "net_sales"
        assert len(client.calls) == calls
        assert result["tokens"] == calls * 30
    with pytest.raises(ValueError):
        validate_proposal(
            {"status": "compute", "request": {"metric": "net_sales", "sql": "DROP TABLE raw"}}
        )


def test_agent_evaluation_resumes_without_omitting_incorrect_attempts(tmp_path):
    contract = build_contract(ROOT / "data/retail")
    contract["tasks"] = [
        {
            "id": "correct",
            "split": "evaluation_a",
            "question": "What is net sales?",
            "expected_status": "compute",
            "request": MetricRequest(metric="net_sales").model_dump(mode="json"),
        },
        {
            "id": "wrong",
            "split": "evaluation_a",
            "question": "What is profit?",
            "expected_status": "unsupported",
        },
    ]
    contract["chains"] = []
    client = FakeClient()
    result = evaluate_local(contract, ROOT / "data/retail", tmp_path, client, ["single"], 2)
    assert result["summaries"][0]["accuracy"] == 0.5
    assert result["summaries"][0]["items"] == 2
    assert result["summaries"][0]["tokens"] == 60
    client.complete = lambda **kw: pytest.fail("resume repeated a completed call")
    assert evaluate_local(contract, ROOT / "data/retail", tmp_path, client, ["single"], 2) == result
    lines = [
        json.loads(line) for line in (tmp_path / "local-attempts.jsonl").read_text().splitlines()
    ]
    assert sum(r["type"] == "result" for r in lines) == 2
    with pytest.raises(ValueError, match="ceiling"):
        evaluate_local(contract, ROOT / "data/retail", tmp_path, client, ["high"], 1)
