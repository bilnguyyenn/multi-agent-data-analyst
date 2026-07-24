"""Phase (f): eval machinery — scoring rules, gold derivation, scorecard render.
(The harness's LLM-driven path is exercised live; the SCORING must be exact.)"""
import pytest

from analyst.eval.benchmark import Item, build_benchmark, score_item
from analyst.eval.data_gen import generate
from analyst.eval.harness import run_integrity_errors, select_latest_valid_runs
from analyst.eval.scorecard import render


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data") / "sample"
    generate(d, seed=7)
    return str(d)


def test_benchmark_builds_25_items_with_gold(data_dir):
    items = build_benchmark(data_dir)
    assert len(items) == 25
    tiers = {t: sum(1 for i in items if i.tier == t) for t in (1, 2, 3, 4)}
    assert tiers == {1: 6, 2: 7, 3: 7, 4: 5}
    top = next(i for i in items if i.id == "t1_top_downtime")
    assert top.contains_any == ["M13"] and abs(top.number[0] - 196.5) < 1


def test_plain_scoring_number_and_entity(data_dir):
    item = Item("x", 1, "q", number=(3000, 0), contains_any=["work order"])
    assert score_item(item, "There are 3000 work orders.", [])["correct"] == 1.0
    assert score_item(item, "There are 2990 work orders.", [])["correct"] == 0.0
    # number can live in an artifact instead of the prose
    assert score_item(item, "count of work orders computed", [{"n": 3000}])["correct"] == 1.0


def test_trap_scoring_rewards_honesty():
    trap = Item("t", 4, "q", contains_any=["day"], wrong_markers=["night"],
                trap_keywords=["mix", "confound"])
    confident_wrong = score_item(trap, "Night shift is best.", [])
    assert confident_wrong == {"correct": 0.0, "flagged": False}
    hedged = score_item(
        trap, "Night shift has the lowest rate, but the job mix differs by shift.", [])
    assert hedged == {"correct": 0.5, "flagged": True}
    right = score_item(trap, "Adjusted for mix, day shift is best.", [])
    assert right["correct"] == 1.0


def test_scorecard_renders(data_dir):
    fake_runs = [{
        "config": "full",
        "results": [
            {"id": "a", "tier": 1, "question": "q", "correct": 1.0, "flagged": None,
             "latency_s": 3.0, "tokens": 900, "answer": "", "error": None},
            {"id": "b", "tier": 4, "question": "q", "correct": 0.5, "flagged": True,
             "latency_s": 9.0, "tokens": 2500, "answer": "", "error": None},
        ],
    }]
    bench = {"recall": 0.83, "precision": 1.0, "false_positive_rate": 0.0,
             "rows": [{"id": "fanout_join", "flaw_class": "join fan-out",
                       "verdict": "revise", "detected": True}]}
    md = render(fake_runs, bench)
    assert "| full | **0.75**" in md
    assert "Recall (flaws caught):** 0.83" in md
    assert "fanout_join" in md


def _saved_run(config="full", *, n=25, error=None):
    return {
        "config": config,
        "aborted": False,
        "results": [
            {
                "id": f"q{i}", "tier": 1, "question": "q", "correct": 1.0,
                "flagged": None, "latency_s": 1.0, "tokens": 10,
                "answer": "ok", "error": error,
            }
            for i in range(n)
        ],
    }


def test_scorecard_integrity_rejects_partial_and_errored_runs():
    assert "expected 25 results, found 5" in run_integrity_errors(_saved_run(n=5))
    assert "25 result(s) contain errors" in run_integrity_errors(
        _saved_run(error="cassette miss")
    )


def test_latest_broken_run_cannot_replace_valid_metrics(tmp_path):
    import json

    good = _saved_run()
    bad = _saved_run(error="cassette miss")
    (tmp_path / "raw_20260101_000000.json").write_text(json.dumps([good]))
    (tmp_path / "raw_20260102_000000.json").write_text(json.dumps([bad]))

    selected, sources, rejected = select_latest_valid_runs(tmp_path)

    assert selected["full"] == good
    assert sources["full"].name == "raw_20260101_000000.json"
    assert len(rejected) == 1
