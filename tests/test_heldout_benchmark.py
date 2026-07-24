from __future__ import annotations

import json

import pytest

from analyst.eval.heldout_benchmark import (
    EXPECTED_HELDOUT_IDS,
    HeldoutItem,
    build_heldout_benchmark,
    number_check,
    score_heldout_item,
    text_check,
)
from analyst.eval.heldout_data import generate_heldout
from analyst.eval.heldout_harness import (
    heldout_integrity_errors,
    render_heldout_scorecard,
    select_latest_valid_heldout_runs,
    summarize_heldout_runs,
)


@pytest.fixture(scope="module")
def heldout_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("heldout") / "seed_20260717"
    generate_heldout(directory, seed=20260717)
    return directory


def test_generator_is_reproducible_and_keeps_metadata_outside_data(tmp_path):
    first = tmp_path / "first" / "data"
    second = tmp_path / "second" / "data"
    third = tmp_path / "third" / "data"
    generate_heldout(first, seed=11)
    generate_heldout(second, seed=11)
    generate_heldout(third, seed=12)

    assert sorted(path.name for path in first.glob("*.csv")) == [
        "experiment_outcomes.csv",
        "feature_rollout.csv",
        "page_events.csv",
        "support_tickets.csv",
        "survey_responses.csv",
        "users.csv",
    ]
    assert (first.parent / "data.metadata.json").exists()
    assert not list(first.glob("*truth*"))
    assert (first / "users.csv").read_bytes() == (second / "users.csv").read_bytes()
    assert (first / "users.csv").read_bytes() != (third / "users.csv").read_bytes()


def test_benchmark_has_balanced_20_item_contract_and_planted_controls(heldout_dir):
    items = build_heldout_benchmark(heldout_dir)

    assert len(items) == 20
    assert tuple(item.id for item in items) == EXPECTED_HELDOUT_IDS
    assert {
        category: sum(item.category == category for item in items)
        for category in {item.category for item in items}
    } == {
        "estimation": 5,
        "testing_effects": 5,
        "confounding": 5,
        "design_quality": 5,
    }

    team = next(item for item in items if item.id == "hc_team_simpson")
    team_values = {check.id: check.expected for check in team.checks}
    assert team_values["alpha_adjusted"] > team_values["beta_adjusted"]
    rollout = next(item for item in items if item.id == "hc_rollout_simpson")
    rollout_values = {check.id: check.expected for check in rollout.checks}
    assert rollout_values["new_adjusted"] > rollout_values["legacy_adjusted"]
    assert any(item.id == "hc_experiment_sound_control" for item in items)


def test_oracle_answer_satisfies_every_component_rubric(heldout_dir):
    for item in build_heldout_benchmark(heldout_dir):
        parts = []
        for check in item.checks:
            if check.kind == "number":
                parts.append(str(check.expected))
            else:
                parts.append(check.any_of[0])
        scored = score_heldout_item(item, " · ".join(parts), [])
        assert scored["score"] == 1.0, item.id
        assert scored["checks_passed"] == scored["checks_total"]


def test_numeric_evidence_is_field_scoped_to_prevent_false_p_value_credit():
    item = HeldoutItem(
        "x",
        "testing_effects",
        "test",
        "question",
        (
            number_check("p_value", "p-value", 0.0, 0.001),
            text_check("decision", "decision", ["significant"]),
        ),
    )
    evidence = [
        {
            "test": {"p_value": 0.5, "null_value": 0.0, "significant": False},
            "intervals": [{"estimate": 0.0, "lower": 0.0, "upper": 1.0}],
            "effects": [{"value": 0.0}],
        }
    ]

    scored = score_heldout_item(item, "not significant", evidence)

    assert scored["check_results"][0]["passed"] is False


def _fake_run(seed=1, score=0.75, error=None):
    categories = ["estimation", "testing_effects", "confounding", "design_quality"]
    results = []
    for index, item_id in enumerate(EXPECTED_HELDOUT_IDS):
        results.append(
            {
                "id": item_id,
                "category": categories[index // 5],
                "method": "method",
                "score": score,
                "checks_passed": 3,
                "checks_total": 4,
                "check_results": [{"passed": True}],
                "latency_s": 2.0,
                "tokens": 100,
                "error": error,
            }
        )
    return {
        "suite": "heldout-statistical-v1",
        "config": "no-critic",
        "seed": seed,
        "dataset_hash": f"hash-{seed}",
        "results": results,
        "aborted": False,
    }


def test_integrity_gate_rejects_partial_errors_and_wrong_item_set():
    assert heldout_integrity_errors(_fake_run()) == []

    partial = _fake_run()
    partial["results"] = partial["results"][:4]
    assert "expected 20 results, found 4" in heldout_integrity_errors(partial)

    errored = _fake_run(error="cassette miss")
    assert "20 result(s) contain errors" in heldout_integrity_errors(errored)

    wrong_ids = _fake_run()
    wrong_ids["results"][0]["id"] = "unknown"
    assert "result ids do not match the held-out v1 contract" in heldout_integrity_errors(
        wrong_ids
    )


def test_newer_broken_heldout_run_cannot_replace_valid_seed(tmp_path):
    valid = _fake_run()
    broken = _fake_run(error="failure")
    (tmp_path / "heldout_raw_20260101_000000.json").write_text(json.dumps([valid]))
    (tmp_path / "heldout_raw_20260102_000000.json").write_text(json.dumps([broken]))

    selected, sources, rejected = select_latest_valid_heldout_runs(tmp_path)

    assert selected[("no-critic", 1)] == valid
    assert sources[("no-critic", 1)].name == "heldout_raw_20260101_000000.json"
    assert len(rejected) == 1


def test_multi_seed_summary_and_scorecard_report_uncertainty(tmp_path):
    first = _fake_run(seed=1, score=0.75)
    second = _fake_run(seed=2, score=0.85)

    summary = summarize_heldout_runs([first, second])[0]
    markdown = render_heldout_scorecard(
        [first, second],
        sources={
            ("no-critic", 1): tmp_path / "first.json",
            ("no-critic", 2): tmp_path / "second.json",
        },
    )

    assert summary["overall"] == 0.8
    assert summary["seeds"] == 2
    assert summary["ci95_halfwidth"] is not None
    assert "**0.800** ±" in markdown
    assert "seed `1`" in markdown
    assert "one no-reversal sound control" in markdown
