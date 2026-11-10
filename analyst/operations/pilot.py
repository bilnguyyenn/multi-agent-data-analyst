"""Prepare a counterbalanced pilot and score actual, consented observations only."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from analyst.operations.metrics import MetricRequest, fingerprint
from analyst.operations.reference import reference_values
from analyst.operations.service import file_hash


def prepare_pilot(data_root: Path, output: Path) -> dict:
    tasks = []
    for variant, split in [("A", "development"), ("B", "refresh")]:
        for index, (metric, country) in enumerate(
            [
                ("net_sales", "United Kingdom"),
                ("order_count", "France"),
                ("cancellation_rate", None),
            ],
            1,
        ):
            request = MetricRequest(metric=metric, country=country).model_dump(mode="json")
            path = data_root / split / "transactions.csv"
            expected = reference_values(path, request)[0]["value"]
            tasks.append(
                {
                    "id": f"pair{index}{variant}",
                    "variant": variant,
                    "split": split,
                    "question": f"Find {metric.replace('_', ' ')} for {country or 'all countries'} in the {split} CSV. Report rates as fractions rounded to four decimal places, not percentages.",
                    "request": request,
                    "source_hash": file_hash(path),
                    "expected": expected,
                }
            )
    protocol = {
        "version": "retail-pilot-v1",
        "minimum_participants": 3,
        "assignment": "Alternate AB and BA by signup order. AB uses manual for A and analyst for B; BA reverses conditions.",
        "tasks": tasks,
        "privacy": "Use aliases only. Obtain consent. No identities, recordings, or quotes are required.",
    }
    protocol["protocol_hash"] = fingerprint(protocol)
    output.mkdir(parents=True, exist_ok=True)
    empty = {"protocol_hash": protocol["protocol_hash"], "participants": [], "trials": []}
    observations_path = output / "observations.json"
    existing = json.loads(observations_path.read_text()) if observations_path.exists() else empty
    if existing.get("protocol_hash") != protocol["protocol_hash"] and (
        existing.get("participants") or existing.get("trials")
    ):
        raise ValueError(
            "cannot replace a protocol with recorded observations; version it in a new directory"
        )
    (output / "facilitator-protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    if not observations_path.exists() or (
        not existing.get("participants") and not existing.get("trials")
    ):
        (output / "observations.json").write_text(json.dumps(empty, indent=2) + "\n")
    participant = {
        **protocol,
        "tasks": [{k: v for k, v in t.items() if k != "expected"} for t in tasks],
    }
    (output / "participant-tasks.json").write_text(json.dumps(participant, indent=2) + "\n")
    return protocol


def analyze(protocol: dict, observations: dict) -> dict:
    if observations.get("protocol_hash") != protocol["protocol_hash"]:
        raise ValueError("pilot protocol changed")
    participants = observations.get("participants", [])
    ids = [p["id"] for p in participants]
    if len(ids) != len(set(ids)) or any(
        not p.get("consent") or p.get("sequence_group") not in {"AB", "BA"} for p in participants
    ):
        raise ValueError(
            "participants need unique aliases, consent, and a counterbalance assignment"
        )
    tasks = {t["id"]: t for t in protocol["tasks"]}
    assignment = {p["id"]: p["sequence_group"] for p in participants}
    scored, seen = [], set()
    for trial in observations.get("trials", []):
        if trial.get("synthetic"):
            raise ValueError("synthetic trials cannot be included in a real user study")
        key = (trial["participant"], trial["task_id"])
        if key in seen or key[0] not in ids or key[1] not in tasks:
            raise ValueError("duplicate trial, unknown participant, or unknown task")
        seen.add(key)
        task = tasks[key[1]]
        manual_variant = "A" if assignment[key[0]] == "AB" else "B"
        expected_condition = "manual" if task["variant"] == manual_variant else "analyst"
        if trial["condition"] != expected_condition:
            raise ValueError("trial violates counterbalance assignment")
        duration = float(trial["elapsed_seconds"])
        if not 0 < duration < 86400:
            raise ValueError("invalid recorded duration")
        answer = trial.get("answer")
        correct = isinstance(answer, (int, float)) and abs(answer - task["expected"]) <= (
            0.00005 if task["request"]["metric"] == "cancellation_rate" else 0.01
        )
        scored.append({**trial, "correct": correct})
    result = {
        "status": "pending_participants"
        if not ids
        else "complete"
        if len(ids) >= 3 and len(seen) == len(ids) * len(tasks)
        else "incomplete",
        "participants": len(ids),
        "observed_trials": len(scored),
        "expected_trials": len(ids) * len(tasks),
        "conditions": {},
        "limitations": "Small volunteer pilot; task learning, period differences, and self-recorded timings limit inference. No causal time-saving claim.",
    }
    for condition in ["manual", "analyst"]:
        rows = [r for r in scored if r["condition"] == condition]
        successful = [r["elapsed_seconds"] for r in rows if r["correct"]]
        result["conditions"][condition] = {
            "trials": len(rows),
            "correct": sum(r["correct"] for r in rows),
            "median_seconds_correct_only": statistics.median(successful) if successful else None,
            "interventions": sum(int(r.get("interventions", 0)) for r in rows),
        }
    return result
