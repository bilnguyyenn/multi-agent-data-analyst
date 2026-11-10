"""Frozen real-data contract checks plus a resumable, explicitly scoped local-agent evaluation."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path

from analyst.operations.metrics import (
    REGISTRY,
    VERSION,
    MetricRequest,
    compute,
    contract_hash,
    connect,
    fingerprint,
    prepare,
    resolve_question,
)
from analyst.operations.reference import reference_values
from analyst.operations.service import file_hash

SUITE = "retail-operations-eval-v1"


def build_contract(data_root: Path) -> dict:
    tasks = []
    # Public temporal splits, not a sealed or model-pretraining-unseen benchmark.
    for split in ["evaluation_a", "evaluation_b"]:
        for metric in REGISTRY:
            for country in [None, "United Kingdom", "France"]:
                for group in [None, "country"]:
                    request = MetricRequest(
                        metric=metric, country=country, group_by=group
                    ).model_dump(mode="json")
                    question = f"What is {metric.replace('_', ' ')}"
                    if country:
                        question += f" for {country}"
                    if group:
                        question += " grouped by country"
                    tasks.append(
                        {
                            "id": f"q{len(tasks) + 1:03d}",
                            "split": split,
                            "question": question + "?",
                            "expected_status": "compute",
                            "request": request,
                        }
                    )
        for question, status in [
            ("What is revenue?", "clarify"),
            ("Which country is best?", "clarify"),
            ("What are sales?", "clarify"),
            ("Which customers should we prioritize?", "clarify"),
            ("What is profit?", "unsupported"),
            ("Forecast sales next year", "unsupported"),
            ("What caused revenue to increase?", "unsupported"),
            ("What is the margin?", "unsupported"),
        ]:
            tasks.append(
                {
                    "id": f"q{len(tasks) + 1:03d}",
                    "split": split,
                    "question": question,
                    "expected_status": status,
                }
            )
    chains = []
    for i in range(20):
        metric = list(REGISTRY)[i % len(REGISTRY)]
        first = {
            "question": f"What is {metric.replace('_', ' ')}?",
            "metric": metric,
            "country": None,
        }
        second = {"question": "And for France?", "metric": metric, "country": "France"}
        if i % 5 == 1:
            first.update(
                question=f"What is {metric.replace('_', ' ')} for France?", country="France"
            )
            second.update(
                question="Actually, use the United Kingdom instead", country="United Kingdom"
            )
        elif i % 5 == 2:
            first.update(
                question=f"What is {metric.replace('_', ' ')} grouped by country?",
                group_by="country",
            )
            second.update(question="Show the overall value instead", country=None)
        elif i % 5 == 3:
            second.update(question="Use net sales instead", metric="net_sales", country=None)
        elif i % 5 == 4:
            second.update(
                question="What caused that value?", expected_status="unsupported", country=None
            )
        chains.append(
            {
                "id": f"chain{i + 1:02d}",
                "split": "evaluation_a" if i < 10 else "evaluation_b",
                "turns": [first, second],
            }
        )
    return {
        "suite": SUITE,
        "metric_version": VERSION,
        "metric_contract_hash": contract_hash(),
        "tasks": tasks,
        "chains": chains,
        "data_hashes": {
            split: file_hash(data_root / split / "transactions.csv")
            for split in ["evaluation_a", "evaluation_b"]
        },
        "scope": "public real-data temporal split; fixed governed requests, not generic SQL/planning accuracy",
    }


def contract_checks(contract: dict, data_root: Path) -> dict:
    results, caches = [], {}
    try:
        for split, expected in contract["data_hashes"].items():
            path = data_root / split / "transactions.csv"
            if file_hash(path) != expected:
                raise ValueError("evaluation source hash mismatch")
            c = connect()
            prepare(c, path)
            caches[split] = c
        for task in contract["tasks"]:
            if task["expected_status"] != "compute":
                resolved = resolve_question(task["question"])["status"]
                expected = (
                    "clarification_required"
                    if task["expected_status"] == "clarify"
                    else "unsupported"
                )
                results.append(
                    {"id": task["id"], "passed": resolved == expected, "kind": "definition_guard"}
                )
                continue
            packet = compute(
                caches[task["split"]],
                MetricRequest.model_validate(task["request"]),
                contract["data_hashes"][task["split"]],
            )
            expected = reference_values(
                data_root / task["split"] / "transactions.csv", task["request"]
            )
            results.append(
                {
                    "id": task["id"],
                    "passed": packet["values"] == expected,
                    "computation_id": packet["computation_id"],
                    "values": packet["values"],
                }
            )
        # Chains test deterministic context inheritance separately from model interpretation.
        for chain in contract["chains"]:
            for index, turn in enumerate(chain["turns"]):
                if turn.get("expected_status") == "unsupported":
                    results.append(
                        {
                            "id": f"{chain['id']}/{index}",
                            "passed": resolve_question(turn["question"])["status"] == "unsupported",
                            "kind": "definition_guard",
                        }
                    )
                    continue
                request = MetricRequest(
                    metric=turn["metric"], country=turn["country"], group_by=turn.get("group_by")
                )
                packet = compute(
                    caches[chain["split"]], request, contract["data_hashes"][chain["split"]]
                )
                expected = reference_values(
                    data_root / chain["split"] / "transactions.csv", request.model_dump(mode="json")
                )
                results.append(
                    {
                        "id": f"{chain['id']}/{index}",
                        "passed": packet["values"] == expected,
                        "computation_id": packet["computation_id"],
                    }
                )
    finally:
        for connection in caches.values():
            connection.close()
    return {
        "scope": "deterministic metric-engine/oracle agreement only; no live-model accuracy",
        "results": results,
        "passed": bool(results) and all(r["passed"] for r in results),
    }


def evaluate_local(
    contract: dict, data_root: Path, output: Path, client, policies: list[str], max_calls: int
) -> dict:
    from analyst.operations.agent import propose

    expected_calls = sum({"single": 1, "standard": 2, "high": 4}[p] for p in policies) * (
        len(contract["tasks"]) + sum(len(c["turns"]) for c in contract["chains"])
    )
    if max_calls < expected_calls:
        raise ValueError(
            f"full evaluation needs a ceiling of {expected_calls} model calls; supplied {max_calls}"
        )
    output.mkdir(parents=True, exist_ok=True)
    contract_hash = fingerprint(contract)
    from analyst.operations import agent

    identity = {
        "contract_hash": contract_hash,
        "policies": policies,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "agent_sha256": hashlib.sha256(Path(agent.__file__).read_bytes()).hexdigest(),
        "model_routing": str(getattr(client, "routing", type(client).__name__)),
    }
    ledger = output / "local-attempts.jsonl"
    prior = (
        [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    )
    if prior and prior[0].get("identity") != identity:
        raise ValueError("resume identity changed; choose a new output directory")
    completed = {(r["policy"], r["id"]): r for r in prior if r.get("type") == "result"}

    def append(event):
        import os

        with ledger.open("a") as f:
            f.write(json.dumps(event, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    if not prior:
        append({"type": "identity", "identity": identity})
    items = list(contract["tasks"])
    for chain in contract["chains"]:
        for i, turn in enumerate(chain["turns"]):
            request = MetricRequest(
                metric=turn["metric"], country=turn["country"], group_by=turn.get("group_by")
            ).model_dump(mode="json")
            items.append(
                {
                    "id": f"{chain['id']}/{i}",
                    "split": chain["split"],
                    "question": turn["question"],
                    "expected_status": turn.get("expected_status", "compute"),
                    "request": request,
                    "prior_id": f"{chain['id']}/{i - 1}" if i else None,
                }
            )
    used_calls = sum(r.get("type") == "call_start" for r in prior)
    required_remaining = sum(
        {"single": 1, "standard": 2, "high": 4}[policy]
        for policy in policies
        for item in items
        if (policy, item["id"]) not in completed
    )
    if used_calls + required_remaining > max_calls:
        raise ValueError(
            "resume would exceed the total model-call ceiling; choose a sufficient explicit ceiling"
        )

    class AuditedClient:
        def complete(self, *, role, system, user):
            nonlocal used_calls
            if used_calls >= max_calls:
                raise ValueError("total model-call ceiling exhausted")
            append({"type": "call_start", "policy": policy, "id": item["id"], "role": role})
            used_calls += 1
            return client.complete(role=role, system=system, user=user)

    for policy in policies:
        for item in items:
            key = (policy, item["id"])
            if key in completed:
                continue
            append({"type": "start", "policy": policy, "id": item["id"], "at": time.time()})
            started = time.monotonic()
            result = {
                "type": "result",
                "policy": policy,
                "id": item["id"],
                "split": item["split"],
                "correct": False,
                "error": None,
                "tokens": 0,
            }
            try:
                old = (
                    completed.get((policy, item.get("prior_id")), {})
                    .get("proposal", {})
                    .get("request")
                )
                proposal = propose(
                    AuditedClient(),
                    item["question"],
                    policy,
                    prior=old,
                    record_call=lambda call: append(
                        {"type": "call", "policy": policy, "id": item["id"], **call}
                    ),
                )
                result.update(
                    proposal={k: v for k, v in proposal.items() if k != "calls"},
                    tokens=proposal["tokens"],
                )
                correct = proposal["status"] == item["expected_status"]
                if correct and proposal["status"] == "compute":
                    wanted = MetricRequest.model_validate(item["request"])
                    actual = MetricRequest.model_validate(proposal["request"])
                    correct = actual == wanted
                    connection = connect()
                    try:
                        path = data_root / item["split"] / "transactions.csv"
                        if file_hash(path) != contract["data_hashes"][item["split"]]:
                            raise ValueError("dataset changed")
                        prepare(connection, path)
                        packet = compute(connection, actual, contract["data_hashes"][item["split"]])
                        result["evidence_id"] = packet["computation_id"]
                        correct = correct and packet["values"] == reference_values(
                            path, item["request"]
                        )
                    finally:
                        connection.close()
                result["correct"] = correct
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            result["latency_s"] = time.monotonic() - started
            append(result)
            completed[key] = result
    recorded_calls = [
        json.loads(line)
        for line in ledger.read_text().splitlines()
        if json.loads(line).get("type") == "call"
    ]
    call_attempts = [
        json.loads(line)
        for line in ledger.read_text().splitlines()
        if json.loads(line).get("type") == "call_start"
    ]
    summary = []
    for policy in policies:
        rows = [completed[(policy, item["id"])] for item in items]
        latencies = sorted(r["latency_s"] for r in rows)
        summary.append(
            {
                "policy": policy,
                "items": len(rows),
                "accuracy": statistics.mean(r["correct"] for r in rows),
                "errors": sum(bool(r["error"]) for r in rows),
                "calls_with_unknown_usage": sum(r["policy"] == policy for r in call_attempts)
                - sum(r["policy"] == policy for r in recorded_calls),
                "tokens": sum(
                    r["input_tokens"] + r["output_tokens"]
                    for r in recorded_calls
                    if r["policy"] == policy
                ),
                "p50_latency_s": statistics.median(latencies),
                "p95_latency_s": latencies[int(0.95 * (len(rows) - 1))],
                "cost_usd": 0,
                "scope": "local governed metric-agent; public temporal split",
            }
        )
    result = {
        "identity": identity,
        "summaries": summary,
        "completed": True,
        "uncertainty": "Only two source-week clusters; point estimates are descriptive, not precision generalization claims.",
    }
    (output / "local-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
