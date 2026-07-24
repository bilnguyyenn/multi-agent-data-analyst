"""Statistical validator: LLM interpretation over deterministic evidence.

Critic vs validator — why two agents, one sentence each:
- The CRITIC reviews each task's METHOD as it completes (fan-out joins, survivorship).
- The VALIDATOR reviews the run's CLAIMS once everything is computed (sample size,
  uncertainty, multiple comparisons, confounded comparisons, denominators).

The split is testable: the phase-f eval ablates each independently.

Confidence intervals, effect sizes, and hypothesis tests are computed by
``analyst.stats.engine`` from executor-supplied sufficient statistics. The LLM still
judges design validity, assumptions, confounding, denominators, and interpretation.

Like the critic, a malformed validator response fails OPEN (everything passes,
loudly logged) — gates must degrade to the ungated pipeline, not deadlock it.
"""
from __future__ import annotations

import json

from analyst.agents import prompts
from analyst.graph.state import (
    Artifact,
    DatasetManifest,
    PredictiveEvidence,
    StatisticalEvidence,
    TaskSpec,
    ValidatedClaim,
)
from analyst.llm.client import LLMClient
from analyst.llm.parsing import ModelOutputError, extract_json

RESULT_MAX_CHARS = 1500


def validate(
    llm: LLMClient,
    question: str,
    done_tasks: dict[str, TaskSpec],
    artifacts: dict[str, Artifact],
    dataset: DatasetManifest,
    statistical_evidence: dict[str, StatisticalEvidence] | None = None,
    predictive_evidence: dict[str, PredictiveEvidence] | None = None,
) -> tuple[list[ValidatedClaim], dict]:
    results_txt = "\n".join(
        f'- task {tid} (goal: "{t.goal}"):\n'
        f"  {json.dumps(artifacts[tid].outcome.result)[:RESULT_MAX_CHARS]}"
        for tid, t in done_tasks.items()
        if tid in artifacts and artifacts[tid].outcome.result is not None
    )
    evidence_txt = json.dumps(
        [item.model_dump() for item in (statistical_evidence or {}).values()],
        indent=2,
    )
    predictive_txt = json.dumps(
        [item.model_dump() for item in (predictive_evidence or {}).values()],
        indent=2,
    )
    resp = llm.complete(
        role="validator",
        system=prompts.VALIDATOR_SYSTEM,
        user=prompts.VALIDATOR_USER.format(
            question=question,
            schema=dataset.schema_text(),
            results=results_txt,
            statistical_evidence=evidence_txt,
            predictive_evidence=predictive_txt,
        ),
    )
    usage = {
        "model": resp.model,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
    }
    try:
        raw = extract_json(resp.text)
        claims = []
        for c in raw.get("claims", []):
            if c.get("verdict") not in ("pass", "caveat", "fail"):
                continue
            tid = str(c.get("task_id", ""))
            if tid not in done_tasks:
                continue  # validator may not invent tasks
            claims.append(
                ValidatedClaim(
                    task_id=tid,
                    claim=str(c.get("claim", "")),
                    verdict=c["verdict"],
                    note=str(c.get("note", "")),
                )
            )
    except ModelOutputError as e:
        # Fail-open, loudly: one pass-with-warning claim per task.
        claims = [
            ValidatedClaim(
                task_id=tid, claim="(validator output unparseable)",
                verdict="pass", note=f"[validator defaulted to pass: {e}]",
            )
            for tid in done_tasks
        ]
    return claims, usage
