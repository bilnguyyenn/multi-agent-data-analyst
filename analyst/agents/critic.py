"""Adversarial critic: reviews each clean execution before it counts as done.

Design decisions worth defending:

- The critic reviews CODE + RESULT against GOAL + SCHEMA. It does not recompute —
  recomputation is the executor's job and would double cost. It reasons about the
  *method* (that's where fan-out joins and survivorship bias live).
- It runs on the REASONING tier: finding subtle flaws is the hardest job in the
  system. Approving is easy; catching a confounded comparison is not.
- A malformed critic response defaults to APPROVE with a logged warning, not to
  rejection: a flaky critic must degrade to "phase-a behavior" rather than
  deadlock the pipeline. The eval harness (phase f) measures whether the critic
  actually adds value — that's the honest check on this default.
"""
from __future__ import annotations

import json

from analyst.agents import prompts
from analyst.graph.state import Artifact, CriticVerdict, DatasetManifest, TaskSpec
from analyst.llm.client import LLMClient
from analyst.llm.parsing import ModelOutputError, extract_json

RESULT_MAX_CHARS = 2000


def review(
    llm: LLMClient,
    question: str,
    task: TaskSpec,
    artifact: Artifact,
    dataset: DatasetManifest,
) -> CriticVerdict:
    resp = llm.complete(
        role="critic",
        system=prompts.CRITIC_SYSTEM,
        user=prompts.CRITIC_USER.format(
            question=question,
            goal=task.goal,
            schema=dataset.schema_text(),
            code=artifact.code,
            result=json.dumps(artifact.outcome.result)[:RESULT_MAX_CHARS],
        ),
    )
    base = dict(
        task_id=task.id,
        model=resp.model,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )
    try:
        raw = extract_json(resp.text)
        verdict = raw.get("verdict")
        if verdict not in ("approve", "revise", "replan"):
            raise ModelOutputError(f"bad verdict {verdict!r}")
    except ModelOutputError as e:
        # Fail-open, loudly: a broken critic must not deadlock the run.
        return CriticVerdict(
            verdict="approve",
            reason=f"[critic output unparseable, defaulted to approve: {e}]",
            **base,
        )
    return CriticVerdict(
        verdict=verdict,
        reason=str(raw.get("reason", "")),
        feedback=str(raw.get("feedback", "")),
        **base,
    )
