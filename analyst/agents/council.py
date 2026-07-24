"""Independent analysis council and evidence adjudication.

The three specialists run as separate LangGraph ``Send`` branches. This module is
deliberately stateless: it turns the same immutable evidence packet into structured
findings, then adjudicates only when deterministic agreement checks find conflict.
"""
from __future__ import annotations

import json
import re

from analyst.agents import prompts
from analyst.graph.state import (
    Artifact,
    CouncilDecision,
    CouncilFinding,
    CouncilRole,
    DatasetManifest,
    PredictiveEvidence,
    StatisticalEvidence,
    TaskSpec,
)
from analyst.llm.client import LLMClient
from analyst.llm.parsing import ModelOutputError, extract_json

SPECIALISTS: tuple[CouncilRole, ...] = ("metric", "statistical", "data_quality")

_SPECIALIST_DETAILS: dict[CouncilRole, tuple[str, str]] = {
    "metric": (
        "METRIC ANALYST",
        "Check whether the result directly answers the business question, uses the "
        "right metric and denominator, and interprets rankings or comparisons correctly.",
    ),
    "statistical": (
        "STATISTICAL ANALYST",
        "Check uncertainty, sample size, confounding, multiple comparisons, effect "
        "size, and whether a causal or comparative claim is stronger than the evidence.",
    ),
    "data_quality": (
        "DATA-QUALITY ANALYST",
        "Check joins and grain, duplicates, missingness, type coercion, coverage, "
        "survivorship, and whether the code used the supplied tables safely.",
    ),
}

ARTIFACT_RESULT_CHARS = 1800
ARTIFACT_CODE_CHARS = 1400
DETERMINISTIC_EVIDENCE_CHARS = 12000


def _canonical_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:100]


def evidence_text(
    tasks: dict[str, TaskSpec],
    artifacts: dict[str, Artifact],
    statistical_evidence: dict[str, StatisticalEvidence] | None = None,
    predictive_evidence: dict[str, PredictiveEvidence] | None = None,
) -> str:
    packets = []
    for task_id, task in tasks.items():
        artifact = artifacts.get(task_id)
        if artifact is None or artifact.outcome.result is None:
            continue
        packets.append(
            f'- task_id={task_id}; goal="{task.goal}"\n'
            f"  result={json.dumps(artifact.outcome.result, default=str)[:ARTIFACT_RESULT_CHARS]}\n"
            f"  executed_code:\n{artifact.code[-ARTIFACT_CODE_CHARS:]}"
        )
    sections = ["\n\n".join(packets) or "(no completed artifacts)"]
    if statistical_evidence:
        payload = {
            key: item.model_dump(mode="json")
            for key, item in statistical_evidence.items()
        }
        sections.append(
            "Deterministic statistical evidence:\n"
            + json.dumps(payload, default=str, indent=2)[:DETERMINISTIC_EVIDENCE_CHARS]
        )
    if predictive_evidence:
        payload = {
            key: item.model_dump(mode="json")
            for key, item in predictive_evidence.items()
        }
        sections.append(
            "Deterministic predictive evidence:\n"
            + json.dumps(payload, default=str, indent=2)[:DETERMINISTIC_EVIDENCE_CHARS]
        )
    return "\n\n".join(sections)


def analyze(
    llm: LLMClient,
    specialist: CouncilRole,
    round_number: int,
    question: str,
    dataset: DatasetManifest,
    tasks: dict[str, TaskSpec],
    artifacts: dict[str, Artifact],
    statistical_evidence: dict[str, StatisticalEvidence] | None = None,
    predictive_evidence: dict[str, PredictiveEvidence] | None = None,
) -> CouncilFinding:
    name, mandate = _SPECIALIST_DETAILS[specialist]
    response = llm.complete(
        role=f"council_{specialist}",
        system=prompts.COUNCIL_SYSTEM.format(
            specialist_name=name, specialist_mandate=mandate
        ),
        user=prompts.COUNCIL_USER.format(
            question=question,
            schema=dataset.schema_text(),
            artifacts=evidence_text(
                tasks,
                artifacts,
                statistical_evidence=statistical_evidence,
                predictive_evidence=predictive_evidence,
            ),
        ),
    )
    base = {
        "specialist": specialist,
        "round": round_number,
        "model": response.model,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
    }
    try:
        raw = extract_json(response.text)
        verdict = raw.get("verdict")
        action = raw.get("recommended_action", "accept")
        if verdict not in ("support", "challenge", "insufficient"):
            raise ModelOutputError(f"bad council verdict {verdict!r}")
        if action not in ("accept", "recompute"):
            raise ModelOutputError(f"bad council action {action!r}")
        valid_ids = set(tasks)
        refs = [str(ref) for ref in raw.get("evidence_refs", []) if str(ref) in valid_ids]
        target = raw.get("target_task_id")
        target = str(target) if target is not None and str(target) in valid_ids else None
        instruction = str(raw.get("instruction", ""))
        if action == "recompute" and (target is None or not instruction.strip()):
            action = "accept"
        conclusion = str(raw.get("conclusion", "")).strip()
        return CouncilFinding(
            verdict=verdict,
            conclusion=conclusion or "No conclusion supplied.",
            answer_key=_canonical_key(str(raw.get("answer_key", "")) or conclusion),
            evidence_refs=refs,
            risks=[str(risk) for risk in raw.get("risks", [])][:8],
            confidence=float(raw.get("confidence", 0.5)),
            recommended_action=action,
            target_task_id=target,
            instruction=instruction if action == "recompute" else "",
            **base,
        )
    except (ModelOutputError, TypeError, ValueError) as exc:
        # A broken specialist is visible as insufficient evidence, but cannot
        # deadlock the run or force arbitrary recomputation.
        return CouncilFinding(
            verdict="insufficient",
            conclusion="Specialist output could not be interpreted.",
            risks=[f"{specialist} output unparseable: {exc}"],
            confidence=0.0,
            **base,
        )


def requires_adjudication(findings: list[CouncilFinding]) -> bool:
    if len(findings) != len(SPECIALISTS):
        return True
    if any(f.verdict != "support" for f in findings):
        return True
    keys = {f.answer_key for f in findings if f.answer_key}
    return len(keys) != 1


def adjudicate(
    llm: LLMClient,
    round_number: int,
    question: str,
    tasks: dict[str, TaskSpec],
    artifacts: dict[str, Artifact],
    findings: list[CouncilFinding],
    statistical_evidence: dict[str, StatisticalEvidence] | None = None,
    predictive_evidence: dict[str, PredictiveEvidence] | None = None,
) -> CouncilDecision:
    if not requires_adjudication(findings):
        metric = next((f for f in findings if f.specialist == "metric"), findings[0])
        return CouncilDecision(
            round=round_number,
            status="consensus",
            conclusion=metric.conclusion,
            rationale="All three independent specialists supported the same answer key.",
        )

    response = llm.complete(
        role="adjudicator",
        system=prompts.ADJUDICATOR_SYSTEM,
        user=prompts.ADJUDICATOR_USER.format(
            question=question,
            artifacts=evidence_text(
                tasks,
                artifacts,
                statistical_evidence=statistical_evidence,
                predictive_evidence=predictive_evidence,
            ),
            findings=json.dumps([f.model_dump() for f in findings], default=str, indent=2),
        ),
    )
    base = {
        "round": round_number,
        "model": response.model,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
    }
    try:
        raw = extract_json(response.text)
        action = raw.get("action")
        if action not in ("accept", "recompute"):
            raise ModelOutputError(f"bad adjudicator action {action!r}")
        task_id = raw.get("task_id")
        task_id = str(task_id) if task_id is not None and str(task_id) in tasks else None
        instruction = str(raw.get("instruction", "")).strip()
        if action == "recompute" and (task_id is None or not instruction):
            raise ModelOutputError("recompute requires a valid task_id and instruction")
        return CouncilDecision(
            status="recompute" if action == "recompute" else "adjudicated",
            conclusion=str(raw.get("conclusion", "")).strip() or "No conclusion supplied.",
            rationale=str(raw.get("rationale", "")),
            dissent=[str(item) for item in raw.get("dissent", [])][:8],
            action=action,
            task_id=task_id,
            instruction=instruction if action == "recompute" else "",
            **base,
        )
    except (ModelOutputError, TypeError, ValueError) as exc:
        dissent = [
            f"{finding.specialist}: {finding.conclusion}"
            for finding in findings
            if finding.verdict != "support"
        ]
        return CouncilDecision(
            status="caveat",
            conclusion="The council disagreed and adjudication was inconclusive.",
            rationale=f"Adjudicator output unparseable: {exc}",
            dissent=dissent,
            **base,
        )
