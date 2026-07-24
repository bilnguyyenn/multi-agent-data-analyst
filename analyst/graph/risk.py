"""Deterministic risk routing for the optional analysis council.

The router spends no model tokens and explains every activation. Signals are broad
analysis-risk categories, not benchmark ids or dataset-specific column names.
"""
from __future__ import annotations

import re

from analyst.graph.state import (
    Artifact,
    CouncilMode,
    CouncilRoutingDecision,
    CriticVerdict,
    TaskSpec,
)

COUNCIL_RISK_THRESHOLD = 4

INFERENCE_TERMS = (
    "significant", "meaningful", "confidence", "uncertain", "reliability",
    "confident", "reliable", "noise", "distribution", "outlier", "statistical",
    "interval", "typical", "mean a good summary",
)
TEMPORAL_CAUSAL_TERMS = (
    "improved", "improve", "changed", "change over time", "trend", "since",
    "before", "after", "early", "effect", "impact", "reduce", "caused",
)
COMPARISON_TERMS = (
    "best", "worst", "better", "different", "compare", "versus", " vs ",
    "most reliable", "quality record", "highest", "lowest",
)
GROUP_TERMS = (
    "shift", "supplier", "operator", "product", "material", "fleet", "machine",
    "group", "segment", "region", "customer", "cohort",
)
RATE_TERMS = (
    "rate", "ratio", "fraction", "percentage", "percent", " per ", "average",
)
POPULATION_TERMS = (
    "current", "active", "fleet", "surviv", "retired", "decommission",
    "population", "cohort",
)


def _contains(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def assess_council_risk(
    *,
    round_number: int,
    mode: CouncilMode,
    question: str,
    dag: dict[str, TaskSpec],
    artifacts: dict[str, Artifact],
    verdicts: dict[str, CriticVerdict] | None = None,
    threshold: int = COUNCIL_RISK_THRESHOLD,
) -> CouncilRoutingDecision:
    text = " ".join([question, *(task.goal for task in dag.values())]).lower()
    code = "\n".join(artifact.code.lower() for artifact in artifacts.values())
    reasons: list[str] = []
    score = 0

    if _contains(text, INFERENCE_TERMS):
        score += 4
        reasons.append("statistical or uncertainty-sensitive claim")
    if _contains(text, TEMPORAL_CAUSAL_TERMS):
        score += 4
        reasons.append("temporal or causal comparison")
    if _contains(text, COMPARISON_TERMS) and _contains(text, GROUP_TERMS):
        score += 4
        reasons.append("group ranking may hide composition differences")
    if _contains(text, RATE_TERMS):
        score += 1
        reasons.append("rate, ratio, or denominator-sensitive metric")
    if _contains(text, TEMPORAL_CAUSAL_TERMS) and _contains(text, POPULATION_TERMS):
        score += 2
        reasons.append("population drift or survivorship exposure")

    csv_files = set(re.findall(r"[\"']([^\"']+\.csv)[\"']", code))
    join_receivers = re.findall(r"\b([a-zA-Z_]\w*)\.join\s*\(", code)
    has_dataframe_join = bool(
        re.search(r"(?:\.|\bpd\.)merge\s*\(", code)
        or any(receiver not in {"path", "str"} for receiver in join_receivers)
    )
    has_sql_join = bool(
        re.search(r"\b(?:inner\s+|left\s+|right\s+|full\s+|cross\s+)?join\s+", code)
    )
    has_join = bool(
        has_dataframe_join
        or has_sql_join
        or re.search(r"\bjoin(?:ed|ing)?\b", text)
    )
    if has_join:
        score += 2
        reasons.append("executed analysis joins tables")
    if len(csv_files) > 1:
        score += 1
        reasons.append("executed analysis reads multiple tables")
    if has_join and _contains(text, RATE_TERMS):
        score += 1
        reasons.append("join plus rate creates fan-out/denominator risk")
    if len(dag) > 1 or any(task.depends_on for task in dag.values()):
        score += 1
        reasons.append("multi-step evidence chain")
    if any(v.verdict != "approve" for v in (verdicts or {}).values()):
        score += 2
        reasons.append("critic previously rejected or reframed a task")

    if mode == "off":
        return CouncilRoutingDecision(
            round=round_number,
            mode=mode,
            invoked=False,
            risk_score=score,
            threshold=threshold,
            reasons=["council disabled", *reasons],
        )
    if mode == "always":
        return CouncilRoutingDecision(
            round=round_number,
            mode=mode,
            invoked=True,
            risk_score=score,
            threshold=threshold,
            reasons=["council forced on", *reasons],
        )
    return CouncilRoutingDecision(
        round=round_number,
        mode=mode,
        invoked=score >= threshold,
        risk_score=score,
        threshold=threshold,
        reasons=reasons or ["no elevated analysis-risk signals"],
    )
