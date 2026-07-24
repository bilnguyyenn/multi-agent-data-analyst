"""Graph state: the single typed object every agent reads and writes.

Two-level design (the core architectural decision, see docs/ARCHITECTURE.md §2):

- The LangGraph control loop is STATIC — plan, schedule, execute, (later: critique,
  validate), finalize. Compiled once, same shape for every question.
- The analysis DAG is DYNAMIC — it is *data* inside this state (`dag`), invented by
  the planner per question. A scheduler node walks it and fans ready tasks out to
  parallel executor instances.

Concurrency note: parallel executor nodes each return a partial state update. LangGraph
merges those updates using the *reducer* attached to each field (see `merge_dicts`).
Executors only ever write keys namespaced by their own task_id, so dict-union is safe —
no locks, no races. That's the interview answer to "how do concurrent agents share state?"
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Inner models (Pydantic: validated, serializable — these cross process
# boundaries later via Postgres checkpoints and WebSocket events)
# ---------------------------------------------------------------------------

TaskKind = Literal["sql", "python", "reuse"]
# "review" = executed cleanly, awaiting the critic's verdict (phase b)
TaskStatus = Literal["pending", "running", "review", "done", "rejected", "failed"]


class TaskSpec(BaseModel):
    """One node of the planner's analysis DAG."""

    id: str
    goal: str  # natural-language sub-task, e.g. "total downtime hours per machine"
    depends_on: list[str] = Field(default_factory=list)
    kind: TaskKind = "python"
    status: TaskStatus = "pending"
    attempts: int = 0    # failed executions (bug loop)
    revisions: int = 0   # critic-ordered rewrites (review loop) — separate budgets
    feedback: Optional[str] = None  # error trace or critic feedback for the retry
    reuse_of: Optional[str] = None  # MemoryHit id, for kind == "reuse" (phase d)


class MemoryHit(BaseModel):
    """A previously-validated sub-result retrieved for a new question."""

    id: str
    goal: str
    result: Any
    run_id: str
    task_id: str
    similarity: float


class ValidatedClaim(BaseModel):
    """Statistical validator's judgment on one report-bound quantitative claim."""

    task_id: str
    claim: str
    verdict: Literal["pass", "caveat", "fail"]
    note: str = ""


class ConfidenceInterval(BaseModel):
    """One deterministic interval attached to a named statistical parameter."""

    parameter: str
    estimate: float
    lower: float
    upper: float
    level: float = Field(default=0.95, gt=0.0, lt=1.0)
    method: str


class EffectSize(BaseModel):
    """Scale-aware effect estimate; interpretation is descriptive, not a verdict."""

    name: str
    value: float
    interpretation: str = ""


class HypothesisTest(BaseModel):
    """Deterministic hypothesis-test result computed from executor sufficient stats."""

    name: str
    statistic: float
    p_value: float = Field(ge=0.0, le=1.0)
    alternative: Literal["two-sided", "less", "greater"] = "two-sided"
    null_value: Optional[float] = None
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    significant: bool


class StatisticalEvidence(BaseModel):
    """Auditable deterministic evidence generated before the LLM validator runs."""

    task_id: str
    request_id: str
    kind: str
    status: Literal["computed", "invalid"]
    sample_size: int = 0
    intervals: list[ConfidenceInterval] = Field(default_factory=list)
    effects: list[EffectSize] = Field(default_factory=list)
    test: Optional[HypothesisTest] = None
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class PredictiveMetric(BaseModel):
    """One out-of-sample model metric with an optional bootstrap interval/baseline."""

    name: str
    value: float
    higher_is_better: bool
    baseline_value: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    confidence: Optional[float] = None


class ModelCandidate(BaseModel):
    """Training-only cross-validation result used for model selection."""

    name: str
    cv_mean: float
    cv_std: float
    folds: int


class FeatureImportance(BaseModel):
    """Holdout permutation importance on an original input feature."""

    feature: str
    importance: float
    std: float


class SliceMetric(BaseModel):
    """Primary holdout metric for one requested subgroup."""

    column: str
    value: str
    n: int
    metric: str
    score: Optional[float] = None
    warning: str = ""


class PredictiveEvidence(BaseModel):
    """Auditable, leakage-aware predictive-model evaluation."""

    task_id: str
    request_id: str
    task_type: str
    status: Literal["computed", "invalid"]
    target: str = ""
    features: list[str] = Field(default_factory=list)
    selected_model: str = ""
    primary_metric: str = ""
    train_size: int = 0
    test_size: int = 0
    split_strategy: str = ""
    metrics: list[PredictiveMetric] = Field(default_factory=list)
    candidates: list[ModelCandidate] = Field(default_factory=list)
    feature_importance: list[FeatureImportance] = Field(default_factory=list)
    slices: list[SliceMetric] = Field(default_factory=list)
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


CouncilRole = Literal["metric", "statistical", "data_quality"]
CouncilMode = Literal["off", "adaptive", "always"]


class CouncilRoutingDecision(BaseModel):
    """Deterministic, auditable decision to spend council tokens for one round."""

    round: int = 0
    mode: CouncilMode
    invoked: bool
    risk_score: int = 0
    threshold: int = 4
    reasons: list[str] = Field(default_factory=list)


class CouncilFinding(BaseModel):
    """One specialist's independent reading of the completed analysis artifacts."""

    specialist: CouncilRole
    round: int = 0
    verdict: Literal["support", "challenge", "insufficient"]
    conclusion: str
    answer_key: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    recommended_action: Literal["accept", "recompute"] = "accept"
    target_task_id: Optional[str] = None
    instruction: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class CouncilDecision(BaseModel):
    """Consensus or adjudication result for one council round."""

    round: int = 0
    status: Literal["consensus", "adjudicated", "recompute", "caveat"]
    conclusion: str
    rationale: str = ""
    dissent: list[str] = Field(default_factory=list)
    action: Literal["accept", "recompute"] = "accept"
    task_id: Optional[str] = None
    instruction: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class CriticVerdict(BaseModel):
    """Structured output of the adversarial critic for one task."""

    task_id: str
    verdict: Literal["approve", "revise", "replan"]
    reason: str = ""
    feedback: str = ""  # concrete instruction for the executor (revise only)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class ExecutionOutcome(BaseModel):
    """What came back from the sandbox for one attempt."""

    status: Literal["ok", "timeout", "oom_or_killed", "error", "contract_violation"]
    stdout: str = ""
    stderr: str = ""
    result: Any = None  # parsed result.json, if status == "ok"
    duration_s: float = 0.0


class Artifact(BaseModel):
    """Everything one task produced: the code, the run, the tokens it cost."""

    task_id: str
    code: str
    outcome: ExecutionOutcome
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class TableInfo(BaseModel):
    name: str
    path: str
    columns: list[str]
    dtypes: dict[str, str]
    row_count: int


QualitySeverity = Literal["error", "warning", "info"]


class QualityIssue(BaseModel):
    """One deterministic, actionable data-quality finding."""

    code: str
    severity: QualitySeverity
    table: str
    column: Optional[str] = None
    message: str
    recommendation: str = ""
    affected_count: int = 0
    affected_rate: float = 0.0


class ValueFrequency(BaseModel):
    """Compact frequency entry used by categorical EDA summaries."""

    value: str
    count: int
    rate: float


class ColumnProfile(BaseModel):
    """Column-level EDA statistics with JSON-safe scalar values."""

    name: str
    dtype: str
    semantic_type: Literal[
        "numeric", "categorical", "text", "datetime", "boolean", "identifier", "unknown"
    ]
    row_count: int
    non_null_count: int
    missing_count: int
    missing_rate: float
    distinct_count: int
    distinct_rate: float
    numeric_summary: dict[str, float] = Field(default_factory=dict)
    datetime_summary: dict[str, str] = Field(default_factory=dict)
    top_values: list[ValueFrequency] = Field(default_factory=list)


class TableProfile(BaseModel):
    """Table-level shape, duplicate checks, keys, and column profiles."""

    name: str
    row_count: int
    column_count: int
    duplicate_rows: int = 0
    duplicate_rate: float = 0.0
    candidate_keys: list[str] = Field(default_factory=list)
    columns: list[ColumnProfile] = Field(default_factory=list)


class RelationshipProfile(BaseModel):
    """Inferred parent-child relationship and referential-integrity result."""

    column: str
    parent_table: str
    child_table: str
    child_non_null_count: int
    orphan_count: int
    orphan_rate: float


class DataQualityReport(BaseModel):
    """Deterministic EDA and data-readiness report for a dataset manifest."""

    quality_score: float = Field(ge=0.0, le=100.0)
    score_method: str
    table_count: int
    total_rows: int
    issue_counts: dict[str, int] = Field(default_factory=dict)
    summary: str
    tables: list[TableProfile] = Field(default_factory=list)
    relationships: list[RelationshipProfile] = Field(default_factory=list)
    issues: list[QualityIssue] = Field(default_factory=list)


class DatasetManifest(BaseModel):
    """What the agents are told about the data. content_hash doubles as the
    staleness guard for the memory layer (phase d): cached sub-results are only
    reusable if the data they were computed on is byte-identical."""

    root: str
    tables: list[TableInfo]
    content_hash: str
    quality_report: Optional[DataQualityReport] = None

    def schema_text(self) -> str:
        lines = []
        for t in self.tables:
            cols = ", ".join(f"{c}:{t.dtypes.get(c, '?')}" for c in t.columns)
            lines.append(f"- {t.name} ({t.row_count} rows): {cols}")
        if self.quality_report is not None:
            report = self.quality_report
            counts = report.issue_counts
            lines.append(
                "DATA QUALITY: "
                f"score {report.quality_score:.1f}/100; "
                f"{counts.get('error', 0)} errors, "
                f"{counts.get('warning', 0)} warnings, "
                f"{counts.get('info', 0)} informational findings"
            )
            for issue in report.issues[:8]:
                location = f"{issue.table}.{issue.column}" if issue.column else issue.table
                lines.append(f"- [{issue.severity}] {location}: {issue.message}")
        return "\n".join(lines)


class Budgets(BaseModel):
    """Termination guarantees for the agent loops (interviewers always ask).
    Exhausting a budget never silently drops an objection — it becomes a caveat."""

    max_task_attempts: int = 2   # execution failures per task (bug loop)
    max_revisions: int = 2       # critic-ordered rewrites per task (review loop)
    replans_left: int = 2        # planner re-invocations per run
    validations_left: int = 2    # validator passes per run (fail -> recompute -> re-validate)
    council_recomputes_left: int = 1  # adjudicator-requested evidence refreshes
    max_steps: int = 40
    steps_used: int = 0


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------

def merge_dicts(left: dict, right: dict) -> dict:
    """Dict-union reducer. Parallel branches write disjoint keys (their own task_id),
    so last-write-wins on a per-key basis is exactly the semantics we want."""
    return {**left, **right}


# ---------------------------------------------------------------------------
# The graph state itself.
#
# TypedDict (not Pydantic) at the top level: LangGraph's reducer machinery via
# Annotated[...] is designed around TypedDict, and partial updates ({"artifacts": {...}})
# are natural. The *values* stay Pydantic for validation.
# ---------------------------------------------------------------------------

class AnalysisState(TypedDict, total=False):
    run_id: str
    question: str
    dataset: DatasetManifest
    dag: Annotated[dict[str, TaskSpec], merge_dicts]
    artifacts: Annotated[dict[str, Artifact], merge_dicts]
    verdicts: Annotated[dict[str, CriticVerdict], merge_dicts]  # latest per task
    claims: Annotated[list[ValidatedClaim], operator.add]  # validator output (phase c)
    statistical_evidence: dict[str, StatisticalEvidence]  # deterministic, replaced per pass
    predictive_evidence: dict[str, PredictiveEvidence]  # deterministic, replaced per pass
    evidence_signature: str  # fingerprint of artifact results used for both evidence maps
    council_findings: Annotated[dict[str, CouncilFinding], merge_dicts]
    council_decisions: Annotated[list[CouncilDecision], operator.add]
    council_round: int
    council_complete: bool
    council_routing: CouncilRoutingDecision
    memory_hits: list[MemoryHit]  # retrieved before planning (phase d)
    budgets: Budgets
    dispatch: list[str]  # scheduler -> router handoff: task ids to fan out this cycle
    replan_request: Optional[dict]  # {"task_ids": [...], "reason": str} critic -> planner
    objections: Annotated[list[str], operator.add]  # unresolved critic objections -> caveats
    answer: str  # phase (a): deterministic synthesis; later: report writer output
    events: Annotated[list[dict], operator.add]  # append-only audit trail
