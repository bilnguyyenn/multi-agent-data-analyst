"""The static LangGraph control loop (phase a: plan -> schedule -> execute* -> finalize).

LangGraph concepts in play (explained properly in docs/build-log.md):

- StateGraph: nodes are functions taking state and returning PARTIAL updates;
  LangGraph merges updates via each field's reducer (see state.py).
- Send API: the scheduler's router returns a list of Send("execute_task", payload)
  objects — one per ready task — and LangGraph runs those node instances in
  PARALLEL within one superstep. This is how a dynamic, per-question DAG executes
  on a static, compiled graph.
- Superstep semantics: all parallel executors finish, their updates merge, and only
  then does the scheduler run again. So the scheduler always sees a consistent
  world — no partially-applied updates, no locks.

Phase (b) will insert the critic between execute and schedule; the loop shape is
built to absorb that without restructuring.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from analyst.agents.council import SPECIALISTS, adjudicate, analyze
from analyst.agents.critic import review
from analyst.agents.executor import run_task
from analyst.agents.planner import PlanError, make_plan, make_replan
from analyst.agents.validator import validate
from analyst.graph.risk import assess_council_risk
from analyst.graph.scheduler import cascade_failures, ready_set
from analyst.graph.state import (
    AnalysisState,
    Budgets,
    CouncilDecision,
    CouncilMode,
    TaskSpec,
)
from analyst.llm.client import LLMClient
from analyst.modeling.engine import evaluate_modeling_artifacts
from analyst.sandbox.runner import SandboxRunner
from analyst.stats.engine import evaluate_artifacts

FEEDBACK_TAIL = 800
CONFOUNDING_TERMS = (
    "confound", "mix", "composition", "simpson", "stratif", "complexity", "material",
)


def _ev(node: str, **kw: Any) -> dict:
    return {"ts": time.time(), "node": node, **kw}


def _descendants_including(dag: dict[str, TaskSpec], root_id: str) -> set[str]:
    affected = {root_id}
    changed = True
    while changed:
        changed = False
        for task in dag.values():
            if task.id not in affected and any(dep in affected for dep in task.depends_on):
                affected.add(task.id)
                changed = True
    return affected


def _validator_retry_feedback(note: str) -> str:
    feedback = (
        "A statistical reviewer rejected the claim from your previous result: "
        f"{note}\nRecompute accordingly."
    )
    if any(term in note.lower() for term in CONFOUNDING_TERMS):
        feedback += (
            "\nThis is a composition/confounding correction. Recompute successes and "
            "the correct denominator within every group × confounder stratum, then "
            "include a statistical_tests request with kind='stratified_proportion', "
            "standardization='pooled', direction='lower' when the event is adverse "
            "('higher' otherwise), and a complete cells grid of "
            "{group, stratum, successes, n}. Keep the crude result in data so the "
            "adjusted conclusion can be compared with it."
        )
    return feedback


def _evidence_signature(artifacts: dict[str, Any]) -> str:
    """Fingerprint the result payloads that deterministic evidence is derived from."""
    payload = {
        task_id: artifact.outcome.result
        for task_id, artifact in sorted(artifacts.items())
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_graph(
    llm: LLMClient,
    sandbox: SandboxRunner,
    exec_timeout_s: int = 60,
    critic_enabled: bool = True,
    validator_enabled: bool = True,
    council_enabled: bool = False,
    council_mode: CouncilMode | None = None,
    memory=None,  # MemoryStore | None — None is the no-memory ablation
):
    """Review components are independent ablation switches.

    Council defaults off at this low-level factory and in the Standard product path.
    Product entry points enable it only for explicit High-assurance runs; historical
    and experimental policies remain available for reproducible ablations.
    """
    resolved_council_mode: CouncilMode = (
        council_mode if council_mode is not None
        else ("always" if council_enabled else "off")
    )

    # ---- nodes ------------------------------------------------------------

    def retrieve_memory(state: AnalysisState) -> dict:
        hits = memory.query(state["question"], state["dataset"].content_hash)
        return {
            "memory_hits": hits,
            "events": [_ev("memory", hits=[(h.id, h.similarity) for h in hits])],
        }

    def update_memory(state: AnalysisState) -> dict:
        stored = memory.store_run(
            run_id=state["run_id"],
            question=state["question"],
            dataset_hash=state["dataset"].content_hash,
            dag=state.get("dag", {}),
            artifacts=state.get("artifacts", {}),
            claims=state.get("claims", []),
        )
        return {"events": [_ev("memory", stored=stored)]}

    def plan(state: AnalysisState) -> dict:
        req = state.get("replan_request")
        if req:  # critic-triggered subtree replan
            rejected = {tid: state["dag"][tid] for tid in req["task_ids"]}
            try:
                updates, usage = make_replan(
                    llm, state["question"], state["dataset"],
                    rejected, req["reason"], known_ids=set(state["dag"]),
                )
            except PlanError as e:
                # Fail-open: keep the original (executed) results, but the
                # objection survives into the final answer as a caveat.
                return {
                    "dag": {
                        tid: t.model_copy(update={"status": "done"})
                        for tid, t in rejected.items()
                    },
                    "replan_request": None,
                    "objections": [
                        f"{tid}: replan failed ({e}); original result kept"
                        for tid in rejected
                    ],
                    "events": [_ev("plan", replan=True, error=str(e))],
                }
            return {
                "dag": updates,
                "replan_request": None,
                "events": [_ev("plan", replan=True, tasks=list(updates), **usage)],
            }

        try:
            dag, usage = make_plan(
                llm, state["question"], state["dataset"],
                memory_hits=state.get("memory_hits"),
            )
        except PlanError as e:
            return {
                "dag": {},
                "answer": f"Planning failed: {e}",
                "events": [_ev("plan", error=str(e))],
            }
        return {
            "dag": dag,
            "budgets": state.get("budgets") or Budgets(),
            "events": [_ev("plan", tasks=list(dag), **usage)],
        }

    def schedule(state: AnalysisState) -> dict:
        """Deterministic, no LLM. Cascades failures, computes the ready set,
        marks it running, enforces the global step budget. Pure logic lives in
        graph/scheduler.py so it unit-tests without LangGraph."""
        dag = state["dag"]
        budgets = state.get("budgets") or Budgets()
        budgets = budgets.model_copy(update={"steps_used": budgets.steps_used + 1})

        updates = cascade_failures(dag)
        merged = {**dag, **updates}
        if budgets.steps_used >= budgets.max_steps:
            dispatch: list[str] = []  # budget exhausted -> drain to finalize
        else:
            dispatch = ready_set(merged)
        for tid in dispatch:
            updates[tid] = merged[tid].model_copy(update={"status": "running"})

        return {
            "dag": updates,
            "budgets": budgets,
            "dispatch": dispatch,
            "events": [_ev("schedule", dispatch=dispatch, step=budgets.steps_used)],
        }

    def route_from_schedule(state: AnalysisState):
        dispatch = state.get("dispatch") or []
        if not dispatch:
            budgets = state.get("budgets") or Budgets()
            done = {tid: t for tid, t in state.get("dag", {}).items() if t.status == "done"}
            if (
                resolved_council_mode != "off"
                and done
                and not state.get("council_complete", False)
            ):
                return "council_gate"
            if (
                validator_enabled
                and budgets.validations_left > 0
                and done
            ):
                return "validate"
            return "finalize"
        dag, artifacts = state["dag"], state.get("artifacts", {})
        hits = {h.id: h for h in state.get("memory_hits") or []}
        return [
            Send(
                "execute_task",
                {
                    "task": dag[tid],
                    "dataset": state["dataset"],
                    "deps": {
                        d: artifacts[d] for d in dag[tid].depends_on if d in artifacts
                    },
                    "memory": hits,
                },
            )
            for tid in dispatch
        ]

    def execute_task(payload: dict) -> dict:
        """Runs as N parallel instances via Send. Input is the Send payload,
        output is a partial state update namespaced by this task's id."""
        task: TaskSpec = payload["task"]

        if task.kind == "reuse":
            # Validated prior result: no LLM, no sandbox, no critic re-review —
            # it already passed both gates on identical data. Provenance is the
            # artifact's model field + the memory hit's run_id.
            hit = payload["memory"][task.reuse_of]
            from analyst.graph.state import Artifact, ExecutionOutcome

            art = Artifact(
                task_id=task.id,
                code=f"# reused from run {hit.run_id} (memory {hit.id})",
                outcome=ExecutionOutcome(status="ok", result=hit.result),
                model="memory",
            )
            return {
                "dag": {task.id: task.model_copy(update={"status": "done"})},
                "artifacts": {task.id: art},
                "events": [_ev("execute", task=task.id, status="reused",
                               source_run=hit.run_id, tokens=0)],
            }

        art = run_task(
            llm, sandbox, task, payload["dataset"], payload["deps"],
            timeout_s=exec_timeout_s,
        )
        if art.outcome.status == "ok":
            # With the critic on, a clean execution is necessary but not sufficient:
            # it goes to "review" and only the critic can promote it to "done".
            new_status = "review" if critic_enabled else "done"
            updated = task.model_copy(update={"status": new_status, "feedback": None})
        else:
            attempts = task.attempts + 1
            if attempts <= (Budgets().max_task_attempts):
                feedback = (
                    f"Previous code:\n{art.code[-FEEDBACK_TAIL:]}\n\n"
                    f"Failure ({art.outcome.status}):\n{art.outcome.stderr[-FEEDBACK_TAIL:]}"
                )
                updated = task.model_copy(
                    update={"status": "pending", "attempts": attempts, "feedback": feedback}
                )
            else:
                updated = task.model_copy(update={"status": "failed", "attempts": attempts})
        return {
            "dag": {task.id: updated},
            "artifacts": {task.id: art},
            "events": [
                _ev(
                    "execute",
                    task=task.id,
                    status=art.outcome.status,
                    duration_s=round(art.outcome.duration_s, 2),
                    model=art.model,
                    tokens=art.input_tokens + art.output_tokens,
                )
            ],
        }

    def critique(state: AnalysisState) -> dict:
        """One critic review per freshly-executed task. Verdicts:
        approve -> done; revise -> back to executor with instructions (budgeted);
        replan -> planner rewrites the task spec (budgeted). Exhausted budgets
        accept the result but carry the objection into the answer as a caveat."""
        dag, artifacts = state["dag"], state["artifacts"]
        budgets = state.get("budgets") or Budgets()
        updates: dict[str, TaskSpec] = {}
        verdicts, events, objections = {}, [], []
        replan_ids: list[str] = []
        replan_reasons: list[str] = []

        for t in dag.values():
            if t.status != "review":
                continue
            v = review(llm, state["question"], t, artifacts[t.id], state["dataset"])
            verdicts[t.id] = v
            events.append(
                _ev("critic", task=t.id, verdict=v.verdict, reason=v.reason[:200],
                    model=v.model, tokens=v.input_tokens + v.output_tokens)
            )
            if v.verdict == "approve":
                updates[t.id] = t.model_copy(update={"status": "done"})
            elif v.verdict == "revise":
                if t.revisions < budgets.max_revisions:
                    fb = (
                        "A reviewer rejected your previous solution.\n"
                        f"Previous code:\n{artifacts[t.id].code[-FEEDBACK_TAIL:]}\n"
                        f"Reviewer's instruction: {v.feedback or v.reason}"
                    )
                    updates[t.id] = t.model_copy(
                        update={"status": "pending", "revisions": t.revisions + 1,
                                "feedback": fb}
                    )
                else:
                    updates[t.id] = t.model_copy(update={"status": "done"})
                    objections.append(f"{t.id}: {v.reason}")
            else:  # replan
                if budgets.replans_left > 0:
                    replan_ids.append(t.id)
                    replan_reasons.append(v.reason)
                    updates[t.id] = t.model_copy(update={"status": "rejected"})
                else:
                    updates[t.id] = t.model_copy(update={"status": "done"})
                    objections.append(f"{t.id}: {v.reason}")

        out: dict = {"dag": updates, "verdicts": verdicts, "events": events,
                     "objections": objections}
        if replan_ids:
            out["budgets"] = budgets.model_copy(
                update={"replans_left": budgets.replans_left - 1}
            )
            out["replan_request"] = {
                "task_ids": replan_ids, "reason": " | ".join(replan_reasons)
            }
        return out

    def route_from_critique(state: AnalysisState):
        return "plan" if state.get("replan_request") else "schedule"

    def council_gate(state: AnalysisState) -> dict:
        round_number = state.get("council_round", 0)
        done = {tid: task for tid, task in state["dag"].items() if task.status == "done"}
        routing = assess_council_risk(
            round_number=round_number,
            mode=resolved_council_mode,
            question=state["question"],
            dag=done,
            artifacts=state.get("artifacts", {}),
            verdicts=state.get("verdicts", {}),
        )
        out = {
            "council_routing": routing,
            "council_complete": not routing.invoked,
            "events": [
                _ev(
                    "council_gate",
                    round=round_number,
                    mode=routing.mode,
                    invoked=routing.invoked,
                    risk_score=routing.risk_score,
                    threshold=routing.threshold,
                    reasons=routing.reasons,
                )
            ],
        }
        if routing.invoked:
            artifacts = state.get("artifacts", {})
            settled_artifacts = {
                task_id: artifacts[task_id]
                for task_id in done
                if task_id in artifacts
            }
            out.update(
                {
                    "statistical_evidence": evaluate_artifacts(settled_artifacts),
                    "predictive_evidence": evaluate_modeling_artifacts(
                        settled_artifacts, state["dataset"]
                    ),
                    "evidence_signature": _evidence_signature(settled_artifacts),
                }
            )
        return out

    def route_from_council_gate(state: AnalysisState):
        routing = state["council_routing"]
        if routing.invoked:
            round_number = state.get("council_round", 0)
            done = {
                tid: task for tid, task in state["dag"].items() if task.status == "done"
            }
            return [
                Send(
                    "council_specialist",
                    {
                        "specialist": specialist,
                        "round": round_number,
                        "question": state["question"],
                        "dataset": state["dataset"],
                        "tasks": done,
                        "artifacts": state.get("artifacts", {}),
                        "statistical_evidence": state.get("statistical_evidence", {}),
                        "predictive_evidence": state.get("predictive_evidence", {}),
                    },
                )
                for specialist in SPECIALISTS
            ]
        if validator_enabled:
            budgets = state.get("budgets") or Budgets()
            if budgets.validations_left > 0:
                return "validate"
        return "finalize"

    def council_specialist(payload: dict) -> dict:
        specialist = payload["specialist"]
        round_number = payload["round"]
        finding = analyze(
            llm=llm,
            specialist=specialist,
            round_number=round_number,
            question=payload["question"],
            dataset=payload["dataset"],
            tasks=payload["tasks"],
            artifacts=payload["artifacts"],
            statistical_evidence=payload.get("statistical_evidence"),
            predictive_evidence=payload.get("predictive_evidence"),
        )
        key = f"{round_number}:{specialist}"
        return {
            "council_findings": {key: finding},
            "events": [
                _ev(
                    "council",
                    specialist=specialist,
                    round=round_number,
                    verdict=finding.verdict,
                    confidence=finding.confidence,
                    action=finding.recommended_action,
                    model=finding.model,
                    tokens=finding.input_tokens + finding.output_tokens,
                )
            ],
        }

    def adjudicate_node(state: AnalysisState) -> dict:
        round_number = state.get("council_round", 0)
        findings = [
            finding
            for key, finding in state.get("council_findings", {}).items()
            if key.startswith(f"{round_number}:")
        ]
        done = {tid: task for tid, task in state["dag"].items() if task.status == "done"}
        decision = adjudicate(
            llm=llm,
            round_number=round_number,
            question=state["question"],
            tasks=done,
            artifacts=state.get("artifacts", {}),
            findings=findings,
            statistical_evidence=state.get("statistical_evidence"),
            predictive_evidence=state.get("predictive_evidence"),
        )
        budgets = state.get("budgets") or Budgets()
        tokens = decision.input_tokens + decision.output_tokens

        if decision.action == "recompute" and budgets.council_recomputes_left > 0:
            affected = _descendants_including(state["dag"], decision.task_id)
            updates: dict[str, TaskSpec] = {}
            for task_id in affected:
                task = state["dag"][task_id]
                feedback = (
                    "The independent analysis council found a material disagreement.\n"
                    f"Adjudicator instruction: {decision.instruction}"
                    if task_id == decision.task_id
                    else f"Recompute because upstream task {decision.task_id} changed."
                )
                updates[task_id] = task.model_copy(
                    update={"status": "pending", "feedback": feedback}
                )
            return {
                "dag": updates,
                "budgets": budgets.model_copy(
                    update={
                        "council_recomputes_left": budgets.council_recomputes_left - 1
                    }
                ),
                "council_decisions": [decision],
                "council_round": round_number + 1,
                "council_complete": False,
                "events": [
                    _ev(
                        "adjudicator",
                        round=round_number,
                        status=decision.status,
                        action="recompute",
                        task=decision.task_id,
                        affected=sorted(affected),
                        model=decision.model,
                        tokens=tokens,
                    )
                ],
            }

        objections: list[str] = []
        if decision.action == "recompute":
            objections.append(
                "analysis council requested recomputation after its budget was exhausted: "
                f"{decision.instruction}"
            )
            decision = decision.model_copy(
                update={
                    "status": "caveat",
                    "action": "accept",
                    "dissent": decision.dissent
                    + ["Requested recomputation could not run: council budget exhausted."],
                }
            )
        return {
            "council_decisions": [decision],
            "council_complete": True,
            "events": [
                _ev(
                    "adjudicator",
                    round=round_number,
                    status=decision.status,
                    action=decision.action,
                    dissent=len(decision.dissent),
                    model=decision.model,
                    tokens=tokens,
                )
            ],
            "objections": objections,
        }

    def route_from_adjudicate(state: AnalysisState):
        if any(task.status == "pending" for task in state["dag"].values()):
            return "schedule"
        if validator_enabled:
            budgets = state.get("budgets") or Budgets()
            if budgets.validations_left > 0:
                return "validate"
        return "finalize"

    def validate_node(state: AnalysisState) -> dict:
        """Runs once all tasks settle. 'fail' claims send their task back through
        the whole loop (execute -> critique -> ...) with the validator's note as
        feedback; 'caveat' claims ride into the answer attached to their numbers.
        At most `validations_left` passes — the last pass publishes claims as-is."""
        dag, artifacts = state["dag"], state.get("artifacts", {})
        budgets = state.get("budgets") or Budgets()
        done = {tid: t for tid, t in dag.items() if t.status == "done"}
        settled_artifacts = {tid: artifacts[tid] for tid in done if tid in artifacts}
        evidence_signature = _evidence_signature(settled_artifacts)
        if state.get("evidence_signature") == evidence_signature:
            statistical_evidence = state.get("statistical_evidence", {})
            predictive_evidence = state.get("predictive_evidence", {})
        else:
            statistical_evidence = evaluate_artifacts(settled_artifacts)
            predictive_evidence = evaluate_modeling_artifacts(
                settled_artifacts, state["dataset"]
            )
        claims, usage = validate(
            llm, state["question"], done, artifacts, state["dataset"],
            statistical_evidence=statistical_evidence,
            predictive_evidence=predictive_evidence,
        )

        budgets = budgets.model_copy(
            update={"validations_left": budgets.validations_left - 1}
        )
        events = [
            _ev(
                "validator",
                claims=[(c.task_id, c.verdict) for c in claims],
                deterministic_tests=sum(
                    item.status == "computed" for item in statistical_evidence.values()
                ),
                invalid_tests=sum(
                    item.status == "invalid" for item in statistical_evidence.values()
                ),
                predictive_models=sum(
                    item.status == "computed" for item in predictive_evidence.values()
                ),
                invalid_models=sum(
                    item.status == "invalid" for item in predictive_evidence.values()
                ),
                passes_left=budgets.validations_left,
                **usage,
            )
        ]

        fails = [c for c in claims if c.verdict == "fail"]
        can_retry = budgets.validations_left > 0
        updates: dict[str, TaskSpec] = {}
        if fails and can_retry:
            for c in fails:
                t = dag[c.task_id]
                updates[t.id] = t.model_copy(
                    update={
                        "status": "pending",
                        "feedback": _validator_retry_feedback(c.note),
                    }
                )
            # don't publish claims from a pass that triggered recomputation
            return {
                "dag": updates,
                "budgets": budgets,
                "statistical_evidence": statistical_evidence,
                "predictive_evidence": predictive_evidence,
                "evidence_signature": evidence_signature,
                "events": events,
            }

        # Final pass: publish everything. Unfixable fails become caveats — honesty
        # over silence, same rule as every other exhausted budget.
        objections = [
            f"{c.task_id} [stats]: {c.note or c.claim}" for c in fails
        ]
        return {
            "claims": claims,
            "statistical_evidence": statistical_evidence,
            "predictive_evidence": predictive_evidence,
            "evidence_signature": evidence_signature,
            "budgets": budgets,
            "events": events,
            "objections": objections,
        }

    def route_from_validate(state: AnalysisState):
        pending = any(t.status == "pending" for t in state["dag"].values())
        return "schedule" if pending else "finalize"

    def finalize(state: AnalysisState) -> dict:
        """Phase (a): deterministic synthesis — no LLM between results and the user.
        The report-writer agent replaces this in a later phase."""
        if state.get("answer"):  # e.g. planning already failed
            return {}
        dag, artifacts = state.get("dag", {}), state.get("artifacts", {})
        depended_on = {d for t in dag.values() for d in t.depends_on}
        lines = []
        for t in dag.values():
            if t.status == "done" and t.id not in depended_on:
                r = artifacts[t.id].outcome.result or {}
                lines.append(f"{r.get('answer', '(no answer field)')}")
                if r.get("data") is not None:
                    lines.append(f"  data: {r['data']}")
        failed = [t.id for t in dag.values() if t.status == "failed"]
        if failed:
            lines.append(f"[!] Unresolved sub-tasks (budget exhausted): {failed}")
        decisions = state.get("council_decisions") or []
        if decisions:
            decision: CouncilDecision = decisions[-1]
            lines.append("")
            lines.append(
                f"[council: {decision.status}] {decision.conclusion}"
            )
            if decision.dissent:
                lines.append("  dissent:")
                lines.extend(f"  - {item}" for item in decision.dissent)
        stat_caveats = [c for c in state.get("claims", []) if c.verdict == "caveat"]
        if stat_caveats:
            lines.append("")
            lines.append("[!] Statistical caveats (validator):")
            lines.extend(f"  - {c.task_id}: {c.note}" for c in stat_caveats)
        statistical_evidence = state.get("statistical_evidence") or {}
        if statistical_evidence:
            lines.append("")
            lines.append("[=] Deterministic statistical evidence:")
            for key, item in statistical_evidence.items():
                lines.append(f"  - {key}: {item.summary}")
                if item.effects:
                    effects = ", ".join(
                        f"{effect.name}={effect.value:.4g}"
                        + (f" ({effect.interpretation})" if effect.interpretation else "")
                        for effect in item.effects
                    )
                    lines.append(f"    effects: {effects}")
                if item.warnings:
                    lines.extend(f"    warning: {warning}" for warning in item.warnings)
        predictive_evidence = state.get("predictive_evidence") or {}
        if predictive_evidence:
            lines.append("")
            lines.append("[>] Predictive modeling evidence:")
            for key, item in predictive_evidence.items():
                lines.append(f"  - {key}: {item.summary}")
                if item.feature_importance:
                    important = ", ".join(
                        f"{feature.feature}={feature.importance:.4g}"
                        for feature in item.feature_importance[:5]
                    )
                    lines.append(f"    permutation importance: {important}")
                if item.warnings:
                    lines.extend(f"    warning: {warning}" for warning in item.warnings)
        objections = state.get("objections") or []
        if objections:
            lines.append("")
            lines.append("[!] Caveats — reviewer objections that could not be resolved:")
            lines.extend(f"  - {o}" for o in objections)
        if not lines:
            lines.append("No sub-task produced a result.")
        return {"answer": "\n".join(lines), "events": [_ev("finalize")]}

    # ---- wiring -----------------------------------------------------------

    from analyst.tracing import traced_node

    g = StateGraph(AnalysisState)
    g.add_node("plan", traced_node("plan")(plan))
    g.add_node("schedule", traced_node("schedule")(schedule))
    g.add_node("execute_task", traced_node("execute")(execute_task))
    g.add_node("finalize", traced_node("finalize")(finalize))

    if memory is not None:
        g.add_node("retrieve_memory", retrieve_memory)
        g.add_node("update_memory", update_memory)
        g.add_edge(START, "retrieve_memory")
        g.add_edge("retrieve_memory", "plan")
        g.add_edge("finalize", "update_memory")
        g.add_edge("update_memory", END)
    else:
        g.add_edge(START, "plan")
        g.add_edge("finalize", END)
    g.add_edge("plan", "schedule")
    schedule_targets = ["execute_task", "finalize"]
    adjudicate_targets = ["schedule", "finalize"]
    council_gate_targets = ["council_specialist", "finalize"]
    if validator_enabled:
        g.add_node("validate", traced_node("validate")(validate_node))
        g.add_conditional_edges("validate", route_from_validate, ["schedule", "finalize"])
        schedule_targets.append("validate")
        adjudicate_targets.append("validate")
        council_gate_targets.append("validate")
    if resolved_council_mode != "off":
        g.add_node("council_gate", traced_node("council_gate")(council_gate))
        g.add_node(
            "council_specialist", traced_node("council_specialist")(council_specialist)
        )
        g.add_node("adjudicate", traced_node("adjudicate")(adjudicate_node))
        g.add_edge("council_specialist", "adjudicate")
        g.add_conditional_edges(
            "adjudicate", route_from_adjudicate, adjudicate_targets
        )
        g.add_conditional_edges(
            "council_gate", route_from_council_gate, council_gate_targets
        )
        schedule_targets.append("council_gate")
    g.add_conditional_edges("schedule", route_from_schedule, schedule_targets)
    if critic_enabled:
        g.add_node("critique", traced_node("critique")(critique))
        g.add_edge("execute_task", "critique")
        g.add_conditional_edges("critique", route_from_critique, ["plan", "schedule"])
    else:
        g.add_edge("execute_task", "schedule")
    return g.compile()
