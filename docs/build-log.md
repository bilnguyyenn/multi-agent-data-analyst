# Build Log

Per-phase notes: what got built, the concepts behind it, and the design decisions
worth being able to defend. Written for future-me and for interview prep.

---

## Phase (a) — planner + executor, end to end · 2026-07-04

**Shipped:** LangGraph control loop (plan → schedule → parallel execute → finalize),
Docker + subprocess sandbox backends, 3-backend LLM client with cassette record/replay,
synthetic manufacturing dataset with 4 planted traps, CLI, 14 tests (all green, $0).

### LangGraph, from zero

A LangGraph `StateGraph` is a state machine where:

- **Nodes are plain functions** `state -> partial_update`. You never mutate state;
  you return only the keys you changed. LangGraph merges the update in.
- **Reducers** (the `Annotated[dict, merge_dicts]` in `graph/state.py`) define *how*
  merging happens per field. Default is overwrite; we use dict-union for
  `dag`/`artifacts` and list-append for `events`. This is the whole answer to
  "how do concurrent agents write shared state without races": parallel executors
  return updates namespaced by their own `task_id`, dict-union merges them, done.
- **Conditional edges** are router functions. Ours (`route_from_schedule`) returns
  either `"finalize"` or a list of `Send` objects.
- **`Send`** is the fan-out primitive: `Send("execute_task", payload)` spawns one
  instance of a node with a custom payload instead of the whole state. N ready
  tasks → N parallel executor instances → this is how a *dynamic* per-question DAG
  runs on a *static* compiled graph. This two-level design is the core
  architecture decision (ARCHITECTURE.md §2.1).
- **Superstep semantics:** all parallel node instances finish and merge before the
  next node runs. The scheduler therefore always sees a consistent world — no
  partially-applied updates, no locks. (Cost: the cycle waits for the slowest
  executor in each wave. Fine here; worth mentioning as a known tradeoff.)
- **`recursion_limit`** caps supersteps as a backstop; our real termination
  guarantees are the explicit budgets (per-task retries, global step cap) because
  "the framework killed my run" is not a defensible failure mode.

### Decisions made this phase (and why)

1. **Scheduler is deterministic code, not an agent.** Computing a DAG ready-set is
   graph theory, not judgment. LLM where judgment lives, code where it doesn't —
   also makes it unit-testable (`tests/test_scheduler.py`) without any model.
2. **Plans are validated structurally before anything executes.** A 7B model will
   eventually emit cycles or dangling deps. `graphlib.TopologicalSorter` catches it
   at plan time — a malformed plan is a planner failure, not an executor mystery.
   (Bug found by test: `static_order()` is lazy and only raises `CycleError` when
   consumed. The test suite caught it; that's what it's for.)
3. **Failure taxonomy routes differently.** timeout / oom / error / contract_violation
   go back to the *executor* with the traceback (bug-fix loop, max 2 retries);
   only clean results will reach the *critic* (review loop, phase b). Different
   failure classes, different loops.
4. **Budget exhaustion degrades to honesty:** a task that keeps failing is reported
   as unresolved in the answer, never silently dropped
   (`test_budget_exhaustion_reports_failure`).
5. **Script-mode vs hash-mode cassettes.** Hash-mode (key = sha256 of prompt) replays
   recorded real runs exactly — any prompt drift is a loud miss. Script-mode (ordered
   responses per role) is for handwritten tests, so a one-word prompt-template tweak
   doesn't invalidate every test. Two modes because they answer different questions:
   "does the system still behave as recorded?" vs "does the machinery work?"
6. **Ground truth lives outside the dataset dir** — that dir is mounted into the
   sandbox, and an agent that can read the answer key isn't being evaluated.
7. **`DATA_DIR` env var is the one contract** both sandbox backends honor, so
   generated code is identical across Docker and dev backends.

### The planted traps (verified in generated data, seed 7)

| Trap | In the data | Naive answer | Truth |
|---|---|---|---|
| Job-mix confounder | night shift: 70% low-complexity jobs | night defect rate 1.39% = "best" | night is *worst* within every complexity class |
| Survivorship | M03/M08/M11 decommissioned mid-2025 | active-only fleet "improved" | the bad machines just left the denominator |
| Fan-out join | defects: 1 row per (wo, defect_type) | join then sum(qty) double-counts | aggregate defects per wo *before* joining |
| Simpson's paradox | S1 ships 80% composite | S1 worst overall (5.2%) | S1 best within *both* materials |

### Running it on your machine (the real thing, not replay)

```bash
pip install -e ".[dev]"
colima start --memory 2 --cpu 2        # 2GB VM: sandbox containers are capped below that
docker build -t analyst-sandbox:latest -f docker/sandbox.Dockerfile .
ollama pull qwen2.5-coder:3b-instruct-q4_K_M     # ~2.2 GB loaded
python -m analyst gen-data --out data/sample
python -m analyst ask "Which machine had the most downtime hours?" \
    --data data/sample --provider local --sandbox docker
# add ANALYST_RECORD_CASSETTES=1 to save the run as replayable cassettes
```

**8GB RAM budget (learned the hard way — the 7B froze the machine):**
macOS ~3.5GB + Colima VM 2GB + 3B model ~2.2GB ≈ full. The 7B (~5GB loaded)
doesn't fit alongside Docker; `num_ctx` is pinned to 4096 in the Ollama client
because an unbounded KV cache can out-eat the model weights. Close the browser
during live runs. If the 3B's output quality becomes the bottleneck, the escape
hatch is a hosted free-tier OpenAI-compatible endpoint behind the same client
interface — the architecture doesn't change.

Dev loop without Docker (not a security boundary — see subprocess_backend.py):
`--sandbox subprocess` after `export ANALYST_SANDBOX=subprocess`.

Tests: `python -m pytest tests/ -q` — 14 tests, no model, no Docker, no network.

### Live-run hardening log (first runs on the real Mac, 2026-07-05)

Four consecutive failures, each at a different layer — none catchable by replay tests:

| # | Symptom | Layer | Root cause → fix |
|---|---|---|---|
| 1 | froze the machine | capacity | 7B model + 3GB Colima VM > 8GB RAM → 3B model, 2GB VM, `num_ctx=4096` |
| 2 | `can't find '__main__' module` in 0.4s | VM file sharing | Docker daemon lives in a Linux VM; macOS `/var/folders` tempdirs aren't shared, Docker silently mounts an empty dir → exchange dir moved under `~` (`~/.cache/analyst-exec`) |
| 3 | `NameError: pd` ×3, ignored feedback | model quality | 3B forgets imports → environment preamble injects them; failure class now impossible (`test_executor_preamble.py`) |
| 4 | `fetch_df` vs `fetchdf` ×3 | API hallucination | compat shim aliases the hallucinated name; prompt now shows a concrete worked duckdb example (small models follow examples, not rules) |
| 5 | t3 read `/data/t2.json` — a file that never existed | interface assumption | model assumed file-based dependency handoff; instead of fighting it, made it TRUE: dep results are now materialized read-only at `$DEPS_DIR/<task_id>.json` (better design anyway — file handoff scales past prompt limits; e2e test now exercises it) |

Meta-lesson for the write-up: the replay test suite validates *logic*; only live runs
validate *infrastructure*. Both are needed, and they catch disjoint failure classes.
Also visible in run #4: t1 succeeded while t2/t3 cascade-failed — the system burns zero
tokens on tasks whose inputs don't exist.

---

## Phase (b) — adversarial critic + revise/replan loop · 2026-07-05

**Shipped:** critic agent (reasoning tier) reviewing every clean execution against a
7-item failure-class checklist; three verdicts wired into the graph
(approve → done; revise → executor retry with the reviewer's instruction, budgeted;
replan → planner rewrites the task spec in place, budgeted); `--no-critic` ablation
flag (this IS the eval switch for phase f); 5 new tests — 24 total, all $0.

Design decisions:

1. **Two loops, two budgets.** Execution failures (bug loop, `attempts`) and critic
   rejections (review loop, `revisions`) are tracked separately — a task that crashed
   twice hasn't used up its right to be *reviewed*.
2. **The critic reviews method, not arithmetic.** It sees goal + code + result +
   schema and reasons about the approach (fan-out joins, survivorship, confounding).
   Recomputing would double cost and duplicate the executor.
3. **Replan is bounded: ids are pinned.** The planner rewrites *what rejected tasks
   do*, not the DAG's shape — dependents' edges stay valid, no cascade invalidation.
   Full-DAG restructuring: costs more than it buys at 1–5 tasks.
4. **Every exhausted budget becomes a caveat**, never silence: "done with unresolved
   objection" flows into the answer's Caveats section verbatim.
5. **A broken critic fails OPEN (approve + logged warning), not closed.** A flaky
   reviewer must degrade to phase-(a) behavior, not deadlock the pipeline. Honest
   check: the phase-f eval measures whether the critic adds value at all.

**First live trap exposure (2026-07-05, Groq 70B critic + 8B executor), question:
"Which shift has the best quality record?"** — the critic caught the planted JOIN
FAN-OUT on its first try (named the exact merge and failure class) and ordered
revisions; the 8B executor couldn't fix it within budget; the system shipped NO
confident wrong answer and carried the objection into the caveats. Recorded gaps
(eval targets, not embarrassments): (1) the job-mix confounder went unnamed — the
naive task framing deserved a `replan`, got `revise`; (2) the 8B repeatedly failed
a groupby it should handle. Trap-question success criterion validated: flag and
hedge beats confidently wrong.

---

## Phase (c) — statistical validator · 2026-07-05

**Shipped:** validator agent (reasoning tier) gating report-bound claims after all
tasks settle; verdicts pass / caveat / fail; `fail` sends the task back through the
whole loop with the statistical note as feedback (budgeted: 2 validation passes);
`caveat` rides into the answer attached to its numbers; `--no-validator` ablation
flag; 5 new tests — 29 total, $0.

Critic vs validator, the one-line answer for interviews: **the critic reviews each
task's method as it completes; the validator reviews the run's claims once
everything is computed** (sample size, uncertainty on comparisons, multiple
comparisons, confounded mix, denominators, overprecision). The split is testable —
each gate is ablated independently in phase (f).

Also this phase, driven by live findings:
- Critic prompt now explicitly treats an unaddressed mix/composition difference as
  a TASK-FRAMING flaw → `replan`, not `revise` (the shift-quality run got this
  half-right: caught fan-out, missed the confounder).
- Hosted mechanical tier flipped 8B → 70B after the 8B hallucinated columns and
  tables across runs. Both are free; the 8B stays in config as the cost-optimized
  variant so the eval can answer "how small can the executor be?" with data.
- v1 validator scope, stated honestly: it *reasons* about statistical soundness;
  it does not yet recompute intervals in the sandbox. That upgrade slots into the
  same interface.

---

## Phase (d) — vector memory layer (Chroma) · 2026-07-05

**Shipped:** `MemoryStore` over a persistent local Chroma collection; retrieval node
before planning; planner can emit `kind="reuse"` tasks that skip LLM + sandbox +
critic entirely (0 tokens, provenance cites the source run); memory write after
finalize; `--no-memory` ablation; 5 new tests (deterministic fake embedder — CI
never downloads models) — 34 total, $0.

The four rules that make memory safe (each is an interview answer):

1. **Write only VALIDATED results** — a sub-result enters memory only if its claim
   passed the statistical validator. A memory of mistakes is worse than no memory.
2. **Staleness guard = dataset content hash.** Queries filter on the exact bytes of
   the data. Change one row → the memory for that dataset is (correctly) invisible.
   Tested: same question + different hash = zero hits.
3. **Chroma stores pointers + small payloads, not truth.** run_id/task_id metadata
   point back at the run ledger; results >4KB aren't stored — memory holds answers,
   not datasets.
4. **Hallucination-proof reuse:** a `reuse_of` id the planner invented (not among
   retrieved hits) silently downgrades to recompute. Trust the memory, not the
   model's memory of the memory.

Also: live prior-run results — both remaining traps (shift confounder, Simpson's
supplier) now get their exact confounder NAMED by the validator in caveats, though
the naive headline number still ships. "Flagged-not-fixed" is the recorded baseline
the eval will quantify; forcing the mix-adjusted recompute is the improvement target.

---

## Phase (e) — FastAPI + WebSocket + React live DAG · 2026-07-05

**Shipped:** async FastAPI (`POST /runs` → 202 + run_id, `GET /runs[/{id}]`,
`WS /runs/{id}/stream`); event bus with Redis pub/sub (prod) and in-process asyncio
(dev/CI) backends; arq worker for the queue path; JSON-file RunStore (Postgres swap
is phase g, interface already fixed); React+TS frontend (Vite + React Flow): live
DAG colored by status — purple = awaiting critic, orange = replanned — with agent
activity log, answer + caveat panel, run history with replay. 3 API tests; 37 total.
Frontend compiles under strict TS.

Concepts introduced this phase (the FastAPI/WS/React learning goals):

1. **The API process never runs agent graphs.** It enqueues and streams. A
   multi-minute run inside an HTTP handler is a timeout waiting to happen; 202 +
   run_id + WebSocket is the pattern.
2. **Replay-then-live WebSocket contract.** On connect the client gets persisted
   history first, then live events. Late joiners and reconnects see the whole
   story. Streaming is UX; the store is truth. (A test initially assumed live-only
   events and failed when the dev run finished before the socket connected — the
   protocol was right, the test was wrong. Instructive race.)
3. **`stream_mode="values"`** — LangGraph yields the full merged state per
   superstep, so every UI update is a consistent snapshot, not a diff to reassemble.
4. **Same event protocol on both execution paths** (dev thread & arq worker) via a
   shared `execute_run` driver — the UI cannot tell the difference, by construction.
5. **Dev-inprocess mode** (`ANALYST_DEV_INPROCESS=1`): the entire system in one
   process with zero infrastructure. This is also how CI tests the API.

Also: the fake embedder caught a real retrieval flaw — memory documents led with
goal + JSON, but queries are *questions*; question-led documents fixed retrieval
(test-driven design of a vector-memory schema, worth retelling).

---

## Phases (f)(g)(h) — eval harness, compose, CI · 2026-07-05

**(f) Eval:** 25-question benchmark, gold values computed from the CSVs at eval
time (can't drift); trap scoring 1.0/0.5/0.0 (correct / naive-but-flagged /
confidently wrong); ablation configs full|no-critic|no-validator|bare; critic-bench
(10 specimens: 6 flawed reproducing the planted trap classes + 4 sound controls →
precision/recall, with the controls punishing a reject-everything critic);
`analyst eval` emits `eval_results/SCORECARD.md`. Memory is measured separately
(repeat-question cost) to avoid confounding accuracy comparisons with cache hits.

**(g) Compose:** redis, postgres (JSONB run ledger via `PostgresRunStore`, selected
by `ANALYST_DB_URL` — same interface as the JSON store), jaeger (OTel span per
graph node, on when `OTEL_EXPORTER_OTLP_ENDPOINT` set), api, worker (docker-socket
mount for sibling sandbox containers — root-equivalent tradeoff documented in the
compose file), nginx-served frontend with WS proxy. K8s deltas documented in the
compose header. `ANALYST_EXEC_TMP` must be volume-backed in the worker — same
VM-file-sharing lesson as hardening item #2, now at the compose layer.

**(h) CI:** every push = 41 tests at $0 (replay, dev sandbox, fake embedder — no
keys, no Docker, no downloads) + strict-TS frontend build. Live benchmark is
`workflow_dispatch` gated on the `ANALYST_HOSTED_API_KEY` secret, uploads the
scorecard as an artifact.

**README** rewritten as the systems-design front door: thesis → live critic catch →
scorecard command → architecture → honest limitations.

### Not built yet, on purpose

The `finalize` node is a deterministic placeholder for the report writer.
The validator reasons over executed results but does not yet run deterministic
interval/power checks in the sandbox. Auth and multi-tenancy remain out of scope for
the single-user deployment.

---

## Analysis council — independent parallel review · 2026-07-10

**Shipped:** three specialist agents (metric, statistical, data quality) as real
parallel LangGraph `Send` branches over the same immutable executed artifacts;
structured findings with answer keys, evidence references, risks, confidence, and
recompute instructions; deterministic consensus that skips an adjudicator call;
evidence adjudication on disagreement; one bounded targeted recomputation including
dependent tasks; fresh council round after recompute; persisted/replayed council
state; and a React evidence/dissent panel with an on/off baseline toggle.

Two integration bugs fell out of the browser test rather than the unit tests:

1. The frontend hard-coded `provider: hosted`, overriding local/replay backend
   configuration. The request now leaves provider selection to the server.
2. Vite proxied `/runs` but not the new `/datasets` upload endpoint. Both routes now
   use a configurable `ANALYST_API_URL`, which also makes side-by-side local testing
   possible when port 8000 is occupied.

**Verification:** consensus and disagreement/recompute e2e tests, API persistence
assertions, routing checks, CLI-help regression coverage, full browser runs, 55 Python tests passing, Ruff clean,
and strict TypeScript + production Vite build passing.

**Measured result:** `full-council` scores 0.84 vs 0.80 for `full` and doubles trap
flagging from 0.40 to 0.80, but costs 2.44× tokens and 1.99× latency. That result
triggered the adaptive router and `council-no-critic` ablations; more agents remain a
costed policy choice, not a default virtue.

### Cost-aware follow-up · 2026-07-11

Implemented `adaptive` as the product default, with `always` and `off` retained as
explicit controls. A zero-token risk gate scores statistical/temporal claims,
group-composition comparisons, joined rates, multi-table code, evidence chains, and
prior critic rejection. Its score, threshold, reasons, and activation decision are
persisted and visible in the UI. On benchmark questions before inspecting generated
code, the gate selects 0/6 tier-1, 1/7 tier-2, 7/7 tier-3, and 5/5 tier-4 questions;
executed join evidence may elevate additional denominator-sensitive tier-2 work.

Added `adaptive-council` and `council-no-critic` configs. The first measures savings;
the second is a deletion test for a critic that now adds zero overall score. No live
result is claimed until those configurations complete the same 25-question gate.

### Final ablation decision · 2026-07-11

Both configurations completed the integrity gate. `council-no-critic` became the
best-scoring policy at **0.86**, with perfect tiers 1–3, versus **0.56** for the bare
baseline. Removing the critic from `full-council` raised overall score from 0.84 to
0.86 and reduced mean tokens from 19,727 to 17,223. The critic remains reproducible
behind an opt-in flag but is no longer part of either product mode.

`adaptive-council` scored 0.80—the same as the validator-only Standard path—while
using 19,892 rather than 3,429 mean tokens. It failed its cost-saving hypothesis and
was demoted to experimental. Final defaults are therefore:

- **Standard:** council off, critic off, validator on.
- **High assurance:** council always on, critic off, validator on.

This is the final architectural result: ablations removed two mechanisms that did
not earn their cost instead of treating a larger agent count as success.

## Deterministic statistical inference · 2026-07-17

Replaced LLM-only statistical arithmetic with an auditable sufficient-statistics
contract. Executor results may request one/two-proportion, one/two-mean, or
contingency-table inference. A deterministic SciPy layer now computes Wilson and
Newcombe-Wilson intervals, Student/Welch intervals, exact binomial and proportion
tests (automatically selecting Fisher for sparse tables), Welch tests, chi-square,
and effect sizes (risk difference, odds ratio, Cohen's h/d, Hedges' g, Cramer's V).

Typed evidence is passed into the validator prompt, stored in graph/run state,
persisted by both JSON and Postgres stores, streamed over WebSockets, displayed in a
dedicated React panel, and appended to the final answer. Invalid requests fail visibly
as evidence warnings instead of failing the run or disappearing. The published live
scorecard is intentionally unchanged until the benchmark is rerun with the new
prompt and computation path.

## Stratified confounder recomputation · 2026-07-17

Added a `stratified_proportion` sufficient-statistics contract and a deterministic
retry policy for validator findings containing confounding, composition, mix,
complexity, material, stratification, or Simpson signals. The retry requires a
complete group × stratum successes/denominator grid instead of another crude rate.

The engine now performs pooled, equal, or custom direct standardization; computes
crude and adjusted rate intervals and rankings; checks sparse cells and overlap;
detects pairwise Simpson reversals; and, for two groups, reports crude/adjusted risk
differences, a Mantel-Haenszel common odds ratio, and a Cochran-Mantel-Haenszel test.
The UI explicitly marks when adjustment changes the winning group.

Tests build sufficient statistics from the generated CSVs and recover both planted
conclusions: night is the crude best shift but day is best after complexity
standardization; S2 is the crude best supplier but S1 is best after material
standardization. No new live score is claimed yet.

## Automatic EDA and data-quality profiling · 2026-07-17

Moved dataset inspection into manifest construction so it runs once, deterministically,
before any agent call. Every table now receives typed shape, duplicate, candidate-key,
missingness, cardinality, distribution, outlier, category-frequency, and datetime
profiles. Rule-based checks also find invalid likely primary keys, whitespace/case
label collisions, mixed numeric/text encodings, infinite values, and orphaned shared
`*_id` relationships without guessing ambiguous parents.

The report includes affected counts/rates, remediations, and a transparent 0–100
triage score. It never mutates or imputes data. A compact issue summary is injected
into agent schema context; the full report is returned on upload, persisted in JSON
and Postgres run records, streamed/replayed over WebSockets, printed by the CLI, and
rendered as expandable React issue/table/relationship cards.

Focused synthetic tests plant duplicate and invalid keys, missing columns, label
variants, orphan joins, invalid dates, and numeric outliers. API tests verify both
upload-time delivery and completed-run persistence.

## Expanded held-out statistical benchmark · 2026-07-17

Added a second evaluation namespace rather than expanding the development benchmark.
Each seed generates a new six-table product-analytics/support dataset with roughly
37,000 rows: randomized experiment outcomes, one-to-many events, support tickets,
an observational rollout, users, and survey invitations. Runtime-derived ground
truth covers intervals, tests, effect sizes, two composition reversals, a stratified
no-reversal sound control, fanout, nonresponse, multiplicity, missingness, and
practical significance.

The 20 questions are balanced across estimation, testing/effects, confounding, and
design/quality. Each has an auditable three-to-five-component rubric. Typed evidence
is field-scoped during scoring—a p-value checker reads `test.p_value`, for example,
so an unrelated zero cannot earn accidental credit. Percentage/proportion forms and
numeric tolerances are explicit.

A separate integrity gate requires the exact v1 item set, generator seed, dataset
content hash, known configuration, valid rubric structure, and zero execution errors.
Valid runs are selected per `(config, seed)` and aggregated across seeds; 95%
uncertainty is withheld until at least two seed means exist. Smoke, partial, crashed,
and quota-aborted attempts remain diagnostic only. CLI generation, resumable runs,
raw persistence, provenance-linked scorecards, oracle-rubric tests, and broken-run
replacement tests are complete. No live held-out score is claimed yet.

## Governed predictive modeling · 2026-07-17

Added a typed predictive-model request path instead of teaching executors to fit a
model and grade their own training score. Requests name a manifest table, target,
features, validated one-to-one/many-to-one joins, split policy, candidates, primary
metric, and optional slices. The trusted engine supports binary classification
(logistic regression/random forest) and regression (ridge/random forest).

Random splits stratify classification. Temporal splits train on the past and expose
their boundary; entity-group splits prove zero train/test group overlap. Candidate
selection uses only CV inside the training partition, after which the selected
pipeline is evaluated once on the untouched holdout. Results include dummy-baseline
comparisons, full task metrics, a seeded bootstrap interval on the primary metric,
train/holdout gap warnings, original-feature permutation importance, and small-slice
handling.

The engine rejects target-as-feature leakage, identifier memorization by default,
unknown/all-missing features, invalid time/group policies, nonbinary targets,
row-increasing joins, ambiguous join columns, and oversized local requests. Semantic
post-outcome leakage still requires validator/domain review and is documented as a
boundary.

`PredictiveEvidence` now flows through graph state, the validator prompt, events,
JSON/Postgres persistence, CLI JSON/final reports, WebSocket replay, and a React
model panel. Tests cover classification/regression, baseline improvement, bootstrap
intervals, model selection, temporal/group isolation, joins, leakage failures,
artifact collection, and graph-to-final-answer integration.
