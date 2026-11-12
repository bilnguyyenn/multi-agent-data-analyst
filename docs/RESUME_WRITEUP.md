# Resume and portfolio writeup — `analyst`

Agent accuracy numbers come from complete, zero-error, 25-question runs in the
committed scorecard, with model routing held constant across configurations.
The predictive reliability result comes from a separate synthetic development
suite of fixed model specifications; it is a behavioral pass count, not accuracy.

**Recommendation for Bill's data-science resume:** replace the county-level obesity
project with this project. The tennis engine already demonstrates conventional
modeling depth; this project adds LLM systems, evaluation design, statistical
reasoning, and safe code execution.

## Project title

**Multi-Agent Autonomous Data Analyst** — LangGraph · FastAPI · React/TypeScript ·
Docker · Pandas · DuckDB

## Resume bullets — selected version

- Built a multi-agent data analyst that converts natural-language questions into
  parallel task DAGs, executes generated Python/SQL in isolated Docker sandboxes,
  and supports contextual follow-up analyses with dataset-hash checks, source-run
  lineage, and portable evidence reports through FastAPI/React.
- Designed a 25-question ablation benchmark with planted statistical traps;
  achieved **0.86 vs. 0.56** for a bare planner/executor and selected a lower-cost
  **0.80** Standard mode based on measured token/accuracy tradeoffs.
- Engineered leakage-aware scikit-learn pipelines with training-only model
  selection, timestamp-safe CV, and baseline comparisons; passed **30/30 synthetic
  reliability cases across three seeds**, including leakage and temporal-shift controls.

Use the first two bullets for an AI engineering resume; include the third for
data-science/MLE roles. The third metric counts case instances, not distinct
scenarios or prediction accuracy. See the
[reproducible report](../predictive_eval_results/REPORT.md).

## Real-data operations evidence — current implementation

For AI/ML engineering applications:

- Built a governed retail-analysis workflow with typed model tool proposals,
  versioned metric contracts, content-addressed snapshots, and query-linked reports;
  implemented durable, idempotent jobs with cancellation, retry budgets, and
  restart recovery.
- Verified 140 deterministic metric/oracle and definition-guard checks across
  100 real-data questions and 20 two-turn scenarios; prepared matched local-agent
  policies with resumable call accounting and explicit cost/resource ceilings.

For data science/analytics applications:

- Defined seven auditable retail KPIs over a 541,909-row historical transaction
  source, preserving cancellation, missing-ID, duplicate, and denominator caveats;
  produced reproducible weekly reviews with independent Decimal reference checks.
- Built exact contribution analysis across 2,619 real product codes and 18 countries,
  reconciling a recorded weekly sales change to the penny; validated every aggregate
  group across eight metric/dimension cases against independent Decimal calculations.

Additional engineering evidence: fixed unknown-versus-zero behavior in truncated
comparisons; added bounded, typed contribution queries, stored evidence, portable
reports, and rejection of nonadditive breakdowns. Added contributor-to-source drill-down
with bounded previews, explicit omitted-row counts, daily/component reconciliation,
parent evidence lineage, and six independent real-data inspection cases. A counterbalanced usability pilot
remains prepared with consent, correctness scoring, and unsuccessful-task accounting.

These bullets describe completed engineering and deterministic validation. Live
model accuracy and measured user time savings remain unclaimed: both measurements
were explicitly deferred. The local resource measurements are one-process runs,
not hosted production or agent-response benchmarks. See [the case study](RETAIL_CASE_STUDY.md).

## Compact one-bullet version

- Built and evaluated a multi-agent data analyst (LangGraph, FastAPI, Docker) that
  executes sandboxed Python/SQL, computes deterministic statistical and predictive
  evidence, and scores **0.86 vs 0.56** for a bare baseline on a 25-question trap
  benchmark; ablations removed review agents that added cost without improving
  accuracy.

## MLE/infra-leaning alternative

- Delivered a containerized agent service with FastAPI, arq/Redis workers, Postgres
  run persistence, WebSocket streaming, and OpenTelemetry/Jaeger traces; untrusted
  generated code runs in networkless, resource-limited Docker sandboxes.
- Instrumented tokens and latency per configuration and built replayable, key-free
  CI plus a provenance-linked scorecard that rejects incomplete or errored runs.
- Added an API-free predictive reliability CI gate with raw model evidence, fixed
  scenario contracts, dataset/source hashes, and no-signal/shift controls; fixed
  temporal leakage at both holdout and CV boundaries.
- Implemented multi-turn analysis with bounded conversation context, explicit
  parent-run lineage, inherited review policies, and dataset-fingerprint checks
  at submission and worker startup to reject stale-data follow-ups.
- Selected a 0.80/3,429-token Standard mode and a 0.86/17,223-token High-assurance
  mode after controlled ablations showed the critic and adaptive router did not earn
  default status.

## Portfolio description

**Multi-Agent Autonomous Data Analyst**

A planner decomposes a business question into a dynamic task DAG. Parallel executor
agents write and run SQL or pandas against uploaded tables inside isolated Docker
containers. A deterministic EDA layer profiles table grain, columns, distributions,
missingness, duplicate/invalid keys, and inferred relationship integrity, then
streams actionable findings and a transparent readiness score through the UI.
For prediction requests, a governed modeling layer—not generated model-training
code—owns validated joins, temporal/group/random holdouts, training-only CV model
selection, dummy baselines, bootstrap metric intervals, permutation importance, and
slice diagnostics. It rejects target/identifier leakage and row-multiplying joins.
A separate engine reliability suite passes 30 synthetic case instances across
three fixed seeds and preserves complete evidence; it checks expected failures
and weak models as well as strong signals. Timestamp cohorts stay together in
holdout and CV, and changing only holdout labels cannot affect model selection.
Standard mode sends the resulting claims through a statistical validator.
High-assurance mode additionally fans the same immutable evidence to
metric, statistical, and data-quality specialists; deterministic consensus avoids
an extra model call, while disagreement invokes an adjudicator that may request one
bounded recomputation. Results, provenance, risks, and dissent stream through a
React interface over WebSockets.
Users can ask a related question from a completed run. Each follow-up becomes a
separate analysis with bounded, caveat-preserving prior context; hash checks at
submission and worker startup reject changed datasets before model access. Parent
lineage and context remain visible in the UI and exported reports.

The central contribution is evaluation rather than agent count. I generated a
25-question manufacturing benchmark with objective gold answers and planted traps
for confounding, survivorship, join fan-out, and Simpson's paradox. Seven complete
ablation configurations show that the selected council-without-critic policy scores
0.86 versus 0.56 for a bare planner/executor. The statistical validator is
load-bearing: removing it drops overall score from 0.80 to 0.68 and eliminates trap
flagging. The critic, by contrast, adds no overall accuracy and false-positives on
50% of sound controls, so it was removed from the product path. An adaptive council
router also failed to save tokens and remains only as a documented negative result.

To reduce development-benchmark overfitting, I added a separate 20-question suite
on independently generated product-analytics data. It covers interval estimation,
hypothesis tests/effect sizes, stratified confounding, and design/data-quality
failure modes; exact item-set/content-hash integrity and a stratified no-reversal
sound control prevent partial runs and indiscriminate trap flagging from earning
credit. Two historical runs pass the gate: 0.719 across 40 question instances, with wide two-seed uncertainty. This is generated-data rubric performance, not external predictive accuracy.

Tech: Python, SQL, LangGraph, FastAPI, React/TypeScript, Docker, pandas, scikit-learn, DuckDB,
Chroma, Postgres, Redis, OpenTelemetry/Jaeger, GitHub Actions.

## Interview one-liner

> I built a multi-agent data analyst, then used a reproducible ablation benchmark to
> prove which agents earned their cost: 0.86 for the selected council policy versus
> 0.56 bare, with the critic and adaptive router removed after negative results.

## Numbers to memorize

- High assurance: **0.86**, 17,223 mean tokens, perfect tiers 1–3.
- Standard: **0.80**, 3,429 mean tokens, 27.6s mean latency.
- Bare planner/executor: **0.56**.
- No validator: **0.68**, with zero trap flagging.
- Predictive engine reliability: **30/30 case instances**, ten synthetic scenarios
  across three development seeds; **not** external predictive accuracy.
- Critic-bench: **0.83 recall**, **0.71 precision**, **0.50 false-positive rate**.
- Held-out statistical suite: **0.719 component-rubric score** across 40 instances
  and two generated-data seeds; very wide uncertainty. Historical implementation only.

Tier-4 uses five planted trap questions per configuration, so describe trap-specific
deltas as directional. The 0.86 and 0.56 headline scores each use all 25 questions.
