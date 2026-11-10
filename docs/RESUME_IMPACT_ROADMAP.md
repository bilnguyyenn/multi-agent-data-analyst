# Roadmap: an operations analyst with measured real-world usefulness

Planning date: November 9, 2026. Implementation was subsequently authorized by the
user. Spending ceiling is $0; live-model evaluation and actual participants are
now explicitly deferred. See [current status](ROADMAP_STATUS.md). Confirmed target roles: AI/ML engineering and data science/analytics.
Effort estimates are approximate engineering time, not promises about elapsed time
or model usage. Targets below are future acceptance criteria, not achieved results.

## Strategic direction

Build a reliable analyst for a recurring operations workflow: ingest a versioned
dataset, answer a defined business question, expose the evidence behind every
important number, and compare results on the next refresh.

Recommended first product: a weekly retail operations review answering:

1. How did net sales and cancellations change?
2. Which products, countries, or customer cohorts contributed to the change?
3. Which conclusions are sensitive to missing identifiers, definitions, or filters?
4. What should a human investigate next, and what evidence supports that suggestion?

Contribution analysis is descriptive, not causal root-cause identification. This
scope avoids promising profit, inventory optimization, or causal effects when the
source data do not contain the necessary information.

The manufacturing generator remains valuable as a controlled regression suite.
The real-data workflow adds a distinct demonstration of usefulness.

## Evidence for both career tracks

Build one shared product and evaluation package, then emphasize different evidence
for each application. Both tracks need correct results and demonstrated usefulness.

| Shared milestone | AI/ML engineering emphasis | Data science / analytics emphasis |
|---|---|---|
| Metric definitions and evidence ledger | Structured contracts, validated tool outputs, resolvable provenance | Correct grain, denominators, units, exclusions, and business interpretation |
| Real-data and multi-turn evaluation | Agent orchestration, context handling, error recovery, cost/latency tradeoffs | Independent ground truth, uncertainty, confounding, missingness, and failure analysis |
| Repeatable snapshot workflow | Idempotency, durable execution, resource limits, reproducible deployment | Comparable periods/cohorts, definition consistency, and interpretable change analysis |
| User pilot | Successful end-to-end task completion and operational reliability | Time to a correct decision, evidence comprehension, and practical usefulness |

For an AI/ML engineering resume, lead with the evaluated agent system and its
reliability/cost tradeoffs. For a data science/analytics resume, lead with the
business question, analytical validity, and measured user outcomes. Use the same
verified results in both versions; change the emphasis rather than the claims.

Do not add predictive models merely to label the project ML. If a pilot workflow
needs prediction, define its decision, target, feature-availability cutoff, and
deployment split first; then evaluate against a naive baseline on external data.
Otherwise, rigorous descriptive/statistical analysis remains the main analytical
contribution, while the agent system supplies the AI engineering contribution.

## Current position and gaps

Already implemented: sandboxed Python/SQL task DAGs; deterministic profiling,
statistics, and predictive evaluation; review-policy ablations; typed evidence;
follow-up context and lineage; portable reports; replay tests and CI.

The strongest gaps are:

- Published evidence is not fully self-contained in a fresh clone. The main
  scorecard cites `raw_20260711_204642.json` and `raw_20260711_163629.json`, which
  exist locally but are ignored. The held-out scorecard's two cited raw files are
  also ignored. Its README/resume text says performance is pending while the
  scorecard reports 0.719. Resolve this through the owning harness, not by choosing
  whichever wording sounds strongest.
- The 25-question result is a small development benchmark. The predictive 30/30
  result is synthetic behavioral coverage. Follow-ups have scripted software
  checks, without a live conversational-quality estimate.
- Business definitions are implicit. Terms such as revenue, cancellation rate,
  active customer, or best shift can hide incompatible denominators and filters.
- Reports preserve evidence packets, but a user cannot yet inspect a reliable
  computation reference attached to each major numerical claim.
- The product has no demonstrated recurring workflow or measured user benefit.
- Execution is designed for one trusted local user, with in-memory profiling and
  explicit file-mutation and sandbox limits. Public arbitrary-code hosting needs
  additional work beyond a frontend deployment.

## Milestones, in order

| Milestone | Deliverable | Completion gate | Approximate effort |
|---|---|---|---|
| 0. Repair the evidence trail | One consistent, reproducible account of existing results | Fresh clone can regenerate each published scorecard from its cited, reviewed raw inputs without API calls | 2–4 hours |
| 1. Build one real-data vertical | Versioned retail dataset, metric definitions, and an evidence-linked operations report | Independent SQL/pandas reference agrees with every canonical KPI; ambiguous requests ask for clarification | 8–12 hours |
| 2. Evaluate the complete analyst | Locked real-data tasks and multi-turn scenarios, matched baselines, cost/latency reporting | All preselected evaluation attempts are accounted for; correctness, abstention, failures, latency, and usage are reproducible | 10–16 hours plus explicitly budgeted live runs |
| 3. Make the workflow repeatable | Saved analysis playbook and snapshot-to-snapshot comparison | A second data snapshot produces a fresh analysis, distinguishes data changes from definition changes, and survives restart without duplicate completion | 8–12 hours |
| 4. Run a small usability pilot | Measured task completion and a portfolio case study | At least 3 people attempt the same predefined tasks; report all outcomes and the measurement protocol | 6–10 hours plus participant availability |

Expected scope: roughly 34–54 engineering hours, spread across bounded sessions.
Finish a reviewable slice and its checks before starting another. Keep early
development API-free. Plan live evaluation only after freezing its task set and
estimating tokens, runtime, and a user-approved spending ceiling.

## Milestone 0: evidence and reproducibility

- Validate all cited saved runs with their existing integrity gates. Inspect them
  for accidental secrets or private content before publishing the specific files.
- Publish the valid raw inputs that substantiate the existing scorecards, or
  explicitly narrow/withdraw an unsupported headline. Preserve historical results
  with their implementation/model/routing versions.
- Reconcile README, held-out README, resume writeup, interview notes, and NEXT_STEPS.
- Add a no-network reproduction check that validates every provenance link and
  recomputes scorecards in a temporary output directory.
- Keep historical and current-version measurements separate. Later features do
  not inherit an earlier accuracy number automatically.

Resume payoff: reviewers can audit the strongest existing claims themselves.

## Milestone 1: business definitions and claim-level evidence

Use UCI Online Retail as an initial real-data integration. Its official record
contains 541,909 transactions and identifies cancellation invoices by a leading
`c` in InvoiceNo. The dataset is historical and does not demonstrate current
business conditions. Attribute and pin the source, file hash, extraction rules,
and deterministic sample used during development.

Source: [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail).
The source spreadsheet is 22.6 MB; the current browser accepts CSVs up to 20 MB.
Start with a documented bounded slice, then implement a measured full-data path.
Do not silently raise limits and assume memory use is solved.

Deliverables:

- A small versioned metric registry: formula, grain, inclusion/exclusion rules,
  units, time zone, source fields, and permitted groupings. Distinguish cancelled
  transactions from verified physical returns. Do not infer margin without costs.
- A clarification path for ambiguous questions. The user chooses the intended
  metric or definition before an expensive analysis begins.
- A deterministic evidence ledger: computation ID, metric-version ID, source
  snapshot hash, query/transformation, filters, row counts, numerator/denominator,
  and aggregate result. Persist execution artifacts in an inspectable local store.
- Clickable references from numerical claims to ledger entries. References must
  resolve to recorded computations; the model cannot invent a source identifier.
- A report that answers the weekly review questions and retains missing-data,
  denominator, small-sample, and causal-interpretation caveats.

Acceptance criteria:

- Canonical KPIs match independently written reference queries on fixed fixtures
  and the pinned real-data slice.
- Test cancellation rules, duplicates, missing customer IDs, empty periods,
  negative quantities, and joins at the wrong grain explicitly.
- Every headline numeric claim either resolves to evidence or is visibly marked
  unsupported. A reference proves provenance, not correctness by itself.
- Loading real data never silently deletes, imputes, or rewrites records.

Resume payoff: semantic data modeling, grounded numerical reasoning, and a real
analytical workflow with understandable business value.

## Milestone 2: external and conversational evaluation

Freeze a development/evaluation split before tuning. Use separate datasets or
database groups where possible, not random near-duplicate questions. A public
benchmark is external to this project's generator, not guaranteed unseen in a
model's pretraining.

Evaluation tracks:

1. Real-data business questions: exact KPIs, filters, joins, time windows,
   incomplete-data caveats, ambiguous requests, and unsupported questions.
2. Multi-turn chains: pronoun references, changed filters, corrections, conflicting
   prior answers, truncated context, and requests that cannot be answered.
3. Optional SQL diagnostic: a pinned subset of BIRD Mini-Dev using the official
   evaluator and supported database dialect. This measures a component, not the
   full statistical/reporting product.

BIRD Mini-Dev provides 500 selected tasks across 11 databases. Its maintainers
announced corrected/replaced annotations on October 28, 2026; pin a release,
dataset hashes, and scorer revision. Label any subset result precisely.
Source: [official BIRD Mini-Dev repository](https://github.com/bird-bench/mini_dev).

Start with a 20–30-task development smoke set. For the final evidence package, aim
for 100–150 preselected business tasks plus about 20 multi-turn chains, expanding
only when the approved evaluation budget supports it.

Compare the same model versions, data, tools, and execution environment:

- a well-designed single-agent baseline;
- Standard mode;
- High assurance on a predeclared subset or the full set if affordable.

Use independent SQL/pandas reference computations as correctness oracles, not as
another LLM baseline. Track tokens/price as a separate axis; if claiming equal-cost
superiority, run an explicitly budget-matched comparison. Replay latency must
never be compared with live-model latency as if they were the same protocol.

Publish answer correctness, confident incorrect answers, abstention/clarification
behavior, evidence-reference validity, execution errors, p50/p95 latency, tokens,
cost per completed task, and cost per correct task. Keep failed/aborted attempts
in the accounting. Publish uncertainty and paired differences, with resampling
at a dataset or conversation level where tasks are dependent. No model-as-judge
alone for objective numerical answers.

Resume payoff: a defensible comparison on external data, including the accuracy,
cost, and reliability tradeoffs an interviewer can interrogate.

## Milestone 3: a recurring workflow people can use

Implement one saved playbook: a named set of metric definitions, parameters,
questions, and report sections. Run it on two immutable data snapshots and produce
a comparison with the same metric versions and windows.

- Treat a refresh as a new root analysis on new data. Existing follow-up hash
  guards must not be bypassed to carry old answers into a changed dataset.
- Store snapshot IDs, timestamps, definition versions, source provenance, and
  links to prior comparable runs. Flag incompatible comparisons explicitly.
- Provide an explicit refresh/run action first. Add scheduling only if pilot users
  need it; avoid unsolicited email or external notifications.
- Add idempotent job submission, cancellation, per-run budgets, durable status,
  atomic state updates, and recoverable failure behavior for the saved workflow.
- Measure peak memory and runtime before enabling full-size real datasets.
  Introduce chunked/database-native ingestion where measurements require it.
- Add at most one ingestion adapter after the file workflow works. A read-only
  Postgres source should use scoped credentials, database permissions, timeouts,
  and a trusted extraction boundary. Generated code must not receive credentials
  or unrestricted network access.

Acceptance: two successful refreshes with validated comparisons, injected failure
and restart tests, no duplicate completion, and a recorded resource envelope.

Resume payoff: a repeatable data product with operational ownership, not only an
interactive demonstration.

## Milestone 4: usage evidence and portfolio delivery

Recruit 3–5 willing users such as classmates, research teammates, or analysts.
No outreach is performed by this plan. Give them predefined tasks and preserve
consent and privacy for any recorded sessions or quoted feedback.

Measure whether each person completes each task, time to a correct answer,
interventions, errors, evidence inspection, and whether they choose to reuse the
workflow. Compare matched manual SQL/spreadsheet tasks with counterbalanced order
where practical. Report small-sample limitations and all participants; describe a
classmate/public-data study accurately rather than implying enterprise adoption.

Portfolio package:

- a 90-second demo showing question → evidence → follow-up → refreshed report;
- a reproducible release and one-command local setup;
- an architecture diagram explaining boundaries and tradeoffs;
- a case study with real-data provenance, baseline comparisons, failure analysis,
  and measured user outcomes;
- a restricted public sample-data demo if useful. Arbitrary uploaded-code hosting
  requires a separate threat model, authentication, resource budgets, and stronger
  isolation; the present Docker socket setup must not be casually exposed.

Resume payoff: measured usefulness and a concrete product story beyond features.

## What to defer

Defer additional agents, fine-tuning, broad model search, Kubernetes, many cloud
connectors, and forecasting until a measured failure or user need justifies them.
Prioritize the above milestones over accumulating technologies.

If time is short, complete milestones 0–2 and a small pilot before expanding
automation. A clear external evaluation and honest user study can provide a
stronger evidence package than a large unmeasured feature set.

## Future resume wording — placeholders only

Do not use these as achieved claims until the corresponding evidence exists:

- Built an evidence-backed operations analyst with versioned metric definitions,
  contextual follow-ups, and snapshot-linked reports; evaluated on **[N] real-data
  tasks**, achieving **[X] vs [Y]** for a matched single-agent baseline.
- Automated **[workflow]** across **[N] data refreshes**; **[U] pilot users** completed
  matched analyses in **[measured time]**, compared with **[manual baseline]**,
  while retaining computation-level provenance and visible caveats.
- Reduced **[latency/tokens/cost per correct answer]** by **[measured change]** at
  **[measured quality level]**, using versioned evaluations and a documented
  failure taxonomy.

## First implementation session

Complete milestone 0, then specify one metric end to end: cancellation-adjusted
sales on a pinned real-data slice. Write its definition and independent reference
query before changing prompts. Implement one resolvable computation reference and
its UI/report presentation. Finish checks, record limitations, and commit that
reviewable slice. Do not begin the full live evaluation or public deployment in
the same session.
