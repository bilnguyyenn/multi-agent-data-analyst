# Next Steps — evaluation complete

The core build and ablation study are complete. The integrity-gated scorecard now
contains seven zero-error, 25-question runs. The evidence-backed product policy is:

- **Standard:** planner + sandboxed executors + statistical validator
  (`no-critic`, council off): 0.80 overall, 3,429 mean tokens, 27.6s mean latency.
- **High assurance:** standard path + parallel analysis council
  (`council-no-critic`): best overall score, 0.86, with perfect tiers 1–3.
- **Legacy critic:** opt-in only. It adds no overall accuracy, uses 2.36× the
  standard path's tokens, and false-positives on 50% of sound critic-bench controls.
- **Adaptive council:** experimental only. It scores 0.80 while using 19,892 mean
  tokens, so it does not earn default status.

No additional agent or evaluation feature is required for the resume version.

## 1. Publish and validate the demo

```bash
git init
git add -A
git commit -m "multi-agent autonomous data analyst"
```

- Create a repository (suggested name: `agentic-analyst`) and push `main`.
- Confirm CI passes both the Python and frontend jobs.
- Run the full Docker stack once and capture one Standard run, one High-assurance
  run, and the corresponding Jaeger trace for interview demonstrations.
- Keep the verified raw evaluation outputs and `eval_results/SCORECARD.md` in the
  repository; they make the headline metrics reproducible without new API calls.

## 2. Interview preparation

Practice this 90-second story:

> A question becomes a dynamic task DAG executed in parallel inside networkless
> Docker sandboxes. A statistical validator reviews the resulting claims, and an
> optional three-specialist council independently analyzes high-assurance requests.
> I evaluated seven configurations on a 25-question benchmark with planted traps.
> The selected council-without-critic policy scored 0.86 versus 0.56 for the bare
> baseline. Ablations showed the validator was load-bearing, while the critic and
> adaptive router did not justify their cost, so I removed both from the default.

Be ready to explain:

- why the dynamic analysis DAG is data inside a static LangGraph control loop;
- how Docker isolation and execution budgets bound generated-code risk;
- why a negative ablation result is a design improvement, not a failed project;
- why Standard and High-assurance modes expose the cost/accuracy tradeoff;
- how deterministic EDA separates actionable quality failures from descriptive
  outlier/skew signals without automatically mutating data;
- why tier-4 results remain directional (five trap questions per configuration).

## 3. Optional future research — not needed for the resume

Completed: deterministic confidence intervals, effect sizes, core hypothesis tests,
and stratified proportion recomputation now run from executor-supplied sufficient
statistics and stream as typed evidence through the validator, API, final report,
and UI. The planted shift/job-complexity and supplier/material Simpson reversals are
corrected in deterministic tests.

Completed: automatic EDA/data-quality profiling now runs during manifest creation.
It computes table/column summaries, missingness, duplicates, key and referential-
integrity checks, label/mixed-type issues, distributions, outliers, date ranges, and
a transparent readiness score. Typed reports reach agent schema context, upload/API
responses, persisted runs, CLI output, WebSocket replay, and the React UI.

Completed: a separate 20-question held-out statistical suite now evaluates five
questions each for estimation, tests/effect sizes, confounding, and design/data-
quality reasoning on independently generated product-analytics datasets. It uses
component rubrics, a no-reversal sound control, seed/content-hash provenance,
multi-seed aggregation, and its own zero-error integrity gate and scorecard.

Completed: governed predictive modeling now supports binary classification and
regression with validated relational joins; random, temporal, or entity-group
holdouts; training-only CV model selection; dummy baselines; holdout metrics with
bootstrap intervals; permutation importance; and subgroup diagnostics. Typed model
evidence is validated, persisted, streamed, included in CLI/final output, and shown
in the React UI. Invalid/leaky specifications fail visibly.

1. Run the selected Standard policy on two held-out seeds before publishing any new
   accuracy number; optionally run the bare baseline on the same seeds.
2. Re-run the original statistical and trap benchmark tiers to measure the new
   deterministic-statistics features before
   updating the published headline score.
3. Repeat the selected council policy's original trap subset for a tighter
   confidence interval.
4. Add survivorship-aware cohort reconstruction for historical fleet comparisons.
5. Add multiple-comparison correction and power/equivalence analysis.
6. Add regression adjustment for continuous or high-cardinality confounders.
7. Add a locked predictive benchmark covering calibration, temporal drift, entity
   leakage, and baseline-relative performance before claiming model accuracy.
8. Add model-artifact serialization/model cards only if the product needs reusable
   batch or online inference.
9. Measure memory reuse on a second pass of repeated questions.

Avoid adding more agents, infrastructure, or routing heuristics unless a measured
failure mode specifically requires them.
