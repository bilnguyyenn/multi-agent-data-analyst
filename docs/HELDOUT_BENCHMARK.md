# Expanded held-out statistical benchmark

## Why it is separate

The original 25-question manufacturing benchmark was used throughout development
and ablation. It remains the right evidence for component deletion, but repeatedly
improving against it risks benchmark overfitting. The held-out suite therefore uses
a different product-analytics/support domain, independent generated data, new table
and column names, and a separate scorecard. Its results never overwrite or silently
blend with the original headline metrics.

"Held out" here means isolated from the development benchmark and regenerated from
independent seeds, not secret. The generator and rubric are public for reproducible
portfolio evaluation. Ground truth is derived from the evaluated CSV bytes at run
time, and mechanism metadata is written beside—never inside—the sandbox-mounted
dataset directory.

## Dataset

Each seed generates six related tables and roughly 37,000 rows:

- users and randomized A/B experiment outcomes;
- a one-to-many page-event log;
- support tickets with team and severity;
- an observational feature rollout with plan composition;
- survey invitations with outcome-dependent nonresponse.

Planted mechanisms include a small randomized conversion uplift, a small continuous
effect, two composition reversals, fanout bias, nonresponse bias, and a rare
subgroup. The suite also includes a stratified no-reversal sound control so a system
does not earn credit merely by declaring Simpson's paradox everywhere.

## Twenty-question coverage

| category | n | coverage |
|---|---:|---|
| estimation | 5 | Wilson/t intervals, conditional estimands, small-subgroup uncertainty |
| tests and effects | 5 | exact/proportion/Welch/chi-square tests; Cohen's h, Hedges' g, Cramer's V |
| confounding | 5 | pooled/equal standardization, two reversals, one sound control, causal restraint |
| design and quality | 5 | join grain, nonresponse, multiplicity, missing data, practical significance |

## Component-level scoring

Each question has three to five explicit checks rather than one LLM-judged grade.
Checks cover the numerical estimate, uncertainty/test/effect size, method or
estimand, and interpretation. The item score is the weighted fraction of checks
satisfied; the overall score is the mean item score.

Numeric checks search the final answer and semantically scoped typed statistical
evidence. For example, a p-value check reads only `test.p_value`, so an unrelated
`null_value=0` cannot accidentally earn credit. Percentage and proportion forms are
accepted with documented tolerances. Text checks use visible marker sets and can
forbid contradictory language. Every saved result contains its individual check
outcomes for audit.

This rubric score is not directly comparable to the original benchmark's
1/0.5/0 trap score.

## Multiple seeds and uncertainty

The CLI accepts any number of independent seeds. The scorecard selects the newest
valid run for each `(configuration, seed)` pair, aggregates category scores, and
shows a 95% t-interval over seed-level mean scores only when at least two seeds are
available. One seed is reported as a point estimate without invented precision.

## Integrity gate

A run enters `heldout_eval_results/SCORECARD.md` only if it has:

- suite id `heldout-statistical-v1`;
- the exact 20 unique v1 item ids;
- a known configuration and integer generator seed;
- the dataset content hash;
- well-formed 0–1 rubric results and zero execution errors;
- no aborted/quota-exhausted status.

Partial, smoke, cassette-miss, and crashed attempts remain diagnostic artifacts and
cannot replace a valid score.

## Commands

```bash
# Optional: inspect one generated held-out dataset.
python -m analyst gen-heldout-data --out .heldout_data/seed_20260717 \
  --seed 20260717

# Cheap four-question plumbing check; never scorecard-eligible.
python -m analyst heldout-eval --provider hosted --subset smoke

# Full selected-policy evaluation on two independent datasets.
python -m analyst heldout-eval --provider hosted --configs no-critic \
  --seeds 20260717,20260718 --resume

# Optionally compare the selected Standard policy with the bare baseline.
python -m analyst heldout-eval --provider hosted --configs no-critic,bare \
  --seeds 20260717,20260718 --resume

# Rebuild without model calls from valid saved runs.
python -m analyst heldout-scorecard
```

No held-out accuracy number should be placed on the resume until at least one full
live run passes the integrity gate; two seeds are preferred before quoting an
uncertainty interval.
