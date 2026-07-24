# Deterministic statistics contract

The validator no longer asks an LLM to calculate inferential statistics. Executor
artifacts may provide compact sufficient statistics under `statistical_tests`; the
application validates them and uses SciPy to compute intervals, effects, and tests.
The LLM validator receives those computed results and judges the study design,
assumptions, denominator, grain, confounding, and interpretation.

This is deliberately a sufficient-statistics boundary: raw rows stay in the sandbox,
while the auditable evidence stored with a run remains small.

## Supported requests

### One proportion

```json
{
  "id": "defect_rate",
  "kind": "one_proportion",
  "groups": [{"label": "day", "successes": 20, "n": 200}],
  "null_value": 0.08,
  "confidence": 0.95,
  "alternative": "two-sided"
}
```

Produces a Wilson score interval, an optional exact binomial test, and Cohen's h.

### Two proportions

```json
{
  "id": "shift_comparison",
  "kind": "two_proportion",
  "groups": [
    {"label": "day", "successes": 20, "n": 200},
    {"label": "evening", "successes": 30, "n": 200}
  ]
}
```

Produces Wilson intervals for both rates, a Newcombe-Wilson interval for the risk
difference, risk difference, odds ratio, Cohen's h, and either a two-proportion
z-test or Fisher's exact test when an expected cell is below five.

### One or two means

```json
{
  "id": "duration_comparison",
  "kind": "two_mean",
  "groups": [
    {"label": "new", "mean": 5.0, "std": 2.0, "n": 40},
    {"label": "old", "mean": 4.0, "std": 2.0, "n": 50}
  ]
}
```

`std` is the sample standard deviation (`ddof=1`). One-group requests produce a
Student-t interval and optional one-sample t-test/Cohen's d. Two-group requests
produce group intervals, a Welch interval and test for the mean difference, the raw
mean difference, and small-sample-corrected Hedges' g.

### Categorical association

```json
{
  "id": "operator_outcome",
  "kind": "chi_square",
  "rows": ["A", "B"],
  "columns": ["defect", "ok"],
  "counts": [[30, 20], [15, 35]]
}
```

Produces Pearson's chi-square test and Cramer's V. The evidence includes a warning
when expected cell counts make the chi-square approximation unreliable.

### Stratified proportion adjustment

```json
{
  "id": "shift_mix_adjusted",
  "kind": "stratified_proportion",
  "direction": "lower",
  "standardization": "pooled",
  "cells": [
    {"group": "day", "stratum": "low", "successes": 12, "n": 2000},
    {"group": "day", "stratum": "high", "successes": 20, "n": 500},
    {"group": "night", "stratum": "low", "successes": 15, "n": 2000},
    {"group": "night", "stratum": "high", "successes": 25, "n": 500}
  ]
}
```

The grid must contain one positive-denominator cell for every group/stratum
combination. The engine computes crude rates and direct-standardized rates using one
of three reference populations:

- `pooled` (default): observed stratum distribution across all groups;
- `equal`: equal weight for each stratum;
- `custom`: normalized non-negative `reference_weights` naming every stratum.

Outputs include crude and adjusted rankings, adjusted-rate intervals, overlap and
sparse-cell warnings, and explicit diagnostics when the winning group changes. The
engine checks every group pair for Simpson reversals: a crude direction opposite to
a consistent within-stratum direction. For exactly two groups it additionally
computes the crude and adjusted risk differences, a direct-standardization interval,
the Mantel-Haenszel common odds ratio, and a Cochran-Mantel-Haenszel test.

The adjusted-rate variance uses a binomial normal approximation conditional on the
chosen reference weights. It does not include uncertainty from estimating pooled
weights, clustering, repeated observations, or an estimated confounder model; the
validator must surface those design limitations when relevant.

## Evidence and failure behavior

Every computation produces a typed `StatisticalEvidence` record containing:

- task/request provenance and total sample size;
- confidence intervals with parameter names and methods;
- effect sizes with a descriptive magnitude label;
- hypothesis-test method, statistic, p-value, alternative, alpha, and significance;
- a human-readable summary and assumption warnings.

Malformed or impossible requests do not crash the graph. They produce an `invalid`
evidence record whose warning is persisted, shown in the UI, included in the final
answer, and supplied to the validator.

## Interpretation boundaries

- A p-value is not an effect size or the probability that the null is true.
- `p >= alpha` is not proof of equivalence or no effect.
- Statistical significance does not imply practical significance.
- Sufficient statistics cannot verify independence, data provenance, distribution
  shape, or whether the executor chose the correct denominator and analysis grain;
  those remain validator responsibilities.
- This increment does not yet implement multiple-comparison correction, power
  analysis, equivalence testing, regression adjustment, continuous-confounder
  adjustment, or causal identification.

The published 0.86 scorecard predates this feature. Re-run the live benchmark before
claiming an accuracy improvement from deterministic statistics.
