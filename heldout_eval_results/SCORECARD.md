# Held-out Statistical Benchmark

_Suite `heldout-statistical-v1` · generated 2026-07-17 16:07._

_Integrity gate: only complete 20-question runs with the exact v1 item set, a dataset content hash, and zero execution errors enter this report. Each question is scored by an explicit component rubric; ground truth is derived from independently generated CSV bytes at evaluation time._

| config | overall rubric score | seeds | items | estimation | tests/effects | confounding | design/quality | checks | mean latency | mean tokens | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no-critic | **0.719** ± 0.524 | 2 | 40 | 0.8 | 0.655 | 0.725 | 0.695 | 121/170 | 26.5s | 7034 | 0 |

## Result provenance

- `no-critic`, seed `20260717`: `heldout_eval_results/heldout_raw_20260717_152930.json`
- `no-critic`, seed `20260718`: `heldout_eval_results/heldout_raw_20260717_153831.json`

## Coverage

- **Estimation:** Wilson and t intervals, conditional estimands, small subgroups.
- **Tests/effects:** exact/proportion/Welch/chi-square tests plus Cohen's h, Hedges' g, and Cramer's V.
- **Confounding:** two planted composition reversals, equal/pooled standardization, and one no-reversal sound control.
- **Design/quality:** one-to-many fanout, outcome-dependent nonresponse, multiplicity, missing data, and practical significance.

## Interpretation

The overall score is the mean fraction of rubric components satisfied, not the original benchmark's binary/trap score. A 95% interval is shown only when at least two independent generated-data seeds are available, using a t-interval over seed-level mean scores.
