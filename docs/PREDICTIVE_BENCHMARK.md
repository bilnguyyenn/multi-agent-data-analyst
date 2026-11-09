# Predictive reliability benchmark

The modeling engine has a versioned, API-free development benchmark: ten fixed
scenarios on three generated-data seeds, for **30 case instances**. It exercises
the real scikit-learn engine, rather than mocked predictions. The published run
passes **30/30**; see the [report](../predictive_eval_results/REPORT.md) and
[raw evidence](../predictive_eval_results/results.json).

This measures implementation reliability on planted synthetic behavior. It does
not measure whether the LLM chooses the right model specification, external
predictive accuracy, production readiness, or statistical generalization across
arbitrary datasets. It is separate from both agent scorecards.

## Run locally

```bash
pip install -e ".[dev]"
python -m analyst predictive-benchmark --out-dir predictive_eval_results
```

The command uses no provider, credentials, Docker, network, or downloaded data.
Generated CSVs live in a temporary directory and are removed when execution ends.
`results.json` and `REPORT.md` remain available for inspection. Exit status is zero
only when the suite is complete and every case passes. CI runs the same command
and uploads its reports even when the benchmark fails.

## Fixed v1 contract

Seeds: `20261109`, `20261110`, `20261111`. Each dataset has 960 observations,
240 entities, 12 timestamp cohorts, three segments, and a one-to-many event table.
The linear signal is `3*x + 0.8*is_north`, with Gaussian noise of standard deviation
0.35. Conversion is a Bernoulli draw with logit `2.5*x + 0.5*is_north`.
All requests use three training CV folds and 100 bootstrap draws. Classification
compares logistic regression with a random forest; regression uses ridge.

| Scenario | Required behavior |
|---|---|
| Classification signal | AUC ≥ 0.75; improvement over dummy AUC ≥ 0.25; log loss better than baseline; all three slices present |
| Regression signal | RMSE ≤ 0.7; baseline minus model RMSE ≥ 1.8 |
| Temporal cohorts | Same regression bounds; strict chronology at holdout and every CV boundary |
| Group holdout | Same regression bounds; no entity overlap between train and holdout |
| No signal | Independent-noise target; holdout R² between −0.2 and 0.1 |
| Temporal shift | Sign reversal in final three cohorts; RMSE ≥ 4.0, at least 0.5 worse than baseline, explicit baseline-failure warning |
| Target leakage | Invalid result with the target-as-feature rejection reason |
| Identifier leakage | Invalid result with the identifier rejection reason |
| Join fanout | Invalid result with the join-validation rejection reason |
| Single timestamp | Invalid result explaining that temporal separation is impossible |

Every computed case also requires finite CV/holdout/baseline metrics, the requested
candidate set, a winner selected by training CV, all source rows accounted for,
and a finite ordered bootstrap interval. Safety cases require the specific
rejection reason and no model or metrics; an unrelated error earns no credit.

The tolerances were written before the first suite run and are stored with the
requests in the versioned contract. These are broad behavioral tolerances based
on the generator, not tuned performance targets. The data seeds are visible
development seeds, not a sealed holdout. Changing the generator, case set, seeds,
or scoring rules requires a suite-version change before comparing results.

## Reproduction and integrity

Raw JSON records the complete contract and its SHA-256 digest, generated CSV
content hashes per seed, engine/benchmark source hashes, Python/library versions,
all typed model evidence, failed requests, and recomputed check verdicts.
The report does not trust a saved success flag: it recomputes checks from evidence
and rejects missing/duplicate case instances, wrong contracts, and execution errors.
Hashes identify the evaluated inputs and implementation; they do not authenticate
an artifact against deliberate modification. Dependency upgrades can change exact
metrics; the saved environment explains the published run.

## Bugs caught and regression coverage

A row-count temporal split could put simultaneous observations in both partitions.
The engine now moves the full boundary cohort into holdout and splits timestamp
cohorts for forward CV. It reports actual holdout fraction and each fold's time
boundaries. If too few distinct training timestamps remain, it rejects the request.
The actual holdout fraction may exceed the requested fraction to keep cohorts whole.

Non-finite CV scores now invalidate a request before model selection. Regression
tests cover both failures. Another test changes only the temporal holdout labels
and verifies that the selected model, candidate CV scores, and training score stay
identical while the holdout result changes.

## Limits

The sign-reversal case checks visible baseline failure; it does not implement a
general drift detector. Log loss tests probability quality, not a full calibration
audit. No-signal bounds are a sound control, not a proof against every spurious fit.
Bootstrap intervals resample rows and condition on the fitted model; temporal or
entity dependence and model-development uncertainty are not represented. Production
use still needs task-specific external data, feature-availability governance,
calibration/drift assessment, and operational safeguards.
