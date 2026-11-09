# Predictive reliability benchmark

Suite `predictive-reliability-v1` · **30/30 case instances passed**

Integrity gate: **PASS**.

Synthetic development benchmark of fixed modeling specifications. This does not
measure LLM planning accuracy, real-world generalization, or a production guarantee.

| case | category | passed seeds |
|---|---|---:|
| classification_signal | signal | 3/3 |
| regression_signal | signal | 3/3 |
| temporal_cohorts | split | 3/3 |
| group_holdout | split | 3/3 |
| no_signal | negative_control | 3/3 |
| temporal_shift | negative_control | 3/3 |
| target_leakage | rejection | 3/3 |
| identifier_leakage | rejection | 3/3 |
| join_fanout | rejection | 3/3 |
| single_timestamp | rejection | 3/3 |

## Holdout metrics by seed

These metrics are separate from behavioral pass counts. Lower RMSE is better;
higher ROC AUC and R² are better. A negative-control model may correctly lose to baseline.

| seed | case | metric | model | baseline | 95% bootstrap interval |
|---|---|---|---:|---:|---|
| 20261109 | classification_signal | roc_auc | 0.9062 | 0.5000 | [0.8638, 0.9437] |
| 20261109 | regression_signal | rmse | 0.3455 | 2.8656 | [0.3114, 0.3778] |
| 20261109 | temporal_cohorts | rmse | 0.3481 | 2.9051 | [0.3161, 0.3845] |
| 20261109 | group_holdout | rmse | 0.3150 | 2.7754 | [0.2878, 0.3458] |
| 20261109 | no_signal | r2 | -0.0010 | -0.0011 | [-0.0269, 0.0133] |
| 20261109 | temporal_shift | rmse | 5.8219 | 2.9955 | [5.3227, 6.3571] |
| 20261110 | classification_signal | roc_auc | 0.8828 | 0.5000 | [0.8300, 0.9309] |
| 20261110 | regression_signal | rmse | 0.3495 | 2.9660 | [0.3023, 0.3912] |
| 20261110 | temporal_cohorts | rmse | 0.3299 | 3.1979 | [0.2990, 0.3614] |
| 20261110 | group_holdout | rmse | 0.3376 | 3.1196 | [0.3044, 0.3762] |
| 20261110 | no_signal | r2 | 0.0056 | -0.0013 | [-0.0239, 0.0173] |
| 20261110 | temporal_shift | rmse | 6.3670 | 3.2069 | [5.7714, 6.9927] |
| 20261111 | classification_signal | roc_auc | 0.8710 | 0.5000 | [0.8127, 0.9176] |
| 20261111 | regression_signal | rmse | 0.3783 | 2.9645 | [0.3296, 0.4183] |
| 20261111 | temporal_cohorts | rmse | 0.3471 | 2.9615 | [0.3098, 0.3722] |
| 20261111 | group_holdout | rmse | 0.3639 | 2.8556 | [0.3256, 0.4136] |
| 20261111 | no_signal | r2 | -0.0110 | -0.0003 | [-0.0526, 0.0088] |
| 20261111 | temporal_shift | rmse | 6.0072 | 3.0957 | [5.5843, 6.4306] |

## Reproduction and provenance

```bash
python -m analyst predictive-benchmark --out-dir predictive_eval_results
```

No API keys, Docker, downloads, or paid services are used. Raw evidence, fixed
requests/tolerances, runtime versions, source hashes, and per-seed CSV content
hashes are in [results.json](results.json). Verdicts are recomputed from evidence;
missing/duplicate cases or execution errors fail the gate and CLI exit status.

Contract SHA-256: `025ce3641bd2e559435a0d2afcc926272a0dca1cca63ce58abe5e9518309b688`.

The three seeds are development seeds. Tolerances test known planted behavior;
they are not confidence bounds or estimates of external predictive accuracy.

Bootstrap intervals condition on the fitted model and resample holdout rows.
They do not account for temporal/entity dependence or model-selection uncertainty.
