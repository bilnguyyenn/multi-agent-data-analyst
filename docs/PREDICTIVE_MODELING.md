# Governed predictive modeling

## Purpose

Predictive questions use a deterministic modeling layer instead of trusting an
executor to fit a model and report its own training score. The executor emits only
a typed specification: source tables, validated joins, target, features, split
policy, candidate models, metric, and optional slices. Application code owns all
training and evaluation.

This design makes the result reproducible and gives the statistical validator a
consistent evidence packet to audit for leakage, target definition, feature timing,
generalization, and practical usefulness.

## Request contract

```json
{
  "modeling_requests": [{
    "id": "conversion_model",
    "task_type": "classification",
    "table": "experiment_outcomes",
    "target": "converted",
    "features": ["variant", "device", "plan", "region"],
    "joins": [
      {"table": "users", "on": ["user_id"], "validate": "one_to_one"}
    ],
    "split_strategy": "random",
    "test_size": 0.2,
    "models": ["logistic_regression", "random_forest"],
    "primary_metric": "roc_auc",
    "positive_label": 1,
    "slice_by": ["device"],
    "random_seed": 42,
    "cv_folds": 5,
    "bootstrap_iterations": 300
  }]
}
```

Supported tasks and models:

| task | candidates | primary metrics |
|---|---|---|
| binary classification | logistic regression, random forest | ROC AUC, F1, balanced accuracy |
| regression | ridge regression, random forest | RMSE, MAE, R² |

If `models` is omitted, both candidates for that task are compared. Candidate
selection uses only cross-validation inside the training partition. The chosen
pipeline is then fitted on all training rows and evaluated once on the untouched
holdout.

## Splits and leakage controls

- `random`: stratified for classification; deterministic seed.
- `temporal`: sorts by `time_column`, trains on the past, tests on the latest rows,
  and records the train/test time boundary. Training CV uses forward time splits.
- `group`: keeps every `group_column` value entirely in train or test and records a
  zero-overlap diagnostic. Training CV also respects groups.

The engine rejects:

- target columns reused as features;
- identifier-like features (`id`/`*_id`) unless explicitly overridden;
- unknown columns/tables, duplicate feature names, and all-missing features;
- temporal/group policies without their required column;
- one-to-many or many-to-many joins and any join that increases row count;
- ambiguous overlapping non-key columns across joins;
- fewer than 50 complete target rows, non-binary classification targets, or a
  train/holdout classification split missing either class;
- more than 100,000 rows or 50 input features in this local product version.

The identifier override is deliberately visible because some legitimate entity
embeddings may need it, but defaulting to rejection prevents easy row memorization.
The validator separately checks semantic leakage that code cannot infer from names,
such as post-outcome revenue being used to predict conversion.

## Preprocessing and evaluation

Each candidate is a scikit-learn `Pipeline`:

- numeric columns: median imputation and standardization;
- categorical columns: most-frequent imputation and unknown-safe one-hot encoding;
- fixed random seeds and single-process model fitting for reproducibility.

Classification reports ROC AUC, accuracy, balanced accuracy, precision, recall,
F1, and log loss. Regression reports RMSE, MAE, and R². Every metric is paired with
the corresponding naive dummy-classifier/regressor baseline. The primary holdout
metric receives a deterministic percentile bootstrap 95% interval. Warnings surface
failure to beat baseline, class imbalance, missing-target row removal, and material
train/holdout gaps. A classifier that predicts only one class at the default 0.5
threshold is flagged; any threshold tuning must use training/validation data rather
than the reported holdout.

Model-agnostic permutation importance is computed on the holdout at the original
feature level. Requested `slice_by` columns receive holdout primary metrics, with
small or single-class slices marked undefined instead of silently omitted.

## Evidence flow

Typed `PredictiveEvidence` includes:

- split policy, train/test sizes, target and features;
- every candidate's training-CV mean/standard deviation and selected model;
- holdout and baseline metrics, bootstrap interval, and train/holdout diagnostics;
- permutation importance and requested slice results;
- invalid-request or modeling warnings.

It is passed to the statistical validator, appended to the deterministic final
answer, persisted in JSON/Postgres run records, streamed/replayed over WebSockets,
included in CLI JSON, and rendered in a dedicated React panel.

## Honest boundaries

- This evaluates models; it does not yet serialize a production model artifact or
  serve online predictions.
- Classification is binary only. Multiclass, multilabel, survival, forecasting,
  and ranking objectives are separate future increments.
- Hyperparameter search is intentionally small and fixed. A larger search needs a
  nested-CV or locked validation design, not repeated holdout peeking.
- Bootstrap intervals condition on the fitted model and observed holdout. They do
  not capture every source of model-development uncertainty.
- Permutation importance is predictive, not causal, and can divide importance among
  correlated features.
- Slice metrics are diagnostics, not a complete fairness audit. Production use
  requires protected-attribute governance, thresholds, calibration, drift checks,
  model cards, and monitoring.
- Random splits are inappropriate for future prediction or repeated entities unless
  the request selects temporal/group splitting. The validator must reject a split
  that does not match deployment.
