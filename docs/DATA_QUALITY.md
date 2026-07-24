# Automatic EDA and data-quality report

Every dataset is profiled deterministically when its CSV manifest is built. The
same CSV bytes therefore produce the same report in browser uploads, CLI runs,
workers, persisted run history, and replayed WebSocket sessions. No LLM call or
generated code is required.

## What the profiler computes

For every table:

- exact row and column counts, fully duplicated rows, and identifier-like
  candidate keys;
- per-column physical and semantic types, non-null/distinct counts, and missing
  rates;
- numeric min, quartiles, median, mean, max, standard deviation, skew, and 1.5×IQR
  outlier counts;
- datetime parse success and observed date range;
- top-value frequencies for categorical and boolean columns.

Deterministic quality rules flag:

- all-missing, highly missing, and moderately missing columns;
- invalid likely primary keys, duplicate rows, and infinite numeric values;
- leading/trailing whitespace, case-normalization label collisions, and mixed
  numeric/text encodings;
- constant columns, high-cardinality text, highly skewed distributions, and IQR
  outliers;
- orphan foreign keys for inferred parent-child relationships. A shared `*_id`
  column is treated as a relationship only when exactly one best parent table has
  a complete unique key; ambiguous relationships are not guessed.

Each finding includes severity, location, affected count/rate, an explanation, and
a remediation. Errors and warnings are sorted before informational EDA signals so
the report remains actionable on wide datasets.

## Readiness score

The 0–100 score is a triage heuristic:

```text
100
- (8 + 12 × affected rate) for each error
- (2 +  6 × affected rate) for each warning
```

Informational distribution signals do not lower the score. The formula is returned
in `score_method` and shown as a UI tooltip; it is deliberately transparent rather
than presented as a learned or scientific measure of fitness.

The score never authorizes automatic deletion, deduplication, or imputation. For
example, a mostly empty decommission date can be valid structural missingness. The
report tells the analyst to define the meaning before choosing a treatment.

## Runtime and API flow

`build_manifest` loads each CSV once, uses the loaded frames for both schema
inference and profiling, then embeds the typed `DataQualityReport` in the
`DatasetManifest`. The compact summary and highest-priority findings are appended
to `schema_text`, so planners and data-quality specialists see known risks before
writing analysis code.

The full report is:

- returned immediately by `POST /datasets` as `data_quality_report`;
- streamed on `node_update` and `answer` WebSocket events;
- stored with JSON and Postgres run records for late replay;
- rendered in the React UI with issue cards, expandable column profiles, and
  inferred relationship checks;
- included in CLI JSON output and `.analyst_runs/last_run.json`.

## Boundaries

- Current profiling performs an exact full-file scan. This is appropriate for the
  local portfolio product and its 20 MB browser-upload limit; a production data
  lake version should use chunked or warehouse-native aggregations.
- Semantic typing is rule-based. Domain constraints such as allowed ranges,
  required columns, units, and business-specific uniqueness need an explicit data
  contract.
- Relationship inference uses shared `*_id` names and unique parent candidates. It
  does not infer fuzzy keys or silently choose an ambiguous parent.
- Outliers and skew are descriptive signals, not proof of bad records.
- Categorical top values may be sensitive in a multi-user deployment; add column
  classification/redaction before exposing profiles outside the current local,
  single-user scope.
