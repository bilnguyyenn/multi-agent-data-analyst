# Versioned retail operations reviews

Install from a clone with `python -m pip install -e ".[retail]"` and Node.js/npm.
Start the local demo with `sh scripts/run_local_demo.sh`. Open the printed local
URL. The committed real weekly slices make the review workflow work without
Docker, credentials, model downloads, or paid services. The generic generated-code
agent retains its separate sandbox/provider requirements.

## Real data and definitions

Data: Chen, D. (2015), [UCI Online Retail](https://doi.org/10.24432/C5BW33), CC BY 4.0.
`data/retail/SOURCE.json` pins the archive, workbook, conversion, row counts, dates,
and all CSV hashes. The historical source has 541,909 rows. Convert it with:

```bash
python -m analyst retail-import --download
```

The converter streams the workbook, retains every source row in the full CSV,
and preserves duplicates and missing cells in the weekly slices. Original files
remain untouched. Public development/refresh weeks precede evaluation weeks.
This is a public temporal split, not a sealed dataset or pretraining-novelty claim.

Eligible metric lines have a valid invoice/time, integer-form quantity (zero-only decimal suffixes are permitted), positive unit
price, and a consistent sign: positive non-cancellation or negative cancellation.
Invoice prefixes are case-insensitive. Unit prices must be ordinary decimal literals with at most four decimal places.
Prices become integer pence using rounded
unit prices. Naive timestamps stay source-local; no timezone is inferred.

| Metric | Contract |
|---|---|
| gross_sales | Positive non-cancellation quantity × unit pence, summed and displayed in GBP |
| cancellation_value | Absolute negative cancellation line value in GBP; not verified physical returns |
| net_sales | Gross sales minus cancellation value; not profit, cash, or causal impact |
| order_count | Distinct non-cancellation invoices with eligible positive lines |
| units_sold | Positive units on eligible non-cancellation lines |
| active_customers | Distinct known customer IDs on eligible positive lines; missing IDs are excluded explicitly |
| cancellation_rate | Distinct cancellation invoices / all eligible distinct invoices; not original-order cohort probability |

Duplicates are retained and counted in diagnostics. Unexpected signs, nonpositive
prices, invalid fields, and missing IDs are reported. The browser shows definitions,
filters, periods, groupings, numerators, denominators, and source-row counts. Empty
rates are undefined; zero observed monetary/count totals do not establish that no
business activity occurred. Grouped distinct counts and rates are not additive.

## Repeatable workflow

1. Create an immutable snapshot from one retail CSV using its server path.
2. Save a playbook with chosen metrics, an optional exact country label, and grouping.
3. Select the snapshot and explicit start/exclusive-end dates, then run the review.
4. Inspect any computation, download its report/evidence, and register the next snapshot.
5. Refresh the same playbook over an equal-length, nonoverlapping period and compare.

After comparing, use **Explain change** to break an overall additive metric down
by country or product code. All groups are aggregated before ranking by absolute
change. Displayed rows plus the explicit remainder reconcile to the saved overall
metric in integer pence (or integer units). Cohorts describe whether eligible
records are present in each window; they do not establish acquisition or churn.
Stock codes can include fees and adjustments, so inspect source descriptions
before interpreting merchandise performance. Distinct invoices, customers, and
rates are excluded because they are not generally additive across these groups.

A plain comparison of top-N lists now marks an absent group as unknown when its
period was truncated. It cannot silently treat an omitted group as zero. The
complete contribution bridge reads the immutable snapshots to recover those
missing aggregates. It requires current engine fingerprints, matching definitions,
one overall metric request, compatible filters, and chronological periods. Each
query is interrupted after 30 seconds, uses one DuckDB thread and the existing
256 MB query setting, and rejects more than 100,000 groups per period. Query
results and hashes are persisted in the evidence ledger; portable HTML/JSON retain
reconciliation, parameters, source hashes, and caveats.

The default demo seeds two completed reviews. CLI equivalents:

```bash
python -m analyst retail-review --start 2011-11-01 --end 2011-11-08
python -m analyst retail-review --data data/retail/refresh/transactions.csv --start 2011-11-08 --end 2011-11-15 --out refresh.html
python -m analyst retail-compare PREVIOUS_JOB_ID CURRENT_JOB_ID
python -m analyst retail-changes PREVIOUS_JOB_ID CURRENT_JOB_ID --group-by product --out change.html
```

Snapshots are content-addressed read-only copies. Evidence includes full preparation
SQL, aggregate SQL, bound parameters, definition and engine hashes, source hash,
quality diagnostics, and deterministic computation IDs. HTML links resolve to
those entries. A reference demonstrates provenance, not correctness by itself.
Independent row-wise Decimal calculations supply the correctness reference.

SQLite transactions make submission idempotent and completion atomic. The same
refresh returns the same job, while reuse of a client key for a different payload
fails. Queued jobs survive restart; expired running leases can be recovered with
`python -m analyst retail-execute JOB_ID`. Ownership tokens prevent stale workers
from publishing duplicate completion. Failed/cancelled jobs support an explicit retry with retained attempt history
and a five-retry cap. Cancellation is cooperative between queries;
the current query can run until its time budget. The default is 60 seconds, maximum
300, with at most 16 metrics. A DuckDB interruption timer enforces the query deadline.
Snapshot and definition hashes are checked again before execution/publication.
Comparisons reject incompatible definitions and unequal or overlapping periods.

The engine uses one DuckDB thread with a 256 MB query-memory setting. This is not
a total-process limit; Python, CSV parsing, and library memory add overhead. See
the actual [resource measurements](../retail_results/runtime.json). Full-data CLI
processing is separate from the unchanged 20 MB browser upload limit.

## Optional language interpretation and evaluation

The non-model definition check asks for clarification when a metric is ambiguous
and marks profit, prediction, and causal questions unsupported. The optional local
model button proposes a typed tool request; it cannot supply SQL or numeric results.
A user confirms the request before saving/running it. No hosted provider is used.
Single, Standard, and High policies respectively use one proposal, one proposal
plus reviewer, or a proposal plus three independently called proposal reviews. In this
local lane calls are serialized to bound memory. It is distinct from generic
Python/SQL graph evaluation; the two results must not be conflated.

```bash
python -m analyst retail-eval
python -m analyst retail-eval --live-local --max-calls 1000
```

The frozen suite has 100 questions and 20 two-turn scenarios. Deterministic checks
measure metric/reference agreement and definition-guard behavior, not live language
accuracy. Local evaluation records every attempt/call before moving on, resumes
only the same contract/model/policy identity, preserves wrong answers and errors,
and reports tokens, p50/p95 latency, and zero dollar cost. Only two week clusters
are represented; precision/generalization claims would be inappropriate.

Resource preflight requires local service/model availability and memory headroom.
A blocked run exits 2 and records why; it does not produce a model-accuracy result.
The current execution environment blocked loading the installed model. No model
accuracy, production adoption, or user time saving is claimed from the new suite.

## Boundaries

This remains one trusted local workspace: no authentication or multi-tenant hosting.
The Compose API stores operations state in its existing `/state` volume and the
frontend proxies operations/dataset routes. The full container stack was not
launched during this memory-constrained run; local API/frontend behavior was tested.
Do not expose the API/Docker socket publicly. No live database connector, scheduler,
notification sending, fine-tuning, or additional infrastructure was needed for this
file-based pilot; those roadmap items were explicitly conditional. New data refreshes
are new analyses and never bypass the generic follow-up stale-data checks.
