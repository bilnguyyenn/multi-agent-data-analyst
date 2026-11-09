# Portable analysis reports

Completed analyses can be exported as a self-contained HTML report or a versioned
JSON evidence packet. This turns a transient browser answer into a document that
can be reviewed offline, printed, or used in a portfolio demonstration.

## Browser

Open a completed run in history. The answer heading offers **Download report**
and **JSON evidence**. HTML opens in a browser without scripts, external assets,
or a running Analyst server. Use the browser's Print command to save a PDF.
Queued, running, or errored runs cannot be exported as completed reports.

The results panel scrolls independently of the sidebar. Its task graph has a
bounded height so graph nodes cannot overlap evidence or download controls when
a run contains several evidence panels.

## CLI

Export a completed API run JSON, a newly saved `.analyst_runs/last_run.json`, or
the output of `ask --json`:

```bash
python -m analyst export-report .analyst_runs/last_run.json --out report.html
python -m analyst export-report .analyst_runs/last_run.json --out evidence.json
```

The output suffix chooses HTML or JSON. `--format html` / `--format json` can
override it. Exporting does not rerun the analysis or contact a model provider.
Invalid input returns exit status 1 and leaves an existing output untouched.
Older CLI snapshots missing completion status need a new complete run; historical
API runs remain exportable, with unavailable provenance labeled **Not recorded**.

## API

```text
GET /runs/{run_id}/report              HTML attachment
GET /runs/{run_id}/report?format=json  JSON attachment
```

Unknown runs return 404, incomplete/errored runs return 409, and unsupported
formats return 422. Both exports use `Content-Disposition: attachment`,
`X-Content-Type-Options: nosniff`, and a restrictive Content Security Policy.

## Evidence contract

`analysis-report-v1` contains:

- question, answer, run identifier, completion status, and available timestamps;
- task goals, dependencies, statuses, attempts/revisions, and memory-source links;
- validator claims, including caveats and failures;
- statistical intervals, effects, tests, warnings, and stratification diagnostics;
- predictive holdout/baseline metrics, CV candidates, bootstrap intervals,
  permutation importance, subgroup slices, and split diagnostics;
- data-quality readiness, issue counts, findings, and recommendations;
- council conclusions, risks, adjudication rationale, and dissent;
- dataset content hash, table names, row counts, columns, and dtypes.

An explicit field list excludes provider options, local CSV paths, generated code,
stdout/stderr, event logs, raw uploaded rows, and profiling sample frequencies.
Analysis text and schema names remain part of the report. The JSON packet contains
the exported evidence in full; HTML presents the main results and includes an
expandable complete evidence packet. An empty claim list is labeled as having no
recorded claims; completion never implies validator approval.

The API/worker persists the dataset hash and schema when execution starts. Export
uses that snapshot rather than rereading current source files. CLI snapshots now
also persist claims and statistical evidence, which were missing from the previous
human-readable CLI save path.

Every dynamic HTML value is escaped, including questions, model output, labels,
warnings, and JSON evidence. Markdown HTML and links in generated answers are
displayed as text. The report has no JavaScript or remote resources.

## Key-free portfolio demonstration

The repository includes [a sample HTML report](../examples/analysis-report.html)
and [its input record](../examples/report_run.json). Download the HTML from GitHub
and open it locally, or export it yourself:

```bash
python -m analyst export-report examples/report_run.json --out report.html
```

To rebuild the example from the published predictive benchmark and regenerated,
hash-verified CSVs:

```bash
python scripts/build_report_demo.py
```

The conversion and temporal-shift model evidence comes from the published
`20261109` benchmark seed. The Wilson conversion interval is computed from the
same regenerated observations. The question, narrative, and task labels are
illustrative; no LLM or council review was run. The example states this explicitly.
It is a report demonstration, not an end-to-end agent benchmark result.
