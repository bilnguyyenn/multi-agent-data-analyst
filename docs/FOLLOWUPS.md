# Follow-up analyses

A completed run can be the starting point for a related question. Ask “break that
down by shift,” “adjust for job mix,” or “does that beat a naive baseline?” without
uploading the same tables or restating the entire previous analysis.

## Browser workflow

1. Open a completed analysis from history.
2. Enter a question under **Ask a follow-up**.
3. Select **Analyze follow-up**.

The application creates a separate run, inherits the source run's dataset and
execution/review options, and records its parent. The new answer displays its own
question and a link back to the source analysis. Historical answers and reports
remain unchanged. Downloaded HTML/JSON reports retain the bounded conversation
context and parent identifier.

A failed follow-up submission leaves the source answer visible. If the data have
changed, start a new independent analysis rather than relying on stale context.

## API

```text
POST /runs/{parent_run_id}/followups
Content-Type: application/json

{"question": "Break that down by shift"}
```

Success returns 202 with a new `run_id`; use the existing run and WebSocket
endpoints to follow its progress. The request accepts only `question`, stripped
of surrounding whitespace, with a maximum of 2,000 characters. It cannot override
the dataset directory, provider, sandbox, or review mode. A new independent run
can choose different execution options.

| Condition | Response |
|---|---|
| Source run not found | 404 |
| Source not completed or has no answer | 409 |
| Historical source has no dataset fingerprint | 409 |
| Source CSV files changed or are unavailable | 409 |
| Empty/oversized question or additional request fields | 422 |

## Data and context controls

The API compares the current CSV content hash with the source run's persisted
fingerprint before creating a job. This byte-level check avoids redoing the full
profiling pass in the request handler. The worker repeats the fingerprint check
before constructing a model client, accessing memory, or executing tasks. A
change while the job is queued fails the child visibly and makes no model call.
These checks do not lock files against mutation after execution has started; this
remains a trusted, single-user local application.

`followup-context-v1` carries up to three recent source turns and at most **6,000
serialized characters**, plus fixed prompt instructions. Each turn identifies its
source run and contains a bounded question, answer, and up to four validator
claims. Unresolved caveats/failures take priority over passing claims. Truncation
is explicit in stored context, prompts, and exported reports. Older turns can fall
out of the window; restate a filter or definition if it is no longer present.

Prior answers are JSON-encoded conversation data, not instructions or newly
validated evidence. The planner must produce self-contained executor goals; the
critic, council, adjudicator, and validator receive the same contextual question.
The existing validated-memory policy still applies. No old generated code,
stdout/stderr, event logs, provider options, or raw source rows enter the context.
Dataset-hash mismatches are also rejected for callers of the graph directly.

This is prompt-based reference handling, not a guarantee against every prompt
injection or a guarantee that a model understands every follow-up. Completion of
a source run does not imply all its claims passed. Explicit caveats are retained.

## Validation

Key-free tests execute follow-ups through the real graph and subprocess runner
using scripted model responses. They verify:

- new computations can correct a disputed source answer instead of copying it;
- context reaches the planner, critic, statistical validator, and council;
- dataset, review settings, and parent lineage survive a multi-turn chain;
- changed data are rejected before creating a child or accessing a model;
- context stays within its budget, prioritizes caveats, and labels truncation;
- independent-run prompt bytes and existing replay behavior remain intact;
- reports preserve lineage and escape source text.

These tests demonstrate the software contract. They are not a live-model benchmark
of conversational accuracy, and the original published accuracy scores have not
been changed to imply one.
