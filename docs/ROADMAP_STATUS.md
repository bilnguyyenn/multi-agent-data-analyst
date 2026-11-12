# Roadmap execution status

Updated November 12, 2026. Target roles: AI/ML engineering and data science/analytics.
Spending limit: **$0**. The user explicitly chose **live evaluation later** and
**pilot testers later**. Neither outcome is represented as completed research.

| Milestone | Status | Evidence / remaining boundary |
|---|---|---|
| 0 · Repair the evidence trail | Complete | Seven development runs and two held-out runs reproduce from their published raw inputs with `python -m analyst verify-results`; CI checks this on a fresh checkout |
| 1 · Real-data vertical | Complete | Pinned UCI source, four complete weekly slices, seven metric definitions, explicit clarification/unsupported outcomes, independent Decimal reference, immutable snapshots, query-linked UI and reports |
| 2 · External/conversational evaluation | Engineering and deterministic validation complete; live run deferred | Frozen 100-question / 20-chain suite; 140/140 metric/oracle and definition-guard checks. Single/Standard/High local-agent policies, durable attempt/call recording, usage accounting, resume identity checks, and resource preflight are prepared. No live-model score is claimed |
| 3 · Repeatable workflow | Complete | Two real weekly reviews and a comparison; saved playbooks, definition fingerprints, SQLite transactions, idempotency, ownership leases, cancellation, retry budgets, restart and corruption tests; full 541,909-row resource run |
| 4 · User pilot / portfolio | Pilot prepared; participants deferred | Counterbalanced matched tasks, consent/alias protocol, facilitator gold, empty observations, non-synthetic scoring, case study, local demo, and a 90-second recorded-state walkthrough. No adoption or time-saving claim |

## Delivered entry points

The initial implementation is pinned by `operations-v1`. The November 12 extension
adds exact contribution analysis, eight independent real-data reconciliations, and
unknown-versus-zero protection for truncated comparisons. See the updated case study.

- `sh scripts/run_local_demo.sh` — local retail workspace with two completed reviews.
- [Workflow and contracts](OPERATIONS.md).
- [Recorded-state 90-second walkthrough](../examples/retail-walkthrough.html) and [narration storyboard](DEMO_SCRIPT.md).
- [Case study](RETAIL_CASE_STUDY.md), [real-data validation](../retail_results/README.md), and [resource measurements](../retail_results/runtime.json).
- [Pilot package](../pilot/README.md); observation data remain local and Git-ignored.
- [Role-specific resume wording](RESUME_WRITEUP.md).

## What remains by explicit user choice

1. Free the required local-model memory headroom and run the fixed live evaluation:
   `python -m analyst retail-eval --live-local --max-calls 1000`. Keep contract,
   model, source, and policy identity unchanged when resuming. Publish actual
   errors and wrong answers together with latency and usage; do not substitute
   deterministic checks or scripted tests for model performance.
2. Recruit 3–5 willing testers, conduct the prepared pilot, and analyze actual
   observations with `python -m analyst retail-pilot --analyze`. The current pilot
   summary correctly records zero participants and pending status.

The optional SQL benchmark, live database adapter, scheduling, external
notifications, fine-tuning, Kubernetes, and public arbitrary-code hosting were
conditional roadmap extensions. They were not needed for the working file-based
pilot and are not silently treated as prerequisites that have been implemented.

## Integrity boundaries

Historical accuracy numbers describe their historical saved runs. Exact original
runtime/source metadata absent from those records cannot be reconstructed with
certainty. The new metric ledger records explicit engine/definition/source hashes.
Public temporal evaluation slices are not sealed or guaranteed absent from model
pretraining. Local runtime/RSS measurements are single-process observations, not
production SLAs or generic-agent response-time claims.
