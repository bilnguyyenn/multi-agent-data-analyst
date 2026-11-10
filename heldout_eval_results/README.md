# Held-out statistical evaluation

Two historical, complete, zero-error 20-question runs are published with their raw
inputs: 40 question instances, overall component-rubric score 0.719. The two-seed
uncertainty interval is very wide; this is early evidence on generated data, not
external real-world accuracy. See [SCORECARD.md](SCORECARD.md).

Reproduce both historical scorecards without a model or network:

```bash
python -m analyst verify-results
```

These July 2026 results describe the evaluated historical implementation, not
subsequent report, follow-up, or operations-workflow changes.
