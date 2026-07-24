# Held-out evaluation results

This directory is reserved for the independent 20-question statistical benchmark.
No live score is committed yet. `SCORECARD.md` is created only after the integrity
gate finds at least one complete, zero-error run.

Run:

```bash
python -m analyst heldout-eval --provider hosted --configs no-critic \
  --seeds 20260717,20260718
```

Generated CSVs live under ignored `.heldout_data/seed_<seed>/`. Raw attempts remain
available locally for diagnosis. When publishing a score, explicitly unignore only
the complete raw files listed in `SCORECARD.md` provenance.
