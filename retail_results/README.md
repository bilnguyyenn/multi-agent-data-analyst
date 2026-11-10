# Retail operations evidence

**140/140 deterministic checks passed** on public UCI retail slices:
100 fixed questions and 20 two-turn scenarios. This is metric-engine/independent
Decimal-oracle agreement and definition-guard behavior, **not live-model accuracy**.

- [Frozen task contract](contract.json), [all check outcomes](contract-results.json)
- [First review](development.html), [refresh](refresh.html), [comparison](comparison.json)
- [Recorded runtime and process memory](runtime.json)
- [Local-model preflight](local-preflight.json)

The reports link every displayed metric to its stored query, transformations,
parameters, definition fingerprint, source hash, and aggregate rows. Raw JSON
packets sit beside each HTML file. The full source conversion is reproduced from
the attributed, hash-pinned UCI download; public weekly slices are in data/retail.

Local-agent evaluation status: **pending**. No model accuracy is inferred from
the deterministic checks. Last preflight: ["Insufficient memory headroom: 1.31 GiB available, 2.50 GiB required"].
User pilot status: **participants deferred by the user**; no time-saving claim.

Rebuild: `python scripts/build_operations_evidence.py`.
Run the guarded free live evaluation later: `python -m analyst retail-eval --live-local`.
Do not tune on the final task outcomes or describe this public temporal split as
sealed or guaranteed absent from model pretraining.
