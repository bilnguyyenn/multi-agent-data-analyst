# Retail operations evidence

**140/140 deterministic checks passed** on public UCI retail slices:
100 fixed questions and 20 two-turn scenarios. This is metric-engine/independent
Decimal-oracle agreement and definition-guard behavior, **not live-model accuracy**.

- [Frozen task contract](contract.json), [all check outcomes](contract-results.json)
- [First review](development.html), [refresh](refresh.html), [comparison](comparison.json)
- [Exact product-code breakdown](change-product.html), [country breakdown](change-country.html)
- [8/8 independent change reconciliations](change-validation.json): every aggregate group checked;
  2,619 product codes / 18 countries; zero-pence residuals. HTML/JSON reports retain all omitted
  groups in an explicit remainder. These checks are separate from the frozen 140-check suite.
- [Source inspection of fee entries](inspection-amazonfee.html), [JSON and parent](inspection-amazonfee.json)
- [6/6 independent source inspections](inspection-validation.json): component totals, source counts,
  sampled-record membership/multiplicity, and exact signed pence for both periods.
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

Rebuild the contribution evidence: `python scripts/build_change_evidence.py`.

Rebuild source-inspection evidence: `python scripts/build_inspection_evidence.py`.
