import { useEffect, useRef, useState } from "react";

type Period = { period: { start: string; end: string; country: string | null }; snapshot_hash: string;
  totals: { eligible_rows: number; gross: string; cancellation: string; net: string; units_sold: number; units_sold_display: string };
  sample: { source: Record<string, string | null>; signed_value: string }[];
  sampled_rows: number; omitted_rows: number;
  daily: { day: string; eligible_rows: number; gross_pence: number; cancellation_pence: number; net_pence: number; gross: string; cancellation: string; net: string }[] };
type Inspection = { computation_id: string; parent_computation_id: string; segment: string; periods: Period[];
  contributor: { delta: string }; metric: string; unit: string; warnings: string[]; reconciliation_residual_base_units: number };

export default function ContributionInspection({ parent, segment, onClose }: { parent: string; segment: string; onClose: () => void }) {
  const root = useRef<HTMLElement>(null);
  const [result, setResult] = useState<Inspection | null>(null);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    setResult(null); setError(""); setPending(true);
    void (async () => {
      try {
        const query = new URLSearchParams({ segment, sample_limit: "10" });
        const response = await fetch(`/operations/changes/${parent}/inspect?${query}`, { signal: controller.signal });
        const payload = await response.json();
        if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Source inspection failed");
        if (!controller.signal.aborted) setResult(payload);
      } catch (e) {
        if (!controller.signal.aborted) setError(e instanceof Error ? e.message : "Source inspection failed");
      } finally { if (!controller.signal.aborted) setPending(false); }
    })();
    return () => controller.abort();
  }, [parent, segment]);
  const matching = result?.parent_computation_id === parent && result?.segment === segment ? result : null;
  useEffect(() => {
    if (matching || error) root.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [matching?.computation_id, error]);
  const fields = ["InvoiceNo", "InvoiceDate", "Description", "Quantity", "UnitPrice", "Country"];
  return <section ref={root} className="ops-inspection" aria-live="polite"><button type="button" onClick={onClose}>Close source inspection</button><h4>Source inspection · {segment}</h4>
    {pending && <p>Checking matching records against the saved contribution…</p>}
    {error && <p role="alert" className="ops-error">{error}</p>}
    {matching && <><p>{matching.metric.replace(/_/g, " ")} change: <strong>{matching.contributor.delta} {matching.unit}</strong> · Reconciliation residual: {matching.reconciliation_residual_base_units} base units.</p>
      <p>Full eligible totals are calculated before sampling. The first matching records are shown in deterministic order.</p>
      {matching.periods.map((period, index) => <article key={index}><h5>{index === 0 ? "Earlier" : "Current"}: {period.period.start} → {period.period.end} (exclusive)</h5>
        <p>Country scope: {period.period.country ?? "All"}</p>
        <p>Gross {period.totals.gross} GBP · Cancellations {period.totals.cancellation} GBP · Net {period.totals.net} GBP · Units sold {period.totals.units_sold_display}</p>
        <p>Showing {period.sampled_rows} of {period.totals.eligible_rows.toLocaleString()} eligible rows; {period.omitted_rows.toLocaleString()} omitted.</p>
        {period.sample.length ? <div className="ops-source-scroll"><table><thead><tr>{fields.map(f => <th key={f}>{f}</th>)}<th>Signed GBP</th></tr></thead><tbody>
          {period.sample.map((row, i) => <tr key={i}>{fields.map(f => <td key={f}>{row.source[f] ?? "(missing)"}</td>)}<td>{row.signed_value}</td></tr>)}
        </tbody></table></div> : <p>No eligible source rows in this window.</p>}
        <details><summary>Daily totals and snapshot reference</summary><p>Snapshot SHA-256: {period.snapshot_hash}</p>
          <p>Daily amounts below are GBP. Unlisted dates are not imputed observations.</p>
          <div className="ops-source-scroll"><table><thead><tr><th>Observed date</th><th>Rows</th><th>Gross GBP</th><th>Cancellation GBP</th><th>Net GBP</th></tr></thead><tbody>
            {period.daily.map(day => <tr key={day.day}><td>{day.day}</td><td>{day.eligible_rows}</td><td>{day.gross}</td><td>{day.cancellation}</td><td>{day.net}</td></tr>)}
          </tbody></table></div></details>
      </article>)}
      {matching.warnings.map(w => <p key={w}>{w}</p>)}
      <a download href={`/operations/inspections/${matching.computation_id}/report`}>Download source inspection</a> · <a target="_blank" rel="noreferrer" href={`/operations/evidence/${matching.computation_id}`}>Inspect complete source evidence</a>
    </>}
  </section>;
}
