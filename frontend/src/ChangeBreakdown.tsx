import { useEffect, useState } from "react";
import ContributionInspection from "./ContributionInspection";

type ChangeRow = { segment: string; cohort: string; previous: string; current: string; delta: string; delta_base_units: number };
type Bridge = { computation_id: string; previous_job: string; current_job: string; metric: string; group_by: string;
  previous: string; current: string; delta: string; unit: string; rows: ChangeRow[]; group_count: number;
  remainder: { group_count: number; previous: string; current: string; delta: string };
  cohorts: { cohort: string; group_count: number; delta: string }[]; warnings: string[];
  reconciliation_residual_base_units: number };

export default function ChangeBreakdown({ previous, current }: { previous: string; current: string }) {
  const [metric, setMetric] = useState("net_sales");
  const [dimension, setDimension] = useState("product");
  const [result, setResult] = useState<Bridge | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [inspected, setInspected] = useState<string | null>(null);
  useEffect(() => { setResult(null); setError(""); setInspected(null); }, [previous, current, metric, dimension]);
  const matching = result?.previous_job === previous && result?.current_job === current
    && result?.metric === metric && result?.group_by === dimension ? result : null;
  const magnitude = Math.max(1, ...(matching?.rows.map(row => Math.abs(row.delta_base_units)) ?? []));

  async function explain() {
    setPending(true); setError(""); setResult(null); setInspected(null);
    try {
      const query = new URLSearchParams({ previous, current, metric, group_by: dimension, limit: "10" });
      const response = await fetch(`/operations/changes?${query}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Check the reviews and metric selection");
      setResult(payload);
    } catch (e) { setError(e instanceof Error ? e.message : "Change breakdown failed"); }
    finally { setPending(false); }
  }

  return <section className="ops-change"><h3>What contributed to the change?</h3>
    <p>Choose an additive metric included as an overall measure in both reviews. Select the earlier review first.</p>
    <div className="ops-change-controls"><label>Measure<select value={metric} disabled={pending} onChange={e => setMetric(e.target.value)}>
      {[["net_sales", "Net sales"], ["gross_sales", "Gross sales"], ["cancellation_value", "Cancellation value"], ["units_sold", "Units sold"]].map(([v, label]) => <option key={v} value={v}>{label}</option>)}
    </select></label><label>Break down by<select value={dimension} disabled={pending} onChange={e => setDimension(e.target.value)}>
      <option value="product">Product code</option><option value="country">Country</option></select></label>
      <button disabled={!previous || pending} onClick={() => void explain()}>{pending ? "Reconciling all groups…" : "Explain change"}</button></div>
    {error && <p role="alert" className="ops-error">{error}</p>}
    {matching && <div aria-live="polite"><h4>{matching.previous} → {matching.current} {matching.unit} · Change {matching.delta}</h4>
      <p>All {matching.group_count.toLocaleString()} groups included. Reconciliation residual: {matching.reconciliation_residual_base_units} base units.</p>
      <p>Showing the 10 largest absolute changes. Bars use the same scale; the remainder retains every other group.</p>
      <table><thead><tr><th>Group / observed cohort</th><th>Earlier</th><th>Current</th><th>Change ({matching.unit})</th></tr></thead><tbody>
        {matching.rows.map(row => <tr key={row.segment}><td><button type="button" className="ops-source-button" aria-label={`Inspect source for ${row.segment}`} onClick={() => setInspected(row.segment)}>{row.segment}</button><small className="ops-change-cohort">{row.cohort.replace(/_/g, " ")}</small></td>
          <td>{row.previous}</td><td>{row.current}</td><td>{row.delta}<span aria-hidden="true" className={`ops-change-bar ${row.delta_base_units < 0 ? "negative" : "positive"}`} style={{ width: `${Math.abs(row.delta_base_units) / magnitude * 100}%` }} /></td></tr>)}
        <tr className="ops-change-remainder"><td>All other groups ({matching.remainder.group_count})</td><td>{matching.remainder.previous}</td><td>{matching.remainder.current}</td><td>{matching.remainder.delta}</td></tr>
      </tbody></table>
      <details><summary>Changes by observed group cohort</summary>{matching.cohorts.map(cohort => <p key={cohort.cohort}>{cohort.cohort.replace(/_/g, " ")} · {cohort.group_count} groups · {cohort.delta} {matching.unit}</p>)}</details>
      {inspected !== null && <ContributionInspection parent={matching.computation_id} segment={inspected} onClose={() => setInspected(null)} />}
      {matching.warnings.map(w => <p key={w}>{w}</p>)}
      <a download href={`/operations/changes/${matching.computation_id}/report`}>Download change report</a> · <a href={`/operations/evidence/${matching.computation_id}`} target="_blank" rel="noreferrer">Inspect queries and complete evidence</a>
    </div>}
  </section>;
}
