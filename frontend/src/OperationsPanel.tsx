import { useEffect, useState } from "react";

type Metric = { unit: string; definition: string };
type Snapshot = { snapshot_id: string; rows: number; observed_start: string; observed_end: string; metadata: { label?: string } };
type Book = { playbook_id: string; name: string };
type Packet = { computation_id: string; metric: string; definition: string; unit: string; sql: string;
  request: { country: string | null; group_by: string | null };
  parameters: unknown[]; snapshot_hash: string; warnings: string[]; quality: Record<string, number>;
  values: { segment: string; value: number | null; numerator: number; denominator: number | null; source_rows: number }[] };
type Job = { job_id: string; status: string; error?: string; duration_s?: number; process_peak_rss_mb?: number;
  spec: { snapshot_id: string; start: string; end: string }; evidence?: Packet[] };

async function api(path: string, body?: unknown) {
  const response = await fetch(`/operations/${path}`, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Check the supplied fields");
  return result;
}

export default function OperationsPanel({ onBack }: { onBack: () => void }) {
  const [registry, setRegistry] = useState<Record<string, Metric>>({});
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [books, setBooks] = useState<Book[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [snapshot, setSnapshot] = useState("");
  const [book, setBook] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [source, setSource] = useState("data/retail/development/transactions.csv");
  const [name, setName] = useState("Weekly retail review");
  const [country, setCountry] = useState("");
  const [grouping, setGrouping] = useState("");
  const [selected, setSelected] = useState(["gross_sales", "cancellation_value", "net_sales", "order_count", "cancellation_rate"]);
  const [start, setStart] = useState("2011-11-01");
  const [end, setEnd] = useState("2011-11-08");
  const [question, setQuestion] = useState("");
  const [clarification, setClarification] = useState("");
  const [proposal, setProposal] = useState<{ metric: string; country?: string | null; group_by?: string | null } | null>(null);
  const [previous, setPrevious] = useState("");
  const [comparison, setComparison] = useState<{ rows: { metric: string; segment: string; previous: number | null; current: number | null; delta: number | null }[]; warnings: string[] } | null>(null);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);

  const refresh = async () => {
    const [r, s, b, j] = await Promise.all([api("metrics"), api("snapshots"), api("playbooks"), api("jobs")]);
    setRegistry(r.metrics); setSnapshots(s); setBooks(b); setJobs(j);
  };
  useEffect(() => { void refresh().catch(e => setError(String(e.message))); }, []);
  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    const timer = setInterval(() => {
      void api(`jobs/${job.job_id}`).then(result => { setJob(result); return refresh(); })
        .catch(e => setError(String(e.message)));
    }, 1000);
    return () => clearInterval(timer);
  }, [job?.job_id, job?.status]);

  async function perform(action: () => Promise<void>) {
    setPending(true); setError("");
    try { await action(); await refresh(); } catch (e) { setError(e instanceof Error ? e.message : "Request failed"); }
    finally { setPending(false); }
  }

  return <main className="operations-page">
    <header><button onClick={onBack}>← Agent analyses</button><span>VERSIONED METRICS · REPEATABLE REVIEWS</span>
      <h1>Retail operations</h1><p>Define the measure, freeze the data, inspect the computation, then compare the next review.</p>
      <small>Review calculations need no model. Contributions describe changes, not their causes.</small></header>
    {error && <p role="alert" className="ops-error">{error}</p>}
    <div className="ops-setup">
      <section><h2>1 · Freeze a data snapshot</h2><label>Server CSV path<input value={source} onChange={e => setSource(e.target.value)} /></label>
        <button disabled={pending} onClick={() => void perform(async () => {
          const result = await api("snapshots", { source_path: source, label: source.split("/").slice(-2).join("/") }); setSnapshot(result.snapshot_id);
        })}>Create immutable snapshot</button>
        <label>Snapshot<select value={snapshot} onChange={e => {
          setSnapshot(e.target.value); const s = snapshots.find(s => s.snapshot_id === e.target.value);
          if (s?.observed_start) setStart(s.observed_start.slice(0, 10));
        }}><option value="">Select snapshot</option>{snapshots.map(s => <option key={s.snapshot_id} value={s.snapshot_id}>
          {s.metadata.label || s.snapshot_id.slice(0, 10)} · {s.rows.toLocaleString()} rows</option>)}</select></label>
        <small>Existing CSV upload directories can also be used. Raw source files are never modified.</small>
      </section>
      <section><h2>2 · Save metric definitions</h2><label>Playbook name<input value={name} onChange={e => setName(e.target.value)} /></label>
        <label>Country filter (exact source label, blank for all)<input value={country} maxLength={160} onChange={e => setCountry(e.target.value)} placeholder="United Kingdom" /></label>
        <label>Group results by<select value={grouping} onChange={e => setGrouping(e.target.value)}>
          <option value="">Overall</option><option value="country">Country</option><option value="product">Product</option><option value="customer">Customer</option></select></label>
        {Object.entries(registry).map(([id, spec]) => <label className="ops-metric-choice" key={id}>
          <input type="checkbox" checked={selected.includes(id)} onChange={e => setSelected(e.target.checked ? [...selected, id] : selected.filter(m => m !== id))} />
          <span><strong>{id.replace(/_/g, " ")}</strong> ({spec.unit})<small>{spec.definition}</small></span></label>)}
        <button disabled={pending || !selected.length} onClick={() => void perform(async () => {
          const result = await api("playbooks", { name, metrics: selected.map(metric => ({ metric, country: country.trim() || null, group_by: grouping || null, limit: 100 })) }); setBook(result.playbook_id);
        })}>Save review playbook</button>
      </section>
      <section><h2>3 · Run or refresh</h2><label>Playbook<select value={book} onChange={e => setBook(e.target.value)}><option value="">Select playbook</option>
        {books.map(b => <option key={b.playbook_id} value={b.playbook_id}>{b.name}</option>)}</select></label>
        <label>Period start<input type="date" value={start} onChange={e => setStart(e.target.value)} /></label>
        <label>Period end (exclusive)<input type="date" value={end} onChange={e => setEnd(e.target.value)} /></label>
        <button disabled={pending || !snapshot || !book} onClick={() => void perform(async () => {
          const created = await api("jobs", { playbook_id: book, snapshot_id: snapshot, start, end });
          setJob(await api(`jobs/${created.job_id}`)); setComparison(null);
        })}>Run review</button>
        <label>Question about a metric<input value={question} onChange={e => setQuestion(e.target.value)} placeholder="What happened to revenue?" /></label>
        <button disabled={!question || pending} onClick={() => void perform(async () => {
          const result = await api("resolve", { question }); setClarification(result.reason || `Use ${result.metric}: ${registry[result.metric]?.definition}`);
        })}>Check the definition</button>
        {clarification && <p className="ops-notice">{clarification}</p>}
        <button disabled={!question || pending} onClick={() => void perform(async () => {
          const result = await api("interpret", { question, prior_request: proposal });
          setProposal(result.status === "compute" ? result.request : null);
          setClarification(result.status === "compute" ? `Proposed request: ${JSON.stringify(result.request)}. Confirm its definition before running.` : `Local model requests ${result.status}. Select a precise metric or restate the question.`);
        })}>Interpret with local model ($0)</button>
        <small>Optional. Memory preflight runs first. Models propose typed requests; they cannot run SQL or supply numeric evidence.</small>
        {proposal && <button disabled={pending} onClick={() => void perform(async () => {
          const result = await api("playbooks", { name: "Confirmed interpreted review", metrics: [proposal] }); setBook(result.playbook_id);
        })}>Confirm interpreted metric</button>}
      </section>
    </div>
    <section className="ops-history"><h2>Review history</h2>{jobs.map(j => <button key={j.job_id} onClick={() => void perform(async () => {
      setJob(await api(`jobs/${j.job_id}`)); setComparison(null);
    })}>{j.spec.start} → {j.spec.end} · {j.status} · {j.job_id.slice(0, 8)}</button>)}</section>
    {job && <section className="ops-results"><header><h2>Review {job.job_id.slice(0, 8)} · {job.status}</h2>
      {job.status === "done" && <div><a download href={`/operations/jobs/${job.job_id}/report`}>Download report</a> · <a download href={`/operations/jobs/${job.job_id}/report?format=json`}>JSON evidence</a></div>}
      {["queued", "running"].includes(job.status) && <button onClick={() => void perform(async () => { setJob(await api(`jobs/${job.job_id}/cancel`, {})); })}>Cancel review</button>}</header>
      {job.error && <p role="alert">{job.error}</p>}
      {["error", "cancelled"].includes(job.status) && <button disabled={pending} onClick={() => void perform(async () => { setJob(await api(`jobs/${job.job_id}/retry`, {})); })}>Retry review</button>}
      {job.evidence?.map(packet => <article key={packet.computation_id}><h3>{packet.metric.replace(/_/g, " ")} <small>{packet.unit}</small></h3>
        <p>{packet.definition}</p><p>Period: {job.spec.start} → {job.spec.end} (exclusive) · Country: {packet.request.country || "All"} · Grouping: {packet.request.group_by || "Overall"}</p><table><thead><tr><th>Segment</th><th>Value</th><th>Numerator</th><th>Denominator</th><th>Rows</th></tr></thead>
          <tbody>{packet.values.map(row => <tr key={row.segment}><td>{row.segment}</td><td>{row.value ?? "Undefined"}</td><td>{row.numerator}</td><td>{row.denominator ?? "—"}</td><td>{row.source_rows}</td></tr>)}</tbody></table>
        <details><summary>Inspect this computation · {packet.computation_id.slice(0, 12)}</summary>
          <p>Snapshot SHA-256: {packet.snapshot_hash}</p><pre>{packet.sql}</pre><p>Parameters: {JSON.stringify(packet.parameters)}</p>
          <p>Snapshot diagnostics: {JSON.stringify(packet.quality)}</p><ul>{packet.warnings.map(w => <li key={w}>{w}</li>)}</ul>
          <a href={`/operations/evidence/${packet.computation_id}`} target="_blank" rel="noreferrer">Open complete ledger entry</a></details></article>)}
      {job.status === "done" && <div className="ops-comparison"><h3>Compare an earlier review</h3><select aria-label="Previous review" value={previous} onChange={e => setPrevious(e.target.value)}>
        <option value="">Select completed review</option>{jobs.filter(j => j.status === "done" && j.job_id !== job.job_id).map(j => <option key={j.job_id} value={j.job_id}>{j.spec.start} → {j.spec.end}</option>)}</select>
        <button disabled={!previous || pending} onClick={() => void perform(async () => {
          setComparison(await api(`compare?previous=${encodeURIComponent(previous)}&current=${job.job_id}`));
        })}>Compare periods</button>
        {comparison && <><table><thead><tr><th>Metric / segment</th><th>Earlier</th><th>Current</th><th>Change</th></tr></thead><tbody>
          {comparison.rows.map((r, i) => <tr key={i}><td>{r.metric} / {r.segment}</td><td>{r.previous ?? "Undefined"}</td><td>{r.current ?? "Undefined"}</td><td>{r.delta ?? "Undefined"}</td></tr>)}</tbody></table>
          {comparison.warnings.map(w => <p key={w}>{w}</p>)}</>}
      </div>}
    </section>}
  </main>;
}
