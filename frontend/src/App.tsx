import { useCallback, useEffect, useRef, useState } from "react";
import DagView from "./DagView";
import type {
  AgentEvent,
  Claim,
  CouncilDecision,
  CouncilFinding,
  CouncilRoutingDecision,
  DataQualityReport,
  PredictiveEvidence,
  RunSummary,
  StatisticalEvidence,
  TaskInfo,
  WsMessage,
} from "./types";

function percent(value: number): string {
  return `${(value * 100).toFixed(value > 0 && value < 0.01 ? 2 : 1)}%`;
}

function eventLine(e: AgentEvent): string {
  const t = new Date((e.ts as number) * 1000).toLocaleTimeString();
  const rest = Object.entries(e)
    .filter(([k]) => k !== "ts" && k !== "node")
    .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`)
    .join(" ");
  return `${t} [${e.node}] ${rest}`;
}

export default function App() {
  const [question, setQuestion] = useState("Which shift has the best quality record?");
  const [dataDir, setDataDir] = useState("data/sample");
  const [files, setFiles] = useState<File[]>([]);
  const [datasetLabel, setDatasetLabel] = useState("sample manufacturing data");
  const [councilMode, setCouncilMode] = useState<"adaptive" | "always" | "off">("off");
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [activeRun, setActiveRun] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("idle");
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [answer, setAnswer] = useState<string | null>(null);
  const [claims, setClaims] = useState<Claim[]>([]);
  const [statisticalEvidence, setStatisticalEvidence] = useState<StatisticalEvidence[]>([]);
  const [predictiveEvidence, setPredictiveEvidence] = useState<PredictiveEvidence[]>([]);
  const [dataQualityReport, setDataQualityReport] = useState<DataQualityReport | null>(null);
  const [councilFindings, setCouncilFindings] = useState<CouncilFinding[]>([]);
  const [councilDecisions, setCouncilDecisions] = useState<CouncilDecision[]>([]);
  const [councilRouting, setCouncilRouting] = useState<CouncilRoutingDecision | null>(null);
  const ws = useRef<WebSocket | null>(null);

  const refreshRuns = useCallback(() => {
    fetch("/runs").then((r) => r.json()).then(setRuns).catch(() => {});
  }, []);
  useEffect(refreshRuns, [refreshRuns]);

  const openStream = useCallback((runId: string, initialReport: DataQualityReport | null = null) => {
    ws.current?.close();
    setActiveRun(runId);
    setTasks([]); setEvents([]); setAnswer(null); setClaims([]); setStatisticalEvidence([]);
    setPredictiveEvidence([]);
    setDataQualityReport(initialReport);
    setCouncilFindings([]); setCouncilDecisions([]);
    setCouncilRouting(null);
    setStatus("connecting");

    const proto = location.protocol === "https:" ? "wss" : "ws";
    const sock = new WebSocket(`${proto}://${location.host}/runs/${runId}/stream`);
    ws.current = sock;

    sock.onmessage = (raw) => {
      const msg: WsMessage = JSON.parse(raw.data);
      switch (msg.type) {
        case "replay": {
          const r = msg.run;
          setStatus(r.status);
          setTasks(Object.values(r.dag ?? {}));
          setEvents(r.events ?? []);
          setAnswer(r.answer);
          setClaims(r.claims ?? []);
          setStatisticalEvidence(r.statistical_evidence ?? []);
          setPredictiveEvidence(r.predictive_evidence ?? []);
          setDataQualityReport(r.data_quality_report ?? initialReport);
          setCouncilFindings(r.council_findings ?? []);
          setCouncilDecisions(r.council_decisions ?? []);
          setCouncilRouting(r.council_routing ?? null);
          break;
        }
        case "status":
          setStatus(msg.status);
          break;
        case "node_update":
          setTasks(msg.tasks);
          setEvents((prev) => [...prev, ...msg.new_events]);
          setCouncilFindings(msg.council_findings ?? []);
          setCouncilDecisions(msg.council_decisions ?? []);
          setCouncilRouting(msg.council_routing ?? null);
          setStatisticalEvidence(msg.statistical_evidence ?? []);
          setPredictiveEvidence(msg.predictive_evidence ?? []);
          setDataQualityReport((current) => msg.data_quality_report ?? current);
          break;
        case "answer":
          setAnswer(msg.answer);
          setClaims(msg.claims);
          setStatisticalEvidence(msg.statistical_evidence ?? []);
          setPredictiveEvidence(msg.predictive_evidence ?? []);
          setDataQualityReport((current) => msg.data_quality_report ?? current);
          setCouncilFindings(msg.council_findings ?? []);
          setCouncilDecisions(msg.council_decisions ?? []);
          setCouncilRouting(msg.council_routing ?? null);
          break;
        case "stream_end":
          sock.close();
          refreshRuns();
          break;
      }
    };
    sock.onerror = () => setStatus("stream error");
  }, [refreshRuns]);

  const submit = useCallback(async () => {
    try {
      setStatus(files.length ? "uploading dataset" : "submitting");
      let selectedDataDir = dataDir;
      let uploadedReport: DataQualityReport | null = null;
      if (files.length) {
        const upload = await fetch("/datasets", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            files: await Promise.all(
              files.map(async (file) => ({ name: file.name, content: await file.text() })),
            ),
          }),
        });
        const payload = await upload.json();
        if (!upload.ok) throw new Error(payload.detail ?? "dataset upload failed");
        selectedDataDir = payload.data_dir;
        uploadedReport = payload.data_quality_report ?? null;
        setDataDir(payload.data_dir);
        setDatasetLabel(
          `${payload.tables.length} table${payload.tables.length === 1 ? "" : "s"} uploaded`,
        );
      }

      const resp = await fetch("/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          data_dir: selectedDataDir,
          council_mode: councilMode,
        }),
      });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload.detail ?? "run could not be created");
      openStream(payload.run_id, uploadedReport);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "request failed");
    }
  }, [question, dataDir, files, councilMode, openStream]);

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>analyst</h1>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={3}
        />
        <label className="dataset-picker">
          <span>CSV tables</span>
          <input
            type="file"
            accept=".csv,text/csv"
            multiple
            onChange={(e) => {
              const selected = Array.from(e.target.files ?? []);
              setFiles(selected);
              setDatasetLabel(
                selected.length
                  ? `${selected.length} file${selected.length === 1 ? "" : "s"} selected`
                  : "sample manufacturing data",
              );
            }}
          />
        </label>
        <small className="dataset-label">{datasetLabel}</small>
        <label className="council-mode">
          <span>Review mode</span>
          <select
            value={councilMode}
            onChange={(e) => setCouncilMode(e.target.value as "adaptive" | "always" | "off")}
          >
            <option value="off">Standard (recommended)</option>
            <option value="always">High assurance</option>
            <option value="adaptive">Adaptive (experimental)</option>
          </select>
        </label>
        <details>
          <summary>advanced: server data folder</summary>
          <input
            aria-label="server data folder"
            value={dataDir}
            onChange={(e) => {
              setDataDir(e.target.value);
              setFiles([]);
              setDatasetLabel("server folder");
            }}
          />
        </details>
        <button onClick={submit} disabled={status === "running"}>
          {status === "running" ? "agents working…" : "Ask"}
        </button>
        <div className={`status status-${status}`}>{status}</div>

        <h2>history</h2>
        <ul className="runs">
          {runs.map((r) => (
            <li
              key={r.run_id}
              className={r.run_id === activeRun ? "active" : ""}
              onClick={() => openStream(r.run_id)}
            >
              <span className={`dot dot-${r.status}`} />
              {r.question.slice(0, 48)}
            </li>
          ))}
        </ul>
      </aside>

      <main className="main">
        <section className="dag">
          <DagView tasks={tasks} />
        </section>

        {councilRouting && (
          <section className={`routing-note routing-${councilRouting.invoked ? "on" : "off"}`}>
            <strong>
              Council {councilRouting.invoked ? "invoked" : "skipped"}
            </strong>
            <span>
              {councilRouting.mode} · risk {councilRouting.risk_score}/{councilRouting.threshold}
              {councilRouting.reasons.length ? ` · ${councilRouting.reasons.join("; ")}` : ""}
            </span>
          </section>
        )}

        {dataQualityReport && (
          <section className="quality-report">
            <div className="section-heading">
              <div>
                <h2>automatic EDA &amp; data quality</h2>
                <p>{dataQualityReport.summary}</p>
              </div>
              <span className={`quality-score ${
                dataQualityReport.quality_score >= 90
                  ? "quality-good"
                  : dataQualityReport.quality_score >= 70
                    ? "quality-watch"
                    : "quality-poor"
              }`} title={dataQualityReport.score_method}>
                {dataQualityReport.quality_score.toFixed(1)} / 100
              </span>
            </div>

            <div className="quality-counts">
              <span className="quality-error">{dataQualityReport.issue_counts.error} errors</span>
              <span className="quality-warning">{dataQualityReport.issue_counts.warning} warnings</span>
              <span className="quality-info">{dataQualityReport.issue_counts.info} EDA signals</span>
              <span>{dataQualityReport.table_count} tables · {dataQualityReport.total_rows.toLocaleString()} rows</span>
            </div>

            {dataQualityReport.issues.length > 0 && (
              <div className="quality-issues">
                {dataQualityReport.issues.map((issue, index) => (
                  <article key={`${issue.table}:${issue.column}:${issue.code}:${index}`} className={`quality-issue quality-${issue.severity}`}>
                    <header>
                      <strong>{issue.table}{issue.column ? `.${issue.column}` : ""}</strong>
                      <span>{issue.severity}</span>
                    </header>
                    <p>{issue.message}</p>
                    <small>{issue.recommendation}</small>
                  </article>
                ))}
              </div>
            )}

            <div className="table-profiles">
              {dataQualityReport.tables.map((table) => (
                <details key={table.name}>
                  <summary>
                    <strong>{table.name}</strong>
                    <span>
                      {table.row_count.toLocaleString()} rows × {table.column_count} columns
                      {table.duplicate_rows ? ` · ${table.duplicate_rows} duplicates` : ""}
                      {table.candidate_keys.length ? ` · keys: ${table.candidate_keys.join(", ")}` : ""}
                    </span>
                  </summary>
                  <div className="column-profile-grid">
                    {table.columns.map((column) => (
                      <article key={column.name}>
                        <header>
                          <strong>{column.name}</strong>
                          <span>{column.semantic_type}</span>
                        </header>
                        <small>
                          {percent(column.missing_rate)} missing · {column.distinct_count.toLocaleString()} distinct
                        </small>
                        {Object.keys(column.numeric_summary).length > 0 && (
                          <small>
                            median {column.numeric_summary.median?.toLocaleString()} · range {column.numeric_summary.min?.toLocaleString()}–{column.numeric_summary.max?.toLocaleString()}
                          </small>
                        )}
                        {Object.keys(column.datetime_summary).length > 0 && (
                          <small>{column.datetime_summary.min} → {column.datetime_summary.max}</small>
                        )}
                        {column.top_values.length > 0 && (
                          <small>
                            Top: {column.top_values.map((value) => `${value.value} (${percent(value.rate)})`).join(" · ")}
                          </small>
                        )}
                      </article>
                    ))}
                  </div>
                </details>
              ))}
            </div>

            {dataQualityReport.relationships.length > 0 && (
              <details className="relationships">
                <summary>inferred relationships ({dataQualityReport.relationships.length})</summary>
                {dataQualityReport.relationships.map((relationship) => (
                  <p key={`${relationship.parent_table}:${relationship.child_table}:${relationship.column}`}>
                    {relationship.parent_table} → {relationship.child_table} on {relationship.column}: {relationship.orphan_count} orphan keys
                  </p>
                ))}
              </details>
            )}
          </section>
        )}

        {(councilFindings.length > 0 || councilDecisions.length > 0) && (
          <section className="council">
            <div className="section-heading">
              <h2>analysis council</h2>
              {councilDecisions.length > 0 && (
                <span className={`decision decision-${councilDecisions[councilDecisions.length - 1].status}`}>
                  {councilDecisions[councilDecisions.length - 1].status}
                </span>
              )}
            </div>
            <div className="council-grid">
              {councilFindings.map((finding) => (
                <article key={`${finding.round}-${finding.specialist}`}>
                  <header>
                    <strong>{finding.specialist.replace("_", " ")}</strong>
                    <span className={`verdict verdict-${finding.verdict}`}>
                      {finding.verdict} · {Math.round(finding.confidence * 100)}%
                    </span>
                  </header>
                  <p>{finding.conclusion}</p>
                  {finding.risks.length > 0 && (
                    <small>Risk: {finding.risks.join(" · ")}</small>
                  )}
                </article>
              ))}
            </div>
            {councilDecisions.length > 0 && (
              <p className="adjudication">
                <strong>Adjudicator:</strong> {councilDecisions[councilDecisions.length - 1].conclusion}
                {councilDecisions[councilDecisions.length - 1].dissent.length
                  ? ` Dissent: ${councilDecisions[councilDecisions.length - 1].dissent.join(" · ")}`
                  : ""}
              </p>
            )}
          </section>
        )}

        {answer && (
          <section className="answer">
            <div className="section-heading">
              <h2>answer</h2>
              {status === "done" && activeRun && (
                <div className="report-actions" aria-label="Download analysis">
                  <a href={`/runs/${encodeURIComponent(activeRun)}/report`} download>
                    Download report
                  </a>
                  <a href={`/runs/${encodeURIComponent(activeRun)}/report?format=json`} download>
                    JSON evidence
                  </a>
                </div>
              )}
            </div>
            <pre>{answer}</pre>
            {claims.filter((c) => c.verdict !== "pass").length > 0 && (
              <div className="caveats">
                {claims.filter((c) => c.verdict !== "pass").map((c, i) => (
                  <p key={i}>
                    <strong>[{c.verdict}] {c.task_id}:</strong> {c.note}
                  </p>
                ))}
              </div>
            )}
          </section>
        )}

        {statisticalEvidence.length > 0 && (
          <section className="statistics">
            <h2>deterministic statistics</h2>
            <div className="statistics-grid">
              {statisticalEvidence.map((item) => (
                <article key={`${item.task_id}:${item.request_id}`} className={`statistics-${item.status}`}>
                  <header>
                    <strong>{item.request_id.replace(/_/g, " ")}</strong>
                    <span>{item.kind} · n={item.sample_size}</span>
                  </header>
                  <p>{item.summary}</p>
                  {item.diagnostics.winner_reversal === true && (
                    <small className="reversal-note">
                      Composition adjustment changed the winning group.
                    </small>
                  )}
                  {item.test && (
                    <small>
                      {item.test.name}: statistic={item.test.statistic.toPrecision(4)}, p={item.test.p_value.toPrecision(4)} · {item.test.significant ? "significant" : "not significant"} at α={item.test.alpha}
                    </small>
                  )}
                  {item.effects.length > 0 && (
                    <small>Effects: {item.effects.map((effect) => `${effect.name}=${effect.value.toPrecision(4)}${effect.interpretation ? ` (${effect.interpretation})` : ""}`).join(" · ")}</small>
                  )}
                  {item.warnings.map((warning) => <small key={warning}>Warning: {warning}</small>)}
                </article>
              ))}
            </div>
          </section>
        )}

        {predictiveEvidence.length > 0 && (
          <section className="predictive-models">
            <h2>predictive model evaluation</h2>
            <div className="predictive-grid">
              {predictiveEvidence.map((item) => {
                const maxImportance = Math.max(
                  ...item.feature_importance.map((feature) => Math.abs(feature.importance)),
                  0,
                );
                return (
                  <article key={`${item.task_id}:${item.request_id}`} className={`predictive-${item.status}`}>
                    <header>
                      <div>
                        <strong>{item.request_id.replace(/_/g, " ")}</strong>
                        <small>{item.task_type} · target: {item.target}</small>
                      </div>
                      <span>{item.selected_model || item.status}</span>
                    </header>
                    <p>{item.summary}</p>

                    {item.metrics.length > 0 && (
                      <div className="model-metrics">
                        {item.metrics.map((metric) => (
                          <div key={metric.name} className={metric.name === item.primary_metric ? "primary-metric" : ""}>
                            <strong>{metric.name.replace(/_/g, " ")}</strong>
                            <span>{metric.value.toPrecision(4)}</span>
                            {metric.lower !== null && metric.upper !== null && (
                              <small>95% CI {metric.lower.toPrecision(3)}–{metric.upper.toPrecision(3)}</small>
                            )}
                            {metric.baseline_value !== null && (
                              <small>baseline {metric.baseline_value.toPrecision(4)}</small>
                            )}
                          </div>
                        ))}
                      </div>
                    )}

                    {item.candidates.length > 0 && (
                      <details>
                        <summary>training-only model selection</summary>
                        {item.candidates.map((candidate) => (
                          <p key={candidate.name}>
                            {candidate.name}: {candidate.cv_mean.toPrecision(4)} ± {candidate.cv_std.toPrecision(3)} ({candidate.folds} folds)
                          </p>
                        ))}
                      </details>
                    )}

                    {item.feature_importance.length > 0 && (
                      <details open>
                        <summary>holdout permutation importance</summary>
                        <div className="importance-list">
                          {item.feature_importance.slice(0, 10).map((feature) => (
                            <div key={feature.feature}>
                              <span>{feature.feature}</span>
                              <i style={{ width: `${maxImportance ? Math.max(2, 100 * Math.abs(feature.importance) / maxImportance) : 2}%` }} />
                              <small>{feature.importance.toPrecision(3)}</small>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}

                    {item.slices.length > 0 && (
                      <details>
                        <summary>holdout slices ({item.slices.length})</summary>
                        {item.slices.map((slice) => (
                          <p key={`${slice.column}:${slice.value}`}>
                            {slice.column}={slice.value} · n={slice.n} · {slice.score === null ? slice.warning : `${slice.metric}=${slice.score.toPrecision(4)}`}
                          </p>
                        ))}
                      </details>
                    )}

                    {item.warnings.map((warning) => (
                      <small className="model-warning" key={warning}>Warning: {warning}</small>
                    ))}
                  </article>
                );
              })}
            </div>
          </section>
        )}

        <section className="log">
          <h2>agent activity</h2>
          <pre>{events.map(eventLine).join("\n")}</pre>
        </section>
      </main>
    </div>
  );
}
