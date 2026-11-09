"""Portable analysis reports built from an explicit subset of persisted evidence.

No JavaScript, external assets, template engine, or Markdown HTML interpretation.
Agent/user text is escaped, and execution logs/code/options are never exported.
"""
from __future__ import annotations

import json
from html import escape
from typing import Any

REPORT_VERSION = "analysis-report-v1"


def _pick(record: dict | None, fields: str) -> dict:
    record = record or {}
    return {key: record[key] for key in fields.split() if key in record}


def _records(value: list | dict | None) -> list[dict]:
    if isinstance(value, dict):
        value = list(value.values())
    return [item for item in (value or []) if isinstance(item, dict)]


def dataset_provenance(dataset: Any) -> dict:
    """Keep source identity/schema, excluding CSV contents and local paths."""
    if hasattr(dataset, "model_dump"):
        dataset = dataset.model_dump()
    dataset = dataset or {}
    return {
        "content_hash": dataset.get("content_hash"),
        "tables": [_pick(table, "name row_count columns dtypes")
                   for table in _records(dataset.get("tables"))],
    }


def report_payload(run: dict) -> dict:
    if not isinstance(run, dict) or run.get("status") != "done":
        raise ValueError("only completed runs can be exported")
    if not isinstance(run.get("answer"), str) or not run["answer"].strip():
        raise ValueError("completed run has no answer")
    quality = run.get("data_quality_report") or {}
    payload = {
        "report_version": REPORT_VERSION,
        **_pick(run, "run_id status question answer created_at started_at finished_at demo_note"),
        "dataset": dataset_provenance(run.get("dataset_provenance") or run.get("dataset")),
        "tasks": [_pick(t, "id goal kind status depends_on attempts revisions reuse_of")
                  for t in _records(run.get("dag"))],
        "claims": [_pick(c, "task_id claim verdict note") for c in _records(run.get("claims"))],
        "data_quality": {
            **_pick(quality, "quality_score score_method table_count total_rows summary issue_counts"),
            "issues": [_pick(issue, "code severity table column message recommendation "
                             "affected_count affected_rate")
                       for issue in _records(quality.get("issues"))],
        },
        "statistical_evidence": [], "predictive_evidence": [],
        "council_findings": [_pick(f, "specialist round verdict conclusion answer_key evidence_refs "
                                  "risks confidence recommended_action")
                             for f in _records(run.get("council_findings"))],
        "council_decisions": [_pick(d, "round status conclusion rationale dissent action task_id")
                              for d in _records(run.get("council_decisions"))],
    }
    for item in _records(run.get("statistical_evidence")):
        payload["statistical_evidence"].append({
            **_pick(item, "task_id request_id kind status sample_size summary warnings"),
            "intervals": [_pick(i, "parameter estimate lower upper level method")
                          for i in _records(item.get("intervals"))],
            "effects": [_pick(e, "name value interpretation") for e in _records(item.get("effects"))],
            "test": _pick(item.get("test"), "name statistic p_value alternative null_value "
                          "alpha significant") if item.get("test") is not None else None,
            "diagnostics": _pick(item.get("diagnostics"), "standardization direction reference_weights "
                                 "crude_rates adjusted_rates stratum_rates crude_ranking adjusted_ranking "
                                 "ranking_reversal winner_reversal simpson_reversal simpson_pairs"),
        })
    for item in _records(run.get("predictive_evidence")):
        payload["predictive_evidence"].append({
            **_pick(item, "task_id request_id task_type status target features selected_model "
                    "primary_metric train_size test_size split_strategy summary warnings"),
            "metrics": [_pick(m, "name value baseline_value higher_is_better lower upper confidence")
                        for m in _records(item.get("metrics"))],
            "candidates": [_pick(c, "name cv_mean cv_std folds")
                           for c in _records(item.get("candidates"))],
            "feature_importance": [_pick(f, "feature importance std")
                                   for f in _records(item.get("feature_importance"))],
            "slices": [_pick(s, "column value n metric score warning")
                       for s in _records(item.get("slices"))],
            "diagnostics": _pick(item.get("diagnostics"), "target_rows_dropped positive_label "
                                 "negative_label positive_rate train_time_max test_time_min "
                                 "actual_test_fraction cv_time_boundaries group_overlap "
                                 "train_primary_metric test_primary_metric baseline_primary_metric "
                                 "baseline_improvement random_seed bootstrap_iterations"),
        })
    return payload


def _text(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return format(value, ".6g")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _e(value: Any) -> str:
    return escape(_text(value), quote=True)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return '<p class="muted">None recorded.</p>'
    return ('<div class="table-wrap"><table><thead><tr>'
            + "".join(f'<th scope="col">{_e(h)}</th>' for h in headers)
            + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{_e(cell)}</td>" for cell in row) + "</tr>" for row in rows)
            + "</tbody></table></div>")


def _warnings(warnings: list[str]) -> str:
    if not warnings:
        return ""
    return '<ul class="warnings">' + "".join(f"<li>{_e(w)}</li>" for w in warnings) + "</ul>"


_STYLE = """
:root{font-family:system-ui,-apple-system,sans-serif;color:#18283b;background:#edf2f6}
*{box-sizing:border-box}body{margin:0}main{max-width:1080px;margin:40px auto;background:white;
padding:48px;border:1px solid #d9e2eb;border-radius:14px}header{border-bottom:3px solid #147d92;
padding-bottom:26px}.eyebrow{text-transform:uppercase;letter-spacing:.16em;color:#147d92;
font-size:12px;font-weight:750}h1{font-size:34px;line-height:1.2;letter-spacing:-.03em;margin:12px 0}
h2{font-size:21px;margin:0 0 14px}h3{font-size:16px;margin:0 0 10px}
p,li{line-height:1.6}p{margin:8px 0}.muted,small{color:#586b80}.question{font-size:18px}
nav{display:flex;gap:18px;flex-wrap:wrap;margin-top:20px}a{color:#126d82}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:26px 0}
.card{padding:16px;border:1px solid #d9e2eb;border-radius:8px}.card strong{display:block;font-size:23px}
.card span{color:#586b80;font-size:12px}.notice,.warnings{background:#fff7e7;border-left:3px solid
#d38a18;padding:14px 18px}.warnings{padding-left:34px}section{margin-top:34px;min-width:0}
article{border:1px solid #d9e2eb;border-radius:8px;padding:20px;margin-top:14px}
.answer{white-space:pre-wrap;overflow-wrap:anywhere;font-size:15px;line-height:1.75}
.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;margin:12px 0}
th{text-align:left;background:#f1f6fa;color:#324960}th,td{padding:10px;border-bottom:1px solid
#e2e9f0;vertical-align:top;overflow-wrap:anywhere}td{max-width:360px}code{overflow-wrap:anywhere}
details{margin-top:16px}summary{cursor:pointer;color:#126d82}pre{white-space:pre-wrap;
overflow-wrap:anywhere;font-size:12px;background:#f4f7fa;padding:16px;line-height:1.5}
footer{margin-top:36px;padding-top:20px;border-top:1px solid #d9e2eb;font-size:12px;color:#586b80}
@media(max-width:700px){main{margin:0;padding:22px;border:0;border-radius:0}h1{font-size:27px}
.cards{grid-template-columns:repeat(2,1fr)}article{padding:14px}th,td{padding:7px}}
@media print{@page{margin:16mm}body{background:white}main{margin:0;padding:0;border:0;max-width:none}
nav{display:none}article,.card,tr{break-inside:avoid}.table-wrap{overflow:visible}
details{display:none}a{color:inherit;text-decoration:none}h2,h3{break-after:avoid}}
"""


def render_report(payload: dict) -> str:
    """Render only a report_payload result; all dynamic values remain plain text."""
    claims = payload["claims"]
    failed_tasks = sum(t.get("status") in ("failed", "rejected") for t in payload["tasks"])
    unresolved = sum(c.get("verdict") != "pass" for c in claims)
    invalid = sum(e.get("status") != "computed" for e in
                  payload["statistical_evidence"] + payload["predictive_evidence"])
    quality = payload["data_quality"]
    body = [
        '<header><div class="eyebrow">Analyst · evidence report</div>',
        '<h1>From question to evidence</h1>',
        f'<p class="question">{_e(payload.get("question"))}</p>',
        f'<p class="muted">Run <code>{_e(payload.get("run_id"))}</code> · completed</p>',
        '<nav aria-label="Report sections"><a href="#answer">Answer</a>'
        '<a href="#evidence">Evidence</a><a href="#quality">Data quality</a>'
        '<a href="#provenance">Provenance</a></nav></header>',
    ]
    if payload.get("demo_note"):
        body.append(f'<p class="notice">{_e(payload["demo_note"])}</p>')
    if failed_tasks or unresolved or invalid:
        body.append(f'<p class="notice">Review before relying on this answer: {unresolved} '
                    f'claim caveat(s)/failure(s), {failed_tasks} failed task(s), '
                    f'{invalid} invalid evidence request(s). Details are preserved below.</p>')
    body.append('<div class="cards">' + "".join(
        f'<div class="card"><strong>{_e(value)}</strong><span>{_e(label)}</span></div>'
        for value, label in [
            (len(payload["tasks"]), "analysis tasks"), (len(claims), "validator claims recorded"),
            (len(payload["predictive_evidence"]), "model evaluations requested"),
            (quality.get("quality_score"), "data readiness / 100"),
        ]) + "</div>")
    body += [f'<section id="answer"><h2>Answer</h2><div class="answer">{_e(payload["answer"])}</div>',
             '<h3>Claim review</h3>', _table(["Task", "Claim", "Verdict", "Note"], [
                 [c.get(k) for k in ("task_id", "claim", "verdict", "note")] for c in claims]),
             '</section><section id="evidence"><h2>Statistical & predictive evidence</h2>']
    if not payload["statistical_evidence"] and not payload["predictive_evidence"]:
        body.append('<p class="muted">No deterministic statistical or predictive requests recorded.</p>')
    for item in payload["statistical_evidence"]:
        body += [f'<article><h3>{_e(item.get("request_id"))} · {_e(item.get("status"))}</h3>',
                 f'<p>{_e(item.get("summary"))}</p>',
                 _warnings(item.get("warnings", [])),
                 _table(["Parameter", "Estimate", "Lower", "Upper", "Level", "Method"], [
                     [i.get(k) for k in ("parameter", "estimate", "lower", "upper", "level", "method")]
                     for i in item["intervals"]]),
                 _table(["Effect", "Value", "Interpretation"], [
                     [e.get(k) for k in ("name", "value", "interpretation")] for e in item["effects"]])]
        if item["test"]:
            body.append(_table(["Test", "Statistic", "p-value", "Alternative", "Alpha"], [
                [item["test"].get(k) for k in ("name", "statistic", "p_value", "alternative", "alpha")]]))
        body.append('</article>')
    for item in payload["predictive_evidence"]:
        body += [f'<article><h3>{_e(item.get("request_id"))} · {_e(item.get("status"))}</h3>',
                 f'<p>{_e(item.get("summary"))}</p>',
                 f'<p class="muted">Target: {_e(item.get("target"))} · '
                 f'Split: {_e(item.get("split_strategy"))} · '
                 f'Train / holdout: {_e(item.get("train_size"))} / {_e(item.get("test_size"))}</p>',
                 _warnings(item.get("warnings", [])),
                 _table(["Metric", "Model", "Naive baseline", "Bootstrap interval", "Direction"], [
                     [m.get("name"), m.get("value"), m.get("baseline_value"),
                      f"[{_text(m['lower'])}, {_text(m['upper'])}] ({_text(m.get('confidence'))})"
                      if m.get("lower") is not None and m.get("upper") is not None else None,
                      "higher is better" if m.get("higher_is_better") else "lower is better"]
                     for m in item["metrics"]]),
                 '<h3>Training-only model selection</h3>',
                 _table(["Candidate", "CV mean", "CV std", "Folds"], [
                     [c.get(k) for k in ("name", "cv_mean", "cv_std", "folds")]
                     for c in item["candidates"]]), '</article>']
    body.append('</section>')
    body += ['<section id="quality"><h2>Data quality</h2>',
             f'<p>{_e(quality.get("summary", "No data-quality report recorded."))}</p>',
             _table(["Severity", "Location", "Finding", "Recommendation"], [
                 [i.get("severity"), f"{i.get('table', '')}.{i.get('column') or ''}".rstrip("."),
                  i.get("message"), i.get("recommendation")] for i in quality["issues"]]), '</section>']
    if payload["council_findings"] or payload["council_decisions"]:
        body += ['<section><h2>Review council & dissent</h2>',
                 _table(["Specialist", "Verdict", "Conclusion", "Risks"], [
                     [f.get(k) for k in ("specialist", "verdict", "conclusion", "risks")]
                     for f in payload["council_findings"]]),
                 _table(["Decision", "Conclusion", "Rationale", "Dissent"], [
                     [d.get(k) for k in ("status", "conclusion", "rationale", "dissent")]
                     for d in payload["council_decisions"]]), '</section>']
    body += ['<section id="provenance"><h2>Dataset & execution provenance</h2>',
             f'<p>Dataset SHA-256: <code>{_e(payload["dataset"]["content_hash"] or "Not recorded")}</code></p>',
             _table(["Table", "Rows", "Columns"], [
                 [t.get("name"), t.get("row_count"), t.get("columns")]
                 for t in payload["dataset"]["tables"]]),
             _table(["Task", "Kind", "Status", "Goal", "Depends on"], [
                 [t.get(k) for k in ("id", "kind", "status", "goal", "depends_on")]
                 for t in payload["tasks"]]),
             '<details><summary>Complete exported evidence packet</summary>',
             f'<pre>{escape(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))}</pre>',
             '</details></section>',
             '<footer>This report preserves recorded results and caveats; completion is not an '
             'independent correctness guarantee. Bootstrap intervals condition on the fitted model '
             'and observed holdout. Permutation importance is predictive, not causal. '
             'Use your browser’s Print command to save a PDF.</footer>']
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="default-src &#39;none&#39;; '
            'style-src &#39;unsafe-inline&#39;; base-uri &#39;none&#39;; form-action &#39;none&#39;">'
            '<title>Analyst · Analysis report</title>'
            f'<style>{_STYLE}</style></head><body><main>{"".join(body)}</main></body></html>')
