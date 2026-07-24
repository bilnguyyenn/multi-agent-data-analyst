"""Runner, integrity gate, and report for the held-out statistical suite."""
from __future__ import annotations

import json
import math
import statistics
import time
import uuid
from pathlib import Path

from scipy import stats

from analyst.data import build_manifest
from analyst.eval.heldout_benchmark import (
    EXPECTED_HELDOUT_IDS,
    EXPECTED_HELDOUT_SIZE,
    HELDOUT_SUITE,
    build_heldout_benchmark,
    score_heldout_item,
)
from analyst.eval.harness import CONFIGS, _tokens_from_events
from analyst.graph.control_loop import build_graph
from analyst.graph.state import Budgets
from analyst.llm.client import DailyQuotaExhausted, make_client
from analyst.sandbox.runner import make_runner

HELDOUT_SMOKE_IDS = {
    "he_b_conversion_ci",
    "ht_conversion_effect",
    "hc_team_simpson",
    "hd_event_fanout",
}


def heldout_integrity_errors(run: dict) -> list[str]:
    """Reject partial, crashed, mislabeled, or structurally incomplete runs."""
    errors: list[str] = []
    if run.get("suite") != HELDOUT_SUITE:
        errors.append("wrong or missing suite version")
    if run.get("config") not in CONFIGS:
        errors.append("unknown config")
    if not isinstance(run.get("seed"), int):
        errors.append("missing integer seed")
    if not run.get("dataset_hash"):
        errors.append("missing dataset hash")
    if run.get("aborted"):
        errors.append("run aborted")
    results = run.get("results")
    if not isinstance(results, list):
        return errors + ["results is not a list"]
    if len(results) != EXPECTED_HELDOUT_SIZE:
        errors.append(
            f"expected {EXPECTED_HELDOUT_SIZE} results, found {len(results)}"
        )
    ids = [result.get("id") for result in results if isinstance(result, dict)]
    if len(ids) != len(results) or any(not item_id for item_id in ids):
        errors.append("one or more results are missing an id")
    elif len(set(ids)) != len(ids):
        errors.append("duplicate result ids")
    elif set(ids) != set(EXPECTED_HELDOUT_IDS):
        errors.append("result ids do not match the held-out v1 contract")
    crashed = sum(
        1
        for result in results
        if not isinstance(result, dict) or result.get("error") is not None
    )
    if crashed:
        errors.append(f"{crashed} result(s) contain errors")
    malformed = sum(
        1
        for result in results
        if not isinstance(result, dict)
        or not isinstance(result.get("score"), (int, float))
        or not 0 <= result.get("score", -1) <= 1
        or not isinstance(result.get("check_results"), list)
    )
    if malformed:
        errors.append(f"{malformed} result(s) have malformed rubric scores")
    return errors


def load_heldout_prior(
    config_name: str,
    seed: int,
    dataset_hash: str,
    out_dir: str | Path = "heldout_eval_results",
) -> dict[str, dict]:
    """Load passing item results only from the same config, seed, and CSV bytes."""
    output = Path(out_dir)
    prior: dict[str, dict] = {}
    if not output.exists():
        return prior
    for path in sorted(output.glob("heldout_raw_*.json")):
        try:
            runs = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(runs, list):
            continue
        for run in runs:
            if (
                run.get("suite") != HELDOUT_SUITE
                or run.get("config") != config_name
                or run.get("seed") != seed
                or run.get("dataset_hash") != dataset_hash
            ):
                continue
            for result in run.get("results", []):
                if result.get("error") is None and result.get("id"):
                    prior[result["id"]] = result
    return prior


def run_heldout_config(
    config_name: str,
    data_dir: str | Path,
    seed: int,
    *,
    subset: str = "all",
    provider: str | None = None,
    timeout: int = 60,
    prior: dict[str, dict] | None = None,
) -> dict:
    """Execute one configuration on one independently generated held-out seed."""
    if config_name not in CONFIGS:
        raise ValueError(f"unknown config {config_name!r}")
    config = CONFIGS[config_name]
    manifest = build_manifest(data_dir)
    items = build_heldout_benchmark(data_dir)
    if subset == "smoke":
        items = [item for item in items if item.id in HELDOUT_SMOKE_IDS]
    prior = prior or {}
    graph = build_graph(
        make_client(provider=provider),
        make_runner(),
        exec_timeout_s=timeout,
        critic_enabled=config["critic"],
        validator_enabled=config["validator"],
        council_mode=config.get("council_mode", "off"),
        memory=None,
    )

    results = []
    aborted = False
    for index, item in enumerate(items, 1):
        if item.id in prior:
            print(f"  [{index}/{len(items)}] {item.id} ... (resumed, skipped)", flush=True)
            results.append(prior[item.id])
            continue
        print(f"  [{index}/{len(items)}] {item.id} ...", end="", flush=True)
        started = time.monotonic()
        try:
            state = graph.invoke(
                {
                    "run_id": f"heldout-{uuid.uuid4().hex[:8]}",
                    "question": item.question,
                    "dataset": manifest,
                    "budgets": Budgets(),
                },
                config={"recursion_limit": 120},
            )
            answer = state.get("answer", "")
            evidence = [
                value.model_dump()
                for value in state.get("statistical_evidence", {}).values()
            ]
            events = state.get("events", [])
            error = None
        except DailyQuotaExhausted as exc:
            print(f" DAILY QUOTA EXHAUSTED ({exc})", flush=True)
            aborted = True
            break
        except Exception as exc:
            answer, evidence, events, error = "", [], [], str(exc)
        latency = time.monotonic() - started
        rubric = score_heldout_item(item, answer, evidence)
        print(
            f" score={rubric['score']:.3f} "
            f"checks={rubric['checks_passed']}/{rubric['checks_total']} "
            f"({latency:.0f}s{', ERROR: ' + error[:80] if error else ''})",
            flush=True,
        )
        results.append(
            {
                "id": item.id,
                "category": item.category,
                "method": item.method,
                "question": item.question,
                **rubric,
                "latency_s": round(latency, 1),
                "tokens": _tokens_from_events(events),
                "answer": answer[:1000],
                "statistical_evidence": evidence,
                "error": error,
            }
        )
    return {
        "suite": HELDOUT_SUITE,
        "config": config_name,
        "seed": seed,
        "dataset_hash": manifest.content_hash,
        "results": results,
        "aborted": aborted,
    }


def save_heldout_raw(
    runs: list[dict], out_dir: str | Path = "heldout_eval_results"
) -> Path:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"heldout_raw_{time.strftime('%Y%m%d_%H%M%S')}.json"
    suffix = 1
    while path.exists():
        path = output / f"heldout_raw_{time.strftime('%Y%m%d_%H%M%S')}_{suffix}.json"
        suffix += 1
    path.write_text(json.dumps(runs, indent=2, default=str))
    return path


def select_latest_valid_heldout_runs(
    out_dir: str | Path = "heldout_eval_results",
) -> tuple[dict[tuple[str, int], dict], dict[tuple[str, int], Path], list[tuple[Path, str, list[str]]]]:
    """Select newest valid run per (configuration, independent data seed)."""
    output = Path(out_dir)
    selected: dict[tuple[str, int], dict] = {}
    sources: dict[tuple[str, int], Path] = {}
    rejected: list[tuple[Path, str, list[str]]] = []
    if not output.exists():
        return selected, sources, rejected
    for path in sorted(output.glob("heldout_raw_*.json")):
        try:
            runs = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            rejected.append((path, "(unreadable)", [str(exc)]))
            continue
        if not isinstance(runs, list):
            rejected.append((path, "(invalid)", ["top-level JSON is not a list"]))
            continue
        for run in runs:
            label = (
                f"{run.get('config', '(missing)')}:seed={run.get('seed', '(missing)')}"
                if isinstance(run, dict)
                else "(invalid)"
            )
            reasons = heldout_integrity_errors(run) if isinstance(run, dict) else ["run is not an object"]
            if reasons:
                rejected.append((path, label, reasons))
                continue
            key = (run["config"], run["seed"])
            selected[key] = run
            sources[key] = path
    return selected, sources, rejected


def summarize_heldout_runs(runs: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for run in runs:
        grouped.setdefault(run["config"], []).append(run)
    summaries = []
    for config_name, config_runs in sorted(grouped.items()):
        results = [result for run in config_runs for result in run["results"]]
        categories = sorted({result["category"] for result in results})
        seed_means = [statistics.mean(result["score"] for result in run["results"]) for run in config_runs]
        mean = statistics.mean(result["score"] for result in results)
        ci_halfwidth = None
        if len(seed_means) >= 2:
            sem = statistics.stdev(seed_means) / math.sqrt(len(seed_means))
            critical = float(stats.t(df=len(seed_means) - 1).ppf(0.975))
            ci_halfwidth = critical * sem
        summaries.append(
            {
                "config": config_name,
                "seeds": len(config_runs),
                "items": len(results),
                "overall": round(mean, 3),
                "ci95_halfwidth": round(ci_halfwidth, 3) if ci_halfwidth is not None else None,
                "by_category": {
                    category: round(
                        statistics.mean(
                            result["score"]
                            for result in results
                            if result["category"] == category
                        ),
                        3,
                    )
                    for category in categories
                },
                "checks": sum(result["checks_passed"] for result in results),
                "checks_total": sum(result["checks_total"] for result in results),
                "latency_mean_s": round(statistics.mean(result["latency_s"] for result in results), 1),
                "tokens_mean": int(statistics.mean(result["tokens"] for result in results)),
                "errors": sum(1 for result in results if result["error"]),
                "seed_means": [round(value, 3) for value in seed_means],
            }
        )
    return summaries


def render_heldout_scorecard(
    runs: list[dict],
    sources: dict[tuple[str, int], str | Path] | None = None,
) -> str:
    summaries = summarize_heldout_runs(runs)
    lines = [
        "# Held-out Statistical Benchmark",
        "",
        f"_Suite `{HELDOUT_SUITE}` · generated {time.strftime('%Y-%m-%d %H:%M')}._",
        "",
        "_Integrity gate: only complete 20-question runs with the exact v1 item set, "
        "a dataset content hash, and zero execution errors enter this report. Each "
        "question is scored by an explicit component rubric; ground truth is derived "
        "from independently generated CSV bytes at evaluation time._",
        "",
        "| config | overall rubric score | seeds | items | estimation | tests/effects | confounding | design/quality | checks | mean latency | mean tokens | errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for summary in summaries:
        category = summary["by_category"]
        overall = f"**{summary['overall']:.3f}**"
        if summary["ci95_halfwidth"] is not None:
            overall += f" ± {summary['ci95_halfwidth']:.3f}"
        lines.append(
            f"| {summary['config']} | {overall} | {summary['seeds']} | {summary['items']} "
            f"| {category.get('estimation', '—')} | {category.get('testing_effects', '—')} "
            f"| {category.get('confounding', '—')} | {category.get('design_quality', '—')} "
            f"| {summary['checks']}/{summary['checks_total']} "
            f"| {summary['latency_mean_s']}s | {summary['tokens_mean']} | {summary['errors']} |"
        )
    if sources:
        lines += ["", "## Result provenance", ""]
        for (config_name, seed), path in sorted(sources.items()):
            lines.append(f"- `{config_name}`, seed `{seed}`: `{Path(path).as_posix()}`")
    lines += [
        "",
        "## Coverage",
        "",
        "- **Estimation:** Wilson and t intervals, conditional estimands, small subgroups.",
        "- **Tests/effects:** exact/proportion/Welch/chi-square tests plus Cohen's h, Hedges' g, and Cramer's V.",
        "- **Confounding:** two planted composition reversals, equal/pooled standardization, and one no-reversal sound control.",
        "- **Design/quality:** one-to-many fanout, outcome-dependent nonresponse, multiplicity, missing data, and practical significance.",
        "",
        "## Interpretation",
        "",
        "The overall score is the mean fraction of rubric components satisfied, not "
        "the original benchmark's binary/trap score. A 95% interval is shown only "
        "when at least two independent generated-data seeds are available, using a "
        "t-interval over seed-level mean scores.",
    ]
    return "\n".join(lines) + "\n"


def write_heldout_scorecard(
    runs: list[dict],
    sources: dict[tuple[str, int], str | Path] | None = None,
    out_dir: str | Path = "heldout_eval_results",
) -> Path:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "SCORECARD.md"
    path.write_text(render_heldout_scorecard(runs, sources=sources))
    return path
