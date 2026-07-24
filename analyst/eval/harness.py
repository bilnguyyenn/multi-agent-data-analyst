"""Eval harness: run the benchmark across ablation configs and prove (or disprove)
that each component earns its complexity. The scorecard is the project's thesis
statement — if the critic doesn't move the numbers, the scorecard says so.

Configs: full | no-critic | no-validator | no-memory | bare
Metrics per config: mean correctness (per tier), trap flag rate, mean latency,
mean tokens per question. Plus critic-bench (precision/recall on flawed specimens),
which isolates the critic from the rest of the pipeline.

Cost control & free-tier budget arithmetic (learned the expensive way):
- The CLI turns cassette recording ON for eval runs — every answered question is
  replayable for free afterwards.
- The full 4-config matrix is ~500-800 requests. Groq's free tier allows 30/min
  and ~1,000/DAY: the per-minute cap makes a full matrix take hours of backoff
  sleep, and the daily cap can kill it near the end. RUN 1-2 CONFIGS PER DAY
  (`--configs full` today, `--configs no-critic,bare` tomorrow) and merge with
  `analyst scorecard`. Results are saved per config, immediately.
"""
from __future__ import annotations

import json
import statistics
import time
import uuid
from pathlib import Path

from analyst.data import build_manifest
from analyst.eval.benchmark import build_benchmark, score_item
from analyst.graph.control_loop import build_graph
from analyst.graph.state import Budgets
from analyst.llm.client import DailyQuotaExhausted, make_client
from analyst.sandbox.runner import make_runner

CONFIGS: dict[str, dict] = {
    "adaptive-council": {"critic": True,  "validator": True,  "council_mode": "adaptive", "memory": False},
    "full-council":     {"critic": True,  "validator": True,  "council_mode": "always",   "memory": False},
    "council-no-critic": {"critic": False, "validator": True, "council_mode": "always",   "memory": False},
    "full":             {"critic": True,  "validator": True,  "council_mode": "off",      "memory": False},
    "no-critic":        {"critic": False, "validator": True,  "council_mode": "off",      "memory": False},
    "no-validator":     {"critic": True,  "validator": False, "council_mode": "off",      "memory": False},
    "bare":             {"critic": False, "validator": False, "council_mode": "off",      "memory": False},
    # memory is evaluated separately (repeat-question latency/tokens), because
    # mixing cache hits into accuracy comparisons would confound them.
}

EXPECTED_BENCHMARK_SIZE = 25

SMOKE_IDS = ["t1_top_downtime", "t2_total_defects", "t2_worst_steel",
             "t3_small_n", "t4_shift_trap"]


def _tokens_from_events(events: list[dict]) -> int:
    total = 0
    for e in events:
        total += e.get("tokens", 0)
        total += e.get("input_tokens", 0) + e.get("output_tokens", 0)
    return total


def load_prior(config_name: str, out_dir: str | Path = "eval_results") -> dict[str, dict]:
    """Collect the best prior result per question id for a config, across all saved
    raw_*.json files (newest last so it wins). 'Best' = a PASSED result (error is
    None); questions that only ever errored are left out so --resume retries them.
    This is what makes a run resumable: a quota-killed run saved everything it
    finished, and the next run reuses those and re-attempts only the gaps."""
    out = Path(out_dir)
    if not out.exists():
        return {}
    prior: dict[str, dict] = {}
    for f in sorted(out.glob("raw_*.json")):  # oldest -> newest
        try:
            runs = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for run in runs:
            if run.get("config") != config_name:
                continue
            for r in run.get("results", []):
                if r.get("error") is None and r.get("id"):
                    prior[r["id"]] = r  # a later passing result supersedes an earlier
    return prior


def run_integrity_errors(run: dict, expected_n: int = EXPECTED_BENCHMARK_SIZE) -> list[str]:
    """Return reasons a saved run is not eligible for the headline scorecard.

    Failed replay attempts are still valuable diagnostics, so they remain in
    ``eval_results/raw_*.json``. They must not, however, overwrite a previously
    complete benchmark and silently turn the published project metrics into zeros.
    """
    errors: list[str] = []
    results = run.get("results")
    if run.get("config") not in CONFIGS:
        errors.append("unknown config")
    if run.get("aborted"):
        errors.append("run aborted")
    if not isinstance(results, list):
        return errors + ["results is not a list"]
    if len(results) != expected_n:
        errors.append(f"expected {expected_n} results, found {len(results)}")

    ids = [r.get("id") for r in results if isinstance(r, dict)]
    if len(ids) != len(results) or any(not item_id for item_id in ids):
        errors.append("one or more results are missing an id")
    elif len(set(ids)) != len(ids):
        errors.append("duplicate result ids")

    crashed = sum(
        1 for r in results if not isinstance(r, dict) or r.get("error") is not None
    )
    if crashed:
        errors.append(f"{crashed} result(s) contain errors")
    return errors


def select_latest_valid_runs(
    out_dir: str | Path = "eval_results",
    expected_n: int = EXPECTED_BENCHMARK_SIZE,
) -> tuple[dict[str, dict], dict[str, Path], list[tuple[Path, str, list[str]]]]:
    """Select the newest complete, zero-error run for each ablation config.

    Returns ``(runs, sources, rejected)``. Sorting oldest-to-newest means a newer
    *valid* run replaces an older one, while a newer broken/partial run is recorded
    in ``rejected`` and cannot poison the scorecard.
    """
    out = Path(out_dir)
    selected: dict[str, dict] = {}
    sources: dict[str, Path] = {}
    rejected: list[tuple[Path, str, list[str]]] = []
    if not out.exists():
        return selected, sources, rejected

    for path in sorted(out.glob("raw_*.json")):
        try:
            saved = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            rejected.append((path, "(unreadable)", [str(exc)]))
            continue
        if not isinstance(saved, list):
            rejected.append((path, "(invalid)", ["top-level JSON is not a list"]))
            continue
        for run in saved:
            config_name = run.get("config", "(missing)") if isinstance(run, dict) else "(invalid)"
            reasons = run_integrity_errors(run, expected_n) if isinstance(run, dict) else ["run is not an object"]
            if reasons:
                rejected.append((path, config_name, reasons))
                continue
            selected[config_name] = run
            sources[config_name] = path
    return selected, sources, rejected


def run_config(
    config_name: str,
    data_dir: str,
    subset: str = "all",
    provider: str | None = None,
    timeout: int = 60,
    prior: dict[str, dict] | None = None,
) -> dict:
    """Run one ablation config over the benchmark.

    resume: pass `prior` (from load_prior) to skip questions that already passed —
    their saved result is carried forward untouched, no LLM calls. Returns
    {"config", "results", "aborted"}; aborted=True means the daily quota ran out
    mid-run, so the caller should save partial progress and stop the whole matrix
    (a re-run tomorrow with --resume continues from exactly here)."""
    cfg = CONFIGS[config_name]
    prior = prior or {}
    items = build_benchmark(data_dir)
    if subset == "smoke":
        items = [i for i in items if i.id in SMOKE_IDS]

    llm = make_client(provider=provider)
    sandbox = make_runner()
    graph = build_graph(
        llm, sandbox, exec_timeout_s=timeout,
        critic_enabled=cfg["critic"], validator_enabled=cfg["validator"],
        council_mode=cfg.get("council_mode", "off"),
        memory=None,
    )
    manifest = build_manifest(data_dir)

    results = []
    aborted = False
    for i, item in enumerate(items, 1):
        if item.id in prior:  # already passed on an earlier run — carry it forward
            print(f"  [{i}/{len(items)}] {item.id} ... (resumed, skipped)", flush=True)
            results.append(prior[item.id])
            continue
        print(f"  [{i}/{len(items)}] {item.id} ...", end="", flush=True)
        start = time.monotonic()
        try:
            state = graph.invoke(
                {"run_id": f"eval-{uuid.uuid4().hex[:8]}", "question": item.question,
                 "dataset": manifest, "budgets": Budgets()},
                config={"recursion_limit": 120},
            )
            answer = state.get("answer", "")
            arts = [a.outcome.result for a in state.get("artifacts", {}).values()
                    if a.outcome.result is not None]
            events = state.get("events", [])
            error = None
        except DailyQuotaExhausted as e:  # can't recover today — stop cleanly
            print(f" DAILY QUOTA EXHAUSTED ({e})", flush=True)
            print("  -> stopping; partial results saved. Re-run with --resume "
                  "after the RPD resets (midnight Pacific) to continue.", flush=True)
            aborted = True
            break
        except Exception as e:  # a crashed question scores 0, run continues
            answer, arts, events, error = "", [], [], str(e)
        latency = time.monotonic() - start
        score = score_item(item, answer, arts)
        flag = "" if score["flagged"] is None else f" flagged={score['flagged']}"
        print(f" correct={score['correct']}{flag} ({latency:.0f}s"
              f"{', ERROR: ' + error[:80] if error else ''})", flush=True)
        results.append({
            "id": item.id, "tier": item.tier, "question": item.question,
            **score, "latency_s": round(latency, 1),
            "tokens": _tokens_from_events(events),
            "answer": answer[:400], "error": error,
        })
    return {"config": config_name, "results": results, "aborted": aborted}


def summarize(run: dict) -> dict:
    rs = run["results"]
    by_tier: dict[int, list] = {}
    for r in rs:
        by_tier.setdefault(r["tier"], []).append(r)
    traps = [r for r in rs if r["tier"] == 4]
    return {
        "config": run["config"],
        "n": len(rs),
        "correct_mean": round(sum(r["correct"] for r in rs) / max(len(rs), 1), 3),
        "correct_by_tier": {
            t: round(sum(r["correct"] for r in v) / len(v), 3)
            for t, v in sorted(by_tier.items())
        },
        "trap_flag_rate": (round(sum(1 for r in traps if r["flagged"]) / len(traps), 3)
                           if traps else None),
        "latency_mean_s": round(sum(r["latency_s"] for r in rs) / max(len(rs), 1), 1),
        "tokens_mean": int(sum(r["tokens"] for r in rs) / max(len(rs), 1)),
        "errors": sum(1 for r in rs if r["error"]),
    }


def save_raw(runs: list[dict], out_dir: str | Path = "eval_results") -> Path:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    p = out / f"raw_{time.strftime('%Y%m%d_%H%M%S')}.json"
    p.write_text(json.dumps(runs, indent=2, default=str))
    return p


# --- Trap-metric variance -------------------------------------------------------
# The tier-4 column is 5 questions at temperature 0.2, so a single run's trap score
# is noisy. This repeats ONLY the traps K times on the same data (temperature drives
# the variation — the dominant "does it reliably flag the trap?" uncertainty) and
# reports a mean with a 95% CI, so the write-up can bound that column instead of
# quoting one draw. Deliberately traps-only + one config to stay cheap on free tiers.

TRAP_TIER = 4

# two-sided t critical values @95% by degrees of freedom (K-1); small-K honest,
# no scipy dependency. Falls back to the normal-approx 1.96 for large K.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _ci95(xs: list[float]) -> tuple[float, float, float]:
    """Return (mean, half_width, sem) for a 95% t-CI. half_width=0 if <2 samples."""
    n = len(xs)
    mean = sum(xs) / n if n else 0.0
    if n < 2:
        return round(mean, 3), 0.0, 0.0
    sd = statistics.stdev(xs)
    sem = sd / (n ** 0.5)
    t = _T95.get(n - 1, 1.96)
    return round(mean, 3), round(t * sem, 3), round(sem, 3)


def run_trap_variance(
    config_name: str,
    data_dir: str,
    provider: str | None = None,
    repeats: int = 5,
    timeout: int = 60,
) -> dict:
    cfg = CONFIGS[config_name]
    items = [i for i in build_benchmark(data_dir) if i.tier == TRAP_TIER]
    llm = make_client(provider=provider)
    sandbox = make_runner()
    graph = build_graph(
        llm, sandbox, exec_timeout_s=timeout,
        critic_enabled=cfg["critic"], validator_enabled=cfg["validator"],
        council_mode=cfg.get("council_mode", "off"),
        memory=None,
    )
    manifest = build_manifest(data_dir)

    per_q: dict[str, dict[str, list]] = {
        it.id: {"question": it.question, "correct": [], "flagged": []} for it in items
    }
    run_means: list[float] = []      # tier-4 mean correctness, one per repeat
    run_flag_rates: list[float] = []  # trap flag rate, one per repeat
    aborted = False

    for r in range(1, repeats + 1):
        print(f"  == repeat {r}/{repeats} ({config_name}, {len(items)} traps) ==",
              flush=True)
        run_correct, run_flags = [], []
        for it in items:
            print(f"    {it.id} ...", end="", flush=True)
            try:
                state = graph.invoke(
                    {"run_id": f"var-{uuid.uuid4().hex[:8]}", "question": it.question,
                     "dataset": manifest, "budgets": Budgets()},
                    config={"recursion_limit": 120},
                )
                answer = state.get("answer", "")
                arts = [a.outcome.result for a in state.get("artifacts", {}).values()
                        if a.outcome.result is not None]
            except DailyQuotaExhausted as e:
                print(f" DAILY QUOTA EXHAUSTED — stopping ({e})", flush=True)
                aborted = True
                break
            except Exception as e:
                answer, arts = "", []
                print(f" ERROR {str(e)[:60]}", flush=True)
            score = score_item(it, answer, arts)
            flagged = bool(score["flagged"])
            per_q[it.id]["correct"].append(score["correct"])
            per_q[it.id]["flagged"].append(flagged)
            run_correct.append(score["correct"])
            run_flags.append(1.0 if flagged else 0.0)
            print(f" correct={score['correct']} flagged={flagged}", flush=True)
        if run_correct:
            run_means.append(sum(run_correct) / len(run_correct))
            run_flag_rates.append(sum(run_flags) / len(run_flags))
        if aborted:
            break

    mean, hw, sem = _ci95(run_means)
    fmean, fhw, _ = _ci95(run_flag_rates)
    return {
        "config": config_name,
        "repeats": len(run_means),
        "aborted": aborted,
        "tier4_correct": {"mean": mean, "ci95_halfwidth": hw, "sem": sem,
                          "per_run": [round(x, 3) for x in run_means]},
        "trap_flag_rate": {"mean": fmean, "ci95_halfwidth": fhw,
                           "per_run": [round(x, 3) for x in run_flag_rates]},
        "per_question": {
            qid: {"question": d["question"],
                  "correct_mean": round(sum(d["correct"]) / len(d["correct"]), 3)
                  if d["correct"] else None,
                  "flag_rate": round(sum(d["flagged"]) / len(d["flagged"]), 3)
                  if d["flagged"] else None,
                  "n": len(d["correct"])}
            for qid, d in per_q.items()
        },
    }


def write_trap_variance(summ: dict, out_dir: str | Path = "eval_results") -> Path:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    c, f = summ["tier4_correct"], summ["trap_flag_rate"]
    lines = [
        "# Trap-metric variance\n",
        f"_Config `{summ['config']}` · {summ['repeats']} repeats · same data, "
        f"temperature-driven variation · 95% t-CI._\n",
        f"**Tier-4 correctness:** {c['mean']:.3f} ± {c['ci95_halfwidth']:.3f} "
        f"(95% CI)  ·  per-run {c['per_run']}\n",
        f"**Trap flag rate:** {f['mean']:.3f} ± {f['ci95_halfwidth']:.3f} "
        f"(95% CI)  ·  per-run {f['per_run']}\n",
        "\n| trap | correctness (mean) | flag rate | n |",
        "|---|---|---|---|",
    ]
    for qid, d in summ["per_question"].items():
        lines.append(f"| {qid} | {d['correct_mean']} | {d['flag_rate']} | {d['n']} |")
    if summ["aborted"]:
        lines.append("\n> ⚠️ Run aborted early (daily quota) — CI reflects the "
                     "repeats that completed.")
    p = out / "TRAP_VARIANCE.md"
    p.write_text("\n".join(lines) + "\n")
    return p
