"""Scorecard generator: turns eval runs into the markdown table that leads the README."""
from __future__ import annotations

import time
from pathlib import Path

from analyst.eval.harness import summarize


def render(
    runs: list[dict],
    critic_bench: dict | None = None,
    sources: dict[str, str | Path] | None = None,
) -> str:
    summaries = [summarize(r) for r in runs]
    lines = [
        "# Evaluation Scorecard",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M')}. "
        "Traps score: 1.0 correct conclusion · 0.5 naive-but-flagged · 0.0 confidently wrong._",
        "",
        "_Integrity gate: headline metrics use only complete 25-question runs with "
        "zero execution errors. Broken or partial attempts remain in `eval_results/` "
        "for diagnosis but cannot replace a valid scorecard input._",
        "",
        "## Ablation comparison",
        "",
        "| config | overall | tier1 | tier2 | tier3 | tier4 (traps) | trap flag rate | mean latency | mean tokens | errors |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        bt = s["correct_by_tier"]
        lines.append(
            f"| {s['config']} | **{s['correct_mean']}** "
            f"| {bt.get(1, '—')} | {bt.get(2, '—')} | {bt.get(3, '—')} | {bt.get(4, '—')} "
            f"| {s['trap_flag_rate'] if s['trap_flag_rate'] is not None else '—'} "
            f"| {s['latency_mean_s']}s | {s['tokens_mean']} | {s['errors']} |"
        )

    if sources:
        lines += ["", "### Result provenance", ""]
        for config_name in sorted(sources):
            path = Path(sources[config_name]).as_posix()
            lines.append(f"- `{config_name}`: `{path}`")

    names = {s["config"] for s in summaries}
    if {"council-no-critic", "no-critic", "bare"}.issubset(names):
        lines += [
            "",
            "### Product decision",
            "",
            "- **Standard:** `no-critic` — validator on, council off.",
            "- **High assurance:** `council-no-critic` — validator and council on.",
            "- The critic is retained only for ablations; adaptive routing remains experimental.",
        ]

    if critic_bench:
        lines += [
            "",
            "## Critic-bench (isolated: flawed-specimen detection)",
            "",
            f"- **Recall (flaws caught):** {critic_bench['recall']:.2f}",
            f"- **Precision:** {critic_bench['precision']:.2f}",
            f"- **False-positive rate on sound analyses:** {critic_bench['false_positive_rate']:.2f}",
            "",
            "| specimen | flaw class | verdict | detected |",
            "|---|---|---|---|",
        ]
        for r in critic_bench["rows"]:
            lines.append(
                f"| {r['id']} | {r['flaw_class'] or '(sound control)'} "
                f"| {r['verdict']} | {'✓' if r['detected'] else '—'} |"
            )

    lines += [
        "",
        "## Reading guide",
        "",
        "- `adaptive-council` vs `no-critic`: whether risk routing beats the Standard path.",
        "- `council-no-critic` vs `full-council`: whether the council subsumes the critic.",
        "- `full-council` vs `full`: the parallel council's incremental value and cost.",
        "- `full` vs `no-critic`: the critic's contribution to trap handling and method quality.",
        "- `full` vs `no-validator`: the validator's contribution to tier-3 statistical framing.",
        "- `bare`: single planner+executor baseline — what a naive agent gets you.",
        "- If a component doesn't move its metric, that finding stays in this table.",
        "",
    ]
    return "\n".join(lines)


def write(
    runs: list[dict],
    critic_bench: dict | None = None,
    path: str | Path = "eval_results/SCORECARD.md",
    sources: dict[str, str | Path] | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(exist_ok=True)
    p.write_text(render(runs, critic_bench, sources=sources))
    return p
