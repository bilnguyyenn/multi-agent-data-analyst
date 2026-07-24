# Evaluation Scorecard

_Generated 2026-07-08 18:26. Traps score: 1.0 correct conclusion · 0.5 naive-but-flagged · 0.0 confidently wrong._

## Ablation comparison

| config | overall | tier1 | tier2 | tier3 | tier4 (traps) | trap flag rate | mean latency | mean tokens | errors |
|---|---|---|---|---|---|---|---|---|---|
| full | **1.0** | 1.0 | 1.0 | — | — | — | 23.7s | 4416 | 0 |

## Reading guide

- `full` vs `no-critic`: the critic's contribution to trap handling and method quality.
- `full` vs `no-validator`: the validator's contribution to tier-3 statistical framing.
- `bare`: single planner+executor baseline — what a naive agent gets you.
- If a component doesn't move its metric, that finding stays in this table.
