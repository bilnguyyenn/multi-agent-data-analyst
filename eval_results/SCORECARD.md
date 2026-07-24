# Evaluation Scorecard

_Generated 2026-07-11 21:28. Traps score: 1.0 correct conclusion · 0.5 naive-but-flagged · 0.0 confidently wrong._

_Integrity gate: headline metrics use only complete 25-question runs with zero execution errors. Broken or partial attempts remain in `eval_results/` for diagnosis but cannot replace a valid scorecard input._

## Ablation comparison

| config | overall | tier1 | tier2 | tier3 | tier4 (traps) | trap flag rate | mean latency | mean tokens | errors |
|---|---|---|---|---|---|---|---|---|---|
| full | **0.8** | 1.0 | 1.0 | 0.857 | 0.2 | 0.4 | 58.7s | 8086 | 0 |
| no-validator | **0.68** | 1.0 | 1.0 | 0.571 | 0.0 | 0.0 | 34.1s | 6775 | 0 |
| no-critic | **0.8** | 1.0 | 0.857 | 1.0 | 0.2 | 0.4 | 27.6s | 3429 | 0 |
| bare | **0.56** | 1.0 | 0.714 | 0.429 | 0.0 | 0.0 | 0.4s | 2007 | 0 |
| full-council | **0.84** | 1.0 | 1.0 | 0.857 | 0.4 | 0.8 | 116.8s | 19727 | 0 |
| adaptive-council | **0.8** | 1.0 | 1.0 | 0.857 | 0.2 | 0.4 | 110.6s | 19892 | 0 |
| council-no-critic | **0.86** | 1.0 | 1.0 | 1.0 | 0.3 | 0.6 | 212.3s | 17223 | 0 |

### Result provenance

- `adaptive-council`: `eval_results/raw_20260711_163629.json`
- `bare`: `eval_results/raw_20260709_070312.json`
- `council-no-critic`: `eval_results/raw_20260711_204642.json`
- `full`: `eval_results/raw_20260710_235446.json`
- `full-council`: `eval_results/raw_20260710_231212.json`
- `no-critic`: `eval_results/raw_20260708_231643.json`
- `no-validator`: `eval_results/raw_20260708_230513.json`

### Product decision

- **Standard:** `no-critic` — validator on, council off.
- **High assurance:** `council-no-critic` — validator and council on.
- The critic is retained only for ablations; adaptive routing remains experimental.

## Critic-bench (isolated: flawed-specimen detection)

- **Recall (flaws caught):** 0.83
- **Precision:** 0.71
- **False-positive rate on sound analyses:** 0.50

| specimen | flaw class | verdict | detected |
|---|---|---|---|
| fanout_join | join fan-out | revise | ✓ |
| survivorship | survivorship bias | revise | ✓ |
| confounded | confounded comparison | approve | — |
| silent_nulls | silent null handling / denominator | revise | ✓ |
| wrong_grain | averaging averages | revise | ✓ |
| goal_mismatch | goal mismatch (count vs hours) | revise | ✓ |
| sound_agg | (sound control) | approve | — |
| sound_dedup | (sound control) | approve | — |
| sound_count | (sound control) | revise | ✓ |
| sound_weighted | (sound control) | revise | ✓ |

## Reading guide

- `adaptive-council` vs `no-critic`: whether risk routing beats the Standard path.
- `council-no-critic` vs `full-council`: whether the council subsumes the critic.
- `full-council` vs `full`: the parallel council's incremental value and cost.
- `full` vs `no-critic`: the critic's contribution to trap handling and method quality.
- `full` vs `no-validator`: the validator's contribution to tier-3 statistical framing.
- `bare`: single planner+executor baseline — what a naive agent gets you.
- If a component doesn't move its metric, that finding stays in this table.
