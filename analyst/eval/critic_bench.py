"""Critic-bench: measures the critic DIRECTLY with handcrafted specimens.

Why this exists in addition to the e2e benchmark: end-to-end accuracy conflates
planner, executor, critic, and validator. This isolates one question — "shown a
flawed analysis, does the critic catch it? shown a sound one, does it stay quiet?"
— as precision/recall over a labeled specimen library.

Each specimen is (goal, code, result, flawed?, expected_class). The flawed ones are
faithful reproductions of the planted trap classes; the sound controls punish a
critic that rejects everything to look vigilant.
"""
from __future__ import annotations

from dataclasses import dataclass

from analyst.agents.critic import review
from analyst.graph.state import (
    Artifact, DatasetManifest, ExecutionOutcome, TaskSpec,
)


@dataclass
class Specimen:
    id: str
    goal: str
    code: str
    result: dict
    flawed: bool
    flaw_class: str = ""


SPECIMENS = [
    Specimen(
        "fanout_join", "total quantity produced per machine",
        code=(
            "wo = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'work_orders.csv'))\n"
            "df = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'defects.csv'))\n"
            "merged = wo.merge(df, on='wo_id')   # one row per defect type!\n"
            "per = merged.groupby('machine_id')['qty_produced'].sum()\n"
            "json.dump({'answer': 'qty per machine', 'data': per.to_dict()}, open('result.json','w'))\n"
        ),
        result={"answer": "qty per machine", "data": {"M01": 812345, "M02": 790123}},
        flawed=True, flaw_class="join fan-out",
    ),
    Specimen(
        "survivorship", "average downtime per machine in the current fleet vs 2025",
        code=(
            "m = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'machines.csv'))\n"
            "active = m[m.status == 'active'].machine_id\n"
            "dt = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'downtime_events.csv'))\n"
            "dt = dt[dt.machine_id.isin(active)]\n"
            "trend = dt.groupby(dt.event_date.str[:7]).downtime_hours.mean()\n"
            "json.dump({'answer': 'downtime is improving', 'data': trend.to_dict()}, open('result.json','w'))\n"
        ),
        result={"answer": "downtime is improving", "data": {"2025-01": 3.4, "2026-05": 2.7}},
        flawed=True, flaw_class="survivorship bias",
    ),
    Specimen(
        "confounded", "which shift has the lowest defect rate",
        code=(
            "wo = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'work_orders.csv'))\n"
            "d = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'defects.csv'))\n"
            "per_wo = d.groupby('wo_id').defect_count.sum()\n"
            "wo['defects'] = wo.wo_id.map(per_wo).fillna(0)\n"
            "rate = wo.groupby('shift').defects.sum() / wo.groupby('shift').qty_produced.sum()\n"
            "json.dump({'answer': f'best shift is {rate.idxmin()}', 'data': rate.to_dict()}, open('result.json','w'))\n"
        ),
        result={"answer": "best shift is night",
                "data": {"day": 0.021, "evening": 0.021, "night": 0.014}},
        flawed=True, flaw_class="confounded comparison",
    ),
    Specimen(
        "silent_nulls", "average defect count per work order",
        code=(
            "wo = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'work_orders.csv'))\n"
            "d = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'defects.csv'))\n"
            "merged = wo.merge(d, on='wo_id', how='inner')  # drops zero-defect WOs\n"
            "avg = merged.groupby('wo_id').defect_count.sum().mean()\n"
            "json.dump({'answer': f'avg defects per WO is {avg:.1f}', 'data': {'avg': avg}}, open('result.json','w'))\n"
        ),
        result={"answer": "avg defects per WO is 4.6", "data": {"avg": 4.6}},
        flawed=True, flaw_class="silent null handling / denominator",
    ),
    Specimen(
        "wrong_grain", "average defect rate across products",
        code=(
            "wo = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'work_orders.csv'))\n"
            "d = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'defects.csv'))\n"
            "per_wo = d.groupby('wo_id').defect_count.sum()\n"
            "wo['rate'] = wo.wo_id.map(per_wo).fillna(0) / wo.qty_produced\n"
            "by_prod = wo.groupby('product_id')['rate'].mean()\n"
            "overall = by_prod.mean()   # average of averages\n"
            "json.dump({'answer': f'average defect rate {overall:.3f}', 'data': {'rate': overall}}, open('result.json','w'))\n"
        ),
        result={"answer": "average defect rate 0.017", "data": {"rate": 0.017}},
        flawed=True, flaw_class="averaging averages",
    ),
    Specimen(
        "goal_mismatch", "total downtime hours for machine M05",
        code=(
            "dt = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'downtime_events.csv'))\n"
            "n = (dt.machine_id == 'M05').sum()\n"
            "json.dump({'answer': f'M05 had {n} downtime events', 'data': {'events': int(n)}}, open('result.json','w'))\n"
        ),
        result={"answer": "M05 had 51 downtime events", "data": {"events": 51}},
        flawed=True, flaw_class="goal mismatch (count vs hours)",
    ),
    # ---- sound controls: the critic must APPROVE these ----
    Specimen(
        "sound_agg", "total downtime hours per machine",
        code=(
            "dt = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'downtime_events.csv'))\n"
            "per = dt.groupby('machine_id').downtime_hours.sum().round(2)\n"
            "json.dump({'answer': f'top is {per.idxmax()}', 'data': per.to_dict()}, open('result.json','w'))\n"
        ),
        result={"answer": "top is M13", "data": {"M13": 196.54, "M07": 150.2}},
        flawed=False,
    ),
    Specimen(
        "sound_dedup", "total defective parts per machine",
        code=(
            "wo = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'work_orders.csv'))\n"
            "d = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'defects.csv'))\n"
            "per_wo = d.groupby('wo_id').defect_count.sum()   # aggregate BEFORE join\n"
            "wo['defects'] = wo.wo_id.map(per_wo).fillna(0)\n"
            "per = wo.groupby('machine_id').defects.sum()\n"
            "json.dump({'answer': 'defects per machine', 'data': per.to_dict()}, open('result.json','w'))\n"
        ),
        result={"answer": "defects per machine", "data": {"M01": 320.0, "M02": 291.0}},
        flawed=False,
    ),
    Specimen(
        "sound_count", "how many work orders ran on the night shift",
        code=(
            "wo = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'work_orders.csv'))\n"
            "n = int((wo['shift'] == 'night').sum())\n"
            "json.dump({'answer': f'{n} night work orders', 'data': {'n': n}}, open('result.json','w'))\n"
        ),
        result={"answer": "743 night work orders", "data": {"n": 743}},
        flawed=False,
    ),
    Specimen(
        "sound_weighted", "defect rate per supplier weighted by lot size",
        code=(
            "lots = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'material_lots.csv'))\n"
            "rate = lots.groupby('supplier_id').apply(\n"
            "    lambda g: g.defective_units.sum() / g.units.sum())\n"
            "json.dump({'answer': 'weighted rate per supplier', 'data': rate.to_dict()}, open('result.json','w'))\n"
        ),
        result={"answer": "weighted rate per supplier",
                "data": {"S1": 0.052, "S2": 0.039, "S3": 0.051}},
        flawed=False,
    ),
]


def run_critic_bench(llm, dataset: DatasetManifest) -> dict:
    """Returns detection metrics + per-specimen detail."""
    rows = []
    for s in SPECIMENS:
        task = TaskSpec(id=s.id, goal=s.goal, status="review")
        art = Artifact(task_id=s.id, code=s.code,
                       outcome=ExecutionOutcome(status="ok", result=s.result))
        try:
            v = review(llm, s.goal, task, art, dataset)
        except Exception as e:  # one specimen failing must not lose the rest
            print(f"  !! specimen {s.id} errored: {e}", flush=True)
            rows.append({"id": s.id, "flawed": s.flawed, "flaw_class": s.flaw_class,
                         "verdict": "error", "detected": False,
                         "reason": str(e)[:200], "tokens": 0})
            continue
        detected = v.verdict in ("revise", "replan")
        print(f"  {s.id}: {v.verdict}", flush=True)
        rows.append({
            "id": s.id, "flawed": s.flawed, "flaw_class": s.flaw_class,
            "verdict": v.verdict, "detected": detected, "reason": v.reason[:200],
            "tokens": v.input_tokens + v.output_tokens,
        })
    tp = sum(1 for r in rows if r["flawed"] and r["detected"])
    fp = sum(1 for r in rows if not r["flawed"] and r["detected"])
    fn = sum(1 for r in rows if r["flawed"] and not r["detected"])
    tn = sum(1 for r in rows if not r["flawed"] and not r["detected"])
    return {
        "rows": rows,
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
    }
