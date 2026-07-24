"""Synthetic manufacturing dataset WITH DELIBERATELY PLANTED TRAPS.

We generate the data ourselves so ground truth is exact and the traps are designed,
not hoped for. Planted (each maps to a benchmark trap question in phase f):

1. JOB-MIX CONFOUNDER — night shift runs mostly low-complexity products, so its
   naive defect rate looks best; within every complexity class night is actually
   slightly WORSE. "Which shift has the best quality record?" has a misleading
   obvious answer.
2. SURVIVORSHIP — three chronically bad machines were decommissioned; they're
   flagged in machines.csv and absent from "current" views, but their history
   lives in downtime/defects. Fleet trends computed on active machines only
   look like improvement that never happened.
3. FAN-OUT JOIN — defects has one row per (work_order, defect_type). Joining
   work_orders->defects then summing qty_produced double-counts production.
4. SIMPSON'S PARADOX — supplier S1 beats S2 within BOTH materials, but S1
   ships mostly the hard material (composite), so S1 looks worse overall.

ground_truth.json stores exact answers (naive AND correct) for the eval harness.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

COMPLEXITY_DEFECT = {"low": 0.006, "medium": 0.016, "high": 0.035}
SHIFT_EFFECT = {"day": 1.0, "evening": 1.1, "night": 1.25}  # night truly worst
# night runs easy jobs (the confounder):
SHIFT_MIX = {
    "day":     {"low": 0.25, "medium": 0.45, "high": 0.30},
    "evening": {"low": 0.35, "medium": 0.45, "high": 0.20},
    "night":   {"low": 0.70, "medium": 0.25, "high": 0.05},
}
DEFECT_TYPES = ["scratch", "dimension", "porosity"]


def generate(out_dir: str | Path, seed: int = 7) -> list[Path]:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- machines (survivorship trap) ----
    n_machines = 14
    machines = pd.DataFrame(
        {
            "machine_id": [f"M{i:02d}" for i in range(1, n_machines + 1)],
            "machine_type": rng.choice(["CNC_mill", "lathe", "press"], n_machines),
            "commissioned": pd.to_datetime("2018-01-01")
            + pd.to_timedelta(rng.integers(0, 1200, n_machines), unit="D"),
            "status": "active",
            "decommissioned": pd.NaT,
        }
    )
    bad = ["M03", "M08", "M11"]  # chronically bad, later removed
    machines.loc[machines.machine_id.isin(bad), "status"] = "decommissioned"
    machines.loc[machines.machine_id.isin(bad), "decommissioned"] = pd.Timestamp(
        "2025-06-30"
    )
    machine_defect_mult = pd.Series(
        rng.uniform(0.8, 1.2, n_machines), index=machines.machine_id
    )
    machine_defect_mult[bad] = rng.uniform(1.8, 2.4, len(bad))  # genuinely worse

    # ---- products ----
    products = pd.DataFrame(
        {
            "product_id": [f"P{i}" for i in range(1, 9)],
            "product_name": [f"part_{c}" for c in "ABCDEFGH"],
            "complexity": ["low", "low", "low", "medium", "medium", "medium", "high", "high"],
        }
    )
    by_complexity = {
        c: products.loc[products.complexity == c, "product_id"].tolist()
        for c in ("low", "medium", "high")
    }

    # ---- work orders (job-mix confounder lives here) ----
    n_wo = 3000
    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 540, n_wo), unit="D"
    )
    shifts = rng.choice(["day", "evening", "night"], n_wo, p=[0.4, 0.35, 0.25])
    prod_ids, machine_ids = [], []
    active_pool = machines.machine_id.tolist()
    for i in range(n_wo):
        mix = SHIFT_MIX[shifts[i]]
        cx = rng.choice(["low", "medium", "high"], p=[mix["low"], mix["medium"], mix["high"]])
        prod_ids.append(rng.choice(by_complexity[cx]))
        # decommissioned machines only have work before 2025-07-01
        pool = (
            active_pool
            if dates[i] < pd.Timestamp("2025-07-01")
            else [m for m in active_pool if m not in bad]
        )
        machine_ids.append(rng.choice(pool))
    qty = rng.integers(50, 400, n_wo)
    work_orders = pd.DataFrame(
        {
            "wo_id": [f"WO{i:05d}" for i in range(1, n_wo + 1)],
            "machine_id": machine_ids,
            "product_id": prod_ids,
            "shift": shifts,
            "wo_date": dates.normalize(),
            "qty_produced": qty,
            "operator_id": [f"OP{n:02d}" for n in rng.integers(1, 25, n_wo)],
        }
    )

    # ---- defects (fan-out trap: one row per wo_id x defect_type) ----
    cx_of = products.set_index("product_id").complexity
    p_defect = (
        work_orders.product_id.map(cx_of).map(COMPLEXITY_DEFECT).to_numpy()
        * work_orders["shift"].map(SHIFT_EFFECT).to_numpy()  # .shift is a method!
        * work_orders.machine_id.map(machine_defect_mult).to_numpy()
    )
    total_defects = rng.binomial(work_orders.qty_produced.to_numpy(), p_defect)
    rows = []
    for wo, nd in zip(work_orders.wo_id, total_defects):
        if nd == 0:
            continue
        split = rng.multinomial(nd, [0.5, 0.3, 0.2])
        for dt, c in zip(DEFECT_TYPES, split):
            if c > 0:
                rows.append({"wo_id": wo, "defect_type": dt, "defect_count": int(c)})
    defects = pd.DataFrame(rows)
    work_orders_defects = pd.Series(total_defects, index=work_orders.wo_id)

    # ---- downtime ----
    n_dt = 700
    dt_machines = rng.choice(machines.machine_id, n_dt)
    base_hours = rng.gamma(2.0, 1.5, n_dt)
    mult = pd.Series(1.0, index=machines.machine_id)
    mult[bad] = 2.2
    downtime = pd.DataFrame(
        {
            "event_id": [f"DT{i:04d}" for i in range(1, n_dt + 1)],
            "machine_id": dt_machines,
            "event_date": (
                pd.to_datetime("2025-01-01")
                + pd.to_timedelta(rng.integers(0, 540, n_dt), unit="D")
            ).normalize(),
            "downtime_hours": np.round(base_hours * mult[dt_machines].to_numpy(), 2),
            "reason": rng.choice(
                ["mechanical", "electrical", "changeover", "material_wait"], n_dt
            ),
        }
    )
    # decommissioned machines have no downtime after removal
    cut = (downtime.machine_id.isin(bad)) & (
        downtime.event_date >= pd.Timestamp("2025-07-01")
    )
    downtime = downtime[~cut].reset_index(drop=True)

    # ---- suppliers + lots (Simpson's paradox) ----
    suppliers = pd.DataFrame(
        {
            "supplier_id": ["S1", "S2", "S3"],
            "supplier_name": ["Apex Metals", "Borealis Supply", "Cyclone Materials"],
            "region": ["domestic", "domestic", "overseas"],
        }
    )
    lot_rows = []
    lot_defect = {  # (supplier, material) -> lot defect probability
        ("S1", "steel"): 0.02, ("S1", "composite"): 0.06,   # S1 better in both...
        ("S2", "steel"): 0.03, ("S2", "composite"): 0.075,
        ("S3", "steel"): 0.028, ("S3", "composite"): 0.07,
    }
    mix = {"S1": 0.8, "S2": 0.2, "S3": 0.5}  # ...but S1 ships mostly composite
    for i in range(1, 901):
        sup = rng.choice(["S1", "S2", "S3"], p=[0.4, 0.4, 0.2])
        material = "composite" if rng.random() < mix[sup] else "steel"
        n_units = int(rng.integers(200, 1200))
        n_def = rng.binomial(n_units, lot_defect[(sup, material)])
        lot_rows.append(
            {
                "lot_id": f"L{i:04d}",
                "supplier_id": sup,
                "material": material,
                "units": n_units,
                "defective_units": n_def,
                "received_date": (
                    pd.Timestamp("2025-01-01")
                    + pd.Timedelta(days=int(rng.integers(0, 540)))
                ).date(),
            }
        )
    lots = pd.DataFrame(lot_rows)

    # ---- write ----
    frames = {
        "machines": machines,
        "products": products,
        "work_orders": work_orders,
        "defects": defects,
        "downtime_events": downtime,
        "suppliers": suppliers,
        "material_lots": lots,
    }
    paths = []
    for name, df in frames.items():
        p = out / f"{name}.csv"
        df.to_csv(p, index=False)
        paths.append(p)

    # ---- exact ground truth (naive AND correct, for trap scoring) ----
    wo = work_orders.assign(defects=work_orders.wo_id.map(work_orders_defects).fillna(0))
    shift_naive = (wo.groupby("shift").defects.sum() / wo.groupby("shift").qty_produced.sum())
    wo_cx = wo.assign(complexity=wo.product_id.map(cx_of))
    shift_by_cx = (
        wo_cx.groupby(["complexity", "shift"])
        .apply(lambda g: g.defects.sum() / g.qty_produced.sum(), include_groups=False)
        .unstack()
    )
    sup_overall = lots.groupby("supplier_id").apply(
        lambda g: g.defective_units.sum() / g.units.sum(), include_groups=False
    )
    sup_by_mat = (
        lots.groupby(["material", "supplier_id"])
        .apply(lambda g: g.defective_units.sum() / g.units.sum(), include_groups=False)
        .unstack()
    )
    dt_per_machine = downtime.groupby("machine_id").downtime_hours.sum()
    truth = {
        "downtime_hours_by_machine": dt_per_machine.round(2).to_dict(),
        "top_downtime_machine": dt_per_machine.idxmax(),
        "shift_defect_rate_naive": shift_naive.round(5).to_dict(),
        "shift_defect_rate_by_complexity": {
            c: shift_by_cx.loc[c].round(5).to_dict() for c in shift_by_cx.index
        },
        "shift_trap_note": "night looks best naively but is worst within every complexity class",
        "supplier_defect_rate_overall": sup_overall.round(5).to_dict(),
        "supplier_defect_rate_by_material": {
            m: sup_by_mat.loc[m].round(5).to_dict() for m in sup_by_mat.index
        },
        "simpson_note": "S1 best within both materials, worst/middling overall due to composite-heavy mix",
        "decommissioned_machines": bad,
        "total_production_qty": int(wo.qty_produced.sum()),
    }
    # Ground truth lives OUTSIDE the dataset dir: that dir gets mounted into the
    # sandbox, and an agent that can read the answer key isn't being evaluated.
    tp = out.parent / f"{out.name}.ground_truth.json"
    tp.write_text(json.dumps(truth, indent=2, default=str))
    paths.append(tp)
    return paths
