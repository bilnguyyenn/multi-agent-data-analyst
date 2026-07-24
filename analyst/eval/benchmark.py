"""Benchmark: 25 questions, 4 tiers, expectations computed FROM THE DATA at eval
time — gold values can never drift from the dataset because they're derived from
the same CSVs the agents see.

Scoring philosophy (the part to defend in interviews):
- Plain questions: full credit iff the expected number/entity appears in the answer
  or in a result artifact (numeric tolerance).
- TRAP questions: a confident naive answer scores 0. Naming the trap in a caveat
  rescues half credit ("flagged"). Full credit requires the *correct* conclusion.
  A system that hedges honestly beats one that's confidently wrong — by design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class Item:
    id: str
    tier: int
    question: str
    number: tuple[float, float] | None = None       # (expected, tolerance)
    contains_any: list[str] = field(default_factory=list)   # correct-entity markers
    wrong_markers: list[str] = field(default_factory=list)  # naive-answer markers (traps)
    trap_keywords: list[str] = field(default_factory=list)  # caveat must mention one


def _nums(obj: Any):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _nums(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _nums(v)


def score_item(item: Item, answer: str, artifacts_results: list[Any]) -> dict:
    text = (answer or "").lower()
    all_numbers = [n for r in artifacts_results for n in _nums(r)]

    number_ok = True
    if item.number:
        expected, tol = item.number
        number_ok = any(abs(n - expected) <= tol for n in all_numbers) or _num_in_text(
            text, expected, tol
        )
    entity_ok = (not item.contains_any) or any(
        m.lower() in text for m in item.contains_any
    )

    if item.tier < 4:
        correct = 1.0 if (number_ok and entity_ok) else 0.0
        return {"correct": correct, "flagged": None}

    # --- trap scoring ---
    flagged = any(k.lower() in text for k in item.trap_keywords)
    said_wrong = any(m.lower() in text for m in item.wrong_markers)
    said_right = entity_ok and item.contains_any != []
    if said_right and not said_wrong:
        correct = 1.0
    elif flagged:
        correct = 0.5   # hedged: naive number but the trap is named
    else:
        correct = 0.0   # confidently wrong (or silent)
    return {"correct": correct, "flagged": flagged}


def _num_in_text(text: str, expected: float, tol: float) -> bool:
    import re

    for m in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "")):
        try:
            if abs(float(m) - expected) <= tol:
                return True
        except ValueError:
            continue
    return False


# ---------------------------------------------------------------------------
# Gold values from the data itself
# ---------------------------------------------------------------------------

def build_benchmark(data_dir: str | Path) -> list[Item]:
    d = Path(data_dir)
    machines = pd.read_csv(d / "machines.csv")
    wo = pd.read_csv(d / "work_orders.csv")
    defects = pd.read_csv(d / "defects.csv")
    dt = pd.read_csv(d / "downtime_events.csv")
    lots = pd.read_csv(d / "material_lots.csv")
    products = pd.read_csv(d / "products.csv")

    dt_per = dt.groupby("machine_id").downtime_hours.sum()
    def_per_wo = defects.groupby("wo_id").defect_count.sum()
    wo2 = wo.assign(defects=wo.wo_id.map(def_per_wo).fillna(0),
                    complexity=wo.product_id.map(products.set_index("product_id").complexity))
    sup_overall = lots.groupby("supplier_id").apply(
        lambda g: g.defective_units.sum() / g.units.sum(), include_groups=False)
    sup_by_mat = lots.groupby(["material", "supplier_id"]).apply(
        lambda g: g.defective_units.sum() / g.units.sum(), include_groups=False).unstack()
    best_within_both = sup_by_mat.idxmin(axis=1)
    simpson_winner = (best_within_both.iloc[0]
                      if best_within_both.nunique() == 1 else None)
    naive_sup = sup_overall.idxmin()
    shift_naive_best = (wo2.groupby("shift").defects.sum()
                        / wo2.groupby("shift").qty_produced.sum()).idxmin()
    true_shift = (wo2[wo2.complexity == "medium"].groupby("shift")
                  .apply(lambda g: g.defects.sum() / g.qty_produced.sum(),
                         include_groups=False)).idxmin()
    sup_name = pd.read_csv(d / "suppliers.csv").set_index("supplier_id").supplier_name

    items = [
        # ---- tier 1: lookups (6) ----
        Item("t1_wo_count", 1, "How many work orders are in the dataset?",
             number=(len(wo), 0)),
        Item("t1_active_machines", 1, "How many machines are currently active?",
             number=(int((machines.status == "active").sum()), 0)),
        Item("t1_top_downtime", 1, "Which machine had the most downtime hours?",
             number=(round(dt_per.max(), 1), 0.5), contains_any=[dt_per.idxmax()]),
        Item("t1_total_downtime", 1, "What is the total downtime across all machines, in hours?",
             number=(round(dt_per.sum(), 0), 2.0)),
        Item("t1_avg_qty", 1, "What is the average quantity produced per work order?",
             number=(round(wo.qty_produced.mean(), 1), 0.5)),
        Item("t1_top_reason", 1, "What is the most common downtime reason?",
             contains_any=[dt.reason.mode()[0]]),
        # ---- tier 2: multi-table (7) ----
        Item("t2_total_defects", 2, "How many defective parts were produced in total?",
             number=(float(defects.defect_count.sum()), 0)),
        Item("t2_worst_machine_defects", 2,
             "Which machine produced the most defective parts in total?",
             contains_any=[wo2.groupby("machine_id").defects.sum().idxmax()]),
        Item("t2_shift_production", 2, "Which shift produced the largest total quantity?",
             contains_any=[wo.groupby("shift").qty_produced.sum().idxmax()]),
        Item("t2_type_downtime", 2,
             "Which machine type accumulated the most downtime hours?",
             contains_any=[dt.merge(machines, on="machine_id")
                           .groupby("machine_type").downtime_hours.sum().idxmax()]),
        Item("t2_complexity_rate", 2,
             "What is the overall defect rate for high-complexity products, as a percentage?",
             number=(round(100 * wo2[wo2.complexity == "high"].defects.sum()
                           / wo2[wo2.complexity == "high"].qty_produced.sum(), 1), 0.3)),
        Item("t2_busiest_operator", 2, "Which operator handled the most work orders?",
             contains_any=[wo.operator_id.mode()[0]]),
        Item("t2_worst_steel", 2, "Which supplier has the highest defect rate for steel lots?",
             contains_any=[sup_by_mat.loc["steel"].idxmax(),
                           sup_name[sup_by_mat.loc["steel"].idxmax()]]),
        # ---- tier 3: statistical (7) ----
        Item("t3_day_vs_evening", 3,
             "Is the evening shift's defect rate meaningfully different from the day shift's?",
             contains_any=["confidence", "interval", "significan", "sample", "uncertain",
                           "not necessarily", "caution"]),
        Item("t3_downtime_outlier", 3,
             "Which machines have downtime significantly above the fleet average?",
             contains_any=["deviation", "significan", "interval", "threshold", "above"]),
        Item("t3_defect_trend", 3,
             "Has the overall defect rate changed over time? Quantify the trend and its reliability.",
             contains_any=["trend", "month", "over time"]),
        Item("t3_supplier_ci", 3,
             "Estimate the defect rate of each supplier with a measure of uncertainty.",
             contains_any=["interval", "uncertain", "margin", "confidence"]),
        Item("t3_operator_effect", 3,
             "Do defect rates differ meaningfully between operators, or is the variation noise?",
             contains_any=["multiple", "noise", "variation", "significan", "sample"]),
        Item("t3_small_n", 3,
             "Which product has the best quality record, and how confident can we be?",
             contains_any=["sample", "confiden", "uncertain", "n="]),
        Item("t3_downtime_dist", 3,
             "What is the typical downtime event duration, and is the mean a good summary of it?",
             contains_any=["skew", "median", "distribution", "outlier"]),
        # ---- tier 4: traps (5) ----
        Item("t4_shift_trap", 4, "Which shift has the best quality record?",
             contains_any=[true_shift] if true_shift != shift_naive_best else [],
             wrong_markers=[shift_naive_best],
             trap_keywords=["mix", "composition", "complexity", "confound", "job type",
                            "product type"]),
        Item("t4_supplier_trap", 4, "Which supplier has the best quality?",
             contains_any=([simpson_winner, sup_name[simpson_winner]]
                           if simpson_winner and simpson_winner != naive_sup else []),
             wrong_markers=[naive_sup, sup_name[naive_sup]],
             trap_keywords=["material", "mix", "simpson", "within", "confound",
                            "composition"]),
        Item("t4_survivorship", 4,
             "Has average machine downtime improved since July 2025?",
             wrong_markers=["improved", "decreas"],
             trap_keywords=["decommission", "surviv", "removed", "retired", "no longer"]),
        Item("t4_fanout", 4,
             "Using the defects table joined to work orders, what fraction of total production was defective?",
             number=(round(float(defects.defect_count.sum())
                           / wo.qty_produced.sum(), 4), 0.002),
             trap_keywords=["duplicate", "fan", "double", "one-to-many", "per work order"]),
        Item("t4_fleet_trap", 4,
             "Is the current fleet more reliable than the fleet was in early 2025?",
             wrong_markers=["more reliable", "improved"],
             trap_keywords=["decommission", "surviv", "retired", "selection"]),
    ]
    return items
