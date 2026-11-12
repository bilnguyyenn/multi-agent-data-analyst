"""Exact, descriptive contribution bridges; no model-generated causes or SQL."""

from __future__ import annotations

import hashlib
import html
import json
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from analyst.operations.metrics import (
    ELIGIBLE,
    PREPARATION_SQL,
    REGISTRY,
    connect,
    fingerprint,
    prepare,
)

VERSION = "retail-change-bridge-v1"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class ChangeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: Literal["gross_sales", "cancellation_value", "net_sales", "units_sold"] = "net_sales"
    group_by: Literal["country", "product"] = "product"
    limit: int = Field(default=10, ge=1, le=100)


def grouped_values(path: Path, packet: dict, group_by: str) -> dict:
    """Read ALL groups, including groups omitted by a review's display limit."""
    request = packet["request"]
    filters = [
        ELIGIBLE,
        "invoice IS NOT NULL",
        "invoice <> ''",
        "occurred IS NOT NULL",
        "occurred >= ?",
        "occurred < ?",
    ]
    parameters = [request["start"], request["end"]]
    if request["country"] is not None:
        filters.append("country = ?")
        parameters.append(request["country"])
    # Both identifiers are validated by ChangeRequest before reaching this function.
    if group_by not in {"country", "product"} or packet["metric"] not in {
        "gross_sales",
        "cancellation_value",
        "net_sales",
        "units_sold",
    }:
        raise ValueError("change breakdown requires an additive metric and supported dimension")
    sql = (
        f"SELECT COALESCE({group_by}, '(missing)') AS segment, COALESCE({REGISTRY[packet['metric']]['formula']}, 0) "
        f"AS numerator, COUNT(*) AS source_rows FROM lines WHERE {' AND '.join(filters)} "
        "GROUP BY 1 ORDER BY segment"
    )
    connection = connect()
    timer = threading.Timer(30, connection.interrupt)
    timer.daemon = True
    timer.start()
    try:
        prepare(connection, path)
        rows = connection.execute(sql, parameters).fetchmany(100_001)
        if len(rows) > 100_000:
            raise ValueError("change breakdown exceeds 100,000 groups")
        return {
            "values": {segment: (int(value), count) for segment, value, count in rows},
            "sql": sql,
            "parameters": parameters,
        }
    finally:
        timer.cancel()
        timer.join()
        connection.close()


def bridge(before: dict, after: dict, request: ChangeRequest, left: dict, right: dict) -> dict:
    """Reconcile integer base units to the two stored overall computations."""
    scale = 100 if after["unit"] == "GBP" else 1

    def display(value):
        # Integer arithmetic preserves pence even above binary float precision.
        return (
            f"{'-' if value < 0 else ''}{abs(value) // 100}.{abs(value) % 100:02d}"
            if scale == 100
            else str(value)
        )

    old_total, new_total = (p["values"][0]["numerator"] for p in [before, after])
    if (
        sum(v[0] for v in left["values"].values()) != old_total
        or sum(v[0] for v in right["values"].values()) != new_total
    ):
        raise ValueError("group totals do not reconcile to the saved overall evidence")
    rows = []
    for segment in sorted(left["values"].keys() | right["values"].keys()):
        x, nx = left["values"].get(segment, (0, 0))
        y, ny = right["values"].get(segment, (0, 0))
        cohort = "continuing" if nx and ny else "newly_observed" if ny else "no_longer_observed"
        rows.append(
            {
                "segment": segment,
                "cohort": cohort,
                "previous_base_units": x,
                "current_base_units": y,
                "delta_base_units": y - x,
                "previous": display(x),
                "current": display(y),
                "delta": display(y - x),
                "previous_source_rows": nx,
                "current_source_rows": ny,
            }
        )
    rows.sort(key=lambda row: (-abs(row["delta_base_units"]), row["segment"]))
    shown, omitted = rows[: request.limit], rows[request.limit :]
    remainder = {
        field: sum(row[field] for row in omitted)
        for field in ["previous_base_units", "current_base_units", "delta_base_units"]
    }
    remainder.update(
        group_count=len(omitted),
        previous=display(remainder["previous_base_units"]),
        current=display(remainder["current_base_units"]),
        delta=display(remainder["delta_base_units"]),
    )
    cohorts = []
    for cohort in ["continuing", "newly_observed", "no_longer_observed"]:
        members = [row for row in rows if row["cohort"] == cohort]
        delta = sum(row["delta_base_units"] for row in members)
        cohorts.append(
            {
                "cohort": cohort,
                "group_count": len(members),
                "delta_base_units": delta,
                "delta": display(delta),
            }
        )
    result = {
        "version": VERSION,
        "metric": request.metric,
        "group_by": request.group_by,
        "unit": after["unit"],
        "base_unit_scale": scale,
        "previous_base_units": old_total,
        "current_base_units": new_total,
        "delta_base_units": new_total - old_total,
        "previous": display(old_total),
        "current": display(new_total),
        "delta": display(new_total - old_total),
        "group_count": len(rows),
        "rows": shown,
        "remainder": remainder,
        "cohorts": cohorts,
        "absolute_movement_base_units": sum(abs(row["delta_base_units"]) for row in rows),
        "reconciliation_residual_base_units": sum(row["delta_base_units"] for row in rows)
        - (new_total - old_total),
        "evidence_refs": [before["computation_id"], after["computation_id"]],
        "snapshot_hashes": [before["snapshot_hash"], after["snapshot_hash"]],
        "metric_contract_hash": after["metric_contract_hash"],
        "source_sha256": SOURCE_SHA256,
        "metric_engine_sha256": after["engine_sha256"],
        "preparation_sql": PREPARATION_SQL,
        "queries": [{"sql": g["sql"], "parameters": g["parameters"]} for g in [left, right]],
        "periods": [
            {key: p["request"][key] for key in ["start", "end", "country"]} for p in [before, after]
        ],
        "warnings": [
            "Contributions are accounting changes, not causal drivers, forecasts, or profit.",
            "Newly/no-longer observed means eligible records in these windows; it does not establish acquisition or churn.",
            "Ranking uses absolute change. Displayed rows plus the explicit remainder equal the overall change.",
            "Source duplicates and eligibility rules are unchanged. Missing source activity is not proof of no business activity.",
            "Stock codes can represent fees and adjustments as well as merchandise; inspect source descriptions before interpreting a product-code change.",
        ],
    }
    result["computation_id"] = fingerprint(result)
    return result


def render_bridge(packet: dict) -> str:
    """Portable report of recorded aggregates, including audit queries and remainder."""
    if (
        fingerprint({k: v for k, v in packet.items() if k != "computation_id"})
        != packet["computation_id"]
    ):
        raise ValueError("change evidence fingerprint mismatch")

    def e(value):
        return html.escape(str(value), quote=True)

    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{e(row[k])}</td>" for k in ["segment", "cohort", "previous", "current", "delta"]
        )
        + "</tr>"
        for row in packet["rows"]
    )
    r = packet["remainder"]
    rows += (
        f"<tr><td>All other groups ({r['group_count']})</td><td>Remainder</td>"
        + "".join(f"<td>{e(r[k])}</td>" for k in ["previous", "current", "delta"])
        + "</tr>"
    )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><title>Retail change breakdown</title>'
        "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:16px;color:#172b44}"
        "table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;"
        "text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}h1{color:#145e65}</style>"
        f"<h1>{e(packet['metric'])} change by {e(packet['group_by'])}</h1>"
        f"<p>{e(packet['previous'])} → {e(packet['current'])} {e(packet['unit'])}; "
        f"change: <strong>{e(packet['delta'])}</strong></p>"
        "<table><thead><tr><th>Group</th><th>Observed cohort</th><th>Previous</th><th>Current</th>"
        f"<th>Change</th></tr></thead><tbody>{rows}</tbody></table>"
        f"<p>Reconciliation residual: {packet['reconciliation_residual_base_units']} base units. "
        f"All {packet['group_count']} groups were included before display limiting.</p>"
        + "".join(f"<p>{e(w)}</p>" for w in packet["warnings"])
        + f"<h2>Complete evidence</h2><pre>{e(json.dumps(packet, indent=2))}</pre></html>"
    )
