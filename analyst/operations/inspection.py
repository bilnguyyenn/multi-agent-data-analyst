"""Bounded source inspection of a recorded contribution, with exact reconciliation."""

from __future__ import annotations

import hashlib
import html
import json
import threading
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from analyst.operations.metrics import (
    COLUMNS,
    ELIGIBLE,
    ENGINE_SHA256,
    PREPARATION_SQL,
    contract_hash,
    connect,
    fingerprint,
    prepare,
)

VERSION = "retail-source-inspection-v1"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
MAX_PACKET_BYTES = 512 * 1024


class InspectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segment: str = Field(max_length=1000)
    sample_limit: int = Field(default=10, ge=1, le=25)


def money(value: int) -> str:
    return f"{'-' if value < 0 else ''}{abs(value) // 100}.{abs(value) % 100:02d}"


def inspect_period(path: Path, parent: dict, index: int, request: InspectionRequest) -> dict:
    dimension = parent["group_by"]
    if dimension not in {"country", "product"}:
        raise ValueError("unsupported inspection dimension")
    period = parent["periods"][index]
    prefix = "CREATE TEMP VIEW lines AS SELECT\n"
    if not PREPARATION_SQL.startswith(prefix):
        raise ValueError("metric preparation changed; update the inspection projection")
    projection = ", ".join(f'raw."{name}" AS "source_{name}"' for name in COLUMNS)
    preparation = (
        "CREATE TEMP VIEW inspected AS SELECT " + projection + ", " + PREPARATION_SQL[len(prefix) :]
    )
    filters = [
        ELIGIBLE,
        "invoice IS NOT NULL",
        "invoice <> ''",
        "occurred IS NOT NULL",
        "occurred >= ?",
        "occurred < ?",
        f"COALESCE({dimension}, '(missing)') = ?",
    ]
    parameters = [period["start"], period["end"], request.segment]
    if period["country"] is not None:
        filters.append("country = ?")
        parameters.append(period["country"])
    where = " AND ".join(filters)
    amounts = """COUNT(*) AS eligible_rows,
        COALESCE(SUM(CASE WHEN NOT cancelled AND qty > 0 THEN qty * pence ELSE 0 END), 0) AS gross_pence,
        COALESCE(SUM(CASE WHEN cancelled AND qty < 0 THEN -qty * pence ELSE 0 END), 0) AS cancellation_pence,
        COALESCE(SUM(qty * pence), 0) AS net_pence,
        COALESCE(SUM(CASE WHEN NOT cancelled AND qty > 0 THEN qty ELSE 0 END), 0) AS units_sold"""
    totals_sql = f"SELECT {amounts} FROM inspected WHERE {where}"
    daily_sql = f"SELECT CAST(occurred AS DATE) AS day, {amounts} FROM inspected WHERE {where} GROUP BY 1 ORDER BY 1"
    source_columns = ", ".join(f'"source_{name}"' for name in COLUMNS)
    sample_sql = (
        f"SELECT {source_columns}, qty * pence AS signed_pence FROM inspected WHERE {where} "
        f"ORDER BY occurred, {source_columns} NULLS FIRST LIMIT ?"
    )
    names = ["eligible_rows", "gross_pence", "cancellation_pence", "net_pence", "units_sold"]
    connection = connect()
    timer = threading.Timer(30, connection.interrupt)
    timer.daemon = True
    timer.start()
    try:
        prepare(connection, path)
        connection.execute(preparation)
        totals = dict(zip(names, map(int, connection.execute(totals_sql, parameters).fetchone())))
        daily = connection.execute(daily_sql, parameters).fetchmany(3661)
        if len(daily) > 3660:
            raise ValueError("inspection exceeds 3,660 observed days")
        sample = connection.execute(sample_sql, [*parameters, request.sample_limit]).fetchall()
        result = {
            "period": period,
            "snapshot_hash": parent["snapshot_hashes"][index],
            "totals": {
                **totals,
                **{
                    key.replace("_pence", ""): money(totals[key])
                    for key in ["gross_pence", "cancellation_pence", "net_pence"]
                },
            },
            "daily": [
                {"day": day.isoformat(), **dict(zip(names, map(int, values)))}
                for day, *values in daily
            ],
            "sample": [
                {
                    "source": dict(zip(COLUMNS, row[:-1])),
                    "signed_pence": int(row[-1]),
                    "signed_value": money(int(row[-1])),
                }
                for row in sample
            ],
            "sample_limit": request.sample_limit,
            "sampled_rows": len(sample),
            "omitted_rows": totals["eligible_rows"] - len(sample),
            "queries": [
                {"sql": totals_sql, "parameters": parameters},
                {"sql": daily_sql, "parameters": parameters},
                {"sql": sample_sql, "parameters": [*parameters, request.sample_limit]},
            ],
            "preparation_sql": preparation,
        }
        result["totals"]["units_sold_display"] = str(totals["units_sold"])
        for day in result["daily"]:
            day.update(
                gross=money(day["gross_pence"]),
                cancellation=money(day["cancellation_pence"]),
                net=money(day["net_pence"]),
                units_sold_display=str(day["units_sold"]),
            )
        for name in names:
            if sum(row[name] for row in result["daily"]) != totals[name]:
                raise ValueError("daily totals do not reconcile")
        return result
    finally:
        timer.cancel()
        timer.join()
        connection.close()


def inspection(parent: dict, request: InspectionRequest, paths: list[Path]) -> dict:
    if parent.get("version") != "retail-change-bridge-v1":
        raise ValueError("inspection requires contribution evidence")
    if (
        parent["metric_contract_hash"] != contract_hash()
        or parent["metric_engine_sha256"] != ENGINE_SHA256
    ):
        raise ValueError("recompute the contribution with the current metric engine")
    matches = [row for row in parent["rows"] if row["segment"] == request.segment]
    if len(matches) != 1:
        raise ValueError("select a contributor displayed in this breakdown")
    contributor = matches[0]
    periods = [inspect_period(path, parent, index, request) for index, path in enumerate(paths)]
    field = {
        "net_sales": "net_pence",
        "gross_sales": "gross_pence",
        "cancellation_value": "cancellation_pence",
        "units_sold": "units_sold",
    }[parent["metric"]]
    for label, period in zip(["previous", "current"], periods):
        if period["totals"][field] != contributor[f"{label}_base_units"]:
            raise ValueError("source totals do not reconcile to recorded contribution")
        if period["totals"]["eligible_rows"] != contributor[f"{label}_source_rows"]:
            raise ValueError("source row counts do not reconcile to recorded contribution")
    result = {
        "version": VERSION,
        "parent_computation_id": parent["computation_id"],
        "metric": parent["metric"],
        "unit": parent["unit"],
        "group_by": parent["group_by"],
        "segment": request.segment,
        "contributor": contributor,
        "periods": periods,
        "metric_contract_hash": contract_hash(),
        "metric_engine_sha256": ENGINE_SHA256,
        "source_sha256": SOURCE_SHA256,
        "reconciliation_residual_base_units": 0,
        "warnings": [
            "Eligible source entries explain the recorded accounting change, not its cause or business impact.",
            "The sample is the first matching rows in deterministic order, not a random or representative sample. Duplicates are retained.",
            "Samples include only lines eligible under the metric contract; they are not the full raw-data quality report.",
            "Descriptions are recorded source text, not verified product classifications. An absent period is not proof of no business activity.",
            "Daily rows cover observed eligible dates; unlisted dates are not imputed observations.",
        ],
    }
    result["computation_id"] = fingerprint(result)
    if len(json.dumps(result).encode()) > MAX_PACKET_BYTES:
        raise ValueError("inspection exceeds 512 KiB; reduce the source sample limit")
    return result


def render_inspection(packet: dict) -> str:
    if (
        packet.get("version") != VERSION
        or fingerprint({k: v for k, v in packet.items() if k != "computation_id"})
        != packet["computation_id"]
    ):
        raise ValueError("inspection evidence fingerprint mismatch")

    def e(value):
        return html.escape(str(value), quote=True)

    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><title>Retail source inspection</title>'
        "<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px;color:#172b44}"
        "table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #ddd;padding:8px;text-align:left}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}.scroll{overflow:auto}</style>",
        f"<h1>Source inspection: {e(packet['segment'])}</h1>",
        f"<p>{e(packet['metric'])} change: {e(packet['contributor']['delta'])} {e(packet['unit'])}</p>",
    ]
    for label, period in zip(["Previous", "Current"], packet["periods"]):
        totals = period["totals"]
        parts += [
            f"<h2>{label}: {e(period['period']['start'])} → {e(period['period']['end'])} (exclusive)</h2>",
            f"<p>Gross: {totals['gross']} GBP · Cancellations: {totals['cancellation']} GBP · Net: {totals['net']} GBP · Units sold: {totals['units_sold']}</p>",
            f"<p>Showing {period['sampled_rows']} of {totals['eligible_rows']} eligible rows; {period['omitted_rows']} omitted.</p>",
            '<div class="scroll"><table><thead><tr>'
            + "".join(f"<th>{e(c)}</th>" for c in COLUMNS)
            + "<th>Signed GBP</th></tr></thead><tbody>",
        ]
        for row in period["sample"]:
            parts.append(
                "<tr>"
                + "".join(
                    f"<td>{e(row['source'][c]) if row['source'][c] is not None else '(missing)'}</td>"
                    for c in COLUMNS
                )
                + f"<td>{e(row['signed_value'])}</td></tr>"
            )
        parts.append("</tbody></table></div>")
        parts.append(
            "<details><summary>Daily eligible totals (GBP)</summary><table><thead><tr>"
            "<th>Date</th><th>Rows</th><th>Gross</th><th>Cancellations</th><th>Net</th>"
            "</tr></thead><tbody>"
        )
        for day in period["daily"]:
            parts.append(
                "<tr>"
                + "".join(
                    f"<td>{e(day[key])}</td>"
                    for key in ["day", "eligible_rows", "gross", "cancellation", "net"]
                )
                + "</tr>"
            )
        parts.append("</tbody></table></details>")
    parts += [f"<p>{e(w)}</p>" for w in packet["warnings"]]
    parts.append(
        f"<h2>Complete evidence and daily totals</h2><pre>{e(json.dumps(packet, indent=2))}</pre></html>"
    )
    return "".join(parts)
