"""Trusted DuckDB metric queries; model/user text never becomes executable SQL."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

import duckdb
from pydantic import BaseModel, ConfigDict, Field, model_validator

VERSION = "retail-metrics-v1"
ENGINE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PREPARATION_SQL = """CREATE TEMP VIEW lines AS SELECT
        trim(InvoiceNo) AS invoice, trim(StockCode) AS product,
        NULLIF(NULLIF(trim(CustomerID), ''), 'nan') AS customer,
        COALESCE(NULLIF(trim(Country), ''), '(missing)') AS country,
        CASE WHEN regexp_full_match(trim(Quantity), '[+-]?[0-9]+([.]0+)?')
             THEN try_cast(Quantity AS BIGINT) ELSE NULL END AS qty,
        CASE WHEN regexp_full_match(trim(UnitPrice), '[+-]?[0-9]+([.][0-9]{1,4})?')
             THEN try_cast(UnitPrice AS DECIMAL(18,4)) ELSE NULL END AS price,
        round(try_cast(UnitPrice AS DECIMAL(18,4)) * 100)::BIGINT AS pence,
        coalesce(try_cast(InvoiceDate AS TIMESTAMP),
                 try_strptime(InvoiceDate, '%m/%d/%Y %H:%M')) AS occurred,
        starts_with(upper(trim(InvoiceNo)), 'C') AS cancelled FROM raw"""
COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]
ELIGIBLE = (
    "price > 0 AND qty IS NOT NULL AND ((NOT cancelled AND qty > 0) OR (cancelled AND qty < 0))"
)
POSITIVE = "NOT cancelled AND qty > 0"
NEGATIVE = "cancelled AND qty < 0"
REGISTRY = {
    "gross_sales": {
        "unit": "GBP",
        "formula": f"SUM(CASE WHEN {POSITIVE} THEN qty * pence ELSE 0 END)",
        "definition": "Positive-quantity, non-cancellation invoice lines at positive prices; rounded line unit prices in pence.",
    },
    "cancellation_value": {
        "unit": "GBP",
        "formula": f"SUM(CASE WHEN {NEGATIVE} THEN -qty * pence ELSE 0 END)",
        "definition": "Absolute value of negative-quantity cancellation invoice lines; not verified physical returns.",
    },
    "net_sales": {
        "unit": "GBP",
        "formula": "SUM(qty * pence)",
        "definition": "Gross sales minus cancellation value on eligible lines; not profit or cash receipts.",
    },
    "order_count": {
        "unit": "invoices",
        "formula": f"COUNT(DISTINCT CASE WHEN {POSITIVE} THEN invoice END)",
        "definition": "Distinct non-cancellation invoices with at least one eligible positive-quantity line.",
    },
    "units_sold": {
        "unit": "units",
        "formula": f"SUM(CASE WHEN {POSITIVE} THEN qty ELSE 0 END)",
        "definition": "Sum of positive quantities on eligible non-cancellation lines.",
    },
    "active_customers": {
        "unit": "customers",
        "formula": f"COUNT(DISTINCT CASE WHEN {POSITIVE} THEN customer END)",
        "definition": "Distinct non-missing customer identifiers on eligible non-cancellation lines.",
    },
    "cancellation_rate": {
        "unit": "fraction",
        "formula": "COUNT(DISTINCT CASE WHEN cancelled THEN invoice END)",
        "definition": "Cancellation invoice count / all eligible distinct invoices; not a cohort cancellation probability.",
    },
}


class MetricRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: Literal[
        "gross_sales",
        "cancellation_value",
        "net_sales",
        "order_count",
        "units_sold",
        "active_customers",
        "cancellation_rate",
    ]
    group_by: Literal["country", "product", "customer"] | None = None
    country: str | None = Field(default=None, max_length=160)
    start: date | None = None
    end: date | None = None
    limit: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def dates(self):
        if bool(self.start) != bool(self.end) or (self.start and self.start >= self.end):
            raise ValueError("supply both dates with start < end (end is exclusive)")
        return self


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def contract_hash() -> str:
    return fingerprint(
        {
            "version": VERSION,
            "registry": REGISTRY,
            "eligibility": ELIGIBLE,
            "preparation_sql": PREPARATION_SQL,
        }
    )


def connect() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(config={"memory_limit": "256MB", "threads": "1"})
    return connection


def prepare(connection, path: Path) -> None:
    # all_varchar prevents identifiers being converted to floats or leading-zero loss.
    connection.read_csv(str(path), all_varchar=True).create_view("raw")
    names = [column[0] for column in connection.execute("DESCRIBE raw").fetchall()]
    if set(names) != set(COLUMNS) or len(names) != len(COLUMNS):
        raise ValueError("retail CSV must contain exactly the eight UCI Online Retail columns")
    connection.execute(PREPARATION_SQL)


def compute(connection, request: MetricRequest, snapshot_hash: str) -> dict:
    definition = REGISTRY[request.metric]
    filters, parameters = (
        [ELIGIBLE, "invoice IS NOT NULL", "invoice <> ''", "occurred IS NOT NULL"],
        [],
    )
    if request.start:
        filters += ["occurred >= ?", "occurred < ?"]
        parameters += [request.start.isoformat(), request.end.isoformat()]
    if request.country is not None:
        filters.append("country = ?")
        parameters.append(request.country)
    where = " AND ".join(filters)
    # group_by comes from a closed enum. No SQL snippets are accepted by this API.
    group = f"COALESCE({request.group_by}, '(missing)')" if request.group_by else "'all'"
    sql = f"""SELECT {group} AS segment, COALESCE({definition["formula"]}, 0) AS numerator,
              COUNT(DISTINCT invoice) AS invoice_denominator, COUNT(*) AS source_rows
              FROM lines WHERE {where} GROUP BY 1 ORDER BY numerator DESC, segment ASC"""
    rows = connection.execute(sql, parameters).fetchall()
    if not rows and not request.group_by:
        rows = [("all", 0, 0, 0)]
    values = []
    for segment, numerator, denominator, row_count in rows[: request.limit]:
        value = (
            float(numerator) / 100
            if definition["unit"] == "GBP"
            else float(numerator) / denominator
            if request.metric == "cancellation_rate" and denominator
            else None
            if request.metric == "cancellation_rate"
            else int(numerator)
        )
        values.append(
            {
                "segment": segment,
                "value": value,
                "numerator": int(numerator),
                "denominator": denominator if request.metric == "cancellation_rate" else None,
                "source_rows": row_count,
            }
        )
    quality = connection.execute("""SELECT COUNT(*),
        COUNT(*) FILTER (WHERE qty IS NULL OR price IS NULL OR occurred IS NULL OR invoice IS NULL OR invoice=''),
        COUNT(*) FILTER (WHERE price <= 0),
        COUNT(*) FILTER (WHERE (cancelled AND qty >= 0) OR (NOT cancelled AND qty <= 0)),
        COUNT(*) FILTER (WHERE customer IS NULL) FROM lines""").fetchone()
    duplicate_rows = connection.execute(
        "SELECT (SELECT COUNT(*) FROM raw) - (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM raw))"
    ).fetchone()[0]
    warnings = [
        "Descriptive contribution only; no causal effect or profit is inferred.",
        "Invoice timestamps are naive source-local times; no timezone conversion is assumed.",
        "Exact duplicate rows are retained; source corrections require a new snapshot.",
    ]
    if request.metric == "active_customers" and quality[4]:
        warnings.append(
            "Missing customer identifiers are excluded from distinct-customer counts; the count covers known IDs only."
        )
    if request.group_by and request.metric in {
        "active_customers",
        "order_count",
        "cancellation_rate",
    }:
        warnings.append(
            "Grouped distinct counts and rates are not additive; do not sum group values to infer an overall result."
        )
    if not values or all(row["source_rows"] == 0 for row in values):
        warnings.append("No eligible rows in this window; a rate is undefined rather than zero.")
    if request.group_by and len(rows) > request.limit:
        warnings.append(
            f"Showing {request.limit} of {len(rows)} groups; displayed groups are not the overall total."
        )
    packet = {
        "metric_version": VERSION,
        "metric": request.metric,
        "unit": definition["unit"],
        "definition": definition["definition"],
        "grain": "source invoice-product line",
        "snapshot_hash": snapshot_hash,
        "request": request.model_dump(mode="json"),
        "sql": sql,
        "preparation_sql": PREPARATION_SQL,
        "metric_contract_hash": contract_hash(),
        "engine_sha256": ENGINE_SHA256,
        "parameters": parameters,
        "values": values,
        "warnings": warnings,
        "quality": dict(
            zip(
                [
                    "source_rows",
                    "invalid_fields",
                    "nonpositive_prices",
                    "unexpected_quantity_sign",
                    "missing_customer_ids",
                ],
                quality,
            )
        ),
        "duplicate_rows_retained": duplicate_rows,
        "group_count": len(rows),
    }
    packet["computation_id"] = fingerprint(packet)
    return packet


def resolve_question(question: str, metric: str | None = None) -> dict:
    """Conservative deterministic front door; never claim general NL understanding."""
    text = question.lower()
    if any(word in text for word in ["profit", "margin", "cause", "forecast", "predict"]):
        return {
            "status": "unsupported",
            "reason": "This workflow supports descriptive retail metrics; the requested outcome needs additional data or methods.",
        }
    if metric:
        if metric not in REGISTRY:
            raise ValueError("unknown metric")
        return {"status": "resolved", "metric": metric}
    matches = [name for name in REGISTRY if name.replace("_", " ") in text]
    if len(matches) == 1:
        return {"status": "resolved", "metric": matches[0]}
    return {
        "status": "clarification_required",
        "reason": "Choose the intended metric and confirm its definition.",
        "choices": [
            {"metric": name, **{k: v for k, v in spec.items() if k != "formula"}}
            for name, spec in REGISTRY.items()
        ],
    }
