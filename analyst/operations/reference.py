"""Independent row-wise Decimal reference for metric evaluation; no DuckDB/engine formulas."""

from __future__ import annotations

import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path


def reference_values(path: Path, request: dict) -> list[dict]:
    groups: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            qparts = row["Quantity"].strip().lstrip("+-").split(".")
            pparts = row["UnitPrice"].strip().lstrip("+-").split(".")
            if (
                len(qparts) > 2
                or not qparts[0].isdigit()
                or (len(qparts) == 2 and (not qparts[1] or set(qparts[1]) != {"0"}))
                or len(pparts) > 2
                or not pparts[0].isdigit()
                or (len(pparts) == 2 and (not pparts[1].isdigit() or len(pparts[1]) > 4))
            ):
                continue
            try:
                quantity, price = Decimal(row["Quantity"]), Decimal(row["UnitPrice"])
                if (
                    not quantity.is_finite()
                    or not price.is_finite()
                    or quantity != quantity.to_integral_value()
                ):
                    continue
                timestamp = datetime.fromisoformat(row["InvoiceDate"])
            except (ValueError, InvalidOperation):
                continue
            invoice = row["InvoiceNo"].strip()
            cancelled = invoice.upper().startswith("C")
            positive = not cancelled and quantity > 0
            negative = cancelled and quantity < 0
            if not invoice or price <= 0 or not (positive or negative):
                continue
            if request.get("start") and not (
                request["start"] <= timestamp.date().isoformat() < request["end"]
            ):
                continue
            country = row["Country"].strip() or "(missing)"
            if request.get("country") is not None and country != request["country"]:
                continue
            customer = row["CustomerID"].strip()
            customer = "" if customer == "nan" else customer
            segment = {
                "country": country,
                "product": row["StockCode"].strip(),
                "customer": customer or "(missing)",
            }.get(request.get("group_by"), "all")
            g = groups.setdefault(
                segment,
                {
                    "gross": 0,
                    "cancel": 0,
                    "units": 0,
                    "rows": 0,
                    "invoices": set(),
                    "orders": set(),
                    "cancellations": set(),
                    "customers": set(),
                },
            )
            amount = int((price * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)) * int(
                quantity
            )
            g["rows"] += 1
            g["invoices"].add(invoice)
            if positive:
                g["gross"] += amount
                g["units"] += int(quantity)
                g["orders"].add(invoice)
                if customer:
                    g["customers"].add(customer)
            else:
                g["cancel"] -= amount
                g["cancellations"].add(invoice)
    if not groups and not request.get("group_by"):
        return [
            {
                "segment": "all",
                "value": None if request["metric"] == "cancellation_rate" else 0,
                "numerator": 0,
                "denominator": 0 if request["metric"] == "cancellation_rate" else None,
                "source_rows": 0,
            }
        ]
    results = []
    for segment, g in groups.items():
        numerators = {
            "gross_sales": g["gross"],
            "cancellation_value": g["cancel"],
            "net_sales": g["gross"] - g["cancel"],
            "units_sold": g["units"],
            "order_count": len(g["orders"]),
            "active_customers": len(g["customers"]),
            "cancellation_rate": len(g["cancellations"]),
        }
        numerator = numerators[request["metric"]]
        denominator = len(g["invoices"]) if request["metric"] == "cancellation_rate" else None
        value = (
            numerator / 100
            if request["metric"] in {"gross_sales", "cancellation_value", "net_sales"}
            else numerator / denominator
            if denominator
            else numerator
        )
        results.append(
            {
                "segment": segment,
                "value": value,
                "numerator": numerator,
                "denominator": denominator,
                "source_rows": g["rows"],
            }
        )
    return sorted(results, key=lambda r: (-r["numerator"], r["segment"]))[
        : request.get("limit", 20)
    ]
