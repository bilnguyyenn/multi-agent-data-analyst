from __future__ import annotations

import pandas as pd

from analyst.data import build_manifest
from analyst.quality import profile_dataset


def test_profile_finds_dirty_keys_labels_missingness_and_orphans():
    customers = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "segment": ["North", " north ", "South"],
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": [10, 11, 12, 12],
            "customer_id": [1, 2, 999, 999],
            "amount": [10.0, None, 1000.0, 1000.0],
            "status": ["Open", "open", "closed", "closed"],
            "unused": [None, None, None, None],
        }
    )

    report = profile_dataset({"customers": customers, "orders": orders})
    codes = {(issue.table, issue.column, issue.code) for issue in report.issues}

    assert report.table_count == 2
    assert report.total_rows == 7
    assert report.quality_score < 100
    assert report.issue_counts["error"] >= 3
    assert ("orders", None, "duplicate_rows") in codes
    assert ("orders", "order_id", "invalid_primary_key") in codes
    assert ("orders", "unused", "all_missing") in codes
    assert ("orders", "amount", "moderate_missingness") in codes
    assert ("customers", "segment", "surrounding_whitespace") in codes
    assert ("customers", "segment", "inconsistent_labels") in codes
    assert ("orders", "status", "inconsistent_labels") in codes
    assert ("orders", "customer_id", "orphan_foreign_keys") in codes

    customers_profile = next(table for table in report.tables if table.name == "customers")
    assert customers_profile.candidate_keys == ["customer_id"]
    relationship = next(item for item in report.relationships if item.column == "customer_id")
    assert relationship.parent_table == "customers"
    assert relationship.child_table == "orders"
    assert relationship.orphan_count == 2
    assert relationship.orphan_rate == 0.5


def test_profile_produces_numeric_datetime_and_frequency_eda():
    frame = pd.DataFrame(
        {
            "event_id": [f"e-{index}" for index in range(21)],
            "event_date": [f"2026-01-{index + 1:02d}" for index in range(20)] + ["bad-date"],
            "value": list(range(20)) + [1000],
            "category": ["A"] * 14 + ["B"] * 7,
        }
    )

    report = profile_dataset({"events": frame})
    table = report.tables[0]
    columns = {column.name: column for column in table.columns}

    assert columns["event_date"].semantic_type == "datetime"
    assert columns["event_date"].datetime_summary["min"].startswith("2026-01-01")
    assert columns["event_date"].datetime_summary["max"].startswith("2026-01-20")
    assert columns["value"].numeric_summary["median"] == 10.0
    assert columns["value"].numeric_summary["max"] == 1000.0
    assert columns["value"].numeric_summary["iqr_outlier_count"] == 1.0
    assert columns["category"].top_values[0].value == "A"
    assert columns["category"].top_values[0].count == 14
    assert any(issue.code == "invalid_datetime" for issue in report.issues)
    assert any(issue.code == "iqr_outliers" for issue in report.issues)


def test_manifest_embeds_report_and_counts_quoted_newline_rows(tmp_path):
    (tmp_path / "notes.csv").write_text('note_id,text\n1,"first line\nsecond line"\n2,plain\n')

    first = build_manifest(tmp_path)
    second = build_manifest(tmp_path)

    assert first.tables[0].row_count == 2
    assert first.quality_report is not None
    assert first.quality_report.total_rows == 2
    assert first.quality_report.model_dump() == second.quality_report.model_dump()
    assert "DATA QUALITY:" in first.schema_text()
