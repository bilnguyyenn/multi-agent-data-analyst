"""Deterministic exploratory-data-analysis and data-quality profiling.

The profiler deliberately uses transparent rules rather than an LLM. Every run on
the same CSV bytes produces the same report, and every issue includes the affected
count/rate and a concrete remediation. The score is a triage heuristic, not a claim
that data quality can be reduced to one scientifically meaningful number.
"""
from __future__ import annotations

import math
import re
import warnings
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from analyst.graph.state import (
    ColumnProfile,
    DataQualityReport,
    QualityIssue,
    RelationshipProfile,
    TableProfile,
    ValueFrequency,
)

_IDENTIFIER_RE = re.compile(r"(^id$|_id$)", re.IGNORECASE)
_DATETIME_NAME_RE = re.compile(r"(^|_)(date|time|timestamp|datetime|created|updated)($|_)", re.I)
_DATETIME_VALUE_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T\s].*)?$")
_SCORE_METHOD = (
    "Heuristic readiness score: 100 minus 8 + 12×affected-rate per error and "
    "2 + 6×affected-rate per warning; informational EDA signals do not reduce the score."
)


def _rate(count: int, denominator: int) -> float:
    return round(count / denominator, 6) if denominator else 0.0


def _number(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return round(converted, 6) if math.isfinite(converted) else None


def _issue(
    code: str,
    severity: str,
    table: str,
    message: str,
    recommendation: str,
    *,
    column: str | None = None,
    affected_count: int = 0,
    denominator: int = 0,
) -> QualityIssue:
    return QualityIssue(
        code=code,
        severity=severity,
        table=table,
        column=column,
        message=message,
        recommendation=recommendation,
        affected_count=affected_count,
        affected_rate=_rate(affected_count, denominator),
    )


def _singular(name: str) -> str:
    lowered = name.casefold()
    if lowered.endswith("ies") and len(lowered) > 3:
        return lowered[:-3] + "y"
    if lowered.endswith("ses") and len(lowered) > 3:
        return lowered[:-2]
    if lowered.endswith("s") and not lowered.endswith("ss"):
        return lowered[:-1]
    return lowered


def _is_primary_key_name(table: str, column: str) -> bool:
    lowered = column.casefold()
    return lowered == "id" or lowered == f"{_singular(table)}_id"


def _datetime_parse(series: pd.Series) -> tuple[pd.Series, float]:
    non_null = series.dropna()
    if non_null.empty:
        return pd.Series(dtype="datetime64[ns, UTC]"), 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        parsed = pd.to_datetime(non_null, errors="coerce", utc=True)
    return parsed, float(parsed.notna().mean())


def _semantic_type(series: pd.Series, column: str, distinct: int, non_null: int) -> str:
    if non_null == 0:
        return "unknown"
    if _IDENTIFIER_RE.search(column):
        return "identifier"
    if ptypes.is_bool_dtype(series.dtype):
        return "boolean"
    if ptypes.is_numeric_dtype(series.dtype):
        return "numeric"
    if ptypes.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    strings = series.dropna().astype(str).str.strip()
    values_look_temporal = bool(
        not strings.empty and strings.str.match(_DATETIME_VALUE_RE).mean() >= 0.8
    )
    if _DATETIME_NAME_RE.search(column) or values_look_temporal:
        _, success_rate = _datetime_parse(series)
        if success_rate >= 0.8:
            return "datetime"
    if distinct <= max(20, int(non_null * 0.2)):
        return "categorical"
    return "text"


def _top_values(series: pd.Series, row_count: int) -> list[ValueFrequency]:
    output = []
    for value, count in series.dropna().astype(str).value_counts().head(5).items():
        output.append(
            ValueFrequency(value=str(value), count=int(count), rate=_rate(int(count), row_count))
        )
    return output


def _profile_column(
    table: str, column: str, series: pd.Series, row_count: int
) -> tuple[ColumnProfile, list[QualityIssue]]:
    issues: list[QualityIssue] = []
    non_null_count = int(series.notna().sum())
    missing_count = row_count - non_null_count
    distinct_count = int(series.nunique(dropna=True))
    semantic_type = _semantic_type(series, column, distinct_count, non_null_count)

    if missing_count:
        if missing_count == row_count:
            severity = "error"
            code = "all_missing"
            message = "Every value is missing."
        elif _rate(missing_count, row_count) >= 0.5:
            severity = "error"
            code = "high_missingness"
            message = f"{_rate(missing_count, row_count):.1%} of values are missing."
        elif _rate(missing_count, row_count) >= 0.05:
            severity = "warning"
            code = "moderate_missingness"
            message = f"{_rate(missing_count, row_count):.1%} of values are missing."
        else:
            severity = "info"
            code = "low_missingness"
            message = f"{_rate(missing_count, row_count):.1%} of values are missing."
        issues.append(
            _issue(
                code,
                severity,
                table,
                message,
                "Define whether missing means unknown, not applicable, or zero before analysis.",
                column=column,
                affected_count=missing_count,
                denominator=row_count,
            )
        )

    if non_null_count and distinct_count <= 1:
        issues.append(
            _issue(
                "constant_column",
                "warning",
                table,
                "The column has only one non-null value and cannot explain variation.",
                "Remove it from modeling features unless the constant is operationally meaningful.",
                column=column,
                affected_count=non_null_count,
                denominator=row_count,
            )
        )

    if semantic_type == "identifier" and _is_primary_key_name(table, column):
        duplicate_ids = max(0, non_null_count - distinct_count)
        if missing_count or duplicate_ids:
            issues.append(
                _issue(
                    "invalid_primary_key",
                    "error",
                    table,
                    f"Likely primary key has {missing_count} missing and {duplicate_ids} duplicate values.",
                    "Enforce non-null uniqueness before joins or deduplication.",
                    column=column,
                    affected_count=missing_count + duplicate_ids,
                    denominator=row_count,
                )
            )

    numeric_summary: dict[str, float] = {}
    datetime_summary: dict[str, str] = {}
    top_values: list[ValueFrequency] = []

    if semantic_type == "numeric":
        numeric = pd.to_numeric(series, errors="coerce").astype(float)
        infinite_count = int(np.isinf(numeric).sum())
        finite = numeric[np.isfinite(numeric)]
        if infinite_count:
            issues.append(
                _issue(
                    "infinite_values",
                    "error",
                    table,
                    f"Found {infinite_count} infinite numeric values.",
                    "Replace infinities with a documented missing-value or capped-value policy.",
                    column=column,
                    affected_count=infinite_count,
                    denominator=row_count,
                )
            )
        if not finite.empty:
            quantiles = finite.quantile([0.25, 0.5, 0.75])
            q1, median, q3 = (float(quantiles.loc[q]) for q in (0.25, 0.5, 0.75))
            iqr = q3 - q1
            outlier_count = 0
            if iqr > 0:
                outlier_count = int(((finite < q1 - 1.5 * iqr) | (finite > q3 + 1.5 * iqr)).sum())
            summary_values = {
                "min": finite.min(),
                "q1": q1,
                "median": median,
                "mean": finite.mean(),
                "q3": q3,
                "max": finite.max(),
                "std": finite.std(ddof=1) if len(finite) > 1 else 0.0,
                "skew": finite.skew() if len(finite) > 2 else 0.0,
                "iqr_outlier_count": outlier_count,
                "iqr_outlier_rate": outlier_count / len(finite),
            }
            numeric_summary = {
                key: converted
                for key, value in summary_values.items()
                if (converted := _number(value)) is not None
            }
            if len(finite) >= 20 and outlier_count:
                issues.append(
                    _issue(
                        "iqr_outliers",
                        "info",
                        table,
                        f"{outlier_count} values fall outside the 1.5×IQR fences.",
                        "Inspect these records; use robust statistics rather than deleting them automatically.",
                        column=column,
                        affected_count=outlier_count,
                        denominator=len(finite),
                    )
                )
            skew = numeric_summary.get("skew", 0.0)
            if len(finite) >= 20 and abs(skew) >= 2:
                issues.append(
                    _issue(
                        "high_skew",
                        "info",
                        table,
                        f"The numeric distribution is highly skewed (skew={skew:.2f}).",
                        "Report medians/quantiles or consider a justified transformation for modeling.",
                        column=column,
                    )
                )
    elif semantic_type == "datetime":
        parsed, success_rate = _datetime_parse(series)
        valid = parsed.dropna()
        invalid_count = non_null_count - len(valid)
        if invalid_count:
            issues.append(
                _issue(
                    "invalid_datetime",
                    "warning",
                    table,
                    f"{invalid_count} non-null values could not be parsed as datetimes.",
                    "Standardize the date format and define a timezone before time-based analysis.",
                    column=column,
                    affected_count=invalid_count,
                    denominator=non_null_count,
                )
            )
        if not valid.empty:
            datetime_summary = {
                "min": valid.min().isoformat(),
                "max": valid.max().isoformat(),
                "parse_success_rate": f"{success_rate:.6f}",
            }
    else:
        if semantic_type in {"categorical", "boolean"}:
            top_values = _top_values(series, row_count)
        if semantic_type in {"categorical", "text", "identifier"}:
            strings = series.dropna().astype(str)
            whitespace_count = int((strings != strings.str.strip()).sum())
            if whitespace_count:
                issues.append(
                    _issue(
                        "surrounding_whitespace",
                        "warning",
                        table,
                        f"{whitespace_count} values contain leading or trailing whitespace.",
                        "Trim values before grouping, joins, or uniqueness checks.",
                        column=column,
                        affected_count=whitespace_count,
                        denominator=non_null_count,
                    )
                )
            if semantic_type != "identifier" and not strings.empty:
                normalized = strings.str.strip().str.casefold()
                raw_distinct = int(strings.nunique())
                normalized_distinct = int(normalized.nunique())
                if normalized_distinct < raw_distinct:
                    issues.append(
                        _issue(
                            "inconsistent_labels",
                            "warning",
                            table,
                            f"Case/whitespace normalization merges {raw_distinct - normalized_distinct} label variants.",
                            "Create a canonical category mapping before aggregation or modeling.",
                            column=column,
                            affected_count=raw_distinct - normalized_distinct,
                            denominator=raw_distinct,
                        )
                    )
            if semantic_type not in {"identifier", "datetime"} and non_null_count:
                numeric_rate = float(pd.to_numeric(strings, errors="coerce").notna().mean())
                if 0.05 <= numeric_rate <= 0.95:
                    issues.append(
                        _issue(
                            "mixed_numeric_text",
                            "warning",
                            table,
                            f"{numeric_rate:.1%} of text values parse as numbers, suggesting mixed types.",
                            "Separate numeric values from labels or document the encoding explicitly.",
                            column=column,
                            affected_count=int(round(numeric_rate * non_null_count)),
                            denominator=non_null_count,
                        )
                    )

    if row_count >= 20 and distinct_count / max(non_null_count, 1) >= 0.95:
        if semantic_type in {"categorical", "text"}:
            issues.append(
                _issue(
                    "high_cardinality",
                    "info",
                    table,
                    f"{_rate(distinct_count, non_null_count):.1%} of non-null values are distinct.",
                    "Avoid naive one-hot encoding; verify whether this is an identifier or free text.",
                    column=column,
                    affected_count=distinct_count,
                    denominator=non_null_count,
                )
            )

    profile = ColumnProfile(
        name=column,
        dtype=str(series.dtype),
        semantic_type=semantic_type,
        row_count=row_count,
        non_null_count=non_null_count,
        missing_count=missing_count,
        missing_rate=_rate(missing_count, row_count),
        distinct_count=distinct_count,
        distinct_rate=_rate(distinct_count, non_null_count),
        numeric_summary=numeric_summary,
        datetime_summary=datetime_summary,
        top_values=top_values,
    )
    return profile, issues


def _profile_table(name: str, frame: pd.DataFrame) -> tuple[TableProfile, list[QualityIssue]]:
    row_count = len(frame)
    issues: list[QualityIssue] = []
    if row_count == 0:
        issues.append(
            _issue(
                "empty_table",
                "error",
                name,
                "The table has no data rows.",
                "Load records or exclude the table from the analysis contract.",
            )
        )
    duplicate_rows = int(frame.duplicated().sum()) if row_count else 0
    if duplicate_rows:
        issues.append(
            _issue(
                "duplicate_rows",
                "warning",
                name,
                f"Found {duplicate_rows} fully duplicated rows.",
                "Confirm the intended row grain, then deduplicate with an auditable key.",
                affected_count=duplicate_rows,
                denominator=row_count,
            )
        )

    columns: list[ColumnProfile] = []
    candidate_keys: list[str] = []
    for raw_column in frame.columns:
        column = str(raw_column)
        profile, column_issues = _profile_column(name, column, frame[raw_column], row_count)
        columns.append(profile)
        issues.extend(column_issues)
        if (
            row_count > 0
            and profile.missing_count == 0
            and profile.distinct_count == row_count
            and profile.semantic_type == "identifier"
        ):
            candidate_keys.append(column)

    return (
        TableProfile(
            name=name,
            row_count=row_count,
            column_count=len(frame.columns),
            duplicate_rows=duplicate_rows,
            duplicate_rate=_rate(duplicate_rows, row_count),
            candidate_keys=candidate_keys,
            columns=columns,
        ),
        issues,
    )


def _infer_relationships(
    frames: dict[str, pd.DataFrame], table_profiles: list[TableProfile]
) -> tuple[list[RelationshipProfile], list[QualityIssue]]:
    profiles_by_table = {profile.name: profile for profile in table_profiles}
    locations: dict[str, list[str]] = defaultdict(list)
    original_names: dict[tuple[str, str], str] = {}
    for table, frame in frames.items():
        for raw_column in frame.columns:
            column = str(raw_column)
            if column.casefold().endswith("_id"):
                locations[column.casefold()].append(table)
                original_names[(table, column.casefold())] = column

    relationships: list[RelationshipProfile] = []
    issues: list[QualityIssue] = []
    for normalized_column, tables in sorted(locations.items()):
        if len(tables) < 2:
            continue
        unique_candidates: list[tuple[int, str]] = []
        stem = normalized_column[:-3]
        for table in tables:
            column = original_names[(table, normalized_column)]
            profile = next(c for c in profiles_by_table[table].columns if c.name == column)
            if profile.missing_count == 0 and profile.distinct_count == profile.row_count:
                name_match = 1 if _singular(table) == stem else 0
                unique_candidates.append((name_match, table))
        if not unique_candidates:
            continue
        unique_candidates.sort(reverse=True)
        best_score = unique_candidates[0][0]
        best = [table for score, table in unique_candidates if score == best_score]
        if len(best) != 1:
            continue
        parent = best[0]
        parent_column = original_names[(parent, normalized_column)]
        parent_values = set(frames[parent][parent_column].dropna().astype(str))
        for child in sorted(table for table in tables if table != parent):
            child_column = original_names[(child, normalized_column)]
            child_values = frames[child][child_column].dropna().astype(str)
            orphan_count = int((~child_values.isin(parent_values)).sum())
            relationship = RelationshipProfile(
                column=child_column,
                parent_table=parent,
                child_table=child,
                child_non_null_count=len(child_values),
                orphan_count=orphan_count,
                orphan_rate=_rate(orphan_count, len(child_values)),
            )
            relationships.append(relationship)
            if orphan_count:
                rate = _rate(orphan_count, len(child_values))
                issues.append(
                    _issue(
                        "orphan_foreign_keys",
                        "error" if rate >= 0.05 else "warning",
                        child,
                        f"{orphan_count} values do not exist in {parent}.{parent_column}.",
                        "Resolve orphan keys before joins so rows are not silently dropped.",
                        column=child_column,
                        affected_count=orphan_count,
                        denominator=len(child_values),
                    )
                )
    return relationships, issues


def profile_dataset(frames: dict[str, pd.DataFrame]) -> DataQualityReport:
    """Create a compact deterministic EDA/data-quality report for loaded CSV tables."""
    table_profiles: list[TableProfile] = []
    issues: list[QualityIssue] = []
    for name in sorted(frames):
        table_profile, table_issues = _profile_table(name, frames[name])
        table_profiles.append(table_profile)
        issues.extend(table_issues)

    relationships, relationship_issues = _infer_relationships(frames, table_profiles)
    issues.extend(relationship_issues)
    severity_order = {"error": 0, "warning": 1, "info": 2}
    issues.sort(
        key=lambda item: (
            severity_order[item.severity], item.table, item.column or "", item.code
        )
    )
    counts = Counter(issue.severity for issue in issues)
    penalties = 0.0
    for issue in issues:
        if issue.severity == "error":
            penalties += 8 + 12 * issue.affected_rate
        elif issue.severity == "warning":
            penalties += 2 + 6 * issue.affected_rate
    score = round(max(0.0, 100.0 - penalties), 1)
    issue_counts = {severity: int(counts.get(severity, 0)) for severity in ("error", "warning", "info")}
    total_rows = sum(len(frame) for frame in frames.values())
    summary = (
        f"Profiled {len(frames)} tables and {total_rows} rows: "
        f"{issue_counts['error']} errors, {issue_counts['warning']} warnings, "
        f"and {issue_counts['info']} informational EDA signals."
    )
    return DataQualityReport(
        quality_score=score,
        score_method=_SCORE_METHOD,
        table_count=len(frames),
        total_rows=total_rows,
        issue_counts=issue_counts,
        summary=summary,
        tables=table_profiles,
        relationships=relationships,
        issues=issues,
    )
