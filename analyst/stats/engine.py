"""Deterministic confidence intervals, effect sizes, and hypothesis tests.

Executors remain responsible for data preparation, but statistical arithmetic no
longer depends on an LLM. An executor may attach a ``statistical_tests`` list to its
small ``result.json`` payload. Each request contains sufficient statistics, not raw
rows; this module validates those values and produces typed, auditable evidence for
the validator and final report.

Supported request kinds:

* ``one_proportion``: one group with ``successes`` and ``n``
* ``two_proportion``: two such groups
* ``one_mean``: one group with ``mean``, sample ``std``, and ``n``
* ``two_mean``: two such groups (Welch inference + Hedges' g)
* ``chi_square``: an r x c non-negative count matrix
* ``stratified_proportion``: group x stratum successes/totals for direct standardization
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, ValidationError
from scipy import stats

from analyst.graph.state import (
    Artifact,
    ConfidenceInterval,
    EffectSize,
    HypothesisTest,
    StatisticalEvidence,
)

Alternative = Literal["two-sided", "less", "greater"]
SUPPORTED_KINDS = {
    "one_proportion", "two_proportion", "one_mean", "two_mean", "chi_square",
    "stratified_proportion",
}


class GroupSummary(BaseModel):
    label: str
    n: int = Field(gt=0)
    successes: int | None = Field(default=None, ge=0)
    mean: float | None = None
    std: float | None = Field(default=None, ge=0.0)


class StratifiedCell(BaseModel):
    group: str
    stratum: str
    successes: int = Field(ge=0)
    n: int = Field(gt=0)


class StatisticalRequest(BaseModel):
    id: str = "test"
    kind: str
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    alternative: Alternative = "two-sided"
    null_value: float | None = None
    groups: list[GroupSummary] = Field(default_factory=list)
    rows: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    counts: list[list[float]] = Field(default_factory=list)
    cells: list[StratifiedCell] = Field(default_factory=list)
    standardization: Literal["pooled", "equal", "custom"] = "pooled"
    reference_weights: dict[str, float] = Field(default_factory=dict)
    direction: Literal["lower", "higher"] = "lower"


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _round(value: float) -> float:
    """Stable JSON-sized precision without hiding small p-values."""
    if value == 0:
        return 0.0
    return float(f"{value:.10g}")


def _p_value(statistic: float, distribution, alternative: Alternative) -> float:
    if alternative == "two-sided":
        return float(2 * distribution.sf(abs(statistic)))
    if alternative == "greater":
        return float(distribution.sf(statistic))
    return float(distribution.cdf(statistic))


def _interpret_standardized(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def _interpret_cramers_v(value: float) -> str:
    if value < 0.1:
        return "negligible"
    if value < 0.3:
        return "small"
    if value < 0.5:
        return "medium"
    return "large"


def _wilson(successes: int, n: int, level: float, parameter: str) -> ConfidenceInterval:
    if successes > n:
        raise ValueError(f"{parameter}: successes cannot exceed n")
    z = float(stats.norm.ppf(1 - (1 - level) / 2))
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return ConfidenceInterval(
        parameter=parameter,
        estimate=_round(p),
        lower=_round(max(0.0, center - half)),
        upper=_round(min(1.0, center + half)),
        level=level,
        method="Wilson score",
    )


def _mean_interval(group: GroupSummary, level: float) -> ConfidenceInterval:
    if group.mean is None or group.std is None:
        raise ValueError(f"{group.label}: mean and std are required")
    if group.n < 2:
        raise ValueError(f"{group.label}: mean inference requires n >= 2")
    mean = _finite(group.mean, f"{group.label}.mean")
    std = _finite(group.std, f"{group.label}.std")
    critical = float(stats.t.ppf(1 - (1 - level) / 2, group.n - 1))
    half = critical * std / math.sqrt(group.n)
    return ConfidenceInterval(
        parameter=f"mean:{group.label}", estimate=_round(mean),
        lower=_round(mean - half), upper=_round(mean + half), level=level,
        method="Student t",
    )


def _hypothesis(
    *, name: str, statistic: float, p_value: float, request: StatisticalRequest,
    null_value: float | None,
) -> HypothesisTest:
    statistic = _finite(statistic, "test statistic")
    p_value = min(1.0, max(0.0, _finite(p_value, "p-value")))
    return HypothesisTest(
        name=name,
        statistic=_round(statistic),
        p_value=_round(p_value),
        alternative=request.alternative,
        null_value=null_value,
        alpha=request.alpha,
        significant=p_value < request.alpha,
    )


def _one_proportion(task_id: str, request: StatisticalRequest) -> StatisticalEvidence:
    if len(request.groups) != 1:
        raise ValueError("one_proportion requires exactly one group")
    group = request.groups[0]
    if group.successes is None:
        raise ValueError(f"{group.label}: successes is required")
    interval = _wilson(group.successes, group.n, request.confidence, f"rate:{group.label}")
    test = None
    effects: list[EffectSize] = []
    if request.null_value is not None:
        null = _finite(request.null_value, "null_value")
        if not 0 <= null <= 1:
            raise ValueError("one_proportion null_value must be between 0 and 1")
        result = stats.binomtest(
            group.successes, group.n, p=null, alternative=request.alternative
        )
        observed = group.successes / group.n
        h = 2 * math.asin(math.sqrt(observed)) - 2 * math.asin(math.sqrt(null))
        effects.append(EffectSize(
            name="Cohen's h", value=_round(h),
            interpretation=_interpret_standardized(h),
        ))
        test = _hypothesis(
            name="exact binomial test", statistic=observed,
            p_value=float(result.pvalue), request=request, null_value=null,
        )
    summary = (f"{group.label} rate={interval.estimate:.4g} "
               f"({request.confidence:.0%} CI {interval.lower:.4g}–{interval.upper:.4g})")
    if test is not None:
        summary += f"; p={test.p_value:.4g}"
    return StatisticalEvidence(
        task_id=task_id, request_id=request.id, kind=request.kind, status="computed",
        sample_size=group.n, intervals=[interval], effects=effects, test=test,
        summary=summary,
    )


def _two_proportion(task_id: str, request: StatisticalRequest) -> StatisticalEvidence:
    if len(request.groups) != 2:
        raise ValueError("two_proportion requires exactly two groups")
    first, second = request.groups
    if first.successes is None or second.successes is None:
        raise ValueError("two_proportion groups require successes")
    first_ci = _wilson(first.successes, first.n, request.confidence, f"rate:{first.label}")
    second_ci = _wilson(second.successes, second.n, request.confidence, f"rate:{second.label}")
    p1, p2 = first.successes / first.n, second.successes / second.n
    difference = p1 - p2
    # Newcombe interval: combine Wilson score limits without a fragile Wald margin.
    lower = difference - math.sqrt(
        (p1 - first_ci.lower) ** 2 + (second_ci.upper - p2) ** 2
    )
    upper = difference + math.sqrt(
        (first_ci.upper - p1) ** 2 + (p2 - second_ci.lower) ** 2
    )
    diff_ci = ConfidenceInterval(
        parameter=f"rate_difference:{first.label}-{second.label}",
        estimate=_round(difference), lower=_round(lower), upper=_round(upper),
        level=request.confidence, method="Newcombe-Wilson",
    )
    null = 0.0 if request.null_value is None else _finite(request.null_value, "null_value")
    if null != 0:
        raise ValueError("two_proportion currently supports null_value=0 only")
    pooled = (first.successes + second.successes) / (first.n + second.n)
    a, b = float(first.successes), float(first.n - first.successes)
    c, d = float(second.successes), float(second.n - second.successes)
    warnings: list[str] = []
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        warnings.append("odds ratio uses a 0.5 continuity correction for a zero cell")
    odds_ratio = (a * d) / (b * c)
    expected = [
        first.n * pooled, first.n * (1 - pooled),
        second.n * pooled, second.n * (1 - pooled),
    ]
    if min(expected) < 5:
        result = stats.fisher_exact(
            [[first.successes, first.n - first.successes],
             [second.successes, second.n - second.successes]],
            alternative=request.alternative,
        )
        test_name = "Fisher exact test"
        test_statistic = odds_ratio
        p_value = float(result.pvalue)
        warnings.append("Fisher exact test selected because an expected cell is below 5")
    else:
        standard_error = math.sqrt(
            pooled * (1 - pooled) * (1 / first.n + 1 / second.n)
        )
        if standard_error == 0:
            raise ValueError("two_proportion test is undefined because pooled variance is zero")
        test_statistic = difference / standard_error
        test_name = "two-proportion z-test"
        p_value = _p_value(test_statistic, stats.norm, request.alternative)
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
    test = _hypothesis(
        name=test_name, statistic=test_statistic, p_value=p_value,
        request=request, null_value=0.0,
    )
    return StatisticalEvidence(
        task_id=task_id, request_id=request.id, kind=request.kind, status="computed",
        sample_size=first.n + second.n,
        intervals=[first_ci, second_ci, diff_ci],
        effects=[
            EffectSize(name="risk difference", value=_round(difference),
                       interpretation="absolute probability difference"),
            EffectSize(name="odds ratio", value=_round(odds_ratio),
                       interpretation=(f"{first.label} vs {second.label}")),
            EffectSize(name="Cohen's h", value=_round(h),
                       interpretation=_interpret_standardized(h)),
        ],
        test=test,
        summary=f"{first.label}−{second.label} rate difference={difference:.4g} "
                f"({request.confidence:.0%} CI {lower:.4g}–{upper:.4g}); p={p_value:.4g}",
        warnings=warnings,
    )


def _one_mean(task_id: str, request: StatisticalRequest) -> StatisticalEvidence:
    if len(request.groups) != 1:
        raise ValueError("one_mean requires exactly one group")
    group = request.groups[0]
    interval = _mean_interval(group, request.confidence)
    test = None
    effects: list[EffectSize] = []
    if request.null_value is not None:
        null = _finite(request.null_value, "null_value")
        if group.std == 0:
            raise ValueError("one_mean test is undefined when std is zero")
        statistic = (float(group.mean) - null) / (float(group.std) / math.sqrt(group.n))
        p_value = _p_value(statistic, stats.t(df=group.n - 1), request.alternative)
        d = (float(group.mean) - null) / float(group.std)
        effects.append(EffectSize(
            name="Cohen's d", value=_round(d), interpretation=_interpret_standardized(d)
        ))
        test = _hypothesis(
            name="one-sample t-test", statistic=statistic, p_value=p_value,
            request=request, null_value=null,
        )
    summary = (f"{group.label} mean={interval.estimate:.4g} "
               f"({request.confidence:.0%} CI {interval.lower:.4g}–{interval.upper:.4g})")
    if test is not None:
        summary += f"; p={test.p_value:.4g}"
    return StatisticalEvidence(
        task_id=task_id, request_id=request.id, kind=request.kind, status="computed",
        sample_size=group.n, intervals=[interval], effects=effects, test=test,
        summary=summary,
    )


def _two_mean(task_id: str, request: StatisticalRequest) -> StatisticalEvidence:
    if len(request.groups) != 2:
        raise ValueError("two_mean requires exactly two groups")
    first, second = request.groups
    first_ci = _mean_interval(first, request.confidence)
    second_ci = _mean_interval(second, request.confidence)
    if first.std == 0 and second.std == 0:
        raise ValueError("two_mean test is undefined when both group std values are zero")
    mean1, mean2 = float(first.mean), float(second.mean)
    var1, var2 = float(first.std) ** 2, float(second.std) ** 2
    difference = mean1 - mean2
    null = 0.0 if request.null_value is None else _finite(request.null_value, "null_value")
    standard_error_sq = var1 / first.n + var2 / second.n
    standard_error = math.sqrt(standard_error_sq)
    statistic = (difference - null) / standard_error
    df_denominator = ((var1 / first.n) ** 2 / (first.n - 1)
                      + (var2 / second.n) ** 2 / (second.n - 1))
    degrees_freedom = standard_error_sq ** 2 / df_denominator
    distribution = stats.t(df=degrees_freedom)
    p_value = _p_value(statistic, distribution, request.alternative)
    critical = float(distribution.ppf(1 - (1 - request.confidence) / 2))
    diff_ci = ConfidenceInterval(
        parameter=f"mean_difference:{first.label}-{second.label}",
        estimate=_round(difference),
        lower=_round(difference - critical * standard_error),
        upper=_round(difference + critical * standard_error),
        level=request.confidence, method="Welch t",
    )

    pooled_df = first.n + second.n - 2
    pooled_std = math.sqrt(
        ((first.n - 1) * var1 + (second.n - 1) * var2) / pooled_df
    )
    effects: list[EffectSize] = [
        EffectSize(name="mean difference", value=_round(difference),
                   interpretation=f"{first.label} minus {second.label}")
    ]
    if pooled_std > 0:
        cohen_d = difference / pooled_std
        correction = 1 - 3 / (4 * pooled_df - 1)
        hedges_g = correction * cohen_d
        effects.append(EffectSize(
            name="Hedges' g", value=_round(hedges_g),
            interpretation=_interpret_standardized(hedges_g),
        ))
    return StatisticalEvidence(
        task_id=task_id, request_id=request.id, kind=request.kind, status="computed",
        sample_size=first.n + second.n,
        intervals=[first_ci, second_ci, diff_ci], effects=effects,
        test=_hypothesis(
            name="Welch's t-test", statistic=statistic, p_value=p_value,
            request=request, null_value=null,
        ),
        summary=f"{first.label}−{second.label} mean difference={difference:.4g} "
                f"({request.confidence:.0%} CI {diff_ci.lower:.4g}–{diff_ci.upper:.4g}); "
                f"p={p_value:.4g}",
    )


def _reference_weights(
    request: StatisticalRequest,
    strata: list[str],
    cells: dict[tuple[str, str], StratifiedCell],
) -> dict[str, float]:
    if request.standardization == "equal":
        return {stratum: 1 / len(strata) for stratum in strata}
    if request.standardization == "custom":
        if set(request.reference_weights) != set(strata):
            raise ValueError("custom reference_weights must name every stratum exactly once")
        weights = {
            stratum: _finite(request.reference_weights[stratum], f"weight:{stratum}")
            for stratum in strata
        }
        if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("custom reference_weights must be non-negative with a positive sum")
        total = sum(weights.values())
        return {stratum: value / total for stratum, value in weights.items()}
    totals = {
        stratum: sum(cell.n for (group, key), cell in cells.items() if key == stratum)
        for stratum in strata
    }
    total = sum(totals.values())
    return {stratum: totals[stratum] / total for stratum in strata}


def _cmh_test(
    request: StatisticalRequest,
    first: str,
    second: str,
    strata: list[str],
    cells: dict[tuple[str, str], StratifiedCell],
) -> tuple[HypothesisTest, float, list[str]]:
    if request.alternative != "two-sided":
        raise ValueError("stratified_proportion currently supports two-sided CMH tests only")
    observed_minus_expected = 0.0
    variance = 0.0
    odds_numerator = 0.0
    odds_denominator = 0.0
    zero_cell = False
    for stratum in strata:
        first_cell, second_cell = cells[(first, stratum)], cells[(second, stratum)]
        a = float(first_cell.successes)
        b = float(first_cell.n - first_cell.successes)
        c = float(second_cell.successes)
        d = float(second_cell.n - second_cell.successes)
        n = a + b + c + d
        row_first = a + b
        row_second = c + d
        success_total = a + c
        failure_total = b + d
        expected_a = row_first * success_total / n
        if n <= 1:
            raise ValueError(f"stratum {stratum!r} has insufficient observations for CMH")
        stratum_variance = (
            row_first * row_second * success_total * failure_total
            / (n * n * (n - 1))
        )
        observed_minus_expected += a - expected_a
        variance += stratum_variance

        if min(a, b, c, d) == 0:
            zero_cell = True
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
            n = a + b + c + d
        odds_numerator += a * d / n
        odds_denominator += b * c / n
    if variance <= 0:
        raise ValueError("CMH test is undefined because stratified variance is zero")
    statistic = observed_minus_expected ** 2 / variance
    p_value = float(stats.chi2.sf(statistic, 1))
    if odds_denominator == 0:
        raise ValueError("Mantel-Haenszel common odds ratio denominator is zero")
    common_odds_ratio = odds_numerator / odds_denominator
    warnings = []
    if zero_cell:
        warnings.append(
            "Mantel-Haenszel common odds ratio uses a 0.5 correction in zero-cell strata"
        )
    test = HypothesisTest(
        name="Cochran-Mantel-Haenszel test (df=1)",
        statistic=_round(statistic),
        p_value=_round(p_value),
        alternative="two-sided",
        null_value=1.0,
        alpha=request.alpha,
        significant=p_value < request.alpha,
    )
    return test, common_odds_ratio, warnings


def _stratified_proportion(
    task_id: str, request: StatisticalRequest
) -> StatisticalEvidence:
    if len(request.cells) < 4:
        raise ValueError("stratified_proportion requires at least a complete 2x2 grid")
    groups = list(dict.fromkeys(cell.group for cell in request.cells))
    strata = list(dict.fromkeys(cell.stratum for cell in request.cells))
    if len(groups) < 2 or len(strata) < 2:
        raise ValueError("stratified_proportion requires at least two groups and strata")
    cells: dict[tuple[str, str], StratifiedCell] = {}
    for cell in request.cells:
        if cell.successes > cell.n:
            raise ValueError(
                f"{cell.group}/{cell.stratum}: successes cannot exceed n"
            )
        key = (cell.group, cell.stratum)
        if key in cells:
            raise ValueError(f"duplicate stratified cell for {cell.group}/{cell.stratum}")
        cells[key] = cell
    missing = [
        f"{group}/{stratum}"
        for group in groups
        for stratum in strata
        if (group, stratum) not in cells
    ]
    if missing:
        raise ValueError(
            "stratified_proportion requires common support; missing cells: "
            + ", ".join(missing)
        )

    weights = _reference_weights(request, strata, cells)
    critical = float(stats.norm.ppf(1 - (1 - request.confidence) / 2))
    crude_rates: dict[str, float] = {}
    adjusted_rates: dict[str, float] = {}
    stratum_rates: dict[str, dict[str, float]] = {}
    intervals: list[ConfidenceInterval] = []
    effects: list[EffectSize] = []
    warnings: list[str] = []
    group_totals: dict[str, int] = {}

    for group in groups:
        group_cells = [cells[(group, stratum)] for stratum in strata]
        total_successes = sum(cell.successes for cell in group_cells)
        total_n = sum(cell.n for cell in group_cells)
        group_totals[group] = total_n
        crude = total_successes / total_n
        crude_rates[group] = crude
        intervals.append(
            _wilson(total_successes, total_n, request.confidence, f"crude_rate:{group}")
        )

        rates = {
            stratum: cells[(group, stratum)].successes / cells[(group, stratum)].n
            for stratum in strata
        }
        stratum_rates[group] = rates
        adjusted = sum(weights[stratum] * rates[stratum] for stratum in strata)
        variance = sum(
            weights[stratum] ** 2
            * rates[stratum]
            * (1 - rates[stratum])
            / cells[(group, stratum)].n
            for stratum in strata
        )
        standard_error = math.sqrt(variance)
        adjusted_rates[group] = adjusted
        intervals.append(ConfidenceInterval(
            parameter=f"adjusted_rate:{group}",
            estimate=_round(adjusted),
            lower=_round(max(0.0, adjusted - critical * standard_error)),
            upper=_round(min(1.0, adjusted + critical * standard_error)),
            level=request.confidence,
            method=f"direct standardization ({request.standardization} weights)",
        ))
        effects.append(EffectSize(
            name=f"standardization change:{group}",
            value=_round(adjusted - crude),
            interpretation="adjusted rate minus crude rate",
        ))
        for stratum in strata:
            cell = cells[(group, stratum)]
            observed_share = cell.n / total_n
            if cell.n < 20:
                warnings.append(
                    f"{group}/{stratum} has n={cell.n}; its stratum rate is unstable"
                )
            if observed_share < 0.01 and weights[stratum] >= 0.05:
                warnings.append(
                    f"limited overlap: {group}/{stratum} is {observed_share:.1%} of "
                    f"that group but receives {weights[stratum]:.1%} reference weight"
                )

    reverse = request.direction == "higher"
    crude_ranking = sorted(groups, key=crude_rates.get, reverse=reverse)
    adjusted_ranking = sorted(groups, key=adjusted_rates.get, reverse=reverse)
    ranking_reversal = crude_ranking != adjusted_ranking
    winner_reversal = crude_ranking[0] != adjusted_ranking[0]
    simpson_pairs: list[list[str]] = []
    for first, second in combinations(groups, 2):
        crude_difference = crude_rates[first] - crude_rates[second]
        within_differences = [
            stratum_rates[first][stratum] - stratum_rates[second][stratum]
            for stratum in strata
        ]
        within_signs = {
            1 if difference > 0 else -1
            for difference in within_differences
            if abs(difference) > 1e-15
        }
        crude_sign = 1 if crude_difference > 0 else -1 if crude_difference < 0 else 0
        if (
            len(within_signs) == 1
            and crude_sign != 0
            and crude_sign == -next(iter(within_signs))
        ):
            simpson_pairs.append([first, second])
    simpson_reversal = bool(simpson_pairs)
    test = None

    if len(groups) == 2:
        first, second = groups
        crude_difference = crude_rates[first] - crude_rates[second]
        adjusted_difference = adjusted_rates[first] - adjusted_rates[second]
        first_variance = sum(
            weights[stratum] ** 2
            * stratum_rates[first][stratum]
            * (1 - stratum_rates[first][stratum])
            / cells[(first, stratum)].n
            for stratum in strata
        )
        second_variance = sum(
            weights[stratum] ** 2
            * stratum_rates[second][stratum]
            * (1 - stratum_rates[second][stratum])
            / cells[(second, stratum)].n
            for stratum in strata
        )
        difference_se = math.sqrt(first_variance + second_variance)
        intervals.append(ConfidenceInterval(
            parameter=f"adjusted_rate_difference:{first}-{second}",
            estimate=_round(adjusted_difference),
            lower=_round(adjusted_difference - critical * difference_se),
            upper=_round(adjusted_difference + critical * difference_se),
            level=request.confidence,
            method=f"direct standardization ({request.standardization} weights)",
        ))
        test, common_odds_ratio, cmh_warnings = _cmh_test(
            request, first, second, strata, cells
        )
        warnings.extend(cmh_warnings)
        effects.extend([
            EffectSize(
                name=f"crude risk difference:{first}-{second}",
                value=_round(crude_difference),
                interpretation="unadjusted probability difference",
            ),
            EffectSize(
                name=f"adjusted risk difference:{first}-{second}",
                value=_round(adjusted_difference),
                interpretation="direct-standardized probability difference",
            ),
            EffectSize(
                name="Mantel-Haenszel common odds ratio",
                value=_round(common_odds_ratio),
                interpretation=f"{first} vs {second}, adjusted for strata",
            ),
        ])

    if simpson_reversal:
        pairs = ", ".join(f"{first} vs {second}" for first, second in simpson_pairs)
        warnings.append(
            "Simpson's paradox detected: the crude direction reverses the "
            f"consistent within-stratum direction for {pairs}"
        )

    if winner_reversal and not simpson_reversal:
        warnings.append(
            "the crude and standardized winners differ; group composition changes "
            "the conclusion"
        )
    elif ranking_reversal and not simpson_reversal:
        warnings.append("group ranking changes after direct standardization")

    rates_text = ", ".join(
        f"{group}={adjusted_rates[group]:.4g}" for group in adjusted_ranking
    )
    summary = (
        f"{request.standardization}-standardized rates: {rates_text}; "
        f"crude winner={crude_ranking[0]}, adjusted winner={adjusted_ranking[0]}"
    )
    if winner_reversal:
        summary += "; winner reversed after adjustment"
    return StatisticalEvidence(
        task_id=task_id,
        request_id=request.id,
        kind=request.kind,
        status="computed",
        sample_size=sum(group_totals.values()),
        intervals=intervals,
        effects=effects,
        test=test,
        summary=summary,
        warnings=list(dict.fromkeys(warnings)),
        diagnostics={
            "standardization": request.standardization,
            "direction": request.direction,
            "reference_weights": {key: _round(value) for key, value in weights.items()},
            "crude_rates": {key: _round(value) for key, value in crude_rates.items()},
            "adjusted_rates": {key: _round(value) for key, value in adjusted_rates.items()},
            "stratum_rates": {
                group: {key: _round(value) for key, value in rates.items()}
                for group, rates in stratum_rates.items()
            },
            "crude_ranking": crude_ranking,
            "adjusted_ranking": adjusted_ranking,
            "ranking_reversal": ranking_reversal,
            "winner_reversal": winner_reversal,
            "simpson_reversal": simpson_reversal,
            "simpson_pairs": simpson_pairs,
        },
    )


def _chi_square(task_id: str, request: StatisticalRequest) -> StatisticalEvidence:
    counts = np.asarray(request.counts, dtype=float)
    if counts.ndim != 2 or min(counts.shape, default=0) < 2:
        raise ValueError("chi_square requires at least a 2x2 counts matrix")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("chi_square counts must be finite and non-negative")
    if len(request.rows) != counts.shape[0] or len(request.columns) != counts.shape[1]:
        raise ValueError("chi_square row/column labels must match the counts matrix")
    if (counts.sum(axis=0) == 0).any() or (counts.sum(axis=1) == 0).any():
        raise ValueError("chi_square rows and columns must have positive totals")
    correction = counts.shape == (2, 2)
    statistic, p_value, degrees_freedom, expected = stats.chi2_contingency(
        counts, correction=correction
    )
    n = int(counts.sum())
    denominator = min(counts.shape[0] - 1, counts.shape[1] - 1)
    cramers_v = math.sqrt(float(statistic) / (n * denominator))
    warnings = []
    low_expected = int((expected < 5).sum())
    if low_expected:
        warnings.append(
            f"{low_expected}/{expected.size} expected cells are below 5; "
            "chi-square approximation may be unreliable"
        )
    method = "Pearson chi-square with Yates correction" if correction else "Pearson chi-square"
    test = _hypothesis(
        name=f"{method} (df={degrees_freedom})",
        statistic=float(statistic), p_value=float(p_value), request=request,
        null_value=None,
    )
    return StatisticalEvidence(
        task_id=task_id, request_id=request.id, kind=request.kind, status="computed",
        sample_size=n,
        effects=[EffectSize(name="Cramer's V", value=_round(cramers_v),
                            interpretation=_interpret_cramers_v(cramers_v))],
        test=test,
        summary=f"chi-square({degrees_freedom})={statistic:.4g}; p={p_value:.4g}; "
                f"Cramer's V={cramers_v:.4g}",
        warnings=warnings,
    )


def evaluate_request(task_id: str, raw: dict[str, Any], index: int = 0) -> StatisticalEvidence:
    """Validate and compute one request; return a visible invalid result on failure."""
    request_id = str(raw.get("id") or f"test_{index + 1}") if isinstance(raw, dict) else f"test_{index + 1}"
    kind = str(raw.get("kind") or "unknown") if isinstance(raw, dict) else "unknown"
    try:
        request = StatisticalRequest.model_validate(raw)
        request.id = request_id
        if request.kind not in SUPPORTED_KINDS:
            raise ValueError(
                f"unsupported kind {request.kind!r}; expected one of {sorted(SUPPORTED_KINDS)}"
            )
        dispatch = {
            "one_proportion": _one_proportion,
            "two_proportion": _two_proportion,
            "one_mean": _one_mean,
            "two_mean": _two_mean,
            "chi_square": _chi_square,
            "stratified_proportion": _stratified_proportion,
        }
        return dispatch[request.kind](task_id, request)
    except (ValidationError, ValueError, TypeError, ArithmeticError) as exc:
        return StatisticalEvidence(
            task_id=task_id, request_id=request_id, kind=kind, status="invalid",
            summary="deterministic statistical request could not be computed",
            warnings=[str(exc)],
        )


def evaluate_artifacts(artifacts: dict[str, Artifact]) -> dict[str, StatisticalEvidence]:
    """Compute every optional ``statistical_tests`` request in settled artifacts."""
    evidence: dict[str, StatisticalEvidence] = {}
    for task_id, artifact in artifacts.items():
        result = artifact.outcome.result
        if not isinstance(result, dict) or "statistical_tests" not in result:
            continue
        requests = result.get("statistical_tests")
        if isinstance(requests, dict):
            requests = [requests]
        if not isinstance(requests, list):
            item = StatisticalEvidence(
                task_id=task_id, request_id="test_1", kind="unknown", status="invalid",
                summary="deterministic statistical request could not be computed",
                warnings=["statistical_tests must be a list or object"],
            )
            evidence[f"{task_id}:{item.request_id}"] = item
            continue
        for index, raw in enumerate(requests):
            item = evaluate_request(task_id, raw, index)
            key = f"{task_id}:{item.request_id}"
            if key in evidence:
                item.warnings.append("duplicate request id; later result replaced an earlier one")
            evidence[key] = item
    return evidence
