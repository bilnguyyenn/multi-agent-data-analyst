"""Held-out statistical benchmark with component-level, runtime-derived scoring.

Unlike the original benchmark's mostly binary entity/number checks, each item here
has a small rubric covering method, estimate, uncertainty/effect, and interpretation.
Partial credit is explicit and auditable. Ground truth is recomputed from the held-
out CSV bytes on every run so it cannot drift from the evaluated dataset.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from scipy import stats

HELDOUT_SUITE = "heldout-statistical-v1"
EXPECTED_HELDOUT_SIZE = 20
EXPECTED_HELDOUT_IDS = (
    "he_b_conversion_ci",
    "he_a_session_ci",
    "he_alpha_sla_ci",
    "he_b_order_value_ci",
    "he_island_small_n",
    "ht_conversion_effect",
    "ht_session_effect",
    "ht_device_conversion",
    "ht_b_target",
    "ht_resolution_effect",
    "hc_team_simpson",
    "hc_rollout_simpson",
    "hc_experiment_sound_control",
    "hc_team_equal_standard",
    "hc_rollout_causality",
    "hd_event_fanout",
    "hd_survey_nonresponse",
    "hd_channel_multiplicity",
    "hd_age_missingness",
    "hd_practical_significance",
)

Category = Literal["estimation", "testing_effects", "confounding", "design_quality"]


@dataclass(frozen=True)
class RubricCheck:
    id: str
    description: str
    kind: Literal["number", "text"]
    expected: float | None = None
    tolerance: float = 0.0
    allow_percent: bool = False
    any_of: tuple[str, ...] = ()
    forbidden_any: tuple[str, ...] = ()
    weight: float = 1.0


@dataclass(frozen=True)
class HeldoutItem:
    id: str
    category: Category
    method: str
    question: str
    checks: tuple[RubricCheck, ...] = field(default_factory=tuple)


def number_check(
    check_id: str,
    description: str,
    expected: float,
    tolerance: float,
    *,
    allow_percent: bool = False,
    weight: float = 1.0,
) -> RubricCheck:
    return RubricCheck(
        id=check_id,
        description=description,
        kind="number",
        expected=float(expected),
        tolerance=float(tolerance),
        allow_percent=allow_percent,
        weight=weight,
    )


def text_check(
    check_id: str,
    description: str,
    any_of: list[str] | tuple[str, ...],
    *,
    forbidden_any: list[str] | tuple[str, ...] = (),
    weight: float = 1.0,
) -> RubricCheck:
    return RubricCheck(
        id=check_id,
        description=description,
        kind="text",
        any_of=tuple(value.casefold() for value in any_of),
        forbidden_any=tuple(value.casefold() for value in forbidden_any),
        weight=weight,
    )


def _numbers(obj: Any):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        value = float(obj)
        if math.isfinite(value):
            yield value
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _numbers(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _numbers(value)


def _text_numbers(text: str) -> list[float]:
    values = []
    pattern = r"(?<![A-Za-z])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    for match in re.findall(pattern, text.replace(",", "")):
        try:
            values.append(float(match))
        except ValueError:
            continue
    return values


def _evidence_numbers(check: RubricCheck, evidence: list[dict[str, Any]]) -> list[float]:
    """Read only semantically relevant typed fields, avoiding accidental matches.

    For example, a tiny expected p-value must not pass merely because an unrelated
    test carries ``null_value=0`` elsewhere in the evidence packet.
    """
    values: list[float] = []
    for item in evidence:
        intervals = item.get("intervals", [])
        effects = item.get("effects", [])
        test = item.get("test") or {}
        if check.id in {"p_value", "global_p"}:
            values.extend(_numbers(test.get("p_value")))
        elif check.id == "lower":
            values.extend(_numbers([interval.get("lower") for interval in intervals]))
        elif check.id == "upper":
            values.extend(_numbers([interval.get("upper") for interval in intervals]))
        elif check.id == "n":
            values.extend(_numbers(item.get("sample_size")))
        elif any(
            marker in check.id
            for marker in ("difference", "cohen", "hedges", "cramers", "observed_diff")
        ):
            values.extend(_numbers([effect.get("value") for effect in effects]))
            values.extend(_numbers([interval.get("estimate") for interval in intervals]))
        else:
            values.extend(_numbers([interval.get("estimate") for interval in intervals]))
            values.extend(_numbers(item.get("diagnostics", {})))
    return values


def score_heldout_item(
    item: HeldoutItem,
    answer: str,
    statistical_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score method components from final prose plus typed deterministic evidence."""
    text = (answer or "").casefold()
    answer_numbers = _text_numbers(text)
    results = []
    earned = 0.0
    possible = sum(check.weight for check in item.checks)
    for check in item.checks:
        if check.kind == "number":
            numeric_values = answer_numbers + _evidence_numbers(
                check, statistical_evidence or []
            )
            targets = [float(check.expected)]
            if check.allow_percent:
                targets.append(float(check.expected) * 100)
            passed = any(
                abs(value - target) <= check.tolerance * (100 if target != check.expected else 1)
                for value in numeric_values
                for target in targets
            )
        else:
            has_required = any(marker in text for marker in check.any_of)
            has_forbidden = any(marker in text for marker in check.forbidden_any)
            passed = has_required and not has_forbidden
        if passed:
            earned += check.weight
        results.append(
            {
                "id": check.id,
                "description": check.description,
                "passed": passed,
                "weight": check.weight,
                "expected": check.expected,
            }
        )
    return {
        "score": round(earned / possible, 3) if possible else 0.0,
        "checks_passed": sum(1 for result in results if result["passed"]),
        "checks_total": len(results),
        "check_results": results,
    }


def _wilson(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def _mean_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float, float]:
    clean = values.dropna().astype(float)
    mean = float(clean.mean())
    sem = float(clean.std(ddof=1) / math.sqrt(len(clean)))
    critical = float(stats.t(df=len(clean) - 1).ppf(1 - (1 - confidence) / 2))
    return mean, mean - critical * sem, mean + critical * sem


def _two_proportion(first: pd.Series, second: pd.Series) -> tuple[float, float, float]:
    n1, n2 = len(first), len(second)
    p1, p2 = float(first.mean()), float(second.mean())
    pooled = float((first.sum() + second.sum()) / (n1 + n2))
    standard_error = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / standard_error
    p_value = float(2 * stats.norm.sf(abs(z)))
    cohen_h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
    return p1 - p2, p_value, cohen_h


def _welch(first: pd.Series, second: pd.Series) -> tuple[float, float, float]:
    first = first.dropna().astype(float)
    second = second.dropna().astype(float)
    difference = float(first.mean() - second.mean())
    _, p_value = stats.ttest_ind(first, second, equal_var=False)
    pooled_df = len(first) + len(second) - 2
    pooled_std = math.sqrt(
        ((len(first) - 1) * first.var(ddof=1) + (len(second) - 1) * second.var(ddof=1))
        / pooled_df
    )
    cohen_d = difference / pooled_std
    hedges_g = (1 - 3 / (4 * pooled_df - 1)) * cohen_d
    return difference, float(p_value), hedges_g


def _chi_square(frame: pd.DataFrame) -> tuple[float, float]:
    statistic, p_value, _, _ = stats.chi2_contingency(frame.to_numpy())
    n = int(frame.to_numpy().sum())
    denominator = min(frame.shape[0] - 1, frame.shape[1] - 1)
    return float(p_value), math.sqrt(float(statistic) / (n * denominator))


def _standardized_rates(
    frame: pd.DataFrame,
    group: str,
    stratum: str,
    outcome: str,
    *,
    equal: bool = False,
) -> dict[str, float]:
    groups = sorted(frame[group].unique())
    strata = sorted(frame[stratum].unique())
    if equal:
        weights = {level: 1 / len(strata) for level in strata}
    else:
        counts = frame.groupby(stratum).size()
        weights = {level: float(counts[level] / len(frame)) for level in strata}
    rates = frame.groupby([group, stratum])[outcome].mean()
    return {
        level: sum(weights[cell] * float(rates.loc[(level, cell)]) for cell in strata)
        for level in groups
    }


def _decision_check(check_id: str, p_value: float) -> RubricCheck:
    if p_value < 0.05:
        return text_check(
            check_id,
            "correct significance decision",
            ["statistically significant", "significant at", "reject the null", "p<", "p <"],
            forbidden_any=["not statistically significant", "not significant"],
        )
    return text_check(
        check_id,
        "correct significance decision",
        ["not statistically significant", "not significant", "fail to reject"],
    )


def build_heldout_benchmark(data_dir: str | Path) -> list[HeldoutItem]:
    directory = Path(data_dir)
    users = pd.read_csv(directory / "users.csv")
    outcomes = pd.read_csv(directory / "experiment_outcomes.csv")
    events = pd.read_csv(directory / "page_events.csv")
    tickets = pd.read_csv(directory / "support_tickets.csv")
    rollout = pd.read_csv(directory / "feature_rollout.csv")
    surveys = pd.read_csv(directory / "survey_responses.csv")

    experiment = outcomes.merge(users, on="user_id", validate="one_to_one")
    a = experiment[experiment.variant == "A"]
    b = experiment[experiment.variant == "B"]
    b_rate = float(b.converted.mean())
    b_lower, b_upper = _wilson(int(b.converted.sum()), len(b))
    a_session = _mean_ci(a.session_minutes)

    alpha = tickets[tickets.team == "Alpha"]
    beta = tickets[tickets.team == "Beta"]
    alpha_rate = float(alpha.met_sla.mean())
    alpha_lower, alpha_upper = _wilson(int(alpha.met_sla.sum()), len(alpha))
    b_orders = b.loc[b.converted == 1, "order_value"]
    b_order_ci = _mean_ci(b_orders)
    island_b = b[b.region == "island"]
    island_rate = float(island_b.converted.mean())

    conversion_effect = _two_proportion(b.converted, a.converted)
    session_effect = _welch(b.session_minutes, a.session_minutes)
    device_table = pd.crosstab(experiment.device, experiment.converted)
    device_effect = _chi_square(device_table)
    b_target_p = float(
        stats.binomtest(int(b.converted.sum()), len(b), p=0.13, alternative="two-sided").pvalue
    )
    resolution_effect = _welch(alpha.resolution_hours, beta.resolution_hours)

    ticket_adjusted = _standardized_rates(tickets, "team", "severity", "met_sla")
    ticket_equal = _standardized_rates(
        tickets, "team", "severity", "met_sla", equal=True
    )
    rollout_full = rollout.merge(users[["user_id", "plan"]], on="user_id", validate="one_to_one")
    rollout_adjusted = _standardized_rates(
        rollout_full, "experience", "plan", "adopted"
    )
    experiment_adjusted = _standardized_rates(
        experiment, "variant", "device", "converted"
    )

    joined = outcomes.merge(events, on="user_id", validate="one_to_many")
    user_conversion = float(outcomes.converted.mean())
    event_weighted_conversion = float(joined.converted.mean())
    survey_full = surveys.merge(outcomes[["user_id", "variant"]], on="user_id")
    response_rates = survey_full.groupby("variant").responded.mean().to_dict()
    observed_satisfaction = survey_full.groupby("variant").satisfaction.mean().to_dict()
    channel_table = pd.crosstab(experiment.acquisition_channel, experiment.converted)
    channel_p, _ = _chi_square(channel_table)
    channel_rates = experiment.groupby("acquisition_channel").converted.mean()
    channel_winner = str(channel_rates.idxmax())
    age_missing_rate = float(users.age.isna().mean())

    items = [
        # ---- estimation: intervals and small-sample uncertainty (5) ---------
        HeldoutItem(
            "he_b_conversion_ci", "estimation", "Wilson proportion interval",
            "Estimate variant B's conversion rate and a 95% confidence interval. Use the user as the unit of analysis.",
            (
                number_check("rate", "conversion-rate point estimate", b_rate, 0.003, allow_percent=True),
                number_check("lower", "Wilson lower bound", b_lower, 0.004, allow_percent=True),
                number_check("upper", "Wilson upper bound", b_upper, 0.004, allow_percent=True),
                text_check("method", "interval method or uncertainty named", ["wilson", "confidence interval", "95% ci"]),
            ),
        ),
        HeldoutItem(
            "he_a_session_ci", "estimation", "one-mean t interval",
            "Estimate the mean session duration for variant A with a 95% confidence interval.",
            (
                number_check("mean", "mean session duration", a_session[0], 0.2),
                number_check("lower", "t-interval lower bound", a_session[1], 0.25),
                number_check("upper", "t-interval upper bound", a_session[2], 0.25),
                text_check("method", "mean interval is identified", ["t interval", "t-interval", "confidence interval", "95% ci"]),
            ),
        ),
        HeldoutItem(
            "he_alpha_sla_ci", "estimation", "Wilson proportion interval",
            "What proportion of Alpha-team tickets met SLA? Report a 95% confidence interval without adjusting for severity yet.",
            (
                number_check("rate", "crude Alpha SLA rate", alpha_rate, 0.004, allow_percent=True),
                number_check("lower", "Wilson lower bound", alpha_lower, 0.005, allow_percent=True),
                number_check("upper", "Wilson upper bound", alpha_upper, 0.005, allow_percent=True),
                text_check("crude", "unadjusted estimand is explicit", ["unadjusted", "crude", "before adjust"]),
            ),
        ),
        HeldoutItem(
            "he_b_order_value_ci", "estimation", "conditional mean t interval",
            "Among converted variant-B users, estimate average order value with a 95% confidence interval. State the conditioning population.",
            (
                number_check("mean", "conditional order-value mean", b_order_ci[0], 1.0),
                number_check("lower", "mean lower bound", b_order_ci[1], 1.25),
                number_check("upper", "mean upper bound", b_order_ci[2], 1.25),
                text_check("population", "conditioning on converted users", ["among converted", "converted users", "conditional on conversion"]),
            ),
        ),
        HeldoutItem(
            "he_island_small_n", "estimation", "small-subgroup proportion",
            "Does variant B appear to convert well in the island region? Estimate its rate, but explain how much confidence the small subgroup supports.",
            (
                number_check("rate", "island/B point estimate", island_rate, 0.02, allow_percent=True),
                number_check("n", "island/B sample size", len(island_b), 0),
                text_check("small_n", "small-sample limitation", ["small sample", "small n", "few users", "limited sample", "wide interval"]),
                text_check("uncertainty", "uncertainty is not overstated", ["uncertain", "confidence interval", "wide", "caution", "insufficient"]),
            ),
        ),
        # ---- tests and effect sizes (5) -------------------------------------
        HeldoutItem(
            "ht_conversion_effect", "testing_effects", "two-proportion comparison",
            "Compare conversion between B and A. Report the absolute effect, a hypothesis test, and a standardized effect size; distinguish statistical from practical importance.",
            (
                number_check("difference", "B-minus-A risk difference", conversion_effect[0], 0.004, allow_percent=True),
                number_check("p_value", "two-proportion p-value", conversion_effect[1], max(0.002, conversion_effect[1] * 0.25)),
                number_check("cohen_h", "Cohen's h", conversion_effect[2], 0.025),
                _decision_check("decision", conversion_effect[1]),
                text_check("practical", "practical magnitude discussed", ["small effect", "small", "modest", "practical", "percentage point"]),
            ),
        ),
        HeldoutItem(
            "ht_session_effect", "testing_effects", "Welch mean comparison",
            "Compare mean session duration for B versus A using Welch's test and an effect size. Is a statistically detectable difference large in practice?",
            (
                number_check("difference", "B-minus-A mean difference", session_effect[0], 0.3),
                number_check("p_value", "Welch p-value", session_effect[1], max(0.002, session_effect[1] * 0.3)),
                number_check("hedges_g", "Hedges' g", session_effect[2], 0.03),
                text_check("method", "Welch comparison named", ["welch"]),
                text_check("magnitude", "small standardized magnitude recognized", ["small", "modest", "limited practical"]),
            ),
        ),
        HeldoutItem(
            "ht_device_conversion", "testing_effects", "chi-square association",
            "Test whether device type and conversion are associated. Include a chi-square p-value and Cramer's V, and avoid causal language.",
            (
                number_check("p_value", "chi-square p-value", device_effect[0], max(0.002, device_effect[0] * 0.3)),
                number_check("cramers_v", "Cramer's V", device_effect[1], 0.025),
                text_check("method", "chi-square method named", ["chi-square", "chi squared", "χ²"]),
                text_check("association", "association rather than causation", ["associated", "association", "not causal", "does not imply caus"]),
            ),
        ),
        HeldoutItem(
            "ht_b_target", "testing_effects", "exact one-proportion test",
            "Test whether variant B's conversion rate differs from a 13% target. Report the test result and uncertainty.",
            (
                number_check("rate", "observed B rate", b_rate, 0.003, allow_percent=True),
                number_check("p_value", "exact-binomial p-value", b_target_p, max(0.003, b_target_p * 0.25)),
                _decision_check("decision", b_target_p),
                text_check("method", "binomial/proportion test named", ["binomial", "proportion test", "exact test"]),
            ),
        ),
        HeldoutItem(
            "ht_resolution_effect", "testing_effects", "unadjusted Welch comparison",
            "As a deliberately unadjusted descriptive comparison, quantify Alpha minus Beta mean resolution time with Welch's test and Hedges' g. Flag the severity-mix limitation.",
            (
                number_check("difference", "Alpha-minus-Beta mean difference", resolution_effect[0], 0.5),
                number_check("p_value", "Welch p-value", resolution_effect[1], max(0.002, resolution_effect[1] * 0.3)),
                number_check("hedges_g", "Hedges' g", resolution_effect[2], 0.04),
                text_check("confounding", "severity composition limitation", ["severity mix", "severity", "confound", "composition"]),
            ),
        ),
        # ---- stratification and causal restraint (5) ------------------------
        HeldoutItem(
            "hc_team_simpson", "confounding", "pooled direct standardization",
            "Which support team has the better SLA record after adjusting for ticket severity? Compare the crude and standardized conclusions and check for Simpson's paradox.",
            (
                text_check("crude", "Beta identified as crude winner", ["crude winner=beta", "beta is the crude", "unadjusted beta", "beta appears better"]),
                text_check("adjusted", "Alpha identified as adjusted winner", ["adjusted winner=alpha", "alpha is better after", "adjusted alpha", "standardized alpha"]),
                number_check("alpha_adjusted", "standardized Alpha rate", ticket_adjusted["Alpha"], 0.008, allow_percent=True),
                number_check("beta_adjusted", "standardized Beta rate", ticket_adjusted["Beta"], 0.008, allow_percent=True),
                text_check("simpson", "Simpson/composition reversal named", ["simpson", "reversal", "severity mix", "composition"]),
            ),
        ),
        HeldoutItem(
            "hc_rollout_simpson", "confounding", "pooled direct standardization",
            "Does the new feature experience have higher adoption than legacy after standardizing for plan? Show whether the crude winner changes.",
            (
                text_check("crude", "legacy identified as crude winner", ["crude winner=legacy", "legacy is the crude", "unadjusted legacy", "legacy appears higher"]),
                text_check("adjusted", "new identified as adjusted winner", ["adjusted winner=new", "new is better after", "adjusted new", "standardized new"]),
                number_check("new_adjusted", "standardized new rate", rollout_adjusted["new"], 0.01, allow_percent=True),
                number_check("legacy_adjusted", "standardized legacy rate", rollout_adjusted["legacy"], 0.01, allow_percent=True),
                text_check("plan", "plan composition/reversal explained", ["plan mix", "plan", "composition", "simpson", "confound"]),
            ),
        ),
        HeldoutItem(
            "hc_experiment_sound_control", "confounding", "stratified sound control",
            "Recompute A/B conversion standardized across device. Does adjustment reverse the experiment's winner, or is the B result robust?",
            (
                text_check("winner", "B remains the adjusted winner", ["adjusted winner=b", "b remains", "b is still", "standardized b"]),
                number_check("a_adjusted", "device-standardized A rate", experiment_adjusted["A"], 0.007, allow_percent=True),
                number_check("b_adjusted", "device-standardized B rate", experiment_adjusted["B"], 0.007, allow_percent=True),
                text_check("no_reversal", "sound control is not falsely called a reversal", ["no reversal", "does not reverse", "same winner", "robust"], forbidden_any=["simpson's paradox detected"]),
            ),
        ),
        HeldoutItem(
            "hc_team_equal_standard", "confounding", "equal-weight direct standardization",
            "Standardize each team's SLA rate to a population that is exactly 50% easy and 50% hard. Report both adjusted rates and the winner.",
            (
                number_check("alpha", "equal-weight Alpha rate", ticket_equal["Alpha"], 0.008, allow_percent=True),
                number_check("beta", "equal-weight Beta rate", ticket_equal["Beta"], 0.008, allow_percent=True),
                text_check("winner", "Alpha wins equal-weight comparison", ["alpha is better", "winner=alpha", "alpha has the higher", "alpha outperforms"]),
                text_check("weights", "50/50 target population explicit", ["50/50", "50% easy", "equal weight", "equal-weight"]),
            ),
        ),
        HeldoutItem(
            "hc_rollout_causality", "confounding", "observational causal restraint",
            "After plan adjustment the new experience is associated with higher adoption. Can we conclude that it caused the increase? Explain the design limitation.",
            (
                text_check("no_cause", "causal claim rejected", ["cannot conclude", "cannot infer caus", "not causal", "does not establish caus", "association, not caus"]),
                text_check("observational", "observational/nonrandomized design named", ["observational", "not randomized", "non-random", "selection"]),
                text_check("residual", "residual confounding acknowledged", ["residual confound", "unmeasured confound", "other confound", "selection bias"]),
                text_check("next_step", "causal validation proposed", ["randomized", "a/b test", "experiment", "propensity", "causal"]),
            ),
        ),
        # ---- design and data-quality traps (5) ------------------------------
        HeldoutItem(
            "hd_event_fanout", "design_quality", "analysis-grain/fanout check",
            "Using experiment outcomes and page events, what is the overall user conversion rate? Explain how to join without event-level fanout bias.",
            (
                number_check("correct_rate", "one-row-per-user conversion rate", user_conversion, 0.003, allow_percent=True),
                number_check("naive_rate", "biased event-weighted rate recognized", event_weighted_conversion, 0.005, allow_percent=True),
                text_check("grain", "user grain/deduplication specified", ["one row per user", "user-level", "deduplic", "aggregate events"]),
                text_check("fanout", "one-to-many inflation named", ["fanout", "one-to-many", "duplicate", "weighted by event", "multiply"]),
            ),
        ),
        HeldoutItem(
            "hd_survey_nonresponse", "design_quality", "missing-outcome bias",
            "Compare observed satisfaction for A and B, but assess whether survey nonresponse permits a population-level conclusion. Quantify response rates.",
            (
                number_check("a_response", "A response rate", response_rates["A"], 0.006, allow_percent=True),
                number_check("b_response", "B response rate", response_rates["B"], 0.006, allow_percent=True),
                number_check("observed_diff", "observed B-minus-A satisfaction", observed_satisfaction["B"] - observed_satisfaction["A"], 0.08),
                text_check("bias", "nonresponse/selection bias named", ["nonresponse", "response bias", "selection bias", "missing not at random", "mnar"]),
                text_check("limit", "population generalization limited", ["cannot generalize", "respondents only", "not representative", "population conclusion", "caution"]),
            ),
        ),
        HeldoutItem(
            "hd_channel_multiplicity", "design_quality", "multiple-comparison control",
            "Which acquisition channel has the highest observed conversion rate, and is there evidence it is truly best after accounting for five-channel comparison?",
            (
                text_check("observed", "observed winner named", [channel_winner]),
                number_check("global_p", "global chi-square p-value", channel_p, max(0.004, channel_p * 0.25)),
                text_check("multiplicity", "multiple comparisons recognized", ["multiple comparison", "multiplicity", "bonferroni", "holm", "false discovery"]),
                text_check("restraint", "winner not overclaimed", ["not enough evidence", "cannot conclude", "observed winner", "descriptive", "not statistically"]),
            ),
        ),
        HeldoutItem(
            "hd_age_missingness", "design_quality", "missing-data audit",
            "Audit the age field before using it in a model. Quantify missingness and recommend an analysis-safe next step without silently filling values.",
            (
                number_check("missing_rate", "age missing rate", age_missing_rate, 0.003, allow_percent=True),
                text_check("missing", "missingness explicitly identified", ["missing", "null"]),
                text_check("mechanism", "missingness mechanism investigated", ["missingness mechanism", "mcar", "mar", "mnar", "pattern", "systematic"]),
                text_check("safe", "no silent imputation", ["indicator", "sensitivity", "document", "imput", "complete-case"], forbidden_any=["fill with zero"]),
            ),
        ),
        HeldoutItem(
            "hd_practical_significance", "design_quality", "practical significance",
            "The A/B conversion result may be statistically significant. Is the effect large enough to matter? Use absolute and standardized effect sizes and state what additional business input is needed.",
            (
                number_check("difference", "absolute conversion uplift", conversion_effect[0], 0.004, allow_percent=True),
                number_check("cohen_h", "standardized effect", conversion_effect[2], 0.025),
                text_check("magnitude", "small/modest magnitude recognized", ["small", "modest", "percentage point"]),
                text_check("business", "business threshold/economics requested", ["business", "cost", "revenue", "threshold", "minimum detectable", "roi"]),
            ),
        ),
    ]
    if len(items) != EXPECTED_HELDOUT_SIZE:
        raise AssertionError(
            f"held-out suite must contain {EXPECTED_HELDOUT_SIZE} items, found {len(items)}"
        )
    return items
