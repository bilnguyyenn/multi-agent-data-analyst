"""Independent product-analytics dataset for the held-out statistical benchmark.

This schema and domain are intentionally separate from the manufacturing benchmark.
Ground truth is derived at evaluation time; the metadata written beside (not inside)
the CSV directory documents planted mechanisms without exposing them to sandboxes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_HELDOUT_SEED = 20260717


def generate_heldout(
    out_dir: str | Path,
    seed: int = DEFAULT_HELDOUT_SEED,
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ---- users and a randomized product experiment -------------------------
    n_users = 5000
    user_ids = np.array([f"U{index:05d}" for index in range(1, n_users + 1)])
    device = rng.choice(["mobile", "desktop"], n_users, p=[0.65, 0.35])
    plan = rng.choice(["basic", "pro"], n_users, p=[0.70, 0.30])
    region = rng.choice(
        ["north", "south", "west", "island"], n_users, p=[0.44, 0.35, 0.20, 0.01]
    )
    channel = rng.choice(
        ["organic", "search", "social", "partner", "email"],
        n_users,
        p=[0.30, 0.25, 0.20, 0.15, 0.10],
    )
    age = np.clip(np.rint(rng.normal(38, 12, n_users)), 18, 80).astype(float)
    age[rng.random(n_users) < 0.02] = np.nan
    users = pd.DataFrame(
        {
            "user_id": user_ids,
            "region": region,
            "device": device,
            "plan": plan,
            "acquisition_channel": channel,
            "age": age,
        }
    )

    variant = rng.choice(["A", "B"], n_users)
    base_conversion = (
        0.09
        + 0.055 * (device == "desktop")
        + 0.025 * (plan == "pro")
    )
    conversion_probability = np.clip(base_conversion + 0.028 * (variant == "B"), 0, 1)
    converted = rng.binomial(1, conversion_probability)
    session_minutes = np.clip(
        rng.normal(
            27 + 6 * (device == "desktop") + 1.25 * (variant == "B"),
            10,
            n_users,
        ),
        1,
        None,
    )
    order_value = np.where(
        converted == 1,
        rng.gamma(shape=3.0, scale=22.0, size=n_users),
        np.nan,
    )
    outcomes = pd.DataFrame(
        {
            "user_id": user_ids,
            "variant": variant,
            "converted": converted,
            "session_minutes": np.round(session_minutes, 2),
            "order_value": np.round(order_value, 2),
        }
    )

    # One-to-many event log. Converted users generate more events, so joining this
    # table before computing user conversion materially biases the denominator.
    event_rows = []
    event_number = 1
    for user_id, did_convert in zip(user_ids, converted):
        count = int(rng.poisson(1.5 + 3.0 * did_convert) + 1)
        for _ in range(count):
            event_rows.append(
                {
                    "event_id": f"E{event_number:07d}",
                    "user_id": user_id,
                    "event_type": rng.choice(
                        ["page_view", "search", "feature_use", "checkout"],
                        p=[0.45, 0.25, 0.22, 0.08],
                    ),
                }
            )
            event_number += 1
    page_events = pd.DataFrame(event_rows)

    # ---- support operations: planted Simpson reversal ----------------------
    n_tickets = 4000
    team = rng.choice(["Alpha", "Beta"], n_tickets)
    hard_probability = np.where(team == "Alpha", 0.80, 0.20)
    severity = np.where(rng.random(n_tickets) < hard_probability, "hard", "easy")
    sla_probability = np.select(
        [
            (team == "Alpha") & (severity == "easy"),
            (team == "Alpha") & (severity == "hard"),
            (team == "Beta") & (severity == "easy"),
        ],
        [0.93, 0.62, 0.85],
        default=0.50,
    )
    met_sla = rng.binomial(1, sla_probability)
    resolution_hours = rng.gamma(
        shape=2.2,
        scale=np.where(severity == "hard", 8.0, 2.5),
    ) * np.where(team == "Alpha", 0.88, 1.0)
    tickets = pd.DataFrame(
        {
            "ticket_id": [f"T{index:05d}" for index in range(1, n_tickets + 1)],
            "user_id": rng.choice(user_ids, n_tickets),
            "team": team,
            "severity": severity,
            "met_sla": met_sla,
            "resolution_hours": np.round(resolution_hours, 2),
        }
    )

    # ---- observational rollout: a second composition reversal --------------
    rollout_users = users.sample(n=3500, random_state=seed).reset_index(drop=True)
    new_probability = np.where(rollout_users.plan.eq("basic"), 0.78, 0.14)
    experience = np.where(rng.random(len(rollout_users)) < new_probability, "new", "legacy")
    adoption_probability = (
        np.where(rollout_users.plan.eq("basic"), 0.07, 0.22)
        + 0.055 * (experience == "new")
    )
    adopted = rng.binomial(1, adoption_probability)
    rollout = pd.DataFrame(
        {
            "user_id": rollout_users.user_id,
            "experience": experience,
            "adopted": adopted,
        }
    )

    # ---- survey invitations: outcome-dependent missingness -----------------
    latent_satisfaction = np.clip(
        np.rint(rng.normal(3.45 + 0.18 * (variant == "B"), 1.05, n_users)), 1, 5
    ).astype(int)
    response_probability = np.clip(
        0.25 + 0.09 * latent_satisfaction + 0.04 * (variant == "B"), 0, 0.95
    )
    responded = rng.binomial(1, response_probability)
    satisfaction = np.where(responded == 1, latent_satisfaction, np.nan)
    surveys = pd.DataFrame(
        {
            "user_id": user_ids,
            "invited": 1,
            "responded": responded,
            "satisfaction": satisfaction,
        }
    )

    frames = {
        "users": users,
        "experiment_outcomes": outcomes,
        "page_events": page_events,
        "support_tickets": tickets,
        "feature_rollout": rollout,
        "survey_responses": surveys,
    }
    paths: list[Path] = []
    for name, frame in frames.items():
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)

    metadata = {
        "suite": "heldout-statistical-v1",
        "seed": seed,
        "domain": "product analytics and support operations",
        "planted_mechanisms": [
            "small randomized conversion uplift",
            "small session-duration effect",
            "support-team severity-mix Simpson reversal",
            "feature-rollout plan-mix Simpson reversal",
            "one-to-many event-log fanout",
            "outcome-dependent survey nonresponse",
        ],
    }
    metadata_path = out.parent / f"{out.name}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    paths.append(metadata_path)
    return paths
