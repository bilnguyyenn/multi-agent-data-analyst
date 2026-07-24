"""Load routing config and environment selection.

Design point: which MODEL serves which AGENT ROLE is data (config/routing.yaml),
not code. Swapping providers is one env var; swapping the routing table is one file.
The eval harness later treats the routing table as an experimental variable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "routing.yaml"

Provider = str  # "local" | "replay" | "anthropic"


@dataclass
class RoutingConfig:
    roles: dict[str, str]                    # role -> tier
    tiers: dict[str, dict[str, str]]         # tier -> provider -> model
    limits: dict = field(default_factory=dict)

    def model_for(self, role: str, provider: Provider) -> str:
        tier = self.roles.get(role)
        if tier is None:
            raise KeyError(f"No routing entry for agent role {role!r}")
        try:
            return self.tiers[tier][provider]
        except KeyError:
            raise KeyError(
                f"Tier {tier!r} has no model for provider {provider!r} "
                f"(role {role!r}). Add it to routing.yaml."
            ) from None


def load_routing(path: str | Path | None = None) -> RoutingConfig:
    p = Path(path or os.environ.get("ANALYST_ROUTING_CONFIG", DEFAULT_CONFIG))
    raw = yaml.safe_load(p.read_text())
    return RoutingConfig(
        roles=raw["roles"], tiers=raw["tiers"], limits=raw.get("limits", {})
    )


def current_provider() -> Provider:
    return os.environ.get("ANALYST_LLM_PROVIDER", "local")
