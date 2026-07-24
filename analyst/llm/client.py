"""LLMClient protocol + factory.

One interface, three backends — this is the zero-budget strategy (ARCHITECTURE.md §6):

- local     -> Ollama on your machine ($0, primary dev path)
- replay    -> recorded cassettes ($0, deterministic; what CI runs)
- anthropic -> wired but dormant until credits exist

Every call is tagged with the agent ROLE; the factory resolves role -> model via
the routing config. Callers never name a model directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from analyst.config import RoutingConfig, current_provider, load_routing


class DailyQuotaExhausted(RuntimeError):
    """Raised when a 429 is the per-DAY quota (RPD), not the per-minute one.

    Per-minute 429s are transient — wait and retry. Per-day 429s are NOT: nothing
    clears until midnight Pacific, so retrying (or hammering the wall for every
    remaining question) just wastes time. The eval harness catches this to stop the
    run cleanly and save partial progress, so --resume can pick up where it left off.
    Lives here (the neutral interface module) so both the hosted backend that raises
    it and the harness that handles it can import it without a layering cycle."""


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    def complete(self, *, role: str, system: str, user: str) -> LLMResponse: ...


def make_client(
    provider: str | None = None, routing: RoutingConfig | None = None
) -> LLMClient:
    provider = provider or current_provider()
    routing = routing or load_routing()

    if provider == "local":
        from analyst.llm.ollama import OllamaClient

        client: LLMClient = OllamaClient(routing)
    elif provider == "hosted":
        from analyst.llm.openai_compat import HostedClient

        client = HostedClient(routing)
    elif provider == "anthropic":
        from analyst.llm.anthropic_backend import AnthropicClient

        client = AnthropicClient(routing)
    elif provider == "replay":
        from analyst.llm.replay import ReplayClient

        return ReplayClient.from_env()  # replay ignores routing: it serves recordings
    else:
        raise ValueError(f"Unknown ANALYST_LLM_PROVIDER {provider!r}")

    # Optional: record live calls into cassettes for later $0 replay.
    if os.environ.get("ANALYST_RECORD_CASSETTES"):
        from analyst.llm.replay import RecordingClient

        return RecordingClient(inner=client)
    return client
