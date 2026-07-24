"""Anthropic backend — wired but dormant until an API key/credits exist.

Kept deliberately thin: same LLMResponse shape, token usage from the API response,
model resolved from the routing table (Fable 5 for reasoning tier, Haiku for
mechanical — see config/routing.yaml).
"""
from __future__ import annotations

from analyst.config import RoutingConfig
from analyst.llm.client import LLMResponse


class AnthropicClient:
    def __init__(self, routing: RoutingConfig):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "pip install 'analyst[anthropic]' to use the anthropic provider"
            ) from e
        self._sdk = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.routing = routing
        self.max_tokens = routing.limits.get("max_output_tokens", 2048)

    def complete(self, *, role: str, system: str, user: str) -> LLMResponse:
        model = self.routing.model_for(role, "anthropic")
        msg = self._sdk.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return LLMResponse(
            text="".join(b.text for b in msg.content if b.type == "text"),
            model=model,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
        )
