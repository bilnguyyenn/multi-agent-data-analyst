"""Ollama backend — talks to a local Ollama server over its native HTTP API.

No SDK dependency; it's a single POST. Token counts come back in the response
(prompt_eval_count / eval_count) so cost accounting works identically to the
hosted providers (cost is just $0/token here).
"""
from __future__ import annotations

import os

import httpx

from analyst.config import RoutingConfig
from analyst.llm.client import LLMResponse


class OllamaClient:
    def __init__(self, routing: RoutingConfig, base_url: str | None = None):
        self.routing = routing
        self.base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.timeout = routing.limits.get("request_timeout_s", 180)
        self.max_tokens = routing.limits.get("max_output_tokens", 2048)

    def complete(self, *, role: str, system: str, user: str) -> LLMResponse:
        model = self.routing.model_for(role, "local")
        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                # num_ctx caps the KV cache — on 8GB machines an unbounded context
                # window can cost more RAM than the model weights themselves.
                "options": {
                    "num_predict": self.max_tokens,
                    "temperature": 0.2,
                    "num_ctx": 4096,
                },
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(
            text=data["message"]["content"],
            model=model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )
