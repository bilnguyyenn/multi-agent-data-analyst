"""Hosted OpenAI-compatible backend (Groq, Google AI Studio, any /chat/completions).

Why generic: every serious inference host speaks the OpenAI chat-completions dialect,
so one client covers Groq today and anything else later — provider choice is two env
vars, not code:

    ANALYST_HOSTED_BASE_URL   e.g. https://api.groq.com/openai/v1   (default)
    ANALYST_HOSTED_API_KEY    from the provider console

Free tiers are rate-limited per minute (Groq: 30 RPM / ~12K TPM on the 70B), so 429s
are EXPECTED operating conditions, not errors — we honor Retry-After and retry a few
times before giving up. Cassette recording (ANALYST_RECORD_CASSETTES=1) makes every
successful call reusable, which is how a 1K-requests/day budget stays comfortable.
"""
from __future__ import annotations

import os
import time

import httpx

from analyst.config import RoutingConfig
from analyst.llm.client import DailyQuotaExhausted, LLMResponse

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
MAX_429_RETRIES = 4


def _is_daily_quota(resp) -> bool:
    """Google tags the exhausted quota in the 429 body. Per-day violations name a
    'PerDay' quota id (e.g. GenerateRequestsPerDayPerProjectPerModel) or say the
    daily limit is used up. Match on that so we don't confuse it with per-minute."""
    body = (resp.text or "")
    return "PerDay" in body or "per day" in body.lower()


def _retry_wait(resp, attempt: int) -> float:
    """Prefer the provider's own guidance: retry-after header, or Google's
    retryDelay inside the 429 body ('retryDelay': '37s'). Fall back to backoff."""
    if resp.headers.get("retry-after"):
        return float(resp.headers["retry-after"])
    import re

    m = re.search(r'retryDelay[\'"]?\s*:\s*[\'"]?(\d+)', resp.text or "")
    if m:
        return float(m.group(1)) + 1
    return float(2 ** (attempt + 1))


class HostedClient:
    def __init__(self, routing: RoutingConfig):
        self.routing = routing
        self.base_url = os.environ.get(
            "ANALYST_HOSTED_BASE_URL", DEFAULT_BASE_URL
        ).rstrip("/")
        # Client-side pacing: gliding at the provider's RPM limit beats slamming
        # into 429 storms and backoff roulette. E.g. Gemini free = 15 RPM -> 4.5s.
        self.min_interval = float(os.environ.get("ANALYST_HOSTED_MIN_INTERVAL_S", 0))
        self._last_request = 0.0
        self.api_key = os.environ.get("ANALYST_HOSTED_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ANALYST_HOSTED_API_KEY is not set. Get a free key (no card) at "
                "https://console.groq.com and `export ANALYST_HOSTED_API_KEY=...`"
            )
        self.timeout = routing.limits.get("request_timeout_s", 120)
        self.max_tokens = routing.limits.get("max_output_tokens", 2048)

    def complete(self, *, role: str, system: str, user: str) -> LLMResponse:
        model = self.routing.model_for(role, "hosted")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.min_interval:
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
        for attempt in range(MAX_429_RETRIES + 1):
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload, headers=headers, timeout=self.timeout,
            )
            # Per-day quota won't clear until midnight Pacific — retrying is pure
            # waste (240s per call x every remaining question). Fail the run fast.
            if resp.status_code == 429 and _is_daily_quota(resp):
                raise DailyQuotaExhausted(
                    f"Daily request quota (RPD) exhausted for model {model} on "
                    f"{self.base_url}. Free-tier RPD resets at midnight Pacific. "
                    f"Options: switch to a higher-RPD model (gemini-2.5-flash-lite = "
                    f"1,000/day), split the eval across days, or replay cassettes. "
                    f"Body: {resp.text[:300]}"
                )
            # 429 = rate limit, 5xx = provider hiccup: both transient, both retried
            if (resp.status_code == 429 or resp.status_code >= 500) and attempt < MAX_429_RETRIES:
                wait = _retry_wait(resp, attempt)
                time.sleep(min(wait, 60))
                continue
            if resp.status_code >= 400:
                # Surface the BODY: providers say which quota was violated there.
                raise RuntimeError(
                    f"HTTP {resp.status_code} from {self.base_url} "
                    f"(model {model}): {resp.text[:600]}"
                )
            data = resp.json()
            usage = data.get("usage", {})
            return LLMResponse(
                text=data["choices"][0]["message"]["content"],
                model=model,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
        raise RuntimeError("unreachable")
