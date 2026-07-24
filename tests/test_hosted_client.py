"""HostedClient: routing resolution, response parsing, 429 retry behavior."""
import httpx
import pytest

from analyst.config import load_routing
from analyst.llm.openai_compat import HostedClient


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("ANALYST_HOSTED_API_KEY", "test-key")
    return HostedClient(load_routing())


def _response(status=200, json_body=None, headers=None):
    return httpx.Response(
        status_code=status,
        json=json_body,
        headers=headers or {},
        request=httpx.Request("POST", "http://test"),
    )


OK_BODY = {
    "choices": [{"message": {"content": "the answer"}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
}


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("ANALYST_HOSTED_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="console.groq.com"):
        HostedClient(load_routing())


def test_roles_resolve_through_the_routing_table(client, monkeypatch):
    """Roles must route via config, never hardcode. (Both tiers may point at the
    same hosted model — that's a config choice the eval varies, not a bug.)"""
    seen = []

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.append(json["model"])
        return _response(json_body=OK_BODY)

    monkeypatch.setattr(httpx, "post", fake_post)
    client.complete(role="planner", system="s", user="u")
    client.complete(role="executor", system="s", user="u")
    client.complete(role="council_statistical", system="s", user="u")
    routing = client.routing
    assert seen[0] == routing.tiers[routing.roles["planner"]]["hosted"]
    assert seen[1] == routing.tiers[routing.roles["executor"]]["hosted"]
    assert seen[2] == routing.tiers[routing.roles["council_statistical"]]["hosted"]
    assert "70b" in seen[0].lower()  # planner is reasoning-tier


def test_parses_text_and_usage(client, monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _response(json_body=OK_BODY))
    r = client.complete(role="planner", system="s", user="u")
    assert r.text == "the answer"
    assert (r.input_tokens, r.output_tokens) == (10, 4)


def test_429_retries_then_succeeds(client, monkeypatch):
    calls = {"n": 0}

    def fake_post(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return _response(status=429, json_body={}, headers={"retry-after": "0"})
        return _response(json_body=OK_BODY)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    r = client.complete(role="executor", system="s", user="u")
    assert r.text == "the answer"
    assert calls["n"] == 2
