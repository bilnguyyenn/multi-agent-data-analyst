"""Cassette record/replay — how CI and eval re-runs cost $0 and stay deterministic.

Two replay modes, on purpose:

- HASH mode (recorded runs): cassette key = sha256(role + system + user). Exact —
  any prompt change is a cache miss, which is what you want when replaying a real
  recorded run.
- SCRIPT mode (handwritten tests): an ordered list of responses per role. Tests that
  hand-author "what the model would have said" shouldn't break every time a prompt
  template gets a word tweaked, so scripts key on (role, call order) instead of
  prompt bytes.

A cassette file is just JSON: {"role", "system", "user", "response": {...}} —
inspectable, diffable, committable.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from analyst.llm.client import LLMClient, LLMResponse

DEFAULT_CASSETTE_DIR = Path("cassettes")


def _key(role: str, system: str, user: str) -> str:
    h = hashlib.sha256()
    for part in (role, system, user):
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()[:20]


class CassetteMiss(RuntimeError):
    pass


@dataclass
class ReplayClient:
    cassette_dir: Path
    script: dict[str, list[dict]] | None = None  # role -> ordered responses
    _script_pos: dict[str, int] | None = None

    @classmethod
    def from_env(cls) -> "ReplayClient":
        d = Path(os.environ.get("ANALYST_CASSETTE_DIR", DEFAULT_CASSETTE_DIR))
        script_file = os.environ.get("ANALYST_CASSETTE_SCRIPT")
        script = json.loads(Path(script_file).read_text()) if script_file else None
        return cls(cassette_dir=d, script=script)

    def complete(self, *, role: str, system: str, user: str) -> LLMResponse:
        if self.script is not None:
            return self._from_script(role)
        f = self.cassette_dir / role / f"{_key(role, system, user)}.json"
        if not f.exists():
            raise CassetteMiss(
                f"No cassette for role={role} at {f}.\n"
                f"Record one: ANALYST_RECORD_CASSETTES=1 ANALYST_LLM_PROVIDER=local "
                f"analyst ask ... (then commit the cassettes/ dir)"
            )
        data = json.loads(f.read_text())
        return LLMResponse(**data["response"])

    def _from_script(self, role: str) -> LLMResponse:
        self._script_pos = self._script_pos or {}
        pos = self._script_pos.get(role, 0)
        responses = (self.script or {}).get(role, [])
        if pos >= len(responses):
            raise CassetteMiss(
                f"Script exhausted for role={role} (call #{pos + 1}, "
                f"only {len(responses)} scripted)"
            )
        self._script_pos[role] = pos + 1
        r = responses[pos]
        return LLMResponse(
            text=r["text"],
            model=r.get("model", "scripted"),
            input_tokens=r.get("input_tokens", 0),
            output_tokens=r.get("output_tokens", 0),
        )


@dataclass
class RecordingClient:
    """Wraps a live client; persists every exchange as a cassette."""

    inner: LLMClient
    cassette_dir: Path = DEFAULT_CASSETTE_DIR

    def complete(self, *, role: str, system: str, user: str) -> LLMResponse:
        resp = self.inner.complete(role=role, system=system, user=user)
        d = self.cassette_dir / role
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{_key(role, system, user)}.json").write_text(
            json.dumps(
                {
                    "role": role,
                    "system": system,
                    "user": user,
                    "response": {
                        "text": resp.text,
                        "model": resp.model,
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                    },
                },
                indent=2,
            )
        )
        return resp
