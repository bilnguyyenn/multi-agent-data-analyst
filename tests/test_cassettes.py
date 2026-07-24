"""Record -> replay round-trip: what makes CI free must actually work."""
import pytest

from analyst.llm.client import LLMResponse
from analyst.llm.replay import CassetteMiss, RecordingClient, ReplayClient


class FakeLive:
    def complete(self, *, role, system, user):
        return LLMResponse(text=f"reply-to:{user}", model="fake-live", input_tokens=3,
                           output_tokens=5)


def test_record_then_replay_hash_mode(tmp_path):
    rec = RecordingClient(inner=FakeLive(), cassette_dir=tmp_path)
    live = rec.complete(role="planner", system="s", user="u1")

    replay = ReplayClient(cassette_dir=tmp_path)
    replayed = replay.complete(role="planner", system="s", user="u1")
    assert replayed.text == live.text
    assert replayed.output_tokens == 5


def test_replay_miss_is_loud(tmp_path):
    replay = ReplayClient(cassette_dir=tmp_path)
    with pytest.raises(CassetteMiss, match="Record one"):
        replay.complete(role="planner", system="s", user="never-recorded")


def test_prompt_change_is_a_miss(tmp_path):
    RecordingClient(inner=FakeLive(), cassette_dir=tmp_path).complete(
        role="executor", system="s", user="original prompt"
    )
    replay = ReplayClient(cassette_dir=tmp_path)
    with pytest.raises(CassetteMiss):
        replay.complete(role="executor", system="s", user="tweaked prompt")
