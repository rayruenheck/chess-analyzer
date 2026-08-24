"""Tests for the model wrapper.

Mostly about money and honesty: that a cached answer is never re-bought, that a
changed prompt does not serve stale text written under the old one, and that a
truncated response produces an actionable error rather than an opaque crash.

No test here reaches the network -- the client is replaced with a fake whose only
job is to record that it was called.
"""

import asyncio
from contextlib import asynccontextmanager

import pytest
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.db import close_db, init_db
from app.services import llm


class Tiny(BaseModel):
    headline: str


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "llm.db"))
    await init_db()
    yield
    await close_db()


class FakeResponse:
    def __init__(self, parsed, stop_reason="end_turn"):
        self.parsed_output = parsed
        self.stop_reason = stop_reason
        self.usage = type("U", (), {"input_tokens": 100, "output_tokens": 20,
                                    "cache_read_input_tokens": 0})()


def fake_client(monkeypatch, response=None, error=None):
    """Installs a client that records calls and returns `response` (or raises)."""
    calls = []

    class Messages:
        @staticmethod
        @asynccontextmanager
        async def stream(**kwargs):
            calls.append(kwargs)
            if error is not None:
                raise error

            class Stream:
                async def get_final_message(self):
                    return response

            yield Stream()

    class Client:
        messages = Messages()

    monkeypatch.setattr(llm, "get_client", lambda: Client())
    return calls


async def test_missing_key_raises_a_named_error(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(llm, "_client", None)

    with pytest.raises(llm.LLMUnavailable, match="ANTHROPIC_API_KEY"):
        llm.get_client()


async def test_result_is_cached_and_not_bought_twice(db, monkeypatch):
    calls = fake_client(monkeypatch, FakeResponse(Tiny(headline="once")))

    first = await llm.generate("move", "subject", {"a": 1}, Tiny)
    second = await llm.generate("move", "subject", {"a": 1}, Tiny)

    assert first == second == {"headline": "once"}
    assert len(calls) == 1, "a cached result must not trigger a second request"


async def test_refresh_bypasses_the_cache(db, monkeypatch):
    calls = fake_client(monkeypatch, FakeResponse(Tiny(headline="x")))

    await llm.generate("move", "subject", {"a": 1}, Tiny)
    await llm.generate("move", "subject", {"a": 1}, Tiny, refresh=True)

    assert len(calls) == 2


async def test_a_different_payload_is_a_different_cache_entry(db, monkeypatch):
    calls = fake_client(monkeypatch, FakeResponse(Tiny(headline="x")))

    await llm.generate("move", "subject", {"a": 1}, Tiny)
    await llm.generate("move", "subject", {"a": 2}, Tiny)

    assert len(calls) == 2


async def test_cache_key_ignores_dict_ordering():
    """The payload is serialized with sorted keys, so two equal payloads built in
    different orders must not miss each other's cache entry."""
    one = llm._cache_key("move", {"a": 1, "b": 2}, "Tiny")
    two = llm._cache_key("move", {"b": 2, "a": 1}, "Tiny")
    assert one == two


async def test_prompt_version_retires_old_cached_text(monkeypatch):
    """Bumping PROMPT_VERSION must invalidate answers written under the old prompt,
    otherwise an edit to the coaching rubric silently keeps serving the old voice."""
    before = llm._cache_key("report", {"a": 1}, "Tiny")
    monkeypatch.setattr(llm, "PROMPT_VERSION", llm.PROMPT_VERSION + 1)
    assert llm._cache_key("report", {"a": 1}, "Tiny") != before


async def test_model_choice_is_part_of_the_cache_key(monkeypatch):
    before = llm._cache_key("report", {"a": 1}, "Tiny")
    monkeypatch.setattr(settings, "llm_model", "claude-haiku-4-5")
    assert llm._cache_key("report", {"a": 1}, "Tiny") != before


async def test_request_carries_the_expected_shape(db, monkeypatch):
    calls = fake_client(monkeypatch, FakeResponse(Tiny(headline="x")))
    await llm.generate("report", "s", {"a": 1}, Tiny, effort="high")

    request = calls[0]
    assert request["model"] == settings.llm_model
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"]["effort"] == "high"
    assert request["output_format"] is Tiny
    # The stable rubric is the cache prefix and must come first, marked cacheable.
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert request["system"][0]["text"] == llm.SYSTEM_PROMPT


async def test_effort_defaults_to_the_configured_value(db, monkeypatch):
    monkeypatch.setattr(settings, "llm_effort", "low")
    calls = fake_client(monkeypatch, FakeResponse(Tiny(headline="x")))
    await llm.generate("move", "s", {"a": 1}, Tiny)
    assert calls[0]["output_config"]["effort"] == "low"


async def test_truncated_json_names_the_token_limit(db, monkeypatch):
    """Pydantic reports a cut-off response as "EOF while parsing", which names the
    symptom and hides the cause. The wrapper must say what to change."""
    error = ValidationError.from_exception_data("Tiny", [])
    fake_client(monkeypatch, error=error)

    with pytest.raises(RuntimeError, match="LLM_MAX_TOKENS"):
        await llm.generate("report", "s", {"a": 1}, Tiny)


async def test_hitting_max_tokens_is_reported_not_returned(db, monkeypatch):
    fake_client(monkeypatch, FakeResponse(Tiny(headline="cut"), stop_reason="max_tokens"))

    with pytest.raises(RuntimeError, match="token limit"):
        await llm.generate("report", "s", {"a": 1}, Tiny)


async def test_a_refusal_is_reported_not_returned(db, monkeypatch):
    fake_client(monkeypatch, FakeResponse(Tiny(headline="no"), stop_reason="refusal"))

    with pytest.raises(RuntimeError, match="declined"):
        await llm.generate("report", "s", {"a": 1}, Tiny)


async def test_a_failed_call_is_not_cached(db, monkeypatch):
    """A failure must leave nothing behind, so a retry after fixing the cause
    actually re-runs instead of serving the error path's absence as success."""
    fake_client(monkeypatch, FakeResponse(Tiny(headline="x"), stop_reason="max_tokens"))
    with pytest.raises(RuntimeError):
        await llm.generate("report", "s", {"a": 1}, Tiny)

    calls = fake_client(monkeypatch, FakeResponse(Tiny(headline="good")))
    assert await llm.generate("report", "s", {"a": 1}, Tiny) == {"headline": "good"}
    assert len(calls) == 1


async def test_usage_is_recorded_for_cost_tracking(db, monkeypatch):
    import sqlite3

    fake_client(monkeypatch, FakeResponse(Tiny(headline="x")))
    await llm.generate("report", "subject-1", {"a": 1}, Tiny)

    connection = sqlite3.connect(settings.db_path)
    row = connection.execute(
        "SELECT kind, subject, input_tokens, output_tokens FROM llm_feedback"
    ).fetchone()
    connection.close()

    assert row == ("report", "subject-1", 100, 20)


def test_system_prompt_forbids_inventing_chess():
    """The guardrails are load-bearing: without them the model narrates plausible
    lines that the engine never produced."""
    prompt = llm.SYSTEM_PROMPT.lower()
    assert "invent" in prompt
    assert "engine" in prompt
    # The opening section deliberately relaxes this, and must say so explicitly.
    assert "opening coaching" in prompt
    assert "theory_confidence" in llm.SYSTEM_PROMPT
