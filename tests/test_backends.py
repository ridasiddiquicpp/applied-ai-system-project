"""Tests for the local-model (Ollama) adapter and for choosing between Claude
and the local model. Everything is faked: no key, no Ollama and no network."""
import io
import json
import urllib.error
from datetime import date, datetime

import pytest

import ai_parser
from ai_parser import (
    AIParserError, Clarification, OllamaClient, ParsedAddPet, ParsedRemovePet,
    ParsedTask, parse_request,
)

TODAY = date(2026, 11, 14)

BASE = {
    "action": "add_task", "status": "ok", "question": "", "pet_name": "",
    "description": "", "due_datetime": "", "duration_minutes": 0,
    "priority": "medium", "frequency": "once", "species": "dog", "breed": "", "age": 0,
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test with no AI-related environment variables."""
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "PAWPAL_AI"):
        monkeypatch.delenv(name, raising=False)


def fake_urlopen(monkeypatch, reply=None, raw=None, error=None):
    """Replace urllib's urlopen; the requests it receives are collected in .sent."""
    sent = []

    def urlopen(request, timeout=None):
        sent.append(SimpleBody(request, timeout))
        if error is not None:
            raise error
        data = raw if raw is not None else json.dumps(reply)
        return io.BytesIO(data.encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return sent


class SimpleBody:
    def __init__(self, request, timeout):
        self.url = request.full_url
        self.body = json.loads(request.data)
        self.timeout = timeout


def ollama_reply(payload, done_reason="stop"):
    return {"message": {"role": "assistant", "content": json.dumps(payload)}, "done_reason": done_reason}


# ---------------------------------------------------------------- choosing a backend

def test_no_key_uses_the_local_model():
    assert ai_parser.active_backend() == "ollama"
    assert isinstance(ai_parser.default_client(), OllamaClient)
    assert "Ollama" in ai_parser.backend_label()


@pytest.mark.parametrize("variable", ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"])
def test_a_claude_credential_uses_claude(monkeypatch, variable):
    monkeypatch.setenv(variable, "x")

    assert ai_parser.active_backend() == "claude"
    assert "claude-haiku-5-5" in ai_parser.backend_label()


def test_pawpal_ai_forces_a_backend(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("PAWPAL_AI", "OLLAMA")  # any casing
    assert ai_parser.active_backend() == "ollama"

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setenv("PAWPAL_AI", "claude")
    assert ai_parser.active_backend() == "claude"


def test_unknown_pawpal_ai_value_is_ignored(monkeypatch):
    monkeypatch.setenv("PAWPAL_AI", "something-else")

    assert ai_parser.active_backend() == "ollama"


# ---------------------------------------------------------------- OllamaClient request

def test_request_goes_to_ollama_with_schema_prompt_and_temperature_zero(monkeypatch):
    sent = fake_urlopen(monkeypatch, ollama_reply(BASE))
    client = OllamaClient(model="my-model", url="http://example:1234/")

    client.messages.create(
        model="claude-haiku-5-5",  # Claude-only arguments are ignored
        max_tokens=500,
        system="SYSTEM TEXT",
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}},
        messages=[{"role": "user", "content": "hello"}],
    )

    request = sent[0]
    assert request.url == "http://example:1234/api/chat"
    assert request.body["model"] == "my-model"
    assert request.body["stream"] is False
    assert request.body["format"] == {"type": "object"}
    assert request.body["options"] == {"temperature": 0, "num_predict": 500}
    assert request.body["messages"] == [
        {"role": "system", "content": "SYSTEM TEXT"},
        {"role": "user", "content": "hello"},
    ]


def test_request_without_a_schema_sends_no_format(monkeypatch):
    sent = fake_urlopen(monkeypatch, ollama_reply(BASE))

    OllamaClient().create(system="s", messages=[{"role": "user", "content": "hi"}])

    assert "format" not in sent[0].body


def test_reply_looks_like_a_claude_response(monkeypatch):
    fake_urlopen(monkeypatch, ollama_reply({"a": 1}))

    response = OllamaClient().create(system="s", messages=[])

    assert response.stop_reason == "end_turn"
    assert response.content[0].type == "text"
    assert json.loads(response.content[0].text) == {"a": 1}


def test_a_length_cutoff_maps_to_max_tokens(monkeypatch):
    fake_urlopen(monkeypatch, ollama_reply({}, done_reason="length"))

    assert OllamaClient().create(system="s", messages=[]).stop_reason == "max_tokens"


# ---------------------------------------------------------------- OllamaClient failures

def http_error(code):
    return urllib.error.HTTPError("http://x", code, "err", {}, None)


def test_missing_model_tells_the_user_how_to_install_it(monkeypatch):
    fake_urlopen(monkeypatch, error=http_error(404))

    with pytest.raises(AIParserError, match="ollama pull my-model"):
        OllamaClient(model="my-model").create(system="s", messages=[])


def test_server_error_becomes_a_friendly_error(monkeypatch):
    fake_urlopen(monkeypatch, error=http_error(500))

    with pytest.raises(AIParserError, match=r"error \(500\)"):
        OllamaClient().create(system="s", messages=[])


@pytest.mark.parametrize("error", [
    urllib.error.URLError("refused"), ConnectionRefusedError(), TimeoutError(),
])
def test_ollama_not_running_becomes_a_friendly_error(monkeypatch, error):
    fake_urlopen(monkeypatch, error=error)

    with pytest.raises(AIParserError, match="Start the Ollama app"):
        OllamaClient().create(system="s", messages=[])


def test_unreadable_server_reply_becomes_a_friendly_error(monkeypatch):
    fake_urlopen(monkeypatch, raw="<html>not json</html>")

    with pytest.raises(AIParserError, match="couldn't read"):
        OllamaClient().create(system="s", messages=[])


# ---------------------------------------------------------------- the whole parser through the local model

def test_parse_request_works_end_to_end_with_the_local_model(monkeypatch):
    payload = {**BASE, "pet_name": "Mochi", "description": "Walk",
               "due_datetime": "2026-11-15T08:00", "duration_minutes": 30,
               "priority": "high", "frequency": "daily"}
    sent = fake_urlopen(monkeypatch, ollama_reply(payload))

    result = parse_request("walk Mochi daily at 8", ["Mochi"], TODAY)  # no client given: auto-selected

    assert result == ParsedTask("Mochi", "Walk", datetime(2026, 11, 15, 8, 0), 30, "high", "daily")
    assert "localhost:11434" in sent[0].url


def test_parse_request_reports_ollama_being_down(monkeypatch):
    fake_urlopen(monkeypatch, error=urllib.error.URLError("refused"))

    with pytest.raises(AIParserError, match="Ollama"):
        parse_request("walk Mochi", ["Mochi"], TODAY)


# ---------------------------------------------------------------- safety net when the model skips a question

def reply_for(monkeypatch, **overrides):
    fake_urlopen(monkeypatch, ollama_reply({**BASE, **overrides}))


def ask(text="x", pets=("Mochi",)):
    return parse_request(text, list(pets), TODAY)


TASK = dict(pet_name="Mochi", description="Walk", due_datetime="2026-11-15T08:00", duration_minutes=30)


@pytest.mark.parametrize("missing, question", [
    (dict(duration_minutes=0), "How long"),
    (dict(duration_minutes=-5), "How long"),
    (dict(pet_name=""), "Which pet"),
    (dict(pet_name="   "), "Which pet"),
    (dict(due_datetime=""), "date and time"),
])
def test_task_with_a_missing_field_becomes_a_question(monkeypatch, missing, question):
    reply_for(monkeypatch, **{**TASK, **missing})

    result = ask()

    assert isinstance(result, Clarification) and question in result.question


def test_add_pet_without_a_name_becomes_a_question(monkeypatch):
    reply_for(monkeypatch, action="add_pet", pet_name="", species="cat")

    assert ask() == Clarification("What is the new pet's name?")


def test_remove_pet_without_a_name_becomes_a_question(monkeypatch):
    reply_for(monkeypatch, action="remove_pet", pet_name="")

    assert ask(pets=("Mochi", "Max")) == Clarification("Which pet should I remove?")


def test_complete_replies_are_not_turned_into_questions(monkeypatch):
    reply_for(monkeypatch, **TASK)
    assert isinstance(ask(), ParsedTask)

    reply_for(monkeypatch, action="add_pet", pet_name="Luna", species="cat")
    assert isinstance(ask(), ParsedAddPet)

    reply_for(monkeypatch, action="remove_pet", pet_name="Max")
    assert ask() == ParsedRemovePet("Max")


# ---------------------------------------------------------------- weekday help in the prompt

def test_request_lists_the_next_seven_days(monkeypatch):
    sent = fake_urlopen(monkeypatch, ollama_reply({**BASE, **TASK}))

    ask()

    user_message = sent[0].body["messages"][1]["content"]
    assert "Next 7 days:" in user_message
    assert "Sunday 2026-11-15" in user_message and "Saturday 2026-11-21" in user_message
    assert "Friday 2026-11-20" in user_message
