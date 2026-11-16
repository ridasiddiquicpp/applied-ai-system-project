"""Tests for ai_parser.py. They use a fake Claude client, so no API key,
network access or credit is needed."""
import json
from datetime import date, datetime
from types import SimpleNamespace

import anthropic
import pytest

import ai_parser
from ai_parser import (
    AIParserError, Clarification, ParsedAddPet, ParsedRemovePet, ParsedTask,
    find_pet_to_remove, parse_request, to_pet, to_task,
)
from pawpal_system import Owner, Pet, Task

TODAY = date(2026, 11, 14)

BASE = {
    "action": "add_task", "status": "ok", "question": "", "pet_name": "",
    "description": "", "due_datetime": "", "duration_minutes": 0,
    "priority": "medium", "frequency": "once", "species": "dog", "breed": "", "age": 0,
}


def fake_client(payload=None, stop_reason="end_turn", raw_text=None, error=None):
    """A stand-in for anthropic.Anthropic whose messages.create returns a canned
    reply (or raises `error`). The kwargs of the last call are kept in .calls."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error
        text = raw_text if raw_text is not None else json.dumps(payload)
        return SimpleNamespace(
            stop_reason=stop_reason,
            content=[SimpleNamespace(type="text", text=text)],
        )

    return SimpleNamespace(messages=SimpleNamespace(create=create), calls=calls)


def task_payload(**overrides):
    return {
        **BASE, "pet_name": "Mochi", "description": "Morning walk",
        "due_datetime": "2026-11-15T08:00", "duration_minutes": 30,
        "priority": "high", "frequency": "daily", **overrides,
    }


def parse(client, text="do something", pets=("Mochi",)):
    return parse_request(text, list(pets), TODAY, client)


def api_error(cls, status):
    """Build an Anthropic API error without a real HTTP response."""
    response = SimpleNamespace(request=None, status_code=status, headers={})
    return cls("boom", response=response, body=None)


# ---------------------------------------------------------------- parse_request: results

def test_task_request_is_parsed():
    result = parse(fake_client(task_payload()))

    assert result == ParsedTask(
        pet_name="Mochi", description="Morning walk",
        due_time=datetime(2026, 11, 15, 8, 0), duration_minutes=30,
        priority="high", frequency="daily",
    )


def test_add_pet_request_is_parsed_and_trimmed():
    payload = {**BASE, "action": "add_pet", "pet_name": " Luna ", "species": "cat",
               "breed": " Siamese ", "age": 3}

    assert parse(fake_client(payload)) == ParsedAddPet("Luna", "cat", "Siamese", 3)


def test_negative_age_is_clamped_to_zero():
    payload = {**BASE, "action": "add_pet", "pet_name": "Luna", "species": "cat", "age": -2}

    assert parse(fake_client(payload)).age == 0


def test_add_pet_works_when_owner_has_no_pets():
    payload = {**BASE, "action": "add_pet", "pet_name": "Luna", "species": "cat"}

    assert parse(fake_client(payload), pets=()) == ParsedAddPet("Luna", "cat", "", 0)


def test_remove_pet_request_is_parsed():
    payload = {**BASE, "action": "remove_pet", "pet_name": " Max "}

    assert parse(fake_client(payload)) == ParsedRemovePet("Max")


def test_missing_information_returns_a_question():
    payload = {**BASE, "status": "needs_clarification", "question": "How long is the vet visit?"}

    assert parse(fake_client(payload)) == Clarification("How long is the vet visit?")


def test_clarification_without_a_question_gets_a_default():
    payload = {**BASE, "status": "needs_clarification", "question": "  "}

    result = parse(fake_client(payload))

    assert isinstance(result, Clarification) and result.question


# ---------------------------------------------------------------- parse_request: the request sent

def test_request_includes_date_pets_text_and_schema():
    client = fake_client(task_payload())

    parse(client, text="walk Mochi", pets=("Mochi", "Max"))

    sent = client.calls[0]
    assert sent["model"] == ai_parser.MODEL
    assert sent["max_tokens"] == ai_parser.MAX_TOKENS
    assert sent["output_config"]["format"]["schema"] == ai_parser.TASK_SCHEMA
    user_message = sent["messages"][0]["content"]
    assert "2026-11-14" in user_message and "Saturday" in user_message
    assert "Mochi, Max" in user_message and "walk Mochi" in user_message


def test_no_pets_is_sent_as_none():
    client = fake_client({**BASE, "action": "add_pet", "pet_name": "Luna", "species": "cat"})

    parse(client, pets=())

    assert "(none)" in client.calls[0]["messages"][0]["content"]


def test_schema_requires_every_field():
    schema = ai_parser.TASK_SCHEMA
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False


# ---------------------------------------------------------------- parse_request: bad input and output

@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_text_is_rejected_without_calling_claude(text):
    client = fake_client(task_payload())

    with pytest.raises(AIParserError, match="type or say"):
        parse_request(text, ["Mochi"], TODAY, client)

    assert client.calls == []


def test_refusal_becomes_a_friendly_error():
    with pytest.raises(AIParserError, match="declined"):
        parse(fake_client(task_payload(), stop_reason="refusal"))


def test_cut_off_answer_becomes_a_friendly_error():
    with pytest.raises(AIParserError, match="cut off"):
        parse(fake_client(task_payload(), stop_reason="max_tokens"))


@pytest.mark.parametrize("raw", ["not json at all", "[]", '"just a string"', ""])
def test_unreadable_answer_becomes_a_friendly_error(raw):
    with pytest.raises(AIParserError, match="couldn't read"):
        parse(fake_client(raw_text=raw))


def test_invalid_date_becomes_a_friendly_error():
    with pytest.raises(AIParserError, match="invalid field"):
        parse(fake_client(task_payload(due_datetime="soon")))


def test_unknown_action_becomes_a_friendly_error():
    with pytest.raises(AIParserError, match="supported action"):
        parse(fake_client({**BASE, "action": "fly"}))


def test_answer_missing_a_field_becomes_a_friendly_error():
    payload = task_payload()
    del payload["duration_minutes"]

    with pytest.raises(AIParserError, match="invalid field"):
        parse(fake_client(payload))


# ---------------------------------------------------------------- parse_request: API failures

@pytest.mark.parametrize("error, message", [
    (TypeError("Could not resolve authentication method"), "No Anthropic API key found"),
    (api_error(anthropic.AuthenticationError, 401), "missing or invalid"),
    (api_error(anthropic.RateLimitError, 429), "Too many requests"),
    (api_error(anthropic.APIStatusError, 500), r"error \(500\)"),
    (anthropic.APIConnectionError(message="down", request=None), "internet connection"),
])
def test_api_failures_become_friendly_errors(error, message):
    with pytest.raises(AIParserError, match=message):
        parse(fake_client(error=error))


# ---------------------------------------------------------------- validation

@pytest.fixture
def owner():
    """An owner with one pet, Mochi, who has two tasks."""
    owner = Owner(1, "Jordan")
    mochi = Pet(1, "Mochi", "dog")
    owner.add_pet(mochi)
    mochi.add_task(Task(1, "walk", "general", datetime(2026, 11, 15, 8, 0), 30))
    mochi.add_task(Task(2, "feed", "general", datetime(2026, 11, 15, 9, 0), 10))
    return owner


def parsed_task(**overrides):
    fields = dict(pet_name="Mochi", description="Walk", due_time=datetime(2026, 11, 16, 8, 0),
                  duration_minutes=30, priority="high", frequency="daily")
    return ParsedTask(**{**fields, **overrides})


def test_to_task_builds_a_task_for_the_matching_pet(owner):
    pet, task = to_task(parsed_task(pet_name=" MOCHI "), owner, 3)

    assert pet is owner.pets[0]
    assert (task.task_id, task.description, task.duration_minutes) == (3, "Walk", 30)
    assert str(task.priority) == "high" and str(task.frequency) == "daily"


def test_to_task_does_not_add_the_task(owner):
    pet, task = to_task(parsed_task(), owner, 3)

    assert len(pet.get_tasks()) == 2 and task.pet is None


def test_to_task_rejects_unknown_pet_and_lists_real_ones(owner):
    with pytest.raises(AIParserError, match=r"'Luna' isn't one of your pets \(your pets: Mochi\)"):
        to_task(parsed_task(pet_name="Luna"), owner, 3)


@pytest.mark.parametrize("duration", [0, -5])
def test_to_task_rejects_non_positive_duration(owner, duration):
    with pytest.raises(AIParserError, match="at least 1 minute"):
        to_task(parsed_task(duration_minutes=duration), owner, 3)


def test_to_task_rejects_empty_description(owner):
    with pytest.raises(AIParserError, match="description"):
        to_task(parsed_task(description=""), owner, 3)


@pytest.mark.parametrize("field, value", [("priority", "urgent"), ("frequency", "monthly")])
def test_to_task_rejects_invalid_priority_or_frequency(owner, field, value):
    with pytest.raises(AIParserError, match="invalid priority or frequency"):
        to_task(parsed_task(**{field: value}), owner, 3)


def test_to_task_accepts_any_casing_for_priority(owner):
    _, task = to_task(parsed_task(priority="HIGH"), owner, 3)

    assert str(task.priority) == "high"


def test_to_pet_builds_a_pet_without_adding_it(owner):
    pet = to_pet(ParsedAddPet("Luna", "cat", "Siamese", 3), owner, 2)

    assert (pet.pet_id, pet.name, pet.species, pet.breed, pet.age) == (2, "Luna", "cat", "Siamese", 3)
    assert [p.name for p in owner.pets] == ["Mochi"]


@pytest.mark.parametrize("name", ["Mochi", "mochi", " MOCHI "])
def test_to_pet_rejects_a_duplicate_name_ignoring_case(owner, name):
    with pytest.raises(AIParserError, match="already have a pet"):
        to_pet(ParsedAddPet(name, "cat"), owner, 2)


def test_to_pet_rejects_empty_name(owner):
    with pytest.raises(AIParserError, match="needs a name"):
        to_pet(ParsedAddPet("", "cat"), owner, 2)


def test_to_pet_rejects_unknown_species(owner):
    with pytest.raises(AIParserError, match="dog, cat or other"):
        to_pet(ParsedAddPet("Rex", "dragon"), owner, 2)


def test_find_pet_to_remove_reports_task_count_without_removing(owner):
    pet, task_count = find_pet_to_remove(ParsedRemovePet("mochi"), owner)

    assert pet is owner.pets[0] and task_count == 2
    assert [p.name for p in owner.pets] == ["Mochi"]  # nothing removed


def test_find_pet_to_remove_rejects_unknown_pet(owner):
    with pytest.raises(AIParserError, match="'Max' isn't one of your pets"):
        find_pet_to_remove(ParsedRemovePet("Max"), owner)


def test_validation_with_no_pets_lists_none_yet():
    with pytest.raises(AIParserError, match="none yet"):
        find_pet_to_remove(ParsedRemovePet("Max"), Owner(1, "Jordan"))
