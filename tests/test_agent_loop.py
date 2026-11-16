"""Tests for the self-correcting loop (parse_and_validate). A scripted fake model
gives a different answer on each attempt, so no key, Ollama or network is needed."""
import json
from datetime import date, datetime
from types import SimpleNamespace

import anthropic
import pytest

import ai_parser
from ai_parser import (
    AgentResult, AIParserError, Clarification, next_pet_id, next_task_id, parse_and_validate,
)
from pawpal_system import Owner, Pet, Task

TODAY = date(2026, 11, 14)

BASE = {
    "action": "add_task", "status": "ok", "question": "", "pet_name": "",
    "description": "", "due_datetime": "", "duration_minutes": 0,
    "priority": "medium", "frequency": "once", "species": "dog", "breed": "", "age": 0,
}
TASK = {**BASE, "pet_name": "Mochi", "description": "Walk",
        "due_datetime": "2026-11-15T08:00", "duration_minutes": 30}


class ScriptedClient:
    """Answers with the next item of `script`: a dict is a model answer, an
    exception is raised. Each call's messages are saved in .conversations."""

    def __init__(self, *script):
        self.script, self.conversations = list(script), []
        self.messages = self

    def create(self, **kwargs):
        self.conversations.append(list(kwargs["messages"]))  # snapshot: the loop appends to it
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            stop_reason=item.pop("_stop", "end_turn") if "_stop" in item else "end_turn",
            content=[SimpleNamespace(type="text", text=json.dumps({k: v for k, v in item.items() if k != "_stop"}))],
        )

    @property
    def calls(self):
        return len(self.conversations)


@pytest.fixture
def owner():
    owner = Owner(1, "Jordan")
    mochi = Pet(1, "Mochi", "dog")
    owner.add_pet(mochi)
    mochi.add_task(Task(7, "feed", "general", datetime(2026, 11, 15, 9, 0), 10))
    return owner


def run(owner, client, text="do it", **kwargs):
    return parse_and_validate(text, owner, TODAY, client, **kwargs)


# ---------------------------------------------------------------- the normal path

def test_a_valid_first_answer_needs_one_call_and_no_corrections(owner):
    client = ScriptedClient(TASK)

    result = run(owner, client)

    assert isinstance(result, AgentResult)
    assert result.corrections == [] and client.calls == 1
    kind, pet, task = result.outcome
    assert kind == "task" and pet is owner.pets[0]
    assert (task.description, task.duration_minutes, task.task_id) == ("Walk", 30, 8)  # max id 7, plus 1


def test_add_pet_and_remove_pet_outcomes(owner):
    add = run(owner, ScriptedClient({**BASE, "action": "add_pet", "pet_name": "Luna", "species": "cat"}))
    assert add.outcome[0] == "add_pet" and add.outcome[1].name == "Luna" and add.outcome[1].pet_id == 2

    remove = run(owner, ScriptedClient({**BASE, "action": "remove_pet", "pet_name": "mochi"}))
    assert remove.outcome == ("remove_pet", owner.pets[0], 1)


def test_nothing_in_the_owner_changes(owner):
    run(owner, ScriptedClient(TASK))
    run(owner, ScriptedClient({**BASE, "action": "remove_pet", "pet_name": "Mochi"}))

    assert [p.name for p in owner.pets] == ["Mochi"] and len(owner.get_all_tasks()) == 1


# ---------------------------------------------------------------- the model fixes its own mistake

def test_the_model_corrects_an_unknown_pet_on_the_second_try(owner):
    client = ScriptedClient({**TASK, "pet_name": "Mocha"}, TASK)

    result = run(owner, client)

    assert result.outcome[0] == "task" and result.outcome[1].name == "Mochi"
    assert client.calls == 2
    assert len(result.corrections) == 1 and "'Mocha' isn't one of your pets" in result.corrections[0]


def test_the_error_is_sent_back_with_the_models_own_answer(owner):
    client = ScriptedClient({**TASK, "pet_name": "Mocha"}, TASK)

    run(owner, client)

    second = client.conversations[1]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert json.loads(second[1]["content"])["pet_name"] == "Mocha"   # what it said before
    assert "'Mocha' isn't one of your pets" in second[2]["content"]  # why that failed
    assert "Never swap in a different pet" in second[2]["content"]   # guard against "fixing" by guessing


@pytest.mark.parametrize("bad", [
    {"due_datetime": "soon"},        # unreadable date
    {"priority": "urgent"},          # priority the app doesn't have
    {"description": ""},             # nothing to call the task
])
def test_other_fixable_mistakes_are_retried(owner, bad):
    client = ScriptedClient({**TASK, **bad}, TASK)

    result = run(owner, client)

    assert result.outcome[0] == "task" and client.calls == 2 and len(result.corrections) == 1


def test_corrections_pile_up_across_attempts(owner):
    client = ScriptedClient({**TASK, "pet_name": "Mocha"}, {**TASK, "priority": "urgent"}, TASK)

    result = run(owner, client)

    assert client.calls == 3 and len(result.corrections) == 2
    assert [m["role"] for m in client.conversations[2]] == ["user", "assistant", "user", "assistant", "user"]


# ---------------------------------------------------------------- giving up

def test_it_gives_up_after_the_attempt_limit_and_raises_the_last_error(owner):
    client = ScriptedClient(*[{**TASK, "pet_name": "Mocha"}] * 5)

    with pytest.raises(AIParserError, match="'Mocha' isn't one of your pets"):
        run(owner, client)

    assert client.calls == ai_parser.MAX_ATTEMPTS == 3


def test_max_attempts_one_means_no_retry(owner):
    client = ScriptedClient({**TASK, "pet_name": "Mocha"}, TASK)

    with pytest.raises(AIParserError):
        run(owner, client, max_attempts=1)

    assert client.calls == 1


# ---------------------------------------------------------------- questions end the loop

def test_a_question_ends_the_loop_at_once(owner):
    client = ScriptedClient({**BASE, "status": "needs_clarification", "question": "Which pet?"}, TASK)

    result = run(owner, client)

    assert result.outcome == Clarification("Which pet?") and client.calls == 1


def test_a_question_after_a_failed_attempt_keeps_the_correction_note(owner):
    client = ScriptedClient(
        {**TASK, "pet_name": "Rex"},
        {**BASE, "status": "needs_clarification", "question": "Which of your pets did you mean?"},
    )

    result = run(owner, client)

    assert result.outcome == Clarification("Which of your pets did you mean?")
    assert client.calls == 2 and len(result.corrections) == 1


# ---------------------------------------------------------------- things that must NOT be retried

def test_adding_a_duplicate_pet_is_not_retried(owner):
    client = ScriptedClient({**BASE, "action": "add_pet", "pet_name": "mochi", "species": "cat"}, TASK)

    with pytest.raises(AIParserError, match="already have a pet"):
        run(owner, client)

    assert client.calls == 1


def test_removing_an_unknown_pet_is_not_retried_so_the_model_cannot_pick_another(owner):
    client = ScriptedClient({**BASE, "action": "remove_pet", "pet_name": "Rex"},
                            {**BASE, "action": "remove_pet", "pet_name": "Mochi"})

    with pytest.raises(AIParserError, match="'Rex' isn't one of your pets"):
        run(owner, client)

    assert client.calls == 1
    assert len(owner.pets) == 1  # nothing was removed either


@pytest.mark.parametrize("failure", [
    anthropic.RateLimitError("boom", response=SimpleNamespace(request=None, status_code=429, headers={}), body=None),
    anthropic.APIConnectionError(message="down", request=None),
    TypeError("no credentials"),
])
def test_api_failures_are_not_retried(owner, failure):
    client = ScriptedClient(failure, TASK)

    with pytest.raises(AIParserError):
        run(owner, client)

    assert client.calls == 1


def test_a_refusal_is_not_retried(owner):
    client = ScriptedClient({**TASK, "_stop": "refusal"}, TASK)

    with pytest.raises(AIParserError, match="declined"):
        run(owner, client)

    assert client.calls == 1


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_text_is_rejected_without_calling_the_model(owner, text):
    client = ScriptedClient(TASK)

    with pytest.raises(AIParserError, match="type or say"):
        run(owner, client, text=text)

    assert client.calls == 0


# ---------------------------------------------------------------- the error flag and id helpers

def test_errors_are_retryable_by_default_and_can_opt_out():
    assert AIParserError("x").retryable is True
    assert AIParserError("x", retryable=False).retryable is False
    assert str(AIParserError("message")) == "message"


def test_id_helpers_avoid_collisions(owner):
    assert next_task_id(owner) == 8 and next_pet_id(owner) == 2
    assert next_task_id(Owner(2, "Empty")) == 1 and next_pet_id(Owner(2, "Empty")) == 1
