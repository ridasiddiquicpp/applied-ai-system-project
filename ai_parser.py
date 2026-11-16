"""Turn a natural-language request into a task or pet action using an AI model.

The model only extracts fields from text. Nothing here saves or deletes
anything: the caller shows a preview, and only after the user confirms does
anything change (tasks go through Scheduler.add_task, which still enforces the
conflict rules).

Two backends, same prompt, schema and validation:
- Claude (Anthropic API), used when ANTHROPIC_API_KEY is set.
- A free local model through Ollama, used when there is no Claude key.
Set PAWPAL_AI=claude or PAWPAL_AI=ollama to force one.

Supported actions: add_task, add_pet, remove_pet.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import anthropic

from pawpal_system import Owner, Pet, Task

# Cheap and fast; plenty for pulling fields out of one sentence.
MODEL = "claude-haiku-5-5"
MAX_TOKENS = 2048

# Free local backend. Install Ollama, then: ollama pull qwen2.5:7b-instruct
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = os.environ.get("PAWPAL_OLLAMA_MODEL", "qwen2.5:7b-instruct")

SYSTEM_PROMPT = """You convert a pet owner's request into exactly one action.

Pick "action":
- add_task: schedule a pet-care task (walk, feeding, medication, vet visit...).
- add_pet: add a new pet to the owner's household.
- remove_pet: remove an existing pet.

Fields (fill every field; for ones that don't apply or can't be determined \
use "" , or 0 for numbers, and the defaults below):
- pet_name:
  - add_task / remove_pet: must be exactly one of the owner's current pet \
names given in the request. If the owner has only one pet and none is named \
for add_task, use that pet.
  - add_pet: the new pet's name.
- description (add_task): a short label such as "Morning walk".
- due_datetime (add_task): local time as YYYY-MM-DDTHH:MM. Resolve words like \
"tomorrow" or "Friday" using the current date given in the request.
- duration_minutes (add_task): whole minutes, only if the owner states how \
long. Never infer it from the kind of task: a vet visit, a walk or a feeding \
has no default length.
- priority (add_task): high, medium or low. Use medium if not said.
- frequency (add_task): once, daily or weekly. Use once if not said.
- species (add_pet): dog, cat or other. Use other for any animal that isn't a \
dog or cat (rabbit, bird, ...).
- breed (add_pet): "" if not said.
- age (add_pet): whole years, 0 if not said.

Asking instead of guessing: if something required is missing or unclear, set \
status to "needs_clarification" and put exactly one short question in \
"question" (ask about only the first missing item). Never guess these:
- add_task: the pet, the date or time, or the duration. If the request does \
not say how long, you MUST ask for the duration.
- add_pet: the name or the species.
- remove_pet: which pet.
If the pet named is not one of the owner's pets, ask which of their pets was \
meant; do not substitute a different pet. If the owner has two or more pets and \
the request names none (for example "remove one of my pets"), leave pet_name \
empty and ask which pet. Removing a pet deletes data, so never pick one for them.
For weekdays, use the date listed in "Next 7 days" in the request.
If the owner has no pets and asks for add_task or remove_pet, ask them to add \
a pet first. If the request is none of the three actions, ask what they would \
like to do. Otherwise set status to "ok" and leave question empty."""

TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["add_task", "add_pet", "remove_pet"]},
        "status": {"type": "string", "enum": ["ok", "needs_clarification"]},
        "question": {"type": "string"},
        "pet_name": {"type": "string"},
        "description": {"type": "string"},
        "due_datetime": {"type": "string"},
        "duration_minutes": {"type": "integer"},
        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
        "frequency": {"type": "string", "enum": ["once", "daily", "weekly"]},
        "species": {"type": "string", "enum": ["dog", "cat", "other"]},
        "breed": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": [
        "action", "status", "question", "pet_name", "description",
        "due_datetime", "duration_minutes", "priority", "frequency",
        "species", "breed", "age",
    ],
    "additionalProperties": False,
}


class AIParserError(Exception):
    """The request couldn't be understood; the message is safe to show the user.

    `retryable` says whether asking the model to try again could fix it. It is
    False for things the model can't fix (the user asked to add a pet that
    already exists) or must never "fix" on its own (which pet to delete).
    """

    def __init__(self, message: str = "", retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class OllamaClient:
    """Makes a local Ollama model look like the Anthropic client.

    It has the same `messages.create(...)` method and returns an object with
    `stop_reason` and `content` blocks, so the rest of this module doesn't
    care which backend answered. Free, runs on this machine, needs no key.
    """

    def __init__(self, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL, timeout: float = 300.0):
        self.model, self.url, self.timeout = model, url.rstrip("/"), timeout
        self.messages = self  # so client.messages.create(...) works

    def create(self, *, system: str = "", messages=(), max_tokens: int = MAX_TOKENS,
               output_config: Optional[dict] = None, **_ignored):
        """Send one chat request to Ollama; extra Claude-only arguments are ignored."""
        body = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0, "num_predict": max_tokens},  # temperature 0: repeatable answers
            "messages": [{"role": "system", "content": system}, *messages],
        }
        schema = ((output_config or {}).get("format") or {}).get("schema")
        if schema:
            body["format"] = schema  # Ollama's structured output: constrains the reply to the schema

        request = urllib.request.Request(
            f"{self.url}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise AIParserError(
                    f"The local model '{self.model}' isn't installed. Run: ollama pull {self.model}"
                )
            raise AIParserError(f"The local model server returned an error ({error.code}).")
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            raise AIParserError(
                "Couldn't reach the local AI model (Ollama). Start the Ollama app, "
                "or set ANTHROPIC_API_KEY to use Claude instead."
            )
        except json.JSONDecodeError:
            raise AIParserError("The local model returned an answer I couldn't read. Please try again.")

        stop_reason = "max_tokens" if data.get("done_reason") == "length" else "end_turn"
        text = (data.get("message") or {}).get("content", "")
        return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])


def has_claude_credentials() -> bool:
    """True when an Anthropic API key or auth token is set in the environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def active_backend() -> str:
    """Which backend will answer: "claude" or "ollama".

    PAWPAL_AI=claude or PAWPAL_AI=ollama forces one; otherwise Claude is used
    when a key is set and the free local model is used when it isn't.
    """
    forced = os.environ.get("PAWPAL_AI", "").strip().lower()
    if forced in ("claude", "ollama"):
        return forced
    return "claude" if has_claude_credentials() else "ollama"


def backend_label() -> str:
    """A short human-readable name of the active model, for the UI and README."""
    return f"Claude ({MODEL})" if active_backend() == "claude" else f"local model ({OLLAMA_MODEL}, via Ollama)"


def default_client():
    """Build the client for the active backend."""
    return anthropic.Anthropic() if active_backend() == "claude" else OllamaClient()


@dataclass
class ParsedTask:
    """add_task: fields Claude extracted, not yet validated or attached to a pet."""

    pet_name: str
    description: str
    due_time: datetime
    duration_minutes: int
    priority: str
    frequency: str


@dataclass
class ParsedAddPet:
    """add_pet: a new pet to create, not yet validated."""

    name: str
    species: str
    breed: str = ""
    age: int = 0


@dataclass
class ParsedRemovePet:
    """remove_pet: the name of a pet to remove, not yet matched to a real pet."""

    pet_name: str


@dataclass
class Clarification:
    """Claude needs more information before an action can be made."""

    question: str


ParseResult = Union[ParsedTask, ParsedAddPet, ParsedRemovePet, Clarification]


def parse_request(
    text: str,
    pet_names: List[str],
    today: date,
    client=None,
) -> ParseResult:
    """Ask the active AI model to turn text into a task or pet action.

    Returns a ParsedTask, ParsedAddPet or ParsedRemovePet, or a Clarification
    when information is missing. Raises AIParserError for any failure (empty
    input, API error, refusal, bad output). Pass `client` to inject a fake in
    tests; otherwise Claude or the local model is chosen by active_backend().
    """
    if not text or not text.strip():
        raise AIParserError("Please type or say what you want to do.")

    if client is None:
        client = default_client()

    data = _ask_claude(client, [{"role": "user", "content": _user_message(text, pet_names, today)}])
    return _interpret(data)


def _user_message(text: str, pet_names: List[str], today: date) -> str:
    """The request sent to the model: date, upcoming weekdays, pets and the user's words."""
    # Models are unreliable at weekday arithmetic, so spell the next week out.
    upcoming = ", ".join(
        f"{day:%A} {day:%Y-%m-%d}" for day in (today + timedelta(days=n) for n in range(1, 8))
    )
    return (
        f"Today is {today:%A, %Y-%m-%d}.\n"
        f"Next 7 days: {upcoming}\n"
        f"Owner's pets: {', '.join(pet_names) if pet_names else '(none)'}\n"
        f"Request: {text.strip()}"
    )


def _interpret(data: dict) -> ParseResult:
    """Turn the model's JSON answer into a Parsed* object or a Clarification.

    Raises AIParserError if a field is missing or invalid.
    """
    if data.get("status") == "needs_clarification":
        question = (data.get("question") or "").strip()
        return Clarification(question or "Could you give more detail?")

    try:
        action = data["action"]
        # Safety net in code: if the model left a required field empty instead of
        # asking (small local models sometimes do), ask the user ourselves.
        if action == "add_task":
            pet_name = data["pet_name"].strip()
            due_raw = data["due_datetime"].strip()
            duration = int(data["duration_minutes"])
            if not pet_name:
                return Clarification("Which pet is this task for?")
            if not due_raw:
                return Clarification("What date and time should it happen?")
            if duration <= 0:
                return Clarification("How long should it take, in minutes?")
            return ParsedTask(
                pet_name=pet_name,
                description=data["description"].strip(),
                due_time=datetime.fromisoformat(due_raw),
                duration_minutes=duration,
                priority=data["priority"],
                frequency=data["frequency"],
            )
        if action == "add_pet":
            name = data["pet_name"].strip()
            if not name:
                return Clarification("What is the new pet's name?")
            return ParsedAddPet(
                name=name,
                species=data["species"],
                breed=data["breed"].strip(),
                age=max(int(data["age"]), 0),
            )
        if action == "remove_pet":
            pet_name = data["pet_name"].strip()
            if not pet_name:
                return Clarification("Which pet should I remove?")
            return ParsedRemovePet(pet_name=pet_name)
    except (KeyError, ValueError, TypeError, AttributeError):
        raise AIParserError("The AI's answer was missing or had an invalid field. Please try again.")

    raise AIParserError("The AI didn't pick a supported action. Please rephrase.")


def _ask_claude(client, messages: list) -> dict:
    """Send the conversation to the model (Claude or the local adapter) and return its
    JSON answer as a dict.

    Turns every API failure, refusal and unreadable answer into AIParserError.
    """
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": TASK_SCHEMA},
            },
            messages=messages,
        )
    except TypeError:
        # The SDK raises a plain TypeError when no credentials are configured at all.
        raise AIParserError("No Anthropic API key found. Set the ANTHROPIC_API_KEY environment variable.")
    except anthropic.AuthenticationError:
        raise AIParserError("The Anthropic API key is missing or invalid. Check ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        raise AIParserError("Too many requests right now. Wait a moment and try again.")
    except anthropic.APIConnectionError:
        raise AIParserError("Couldn't reach the Anthropic API. Check your internet connection.")
    except anthropic.APIStatusError as error:
        raise AIParserError(f"The AI service returned an error ({error.status_code}).")

    if response.stop_reason == "refusal":
        raise AIParserError("The AI declined to process that request.")
    if response.stop_reason == "max_tokens":
        raise AIParserError("The AI's answer was cut off. Try a shorter request.")

    raw = next((b.text for b in response.content if b.type == "text"), "")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise AIParserError("The AI returned an answer I couldn't read. Please try again.")
    if not isinstance(data, dict):
        raise AIParserError("The AI returned an answer I couldn't read. Please try again.")
    return data


# --- Validation -------------------------------------------------------------
# These check what Claude extracted against the owner's real data. None of them
# change the owner: the caller adds or removes things only after the user
# confirms (tasks via Scheduler.add_task so conflict rules still apply).

SPECIES = ("dog", "cat", "other")


def _find_pet(owner: Owner, name: str) -> Optional[Pet]:
    """Return the owner's pet with this name (ignoring case and spaces), or None."""
    wanted = name.strip().lower()
    return next((p for p in owner.pets if p.name.strip().lower() == wanted), None)


def _pet_list(owner: Owner) -> str:
    return ", ".join(p.name for p in owner.pets) or "none yet"


def to_task(parsed: ParsedTask, owner: Owner, task_id: int) -> Tuple[Pet, Task]:
    """Validate a ParsedTask and build the Task and the pet it belongs to.

    Does not add the task to the pet; the caller does that through
    Scheduler.add_task after the user confirms. Raises AIParserError if the
    pet is unknown, the description is empty or the duration isn't positive.
    """
    pet = _find_pet(owner, parsed.pet_name)
    if pet is None:
        raise AIParserError(
            f"'{parsed.pet_name}' isn't one of your pets (your pets: {_pet_list(owner)})."
        )
    if not parsed.description:
        raise AIParserError("The task needs a description.")
    if parsed.duration_minutes <= 0:
        raise AIParserError("The duration must be at least 1 minute.")

    try:
        task = Task(
            task_id,
            parsed.description,
            "general",
            parsed.due_time,
            parsed.duration_minutes,
            parsed.priority,   # Task.__post_init__ rejects unknown values
            parsed.frequency,
        )
    except ValueError:
        raise AIParserError("The AI chose an invalid priority or frequency. Please try again.")
    return pet, task


def to_pet(parsed: ParsedAddPet, owner: Owner, pet_id: int) -> Pet:
    """Validate a ParsedAddPet and build the Pet (not yet added to the owner).

    Raises AIParserError for an empty name, a name an existing pet already
    uses (so "remove Mochi" can never be ambiguous) or an unknown species.
    """
    if not parsed.name:
        raise AIParserError("The new pet needs a name.")
    if _find_pet(owner, parsed.name) is not None:
        # Retrying can't help: the user asked for something that conflicts with their data.
        raise AIParserError(f"You already have a pet named '{parsed.name}'.", retryable=False)
    if parsed.species not in SPECIES:
        raise AIParserError("The species must be dog, cat or other.")
    return Pet(pet_id, parsed.name, parsed.species, parsed.breed, parsed.age)


def find_pet_to_remove(parsed: ParsedRemovePet, owner: Owner) -> Tuple[Pet, int]:
    """Match a ParsedRemovePet to a real pet; return it and how many tasks go with it.

    The count lets the preview warn "this also removes N tasks". Nothing is
    removed here. Raises AIParserError if the pet is unknown.
    """
    pet = _find_pet(owner, parsed.pet_name)
    if pet is None:
        # Never retryable: the model must not "correct" which pet gets deleted.
        raise AIParserError(
            f"'{parsed.pet_name}' isn't one of your pets (your pets: {_pet_list(owner)}).",
            retryable=False,
        )
    return pet, len(pet.get_tasks())


# --- Self-correcting loop ---------------------------------------------------
# The model proposes, the code checks the proposal against the owner's real
# data, and if the check fails the error goes back to the model so it can fix
# its own answer. A person still confirms before anything changes.

MAX_ATTEMPTS = 3

RETRY_PROMPT = (
    "That answer can't be used: {error}\n"
    "Fix it and answer again. Only change what the error says is wrong. If the "
    "request doesn't contain enough information to fix it, set status to "
    '"needs_clarification" and ask the user one short question instead of '
    "guessing. Never swap in a different pet than the user asked for."
)


def next_task_id(owner: Owner) -> int:
    """An id no existing task uses (count + 1 can collide once recurring tasks exist)."""
    return max((t.task_id for t in owner.get_all_tasks()), default=0) + 1


def next_pet_id(owner: Owner) -> int:
    """An id no existing pet uses."""
    return max((p.pet_id for p in owner.pets), default=0) + 1


@dataclass
class AgentResult:
    """What parse_and_validate produced.

    `outcome` is a Clarification (the user must answer a question) or a
    proposal tuple the UI previews: ("task", pet, task), ("add_pet", pet) or
    ("remove_pet", pet, task_count). `corrections` lists, in order, the errors
    the model was asked to fix before this answer was accepted.
    """

    outcome: Union[Clarification, tuple]
    corrections: List[str]


def _validate(result: ParseResult, owner: Owner) -> Union[Clarification, tuple]:
    """Check a parsed result against the owner's data; raise AIParserError if it fails."""
    if isinstance(result, Clarification):
        return result
    if isinstance(result, ParsedTask):
        pet, task = to_task(result, owner, next_task_id(owner))
        return ("task", pet, task)
    if isinstance(result, ParsedAddPet):
        return ("add_pet", to_pet(result, owner, next_pet_id(owner)))
    pet, task_count = find_pet_to_remove(result, owner)
    return ("remove_pet", pet, task_count)


def parse_and_validate(
    text: str,
    owner: Owner,
    today: date,
    client=None,
    max_attempts: int = MAX_ATTEMPTS,
) -> AgentResult:
    """Parse a request, check it against the owner's data, and let the model fix its own mistakes.

    Loop: ask the model, validate the answer, and if validation fails send the
    error back and ask again, up to `max_attempts` answers in total. A question
    for the user ends the loop at once. Problems that retrying can't fix
    (the API is down, a refusal, adding a pet that already exists, naming an
    unknown pet to delete) are raised straight away. If the model never
    produces a valid answer, the last error is raised. Nothing is changed.
    """
    if not text or not text.strip():
        raise AIParserError("Please type or say what you want to do.")
    if client is None:
        client = default_client()

    messages = [{"role": "user", "content": _user_message(text, [p.name for p in owner.pets], today)}]
    corrections: List[str] = []
    last_error: Optional[AIParserError] = None

    for _attempt in range(max(max_attempts, 1)):
        data = _ask_claude(client, messages)  # API failures and refusals are not retried
        try:
            outcome = _validate(_interpret(data), owner)
        except AIParserError as error:
            if not error.retryable:
                raise
            last_error = error
            corrections.append(str(error))
            messages += [
                {"role": "assistant", "content": json.dumps(data)},
                {"role": "user", "content": RETRY_PROMPT.format(error=error)},
            ]
            continue
        return AgentResult(outcome, corrections)

    raise last_error
