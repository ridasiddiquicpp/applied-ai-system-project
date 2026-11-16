"""Tests for the evaluation harness itself (evals/run_eval.py) and for the labels
in evals/cases.json. An eval you can't trust is worse than none, so these check
the scoring and the labels. They use a fake model: no key, Ollama or network."""
import json
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_parser import Clarification
from evals import run_eval as ev
from pawpal_system import Pet

DATA = json.loads((Path(ev.HERE) / "cases.json").read_text(encoding="utf-8"))
SETUP = DATA["setup"]
CASES = DATA["cases"]
KINDS = {"task", "add_pet", "remove_pet", "question", "error"}
TODAY = date.fromisoformat(SETUP["today"])
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


# ---------------------------------------------------------------- the labels

def test_case_ids_are_unique_and_splits_valid():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))
    assert {c["split"] for c in CASES} == {"dev", "test"}
    assert sum(c["split"] == "test" for c in CASES) >= 10  # a real hold-out set


def test_every_case_has_valid_expected_outcomes():
    for case in CASES:
        assert case["text"].strip() and case["expect"], case["id"]
        for outcome in case["expect"]:
            assert outcome["kind"] in KINDS, case["id"]
            if outcome["kind"] == "task":
                datetime.fromisoformat(outcome["due"])  # must be a real date
                assert outcome["duration"] > 0, case["id"]


def test_both_splits_cover_every_kind_of_request():
    for split in ("dev", "test"):
        kinds = {c["expect"][0]["kind"] for c in CASES if c["split"] == split}
        assert kinds == KINDS, split


def test_expected_dates_agree_with_the_words_in_the_sentence():
    """A wrong label would unfairly fail the model, so check weekdays, tomorrow and tonight."""
    checked = 0
    for case in CASES:
        task = next((o for o in case["expect"] if o["kind"] == "task"), None)
        if task is None:
            continue
        due = datetime.fromisoformat(task["due"])
        text = case["text"].lower()
        assert due.date() >= TODAY, case["id"]
        for number, name in enumerate(WEEKDAYS):
            if name in text:
                assert due.weekday() == number, f"{case['id']}: expected {task['due']} is not a {name}"
                checked += 1
        if "tomorrow" in text:
            assert (due.date() - TODAY).days == 1, case["id"]
            checked += 1
        if "tonight" in text:
            assert due.date() == TODAY, case["id"]
            checked += 1
    assert checked >= 10  # the check really looked at something


def test_setup_describes_two_pets_one_with_a_task():
    owner = ev.build_owner(SETUP)

    assert [p.name for p in owner.pets] == ["Mochi", "Max"]
    assert [len(p.get_tasks()) for p in owner.pets] == [1, 0]
    assert ev.build_owner(SETUP) is not owner  # fresh each time


# ---------------------------------------------------------------- scoring

TASK = {"kind": "task", "pet": "Mochi", "due": "2026-11-15T08:00", "duration": 30,
        "priority": "medium", "frequency": "once"}


def test_matching_ignores_case_and_unlisted_fields():
    assert ev.matches(TASK, {"kind": "task", "pet": "mochi"})
    assert ev.matches(TASK, {"kind": "task", "pet": " MOCHI ", "duration": 30})


@pytest.mark.parametrize("expected", [
    {"kind": "task", "pet": "Max"},
    {"kind": "task", "duration": 31},
    {"kind": "task", "due": "2026-11-16T08:00"},
    {"kind": "add_pet"},
])
def test_matching_rejects_differences(expected):
    assert not ev.matches(TASK, expected)


def test_a_case_passes_if_any_alternative_matches():
    expect = [{"kind": "question"}, {"kind": "task", "pet": "Mochi"}]

    assert ev.passes({"kind": "question", "text": "?"}, expect)
    assert ev.passes(TASK, expect)
    assert not ev.passes({"kind": "error", "text": "x"}, expect)


def test_outcomes_are_summarized_for_comparison():
    owner = ev.build_owner(SETUP)
    mochi = owner.pets[0]
    task = mochi.tasks[0]

    assert ev.summarize_outcome(Clarification("How long?")) == {"kind": "question", "text": "How long?"}
    assert ev.summarize_outcome(("add_pet", Pet(9, "Luna", "cat"))) == {"kind": "add_pet", "name": "Luna", "species": "cat"}
    assert ev.summarize_outcome(("remove_pet", mochi, 1)) == {"kind": "remove_pet", "pet": "Mochi"}
    summary = ev.summarize_outcome(("task", mochi, task))
    assert summary["kind"] == "task" and summary["pet"] == "Mochi" and summary["duration"] == 10
    assert summary["due"] == "2026-12-01T08:00" and summary["priority"] == "medium"


# ---------------------------------------------------------------- running a case

class ScriptedClient:
    def __init__(self, *answers):
        self.answers, self.messages, self.calls = list(answers), self, 0

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text=json.dumps(self.answers.pop(0)))])


BASE = {"action": "add_task", "status": "ok", "question": "", "pet_name": "", "description": "",
        "due_datetime": "", "duration_minutes": 0, "priority": "medium", "frequency": "once",
        "species": "dog", "breed": "", "age": 0}
GOOD = {**BASE, "pet_name": "Mochi", "description": "Walk", "due_datetime": "2026-11-15T08:00", "duration_minutes": 30}
CASE = {"id": "c1", "text": "walk Mochi tomorrow at 8 for 30 minutes",
        "expect": [{"kind": "task", "pet": "Mochi", "due": "2026-11-15T08:00", "duration": 30}]}


def test_a_correct_answer_passes():
    result = ev.run_case(CASE, SETUP, max_attempts=3, client=ScriptedClient(GOOD))

    assert result["passed"] and result["corrections"] == [] and result["expected_kind"] == "task"


def test_a_wrong_answer_fails():
    result = ev.run_case(CASE, SETUP, max_attempts=3, client=ScriptedClient({**GOOD, "duration_minutes": 45}))

    assert not result["passed"] and result["got"]["duration"] == 45


def test_a_rejected_request_is_recorded_as_an_error():
    case = {"id": "c2", "text": "add a dog named Mochi", "expect": [{"kind": "error"}]}
    answer = {**BASE, "action": "add_pet", "pet_name": "Mochi", "species": "dog"}

    result = ev.run_case(case, SETUP, max_attempts=3, client=ScriptedClient(answer))

    assert result["passed"] and result["got"]["kind"] == "error"


def test_the_loop_rescues_a_case_that_fails_without_it():
    wrong_then_right = lambda: ScriptedClient({**GOOD, "pet_name": "Mocha"}, GOOD)

    without_loop = ev.run_case(CASE, SETUP, max_attempts=1, client=wrong_then_right())
    with_loop = ev.run_case(CASE, SETUP, max_attempts=3, client=wrong_then_right())

    assert not without_loop["passed"] and without_loop["got"]["kind"] == "error"
    assert with_loop["passed"] and len(with_loop["corrections"]) == 1


# ---------------------------------------------------------------- metrics

def result(id, expected_kind, got_kind, passed, corrections=(), seconds=1.0):
    return {"id": id, "expected_kind": expected_kind, "got": {"kind": got_kind}, "passed": passed,
            "corrections": list(corrections), "seconds": seconds}


def test_metrics_count_accuracy_wrong_actions_and_retries():
    results = [
        result("a", "task", "task", True),
        result("b", "task", "task", False),                                    # wrong action
        result("c", "question", "question", True),
        result("d", "question", "remove_pet", False),                          # wrong deletion
        result("e", "task", "task", True, corrections=["x"], seconds=3.0),     # rescued by a retry
        result("f", "add_pet", "error", False, corrections=["y"]),             # retried, still failed
    ]

    stats = ev.summarize(results)

    assert (stats["total"], stats["passed"]) == (6, 3)
    assert stats["accuracy"] == 0.5
    assert stats["wrong_actions"] == ["b", "d"] and stats["wrong_deletions"] == ["d"]
    assert stats["used_retries"] == 2 and stats["retries_that_ended_in_pass"] == 1
    assert stats["by_kind"]["task"] == {"passed": 2, "total": 3}
    assert stats["by_kind"]["question"] == {"passed": 1, "total": 2}
    assert stats["avg_seconds"] == 1.33


def test_metrics_handle_no_results():
    stats = ev.summarize([])

    assert stats["total"] == 0 and stats["accuracy"] is None and stats["avg_seconds"] is None


# ---------------------------------------------------------------- report and command line

def test_report_contains_the_numbers_and_the_failures():
    runs = {"loop off": [result("a", "task", "task", True), result("b", "task", "error", False)]}
    runs["loop off"][1].update(text="walk someone", expect=[{"kind": "task", "pet": "Mochi"}],
                               got={"kind": "error", "text": "no pet"})
    meta = {"model": "test model", "when": "now", "split": "dev", "cases": 2, "today": "2026-11-14", "pets": "Mochi"}

    report = ev.report_markdown(runs, meta)

    assert "test model" in report and "| loop off | 1 of 2 | 50% |" in report
    assert "**b** \"walk someone\"" in report and "no pet" in report


def test_report_can_list_a_failure_whose_expected_outcome_has_no_text():
    """Expected outcomes like {"kind": "question"} have no text; this once crashed the report."""
    failed = result("q", "question", "task", False)
    failed.update(text="vet tomorrow", expect=[{"kind": "question"}, {"kind": "error"}],
                  got={"kind": "task", "pet": "Mochi", "duration": 60})
    meta = {"model": "m", "when": "now", "split": "dev", "cases": 1, "today": "2026-11-14", "pets": "Mochi"}

    report = ev.report_markdown({"loop on": [failed]}, meta)

    assert "expected: question or error" in report
    assert '"pet": "Mochi"' in report


def test_describe_shows_text_only_when_there_is_some():
    assert ev.describe({"kind": "question"}) == "question"
    assert ev.describe({"kind": "error", "text": "no pet"}) == "error: no pet"
    assert ev.describe({"kind": "add_pet", "name": "Luna"}) == '{"name": "Luna"} (add_pet)'


def test_the_script_refuses_to_spend_claude_credits_without_permission(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    assert ev.main([]) == 2
    assert "--allow-claude" in capsys.readouterr().out
