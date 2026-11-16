"""Measure how well the AI parser performs on labeled sentences.

Run from the project folder:

    python -m evals.run_eval                 # dev cases, with and without the retry loop
    python -m evals.run_eval --split test    # hold-out cases: use for the final score only
    python -m evals.run_eval --loop on       # only the self-correcting loop
    python -m evals.run_eval --split all --loop both

It uses whichever model ai_parser picks (the free local model unless
ANTHROPIC_API_KEY is set). Using Claude costs credits, so that needs
--allow-claude. Results are printed and saved to evals/results.md and
evals/results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import ai_parser
from ai_parser import AIParserError, Clarification, parse_and_validate
from pawpal_system import Owner, Pet, Task

HERE = Path(__file__).parent
ACTIONS = ("task", "add_pet", "remove_pet")  # outcomes that would change data if confirmed


# ---------------------------------------------------------------- setup and summaries

def build_owner(setup: dict) -> Owner:
    """A fresh owner for every case, so one case can't affect another."""
    owner = Owner(1, "Eval")
    for index, spec in enumerate(setup["pets"], 1):
        pet = Pet(index, spec["name"], spec["species"])
        owner.add_pet(pet)
        for n in range(spec.get("tasks", 0)):
            pet.add_task(Task(len(owner.get_all_tasks()) + 1, f"existing task {n + 1}", "general",
                              datetime(2026, 12, 1, 8 + n), 10))
    return owner


def summarize_outcome(outcome) -> dict:
    """Turn the parser's result into a flat dict that cases can be compared against."""
    if isinstance(outcome, Clarification):
        return {"kind": "question", "text": outcome.question}
    kind = outcome[0]
    if kind == "task":
        _, pet, task = outcome
        return {"kind": "task", "pet": pet.name, "due": task.due_time.isoformat(timespec="minutes"),
                "duration": task.duration_minutes, "priority": str(task.priority),
                "frequency": str(task.frequency)}
    if kind == "add_pet":
        return {"kind": "add_pet", "name": outcome[1].name, "species": outcome[1].species}
    return {"kind": "remove_pet", "pet": outcome[1].name}


def matches(summary: dict, expected: dict) -> bool:
    """True when every field listed in `expected` equals the one in `summary`."""
    for key, want in expected.items():
        got = summary.get(key)
        if isinstance(want, str) and isinstance(got, str):
            if want.strip().lower() != got.strip().lower():
                return False
        elif got != want:
            return False
    return True


def passes(summary: dict, expect: List[dict]) -> bool:
    """A case passes when the result matches any acceptable outcome."""
    return any(matches(summary, alternative) for alternative in expect)


# ---------------------------------------------------------------- running

def run_case(case: dict, setup: dict, max_attempts: int, client=None) -> dict:
    """Run one sentence through the parser and score it."""
    owner = build_owner(setup)
    started = time.time()
    corrections: List[str] = []
    try:
        result = parse_and_validate(case["text"], owner, date.fromisoformat(setup["today"]),
                                    client, max_attempts=max_attempts)
        summary = summarize_outcome(result.outcome)
        corrections = result.corrections
    except AIParserError as error:
        summary = {"kind": "error", "text": str(error)}
    return {
        "id": case["id"], "text": case["text"], "expect": case["expect"],
        "expected_kind": case["expect"][0]["kind"], "got": summary,
        "passed": passes(summary, case["expect"]), "corrections": corrections,
        "seconds": round(time.time() - started, 2),
    }


def run_all(cases: List[dict], setup: dict, max_attempts: int,
            client=None, progress: Optional[Callable[[dict], None]] = None) -> List[dict]:
    results = []
    for case in cases:
        results.append(run_case(case, setup, max_attempts, client))
        if progress:
            progress(results[-1])
    return results


# ---------------------------------------------------------------- metrics

def summarize(results: List[dict]) -> dict:
    """Counts and rates for one run. Rates are None when there is nothing to divide by."""
    def rate(part, whole):
        return None if not whole else part / whole

    by_kind = defaultdict(lambda: [0, 0])
    for r in results:
        by_kind[r["expected_kind"]][1] += 1
        by_kind[r["expected_kind"]][0] += r["passed"]

    wrong_actions = [r for r in results if not r["passed"] and r["got"]["kind"] in ACTIONS]
    return {
        "total": len(results),
        "passed": sum(r["passed"] for r in results),
        "accuracy": rate(sum(r["passed"] for r in results), len(results)),
        "by_kind": {k: {"passed": p, "total": t} for k, (p, t) in sorted(by_kind.items())},
        # Acting on a request that should have been a question or rejection, or on the wrong thing.
        "wrong_actions": [r["id"] for r in wrong_actions],
        # The worst kind: proposing to delete the wrong pet.
        "wrong_deletions": [r["id"] for r in wrong_actions if r["got"]["kind"] == "remove_pet"],
        "used_retries": sum(1 for r in results if r["corrections"]),
        "retries_that_ended_in_pass": sum(1 for r in results if r["corrections"] and r["passed"]),
        "avg_seconds": round(sum(r["seconds"] for r in results) / len(results), 2) if results else None,
    }


# ---------------------------------------------------------------- reporting

def pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def describe(summary: dict) -> str:
    if summary["kind"] in ("question", "error"):
        text = summary.get("text")  # expected outcomes carry no text, only actual results do
        return summary["kind"] if not text else f"{summary['kind']}: {text[:90]}"
    return json.dumps({k: v for k, v in summary.items() if k != "kind"}) + f" ({summary['kind']})"


def report_markdown(runs: Dict[str, List[dict]], meta: dict) -> str:
    lines = [
        "# AI parser evaluation", "",
        f"- Model: **{meta['model']}**",
        f"- Date: {meta['when']}",
        f"- Cases: {meta['split']} split ({meta['cases']} sentences)",
        f"- Fixed setup: today is {meta['today']}; pets {meta['pets']}",
        "- A case passes when the result matches any acceptable outcome listed for it.", "",
        "## Results", "",
        "| Mode | Passed | Accuracy | Wrong actions | Wrong deletions | Used retries | Avg seconds |",
        "|---|---|---|---|---|---|---|",
    ]
    stats = {name: summarize(results) for name, results in runs.items()}
    for name, s in stats.items():
        lines.append(f"| {name} | {s['passed']} of {s['total']} | {pct(s['accuracy'])} | "
                     f"{len(s['wrong_actions'])} | {len(s['wrong_deletions'])} | {s['used_retries']} | {s['avg_seconds']} |")
    lines += ["", "## By kind of request (passed / total)", ""]
    kinds = sorted({k for s in stats.values() for k in s["by_kind"]})
    lines += ["| Expected | " + " | ".join(stats) + " |", "|---|" + "---|" * len(stats)]
    for kind in kinds:
        cells = [f"{s['by_kind'][kind]['passed']} / {s['by_kind'][kind]['total']}" if kind in s["by_kind"] else "-"
                 for s in stats.values()]
        lines.append(f"| {kind} | " + " | ".join(cells) + " |")
    lines += ["", '"Wrong actions" counts results that would change data if confirmed but did not match '
                  "any acceptable outcome. A wrong deletion is the most serious kind.", ""]
    for name, results in runs.items():
        failures = [r for r in results if not r["passed"]]
        lines += [f"## Failures: {name} ({len(failures)})", ""]
        if not failures:
            lines += ["None.", ""]
        for r in failures:
            lines += [f"- **{r['id']}** \"{r['text']}\"",
                      f"  - expected: {' or '.join(describe(e) for e in r['expect'])}",
                      f"  - got: {describe(r['got'])}"]
            if r["corrections"]:
                lines.append(f"  - corrections before this answer: {len(r['corrections'])}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- command line

def ollama_is_running() -> bool:
    try:
        urllib.request.urlopen(f"{ai_parser.OLLAMA_URL}/api/tags", timeout=5).close()
        return True
    except Exception:
        return False


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--loop", choices=["on", "off", "both"], default="both",
                        help="on = self-correcting retry loop, off = a single attempt")
    parser.add_argument("--allow-claude", action="store_true", help="allow spending Claude API credits")
    args = parser.parse_args(argv)

    if ai_parser.active_backend() == "claude" and not args.allow_claude:
        print("The active model is Claude, which costs credits. Re-run with --allow-claude to continue,\n"
              "or unset ANTHROPIC_API_KEY (or set PAWPAL_AI=ollama) to use the free local model.")
        return 2
    if ai_parser.active_backend() == "ollama" and not ollama_is_running():
        print("Ollama isn't running. Start the Ollama app and try again.")
        return 2

    data = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    setup = data["setup"]
    cases = [c for c in data["cases"] if args.split == "all" or c["split"] == args.split]
    modes = {"on": {"loop on": ai_parser.MAX_ATTEMPTS}, "off": {"loop off": 1},
             "both": {"loop off": 1, "loop on": ai_parser.MAX_ATTEMPTS}}[args.loop]

    print(f"Model: {ai_parser.backend_label()} | {len(cases)} cases ({args.split}) | modes: {', '.join(modes)}")
    runs = {}
    for name, attempts in modes.items():
        print(f"\n== {name} ==")
        runs[name] = run_all(cases, setup, attempts, progress=lambda r: print(
            f"  {'PASS' if r['passed'] else 'FAIL'} {r['id']:4} {r['seconds']:5.1f}s  {r['text'][:60]}"))

    meta = {"model": ai_parser.backend_label(), "when": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "split": args.split, "cases": len(cases), "today": setup["today"],
            "pets": ", ".join(p["name"] for p in setup["pets"])}
    markdown = report_markdown(runs, meta)
    (HERE / "results.md").write_text(markdown, encoding="utf-8")
    (HERE / "results.json").write_text(json.dumps({"meta": meta, "runs": runs}, indent=2), encoding="utf-8")
    print("\n" + markdown)
    print("Saved evals/results.md and evals/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
