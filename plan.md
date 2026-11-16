# PawPal+ AI Feature Plan

Add natural-language entry, by typing or by voice, that turns a sentence like *"walk Mochi every day at 8am for 30 minutes, high priority"* into a `Task`. The same input can also add or remove pets ("add a cat named Luna", "remove Max").

## Decisions

- **Feature:** natural-language entry first. Actions: add task, add pet, remove pet. Deleting a single task or completing a task by voice is out of scope for now.
- **Provider:** Anthropic Claude API (`ANTHROPIC_API_KEY` from the environment) when a key is set. With no key, a free local model (`qwen2.5:7b-instruct` through Ollama) is used instead. `PAWPAL_AI=claude` or `PAWPAL_AI=ollama` forces one.
- **Model:** `claude-haiku-5-5`, kept in one constant in `ai_parser.py`. It is cheap and enough for extracting fields from one sentence.
- **Voice:** `st.audio_input` for recording, local `faster-whisper` (small model) for transcription. Claude only receives text.
- **Input:** voice and typed text sit side by side. A transcript fills the same editable text box.
- **Cost control:** no Claude call until the user clicks **Parse**, so a bad transcript can be fixed first.
- **Safety:** nothing is added until the user clicks **Confirm**, and it goes through `Scheduler.add_task`, so conflict rules still apply.

## Flow

```
mic -> st.audio_input -> faster-whisper -> editable transcript
typed text ----------------------------> editable transcript
editable transcript -> [Parse] -> Claude (structured output) -> preview -> [Confirm] -> Scheduler.add_task
```

## Steps

- [x] **1. Dependencies and config** (done; you still need to set the API key, see below)
  - [x] Add `anthropic` and `faster-whisper` to `requirements.txt` (only `anthropic` is installed in `.venv`; `faster-whisper` waits for step 4)
  - [x] Raise Streamlit to `>=1.39` (needed for `st.audio_input`)
  - [x] Read `ANTHROPIC_API_KEY` from the environment (the SDK does this by default; the key is not set in this shell yet)
- [x] **2. `ai_parser.py`** (written; checked only with a fake client, never against the real API)
  - [x] `parse_request(text, pet_names, today) -> ParsedTask | ParsedAddPet | ParsedRemovePet | Clarification`
  - [x] Three actions: `add_task`, `add_pet`, `remove_pet` (the model picks one per request)
  - [x] JSON schema covers task fields (pet, description, due datetime, duration, priority, frequency) and pet fields (name, species, breed, age)
  - [x] Return a clarification question when input is ambiguous or missing a field (never guess pet, time, duration, species)
  - [x] Handle refusals, rate limits and connection errors without crashing
- [x] **3. Validation** (in `ai_parser.py`: `to_task`, `to_pet`, `find_pet_to_remove`; hand-checked, pytest tests come in step 6)
  - [x] Convert `ParsedTask` to `Task` (existing enums reject bad priority/frequency)
  - [x] Reject unknown pets and non-positive durations
  - [x] Add pet: reject empty or duplicate names; build the `Pet`
  - [x] Remove pet: reject unknown pets; report how many tasks would be removed
  - [x] Do not change anything until the user confirms (these functions never modify the owner; the UI will, after Confirm)
- [x] **4. `transcribe.py`** (written; real microphone transcription confirmed working in the app after pinning `av<17`)
  - [x] `transcribe(audio_bytes, pet_names) -> str` using the `base.en` Whisper model (`faster-whisper` installed in `.venv`)
  - [x] Load the model once (`functools.lru_cache`, so the file doesn't depend on Streamlit)
  - [x] Pass pet names as an initial prompt to improve name recognition
  - [x] Clear error if the model is not installed, can't download, or the recording is empty or silent
- [x] **5. `app.py` UI** (written; flows checked headlessly with a fake parser. The mic widget, real transcription and real Claude calls are untested)
  - [x] "Quick add" section: `st.audio_input` plus text box
  - [x] **Parse** button, preview, then **Confirm** / **Cancel** (a different preview per action; removing a pet warns "this also removes N tasks")
  - [x] Confirm calls `Scheduler.add_task` (rejections and warnings still show)
  - [x] Keep the manual form as a fallback (no API key, no mic)
  - [x] Fix the `NameError` when "Add task" is clicked with no pets
  - [x] New task and pet ids use max + 1 instead of count + 1
  - [x] Fixed: `st.table` crashed on this machine because Windows Application Control blocks pyarrow (needed by pandas). Both tables now use a plain markdown `show_table` helper instead.
- [x] **6. Tests** (82 pass in about 1 second with no API key, model or network)
  - [x] Validation and conversion tests (tasks, add pet, remove pet including a pet with tasks) in `tests/test_ai_parser.py`
  - [x] Parser error handling with a mocked client (refusal, bad JSON, rate limit, no key, bad date, unknown action)
  - [x] Transcriber tests with a fake Whisper model in `tests/test_transcribe.py`
  - [x] Existing 29 tests still pass
  - [ ] Not covered: the Streamlit UI flow (checked by hand with a headless script, not saved as a test)
- [ ] **7. Docs**
  - [ ] README: setup, API key, mic permission, voice usage
  - [ ] Update `CLAUDE.md` with the new files
  - [ ] Update `diagrams/uml.mmd` if classes change

- [x] **8. Free local model backend** (no key needed; tried with real sentences headlessly, Claude itself still untried)
  - [x] Install Ollama and pull `qwen2.5:7b-instruct` (about 4.7 GB)
  - [x] `OllamaClient` adapter with the same `messages.create` interface as the Claude client
  - [x] Automatic choice: Claude when `ANTHROPIC_API_KEY` is set, otherwise the local model
  - [x] The app shows which model is in use
  - [x] Prompt tightened (never infer duration, ask which pet when several) plus a code safety net that turns an empty pet, time or duration into a question
  - [x] Next 7 days listed in the request to help with weekdays
  - [x] 79 more tests with fakes (108 total)
  - [ ] Known weakness: the local model can still get a weekday wrong (said Nov 17 for "Friday"; the preview shows the date so you can catch it)

- [x] **9. Self-correcting loop (agentic feature, option A)** (`parse_and_validate` in `ai_parser.py`)
  - [x] Model proposes, code validates against the owner's real data, and a validation error is sent back so the model can fix its own answer (up to 3 answers in total)
  - [x] A question for the user ends the loop at once
  - [x] Never retried: API failures and refusals, adding a pet that already exists, and naming an unknown pet to delete (the model must not "correct" which pet is removed)
  - [x] The app uses it and shows "The AI corrected itself after: ..." when a retry happened
  - [x] 24 more tests with a scripted fake model (132 total)
  - [x] Tried with the real local model by planting wrong first answers (misspelled pet, invalid priority, unreadable date, nonexistent pet): all four were corrected on the second call, and the nonexistent pet became a question instead of a swap
  - [ ] Not yet shown: on real sentences the local model rarely makes a validation mistake in the first place (it fixed "Mocha" and "Maxx" itself), so the loop's benefit still needs the evaluation script to measure

## Build order

1. Parser and validation, with tests
2. Transcriber
3. UI

## Out of scope for now

- Multiple tasks in one sentence
- AI schedule explainer
- Pet care advisor chat
- Auto-planner

The module layout should leave room to add these later.

## Open items

- Real mic and Claude calls need your `ANTHROPIC_API_KEY` and a browser mic, so they must be tried by hand after the code is written.
- Optional: a small set of sample sentences with expected parses to check parser quality.
