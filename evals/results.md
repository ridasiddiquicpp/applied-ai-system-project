# AI parser evaluation

- Model: **local model (qwen2.5:7b-instruct, via Ollama)**
- Date: 2026-11-15 19:32
- Cases: dev split (21 sentences)
- Fixed setup: today is 2026-11-14; pets Mochi, Max
- A case passes when the result matches any acceptable outcome listed for it.

## Results

| Mode | Passed | Accuracy | Wrong actions | Wrong deletions | Used retries | Avg seconds |
|---|---|---|---|---|---|---|
| loop off | 12 of 21 | 57% | 8 | 0 | 0 | 3.95 |
| loop on | 12 of 21 | 57% | 8 | 0 | 0 | 3.97 |

## By kind of request (passed / total)

| Expected | loop off | loop on |
|---|---|---|
| add_pet | 2 / 3 | 2 / 3 |
| error | 1 / 1 | 1 / 1 |
| question | 3 / 6 | 3 / 6 |
| remove_pet | 2 / 2 | 2 / 2 |
| task | 4 / 9 | 4 / 9 |

"Wrong actions" counts results that would change data if confirmed but did not match any acceptable outcome. A wrong deletion is the most serious kind.

## Failures: loop off (9)

- **t01** "starting tomorrow, walk Mochi every day at 8am for 30 minutes, high priority"
  - expected: {"pet": "Mochi", "due": "2026-11-15T08:00", "duration": 30, "priority": "high", "frequency": "daily"} (task)
  - got: {"pet": "Mochi", "due": "2026-11-16T08:00", "duration": 30, "priority": "high", "frequency": "daily"} (task)
- **t02** "feed Max tomorrow at 7:30am for 10 minutes"
  - expected: {"pet": "Max", "due": "2026-11-15T07:30", "duration": 10} (task)
  - got: {"pet": "Max", "due": "2026-11-16T07:30", "duration": 10, "priority": "medium", "frequency": "once"} (task)
- **t05** "groom Mochi on Wednesday at 10am for 45 minutes, low priority"
  - expected: {"pet": "Mochi", "due": "2026-11-18T10:00", "duration": 45, "priority": "low"} (task)
  - got: {"pet": "Mochi", "due": "2026-11-17T10:00", "duration": 45, "priority": "low", "frequency": "once"} (task)
- **t06** "play with Max tonight at 7pm for 20 minutes"
  - expected: {"pet": "Max", "due": "2026-11-14T19:00", "duration": 20} (task)
  - got: {"pet": "Max", "due": "2026-11-19T19:00", "duration": 20, "priority": "medium", "frequency": "once"} (task)
- **t07** "uh walk mochi tomorrow at eight in the morning for like thirty minutes"
  - expected: {"pet": "Mochi", "due": "2026-11-15T08:00", "duration": 30} (task)
  - got: {"pet": "Mochi", "due": "2026-11-16T08:00", "duration": 30, "priority": "medium", "frequency": "once"} (task)
- **p02** "I got a new dog, his name is Biscuit"
  - expected: {"name": "Biscuit", "species": "dog"} (add_pet)
  - got: question: What is the new pet's name?
- **p04** "add a pet named Pixel"
  - expected: question
  - got: {"name": "Pixel", "species": "other"} (add_pet)
- **q02** "walk Mochi for 30 minutes"
  - expected: question
  - got: {"pet": "Mochi", "due": "2026-11-15T09:00", "duration": 30, "priority": "medium", "frequency": "once"} (task)
- **q03** "feed tomorrow at 7am for 10 minutes"
  - expected: question
  - got: {"pet": "Mochi", "due": "2026-11-15T07:00", "duration": 10, "priority": "medium", "frequency": "once"} (task)

## Failures: loop on (9)

- **t01** "starting tomorrow, walk Mochi every day at 8am for 30 minutes, high priority"
  - expected: {"pet": "Mochi", "due": "2026-11-15T08:00", "duration": 30, "priority": "high", "frequency": "daily"} (task)
  - got: {"pet": "Mochi", "due": "2026-11-16T08:00", "duration": 30, "priority": "high", "frequency": "daily"} (task)
- **t02** "feed Max tomorrow at 7:30am for 10 minutes"
  - expected: {"pet": "Max", "due": "2026-11-15T07:30", "duration": 10} (task)
  - got: {"pet": "Max", "due": "2026-11-16T07:30", "duration": 10, "priority": "medium", "frequency": "once"} (task)
- **t05** "groom Mochi on Wednesday at 10am for 45 minutes, low priority"
  - expected: {"pet": "Mochi", "due": "2026-11-18T10:00", "duration": 45, "priority": "low"} (task)
  - got: {"pet": "Mochi", "due": "2026-11-17T10:00", "duration": 45, "priority": "low", "frequency": "once"} (task)
- **t06** "play with Max tonight at 7pm for 20 minutes"
  - expected: {"pet": "Max", "due": "2026-11-14T19:00", "duration": 20} (task)
  - got: {"pet": "Max", "due": "2026-11-19T19:00", "duration": 20, "priority": "medium", "frequency": "once"} (task)
- **t07** "uh walk mochi tomorrow at eight in the morning for like thirty minutes"
  - expected: {"pet": "Mochi", "due": "2026-11-15T08:00", "duration": 30} (task)
  - got: {"pet": "Mochi", "due": "2026-11-16T08:00", "duration": 30, "priority": "medium", "frequency": "once"} (task)
- **p02** "I got a new dog, his name is Biscuit"
  - expected: {"name": "Biscuit", "species": "dog"} (add_pet)
  - got: question: What is the new pet's name?
- **p04** "add a pet named Pixel"
  - expected: question
  - got: {"name": "Pixel", "species": "other"} (add_pet)
- **q02** "walk Mochi for 30 minutes"
  - expected: question
  - got: {"pet": "Mochi", "due": "2026-11-15T09:00", "duration": 30, "priority": "medium", "frequency": "once"} (task)
- **q03** "feed tomorrow at 7am for 10 minutes"
  - expected: question
  - got: {"pet": "Mochi", "due": "2026-11-15T07:00", "duration": 10, "priority": "medium", "frequency": "once"} (task)
