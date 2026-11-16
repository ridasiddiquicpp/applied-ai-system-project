
# Project to Expand

I decided to expand on the PawPal+ project from Module 2. Originally this project was aimed to help owners stay consistent with pet care. Owners could enter their pets' details, add tasks, times, frequency, and priority. The app would give an error or warning if any task times conflict and automatically add next task based on frequency. The app would also build the owner a schedule from this information.

# Title and Summary

Title is PawPal+ AI

PawPal AI is an extension of PawPal, which adds features for the owner to be able to simply type or speak a request and using AI it gets added to the owner's schedule. 

A request like "Add a cat named Mochi" or "add walk Mochi everyday from 8 to 9 am"

The system has a small agentic workflow where after a request is made, the app checks that request against the owner's real pets and tasks, and any error is sent back so the model can fix its own answer, with a person confirming the final result

This workflow is part of a larger reliability system, which is the advanced feature of this project. Alongside the retry loop, guardrails make a missing pet, time or duration become a question instead of a guess, and an evaluation script measures the AI on 33 labeled sentences so I can see how often it is right.



## Architecture Overview

The system diagram is saved in `diagrams/system_diagram.mmd`.

The system flows from input, to AI, to output, with a person checking the AI's work at three points.

Input component: The user speaks or types a request, or uses the original manual forms, which work without AI.

AI component: When Whisper turns speech into text, and a language model (Qwen2.5-7B through Ollama by default, or Claude if an API key is set) turns the text into an action: add a task, add a pet or remove a pet. The answer is checked against the real pets and tasks, and a failed check is sent back to the model to fix, up to three tries.

Human Check component: The user gets to edit the transcript, answers any question the AI asks, and confirms a preview before anything changes.

Output component: A confirmed task goes through the scheduler's conflict check, a pet change goes straight to the owner, and the results appear in the task list and schedule.

Testing component: Pytest checks the code with fake models, and an evaluation script measures the AI on 33 labeled sentences.


## Setup Instructions

1. Open a terminal in the project folder (`applied-ai-final`).
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install [Ollama](https://ollama.com), start it, and download the free local model:
   ```bash
   ollama pull qwen2.5:7b-instruct
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

7. Run the tests with `python -m pytest -v`. 

Notes:
The first voice recording downloads the Whisper `base.en` speech model. After that, speech recognition runs locally. The browser must allow microphone access.

To use Claude instead of the local model, set ANTHROPIC_API_KEY before running the app. Claude costs credits, and this path has not been tested against the real API.


## Sample Interactions


Input: "walk Mochi every day at 8am for 30 minutes, high priority"
What the AI Returned: Task: Mochi, daily, 8:00 AM, 30 min, high
Result: Correct.

Input:  "feed Luna at 7 for 10 minutes" 
What the AI returned: "Luna is not one of the owner's pets. Which pet is meant to be fed at 7 for 10 minutes?"
Result: Nothing is added.


How the Guardrails behave:
Real output from the local model, with Mochi and Max as the owner's pets. When the AI asks a question or rejects a request, nothing is saved.

| Input | Guardrail | Behavior | Result |
|---|---|---|---|
Input: "vet appointment for Max tomorrow at 9" 
Guardrail: Missing duration 
Behavior: The AI asks "How long should it take, in minutes?"
Result: Works

Input: "remove my pet" 
Guardrail: Never pick a pet to delete 
Behavior: The AI asks "Which pet should I remove?" 
Result: Works. Nothing is deleted.

Input: "feed Rex tomorrow at 7am for 10 minutes" 
Guardrail: Unknown pet
Behavior: The AI asks "Rex is not one of the owner's pets. Which pet would you like to feed?"
Result: Works.

Input: "add a dog named Mochi" 
Guardrail: Duplicate pet 
Behavior: Rejected: "You already have a pet named 'Mochi'." 
Result: Works.

Input: "feed Mocha tomorrow at 7am for 10 minutes"
Guardrail: Name check 
Behavior: The AI corrected the typo to Mochi and returned a task for Mochi. 
Result: Works, and the user sees Mochi in the preview.

Input: "walk Mochi for 30 minutes" (no time) 
Guardrail: Missing time 
Behvaior: The AI picked 9:00 AM and returned a task.
Result: Fails. The guardrail only catches empty fields, and the model filled this one in. 

Input: "feed tomorrow at 7am for 10 minutes" (no pet) 
Guardrail: Missing pet 
Behavior: The AI picked Mochi and returned a task.
Result: Fails. the model filled in the pet


## More Example Commands and Sample Outputs


These are real results from the local model, shown the way the parser returns them, with Max already added as the owner's. After seeing the preview, the user can select Confirm.

Example 1, add a task:
```
You say:  "Add a task to walk Mochi every day at 8 a.m. for 30 minutes and make it high priority."
Transcript: "Add a task to walk Mochi every day at 8 a.m. for 30 minutes and make it high priority."
Result:    Add task for Mochi: Walk, daily, Nov 16, 08:00 AM, 30 min, high priority
```

Example 2, add a new pet:
```
You say:  "add a cat named Luna"
Transcript: I had a cat named Luna.
Result:    Add pet Luna (cat)
```

Example 3, a missing detail (nothing is saved):
```
You say:  "Make a vet appointment for Max tomorrow at 9"
Transcript:  "Make a vet appointment for Max tomorrow at 9"
Result:   How long should it take, in minutes? (Add your answer to the text above and click Parse again.)
```

Example 3, make an appointment:
```
You say:  "Make a vet appointment for Max tomorrow at 9 for one hour"
Transcript:  "Make a bit appointment for Max tomorrow at 9 for one hour."
Result:  Add task for Max: Vet appointment, once, Nov 17, 09:00 AM, 60 min, high priority
```

Example 4, delete a pet:
```
You say:  "Delete pet Max"
Transcipt: "Delete Pet Max:
Preview:   Remove Max? This also removes its 1 task(s) and cannot be undone.
```


## Design Decisions

Why I built it this way:
I used a free local model, Qwen2.5-7B, run through Ollama, so no API key is needed. 


When the owner makes a request, if the pet, time, or duration is empty, then the app asks a question instead of guessing. Which I think it better than guessing and adding the wrong task information.

The code checks the model's answer against the real pets and tasks. If the check fails, the error message is sent back to the model so it can correct its answer, up to three tries. This ensures small mistakes (particularly transcription mistakes) like the pet's name being spelled differently can be easily fixed. 

The owner confirms every change, because speech recognition and models can make mistakes

If a task is added for the same pet at the same time then it is rejected. But if a task for another pet is added at the same time then it results in a warning. This is because it is assumed that an owner cannot do two things with the same pet at the same time, but can possibly do two things with different pets at the same time. 

This technically isn't perfect especially if an owner adds two tasks at different locations, then the owner of course cannot be in both places

The manual form is still there and can be used. So the app can work without a model or microphone.



Trade-offs:
I used a free model, but the user has to install Ollama and download the model, whereas using a Claude API would just use an API key.
Also this model isn;t very accurate when it comes to dates; it made many mistakes with the date tests. So it is possible that if the Claude API was used it would be more accurate.

The AI cannot suggest fixes, such as moving a task to avoid a conflict. Also, asking instead of guessing means the user sometimes answers something the model could have guessed.

The user has to make only one request, like add a pet, or schedule a task, it cant do both on the same request


## Testing Summary

What worked:
154 automated tests pass. They use fake models, so they need no Ollama, key or network. They cover the scheduler, the parser, the retry loop, speech-to-text and the evaluation scoring.

An evaluation script scored the local model on 21 labeled sentences (the dev set): 12 passed (57%), with no wrong deletions. Running it a second time gave identical results. It handled add pet, remove pet and unknown pet requests well.

What didn't work:
Dates were the main problem for this model, causing 5 of the 9 failures. "Tomorrow" and "tonight" came out one day too late many times, and weekdays were right only some of the time. 

In 3 cases the model guessed a missing detail instead of asking, and in 1 case it missed a name.

What I learned:
Fake-model tests show that my code works, but only a labeled evaluation shows how well the AI works. I needed both for proper testing.

Adding stricter instructions and a code check for empty fields stopped the model from making up durations and picking a pet to delete.


## Reflection

This project taught me that AI is useful for things like turning a spoken sentence into fields, but it can't akways be trusted on its own. The model was good at understanding requests like "add a cat named Luna", but it got dates wrong and sometimes guessed a missing detail. I learned to put checks around the model, like validate its answer against real data, ask a question when something is missing, and have a person confirm every change.

This taught me that sometimes human cofirmation is necessary and AI cannot always be trusted to fully make decision by itself

The full responsible-AI reflection is in `model_card.md`.



