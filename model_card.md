
# Model Card: PawPal+ AI

 The AI model in this project turns a typed or spoken request into an action (add a task, add a pet or remove a pet). Whisper turns speech into text, and a language model (Qwen2.5-7B through Ollama by default) reads the text.


## Limitations and Biases

Limitations:
Some limitations are that dates are unreliable. The model often gets "tomorrow" and "tonight" wrong by a day, and weekdays only some of the time. On my 21 dev sentences it passed 12 (57%), and 5 of the 9 failures were dates.

Along with this, the model sometimes guesses a missing detail. For "walk Mochi for 30 minutes" it picked 9:00 AM, and for "feed tomorrow at 7am" it picked Mochi. My check for empty fields can't catch a value the model made up.

The Qwen2.5-7B is much smaller than models like Claude, so its possible that it has lower accuracy. I never ran Claude against the real API, so I can't say by how much.

Another limitation is the user can give one request at a time. A request can add a task, add a pet or remove a pet, but not two of these together.

Biases:
The speech recognition is for English only. I used the English-only Whisper model. The speech recognition is usually less accurate for strong accents and background noises, which can make it less reliable for some users.



## Misuse and Prevention


I don't think it can be misused but it can defintely result in accidents. 

A misheard or vague request could remove a pet, and removing a pet also deletes its tasks. 

Also, a wrong date could make an owner miss a medicine or a vet visit. 

But the good thing is the user always has to confirm any request first, so this can help prevent any incorrect deletions or wrong scheduling. 


I prevent this by ensuring that the app can never pick a pet to delete on its own. If the request doesn't name one, the app asks. Like I said earlier, a person sees a preview and must click Confirm before anything changes, and the removal preview says how many tasks will be deleted.

Before the user is shown the preview, the request is checked against the owner's real pets and tasks, this helps the app ask the user any clarifying questions. 


## What Surprised Me While Testing

All 154 tests passed while the AI was still wrong. The tests use fake models, so they showed that my code worked and said nothing about the model. Only the evaluation on labeled sentences revealed the date problem.

The date error was consistent. "Tomorrow" came out one day too late every time, on both a fixed date and the real date. It wasn't random, and it wasn't caused by my fixed evaluation date, which is what I first thought.

The model fixed some mistakes itself. When I gave it "feed Mocha", it returned a task for Mochi without a retry. 

## Collaboration with AI

I worked with Claude through the whole project: planning the features, writing the code, writing the tests, and running the evaluation. I made the decisions about scope and cost, and I edited the README myself.

I had no API key and didn't want to spend credits, so Claude suggested running Qwen2.5-7B through Ollama. I accepted it because anyone can run the project without an account. It cost accuracy, mostly on dates, but it made the project free to build, test and share.

I asked Claude to describe each step before writing it, and I rejected the first version of the AI parser file until I understood the plan. 
