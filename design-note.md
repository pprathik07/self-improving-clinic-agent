# Design Note

**1. Key design choices.** Safety lives in code, not only in the prompt. Tools never accept
a `patient_id` from the model; a write needs a verified session, a confirmed action,
ownership of the appointment and an open slot. An injected instruction can change what the
model says, but not what the tools permit. An explicit state machine (greeting, identify,
verified, intent, slot selection, confirm, execute, done, escalated) limits which tools
exist in each state. Policy is a YAML file, so the improvement loop has exactly one thing it
is allowed to change.

**2. How the eval works and where it is blind.** 15 scenarios (9 train, 6 heldout), k runs
each, scored in three layers in order of trust: database and session state, tool-call
trace, then an LLM judge on a different model. State is ground truth because the judge sees
only text and can be fooled by "you're booked!" when no write happened. Blind spots: the
judge cannot see tool arguments or DB state, the simulator is an LLM and can drift, k is
small, and the data is synthetic. Judge agreement with my hand labels: not measured.

**3. How the improvement loop works.** The reflector sees train failures only (transcript,
tool calls, failing checks, current policy); a leak check runs on the final prompt so
heldout scenarios cannot reach it. It returns at most 3 changes of 600 characters each,
limited to known policy sections, and a patch that names a scenario, patient or quoted line
is rejected. I read the diff and approve it. The full eval then re-runs with the same k and
scenario set. The gate accepts only if targets strictly improve, no fully-passing scenario
drops (even by one run), heldout does not drop and there are no error runs; otherwise it
rolls back and keeps v1 active. The changelog is generated from `results.json`.

**4. Before and after scores.** None. I did not complete a real run. Every attempt was
stopped by free-tier provider limits (OpenRouter free models: 50 requests/day; Gemini judge
model: 20/day), and a baseline needs several hundred calls. The loop and gate are verified
with fake LLMs and mutation checks only. I would rather report no score than a mock one.

**5. One thing I would change for a real clinic.** Bind confirmation to the exact action.
Today `confirmed` is a boolean set when the patient's message contains words like "yes",
"sure" or "please" while the state is CONFIRM. It is a substring match, so "I'm not sure"
contains "sure" and "yesterday" contains "yes". It is also not tied to the specific slot or
appointment being confirmed. In a real clinic I would store a pending action (type,
patient, slot id), show a read-back, accept only an explicit affirmative parsed against that
pending action, and have the guard re-check the arguments at write time. A wrong-patient or
wrong-slot booking is the failure with real harm, and the keyword shortcut is the weakest
link in an otherwise code-enforced design.

**6. Where AI helped and where I overrode it.** Claude (chat) helped plan and review;
Cursor and other coding agents wrote most of the code. I overrode them when: an edit let
`add` overwrite a protected policy section; guards were marked "protected" with no failing
test; an agent hand-commented guards to "mutate" them (I banned it and ran the script
myself); a decisions entry described a change that `git diff` showed was never made;
checklist items were ticked before any mutation had been run; and a test docstring said
"human-approved" for a change I never approved. Details are in `AI_USAGE.md`.