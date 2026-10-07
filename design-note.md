# Design Note

**1. Key design choices.** Safety lives in code, not only in the prompt. Tools never accept a `patient_id` from the model. A write needs a verified session, a confirmed action, ownership of the appointment and an open slot. An injected instruction can change what the model says, but not what the tools permit. An explicit state machine limits which tools exist in each state. Policy is a YAML file, so the improvement loop has exactly one thing it may change.

**2. How the eval works and where it is blind.** 15 scenarios (9 train, 6 heldout), scored in three layers in order of trust: database and session state, tool-call trace, then an LLM judge. State is ground truth because the judge sees only text and can be fooled by "you're booked!" when nothing was written. Blind spots: the judge cannot see tool arguments or DB state, the simulator is an LLM and can drift, k is small, and the data is synthetic. Judge agreement with hand labels: not measured.

**3. How the improvement loop works.** The reflector sees train failures only, and a leak check runs on its final prompt. It returns at most 3 policy changes; I read the diff and approve. The full eval then re-runs with the same k and scenarios. The gate accepts only if targets strictly improve, no fully-passing scenario drops, heldout does not drop and there are no error runs. Otherwise it rolls back and v1 stays active.

**4. Before and after.** All runs are real, k=1, on free-tier Gemini models (agent gemini-3.1-flash-lite, simulator gemini-3.5-flash-lite, judge gemini-3.6-flash).

| Run | Code | Policy | Overall | Train | Heldout |
|---|---|---|---|---|---|
| 012952 | original | v1 | 10/15 | 7/9 | 3/6 |
| 015628 | original | v2 | 8/15 | 5/9 | 3/6 |
| 021916 | state fix | v1 | 11/15 | 6/9 | 5/6 |

The reflector saw `ambiguous_date` fail and proposed one sentence: try tools before escalating. The gate **rejected** it. The target stayed at 0%, and `happy_path_book` and `medical_advice` dropped from passing to failing. v1 stayed active.

The trace showed why. The patient gave identity and request in one message, so the state stayed VERIFIED and the code blocked `list_providers`. No policy sentence can grant a tool the code withholds. I fixed this in code, with three tests (run 021916). The scenario then reached `check_availability` but still failed, because of a second bug: a same-day range (`date_start == date_end`) returns no slots. That bug is not fixed.

Read run 021916 with care. The code changed, so it is not a clean before/after of the loop. At k=1 a one-scenario swing is noise (train fell 7/9 to 6/9 while heldout rose). Two scenarios in that run had judge parse failures. I stopped at k=1 because the free tier allows about 20 judge calls per day.

**5. One thing I would change for a real clinic.** Bind confirmation to the exact action. Today `confirmed` is set when the patient's message contains words like "yes" or "sure" while the state is CONFIRM. It is a substring match: "I'm not sure" contains "sure". It is also not tied to the slot being confirmed. I would store a pending action (type, patient, slot id), accept only an explicit affirmative against it, and have the guard re-check the arguments at write time. A wrong-patient or wrong-slot booking is the failure with real harm.

**6. Where AI helped and where I overrode it.** Claude (chat) helped plan and review; Cursor and other agents wrote most of the code. I overrode them when an edit let `add` overwrite a protected policy section, when guards were marked "protected" with no failing test, and when a decisions entry described a change `git diff` showed was never made. Details are in `AI_USAGE.md`.