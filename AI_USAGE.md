# AI Usage

I used Claude (chat) to plan and review, and Cursor and other coding agents to write most
of the code. The AI-written parts are listed below, along with where I checked, rejected
or corrected the output.

| Area | What AI did | What I changed or overrode |
|---|---|---|
| Scaffolding, DB, tools, state machine, policy_v1 | Wrote the project structure, clinic DB, 7 tools with guards, state machine and the first policy | Reviewed the guard design (the model never supplies a patient_id; writes need verified identity and confirmation). Verified with the guard and machine tests. |
| Eval harness | Wrote 15 scenarios, simulator, three scorers and judge, runner | Required state to rank above trace and judge as ground truth, since the judge sees only text. |
| Improvement loop | Wrote reflector, patch validator, gate, improve CLI | Rejected an `add` operation that overwrote a protected policy section and required append semantics plus a test. Rejected a reflector that received no policy or transcript (my printed prompt showed it). Required empty targets, mismatched k and scenario sets to be gate rejections. |
| Mutation checks | Wrote `mutation_check.py` and `mutation_table.py` | Restored `mutation_check.py` after an agent replaced it and verified its sha256. Banned hand-commenting guards to "mutate" them and ran the script myself. Reproduced one reflector mutation by hand (34 tests failed). |
| Guards with no proof | Marked guards "protected" | Demanded a named failing test for every guard. |
| Test edits | Deleted a failing heldout-persona test, then injected the persona to pass it; retargeted a cap test to fit a bug | Required a code fix and slice tests instead. A probe showed the code still failed. |
| Docs and checklists | Ticked every checklist item DONE before any mutation ran; wrote a decisions entry for a change that `git diff` showed was never made; wrote "human-approved" for a test change I never approved | Reset the checklist, removed the entry and removed the claim. |
| LLM adapter | Wrote the Gemini and OpenRouter paths, retry and quota handling, fatal-error aborts | Ran the smoke tests and live conversations myself. Found a live run that escalated incorrectly (missing state transition) and a per-day quota that was retried three times before aborting. |
| Real runs | Suggested models, commands and checks | I ran every real run. I rejected one run as invalid (judge daily quota exhausted mid-run, scenarios unjudged) and moved it out of the results. Only the valid k=1 baseline is reported. |
| Design note and README | Drafted structure and wording | I rewrote the status and limitation sections to say exactly what was and wasn't measured. |