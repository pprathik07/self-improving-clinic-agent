# Self-Improving Clinic Scheduling Agent

A multi-turn patient scheduling agent (book / reschedule / cancel) with an evaluation
harness and an improvement loop. When a scenario fails, a reflector proposes a small,
validated patch to the agent's policy text. The same scenarios are re-run, and a gate
accepts the patch or rolls it back.

## Run it

    git clone https://github.com/pprathik07/self-improving-clinic-agent
    cd self-improving-clinic-agent
    cp .env.example .env        # add a provider key and model slugs
    uv sync --all-extras

    make agent      # talk to the agent:        uv run python -m clinic_agent
    make improve    # run the improvement loop: uv run python -m clinic_agent.loop.improve

Also available: `make eval` (score the current policy) and `make test` (302 tests,
no network, mock LLM).

## Status (read this first)

**The agent, eval harness and improvement loop are built and tested. I have not completed
a real before/after run, so this repo contains no measured scores.**

What is verified:

- 302 passing tests, 0 skipped, run with mock LLMs and no network.
- Hand-run mutation checks on the policy-patch validator: 11 guards, 0 untested.
- Frozen scenario, scorer and policy hashes (`scripts/freeze_manifest.py --check` passes).
- One live end-to-end booking on a real model: identity verified, slot found, explicit
  confirmation, appointment written and confirmed in the database.

What is not verified: the improvement loop (reflector, patch, re-run, gate) has only been
exercised with fake LLMs.

Why there is no baseline: every real attempt was cut off by free-tier provider limits.
OpenRouter free models allowed 50 requests/day and 20/minute; the Gemini judge model allowed
5/minute and 20/day; another Gemini model returned 503 demand spikes. One baseline needs
several hundred model calls. With enough quota it is one command:

    uv run python -m clinic_agent.evals policy/policy_v1.yaml 3

No mock-LLM number is reported as a result anywhere in this repo.

## How it works

- **Safety is enforced in code, not only in the prompt.** A state machine limits which tools
  exist in each state. Tool guards block writes when the patient is unverified, the action
  is unconfirmed, the appointment belongs to someone else, or the slot is taken. The model
  never supplies a `patient_id`; identity comes from session state. Three failed
  verifications lock the session and escalate to a human.
- **Policy is data.** Behaviour text lives in `policy/policy_v1.yaml`. The loop may edit
  only this file.
- **Three scoring layers, in order of trust:** (1) database and session state, (2) tool-call
  trace, (3) an LLM judge that sees only the transcript. A scenario passes only if all three
  pass.
- **15 scenarios** (9 train, 6 heldout). Adversarial cases outnumber happy paths: wrong DOB,
  lockout, emergency symptoms, medical advice, prompt injection, another patient's records,
  ambiguous dates, "are you a robot".
- **The loop:**
  1. `failures.md` is generated from a real baseline run.
  2. The reflector reads train failures only and returns one JSON patch.
  3. `patch.py` validates it: at most 3 changes, 600 characters each, known sections only,
     protected sections are add-only, no scenario ids, patient names or quoted lines,
     `policy_v1.yaml` is never overwritten.
  4. I approve the printed diff.
  5. The full eval is re-run with the same k and scenario set.
  6. `gate.py` accepts or rolls back, and a CHANGELOG entry is generated from `results.json`.
- **Gate rules:** every target scenario must strictly improve; no fully-passing scenario may
  drop (even by one run); the heldout aggregate must not drop; no error runs.

## Configuration

| Variable | Meaning |
|---|---|
| `MOCK_LLM` | `1` uses a deterministic fake LLM (tests); `0` uses a real provider |
| `LLM_PROVIDER` | `openrouter`, `gemini` or `auto`. Use an explicit value for real evals |
| `OPEN_ROUTER_API_KEY`, `GEMINI_API_KEY` | Provider keys. Keep them in `.env`, which is untracked |
| `AGENT_MODEL`, `JUDGE_MODEL`, `SIM_MODEL`, `REFLECTOR_MODEL` | Model slugs. Real mode requires agent, judge and simulator to be three different models |

Fatal auth/model errors (401/403/404) abort a run immediately. Rate limits (429) are
classified by quota id: per-day limits abort, per-minute limits retry after the server's
retry delay, up to 3 times.

## Results

<!-- RESULTS:START -->
PENDING
<!-- RESULTS:END -->

## Verification

    make test                              # 302 tests, mock LLM
    uv run python scripts/mutation_check.py   # patch.py guards: each must have a failing test
    uv run python scripts/freeze_manifest.py --check
    uv run python scripts/preflight.py --pre-real

A caught mutation proves that some test fails, not that the test is precise.

## Known limitations

- The judge is transcript-only; it cannot see database state or tool arguments.
- The simulator is an LLM and can drift. k is small. Data is synthetic.
- The gate protects only scenarios that were fully passing before the patch; partially
  passing scenarios can still get worse.
- The heldout leak check is lexical (whole-text plus 8-word runs). Paraphrases pass, and
  single-word names are missed.
- Confirmation is detected by keyword match on the patient's message, not bound to the exact
  action (see `design-note.md`).
- `check_availability` returned no slots for a same-day start/end range that a
  specialty-only query did return. Date scenarios may fail for reasons no policy patch can fix.
- Not production-ready: no UI, synthetic data only.

## Layout

    clinic_agent/
      clinic/    DB schema, seed data, frozen clock
      tools/     tool functions and guards
      agent/     state machine, runner, policy loader
      evals/     scenarios, simulator, scorers, judge, failures/report/calibrate
      loop/      patch, reflector, gate, improve, CHANGELOG
      llm.py     the only file that imports an LLM SDK
    policy/      policy_v1.yaml (policy_v2.yaml is generated by the loop)
    scripts/     mutation_check, freeze_manifest, smoke_llm, preflight
    tests/

See `design-note.md` for design choices and `AI_USAGE.md` for where AI helped and where I
overrode it.