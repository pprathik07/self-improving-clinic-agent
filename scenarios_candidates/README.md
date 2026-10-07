# Scenario candidates (not part of the evaluation)

These YAML drafts are **not loaded by the harness and were not used in any reported
result**. The eval harness loads only `clinic_agent/evals/scenarios/*.yaml`, which holds
the 15 frozen scenarios (9 train, 6 heldout) used for the baseline in
`runs/20261008_012952`.

They are ideas for extending coverage later. I did not add them, because scenarios and
scorers are frozen once a baseline starts. Adding a scenario afterwards would change the
test set between the baseline and the re-run, and the before/after comparison would no
longer be valid.

## If you extend the suite

1. Review each file and edit personas, openings and expectations.
2. Assign splits so that at least 2 of any added scenarios are heldout.
3. Move the chosen files into `clinic_agent/evals/scenarios/`.
4. Re-run `uv run python scripts/freeze_manifest.py`, then run a **new** baseline.
   Results from before the change are not comparable with results after it.
5. Do not move files marked "needs harness support" until that support exists.

## Inventory

| file | needs harness support? | suggested split |
|---|---|---|
| slot_taken_between_check_and_book.yaml | no (similar to the existing slot_race_condition) | heldout |
| prompt_injection_jailbreak_v2.yaml | no | heldout |
| multi_intent_cancel_and_book.yaml | no | train |
| tool_failure_timeout.yaml | **yes**: needs fault injection for tool timeouts | n/a until the harness supports it |