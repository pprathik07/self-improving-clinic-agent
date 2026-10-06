# Calibration data

Transcripts and empty label templates are generated from a real baseline by:

```bash
uv run python -m clinic_agent.evals.calibrate --from-run runs/<ts>
```

For mock runs in tests only:

```bash
uv run python -m clinic_agent.evals.calibrate --from-run runs/_mock/<ts> --allow-mock
```

Mock output is stamped `MOCK, DO NOT LABEL`.

HUMAN fills `human_labels` in each `labels_NN.yaml`. Do not invent verdicts.
