"""Tests for eval runner error handling and fail-fast behavior."""

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from clinic_agent.evals.run_eval import run_scenario, run_eval
from clinic_agent.evals.scenario_schema import Scenario
from clinic_agent.evals.scorers import ScoreReport
from clinic_agent.llm import FatalLLMError


class TestAuthErrorFailFast:
    """FatalLLMError aborts the run immediately (class-based, not substring)."""

    def test_auth_error_aborts_run_real_mode(self, tmp_path, monkeypatch):
        """In real mode, FatalLLMError from run_turn aborts the scenario."""
        scenario = Scenario(
            id="test_scenario",
            split="train",
            tags=[],
            seed_db="default",
            patient_persona="Test patient",
            opening="Hello",
            max_turns=5,
            expect={"state": {}, "trace": {}, "judge": []},
        )

        with patch("clinic_agent.evals.run_eval.run_turn") as mock_run:
            mock_run.side_effect = FatalLLMError(
                "LLM API fatal error (abort immediately): 403 PERMISSION_DENIED"
            )

            monkeypatch.setenv("MOCK_LLM", "0")

            with pytest.raises(FatalLLMError) as exc_info:
                run_scenario(
                    scenario,
                    "policy/policy_v1.yaml",
                    trace_dir=tmp_path,
                )

            assert "403" in str(exc_info.value) or "PERMISSION_DENIED" in str(exc_info.value)

    def test_run_eval_aborts_on_auth_error_real_mode(self, tmp_path, monkeypatch):
        """run_eval returns {} and does not write results.json on FatalLLMError."""
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        scenario_file = scenarios_dir / "test.yaml"
        scenario_file.write_text("""
id: test
split: train
tags: []
seed_db: default
patient_persona: Test
opening: Hello
max_turns: 5
expect:
  state: {}
  trace: {}
  judge: []
""")

        with patch("clinic_agent.evals.run_eval.run_scenario") as mock_run:
            mock_run.side_effect = FatalLLMError(
                "LLM API fatal error (abort immediately): 403 PERMISSION_DENIED"
            )

            monkeypatch.setenv("MOCK_LLM", "0")

            results = run_eval(
                scenarios_dir=str(scenarios_dir),
                policy_path="policy/policy_v1.yaml",
                k=1,
            )

            assert results == {}


class TestJudgeErrorHandling:
    """Test that judge errors are marked as infrastructure errors, not scenario failures."""

    def test_judge_error_marked_as_infrastructure_error(self, tmp_path):
        """Judge errors should be marked in metadata and excluded from pass rate."""
        scenario = Scenario(
            id="test_scenario",
            split="train",
            tags=[],
            seed_db="default",
            patient_persona="Test patient",
            opening="Hello",
            max_turns=5,
            expect={"state": {}, "trace": {}, "judge": ["Test rubric"]},
        )

        # Mock judge to raise an error
        with patch("clinic_agent.evals.run_eval.judge_transcript") as mock_judge:
            mock_judge.side_effect = RuntimeError("Judge API error")

            report, run_meta = run_scenario(
                scenario,
                "policy/policy_v1.yaml",
                trace_dir=tmp_path,
            )

            # Should have error field set
            assert run_meta.get("error") is not None
            assert "judge" in run_meta["error"]

            # Should not be marked as passed
            assert run_meta.get("passed") is False

            # Scores should be None (not calculated)
            assert run_meta.get("scores") is None


class TestModelValidation:
    """Test model name validation in real mode."""

    def test_requires_different_models_in_real_mode(self, tmp_path, monkeypatch):
        """Real mode should require all three models to be different."""
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        scenario_file = scenarios_dir / "test.yaml"
        scenario_file.write_text("""
id: test
split: train
tags: []
seed_db: default
patient_persona: Test
opening: Hello
max_turns: 5
expect:
  state: {}
  trace: {}
  judge: []
""")

        # Set all models to the same value
        monkeypatch.setenv("MOCK_LLM", "0")
        monkeypatch.setenv("AGENT_MODEL", "gemini-2.0-flash")
        monkeypatch.setenv("JUDGE_MODEL", "gemini-2.0-flash")
        monkeypatch.setenv("SIM_MODEL", "gemini-2.0-flash")

        results = run_eval(
            scenarios_dir=str(scenarios_dir),
            policy_path="policy/policy_v1.yaml",
            k=1,
        )

        # Should return empty dict (validation failed)
        assert results == {}


class TestFatalAbortOnFirstCall:
    """Fatal 403/404/quota errors abort on call 1; no scenario is scored."""

    @pytest.mark.parametrize(
        "error_msg",
        [
            "403 PERMISSION_DENIED",
            "404 NOT_FOUND model not found",
            "RESOURCE_EXHAUSTED quota exceeded",
        ],
        ids=["403", "404", "quota"],
    )
    def test_fatal_error_aborts_on_call_1(self, tmp_path, monkeypatch, error_msg):
        """run_eval aborts on first fatal error; call_count==1; no results written."""
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        # Two scenarios — second must never be scored if abort is immediate
        for sid in ("first", "second"):
            (scenarios_dir / f"{sid}.yaml").write_text(f"""
id: {sid}
split: train
tags: []
seed_db: default
patient_persona: Test
opening: Hello
max_turns: 5
expect:
  state: {{}}
  trace: {{}}
  judge: []
""")

        call_count = {"n": 0}

        def fake_run_scenario(*args, **kwargs):
            call_count["n"] += 1
            raise FatalLLMError(f"LLM API fatal error (abort immediately): {error_msg}")

        monkeypatch.setenv("MOCK_LLM", "0")
        monkeypatch.setenv("AGENT_MODEL", "model-a")
        monkeypatch.setenv("JUDGE_MODEL", "model-b")
        monkeypatch.setenv("SIM_MODEL", "model-c")

        # Keep policy reachable after chdir
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()
        shutil.copy(
            Path(__file__).resolve().parents[1] / "policy" / "policy_v1.yaml",
            policy_dir / "policy_v1.yaml",
        )
        monkeypatch.chdir(tmp_path)

        with patch("clinic_agent.evals.run_eval.run_scenario", side_effect=fake_run_scenario):
            results = run_eval(
                scenarios_dir=str(scenarios_dir),
                policy_path="policy/policy_v1.yaml",
                k=1,
            )

        assert results == {}
        assert call_count["n"] == 1, f"Expected abort on call 1, got {call_count['n']}"
        # No results.json under runs/
        runs_dir = tmp_path / "runs"
        if runs_dir.exists():
            assert list(runs_dir.rglob("results.json")) == []


class TestRequireRealVsMock:
    """REQUIRE_REAL=1 with MOCK_LLM=1 must abort."""

    def test_require_real_aborts_when_mock_on(self, tmp_path, monkeypatch):
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        (scenarios_dir / "t.yaml").write_text("""
id: t
split: train
tags: []
seed_db: default
patient_persona: Test
opening: Hello
max_turns: 5
expect:
  state: {}
  trace: {}
  judge: []
""")
        monkeypatch.setenv("MOCK_LLM", "1")
        monkeypatch.setenv("REQUIRE_REAL", "1")
        results = run_eval(
            scenarios_dir=str(scenarios_dir),
            policy_path="policy/policy_v1.yaml",
            k=1,
        )
        assert results == {}

    """Test that mock runs go to runs/_mock/ and real runs go to runs/."""

    def test_mock_runs_go_to_mock_subdirectory(self, tmp_path, monkeypatch):
        """Mock mode should write results to runs/_mock/timestamp/."""
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        scenario_file = scenarios_dir / "test.yaml"
        scenario_file.write_text("""
id: test
split: train
tags: []
seed_db: default
patient_persona: Test
opening: Hello
max_turns: 5
expect:
  state: {}
  trace: {}
  judge: []
""")

        # Copy policy file to tmp_path
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()
        shutil.copy("policy/policy_v1.yaml", policy_dir / "policy_v1.yaml")

        # Set mock mode
        monkeypatch.setenv("MOCK_LLM", "1")

        # Change to temp directory for this test
        monkeypatch.chdir(tmp_path)

        results = run_eval(
            scenarios_dir=str(scenarios_dir),
            policy_path="policy/policy_v1.yaml",
            k=1,
        )

        # Check that results were written to runs/_mock/
        mock_runs_dir = tmp_path / "runs" / "_mock"
        assert mock_runs_dir.exists()
        assert list(mock_runs_dir.glob("*/results.json"))  # At least one results.json

        # Check that no results were written to runs/ directly
        runs_dir = tmp_path / "runs"
        runs_results = list(runs_dir.glob("*/results.json"))
        for path in runs_results:
            assert "_mock" not in str(path), f"Found results in wrong location: {path}"

    def test_real_runs_go_to_runs_directory(self, tmp_path, monkeypatch):
        """Real mode should write results to runs/timestamp/ (not _mock)."""
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        scenario_file = scenarios_dir / "test.yaml"
        scenario_file.write_text("""
id: test
split: train
tags: []
seed_db: default
patient_persona: Test
opening: Hello
max_turns: 5
expect:
  state: {}
  trace: {}
  judge: []
""")

        # Copy policy file to tmp_path
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()
        shutil.copy("policy/policy_v1.yaml", policy_dir / "policy_v1.yaml")

        # Set real mode with different models
        monkeypatch.setenv("MOCK_LLM", "0")
        monkeypatch.setenv("AGENT_MODEL", "model1")
        monkeypatch.setenv("JUDGE_MODEL", "model2")
        monkeypatch.setenv("SIM_MODEL", "model3")

        # Change to temp directory for this test
        monkeypatch.chdir(tmp_path)

        # Mock run_scenario to avoid actual LLM calls
        with patch("clinic_agent.evals.run_eval.run_scenario") as mock_run:
            mock_report = ScoreReport()
            mock_report.state_checks = []
            mock_report.trace_checks = []
            mock_report.judge_checks = []
            mock_run.return_value = (mock_report, {
                "scenario_id": "test",
                "split": "train",
                "tags": [],
                "turns": 1,
                "final_state": "done",
                "passed": True,
                "error": None,
                "scores": mock_report.summary
            })

            results = run_eval(
                scenarios_dir=str(scenarios_dir),
                policy_path="policy/policy_v1.yaml",
                k=1,
            )

            # Check that results were written to runs/ (not _mock)
            runs_dir = tmp_path / "runs"
            assert runs_dir.exists()
            runs_results = list(runs_dir.glob("*/results.json"))
            assert len(runs_results) > 0, "No results.json found in runs/"

            # Verify none are in _mock
            for path in runs_results:
                assert "_mock" not in str(path), f"Found results in _mock: {path}"


class TestInteractiveDbRelaunch:
    """run_interactive second launch must not crash on leftover clinic.db."""

    def test_run_interactive_setup_twice_same_file_no_raise(self, tmp_path, monkeypatch):
        """Call run_interactive setup twice against same tmp_path/clinic.db with MOCK_LLM=1.

        Simulates a user launching the CLI twice (leftover clinic.db). With the bug
        the second call raises sqlite3.IntegrityError: UNIQUE constraint on providers.id
        (seed_db re-INSERTs). With reset_db used instead, both calls cleanly succeed.

        Uses scripted input: input() returns "quit" on first user prompt so the loop
        exits cleanly without LLM tool follow-ups.
        """
        from pathlib import Path
        from clinic_agent.clinic.db import init_db, reset_db

        policy_src = Path("policy/policy_v1.yaml").resolve()
        assert policy_src.exists(), f"Need policy v1 for run_interactive setup, missing: {policy_src}"

        monkeypatch.setenv("MOCK_LLM", "1")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "policy").mkdir()
        import shutil
        shutil.copy(policy_src, tmp_path / "policy" / "policy_v1.yaml")

        input_calls = {"n": 0}

        def fake_input(prompt=""):
            input_calls["n"] += 1
            return "quit"

        import clinic_agent.agent.runner as runner_mod

        with patch("builtins.input", side_effect=fake_input):
            # First launch — creates & seeds clinic.db
            runner_mod.run_interactive()
            # Second launch — leftover clinic.db now exists in tmp_path
            runner_mod.run_interactive()

        # Both launches should have prompted once and exited cleanly
        assert input_calls["n"] >= 2

        # And DB now has exactly the 3 default providers (no duplicates)
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "clinic.db"))
        try:
            n = conn.execute("SELECT COUNT(*) FROM providers").fetchone()[0]
        finally:
            conn.close()
        assert n == 3


class TestRunnerNoDuplicateAssistantAppends:
    """Prove that _handle_tool_calls does not append duplicate assistant messages.

    Scenario: mock LLM returns tool_call -> tool_call -> final_text (three model turns).
    Assert session.history has exactly one assistant Message per model turn (3 total),
    tool_result Messages are present in execution order, and no role-duplicates occur.
    """

    def test_chained_tool_calls_no_duplicate_assistant_entries(self, monkeypatch):
        """run_turn with LLM sequence (verify_patient TC, check_availability TC, text) yields exactly 3 assistant msgs."""
        from types import SimpleNamespace
        from clinic_agent.agent.runner import run_turn, TraceLogger
        from clinic_agent.agent.state import SessionState, AgentState, Message
        from clinic_agent.clinic.db import init_db, reset_db
        from clinic_agent.agent.policy import load_policy, policy_to_prompt
        from clinic_agent.llm import ToolCall, LLMResponse

        # --- Setup real DB so tool execution (verify_patient) works ---
        conn = init_db(in_memory=True)
        reset_db(conn, "default")

        policy = load_policy("policy/policy_v1.yaml")
        policy_text = policy_to_prompt(policy)

        # --- Opening: user provides name + dob so verify_patient passes state guard ---
        session = SessionState()
        session.state = AgentState.IDENTIFY
        session.messages.append(Message(
            role="user",
            content="Hi my name is Asha Rao, DOB 1990-05-14. I need a cardiology appointment.",
        ))
        trace = TraceLogger()

        # --- Programmed chat() sequence: 3 turns (TC1, TC2, final text) ---
        chat_calls = {"n": 0}

        def fake_chat(**kwargs):
            chat_calls["n"] += 1
            turn = chat_calls["n"]
            assert turn <= 3, f"Too many chat() calls: {turn}"
            if turn == 1:
                # Model turn 1: call verify_patient
                return LLMResponse(
                    content="Let me verify your identity first.",
                    tool_calls=[ToolCall(
                        id="c_ver_1", name="verify_patient",
                        arguments={"name": "Asha Rao", "dob": "1990-05-14"},
                    )],
                    usage={"input_tokens": 50, "output_tokens": 20},
                )
            elif turn == 2:
                # Model turn 2: call check_availability (after verify succeeded)
                return LLMResponse(
                    content="Now checking cardiology slots for you.",
                    tool_calls=[ToolCall(
                        id="c_avl_2", name="check_availability",
                        arguments={"specialty": "Cardiology"},
                    )],
                    usage={"input_tokens": 60, "output_tokens": 25},
                )
            else:
                # Model turn 3: final text only (no tool calls)
                return LLMResponse(
                    content="I found cardiology slots. The first one is Monday at 9am with Dr. Patel — would you like me to book that?",
                    usage={"input_tokens": 70, "output_tokens": 30},
                )

        monkeypatch.setattr("clinic_agent.agent.runner.chat", fake_chat)

        # --- Execute one run_turn (runner recurses internally for followups) ---
        run_turn(session, conn, policy_text, trace)

        # --- Assert exactly 3 chat() invocations (TC1, TC2, final text) ---
        assert chat_calls["n"] == 3, f"Expected 3 model turns, got {chat_calls['n']}"

        # --- Count assistant messages in session history ---
        assistant_msgs = [m for m in session.messages if m.role == "assistant"]
        assert len(assistant_msgs) == 3, (
            f"Expected exactly 3 assistant messages (one per model turn), "
            f"got {len(assistant_msgs)}. This indicates a duplicate-append bug "
            f"in _handle_tool_calls. Messages: {[(m.role, m.tool_name) for m in session.messages]!r}"
        )

        # --- Assert assistant turns each have correct tool_name association ---
        assert assistant_msgs[0].tool_name == "verify_patient", \
            f"Turn 1 assistant should carry verify_patient tool_name, got {assistant_msgs[0].tool_name!r}"
        assert assistant_msgs[1].tool_name == "check_availability", \
            f"Turn 2 assistant should carry check_availability tool_name, got {assistant_msgs[1].tool_name!r}"
        assert assistant_msgs[2].tool_name is None, \
            f"Turn 3 (final text) should have no tool_name, got {assistant_msgs[2].tool_name!r}"

        # --- Assert tool_result messages are present and ORDERED correctly ---
        tool_results = [m for m in session.messages if m.role == "tool_result"]
        assert len(tool_results) == 2, \
            f"Expected 2 tool_result messages (one per tool call), got {len(tool_results)}"
        assert tool_results[0].tool_name == "verify_patient", \
            f"First tool_result should be verify_patient, got {tool_results[0].tool_name!r}"
        assert tool_results[1].tool_name == "check_availability", \
            f"Second tool_result should be check_availability, got {tool_results[1].tool_name!r}"

        # --- Assert role ordering in session is strictly alternating, no back-to-back assistant ---
        roles = [m.role for m in session.messages]
        for i in range(len(roles) - 1):
            current, nxt = roles[i], roles[i + 1]
            assert not (current == "assistant" and nxt == "assistant"), (
                f"Back-to-back assistant messages at positions [{i},{i+1}]: "
                f"roles={roles!r}. This is the duplicate-append bug."
            )

        # --- Final text assistant message should contain the LLM's natural-language response ---
        final_resp_text = assistant_msgs[2].content
        assert "cardiology" in final_resp_text.lower() or "slots" in final_resp_text.lower(), (
            f"Final assistant message missing LLM text content. Got: {final_resp_text!r}"
        )


class TestClinicDebugPerTurnPrinting:
    """CLINIC_DEBUG=1 prints per-turn tool calls (name,args), result (trunc 200), guard rejection.

    Default (unset / 0): default output unchanged, no [DEBUG] lines emitted anywhere,
    run_turn debug_events list remains None, run_interactive produces only Agent: lines.
    """

    @staticmethod
    def _fake_chat_sequence(monkeypatch, responses):
        """Override clinic_agent.agent.runner.chat with a pop-from-list callable.

        Each response is either an LLMResponse or a dict with content/tool_calls.
        """
        call_idx = {"n": 0}

        from clinic_agent.llm import LLMResponse, ToolCall
        import clinic_agent.agent.runner as runner_mod

        def _chat(messages=None, system="", tools=None, model_key="AGENT_MODEL", **_):
            i = call_idx["n"]
            call_idx["n"] += 1
            resp = responses[i % len(responses)]
            if isinstance(resp, LLMResponse):
                return resp
            return LLMResponse(
                content=resp.get("content", ""),
                tool_calls=resp.get("tool_calls", []),
                usage={},
                stop_reason="stop",
                provider=getattr(resp, "provider", ""),
            )

        monkeypatch.setattr(runner_mod, "chat", _chat)
        return call_idx

    def test_run_turn_debug_events_populated_when_list_passed(self, monkeypatch):
        """run_turn with debug_events=[] appends every tool event, including guard block."""
        from clinic_agent.agent.runner import run_turn, TraceLogger
        from clinic_agent.agent.state import SessionState, AgentState, Message
        from clinic_agent.clinic.db import init_db, reset_db
        from clinic_agent.agent.policy import load_policy, policy_to_prompt
        from clinic_agent.llm import LLMResponse, ToolCall
        import clinic_agent.agent.runner as runner_mod

        conn = init_db(in_memory=True)
        reset_db(conn, "default")
        policy = load_policy("policy/policy_v1.yaml")
        policy_text = policy_to_prompt(policy)

        # --- Session prepped in EXECUTE state so book_appt is state-allowed,
        # but slot_id nonexistent -> GuardError fires inside the tool. ---
        session = SessionState()
        session.state = AgentState.EXECUTE
        session.confirmed = True
        session.verified_patient_id = "PAT-001"
        session.messages.append(Message(role="user", content="Confirm my booking"))

        chat_calls = {"n": 0}

        def fake_chat(messages=None, system="", tools=None, model_key="AGENT_MODEL", **_):
            idx = chat_calls["n"]
            chat_calls["n"] += 1
            if idx == 0:
                # Parallel tool call in EXECUTE state (allowed set = {book, cancel,
                # reschedule, escalate}). Use integer ids so args_model validates,
                # then the guard fires with GuardError:
                #   (1) cancel_appointment with appointment_id=9999 → not in DB
                #       → appointment-not-found GuardError.
                #   (2) book_appointment with slot_id=8888 → slot not in DB
                #       → slot_not_found GuardError.
                # _handle_tool_calls loop runs both → 2 debug events:
                #   type=tool_guard_blocked + type=tool_guard_blocked.
                return LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="call_c1", name="cancel_appointment",
                                 arguments={"appointment_id": 9999}),
                        ToolCall(id="call_b1", name="book_appointment",
                                 arguments={"slot_id": 8888,
                                            "patient_id": "PAT-001",
                                            "reason": "checkup"}),
                    ],
                    usage={},
                    stop_reason="tool_use",
                    provider="openrouter",
                )
            # Follow-up chat after tool results: pure natural-language text.
            return LLMResponse(
                content="Looks like neither action matched valid records — let me check the list.",
                tool_calls=[],
                usage={},
                stop_reason="stop",
                provider="openrouter",
            )

        monkeypatch.setattr(runner_mod, "chat", fake_chat)

        debug_events: list[dict] = []
        run_turn(session, conn, policy_text, TraceLogger(), debug_events=debug_events)

        # --- 2 events recorded: 2 guard-blocked calls (nonexistent appointment id
        # and nonexistent slot id) both in EXECUTE state (state-allowed so guards
        # run, not blocked as tool_blocked_state) ---
        assert len(debug_events) == 2, (
            f"Expected 2 debug events (2 guard-blocked tool calls), got {len(debug_events)}: {debug_events!r}"
        )
        ev_cancel, ev_book = debug_events[0], debug_events[1]

        assert ev_cancel["type"] == "tool_guard_blocked", (
            f"Expected tool_guard_blocked (appointment id 9999 not in DB → cancel guard), "
            f"got type={ev_cancel['type']!r} reason={ev_cancel.get('reason')!r}"
        )
        assert ev_cancel["name"] == "cancel_appointment"
        assert ev_cancel["args"] == {"appointment_id": 9999}
        assert (
            "appointment" in ev_cancel["reason"].lower()
            or "not found" in ev_cancel["reason"].lower()
        ), (
            f"Expected cancel guard reason to mention appointment/not found, got {ev_cancel['reason']!r}"
        )

        assert ev_book["type"] == "tool_guard_blocked", (
            f"Expected tool_guard_blocked (slot 8888 not in DB → book guard), "
            f"got type={ev_book['type']!r} reason={ev_book.get('reason')!r}"
        )
        assert ev_book["name"] == "book_appointment"
        assert ev_book["args"] == {"slot_id": 8888,
                                   "patient_id": "PAT-001",
                                   "reason": "checkup"}
        assert "slot" in ev_book["reason"].lower() or "not found" in ev_book["reason"].lower(), (
            f"Expected book guard reason to mention slot / not found, got {ev_book['reason']!r}"
        )

        # --- Also add a success event for testing: rerun run_turn on a fresh
        # session2 in IDENTIFY state with verify_patient (allowed, DB seeded with
        # PAT-001 / PAT-002 / PAT-003) so the tool call succeeds. ---
        session2 = SessionState()
        session2.state = AgentState.IDENTIFY
        session2.messages.append(Message(role="user", content="Hi I'm Asha Rao, 1990-05-14"))
        chat_calls2 = {"n": 0}

        def fake_chat2(messages=None, system="", tools=None, model_key="AGENT_MODEL", **_):
            idx = chat_calls2["n"]
            chat_calls2["n"] += 1
            if idx == 0:
                return LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="call_v", name="verify_patient",
                                 arguments={"name": "Asha Rao", "dob": "1990-05-14"}),
                    ],
                    usage={},
                    stop_reason="tool_use",
                    provider="openrouter",
                )
            return LLMResponse(content="Verified — thanks!", tool_calls=[], usage={},
                               stop_reason="stop")

        monkeypatch.setattr(runner_mod, "chat", fake_chat2)
        debug_events2: list[dict] = []
        run_turn(session2, conn, policy_text, TraceLogger(), debug_events=debug_events2)
        assert len(debug_events2) == 1, (
            f"Expected 1 debug event (1 success verify_patient), got {len(debug_events2)}: {debug_events2!r}"
        )
        ev_ok = debug_events2[0]
        assert ev_ok["type"] == "tool_call"
        assert ev_ok["name"] == "verify_patient"
        assert ev_ok["args"] == {"name": "Asha Rao", "dob": "1990-05-14"}
        assert isinstance(ev_ok["result"], str) and len(ev_ok["result"]) > 0, (
            f"Expected non-empty result JSON, got {ev_ok['result']!r}"
        )

        # --- _truncate helper: 500-char string becomes 200-char with "..." suffix; 3-char stays 3 ---
        long_result = "x" * 500
        assert len(runner_mod._truncate(long_result, 200)) == 200
        assert runner_mod._truncate(long_result, 200).endswith("...")
        assert runner_mod._truncate("abc", 200) == "abc"

        # --- Formatting produces [DEBUG] prefixed lines with both result + guard info ---
        lines = runner_mod._format_debug_events(debug_events + debug_events2)
        joined = "\n".join(lines)
        for ln in lines:
            assert ln.startswith("[DEBUG]"), f"Bad [DEBUG] prefix on: {ln!r}"
        # Tool call header line with sort_keys JSON args
        assert "tool_call: verify_patient(" in joined
        assert '"dob": "1990-05-14"' in joined and '"name": "Asha Rao"' in joined
        # Success result line present
        assert "[DEBUG]   result:" in joined
        # BLOCKED (guard) lines for both cancel and book
        assert "[DEBUG]   BLOCKED (guard):" in joined
        # Both internal reasons surface in the formatted text
        assert "9999" in joined or "appointment" in joined.lower()
        assert "8888" in joined or "slot" in joined.lower() or "not found" in joined.lower()

    def test_run_turn_debug_events_none_default_no_populate(self, monkeypatch):
        """run_turn default call (debug_events=None, i.e. CLINIC_DEBUG off) never
        allocates, never touches debug_events arg; _execute_tool still returns
        result_text cleanly (tuple unpacking handled)."""
        from clinic_agent.agent.runner import run_turn, TraceLogger
        from clinic_agent.agent.state import SessionState, AgentState, Message
        from clinic_agent.clinic.db import init_db, reset_db
        from clinic_agent.agent.policy import load_policy, policy_to_prompt
        from clinic_agent.llm import LLMResponse, ToolCall
        import clinic_agent.agent.runner as runner_mod

        conn = init_db(in_memory=True)
        reset_db(conn, "default")
        policy = load_policy("policy/policy_v1.yaml")
        policy_text = policy_to_prompt(policy)

        chat_calls = {"n": 0}

        def fake_chat(messages=None, system="", tools=None, model_key="AGENT_MODEL", **_):
            idx = chat_calls["n"]
            chat_calls["n"] += 1
            if idx == 0:
                return LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="call_v1", name="verify_patient",
                                 arguments={"name": "Asha Rao", "dob": "1990-05-14"}),
                    ],
                    usage={},
                    stop_reason="tool_use",
                )
            return LLMResponse(content="Welcome verified!", tool_calls=[], usage={},
                               stop_reason="stop")

        monkeypatch.setattr(runner_mod, "chat", fake_chat)

        session = SessionState()
        session.state = AgentState.IDENTIFY
        session.messages.append(Message(role="user", content="Hi"))

        # Default — no debug_events arg at all (unset)
        resp = run_turn(session, conn, policy_text, TraceLogger())
        assert isinstance(resp, str) and "Welcome" in resp, (
            f"Expected text response, got {resp!r}"
        )
        # No side effects: _clinic_debug_enabled() returns False by default (env unset)
        assert runner_mod._clinic_debug_enabled() is False

    def test_run_interactive_debug_on_prints_debug_lines_off_silent(self, tmp_path, monkeypatch, capsys):
        """run_interactive with CLINIC_DEBUG=1 captures [DEBUG] lines per turn on stdout;
        with CLINIC_DEBUG unset/unset=0 stdout has only Agent: lines (no [DEBUG] prefix).

        Uses the same tmp_path/policy-copy trick as TestInteractiveDbRelaunch, with
        fake_input returning "quit" on first user prompt so only the greeting turn runs.
        chat() mock: greeting turn returns a verify_patient TC -> followup text, so the
        first turn has a real tool call + tool_result and debug_events become non-empty.
        """
        from pathlib import Path
        from clinic_agent.llm import LLMResponse, ToolCall
        import clinic_agent.agent.runner as runner_mod

        policy_src = Path("policy/policy_v1.yaml").resolve()
        assert policy_src.exists()

        (tmp_path / "policy").mkdir()
        import shutil as _shutil
        _shutil.copy(policy_src, tmp_path / "policy" / "policy_v1.yaml")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MOCK_LLM", "1")

        def fake_input(prompt=""):
            return "quit"

        # --- OFF run first: CLINIC_DEBUG unset ---
        monkeypatch.delenv("CLINIC_DEBUG", raising=False)
        off_chat_calls = {"n": 0}

        def fake_chat_off(messages=None, system="", tools=None, model_key="AGENT_MODEL", **_):
            idx = off_chat_calls["n"]
            off_chat_calls["n"] += 1
            if idx == 0:
                return LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="call_v1", name="verify_patient",
                                 arguments={"name": "Asha Rao", "dob": "1990-05-14"}),
                    ],
                    usage={},
                    stop_reason="tool_use",
                )
            return LLMResponse(content="Greeting text", tool_calls=[], usage={},
                               stop_reason="stop")

        monkeypatch.setattr(runner_mod, "chat", fake_chat_off)
        with patch("builtins.input", side_effect=fake_input):
            runner_mod.run_interactive()
        out_off = capsys.readouterr().out
        assert "Agent: Greeting text" in out_off, (
            f"Expected Agent line in OFF output, got snippet: {out_off[:300]!r}"
        )
        assert "[DEBUG]" not in out_off, (
            f"CLINIC_DEBUG=off must not emit [DEBUG] lines. Got: {out_off[:800]!r}"
        )

        # --- ON run: CLINIC_DEBUG=1 ---
        monkeypatch.setenv("CLINIC_DEBUG", "1")
        on_chat_calls = {"n": 0}

        def fake_chat_on(messages=None, system="", tools=None, model_key="AGENT_MODEL", **_):
            idx = on_chat_calls["n"]
            on_chat_calls["n"] += 1
            if idx == 0:
                return LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="call_v1", name="verify_patient",
                                 arguments={"name": "Asha Rao", "dob": "1990-05-14"}),
                    ],
                    usage={},
                    stop_reason="tool_use",
                )
            return LLMResponse(content="Greeting text", tool_calls=[], usage={},
                               stop_reason="stop")

        monkeypatch.setattr(runner_mod, "chat", fake_chat_on)
        with patch("builtins.input", side_effect=fake_input):
            runner_mod.run_interactive()
        out_on = capsys.readouterr().out
        assert "Agent: Greeting text" in out_on
        # ON output now contains [DEBUG] lines with both the tool call header and truncated result
        assert "[DEBUG] tool_call: verify_patient(" in out_on, (
            f"Expected [DEBUG] verify_patient call in ON output. Got snippet: {out_on[:800]!r}"
        )
        assert "[DEBUG]   result:" in out_on, (
            f"Expected [DEBUG]   result: line in ON output. Got: {out_on[:800]!r}"
        )
        # JSON args present with stable sort_keys ordering
        assert '"dob": "1990-05-14"' in out_on and '"name": "Asha Rao"' in out_on

    def test_state_blocked_tool_shows_blocked_state_reason_in_debug(self, monkeypatch):
        """State-blocked tool (tc.name not in allowed for current state) produces
        tool_blocked_state event with reason mentioning state name, formatted to
        [DEBUG] BLOCKED (state): line.
        """
        from clinic_agent.agent.runner import run_turn, TraceLogger
        from clinic_agent.agent.state import SessionState, AgentState, Message
        from clinic_agent.clinic.db import init_db, reset_db
        from clinic_agent.agent.policy import load_policy, policy_to_prompt
        from clinic_agent.llm import LLMResponse, ToolCall
        import clinic_agent.agent.runner as runner_mod

        conn = init_db(in_memory=True)
        reset_db(conn, "default")
        policy = load_policy("policy/policy_v1.yaml")
        policy_text = policy_to_prompt(policy)

        chat_calls = {"n": 0}

        def fake_chat(messages=None, system="", tools=None, model_key="AGENT_MODEL", **_):
            idx = chat_calls["n"]
            chat_calls["n"] += 1
            if idx == 0:
                # In IDENTIFY state, book_appointment is NOT allowed; this will be
                # blocked by the state check (not a guard, since we never enter _execute_tool)
                return LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="call_bad", name="book_appointment",
                                 arguments={"slot_id": "slot-1", "patient_id": "p-1",
                                            "reason": "x"}),
                    ],
                    usage={},
                    stop_reason="tool_use",
                )
            return LLMResponse(content="Let me verify first.", tool_calls=[], usage={},
                               stop_reason="stop")

        monkeypatch.setattr(runner_mod, "chat", fake_chat)

        session = SessionState()
        session.state = AgentState.IDENTIFY  # disallows book_appointment
        session.messages.append(Message(role="user", content="Book slot 1 please"))

        debug_events: list[dict] = []
        run_turn(session, conn, policy_text, TraceLogger(), debug_events=debug_events)

        assert len(debug_events) == 1
        ev = debug_events[0]
        assert ev["type"] == "tool_blocked_state"
        assert ev["name"] == "book_appointment"
        assert "identify" in ev["reason"].lower()
        lines = runner_mod._format_debug_events(debug_events)
        joined = "\n".join(lines)
        assert "[DEBUG]   BLOCKED (state):" in joined
        assert "identify" in joined.lower()


class TestResultsJsonLlmProvidersUsed:
    """Ensure results.json records provider-per-role and mixed-provider runs invalidate (real mode)."""

    def _write_scenario(self, scenarios_dir: Path) -> None:
        (scenarios_dir / "sc.yaml").write_text("""
id: mixed_test
split: train
tags: []
seed_db: default
patient_persona: Cooperative patient
opening: Hi, I need an appointment.
max_turns: 4
expect:
  state: {}
  trace: {}
  judge: []
""")

    def test_results_contains_llm_providers_used_from_snapshot(self, tmp_path, monkeypatch):
        """run_eval() includes snapshot_provider_usage() output as results["llm_providers_used"]."""
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        self._write_scenario(scenarios_dir)

        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()
        shutil.copy("policy/policy_v1.yaml", policy_dir / "policy_v1.yaml")

        monkeypatch.setenv("MOCK_LLM", "1")
        monkeypatch.chdir(tmp_path)

        results = run_eval(
            scenarios_dir=str(scenarios_dir),
            policy_path="policy/policy_v1.yaml",
            k=1,
        )

        assert "llm_providers_used" in results, (
            "results dict missing 'llm_providers_used' key"
        )
        # Provider registry is global; in mock mode each labelled role should record "mock"
        usage = results["llm_providers_used"]
        assert isinstance(usage, dict)
        # agent role definitely calls chat (run_turn); provider list should be ["mock"]
        assert "agent" in usage
        assert usage["agent"] == ["mock"], f"Expected ['mock'] for agent, got {usage['agent']!r}"

        # Verify written results.json has the same field
        runs_dir = tmp_path / "runs"
        json_files = list(runs_dir.rglob("results.json"))
        assert json_files, "No results.json written"
        import json as _json
        written = _json.loads(json_files[0].read_text(encoding="utf-8"))
        assert written.get("llm_providers_used") == usage, (
            "Written results.json llm_providers_used does not match returned value"
        )

    def test_real_mode_mixed_providers_invalidates_run_no_results_json(self, tmp_path, monkeypatch):
        """REAL mode: if a role used >1 provider, run_eval returns {} + removes run dir."""
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        self._write_scenario(scenarios_dir)

        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()
        shutil.copy("policy/policy_v1.yaml", policy_dir / "policy_v1.yaml")

        # Fake "real" mode (MOCK_LLM=0) and different AGENT/JUDGE/SIM models so the
        # distinct-model guard passes.
        monkeypatch.setenv("MOCK_LLM", "0")
        monkeypatch.setenv("AGENT_MODEL", "a1")
        monkeypatch.setenv("JUDGE_MODEL", "j1")
        monkeypatch.setenv("SIM_MODEL", "s1")

        # --- Program chat() to use different providers on successive agent calls ---
        # We accomplish this by making the snapshot_provider_usage() lie: it returns
        # a mixed dict (e.g. agent=["openrouter", "gemini"]) BEFORE run_eval's
        # invalidation check runs. We monkeypatch snapshot_provider_usage at the
        # run_eval import site.
        from clinic_agent.evals import run_eval as run_eval_mod
        monkeypatch.setattr(
            run_eval_mod,
            "snapshot_provider_usage",
            lambda: {
                "agent": ["openrouter", "gemini"],  # <-- mixed: 2 providers in one role
                "sim": ["gemini"],
                "judge": ["gemini"],
            },
        )

        # Patch run_scenario so it always returns a pass (no real LLM calls in MOCK_LLM=0
        # would otherwise crash without an API key)
        with patch.object(run_eval_mod, "run_scenario") as mock_run_sc:
            report = ScoreReport()
            run_meta_dict = {
                "scenario_id": "mixed_test",
                "split": "train",
                "tags": [],
                "turns": 2,
                "final_state": "BOOKED",
                "passed": True,
                "error": None,
                "scores": None,
            }
            mock_run_sc.return_value = (report, run_meta_dict)

            monkeypatch.chdir(tmp_path)

            results = run_eval(
                scenarios_dir=str(scenarios_dir),
                policy_path="policy/policy_v1.yaml",
                k=1,
            )

        # --- ASSERTIONS ---
        assert results == {}, (
            f"Expected run_eval to return {{}} on mixed-provider invalidation, "
            f"got keys: {list(results.keys())!r}"
        )
        runs_root = tmp_path / "runs"
        if runs_root.exists():
            leftover = [p for p in runs_root.rglob("results.json")]
            assert not leftover, (
                f"Expected no results.json after mixed-provider invalidation, "
                f"found: {[str(p) for p in leftover]!r}"
            )

    def test_real_mode_mixed_providers_via_fake_chat_invalidates(self, tmp_path, monkeypatch):
        """REAL mode: actual fake chat() records 2 providers for a role → run invalid.

        Unlike the test above (which patches snapshot_provider_usage), this test
        installs a fake chat() that records DIFFERENT providers on successive
        calls that share the SAME role label. Uses a per-role call counter so
        role='agent' call 1 → openrouter, role='agent' call 2 → gemini (mix!).
        Verifies: chat_role_label context → _record_provider_usage registry →
        snapshot_provider_usage → mixed-role detection → run deleted, no results.json.
        """
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        self._write_scenario(scenarios_dir)

        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()
        shutil.copy("policy/policy_v1.yaml", policy_dir / "policy_v1.yaml")

        # Fake "real" mode with distinct models so the distinct-model guard passes
        monkeypatch.setenv("MOCK_LLM", "0")
        monkeypatch.setenv("AGENT_MODEL", "a1")
        monkeypatch.setenv("JUDGE_MODEL", "j1")
        monkeypatch.setenv("SIM_MODEL", "s1")
        monkeypatch.chdir(tmp_path)

        # Install a chat() fake that records different providers on successive
        # calls within the SAME role (per-role counter). This drives the real
        # registry path without patching the snapshot.
        import clinic_agent.llm as llm_mod
        from clinic_agent.llm import chat_role_label, clear_provider_usage, _record_provider_usage, _current_chat_role

        # Reset the global registry at start of test (runs within chat_role_label in run_eval)
        clear_provider_usage()

        per_role_calls: dict[str, int] = {}
        total_calls = {"n": 0}

        def fake_chat_recording_providers(messages, system="", tools=None, model_key="AGENT_MODEL", **_):
            total_calls["n"] += 1
            role = _current_chat_role() or "none"
            # Use a PER-ROLE counter so the SAME role gets alternating providers
            # regardless of interleaving with other roles. Role call 1 → openrouter,
            # role call 2 → gemini → that role has len(providers) = 2 → invalid.
            per_role_calls[role] = per_role_calls.get(role, 0) + 1
            call_n_for_this_role = per_role_calls[role]
            provider = "openrouter" if (call_n_for_this_role % 2 == 1) else "gemini"
            _record_provider_usage(provider)
            # Return a minimal valid LLMResponse: text only so run_scenario progresses.
            return llm_mod.LLMResponse(
                content="Welcome, let me verify your identity first.",
                tool_calls=[],
                usage={"input_tokens": 1, "output_tokens": 1},
                stop_reason="stop",
                provider=provider,
            )

        # Patch chat at the runner/eval call sites (run_turn, simulate_patient, judge_transcript)
        with patch("clinic_agent.agent.runner.chat", fake_chat_recording_providers), \
             patch("clinic_agent.evals.simulator.chat", fake_chat_recording_providers), \
             patch("clinic_agent.evals.judge.chat", fake_chat_recording_providers):

            results = run_eval(
                scenarios_dir=str(scenarios_dir),
                policy_path="policy/policy_v1.yaml",
                k=1,
            )

        # Sanity: fake chat was called enough times that at least one role ran twice
        assert total_calls["n"] >= 3, (
            f"Expected >=3 calls so some role runs twice; got {total_calls['n']}. "
            f"Per-role counts: {per_role_calls!r}"
        )
        assert any(c >= 2 for c in per_role_calls.values()), (
            f"Expected at least one role with >=2 calls. Per-role: {per_role_calls!r}"
        )

        # End-to-end: run is invalid, no results returned, no results.json on disk
        assert results == {}, (
            f"Mixed providers via real chat registry should invalidate run; "
            f"got keys: {list(results.keys())!r}. Per-role counts={per_role_calls!r}"
        )
        runs_root = tmp_path / "runs"
        if runs_root.exists():
            leftover = [p for p in runs_root.rglob("results.json")]
            assert not leftover, (
                f"Expected no results.json after real-mixed-chat invalidation, "
                f"found: {[str(p) for p in leftover]!r}"
            )
        # Clean up any leftover registry state for subsequent tests
        clear_provider_usage()
