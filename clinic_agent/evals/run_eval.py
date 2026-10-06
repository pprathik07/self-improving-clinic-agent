"""Eval runner — loads scenarios, runs agent+simulator, scores, reports.

Entry point: python -m clinic_agent.evals
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from clinic_agent.agent.machine import is_terminal
from clinic_agent.agent.policy import load_policy, policy_hash, policy_to_prompt
from clinic_agent.agent.runner import TraceLogger, _build_system_prompt, _build_tool_definitions, _handle_tool_calls, _maybe_transition_from_user_input, run_turn
from clinic_agent.agent.state import AgentState, Message, SessionState
from clinic_agent.clinic.db import frozen_now, init_db, reset_db, seed_db
from clinic_agent.evals.judge import format_transcript, judge_transcript
from clinic_agent.evals.scenario_schema import Scenario
from clinic_agent.evals.scorers import ScoreReport, score_judge, score_state, score_trace
from clinic_agent.evals.simulator import build_sim_history, simulate_patient
from clinic_agent.llm import (
    FatalLLMError,
    chat_role_label,
    clear_provider_usage,
    snapshot_provider_usage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario loader
# ---------------------------------------------------------------------------

def load_scenarios(scenarios_dir: str | Path) -> list[Scenario]:
    """Load all scenario YAML files from a directory."""
    scenarios_dir = Path(scenarios_dir)
    scenarios = []
    for path in sorted(scenarios_dir.glob("*.yaml")):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        scenarios.append(Scenario(**data))
    return scenarios


# ---------------------------------------------------------------------------
# Single scenario run
# ---------------------------------------------------------------------------

def run_scenario(
    scenario: Scenario,
    policy_path: str,
    trace_dir: Path | None = None,
) -> tuple[ScoreReport, dict[str, Any]]:
    """Run a single scenario once and score it.

    Returns:
        (score_report, run_metadata)

    Raises:
        RuntimeError: On auth/permission errors (fail-fast)
    """
    # Fresh DB for each run
    conn = init_db(in_memory=True)
    seed_db(conn, scenario.seed_db)

    policy = load_policy(policy_path)
    policy_text = policy_to_prompt(policy)

    session = SessionState()
    trace = TraceLogger(
        trace_dir / f"{scenario.id}.jsonl" if trace_dir else None
    )

    # Start conversation
    session.state = AgentState.IDENTIFY
    session.messages.append(Message(role="user", content=scenario.opening))

    trace.log({
        "event": "conversation_start",
        "scenario_id": scenario.id,
        "opening": scenario.opening,
    })

    # Simulate conversation
    sim_history: list[tuple[str, str]] = []
    turn = 0
    infrastructure_error = None

    while not is_terminal(session.state) and turn < scenario.max_turns:
        turn += 1

        # Agent turn
        try:
            with chat_role_label("agent"):
                agent_response = run_turn(session, conn, policy_text, trace)
        except FatalLLMError:
            raise  # Abort entire eval — typed fatal from llm.py
        except RuntimeError as e:
            logger.error("Agent error on turn %d of %s: %s", turn, scenario.id, e)
            infrastructure_error = f"agent: {e}"
            trace.log({"event": "agent_error", "turn": turn, "error": str(e)})
            break
        except Exception as e:
            logger.error("Agent error on turn %d of %s: %s", turn, scenario.id, e)
            infrastructure_error = f"agent: {e}"
            trace.log({"event": "agent_error", "turn": turn, "error": str(e)})
            break

        trace.log({
            "event": "agent_turn",
            "turn": turn,
            "response": agent_response,
            "state": session.state.value,
        })

        if is_terminal(session.state):
            break

        # Patient simulator turn
        try:
            history = build_sim_history(sim_history)
            with chat_role_label("sim"):
                patient_response = simulate_patient(
                    persona=scenario.patient_persona,
                    agent_message=agent_response,
                    conversation_history=history,
                    turn=turn,
                    max_turns=scenario.max_turns,
                )
        except FatalLLMError:
            raise
        except RuntimeError as e:
            logger.error("Simulator error on turn %d of %s: %s", turn, scenario.id, e)
            infrastructure_error = f"simulator: {e}"
            patient_response = "I need to go. Goodbye."
        except Exception as e:
            logger.error("Simulator error on turn %d of %s: %s", turn, scenario.id, e)
            infrastructure_error = f"simulator: {e}"
            patient_response = "I need to go. Goodbye."

        sim_history.append((agent_response, patient_response))

        trace.log({
            "event": "patient_turn",
            "turn": turn,
            "response": patient_response,
        })

        session.messages.append(Message(role="user", content=patient_response))
        _maybe_transition_from_user_input(session, patient_response)

    trace.log({
        "event": "conversation_end",
        "final_state": session.state.value,
        "total_turns": turn,
    })

    # Score
    report = ScoreReport()

    # Layer 1: State checks
    report.state_checks = score_state(scenario, session, conn, trace.entries)

    # Layer 2: Trace checks
    report.trace_checks = score_trace(scenario, trace.entries)

    # Layer 3: LLM Judge
    judge_error = None
    if scenario.expect.judge:
        try:
            transcript_text = format_transcript(
                [{"role": m.role, "content": m.content, "tool_name": m.tool_name}
                 for m in session.messages]
            )
            with chat_role_label("judge"):
                judge_results = judge_transcript(transcript_text, scenario.expect.judge)
            report.judge_checks = score_judge(judge_results)
        except FatalLLMError:
            raise
        except RuntimeError as e:
            logger.error("Judge error for %s: %s", scenario.id, e)
            judge_error = str(e)
            infrastructure_error = infrastructure_error or f"judge: {e}"
        except Exception as e:
            logger.error("Judge error for %s: %s", scenario.id, e)
            judge_error = str(e)
            infrastructure_error = infrastructure_error or f"judge: {e}"

    # Metadata
    run_meta = {
        "scenario_id": scenario.id,
        "split": scenario.split,
        "tags": scenario.tags,
        "turns": turn,
        "final_state": session.state.value,
        "passed": report.passed and infrastructure_error is None,
        "error": infrastructure_error,
        "scores": report.summary if infrastructure_error is None else None,
    }

    return report, run_meta


# ---------------------------------------------------------------------------
# Full eval run
# ---------------------------------------------------------------------------

def run_eval(
    scenarios_dir: str = "clinic_agent/evals/scenarios",
    policy_path: str = "policy/policy_v1.yaml",
    k: int | None = None,
    split_filter: str | None = None,
) -> dict[str, Any]:
    """Run all scenarios k times each and aggregate results.

    Args:
        scenarios_dir: Path to scenario YAML files.
        policy_path: Path to the policy YAML to use.
        k: Runs per scenario (defaults to RUNS_PER_SCENARIO env var or 3).
        split_filter: If set, only run scenarios with this split (train/heldout).

    Returns:
        Full results dict with per-scenario and aggregate scores.
    """
    k = k or int(os.environ.get("RUNS_PER_SCENARIO", "3"))

    # Record LLM mode and model names
    llm_mode = "mock" if os.environ.get("MOCK_LLM", "0") == "1" else "real"
    agent_model = os.environ.get("AGENT_MODEL", "unset")
    judge_model = os.environ.get("JUDGE_MODEL", "unset")
    sim_model = os.environ.get("SIM_MODEL", "unset")
    reflector_model = os.environ.get("REFLECTOR_MODEL", "unset")

    # Print mode and models at start
    print(f"\n{'='*60}")
    print(f"LLM MODE: {llm_mode.upper()}")
    print(f"AGENT_MODEL: {agent_model}")
    print(f"JUDGE_MODEL: {judge_model}")
    print(f"SIM_MODEL: {sim_model}")
    print(f"REFLECTOR_MODEL: {reflector_model}")
    print(f"{'='*60}\n")

    # Real-run request with MOCK_LLM on is inconsistent — abort
    if os.environ.get("REQUIRE_REAL", "0") == "1" and llm_mode == "mock":
        print("ERROR: REQUIRE_REAL=1 but MOCK_LLM=1. Turn off MOCK_LLM for a real run.")
        return {}

    # In real mode, require all three models to be different
    if llm_mode == "real":
        if len({agent_model, judge_model, sim_model}) < 3:
            print(f"ERROR: In real mode, AGENT_MODEL, JUDGE_MODEL, and SIM_MODEL must all be different.")
            print(f"Current: agent={agent_model}, judge={judge_model}, sim={sim_model}")
            print(f"Set different models in .env or use MOCK_LLM=1 for testing.")
            return {}

    scenarios = load_scenarios(scenarios_dir)
    if split_filter:
        scenarios = [s for s in scenarios if s.split == split_filter]

    policy = load_policy(policy_path)
    p_hash = policy_hash(policy)

    # Reset the per-role provider usage tracker before this run
    clear_provider_usage()

    # Create run directory (mock runs go to runs/_mock/, real runs to runs/)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if llm_mode == "mock":
        run_dir = Path("runs") / "_mock" / timestamp
    else:
        run_dir = Path("runs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"EVAL RUN: {timestamp}")
    print(f"Policy: {policy_path} (hash: {p_hash})")
    print(f"Scenarios: {len(scenarios)} | Runs per scenario: {k}")
    print(f"{'='*60}\n")

    all_results: dict[str, list[dict]] = {}
    infrastructure_errors = 0

    for scenario in scenarios:
        print(f"  [{scenario.split.upper():7s}] {scenario.id}...", end=" ", flush=True)
        scenario_results = []

        for run_idx in range(k):
            try:
                report, run_meta = run_scenario(
                    scenario, policy_path,
                    trace_dir=trace_dir,
                )
                run_meta["run_index"] = run_idx
                scenario_results.append(run_meta)
                if run_meta.get("error"):
                    infrastructure_errors += 1
            except FatalLLMError as e:
                # Typed fatal from llm.py — abort entire run (real or mock)
                print(f"\n\nFATAL: {e}")
                print(f"Aborting run. No results.json written.")
                print(f"Fix your API key or set MOCK_LLM=1 for testing.")
                import shutil
                shutil.rmtree(run_dir, ignore_errors=True)
                return {}
            except RuntimeError as e:
                # Non-fatal infra RuntimeError — mark run, continue
                logger.error("Run %d of %s failed: %s", run_idx, scenario.id, e)
                scenario_results.append({
                    "scenario_id": scenario.id,
                    "run_index": run_idx,
                    "passed": False,
                    "error": str(e),
                })
                infrastructure_errors += 1
            except Exception as e:
                logger.error("Run %d of %s failed: %s", run_idx, scenario.id, e)
                scenario_results.append({
                    "scenario_id": scenario.id,
                    "run_index": run_idx,
                    "passed": False,
                    "error": str(e),
                })
                infrastructure_errors += 1

        pass_count = sum(1 for r in scenario_results if r.get("passed") and not r.get("error"))
        status = "PASS" if pass_count == k else f"{pass_count}/{k}"
        if any(r.get("error") for r in scenario_results):
            status = f"{status} (ERROR)"
        print(f"{status}")

        all_results[scenario.id] = scenario_results

    # Snapshot provider usage after ALL scenario runs are complete
    llm_providers_used = snapshot_provider_usage()

    # In real mode, invalidate if any role used more than one distinct provider
    if llm_mode == "real":
        mixed_roles = {role for role, provs in llm_providers_used.items() if len(provs) > 1}
        if mixed_roles:
            print(f"\nFATAL: Some LLM roles used more than one provider in a single real run. "
                  f"Mixed roles: {sorted(mixed_roles)} -> {llm_providers_used}")
            print("Run is invalid. No results.json written. "
                  "Set LLM_PROVIDER=openrouter or LLM_PROVIDER=gemini explicitly to avoid fallback mixing.")
            import shutil
            shutil.rmtree(run_dir, ignore_errors=True)
            return {}

    # In real mode, if any infrastructure errors occurred, mark run invalid
    if llm_mode == "real" and infrastructure_errors > 0:
        print(f"\nFATAL: {infrastructure_errors} infrastructure error(s) occurred in real mode.")
        print(f"Run is invalid. No results.json written.")
        import shutil
        shutil.rmtree(run_dir, ignore_errors=True)
        return {}

    # Aggregate
    results = _aggregate_results(
        all_results, scenarios, p_hash, timestamp, k,
        llm_mode, agent_model, judge_model, sim_model,
        llm_providers_used=llm_providers_used,
    )

    # Save
    results_path = run_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Human-readable failure packet (always generated from results + traces)
    try:
        from clinic_agent.evals.failures import write_failures_md
        write_failures_md(run_dir, results)
    except Exception as e:
        logger.warning("Failed to write failures.md: %s", e)

    # Print summary
    _print_summary(results)

    return results


def _aggregate_results(
    all_results: dict[str, list[dict]],
    scenarios: list[Scenario],
    p_hash: str,
    timestamp: str,
    k: int,
    llm_mode: str,
    agent_model: str,
    judge_model: str,
    sim_model: str,
    *,
    llm_providers_used: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Compute aggregate scores from per-scenario results."""
    scenario_map = {s.id: s for s in scenarios}

    per_scenario = {}
    train_pass = 0
    train_total = 0
    heldout_pass = 0
    heldout_total = 0

    for scenario_id, runs in all_results.items():
        # Error runs are excluded from pass rates (neither pass nor fail)
        scored_runs = [r for r in runs if not r.get("error")]
        pass_count = sum(1 for r in scored_runs if r.get("passed"))
        total_scored = len(scored_runs)
        pass_rate = pass_count / total_scored if total_scored else 0.0
        scenario = scenario_map.get(scenario_id)
        split = scenario.split if scenario else "unknown"

        per_scenario[scenario_id] = {
            "split": split,
            "tags": scenario.tags if scenario else [],
            "pass_count": pass_count,
            "total_runs": total_scored,
            "error_runs": len(runs) - total_scored,
            "pass_rate": pass_rate,
            "runs": runs,
        }

        if split == "train":
            train_pass += pass_count
            train_total += total_scored
        elif split == "heldout":
            heldout_pass += pass_count
            heldout_total += total_scored

    overall_pass = train_pass + heldout_pass
    overall_total = train_total + heldout_total

    return {
        "timestamp": timestamp,
        "policy_hash": p_hash,
        "k": k,
        "llm_mode": llm_mode,
        "agent_model": agent_model,
        "judge_model": judge_model,
        "sim_model": sim_model,
        # Per-role list of providers actually used in this run.
        # Mock runs produce {"agent":["mock"],"sim":["mock"],"judge":["mock"]} if those roles called chat().
        # Real runs must have exactly one provider per role (e.g., {"agent":["gemini"],...}),
        # or the run was already invalidated before reaching _aggregate_results.
        "llm_providers_used": llm_providers_used or {},
        "overall": {
            "pass_rate": overall_pass / overall_total if overall_total else 0.0,
            "passed": overall_pass,
            "total": overall_total,
        },
        "train": {
            "pass_rate": train_pass / train_total if train_total else 0.0,
            "passed": train_pass,
            "total": train_total,
        },
        "heldout": {
            "pass_rate": heldout_pass / heldout_total if heldout_total else 0.0,
            "passed": heldout_pass,
            "total": heldout_total,
        },
        "per_scenario": per_scenario,
    }


def _print_summary(results: dict[str, Any]) -> None:
    """Print a formatted results summary."""
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Scenario':<30s} {'Split':<8s} {'Pass Rate':<12s}")
    print(f"{'-'*30} {'-'*8} {'-'*12}")

    for scenario_id, data in results["per_scenario"].items():
        rate = f"{data['pass_count']}/{data['total_runs']}"
        print(f"{scenario_id:<30s} {data['split']:<8s} {rate:<12s}")

    print(f"{'-'*50}")
    overall = results["overall"]
    print(f"{'Overall':<30s} {'':8s} {overall['passed']}/{overall['total']} ({overall['pass_rate']:.0%})")
    train = results["train"]
    print(f"{'Train':<30s} {'':8s} {train['passed']}/{train['total']} ({train['pass_rate']:.0%})")
    heldout = results["heldout"]
    print(f"{'Heldout':<30s} {'':8s} {heldout['passed']}/{heldout['total']} ({heldout['pass_rate']:.0%})")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for running evals."""
    load_dotenv()

    policy_path = sys.argv[1] if len(sys.argv) > 1 else "policy/policy_v1.yaml"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else None

    logging.basicConfig(level=logging.WARNING)
    run_eval(policy_path=policy_path, k=k)


if __name__ == "__main__":
    main()
