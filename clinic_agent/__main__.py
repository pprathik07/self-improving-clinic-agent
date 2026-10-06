"""CLI entry point: python -m clinic_agent."""

import sys

from clinic_agent.agent.runner import run_interactive


def main() -> None:
    """Run the clinic scheduling agent in interactive mode."""
    policy_path = None
    if len(sys.argv) > 1:
        policy_path = sys.argv[1]
    run_interactive(policy_path)


if __name__ == "__main__":
    main()
