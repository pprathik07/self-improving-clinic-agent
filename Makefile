.PHONY: agent eval improve test clean setup

setup:
	uv sync --all-extras

agent:
	uv run python -m clinic_agent

eval:
	uv run python -m clinic_agent.evals

improve:
	uv run python -m clinic_agent.loop.improve

test:
	uv run pytest -v

clean:
	rm -rf runs/__pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
