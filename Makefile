# Shortcuts for setting up and testing. Everything else is `uv run sbf <command>` (see `uv run sbf --help`).

install:
	uv sync
install_rl:
	uv sync --extra rl
test:
	uv run pytest -n 3
lint:
	uv run ruff check --fix .
	uv run ruff format .

.PHONY: install install_rl test lint
