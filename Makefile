# Participant commands. AGENT is an agent's name (a folder of agents/, else template, random or heuristic) or a
# path; TASK is tiny (the default), small (the public board's network) or full (the private board's).
#   make new-agent NAME=mine    make check AGENT=mine TASK=small    make compare A=mine B=template
AGENT ?= template
A ?= $(AGENT)
B ?= template
TASK ?=
NAME ?=
ARGS ?=
ZIP ?= outputs/$(notdir $(patsubst %/,%,$(AGENT))).zip
T = $(if $(TASK),--task=$(TASK))

install:
	uv sync
install_rl:
	uv sync --extra rl
update:
	bash scripts/bash/update.sh $(ARGS)
new-agent:
	@test -n "$(NAME)" || { echo "usage: make new-agent NAME=<name>"; exit 2; }
	@test ! -e agents/$(NAME) || { echo "agents/$(NAME) exists already"; exit 1; }
	mkdir -p agents/$(NAME)
	cp src/sbf_starter/agents/template/agent.py agents/$(NAME)/agent.py
	@echo "agents/$(NAME)/agent.py: edit act, then make check AGENT=$(NAME) and make evaluate AGENT=$(NAME)"
quickstart:
	uv run python scripts/python/01_quickstart.py $(if $(TASK),task=$(TASK))
evaluate:
	uv run sbf evaluate $(AGENT) $(T)
evaluate_quick:
	uv run sbf evaluate $(AGENT) --quick $(T)
compare:
	uv run sbf compare $(A) $(B) $(T)
check:
	uv run sbf check $(AGENT) $(T)
pack:
	uv run sbf pack $(AGENT) --out=$(ZIP)
upload:
	uv run sbf upload $(ZIP)
upload_dry_run:
	uv run sbf upload $(ZIP) --dry_run
status:
	uv run sbf status
runs:
	uv run sbf runs
version:
	uv run sbf version
test:
	uv run pytest -n 3
lint:
	uv run ruff check --fix .
	uv run ruff format .

.PHONY: install install_rl update new-agent quickstart evaluate evaluate_quick compare check pack upload upload_dry_run \
	status runs version test lint

# The template's uv and pre-commit targets
uv_install_deps:
	uv sync --frozen --all-extras
uv_install_deps_with_upgrade:
	uv sync --all-extras -U
uv_show_deps:
	uv pip list
uv_show_deps_tree:
	uv tree
uv_create_venv:
	uv venv --python 3.13

pre_commit_install: .pre-commit-config.yaml
	uv run pre-commit install
pre_commit_run: .pre-commit-config.yaml
	uv run pre-commit run --all-files
pre_commit_rm_hooks:
	uv run pre-commit --uninstall-hooks
