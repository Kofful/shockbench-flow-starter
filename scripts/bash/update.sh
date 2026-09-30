#!/usr/bin/env bash
# `make update`: pull the organisers' changes (a new wheel in vendor/, fixes to the examples and tools), relock and
# resync. A conflict in uv.lock alone (your `mine` dependencies against a new wheel) is resolved by taking theirs and
# relocking, which puts your dependencies back; a conflict in any other file stops, for you to resolve. The rl extra
# stays installed when it was. Extra arguments go to `git pull` (`make update ARGS="upstream main"`).
set -euo pipefail

if ! git pull --no-rebase --ff --no-edit "$@"; then
  conflicted="$(git diff --name-only --diff-filter=U)"
  if [ "$conflicted" != "uv.lock" ]; then
    [ -n "$conflicted" ] || conflicted="(no file: see the message of git above)"
    echo "make update: git pull stopped with conflicts in: $conflicted" >&2
    echo "  resolve them (or 'git merge --abort'), then run make update again" >&2
    exit 1
  fi
  echo "make update: uv.lock conflicts: taking the organisers' and relocking with your dependencies"
  git checkout --theirs uv.lock
  uv lock --upgrade-package shockbench-flow
  git add uv.lock
  git commit --no-edit
else
  uv lock --upgrade-package shockbench-flow # re-read from vendor/; your other locked versions stay
fi

rl=""
if uv run --no-sync python -c "import stable_baselines3" >/dev/null 2>&1; then
  rl="--extra rl" # keep the rl extra: a plain `uv sync` would remove it
fi
# shellcheck disable=SC2086 # $rl is empty or two words
uv sync $rl
uv run sbf version
if ! git diff --quiet -- uv.lock; then
  echo "make update: uv.lock changed (your dependencies relocked against the new wheel): commit it"
fi
