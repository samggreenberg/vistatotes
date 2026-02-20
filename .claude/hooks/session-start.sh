#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Install lightweight dev tools only — linter and formatter.
# Heavy dependencies (PyTorch, transformers, etc.) are installed lazily
# by ensure-test-deps.sh the first time tests or the app are run.
pip install ruff -q
