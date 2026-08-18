#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the Eleusis LLM Benchmark.
# Installs uv (if missing) and syncs the locked virtualenv from uv.lock.
set -euo pipefail

# Ensure a user-writable bin dir is on PATH for this run and for future shells.
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# The installer drops a PATH shim here; source it so `uv` resolves immediately.
if [ -f "$HOME/.local/bin/env" ]; then
  # shellcheck disable=SC1091
  . "$HOME/.local/bin/env"
fi

uv --version

# Create/refresh the project's .venv from the committed lockfile. Idempotent:
# a second run is a no-op when nothing changed.
uv sync --frozen

echo "Eleusis environment ready."
