#!/usr/bin/env bash
# Set the project up with uv: install uv if missing, sync the environment, seed .env.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found — installing it."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        echo "curl is required to install uv. See https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    # The installer drops uv in ~/.local/bin, which may not be on PATH yet.
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv $(uv --version | awk '{print $2}')"

# Creates .venv, installs the project editable plus the dev group, from uv.lock.
# Some machines sit behind TLS inspection that uv's bundled cert store rejects;
# retry against the system cert store before giving up (see docs/development.md).
if ! uv sync; then
    echo "uv sync failed — retrying with --native-tls"
    uv sync --native-tls
    echo "Tip: export UV_NATIVE_TLS=1 to make this the default for uv on this machine."
fi

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example — fill in any secrets."
fi

echo
echo "Done. Run things with 'uv run', no activation needed:"
echo "  uv run joel-nvk config"
echo "  uv run pytest"
