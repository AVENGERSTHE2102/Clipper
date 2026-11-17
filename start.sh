#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$PROJECT_ROOT/backend/.venv/bin/python" ]]; then
    printf 'Missing backend virtual environment. Run: cd "%s/backend" && uv sync\n' "$PROJECT_ROOT" >&2
    exit 1
fi
exec "$PROJECT_ROOT/backend/.venv/bin/python" "$PROJECT_ROOT/scripts/start-local.py" "$@"
