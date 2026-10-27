#!/usr/bin/env bash
# Convenience wrapper. The Python module can be run directly too:
#   .venv/bin/python -m ndb_host.tools.powerloss.harness --help
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../.." && pwd)"
PYTHON="${NEBULONDB_PYTHON:-$ROOT/.venv/bin/python}"

cd "$ROOT"
exec "$PYTHON" -m ndb_host.tools.powerloss.harness "$@"
