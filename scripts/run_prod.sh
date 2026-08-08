#!/usr/bin/env bash
# Headless Streamlit (local or when systemd is unavailable).
# On arya production, prefer: systemctl start queuescore
# Usage (from repo root): ./scripts/run_prod.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8501}"
ADDRESS="${ADDRESS:-0.0.0.0}"

if [[ -d "$ROOT/.venv" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  :
else
  echo "warning: no .venv found — using system Python ($(command -v python3))" >&2
fi

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

exec streamlit run src/app.py \
  --server.port "$PORT" \
  --server.address "$ADDRESS" \
  --server.headless true \
  --browser.gatherUsageStats false
