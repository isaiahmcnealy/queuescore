#!/usr/bin/env bash
# Kill and restart the QueueScore Streamlit service (used by deploy + manual ops).
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/queuescore}"
SERVICE="${SERVICE:-queuescore}"

if [[ ! -d "$APP_DIR" ]]; then
  echo "error: app dir not found: $APP_DIR" >&2
  exit 1
fi

if command -v systemctl >/dev/null 2>&1 && { systemctl cat "$SERVICE" >/dev/null 2>&1 || sudo -n systemctl cat "$SERVICE" >/dev/null 2>&1; }; then
  if systemctl restart "$SERVICE" 2>/dev/null; then
    :
  else
    sudo -n systemctl restart "$SERVICE"
  fi
  systemctl --no-pager --full status "$SERVICE" 2>/dev/null | head -20 \
    || sudo -n systemctl --no-pager --full status "$SERVICE" | head -20
else
  # Fallback when systemd unit is missing (local/dev): kill by port, then run_prod.
  PORT="${PORT:-8501}"
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" 2>/dev/null || true
  else
    pkill -f "streamlit run src/app.py" 2>/dev/null || true
  fi
  sleep 1
  cd "$APP_DIR"
  nohup ./scripts/run_prod.sh >>"$APP_DIR/streamlit.log" 2>&1 &
  echo "started streamlit (pid $!) → log $APP_DIR/streamlit.log"
fi
