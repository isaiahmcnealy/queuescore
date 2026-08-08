#!/usr/bin/env bash
# QueueScore app control: start | stop | restart | status | logs
# Usage: ./run.sh start   (then open http://localhost:8501)
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8501}"
PIDFILE=".app.pid"
LOG="/tmp/queuescore.log"
STREAMLIT=".venv/bin/streamlit"

running() {
  [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

start() {
  if [ ! -x "$STREAMLIT" ]; then
    echo "✗ $STREAMLIT not found. Create the venv first:"
    echo "    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
  fi
  if running; then
    echo "already running (pid $(cat "$PIDFILE")) → http://localhost:$PORT"
    return 0
  fi
  "$STREAMLIT" run src/app.py --server.port "$PORT" --server.headless true >"$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  sleep 2
  if running; then
    echo "✓ started (pid $(cat "$PIDFILE")) → http://localhost:$PORT"
    echo "  logs: $LOG   stop: ./run.sh stop"
  else
    echo "✗ failed to start — see $LOG"; tail -n 15 "$LOG"; exit 1
  fi
}

stop() {
  local hit=0
  if running; then
    kill "$(cat "$PIDFILE")" 2>/dev/null && hit=1
  fi
  rm -f "$PIDFILE"
  # Always sweep the port too, in case an untracked instance is also on it.
  if lsof -ti "tcp:$PORT" >/dev/null 2>&1; then
    lsof -ti "tcp:$PORT" | xargs kill 2>/dev/null && hit=1
  fi
  [ "$hit" = 1 ] && echo "✓ stopped (port $PORT free)" || echo "not running"
}

status() {
  if running; then
    echo "running (pid $(cat "$PIDFILE")) → http://localhost:$PORT"
  else
    echo "stopped"
  fi
}

case "${1:-}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; sleep 1; start ;;
  status)  status ;;
  logs)    tail -f "$LOG" ;;
  *) echo "Usage: ./run.sh {start|stop|restart|status|logs}"; exit 1 ;;
esac
