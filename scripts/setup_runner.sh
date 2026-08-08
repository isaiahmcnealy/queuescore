#!/usr/bin/env bash
# Configure the repo-scoped self-hosted runner on arya.
# Get a registration token from:
#   GitHub → isaiahmcnealy/queuescore → Settings → Actions → Runners → New self-hosted runner
#
# Usage (on arya as root):
#   REGISTRATION_TOKEN=AAAA ./scripts/setup_runner.sh
set -euo pipefail

RUNNER_DIR="${RUNNER_DIR:-/opt/actions-queuescore}"
REPO_URL="${REPO_URL:-https://github.com/isaiahmcnealy/queuescore}"
NAME="${RUNNER_NAME:-arya-queuescore}"
LABELS="${RUNNER_LABELS:-queuescore}"
RUN_USER="${RUNNER_USER:-actions-runner}"

if [[ -z "${REGISTRATION_TOKEN:-}" ]]; then
  echo "error: set REGISTRATION_TOKEN to a fresh GitHub runner registration token" >&2
  exit 1
fi

if [[ ! -x "$RUNNER_DIR/config.sh" ]]; then
  echo "error: runner not extracted at $RUNNER_DIR" >&2
  exit 1
fi

if [[ -f "$RUNNER_DIR/.runner" ]]; then
  echo "runner already configured at $RUNNER_DIR (.runner exists); remove it first to reconfigure" >&2
  exit 1
fi

# config.sh refuses EUID 0 unless RUNNER_ALLOW_RUNASROOT is set; prefer dedicated user.
if [[ ! -f "$RUNNER_DIR/svc.sh" ]]; then
  echo "error: svc.sh missing in $RUNNER_DIR (re-extract the Actions runner tarball)" >&2
  exit 1
fi

chown -R "$RUN_USER:$RUN_USER" "$RUNNER_DIR"

sudo -u "$RUN_USER" -H bash -lc "
  cd '$RUNNER_DIR'
  ./config.sh --unattended \
    --url '$REPO_URL' \
    --token '$REGISTRATION_TOKEN' \
    --name '$NAME' \
    --labels '$LABELS' \
    --work _work
"

cd "$RUNNER_DIR"
./svc.sh install "$RUN_USER"
./svc.sh start
./svc.sh status
echo "runner $NAME online — label: $LABELS (user: $RUN_USER)"
