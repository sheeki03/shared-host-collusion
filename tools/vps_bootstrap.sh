#!/usr/bin/env bash
# Bootstrap a Linux VPS (Ubuntu 22.04/24.04) as an extra game runner for this study.
#
# Run on the VPS as a sudo-capable user:  bash vps_bootstrap.sh <git-commit>
# It installs Docker (Ubuntu packages) and uv, clones this repo at the given commit, installs the pinned Python
# environment, and runs the $0 mock benchmark as a gate. It never makes a paid model call and never touches the API
# key; the owner copies ~/.nanogpt-api-key (chmod 600) separately. Ungated arms only: tirith-gated arms run on the
# primary machine, so the pinned tirith binary is not needed here.
set -euo pipefail

COMMIT=${1:?usage: vps_bootstrap.sh <git-commit>}
REPO=https://github.com/sheeki03/shared-host-collusion.git
DIR=$HOME/shared-host-collusion

echo "== packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io git pipx ca-certificates >/dev/null
if ! docker compose version >/dev/null 2>&1; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker-compose-v2 >/dev/null \
    || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker-compose-plugin >/dev/null
fi
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
pipx install uv >/dev/null 2>&1 || true
export PATH=$HOME/.local/bin:$PATH

echo "== repo at $COMMIT"
if [ ! -d "$DIR/.git" ]; then git clone -q "$REPO" "$DIR"; fi
git -C "$DIR" fetch -q origin
git -C "$DIR" checkout -q --detach "$COMMIT"

echo "== python env"
cd "$DIR"
uv venv -q -p 3.13 .venv
uv pip install -q --python .venv/bin/python -r requirements-lock.txt

echo "== host"
nproc; free -g | head -2; df -h "$HOME" | tail -1
sg docker -c "docker info --format 'docker {{.ServerVersion}} cpus {{.NCPU}} mem {{.MemTotal}}'"

echo "== mock gate (no paid calls)"
sg docker -c ".venv/bin/python tools/bench_offline.py run-mock --confirm-docker --games 6 --rounds 4 --concurrency 6"
echo "BOOTSTRAP OK $(hostname) $(git rev-parse --short HEAD)"
