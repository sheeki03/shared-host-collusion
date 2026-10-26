#!/usr/bin/env bash
# On a host with NO live harness runner: bring down orphaned inspect-shc_games-* compose projects only.
set -uo pipefail
if ps -eo args | grep -q "[h]arness.run"; then echo "a harness runner is alive: refusing"; exit 3; fi
projects=$(docker ps -a --format '{{.Label "com.docker.compose.project"}}' | grep '^inspect-shc_games-' | sort -u)
n=0
for p in $projects; do docker compose -p "$p" down --volumes --timeout 5 >/dev/null 2>&1 && n=$((n+1)); done
echo "brought down $n projects; inspect containers left: $(docker ps -a --format '{{.Names}}' | grep -c '^inspect-shc_games-')"
