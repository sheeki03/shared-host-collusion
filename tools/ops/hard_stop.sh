#!/usr/bin/env bash
# Stop the confirmatory runner on this machine (TERM, then KILL) and remove only Inspect game containers/volumes.
# Never touches containers outside the inspect-shc_games-* compose projects.
pids=$(pgrep -f "[Pp]ython -m harness.run --plan full" || true)
[ -n "$pids" ] && kill -TERM $pids 2>/dev/null
for i in $(seq 1 15); do pgrep -f "[Pp]ython -m harness.run --plan full" >/dev/null || break; /bin/sleep 2; done
pids=$(pgrep -f "[Pp]ython -m harness.run --plan full" || true)
[ -n "$pids" ] && kill -KILL $pids 2>/dev/null && echo "killed $pids"
/bin/sleep 2
projects=$(docker ps -a --format '{{.Label "com.docker.compose.project"}}' | grep '^inspect-shc_games-' | sort -u)
n=0
for p in $projects; do
  ids=$(docker ps -aq --filter "label=com.docker.compose.project=$p")
  [ -n "$ids" ] && docker rm -f $ids >/dev/null 2>&1
  vols=$(docker volume ls -q --filter "label=com.docker.compose.project=$p")
  [ -n "$vols" ] && docker volume rm -f $vols >/dev/null 2>&1
  n=$((n+1))
done
echo "runner alive: $(pgrep -f '[Pp]ython -m harness.run --plan full' | wc -l | tr -d ' ') | removed projects: $n | containers left: $(docker ps -q | wc -l | tr -d ' ')"
docker ps --format '{{.Names}}' | grep -vE '^inspect-' | head -5
