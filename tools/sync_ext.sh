#!/usr/bin/env bash
# Read-only pull of every host's runs/ext (extension logs, ledgers, runner output), ext shard files and calibration
# output to runs/ext-remote/<host>/ (a tar stream over ssh; the instabox hosts have no rsync). Final pulls happen
# after the runners exit, so no .eval copy is torn; a copy taken while a runner writes may be.
# Each stream is unpacked into a fresh temp directory and merged into runs/ext-remote/<host>/ (nothing local is
# deleted) only if both ends of the pipe exit 0. A failed host prints FAILED, the other hosts are still tried, and
# the script exits 1 at the end.
# usage: tools/sync_ext.sh [--verify]
#   --verify copies nothing: sha256 of the same files on each host (read-only find + sha256sum) and under
#            runs/ext-remote/<host>/; lists files only remote, only local and hash mismatches; exits 1 on any.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=sync
case "${1:-}" in "") ;; --verify) mode=verify ;; *) echo "usage: $0 [--verify]" >&2; exit 2 ;; esac
# the pulled set, relative to ~/shared-host-collusion/runs on the host; the remote command fails if it is empty
REMOTE='cd ~/shared-host-collusion/runs && p=$(ls -d ext shards/*ext*.json calibration 2>/dev/null);
  [ -n "$p" ] || { echo "nothing to pull: no ext, shards/*ext*.json or calibration in $PWD" >&2; exit 3; }'
SSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
if command -v sha256sum >/dev/null; then SHA=sha256sum; else SHA='shasum -a 256'; fi
TAB=$(printf '\t')
failed="" tmp=""
trap 'if [ -n "$tmp" ]; then rm -rf "$tmp"; fi' EXIT; trap 'trap "" PIPE; exit 1' HUP INT PIPE TERM

local_sums() {   # the same set under runs/ext-remote/<host>/ (empty if nothing was pulled)
  [ -d "runs/ext-remote/$1" ] || return 0
  ( cd "runs/ext-remote/$1" || exit 1; p=$(ls -d ext shards/*ext*.json calibration 2>/dev/null) || true
    [ -z "$p" ] || find $p -type f -exec $SHA {} + )
}
by_path() {      # "<hash>  <path>" lines -> "<path>\t<hash>", sorted by path
  awk '{ print substr($0, 67) "\t" substr($0, 1, 64) }' "$1" | LC_ALL=C sort
}
list() { [ -z "$2" ] || printf '%s\n' "$2" | sed "s|^|  $1: |"; }

while read -r name dest; do
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/sync_ext.$name.XXXXXX")
  if [ "$mode" = sync ]; then
    set +e
    "${SSH[@]}" "$dest" "$REMOTE"' && tar cf - $p' | tar xf - -C "$tmp"
    st=("${PIPESTATUS[@]}")
    set -e
    if [ "${st[0]}" -ne 0 ] || [ "${st[1]}" -ne 0 ]; then
      echo "$name: FAILED (ssh/remote tar exit ${st[0]}, local tar exit ${st[1]}); runs/ext-remote/$name not changed"
      failed="$failed $name"
    elif mkdir -p "runs/ext-remote/$name" && cp -pR "$tmp"/* "runs/ext-remote/$name/"; then
      echo "$name: $(find "$tmp" -type f | wc -l | tr -d ' ') files pulled," \
        "$(ls "runs/ext-remote/$name"/ext/logs-*/*.eval 2>/dev/null | wc -l | tr -d ' ') eval files"
    else
      echo "$name: FAILED merging the pull into runs/ext-remote/$name (it may be partly updated)"
      failed="$failed $name"
    fi
  else
    rc=0
    "${SSH[@]}" "$dest" "$REMOTE"' && find $p -type f -exec sha256sum {} +' > "$tmp/remote.sha" || rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "$name: FAILED (ssh/remote find+sha256sum exit $rc)"
      failed="$failed $name"
    elif ! local_sums "$name" > "$tmp/local.sha"; then
      echo "$name: FAILED (local find+$SHA under runs/ext-remote/$name)"
      failed="$failed $name"
    else
      by_path "$tmp/remote.sha" > "$tmp/r"; by_path "$tmp/local.sha" > "$tmp/l"
      cut -f1 "$tmp/r" > "$tmp/rp"; cut -f1 "$tmp/l" > "$tmp/lp"
      only_r=$(LC_ALL=C comm -23 "$tmp/rp" "$tmp/lp")
      only_l=$(LC_ALL=C comm -13 "$tmp/rp" "$tmp/lp")
      LC_ALL=C join -t "$TAB" "$tmp/r" "$tmp/l" > "$tmp/both"
      differ=$(awk -F'\t' '$2 != $3 { print $1 }' "$tmp/both")
      same=$(awk -F'\t' '$2 == $3' "$tmp/both" | wc -l | tr -d ' ')
      counts="remote $(wc -l < "$tmp/r" | tr -d ' ') files, local $(wc -l < "$tmp/l" | tr -d ' '), identical $same"
      if [ -z "$only_r$only_l$differ" ]; then
        echo "$name: OK, $counts"
      else
        echo "$name: DIFFERS, $counts"
        list "only remote" "$only_r"; list "only local" "$only_l"; list "hash mismatch" "$differ"
        failed="$failed $name"
      fi
    fi
  fi
  rm -rf "$tmp"; tmp=""
done <<HOSTS
instabox instabox
instabox2 instabox2
instabox3 instabox3
vps1 root@161.97.142.53
HOSTS
if [ -n "$failed" ]; then
  echo "FAILED ($mode):$failed" >&2
  exit 1
fi
