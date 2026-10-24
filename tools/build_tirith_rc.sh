#!/bin/bash
# Build tirith 0.5.0 RC (PR #278 head 8856dc60) in a separate detached worktree, offline, reusing the existing
# target dir as a build cache. Copies the binary to a pinned path and records its sha256.
set -eu
SRC=/Users/home/tirith-next-cycle
WT=/Users/home/tirith-rc-8856dc60
OUT=/Users/home/shared-host-collusion/bin
mkdir -p "$OUT"
if [ ! -d "$WT" ]; then
  git -C "$SRC" worktree add --detach "$WT" 8856dc60
fi
cd "$WT"
echo "worktree HEAD: $(git rev-parse HEAD)"
start=$(date +%s)
CARGO_TARGET_DIR="$SRC/target" cargo build --release --locked --offline --bin tirith
cp "$SRC/target/release/tirith" "$OUT/tirith-0.5.0-rc-8856dc60"
echo "build secs: $(( $(date +%s) - start ))"
"$OUT/tirith-0.5.0-rc-8856dc60" --version
shasum -a 256 "$OUT/tirith-0.5.0-rc-8856dc60" | tee "$OUT/tirith-0.5.0-rc-8856dc60.sha256"
