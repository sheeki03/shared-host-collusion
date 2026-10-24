# Step 3 readiness review — 25 Oct 2026, ~03:09 IST

## Decision

**GO to begin Step 3 implementation and offline self-tests. NO-GO to the paid pilot or confirmatory runs until the gates below pass.** The Step 2 spike demonstrates that Inspect can run two containers, expose a shared volume in both directions, isolate private volumes, and save an evaluation log. Its combined shared-sample scorer did not pass because one price was unparseable. The current code is a spike and prototype, not the finished v6.4 harness.

This review did not run new model calls, install dependencies, or alter experimental data. The available secret scan found no leaks in the three reachable commits. It does not prove that secrets are absent from future logs or containers.

## Required gates before Step 4 pilot

### R1 — Critical: global spending cap is unproven

The raw provider response has `usage.cost`, but the saved Inspect usage has `total_cost=None`; [spike_task.py](spike/spike_task.py#L70) also records only the first generate call when a second occurs. Inspect's configured [cost limit](https://inspect.aisi.org.uk/setting-limits.html) is per sample, not a global exact bill cap. At 20–30 concurrent games, post-call accounting alone can exceed a limit through in-flight calls.

**Acceptance:** surface or reconcile every provider charge, including retries; atomically reserve a conservative maximum before each call; release/reconcile after completion; fail closed on missing cost, ledger error, or exhaustion. Test simultaneous calls, crashes, retries, and the agreed cap. If exact charges remain unavailable through Inspect, call the cap conservative/estimated and enforce it using worst-case token-price reservations.

### R2 — High: spike did not pass its full game scorer

The shared sample completed but scored zero because one price parsed as `None`; the private sample scored one. The shared/private nonce and network-A checks still support the Inspect route. [The spike parser and scorer](spike/spike_task.py#L27) do not yet provide a valid game outcome for every turn.

**Acceptance:** schema-validate JSON and finite, bounded prices; retry once with separately logged usage; make a first-round fallback explicit; flag/exclude failed turns by the pre-registered rule. A smoke game must pass the actual scorer before any pilot result is treated as a positive control.

### R3 — High: mutating shell calls must be single-execution

[The spike helper](spike/spike_task.py#L43) uses Inspect `sandbox.exec` without `timeout_retry=False`. Inspect can retry timed-out commands, which can duplicate writes and contaminate the trajectory. [The shell runner](spike/run_spike.sh#L3) also pipes through `tail` without `pipefail`, so it can hide an Inspect failure.

**Acceptance:** disable timeout retry for all agent-controlled commands, test write-then-timeout behavior and surviving child processes, and make the runner fail on every failed stage. Preserve command exit status and full research-side output; separately cap what agents see.

### R4 — High: harden the container and shared bulletin boundary

The spike [Compose services](spike/compose-shared.yaml#L3) have no network and command execution uses non-root UIDs, but the service process starts as root, the volume is mode 1777, and the image is not yet pinned by digest. Only firm A received a dynamic network probe. A writable shared directory can also exhaust the 12 GB free disk or tamper with a bulletin if its ownership and update path are wrong.

**Acceptance:** verify both firms have no network, host keys, Docker socket or host-secret mount; enforce non-root agent commands, `no-new-privileges`, appropriate capability drops and writable-path limits; make the bulletin root-owned, not agent-writable, symlink-safe, and restricted to completed rounds. Bound per-game file/disk use and stop/clean up on exhaustion. Test concurrent shared and isolated games. See [Docker Compose controls](https://docs.docker.com/reference/compose-file/services/).

### R5 — High: independent trace labels cannot come from the old snapshot parser

[The prototype](harness/sandbox.py#L76) keeps only the first 2 KB of a file and parses `@@FILE` markers in agent-controlled text; a crafted file can forge apparent paths or ownership. UID ownership is not proof of the last writer, and round-end snapshots miss temporary files or suppressed-output reads.

**Acceptance:** implement structured per-command provenance and content hashes without parsing attacker-controlled delimiters. Test newline and marker filenames/content, overwrites, create-read-delete, symlinks, filename-only transfer, and canary crossing. Mark reads that instrumentation cannot establish as unknown rather than negative. Never derive labels from the tirith verdict being evaluated.

### R6 — High: the Step 3 statistical wording conflicts with the plan

The pasted checklist requests “exact permutation tests.” The local study plan v6.4 explicitly withdrew a simple across-round exact shuffle because agents adapt to earlier rounds. The current [economics prototype](harness/econ.py#L25) is symmetric only; its information-pooling simulation is not a unilateral-disclosure proof for the revised truncated, two-cost game.

**Acceptance:** precompute matched exogenous schedules with separate RNG streams; implement the two-cost Nash and joint-profit solvers and independent benchmark checks; verify dollar inbox units and the completed-round-only bulletin. Make matched-seed game-level SHR-versus-ISO contrasts primary. Treat pooled β with game-clustered uncertainty, adaptive-null simulations and first-mover placebos as supporting analysis. The private-cost incentive is a numerical, model-specific hypothesis until its payoff check passes. Welfare's primary comparison should include **all** matched seed pairs; transfer-positive subsets are descriptive only.

### R7 — Medium: preregistration and scope need exact wording

Build **five economic engines with ten scenario framings**, not ten unrelated engines. Each engine needs a payoff, information and benchmark specification, a suitable randomised signal, and passing tests before admission. Draft endpoints, seeds, exclusions and pilot-trigger rules before the paid pilot; exclude all pilot seeds, disclose any pilot-driven changes, and freeze/publish the final registration before confirmatory calls.

## Stack and Step 2 evidence

Python plus Inspect is a sound choice for this research harness; tirith stays a pinned Rust binary. Use Inspect's [asynchronous parallelism](https://inspect.aisi.org.uk/parallelism.html), not the prototype's synchronous `subprocess.run`, for 20–30 concurrent games. The four tiny raw model probes (1.4–3.7 s) do not establish full-game throughput; the two-sample Inspect spike took 2:36. Measure throughput in the full-length/concurrency pilot.

Local `main` matched `origin/main` at review time, with three reachable commits using the configured noreply identity. The saved tirith hash matches the recorded binary. A redacted Gitleaks scan of those three commits reported no leaks. The secsuite offline detector could not start because its Go dependencies were not cached; no broad scanner result is claimed.
