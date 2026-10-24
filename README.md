# Shared-host collusion study (Apart AI Collusion Research Sprint, Oct 2026)

Do co-located LLM agents that work for competing firms collude through the machine they share? This
repository holds all sprint work: harness, market engines, pre-registration, analysis and paper.

**Timeline.** Built for the Apart Research AI Collusion Research Sprint (23–25 Oct 2026; submissions close
25 Oct 23:59 AoE). Work started on 25 Oct 2026, ~00:58 IST. The commit history is the record of what was done
and when. The pre-registration is committed before any confirmatory run.
The study design is published in `prereg/` before any confirmatory run.

## Layout

| Path | What |
|---|---|
| `STATUS.md` | Step-by-step progress log with timestamps (read this to track the work) |
| `harness/` | Core game: `econ.py` (logit Bertrand benchmarks), `sandbox.py` (Docker pair), game loop, prompts, gate |
| `engines/` | Cross-domain market modules (auction, reporting, pricing re-framings, market making, bargaining) |
| `spike/` | 90-minute Inspect go/no-go spike |
| `prereg/` | `PREREGISTRATION.md` and its frozen hash manifest |
| `runs/pilot/`, `runs/full/` | Run logs (pilot seeds are never used in final estimates) |
| `analysis/` | Statistics, monitor replay, figures |
| `paper/` | Report drafts and figures for the PDF |
| `tools/` | Budget, model list, balance, Firecrawl search, preflight, tirith RC build |
| `bin/` | Pinned tirith 0.5.0 RC binary (PR #278 head `8856dc60`) and its sha256 |

## Preexisting vs sprint work
- **Preexisting (written before the sprint):** tirith (github.com/sheeki03/tirith, AGPL-3.0), used unmodified as an
  instrument. The binary is built from PR #278 head `8856dc60` in a separate worktree (`~/tirith-rc-8856dc60`).
- **Third-party (credited):** the Fish-style prompt starter by Lucia Sauer / Poseidon Research (MIT), UK AISI Inspect
  (MIT), ControlArena monitor prompt (MIT), MUTE nonce-test design (MIT).
- **Sprint work (25–26 Oct 2026):** everything else in this folder.
