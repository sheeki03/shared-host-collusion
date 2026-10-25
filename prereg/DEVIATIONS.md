# Deviations and pilot-time changes

Times are IST, Sun 25 Oct 2026. Entries before the freeze changed the harness, not the confirmatory design; they are
listed so the frozen design can be read against them.

| ID | Time | Change | Reason | Effect on confirmatory games |
|---|---|---|---|---|
| D1 | ~21:58 | Provider settles a call whose final cost is pending (nanoGPT `x_nanogpt_pricing.isEstimate`) at max(provider estimate, token bound at pinned top-tier prices), never above the reservation | nanoGPT stopped returning `usage.cost` on some responses on 25 Oct; such calls settled at the full reservation and would exhaust any cap | Budget bookkeeping only (conservative); no effect on play |
| D2 | ~22:03 | Reservation bound changed from the full advertised context window to min(context, UTF-8 bytes + 32/message + 2,048) input tokens and max_tokens + 8,192 output tokens | A 97-second provider burst of 21 unbilled ReadErrors settled at $1–6 each and halted 9 of 14 probes on the local cap ($57 ledger vs $0.75 real, matched to the balance) | Budget bookkeeping only; the bound is still an upper bound for adversarial text |
| D3 | ~22:16 | Output caps raised: default 800; Gemini 3.1 Pro, Grok 4.7, Kimi K3.1 Pro 1,500; Qwen 3.8 Max Prime 2,000 (from 500 / 1,000) | Probe truncation: Gemini 11/27 calls (4/10 steps unparsed), Qwen 3/10 unparsed, Opus 6/26 truncated | Applies to all confirmatory games equally across arms |
| D4 | ~22:22 | Core games keep 40 rounds although the projection misses the 02:00 completion rule for the 30-round branch | Owner decision: two VPS added for capacity and the balance topped up; the pre-registered trimming order applies if the run is late | Design unchanged; completion risk accepted |
| D5 | ~22:30 | Games run on three machines, sharded by whole (model, scenario) stream; gated arms on the primary machine | Capacity | Host recorded per game and used as a sensitivity stratum |
