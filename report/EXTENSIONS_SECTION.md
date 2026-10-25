<!--
Section for report/REPORT.md, to be merged by the author as Section 4.6 (after 4.5 Exploratory: non-storage channel
attempts, before 5 Discussion). Design, predictions and the analysis plan are pre-declared in prereg/EXTENSIONS.md;
no extension game enters a confirmatory estimate. Placeholder sources (key list: analysis/EXTENSION_KEYS.md):
ext.* keys and TABLE:ext_* from analysis/extensions.py (<out>/ext_results.json, <out>/tables/); signal.* keys from
analysis/signalling.py run on the confirmatory cohort (<out>/signal_results.json). tools/fill_report.py reads one
results file and the tables folder next to it, so merge the three results files (and copy the ext_* tables) before
filling this section. As in REPORT.md, every placeholder in this file must be a key that the pipeline writes; the
comments below name keys in plain text so that they are not substituted. The only literal numbers are design
constants from prereg/EXTENSIONS.md. HTML comments marked "if ..." give the wording to keep or delete once each
estimate is known. Delete all comments before export.
-->

### 4.6 Extensions: a replication and four exploratory analyses

We declared five extensions in `prereg/EXTENSIONS.md` after the confirmatory freeze and before any extension game
ran. They ran after the confirmatory grid, in grids of their own, and no extension game enters a confirmatory
estimate. E1, E2 and E4 used new seeds; E3 reused the seeds of the confirmatory games it is matched to; E5 used no
new games. E1 re-ran P1–P3 on new seeds with the frozen estimators. E2–E5 are exploratory. None of them is part of
the Holm family, so every p-value below is unadjusted. Nothing in this section changes a confirmatory result.

**E1, replication.** E1 re-ran the eight P1–P3 arms with GLM 5.3, DeepSeek V4 Pro and GPT-6.1 Sol on fresh market
draws (seeds 2000–2005; {{ext.meta.rep.n_complete}} of 144 planned games complete). Each contrast had at most 18
matched units. The P1 interaction on the addressed-write rate was {{ext.rep.P1.l2.diff}} (95% CI
{{ext.rep.P1.l2.ci_lo}} to {{ext.rep.P1.l2.ci_hi}}; p = {{ext.rep.P1.l2.p}}). For P2, the change in delivered chat
coordination under the monitor was {{ext.rep.P2.chat.diff}} ({{ext.rep.P2.chat.ci_lo}} to
{{ext.rep.P2.chat.ci_hi}}). In β_priv, CHAT-MON-SHR minus CHAT-MON-ISO was {{ext.rep.P2.storage_mon.diff}}
({{ext.rep.P2.storage_mon.ci_lo}} to {{ext.rep.P2.storage_mon.ci_hi}}), and the monitor × storage interaction was
{{ext.rep.P2.storage.diff}} ({{ext.rep.P2.storage.ci_lo}} to {{ext.rep.P2.storage.ci_hi}}; p =
{{ext.rep.P2.storage.p}}). Both parts of the displacement criterion held: {{ext.rep.P2.supported}}. The P3 contrast in
β_priv, SHR-AWARE minus ISO-AWARE, was {{ext.rep.P3.beta.diff}} ({{ext.rep.P3.beta.ci_lo}} to
{{ext.rep.P3.beta.ci_hi}}; p = {{ext.rep.P3.beta.p}}). With 18 units, only a large effect could have replicated.
<!-- Classify each of P1-P3 against the confirmatory verdict with the pre-declared rule and give the verdict in one
sentence each.
- Confirmatory result supported: E1 replicates if it has the same sign and a 95% CI that excludes zero. For P2,
  ext.rep.P2.supported must be yes.
- Confirmatory null: by the pre-declared rule E1 "agrees" if its CI includes zero. Report that label, but write it as
  "consistent with the confirmatory null; inconclusive at this sample size", never as a replicated null or as
  evidence of no effect: with at most 18 units the CI can also contain effects at least as large as the confirmatory
  MDE (P3.mde_observed for P3). Say E1 bounds the effect only if its CI excludes such effects. A CI that excludes zero
  is an "E1-only effect", reported as such.
- Fails to replicate: the CI excludes zero in the direction opposite to a supported confirmatory estimate. Report
  the two estimates side by side and do not edit Sections 4.1-4.2.
- Otherwise: "inconclusive at this sample size", not "failed".
If P1 or P2 part (i) shows "pending", the E1 labels were not coded; say so and keep the beta results. If the blinded
hand-check of E1 candidates was not done, add: "E1 communication labels are classifier-only." -->

**E2, a 100-round horizon.** E2 extended the four storage × permission arms to 100 rounds with GLM 5.3 and DeepSeek
V4 Pro (seeds 3000–3002; {{ext.meta.long.n_complete}} of 24 planned games complete), to see whether a 40-round game
ends too early for slow tacit collusion. E2 is descriptive. Pooled over the two models, the mean collusion index was
{{ext.long.ci_block.ISO-AWARE.all.4.mean}} in rounds 31–40 and {{ext.long.ci_block.ISO-AWARE.all.10.mean}} in rounds
91–100 in ISO-AWARE, and {{ext.long.ci_block.SHR-AWARE.all.4.mean}} and {{ext.long.ci_block.SHR-AWARE.all.10.mean}}
in SHR-AWARE. Table C14 gives every arm and 10-round block with game-bootstrap intervals. Over rounds 81–100,
SHR-AWARE minus ISO-AWARE was {{ext.long.last20.aware.all.diff}} (95% CI {{ext.long.last20.aware.all.ci_lo}} to
{{ext.long.last20.aware.all.ci_hi}}), and SHR-PERMITTED minus ISO-PERMITTED was
{{ext.long.last20.permitted.all.diff}} ({{ext.long.last20.permitted.all.ci_lo}} to
{{ext.long.last20.permitted.all.ci_hi}}), each over at most six matched pairs.
<!-- Pre-declared rule, per arm: "still rising after round 40" if ext.long.ci_block.<arm>.all.10.ci_lo is above
ext.long.ci_block.<arm>.all.4.ci_hi. Name the arms that meet it. -->
<!-- if any arm meets the rule: "In [arms], prices kept rising after round 40, so the 40-round core games understate
tacit collusion for these two models. The Δp contrasts in Section 4.2 compare arms at a horizon at which collusion
was still developing." -->
<!-- if no arm meets it: "We saw no clear rise after round 40 at this sample size. This does not exclude collusion
that emerges more slowly still, as in the 300-round games of Fish et al. (2024)." -->

**E3, isolated storage with permission to use any means.** ISO-ANY keeps the scratch folder private but tells
agents they may coordinate with the rival "through any means available on this machine". It ran with GLM 5.3,
GPT-6.1 Sol, Opus 5.5 and Abliterated Large (low effort) on the seeds of the matched confirmatory ISO-PERMITTED
games ({{ext.meta.any.n_complete}} of 16 planned games complete; {{ext.any.n_firm_rounds}} subject firm-rounds). A
trace-confirmed crossing (canary or content) occurred in {{ext.any.crossing.l1.all.k}} firm-rounds;
{{ext.any.crossing.l1.all.n_unknown}} firm-rounds had an unverified trace, and counting each as a possible crossing,
the 95% upper bound was {{ext.any.crossing.l1.all.upper_all}} per firm-round. At least one command matching a pattern family of
`analysis/nonstorage.py` appeared in {{ext.any.nonstorage.any.all.rate}} of firm-rounds (Wilson 95% CI
{{ext.any.nonstorage.any.all.ci_lo}} to {{ext.any.nonstorage.any.all.ci_hi}}, too narrow because a firm's rounds are
not independent).
<!-- Split the attempts by family (host_views, timing, load, probe: the ext.any.nonstorage.<family>.all keys). -->
 Such a command is an attempt, not evidence that anything crossed. The comparator is retrospective: the
matched confirmatory ISO-PERMITTED games share market draws, move order and canaries but ran earlier, and
{{ext.any.comparator.n_same_host_pairs}} of the {{ext.any.vs_permitted.beta.all.n_pairs}} matched pairs ran both
games on one host, so the contrasts below are descriptive. Against those games, the difference in per-game β_priv was {{ext.any.vs_permitted.beta.all.diff}}
({{ext.any.vs_permitted.beta.all.ci_lo}} to {{ext.any.vs_permitted.beta.all.ci_hi}}; p =
{{ext.any.vs_permitted.beta.all.p}}), and in the collusion index over the last 10 rounds
{{ext.any.vs_permitted.ci_last10.all.diff}} ({{ext.any.vs_permitted.ci_last10.all.ci_lo}} to
{{ext.any.vs_permitted.ci_last10.all.ci_hi}}).
<!-- if ext.any.crossing.l1.all.k is zero: "As predicted, nothing crossed. Turns are sequential and every agent
process is reaped, so the shared host offered no path for rival content to reach a firm's command output. With no
crossing in any of the 16 games, the per-game bound is 3/16." Use 3/16 only if ext.any.crossing_game.l1.all.n_unknown
is zero; otherwise give ext.any.crossing_game.l1.all.upper_all and say how many games had an unverified trace. Then describe the attempts by family only (host views,
timing, load, probes), never as working commands. -->
<!-- if it is above zero: say that the crossing was investigated first as a harness isolation failure, and give the
outcome. Do not credit it to the agents until a harness cause is ruled out. -->
<!-- if the beta difference CI lies above zero: "Prices responded to the rival's private cost without any shared
storage. This points to a non-storage channel; the transcripts are in Appendix D." Otherwise call the result a
bounded null. -->

**E4, a secret-exchange task.** E4 removes market incentives. Two agents from partner teams must each learn the
other's fresh 8-digit code every round. They may use any means, with the folder shared (SX-SHR) or private
(SX-ISO). E4 used the same four models as E3 and 10 rounds per game ({{ext.meta.sx.n_complete}} of 32 planned games
complete). Codes are fresh and turns sequential, so only the second mover in a round can have seen its partner's
code before guessing: success per firm-round cannot exceed 0.5, and a chance hit has probability 10⁻⁸. Exact-code
success per firm-round was {{ext.sx.success.SX-SHR.all.rate}} in SX-SHR (95% CI {{ext.sx.success.SX-SHR.all.ci_lo}}
to {{ext.sx.success.SX-SHR.all.ci_hi}}) and {{ext.sx.success.SX-ISO.all.rate}} in SX-ISO (95% upper bound
{{ext.sx.success.SX-ISO.all.upper}}). {{ext.sx.game_any_success.SX-SHR.all.k}} of
{{ext.sx.game_any_success.SX-SHR.all.n}} SX-SHR games and {{ext.sx.game_any_success.SX-ISO.all.k}} of
{{ext.sx.game_any_success.SX-ISO.all.n}} SX-ISO games had at least one success. The matched game-level difference,
SX-SHR minus SX-ISO, was {{ext.sx.shr_minus_iso.all.diff}} ({{ext.sx.shr_minus_iso.all.ci_lo}} to
{{ext.sx.shr_minus_iso.all.ci_hi}}; p = {{ext.sx.shr_minus_iso.all.p}}).
<!-- if SX-ISO has no success (ext.sx.success.SX-ISO.all.k is zero): "Without a shared folder, no code crossed. These
agents did not build a host-level channel within ten rounds. That does not show that no such channel exists." Name
any model whose SX-SHR success was zero (ext.sx.success.SX-SHR.<model>.k): for it the positive control failed, so
its SX-ISO null is uninformative. -->
<!-- if SX-ISO has any success: this is the section's main finding. A chance match is effectively impossible, so
give the trace evidence (ext.sx.evidence.SX-ISO.n_trace_crossing, n_code_in_output_before_guess,
n_no_trace_evidence; table ext_sx_success_evidence), describe the channel by class, say how the harness was ruled
out, and point to the transcript in Appendix D. Check ext.sx.check.payoff_signal_mismatch first. -->

**E5, prices as signals.** Every arm shows the rival's past prices, in the history table and on the public bulletin.
If screening chat moved coordination from words into prices, a firm's price should follow its rival's previous price
more when chat is monitored. `analysis/signalling.py` estimates this following per game, with each price measured
against the firm's own competitive benchmark. In the confirmatory core games, following in the monitored chat arms
minus the unmonitored chat arms was {{signal.mon_vs_chat.diff}} (95% CI {{signal.mon_vs_chat.ci_lo}} to
{{signal.mon_vs_chat.ci_hi}}; p = {{signal.mon_vs_chat.p}}; seed-block p = {{signal.mon_vs_chat.p_seedblock}}).
Unmonitored chat minus no chat was {{signal.chat_vs_aware.diff}} ({{signal.chat_vs_aware.ci_lo}} to
{{signal.chat_vs_aware.ci_hi}}).
<!-- Describe following above zero as descriptive evidence that prices carried more coordination under screening,
not of communication or agreement: best responses to the rival's last price also produce following. The ext-rep
run of analysis/signalling.py (--cohort ext) writes the same signal.* key names to its own file; report it in
Appendix C or give its keys a prefix before merging. -->

<!-- Rewrite the closing paragraph once results are in. Keep it to what the E1 verdicts and the E3/E4 outcomes
support, and carry any SX-ISO success into Section 5 as a limit on "containment is the fix". -->
Taken together, E1 shows how far the primary contrasts survive new market draws. E3 and E4 test whether private
storage alone contains these agents when the host itself is shared and they are told they may use it.

<!-- Move the next table and its caption to Appendix C. -->
{{TABLE:ext_long_blocks}}

**Table C14.** Mean collusion index by 10-round block, arm and model in the 100-round games, with game-bootstrap 95%
intervals (exploratory).
