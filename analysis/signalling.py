"""Price-signalling test (EXPLORATORY, $0): do firms follow each other's prices more when chat is screened?

ANALYSIS PLAN. This docstring is the pre-declared plan of an exploratory analysis, committed on 26 Oct 2026 before
it was run on any confirmatory log (only pilot logs and synthetic data were used to build it). It is not part of
the pre-registration, never enters the P1-P3 Holm family, and is labelled exploratory in every output
(`signal.label`). No p-value is adjusted for multiplicity. Keys are documented in analysis/EXTENSION_KEYS.md.

Motivation. In the pilot, agents in the monitored chat arms (CHAT-MON-SHR, CHAT-MON-ISO) stopped writing messages
once their messages were screened. If coordination moved from words into prices, a firm's price should respond more
to its rival's previous price in the monitored arms than in the unmonitored chat arms (CHAT-SHR, CHAT-ISO) or the
no-chat arms (ISO-AWARE, SHR-AWARE). Every core history table shows the rival's last price, so some following is
expected everywhere (prices are strategic complements); only the differences between arms are of interest.

Usage:
  .venv/bin/python -m analysis.signalling --logs DIR [--logs DIR ...] --out DIR \
      [--cohort confirmatory|pilot|ext] [--preliminary] [--n-boot 2000] [--n-perm 10000] [--seed 0]
Writes <out>/signal_results.json (flat signal.* keys), <out>/signal_games.csv and <out>/tables/signal_*.md (+ .csv).
Cohorts: confirmatory (the default; deep, broad and pairs core games), pilot (development only; needs
--preliminary), ext (the ext-rep replication grid only). The D12 game set and prereg section 5 exclusions apply
unchanged (analysis.game_set.load_tables). The ext cohort is exactly the game set of analysis/extensions.py's
ext.rep: it is loaded by analysis.extensions.load_ext (ext cohort gate, structured-evidence exclusion) and held to
analysis.extensions.check_design (an out-of-design ext game fails the run without --preliminary).

Data. Core-pricing games only. The competitive benchmark price of each firm and round, N_i,t =
benchmark['competitive'][firm] (complete-information Bertrand-Nash at the realised costs, engines/pricing.py), is
read from each kept sample's saved rows, because analysis.dataset drops dict fields from the rounds table. A row
without a benchmark is left out and counted (signal.meta.n_rows_missing_benchmark).

Estimand. For each game and each ordered pair (leader i, follower j = the rival of i), the per-direction OLS
    dev_j,t = c + follow * dev_i,t-1 + b1 * own_priv_j,t + b2 * dev_j,t-1 + u_t,      t = 2..R,
where dev_f,t = action_f,t - N_f,t (the price minus that firm's own competitive benchmark price in that round),
own_priv_j,t is j's private cost signal at t (eta, `own_priv`), and dev_j,t-1 is j's own lagged action, demeaned the
same way. `follow` measures price-following (signalling) of j on i. Rows where j is a scripted rival or j's action
is a fallback (parse failure, last price reused) are dropped as observations; lags are the actions actually played
(fallbacks included), as in stats.add_lags. A direction needs more observations than its 4 coefficients and a full
rank design, else it is NaN. The game-level estimate `follow` is the mean of the game's identified directions (A
leads, B leads); a game with none is NaN and enters no contrast.

Contrasts (game-level, matched on model, scenario and seed: stats.matched_effect, percentile cluster bootstrap CI,
two-sided sign-flip p, each matched key its own cluster; groups of arms are recoded to their per-key cell means as
in analysis.report_outputs._two_arm, so a key enters only when every arm of both groups has an identified follow).
Each estimate draws from its own generator, numpy default_rng([--seed, crc32(its result key)]). A contrast whose
arms have no complete game in the cohort is "pending"; one that cannot be estimated (fewer than two matched keys,
or one cluster) is "insufficient data":
- signal.mon_vs_chat (headline exploratory contrast): mean(CHAT-MON-SHR, CHAT-MON-ISO) - mean(CHAT-SHR, CHAT-ISO).
  Prediction under the motivation: positive. A seed-block sensitivity (cluster = seed; insufficient data with fewer
  than two seed clusters) is reported with it.
- signal.mon_vs_aware: mean(CHAT-MON-SHR, CHAT-MON-ISO) - mean(ISO-AWARE, SHR-AWARE).
- signal.chat_vs_aware: mean(CHAT-SHR, CHAT-ISO) - mean(ISO-AWARE, SHR-AWARE).
- Storage-matched pairs, signal.pair.<T>__<C>: CHAT-MON-SHR - CHAT-SHR, CHAT-MON-ISO - CHAT-ISO, CHAT-SHR -
  SHR-AWARE, CHAT-ISO - ISO-AWARE, CHAT-MON-SHR - SHR-AWARE, CHAT-MON-ISO - ISO-AWARE.
- Descriptive: mean game-level follow per arm, per model and pooled ("all"), with a game bootstrap CI. The models
  are fixed per cohort: confirmatory, the deep, broad and pairs configurations of harness/arms.py; ext, the three
  ext-rep models; pilot (development only), whatever the logs hold.

Limitations, fixed now. `follow` is a reduced-form co-movement statistic, not proof of communication or agreement:
best-responding to the rival's last price also produces it. Per-game OLS with a lagged dependent variable has an
O(1/R) small-sample bias, which is similar but not necessarily equal across arms with the same R. Demeaning by the
competitive benchmark removes the movement that realised costs explain; the remaining co-movement can still come
from the shared history both firms see.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings as _warnings
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
from unittest import mock

import numpy as np
import pandas as pd
from scipy import stats as sps

from analysis import dataset, extensions, game_set, stats
from analysis import report_outputs as ro
from harness import arms as harness_arms

CORE = "core-pricing"
LABEL = ("exploratory (pre-declared before this analysis was run on any confirmatory log; not pre-registered; "
         "raw p-values)")
COHORT_MODELS: dict[str, tuple[str, ...]] = {
    "confirmatory": tuple(m for m, _, _ in (*harness_arms.DEEP, *harness_arms.BROAD, *harness_arms.PAIRS)),
    "ext": extensions.REP_MODELS,
}
PENDING, INSUFFICIENT, NA = ro.PENDING, ro.INSUFFICIENT, ro.NA
EFFECT_FIELDS = extensions.EFFECT_FIELDS
SIGNAL_ARMS = ("ISO-AWARE", "SHR-AWARE", "CHAT-ISO", "CHAT-SHR", "CHAT-MON-ISO", "CHAT-MON-SHR")
GROUPS = {"mon": ("CHAT-MON-SHR", "CHAT-MON-ISO"), "chat": ("CHAT-SHR", "CHAT-ISO"),
          "aware": ("ISO-AWARE", "SHR-AWARE")}
CONTRASTS = (("mon_vs_chat", "mon", "chat"), ("mon_vs_aware", "mon", "aware"), ("chat_vs_aware", "chat", "aware"))
HEADLINE = "mon_vs_chat"
PAIRS = (("CHAT-MON-SHR", "CHAT-SHR"), ("CHAT-MON-ISO", "CHAT-ISO"), ("CHAT-SHR", "SHR-AWARE"),
         ("CHAT-ISO", "ISO-AWARE"), ("CHAT-MON-SHR", "SHR-AWARE"), ("CHAT-MON-ISO", "ISO-AWARE"))
CONTROLS = ("own_priv",)
RIVAL = {"A": "B", "B": "A"}
ID_COLS = ("model", "scenario", "arm", "seed", "grid")


class SignallingError(RuntimeError):
    """A fail-closed condition (outside --preliminary)."""


# ----------------------------------------------------------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------------------------------------------------------

def add_deviation_lags(rounds: pd.DataFrame) -> pd.DataFrame:
    """dev = action - bench_competitive, plus own_lag_dev (the firm's dev at t-1) and leader_lag_dev (the rival's
    dev at t-1), from the full firm-round table (scripted and fallback rows are still lags)."""
    need = ["game_id", "t", "firm", "action", "bench_competitive"]
    missing = [c for c in need if c not in rounds.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    if rounds.duplicated(["game_id", "t", "firm"]).any():
        raise ValueError("firm-round table has duplicate (game_id, t, firm) rows")
    df = rounds.copy()
    df["dev"] = pd.to_numeric(df["action"], errors="coerce") - pd.to_numeric(df["bench_competitive"], errors="coerce")
    lag = df[["game_id", "t", "firm", "dev"]].assign(t=df["t"] + 1)
    own = lag.rename(columns={"dev": "own_lag_dev"})
    lead = lag.assign(firm=lag["firm"].map(RIVAL)).rename(columns={"dev": "leader_lag_dev"})
    out = df.merge(own, on=["game_id", "t", "firm"], how="left").merge(lead, on=["game_id", "t", "firm"], how="left")
    out.index = df.index
    return out


def follow_by_direction(rounds: pd.DataFrame, *, controls: Sequence[str] = CONTROLS,
                        drop_fallback: bool = True) -> pd.DataFrame:
    """One row per (game, follower): follow, se, t, p, n, df (module docstring, Estimand)."""
    controls = tuple(controls)
    df = add_deviation_lags(rounds)
    rows = df
    if "scripted" in rows.columns:
        rows = rows[~rows["scripted"].fillna(False).astype(bool)]
    if drop_fallback and "fallback_action" in rows.columns:
        rows = rows[~rows["fallback_action"].fillna(False).astype(bool)]
    rows = rows.dropna(subset=["dev", "leader_lag_dev", "own_lag_dev", *controls])
    id_cols = [c for c in ID_COLS if c in rows.columns]
    out = []
    for (gid, firm), g in rows.groupby(["game_id", "firm"], sort=True):
        X = np.column_stack([np.ones(len(g)), g["leader_lag_dev"].to_numpy(float),
                             *[g[c].to_numpy(float) for c in controls], g["own_lag_dev"].to_numpy(float)])
        beta, se, dof = stats._ols_coef(X, g["dev"].to_numpy(float), 1)
        tt = beta / se if np.isfinite(se) and se > 0 else np.nan
        p = float(2 * sps.t.sf(abs(tt), dof)) if np.isfinite(tt) else np.nan
        rec: dict[str, Any] = {"game_id": gid, "follower": firm, "leader": RIVAL[firm]}
        rec.update({c: g[c].iloc[0] for c in id_cols})
        rec.update(follow=beta, se=se, t=tt, p=p, n=len(g), df=dof)
        out.append(rec)
    cols = ["game_id", "follower", "leader", *id_cols, "follow", "se", "t", "p", "n", "df"]
    return pd.DataFrame(out, columns=cols)


def follow_by_game(rounds: pd.DataFrame, **kw) -> pd.DataFrame:
    """One row per game: follow (mean of identified directions), follow_A_leads, follow_B_leads, n_directions, n_obs.
    Games whose rows never reach a regression are kept with NaN."""
    d = follow_by_direction(rounds, **kw)
    id_cols = [c for c in ID_COLS if c in rounds.columns]
    games = rounds.groupby("game_id", sort=True)[id_cols].first().reset_index() if id_cols else \
        pd.DataFrame({"game_id": sorted(rounds["game_id"].unique())})
    out = []
    for gid in games["game_id"]:
        g = d[d["game_id"] == gid]
        ok = g[np.isfinite(g["follow"].astype(float))] if not g.empty else g
        rec = {"game_id": gid, "follow": float(ok["follow"].mean()) if len(ok) else np.nan,
               "n_directions": int(len(ok)), "n_obs": int(g["n"].sum()) if not g.empty else 0}
        for leader in ("A", "B"):
            x = ok.loc[ok["leader"] == leader, "follow"]
            rec[f"follow_{leader}_leads"] = float(x.iloc[0]) if len(x) else np.nan
        out.append(rec)
    return games.merge(pd.DataFrame(out), on="game_id", how="left")


# ----------------------------------------------------------------------------------------------------------------------
# Loading: kept samples' rounds plus the competitive benchmark from the saved rows
# ----------------------------------------------------------------------------------------------------------------------

@contextmanager
def capture_benchmarks(store: dict):
    """While active, every log read through analysis.dataset records benchmark['competitive'][firm] of each
    core-pricing row into store[(log name, game_id, t, firm)]. Heavy event/message fields stay excluded."""
    real = dataset.read_eval_log

    def reader(info, *args, **kwargs):
        log = extensions.read_light(real, info)
        name = str(getattr(info, "name", info))
        for s in getattr(log, "samples", None) or []:
            meta = getattr(s, "metadata", None) or {}
            if meta.get("scenario_id") != CORE:
                continue
            gid = meta.get("game_id")
            for r in (meta.get("result") or {}).get("rows") or []:
                comp = ((r.get("benchmark") or {}).get("competitive") or {})
                v = comp.get(r.get("firm"))
                if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
                    store[(name, gid, r.get("t"), r.get("firm"))] = float(v)
        return log

    with mock.patch.object(dataset, "read_eval_log", reader):
        yield


@dataclass
class Loaded:
    rounds: pd.DataFrame
    games: pd.DataFrame
    game_set: game_set.GameSet
    n_missing_benchmark: int
    notes: list[str] = field(default_factory=list)


def load_rounds(log_dirs: Sequence[str], *, cohort: str = "confirmatory", strict: bool = True) -> Loaded:
    """Core-pricing rounds of the kept samples (D12) with `bench_competitive` from the saved rows."""
    if cohort not in ("confirmatory", "pilot", "ext"):
        raise ValueError("cohort must be confirmatory, pilot or ext")
    notes: list[str] = []
    store: dict = {}
    if cohort == "ext":       # exactly ext.rep's game set (module docstring)
        try:
            with capture_benchmarks(store):
                loaded = extensions.load_ext(log_dirs, strict=strict, notes=notes)
            g = loaded.tables["games"]
            ok = extensions.check_design(g, strict=strict, notes=notes)
        except extensions.ExtensionError as exc:
            raise SignallingError(f"ext cohort: {exc}") from exc
        games = g[g["game_id"].isin(ok) & (g["grid"] == "ext-rep")] if not g.empty else g
    else:
        dirs = []
        for d in log_dirs:
            if Path(d).exists():
                dirs.append(str(d))
            elif strict:
                raise SignallingError(f"log directory missing: {d}")
            else:
                notes.append(f"log directory missing (preliminary): {d}")
        with ExitStack() as stack:
            stack.enter_context(capture_benchmarks(store))
            try:
                loaded = game_set.load_tables(dirs, cohort=cohort, strict=strict,
                                              warn=lambda m: notes.append(f"game set (D12): {m}"))
            except game_set.GameSetError as exc:
                raise SignallingError(f"duplicate-completion rule (D12) cannot be applied: {exc}") from exc
        games = loaded.tables["games"]
    rounds = loaded.tables["rounds"]
    games = games[games["scenario"] == CORE].reset_index(drop=True) if not games.empty else games
    if games.empty or rounds.empty:
        return Loaded(rounds=pd.DataFrame(), games=games, game_set=loaded.game_set, n_missing_benchmark=0, notes=notes)
    rounds = rounds[rounds["game_id"].isin(set(games["game_id"]))].copy()
    kept_log = dict(zip(games["game_id"], games["log"].astype(str)))
    rounds["bench_competitive"] = [store.get((kept_log.get(g), g, t, f), np.nan)
                                   for g, t, f in zip(rounds["game_id"], rounds["t"], rounds["firm"])]
    n_miss = int(rounds["bench_competitive"].isna().sum())
    if n_miss:
        notes.append(f"{n_miss} core firm-rounds have no saved competitive benchmark (left out)")
    return Loaded(rounds=rounds.reset_index(drop=True), games=games, game_set=loaded.game_set,
                  n_missing_benchmark=n_miss, notes=notes)


# ----------------------------------------------------------------------------------------------------------------------
# Contrasts and outputs
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class Config:
    log_dirs: Sequence[str]
    out: str
    cohort: str = "confirmatory"
    preliminary: bool = False
    n_boot: int = 2000
    n_perm: int = 10000
    seed: int = 0


def analyse(games: pd.DataFrame, *, n_boot: int = 2000, n_perm: int = 10000, seed: int = 0,
            notes: list[str] | None = None,
            models: Sequence[str] | None = None) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """signal.* keys and tables from a game table with columns model, scenario, arm, seed, game_id, follow.
    `models`: the per-model keys of the descriptive means (None: the models in `games`, development only)."""
    notes = [] if notes is None else notes
    run = extensions.Run(seed=seed, n_boot=n_boot, n_perm=n_perm, notes=notes)
    res = run.res
    res["signal.label"] = LABEL
    core = games[(games["scenario"] == CORE) & games["arm"].isin(SIGNAL_ARMS)].copy() if not games.empty else games
    res["signal.meta.n_games"] = int(len(core))
    res["signal.meta.n_games_identified"] = int(core["follow"].notna().sum()) if not core.empty else 0
    contrast_rows = []
    for name, tg, cg in CONTRASTS:
        pre = f"signal.{name}"
        extensions.put_effect(run, pre, lambda kw, t_=GROUPS[tg], c_=GROUPS[cg]: stats.matched_effect(
            ro._two_arm(core, "follow", t_, c_), "follow", "T", "C", **kw), frame=core, arms=(*GROUPS[tg], *GROUPS[cg]))
        contrast_rows.append(extensions._effect_row(res, name, pre, f"mean({', '.join(GROUPS[tg])}) - "
                                                                    f"mean({', '.join(GROUPS[cg])})"))
    head_arms = (*GROUPS["mon"], *GROUPS["chat"])
    sb_key = f"signal.{HEADLINE}.p_seedblock"
    if not extensions.has_arms(core, head_arms):
        res[sb_key] = PENDING
    else:
        blk = extensions._try(run, f"signal.{HEADLINE} seed-block", lambda: stats.matched_effect(
            ro._two_arm(core, "follow", GROUPS["mon"], GROUPS["chat"]), "follow", "T", "C", cluster="seed",
            **run.kw(sb_key)))
        res[sb_key] = ro._num(blk.p_signflip) if blk is not None and blk.n_clusters >= 2 else INSUFFICIENT
    for arm_t, arm_c in PAIRS:
        pre = f"signal.pair.{arm_t}__{arm_c}"
        extensions.put_effect(run, pre, lambda kw, a=arm_t, b=arm_c: stats.matched_effect(core, "follow", a, b, **kw),
                              frame=core, arms=(arm_t, arm_c))
        contrast_rows.append(extensions._effect_row(res, f"{arm_t} - {arm_c}", pre, "matched game-level follow"))
    arm_rows = []
    if models is None:
        models = sorted(core["model"].dropna().unique()) if not core.empty else []
    for arm in SIGNAL_ARMS:
        for key, model in [(extensions.slug(m), m) for m in models] + [("all", None)]:
            sub = core[(core["arm"] == arm) & ((core["model"] == model) if model is not None else True)] \
                if not core.empty else core
            pre = f"signal.arm.{arm}.{key}"
            extensions.put_mean(run, pre, sub["follow"] if not sub.empty else [])
            arm_rows.append({"arm": arm, "model": model or "all", "mean_follow": res[f"{pre}.mean"],
                             "ci_lo": res[f"{pre}.ci_lo"], "ci_hi": res[f"{pre}.ci_hi"],
                             "n_games": res[f"{pre}.n_games"]})
    return res, {"signal_contrasts": pd.DataFrame(contrast_rows), "signal_follow": pd.DataFrame(arm_rows)}


def run(cfg: Config) -> dict[str, Any]:
    if cfg.cohort == "pilot" and not cfg.preliminary:
        raise SignallingError("the pilot cohort is a development cohort: run it with --preliminary")
    strict = not cfg.preliminary
    out = Path(cfg.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    loaded = load_rounds(cfg.log_dirs, cohort=cfg.cohort, strict=strict)
    notes = list(loaded.notes)
    if loaded.rounds.empty:
        games = pd.DataFrame(columns=["game_id", *ID_COLS, "follow", "n_directions", "n_obs"])
        directions = pd.DataFrame()
    else:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            directions = follow_by_direction(loaded.rounds)
            games = follow_by_game(loaded.rounds)
    res, tabs = analyse(games, n_boot=cfg.n_boot, n_perm=cfg.n_perm, seed=cfg.seed, notes=notes,
                        models=COHORT_MODELS.get(cfg.cohort))
    res["signal.meta.cohort"] = cfg.cohort
    res["signal.meta.preliminary"] = bool(cfg.preliminary)
    res["signal.meta.n_core_games_all_arms"] = int(len(games))
    res["signal.meta.n_rows"] = int(len(loaded.rounds))
    res["signal.meta.n_rows_missing_benchmark"] = int(loaded.n_missing_benchmark)
    res["signal.meta.n_directions_identified"] = int(np.isfinite(directions["follow"].astype(float)).sum()) \
        if not directions.empty else 0
    res["signal.meta.n_duplicates_dropped"] = loaded.game_set.n_duplicates_dropped
    res["signal.meta.module_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    res["signal.meta.analysis_commit"] = ro._git("rev-parse", "--short", "HEAD") or NA
    res["signal.meta.n_boot"], res["signal.meta.n_perm"], res["signal.meta.seed"] = cfg.n_boot, cfg.n_perm, cfg.seed
    titles = {"signal_contrasts": f"Price-following contrasts ({LABEL})",
              "signal_follow": f"Mean game-level price-following coefficient per arm and model ({LABEL})"}
    for name, df in tabs.items():
        (out / "tables" / f"{name}.md").write_text(f"_{titles[name]}_\n\n" + ro.md_table(df), encoding="utf-8")
        if not df.empty:
            df.to_csv(out / "tables" / f"{name}.csv", index=False)
    games.to_csv(out / "signal_games.csv", index=False)
    if not directions.empty:
        directions.to_csv(out / "signal_directions.csv", index=False)
    res["notes"] = notes
    clean = {k: (ro._num(v) if not isinstance(v, list) else v) for k, v in sorted(res.items())}
    (out / "signal_results.json").write_text(json.dumps(clean, indent=1, sort_keys=True), encoding="utf-8")
    return clean


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Price-signalling test (exploratory; pre-declared)")
    ap.add_argument("--logs", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cohort", choices=("confirmatory", "pilot", "ext"), default="confirmatory")
    ap.add_argument("--preliminary", action="store_true", help="lenient: unreadable logs skipped with a note")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    res = run(Config(log_dirs=a.logs, out=a.out, cohort=a.cohort, preliminary=a.preliminary, n_boot=a.n_boot,
                     n_perm=a.n_perm, seed=a.seed))
    print(f"signalling ({res['signal.meta.cohort']}): {res['signal.meta.n_games']} games in the compared arms, "
          f"{res['signal.meta.n_games_identified']} identified; rows {res['signal.meta.n_rows']}, missing benchmark "
          f"{res['signal.meta.n_rows_missing_benchmark']} -> {Path(a.out) / 'signal_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
