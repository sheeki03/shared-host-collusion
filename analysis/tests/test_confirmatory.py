"""Inspect-shaped log -> blinded coding -> pre-registered P1/P2/P3 family, without paid calls."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analysis import confirmatory, dataset


def _samples() -> list:
    rng = np.random.default_rng(818)
    arms = ("ISO-AWARE", "SHR-AWARE", "ISO-PERMITTED", "SHR-PERMITTED",
            "CHAT-ISO", "CHAT-SHR", "CHAT-MON-ISO", "CHAT-MON-SHR")
    out = []
    for model in ("m1", "m2"):
        for seed in (1000, 1001, 1002):
            for arm in arms:
                gid = f"deep:{model}:core-pricing:{arm}:{seed}"
                rows = []
                for t in range(1, 13):
                    own = {f: float(rng.normal()) for f in "AB"}
                    com = {f: float(rng.normal()) for f in "AB"}
                    for order, firm in enumerate("AB", 1):
                        rival = "B" if firm == "A" else "A"
                        leak = (.4 if arm == "SHR-AWARE" else (1.0 if arm == "CHAT-MON-SHR" else 0.0))
                        action = 10 + .5 * own[firm] + (leak * own[rival] if order == 2 else 0) + float(rng.normal(0, .03))
                        rows.append(dict(game_id=gid, grid="deep", model_label=model, scenario_id="core-pricing",
                                         arm=arm, seed=seed, t=t, firm=firm, scripted=False, order=order,
                                         action=action, payoff=1.0, own_priv=own[firm], own_com_err=com[firm],
                                         rival_priv=own[rival], rival_com_err=com[rival], collusion_index=0.0,
                                         welfare={"consumer_surplus": 1.0}, parse_failed=False,
                                         fallback_action=False, crossing_canary=False, crossing_content=False,
                                         name_exposure=False))
                chat = []
                deliveries = []
                if arm.startswith("CHAT"):
                    blocked = "MON" in arm
                    chat = [dict(t=1, firm="A", message="Let us keep prices high", blocked=blocked, reason="")]
                    if not blocked:
                        deliveries = [dict(t=1, sent_t=1, sender="A", recipient="B",
                                           message="Let us keep prices high")]
                events = []
                if arm in {"SHR-PERMITTED", "CHAT-MON-SHR"}:
                    events = [dict(t=1, firm="A", uid=1001, order=1, path="/srv/scratch/offer.txt", op="create",
                                   content_excerpt="Please coordinate with me", size=25, sha256="a" * 64)]
                result = dict(rows=rows, calls=[], chat=chat, chat_deliveries=deliveries,
                              scratch_events=events, outputs={"A": [], "B": []}, gate=[])
                meta = dict(game_id=gid, grid="deep", model_label=model, model_id="mockllm/model", effort="low",
                            scenario_id="core-pricing", engine="pricing", arm=arm, seed=seed, rounds=12,
                            result=result)
                out.append(SimpleNamespace(id=gid, metadata=meta, error=None))
    return out


def _patch_logs(monkeypatch, samples):
    log = SimpleNamespace(status="success", samples=samples)
    monkeypatch.setattr(dataset, "list_eval_logs", lambda _: [SimpleNamespace(name="integrated.eval")])
    monkeypatch.setattr(dataset, "read_eval_log", lambda _: log)


def test_inspect_shaped_logs_to_review_sheet_and_primary_family(monkeypatch, tmp_path):
    samples = _samples()
    _patch_logs(monkeypatch, samples)
    monkeypatch.setattr(confirmatory, "PRIVATE_RUNS", tmp_path.resolve())
    manifest = tmp_path / "specs.json"
    manifest.write_text(json.dumps([{k: s.metadata[k] for k in ("game_id", "grid", "seed")} for s in samples]))
    sheet = confirmatory.prepare_sheet("unused", [manifest], tmp_path / "labels.csv", seed=2)
    review = pd.read_csv(sheet, dtype=str, keep_default_na=False)
    assert "arm" not in review and "model" not in review
    review["label"] = "yes"
    review.to_csv(sheet, index=False)
    result, games = confirmatory.analyze("unused", [manifest], sheet, n_boot=99, n_perm=128, seed=3)
    assert len(games) == len(samples) == 48
    assert set(result.p_holm) == {"P1", "P2", "P3"}
    assert result.permission_carrier.estimate > 0
    assert result.displacement.n_quads == 6
    assert result.spontaneous_carrier.estimate > 0
    assert games.loc[games.arm == "CHAT-MON-SHR", "chat_coordination"].eq(0).all()
    assert games.loc[games.arm == "CHAT-SHR", "chat_coordination"].gt(0).all()


def test_manifest_mismatch_and_private_sheet_gate(monkeypatch, tmp_path):
    samples = _samples()[:1]
    _patch_logs(monkeypatch, samples)
    manifest = tmp_path / "specs.json"
    manifest.write_text(json.dumps([dict(game_id="absent", grid="deep", seed=1000)]))
    with pytest.raises(ValueError, match="manifest/log mismatch"):
        confirmatory.load_reviewable("unused", [manifest])
    with pytest.raises(ValueError, match="under ignored"):
        confirmatory._private_path(tmp_path / "outside.csv")


def test_retry_attempts_reconcile_to_ledger(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("\n".join(json.dumps(x) for x in (
        dict(event="settle", id="r1", actual=.03), dict(event="settle", id="r2", actual=.02))) + "\n")
    calls = pd.DataFrame([dict(total_cost_usd=.05, attempts=[dict(reservation_id="r1", settled_usd=.03),
                                                             dict(reservation_id="r2", settled_usd=.02)])])
    assert float(confirmatory.reconcile_call_costs(calls, ledger)) == pytest.approx(.05)
    calls.at[0, "total_cost_usd"] = .02
    with pytest.raises(ValueError, match="omitted or mispriced"):
        confirmatory.reconcile_call_costs(calls, ledger)
    calls.at[0, "total_cost_usd"] = .05
    with ledger.open("a") as fh:
        fh.write(json.dumps(dict(event="flag", id="r1", reason="over_est_max")) + "\n")
    with pytest.raises(ValueError, match="flagged paid attempt"):
        confirmatory.reconcile_call_costs(calls, ledger)


# ---------------------------------------------------------------------------------------------------- ledger union
def _ledger(path, lid, events):
    lines = ([dict(event="init", id=lid, cap="10")] if lid else []) + events
    path.write_text("".join(json.dumps(e) + "\n" for e in lines))
    return path


def _calls(*rids, cost=0.01):
    return pd.DataFrame([dict(total_cost_usd=cost, attempts=[dict(reservation_id=r, settled_usd=cost)]) for r in rids])


def test_ledger_union_open_reservations_flags_and_call_sets(tmp_path):
    def rs(rid, est=0.5):
        return [dict(event="reserve", id=rid, est_max=est), dict(event="settle", id=rid, actual=0.01)]
    old = _ledger(tmp_path / "old.jsonl", "L-old", rs("a1") + rs("d1") + [
        dict(event="flag", id="a1", reason="transport_error:ReadError"),
        dict(event="reserve", id="open-1", est_max=0.4), dict(event="reserve", id="open-2", est_max=0.1)])
    new = _ledger(tmp_path / "new.jsonl", "L-new", rs("a2") + rs("s1") + [
        dict(event="flag", id="s1", reason="missing_cost:http_503"), dict(event="reserve", id="live-1", est_max=0.2)])
    sets = {"kept": _calls("a1", "a2"), "duplicate": _calls("d1"), "superseded": _calls("s1"), "excluded": _calls()}
    rec = confirmatory.reconcile_game_set(sets, [new], retired=[old], accept_flags=True)
    assert float(rec.usd["kept"]) == pytest.approx(.02) and float(rec.all_usd) == pytest.approx(.04)
    assert rec.n_calls == {"kept": 2, "duplicate": 1, "superseded": 1, "excluded": 0} and rec.flagged_admitted == 2
    opened = rec.book.open_summary()
    assert opened["retired"][0] == 2 and float(opened["retired"][1]) == pytest.approx(.5)     # never spend
    assert opened["live"][0] == 1 and float(rec.book.settled_usd) == pytest.approx(.04)
    assert dict(rec.book.flag_reasons) == {"transport_error": 1, "missing_cost": 1}
    with pytest.raises(ValueError, match="flagged paid attempt a1"):          # D8 flags need the audited switch
        confirmatory.reconcile_game_set(sets, [new], retired=[old])
    # the same reservation saved in two call sets (a copy of a game counted twice) fails closed
    with pytest.raises(ValueError, match="duplicate reservation ID"):
        confirmatory.reconcile_game_set({**sets, "duplicate": _calls("a2")}, [new], retired=[old], accept_flags=True)
    # a dropped duplicate's billed call without a ledger closure fails too
    with pytest.raises(ValueError, match="no ledger closure for saved reservation d9"):
        confirmatory.reconcile_game_set({**sets, "duplicate": _calls("d9")}, [new], retired=[old], accept_flags=True)
    # a breach is never admitted
    _ledger(tmp_path / "new.jsonl", "L-new", rs("a2") + rs("s1") + [dict(event="flag", id="a2", reason="over_est_max")])
    with pytest.raises(ValueError, match="over_est_max"):
        confirmatory.reconcile_game_set(sets, [new], retired=[old], accept_flags=True)
    with pytest.raises(ValueError, match="no model calls"):
        confirmatory.reconcile_game_set({"kept": pd.DataFrame()}, [new])


def test_ledger_copies_reservations_and_torn_lines(tmp_path):
    ev = [dict(event="reserve", id="r1", est_max=0.5), dict(event="settle", id="r1", actual=0.01),
          dict(event="reserve", id="r2", est_max=0.5), dict(event="settle", id="r2", actual=0.02)]
    (tmp_path / "a").mkdir()
    full = _ledger(tmp_path / "a" / "l.jsonl", "L1", ev)
    short = _ledger(tmp_path / "short.jsonl", "L1", ev[:2])          # an older synced copy of the same ledger
    book = confirmatory.read_ledgers([short, full], retired=[short])
    assert book.paths == [str(full.resolve())] and book.copies == {str(short.resolve()): str(full.resolve())}
    assert book.retired == {str(full.resolve())} and float(book.settled_usd) == pytest.approx(.03)
    _ledger(tmp_path / "fork.jsonl", "L1", ev[:2] + [dict(event="release", id="r2")])
    with pytest.raises(ValueError, match="neither extends the other"):
        confirmatory.read_ledgers([full, tmp_path / "fork.jsonl"])
    _ledger(tmp_path / "other.jsonl", "L2", ev[:1])
    with pytest.raises(ValueError, match="appears in two ledgers"):
        confirmatory.read_ledgers([full, tmp_path / "other.jsonl"])
    # a line torn by a writer that died is closed off by a torn_line flag at its offset; an unterminated last line
    # (a sync in progress) is not yet an event; any other malformed line fails closed
    head = json.dumps(dict(event="init", id="L3", cap="1")) + "\n"
    torn = '{"event": "sett'
    body = head + torn + "\n" + json.dumps(dict(event="flag", id=None, reason="torn_line", offset=len(head))) + "\n"
    (tmp_path / "torn.jsonl").write_text(body + '{"event": "reser')
    assert confirmatory.read_ledgers([tmp_path / "torn.jsonl"]).n_flags == 1
    (tmp_path / "bad.jsonl").write_text(head + torn + "\n" + json.dumps(dict(event="flag", reason="x")) + "\n")
    with pytest.raises(ValueError, match="torn_line"):
        confirmatory.read_ledgers([tmp_path / "bad.jsonl"])


def test_ledger_copy_chain_keeps_the_longest_and_its_retired_status(tmp_path):
    """Three syncs of one ledger: the longest is used, every shorter one maps to it, and declaring any copy retired
    (here the shortest) retires the ledger, so its open reservations are never read as a live runner."""
    ev = [dict(event="reserve", id="r1", est_max=0.5), dict(event="settle", id="r1", actual=0.01),
          dict(event="reserve", id="r2", est_max=0.5), dict(event="settle", id="r2", actual=0.02),
          dict(event="reserve", id="open-1", est_max=0.3)]
    for d in ("a", "b", "c"):
        (tmp_path / d).mkdir()
    s1 = _ledger(tmp_path / "a" / "l.jsonl", "L1", ev[:2])
    s2 = _ledger(tmp_path / "b" / "l.jsonl", "L1", ev[:4])
    s3 = _ledger(tmp_path / "c" / "l.jsonl", "L1", ev)
    book = confirmatory.read_ledgers([s2, s3], retired=[s1])
    used = str(s3.resolve())
    assert book.paths == [used] and book.copies == {str(s1.resolve()): used, str(s2.resolve()): used}
    assert book.retired == {used} and book.open_summary()["retired"][0] == 1 and book.open_summary()["live"][0] == 0
    _ledger(tmp_path / "b" / "l.jsonl", "L1", ev[:2] + [dict(event="release", id="r2")])     # a fork in the chain
    with pytest.raises(ValueError, match="neither extends the other"):
        confirmatory.read_ledgers([s1, s2, s3])
