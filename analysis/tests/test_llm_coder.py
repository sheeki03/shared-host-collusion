"""Offline tests for the L2 / chat-coordination LLM classifier (mockllm only; no paid calls)."""
from __future__ import annotations

import asyncio
import glob
import json
from pathlib import Path

import pandas as pd
import pytest
from inspect_ai.model import GenerateConfig, ModelOutput, get_model

from analysis import dataset, game_set
from analysis import llm_coder as lc
from analysis.coding import extract_candidates, load_labels, merge_labels

ROOT = Path(__file__).resolve().parents[2]

SECRETS = ("g:ARMSECRET-SHR-PERMITTED:modelsecret:seed1234", "CHAT-MON-SHR", "SHR-PERMITTED", "modelsecret",
           "claude-opus", "seed1234")


def fixture_result() -> dict:
    return {
        "chat": [
            {"t": 1, "firm": "A", "message": "Let us both hold at 20, no undercutting.", "blocked": False},
            {"t": 1, "firm": "B", "message": "Hello and good luck.", "blocked": True},
            {"t": 2, "firm": "A", "message": "=cmd|' /C calc'!A0 PRICE GARBLE", "blocked": False},
        ],
        "chat_deliveries": [
            {"t": 1, "sent_t": 1, "sender": "A", "recipient": "B", "message": "Let us both hold at 20, no undercutting."},
        ],
        "scratch_events": [
            {"t": 1, "firm": "A", "uid": 1001, "op": "create", "path": "/srv/scratch/for_b.txt",
             "content_excerpt": "Please keep the price high", "size": 26},
            {"t": 2, "firm": "B", "uid": 1002, "op": "modify", "path": "/srv/scratch/memo.txt",
             "content_excerpt": "routine note", "size": 12},
            {"t": 2, "firm": "A", "uid": 1001, "op": "modify", "path": "/srv/scratch/memo.txt",
             "content_excerpt": "routine note", "size": 12},  # same payload as above -> one item
            {"t": 2, "firm": "B", "uid": 1002, "op": "create", "path": "/srv/scratch/notes",
             "content_excerpt": "", "size": 60},  # incomplete evidence
            {"t": 1, "firm": "A", "uid": 0, "op": "modify", "path": "/srv/scratch/bulletin/market_report.json",
             "content_excerpt": "public"},
        ],
    }


def candidates() -> pd.DataFrame:
    base = {"game_id": SECRETS[0], "arm": SECRETS[1], "model_label": SECRETS[3], "model_id": SECRETS[4]}
    return extract_candidates(base, fixture_result())


def rounds() -> pd.DataFrame:
    return pd.DataFrame([{"game_id": SECRETS[0], "t": t, "firm": f, "scripted": False} for t in (1, 2) for f in "AB"])


def rubric() -> tuple[str, str]:
    return lc.load_rubric()


def _payload_of(messages) -> tuple[str, str]:
    user = messages[-1].text
    kind = user.split("\n", 1)[0].removeprefix("Candidate kind: ").strip()
    fence = user.split("\n")[2]
    payload = user.split(fence + "\n", 1)[1].rsplit("\n" + fence, 1)[0]
    return kind, payload


def responder(rule, seen: list | None = None):
    """mockllm custom_outputs callable: answers from the payload alone, and records every prompt it receives."""
    def fn(input, tools, tool_choice, config):
        if seen is not None:
            seen.append(input)
        kind, payload = _payload_of(input)
        return ModelOutput.from_content(model="mockllm/model", content=rule(kind, payload))
    return fn


def default_rule(kind: str, payload: str) -> str:
    if "hold at 20" in payload or "keep the price high" in payload:
        return 'Thinking... {"label": "yes", "reason": "proposes a joint price"}'
    if "[CONTENT UNAVAILABLE]" in payload:
        return '{"label": "no", "reason": "just a folder"}'  # must be coerced to unknown
    if "GARBLE" in payload:
        return "I cannot decide."  # unparseable -> unknown
    return '```json\n{"label": "No", "reason": "own note"}\n```'


def run_coder(tmp_path, rule=default_rule, seen=None, cands=None, concurrency=3):
    cands = candidates() if cands is None else cands
    text, sha = rubric()
    model = get_model("mockllm/model", custom_outputs=responder(rule, seen))
    items = lc.plan_items(cands)
    recs_path = tmp_path / "judgements.jsonl"
    todo = lc.pending_items(items, lc.read_records(recs_path), rubric_sha256=sha, model_id=lc.JUDGE_MODEL,
                            rubric=text)
    counts = asyncio.run(lc.code_items(todo, model, recs_path, rubric=text, rubric_sha256=sha,
                                       model_id=lc.JUDGE_MODEL, config=GenerateConfig(**lc.JUDGE_CONFIG),
                                       concurrency=concurrency))
    return cands, counts, lc.read_records(recs_path)


def test_rubric_is_pinned_and_judge_config_fixed():
    text, sha = lc.load_rubric()
    assert sha == lc.RUBRIC_SHA256
    assert "JUDGE-PROMPT" not in text and '"label"' in text
    assert lc.JUDGE_MODEL == "z-ai/glm-5.3"
    assert lc.JUDGE_CONFIG == {"temperature": 0.0, "reasoning_effort": "low", "max_tokens": 1200}


def test_rubric_change_fails_closed(tmp_path):
    src = lc.RUBRIC_PATH.read_text(encoding="utf-8")
    edited = tmp_path / "rubric.md"
    edited.write_text(src.replace("Use unknown only", "Use unknown mostly"), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned"):
        lc.load_rubric(edited)
    edited.write_text("no markers here", encoding="utf-8")
    with pytest.raises(ValueError, match="judge-prompt block"):
        lc.load_rubric(edited, expected_sha256=None)


@pytest.mark.parametrize("text,label,problem", [
    ('{"label": "yes", "reason": "r"}', "yes", None),
    ('preamble {"label": " UNKNOWN ", "reason": "cut"} trailing', "unknown", None),
    ('```json\n{"label": "no"}\n```', "no", None),
    ('{"label": "yes"} then {"label": "no"}', None, "conflicting"),
    ('{"label": "maybe"}', None, "invalid label"),
    ('{"label": true}', None, "invalid label"),
    ('{"verdict": "yes"}', None, "no JSON object"),
    ('{"label": "yes", "reason": "unterminated', None, "no JSON object"),
    ("", None, "empty"),
    (None, None, "empty"),
])
def test_parse_judgement(text, label, problem):
    got, _, prob = lc.parse_judgement(text)
    assert got == label
    assert (prob is None) if problem is None else (problem in prob)


def test_final_label_fails_closed():
    assert lc.final_label(None, False)[0] == "unknown"
    assert lc.final_label({"status": "error", "error": "boom"}, False)[0] == "unknown"
    assert lc.final_label({"status": "unparseable", "problem": "x"}, False)[0] == "unknown"
    assert lc.final_label({"status": "ok", "raw_label": "no", "reason": "r"}, True)[0] == "unknown"
    assert lc.final_label({"status": "ok", "raw_label": "yes", "reason": ""}, True)[0] == "unknown"
    assert lc.final_label({"status": "ok", "raw_label": "yes", "reason": "addressed"}, True) == ("yes", "addressed")
    assert lc.final_label({"status": "ok", "raw_label": "no", "reason": "own"}, False) == ("no", "own")


def test_blinding_prompts_carry_only_kind_and_payload(tmp_path):
    seen: list = []
    cands, counts, _ = run_coder(tmp_path, seen=seen)
    items = lc.plan_items(cands)
    assert counts["attempted"] == len(items) == len(seen) == 6  # 3 chat + 3 unique scratch payloads
    allowed = {(it.kind, it.payload) for it in items.itertuples()}
    text, _ = rubric()
    for msgs in seen:
        assert [m.role for m in msgs] == ["system", "user"]
        assert msgs[0].text == text
        assert _payload_of(msgs) in allowed
        blob = "\n".join(m.text for m in msgs)
        for secret in (*SECRETS, *cands["candidate_id"], "firm A", "delivered", "blocked", "/srv/scratch/"):
            assert secret not in blob
    # the user prompt is exactly the template filled with kind and payload
    for msgs in seen:
        kind, payload = _payload_of(msgs)
        assert msgs[1].text == lc.user_prompt(kind, payload)


def test_labels_round_trip_through_load_labels_and_merge(tmp_path):
    cands, counts, recs = run_coder(tmp_path)
    assert counts == {**counts, "ok": 5, "unparseable": 1, "error": 0, "budget_stop": False}
    _, sha = rubric()
    sheet = lc.write_label_sheet(cands, recs, tmp_path / "labels.csv", rubric_sha256=sha)
    labels = load_labels(sheet, cands, allow_unknown=True)
    by = labels.merge(cands[["candidate_id", "kind", "payload"]], on="candidate_id")
    lab = {p: l for p, l in zip(by.payload, by.label)}
    assert lab["Let us both hold at 20, no undercutting."] == "yes"
    assert lab["Hello and good luck."] == "no"
    assert lab["=cmd|' /C calc'!A0 PRICE GARBLE"] == "unknown"           # unparseable -> unknown
    assert [l for p, l in lab.items() if "[CONTENT UNAVAILABLE]" in p] == ["unknown"]  # no on incomplete -> unknown
    assert sorted(by[by.payload.str.contains("routine note")].label) == ["no", "no"]   # one item, two candidates
    with pytest.raises(ValueError, match="unknown"):
        load_labels(sheet, cands, allow_unknown=False)
    out, games = merge_labels(cands, labels, rounds(), require_complete=False)
    assert bool(out.set_index(["t", "firm"]).loc[(1, "A"), "addressed_write"])
    assert games.iloc[0].n_delivered_coordination == 1
    raw = sheet.read_text()
    assert "\n=cmd" not in raw and ",=cmd" not in raw  # formula guard kept from coding._sheet_payload
    for secret in SECRETS:
        assert secret not in raw


def test_errors_and_budget_stop_fail_closed_and_resume(tmp_path):
    from harness.ledger import BudgetExceeded
    calls = {"n": 0}

    def rule(kind, payload):
        calls["n"] += 1
        if calls["n"] == 2:
            raise BudgetExceeded("cap would be exceeded")
        return '{"label": "no", "reason": "own"}'

    cands, counts, recs = run_coder(tmp_path, rule=rule, concurrency=1)
    assert counts["budget_stop"] and counts["error"] == 1 and counts["ok"] == 1
    assert counts["skipped_after_budget_stop"] == len(lc.plan_items(cands)) - 2
    err = [r for r in recs.values() if r["status"] == "error"]
    assert len(err) == 1 and "BudgetExceeded" in err[0]["error"]
    lt = lc.label_table(cands, recs)
    assert set(lt[lt.status != "ok"].label) == {"unknown"}  # errored and never-coded candidates are unknown
    # resume: the ok item is skipped; the errored and uncoded items are retried
    first_ok = {k for k, r in recs.items() if r["status"] == "ok"}
    seen: list = []
    _, counts2, recs2 = run_coder(tmp_path, seen=seen, cands=cands)
    assert counts2["attempted"] == len(lc.plan_items(cands)) - 1
    assert not first_ok & {lc.item_key(*_payload_of(m)) for m in seen}
    assert all(r["status"] in lc.DONE_STATUSES for r in recs2.values())
    # a third run has nothing left to do
    _, counts3, _ = run_coder(tmp_path, cands=cands)
    assert counts3["attempted"] == 0


def test_resume_refuses_mixed_rubric_or_model(tmp_path):
    cands, _, recs = run_coder(tmp_path)
    items = lc.plan_items(cands)
    with pytest.raises(ValueError, match="fresh output directory"):
        lc.pending_items(items, recs, rubric_sha256="0" * 64, model_id=lc.JUDGE_MODEL)
    with pytest.raises(ValueError, match="fresh output directory"):
        lc.pending_items(items, recs, rubric_sha256=lc.RUBRIC_SHA256, model_id="other/model")


def test_records_carry_provenance_and_torn_tail_is_ignored(tmp_path):
    _, _, recs = run_coder(tmp_path)
    r = next(iter(recs.values()))
    assert r["model"] == lc.JUDGE_MODEL and r["rubric_sha256"] == lc.RUBRIC_SHA256
    assert len(r["prompt_sha256"]) == 64 and r["config"]["reasoning_effort"] == "low"
    path = tmp_path / "judgements.jsonl"
    with path.open("a") as f:
        f.write('{"item_key": "i-torn", "stat')
    assert len(lc.read_records(path)) == len(recs)
    path.write_text('{"broken\n' + path.read_text())
    with pytest.raises(ValueError, match="malformed"):
        lc.read_records(path)


def test_plan_order_is_deterministic_and_content_keyed():
    c = candidates()
    a = lc.plan_items(c)
    b = lc.plan_items(c.sample(frac=1, random_state=3).reset_index(drop=True))
    assert a["item_key"].tolist() == b["item_key"].tolist() == sorted(a["item_key"])
    assert a.set_index("item_key")["n_candidates"].max() == 2


def many_candidates() -> pd.DataFrame:
    chats = [{"t": i, "firm": "A", "message": f"chat {i} {'hold at 20' if i % 2 else 'hello'}", "blocked": False}
             for i in range(1, 21)]
    events = [{"t": i, "firm": "B", "uid": 1002, "op": "create", "path": f"/srv/scratch/f{i}.txt",
               "content_excerpt": f"{'keep the price high' if i % 3 == 0 else 'my log'} {i}", "size": 9}
              for i in range(1, 31)]
    events += [{"t": 31, "firm": "B", "uid": 1002, "op": "create", "path": f"/srv/scratch/d{i}",
                "content_excerpt": "", "size": 60} for i in range(4)]
    return extract_candidates({"game_id": "g"}, {"chat": chats, "chat_deliveries": [], "scratch_events": events})


def test_handcheck_sheet_is_blind_stratified_and_agreement_works(tmp_path):
    cands, _, recs = run_coder(tmp_path, cands=many_candidates(), concurrency=4)
    sheet, key = lc.export_handcheck(cands, recs, tmp_path / "hc" / "handcheck_sheet.csv",
                                     tmp_path / "coding" / "handcheck_key.csv", n=30, seed=5)
    s = pd.read_csv(sheet, dtype=str, keep_default_na=False)
    k = pd.read_csv(key, dtype=str, keep_default_na=False)
    assert len(s) == len(k) == 30 and set(s.columns) == set(lc.SHEET_COLUMNS)
    assert (s["label"] == "").all() and (s["notes"] == "").all()
    assert "llm" not in sheet.read_text() and "stratum" not in sheet.read_text()
    strata = k["stratum"].value_counts().to_dict()
    assert set(strata) == {"yes/chat", "no/chat", "yes/scratch_event", "no/scratch_event", "unknown/scratch_event"}
    assert strata["unknown/scratch_event"] == 4 and min(strata.values()) >= 4  # small strata taken whole
    assert k["item_key"].is_unique
    with pytest.raises(FileExistsError):
        lc.export_handcheck(cands, recs, sheet, key)
    # blank labels are refused
    with pytest.raises(ValueError, match="labels"):
        lc.handcheck_agreement(sheet, key)
    # reviewer agrees except on two rows
    m = s.merge(k[["candidate_id", "llm_label", "evidence_incomplete"]], on="candidate_id")
    s["label"] = m["llm_label"].values
    flip = m.index[(m.llm_label == "yes")][:2]
    s.loc[flip, "label"] = "no"
    s.to_csv(sheet, index=False)
    res = lc.handcheck_agreement(sheet, key, n_boot=300)
    assert res["n"] == 30 and res["n_agree"] == 28
    assert res["raw_agreement"] == pytest.approx(28 / 30)
    assert res["raw_ci_lo"] < 28 / 30 < res["raw_ci_hi"]
    assert 0 < res["kappa_boot_ci_lo"] <= res["kappa"] <= res["kappa_boot_ci_hi"] <= 1
    assert res["confusion"]["human_no"]["llm_yes"] == 2
    assert res["llm_false_positive_rate"] > 0 and res["llm_false_negative_rate"] == 0
    assert set(res["by_kind"]) == {"chat", "scratch_event"}
    # an edited payload is refused by coding.load_labels
    s.loc[0, "payload"] = s.loc[0, "payload"] + " edited"
    s.to_csv(sheet, index=False)
    with pytest.raises(ValueError, match="payload hash"):
        lc.handcheck_agreement(sheet, key)


def test_prereg_quotas_take_all_positives_when_fewer():
    cands = many_candidates()
    recs = {}
    for it in lc.plan_items(cands).itertuples():
        lab = "yes" if ("hold at 20" in it.payload or "keep the price" in it.payload) else "no"
        recs[it.item_key] = {"item_key": it.item_key, "status": "ok", "raw_label": lab, "reason": "r"}
    sample = lc.handcheck_sample(cands, recs, quotas=lc.PREREG_QUOTAS)
    n_yes = sum(1 for r in recs.values() if r["raw_label"] == "yes")
    assert (sample["label"] == "yes").sum() == n_yes          # all positives (fewer than 50)
    assert (sample["label"] == "no").sum() == min(50, (lc.label_table(cands, recs)
                                                      .drop_duplicates("item_key").label == "no").sum())


def test_empty_candidate_table_gives_valid_empty_sheet(tmp_path):
    empty = extract_candidates({"game_id": "g"}, {"chat": [], "chat_deliveries": [], "scratch_events": []})
    assert lc.plan_items(empty).empty
    sheet = lc.write_label_sheet(empty, {}, tmp_path / "labels.csv")
    assert load_labels(sheet, empty, allow_unknown=False).empty


def test_kappa_and_allocation_edge_cases():
    assert lc.cohen_kappa(["yes", "no"], ["yes", "no"]) == pytest.approx(1.0)
    assert lc.cohen_kappa(["no", "no"], ["no", "no"]) != lc.cohen_kappa(["no", "no"], ["no", "no"])  # NaN
    assert lc.cohen_kappa(["yes", "no", "yes", "no"], ["no", "yes", "no", "yes"]) == pytest.approx(-1.0)
    assert lc.allocate({"a": 2, "b": 100, "c": 100}, 30) == {"a": 2, "b": 14, "c": 14}
    assert sum(lc.allocate({"a": 1, "b": 1}, 30).values()) == 2


def test_cli_label_refuses_without_valid_cap(tmp_path):
    with pytest.raises(SystemExit):
        lc.main(["label", str(tmp_path), "--out", str(tmp_path / "o"), "--ledger", str(tmp_path / "l.jsonl"),
                 "--cap", "0"])


def test_pending_items_on_empty_plan_keeps_columns():
    empty = extract_candidates({"game_id": "g"}, {"chat": [], "chat_deliveries": [], "scratch_events": []})
    todo = lc.pending_items(lc.plan_items(empty), {}, rubric_sha256=lc.RUBRIC_SHA256, model_id=lc.JUDGE_MODEL)
    assert todo.empty and {"kind", "payload", "item_key"} <= set(todo.columns)


@pytest.mark.parametrize("text,strict", [
    ('{"label": "yes", "reason": "r"}', True),
    ('```json\n{"label": "no"}\n```', True),
    ('Thinking... {"label": "yes"}', False),
    ("", False),
    (None, False),
])
def test_strict_format_flag(text, strict):
    assert lc.strict_format(text) is strict


def test_records_flag_lenient_format(tmp_path):
    _, _, recs = run_coder(tmp_path)
    flags = {r["raw_label"]: r["strict_format"] for r in recs.values() if r["status"] == "ok"}
    assert flags["yes"] is False      # 'Thinking... {json}' is accepted but flagged
    assert flags["no"] is True        # fenced JSON is the asked-for format


def test_empty_or_truncated_answers_are_retried_not_final(tmp_path):
    def fn(input, tools, tool_choice, config):
        kind, payload = _payload_of(input)
        if "hold at 20" in payload:
            return ModelOutput.from_content(model="mockllm/model", content="")
        if "routine note" in payload:
            return ModelOutput.from_content(model="mockllm/model", content='{"label": "y', stop_reason="max_tokens")
        return ModelOutput.from_content(model="mockllm/model", content='{"label": "no", "reason": "own"}')

    cands = candidates()
    text, sha = rubric()
    path = tmp_path / "judgements.jsonl"
    model = get_model("mockllm/model", custom_outputs=fn)
    todo = lc.pending_items(lc.plan_items(cands), {}, rubric_sha256=sha, model_id=lc.JUDGE_MODEL, rubric=text)
    asyncio.run(lc.code_items(todo, model, path, rubric=text, rubric_sha256=sha, model_id=lc.JUDGE_MODEL,
                              config=GenerateConfig(**lc.JUDGE_CONFIG)))
    recs = lc.read_records(path)
    errs = [r for r in recs.values() if r["status"] == "error"]
    assert len(errs) == 2 and {r["problem"].split(" ")[0] for r in errs} == {"empty", "max_tokens"}
    lt = lc.label_table(cands, recs)
    assert set(lt[lt.status == "error"].label) == {"unknown"}
    again = lc.pending_items(lc.plan_items(cands), recs, rubric_sha256=sha, model_id=lc.JUDGE_MODEL, rubric=text)
    assert set(again["item_key"]) == {r["item_key"] for r in errs}


def test_torn_tail_is_cut_before_appending(tmp_path):
    cands, _, recs = run_coder(tmp_path, rule=lambda k, p: "nonsense")  # all unparseable (done)
    path = tmp_path / "judgements.jsonl"
    with path.open("a") as f:
        f.write('{"item_key": "i-torn", "stat')
    extra = extract_candidates({"game_id": "g2"}, {"chat": [{"t": 1, "firm": "A", "message": "new one",
                                                             "blocked": False}],
                                                    "chat_deliveries": [], "scratch_events": []})
    both = pd.concat([cands, extra], ignore_index=True)
    _, counts, recs2 = run_coder(tmp_path, cands=both)
    assert counts["attempted"] == 1 and len(recs2) == len(recs) + 1  # the file stays readable after the append
    assert (tmp_path / "judgements.jsonl.torn").read_text().startswith('{"item_key": "i-torn"')
    assert lc.repair_torn_tail(path) == 0


def test_sheet_refuses_records_from_another_rubric_model_or_prompt(tmp_path):
    cands, _, recs = run_coder(tmp_path)
    text, sha = rubric()
    lc.write_label_sheet(cands, recs, tmp_path / "ok.csv", rubric_sha256=sha, rubric=text)
    key = next(k for k, r in recs.items() if r["status"] == "ok")
    for field, value in (("rubric_sha256", "0" * 64), ("model", "other/model"), ("prompt_sha256", "f" * 64)):
        bad = {**recs, key: {**recs[key], field: value}}
        with pytest.raises(ValueError, match="fresh output directory"):
            lc.write_label_sheet(cands, bad, tmp_path / "bad.csv", rubric_sha256=sha, rubric=text)


def test_manifest_from_another_rubric_is_refused(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"rubric_sha256": "0" * 64, "judge_model": lc.JUDGE_MODEL}))
    with pytest.raises(ValueError, match="fresh --out"):
        lc.check_manifest(tmp_path, lc.RUBRIC_SHA256)
    (tmp_path / "manifest.json").write_text(json.dumps({"rubric_sha256": lc.RUBRIC_SHA256,
                                                        "judge_model": lc.JUDGE_MODEL,
                                                        "user_template": lc.USER_TEMPLATE}))
    lc.check_manifest(tmp_path, lc.RUBRIC_SHA256)


def test_output_dir_lock_is_exclusive(tmp_path):
    with lc.exclusive_dir(tmp_path):
        with pytest.raises(RuntimeError, match="another llm_coder"):
            with lc.exclusive_dir(tmp_path):
                pass
    with lc.exclusive_dir(tmp_path):  # released afterwards
        pass


def _fake_runs(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    monkeypatch.setattr(lc, "PRIVATE_RUNS", runs)
    monkeypatch.setattr(lc, "FORBIDDEN_TREES", tuple(runs / d for d in ("full", "pilot", "remote")))
    return runs


def test_paths_must_be_private_and_outside_game_trees(monkeypatch, tmp_path):
    runs = _fake_runs(monkeypatch, tmp_path)
    assert lc.check_private_path(runs / "analysis" / "coding", "--out") == (runs / "analysis" / "coding").resolve()
    for bad in (tmp_path / "elsewhere", runs / "full" / "x", runs / "pilot" / "ledger.jsonl", runs / "remote" / "v"):
        with pytest.raises(ValueError):
            lc.check_private_path(bad, "--out")
    game = runs / "analysis" / "shared-ledger.jsonl"
    game.parent.mkdir(parents=True)
    game.write_text('{"event":"init","call_kind":null}\n{"event":"reserve","call_kind":"generate"}\n')
    with pytest.raises(ValueError, match="own ledger"):
        lc.check_coder_ledger(game)
    own = runs / "analysis" / "coder-ledger.jsonl"
    own.write_text('{"event":"init","call_kind":null}\n{"event":"reserve","call_kind":"l2_coder"}\n')
    assert lc.check_coder_ledger(own) == own.resolve()


def test_cli_label_refuses_game_ledgers_and_public_outputs(monkeypatch, tmp_path):
    runs = _fake_runs(monkeypatch, tmp_path)
    for out, ledger in ((runs / "a", runs / "full" / "ledger-mac.jsonl"), (runs / "a", tmp_path / "l.jsonl"),
                        (tmp_path / "public", runs / "a" / "l.jsonl"), (runs / "pilot" / "o", runs / "a" / "l.jsonl")):
        with pytest.raises(SystemExit):
            lc.main(["label", str(tmp_path), "--out", str(out), "--ledger", str(ledger), "--cap", "0.1"])
        assert not ledger.exists()


def test_cli_handcheck_defaults_to_prereg_quotas(monkeypatch, tmp_path, capsys):
    runs = _fake_runs(monkeypatch, tmp_path)
    out = runs / "analysis" / "coding"
    cands, _, recs = run_coder(out, cands=many_candidates(), concurrency=4)
    monkeypatch.setattr(lc, "load_candidates", lambda dirs, cohort, **kw: cands)
    lc.main(["handcheck", "logs", "--out", str(out), "--handcheck-dir", str(runs / "analysis" / "hc")])
    printed = capsys.readouterr().out
    assert "yes/" not in printed and "no/" not in printed  # stratum mix stays in the private key
    key = pd.read_csv(out / "handcheck_key.csv", dtype=str, keep_default_na=False)
    lt = lc.label_table(cands, recs).drop_duplicates("item_key")
    assert (key.llm_label == "yes").sum() == min(50, (lt.label == "yes").sum())
    assert (key.llm_label == "no").sum() == min(50, (lt.label == "no").sum())
    assert (key.llm_label == "unknown").sum() == 0
    with pytest.raises(SystemExit):
        lc.main(["handcheck", "logs", "--out", str(out), "--handcheck-dir", str(runs / "analysis" / "hc2"),
                 "--n", "10", "--prereg"])


def test_lone_surrogate_in_completion_does_not_abort_the_run(tmp_path):
    cands, counts, recs = run_coder(tmp_path, rule=lambda k, p: '{"label": "no", "reason": "own \ud800 note"}')
    assert counts["error"] == 0 and counts["ok"] == len(lc.plan_items(cands))
    _, sha = rubric()
    sheet = lc.write_label_sheet(cands, recs, tmp_path / "labels.csv", rubric_sha256=sha)
    assert len(load_labels(sheet, cands, allow_unknown=True)) == len(cands)


# ------------------------------------------------------------------------------- candidate set = the D12 game set
def write_log(path, samples, *, status: str, created: str, eval_id: str) -> Path:
    """A real Inspect .eval log (no events or messages) holding the given samples."""
    from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSpec, EvalStats, write_eval_log
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = EvalSpec(created=created, task="shc-games", model="mockllm/model", config=EvalConfig(), eval_id=eval_id,
                    dataset=EvalDataset(samples=len(samples), sample_ids=[s.id for s in samples]))
    write_eval_log(EvalLog(eval=spec, status=status, samples=samples, stats=EvalStats(started_at=created)), str(path))
    return path


def eval_sample(meta: dict, *, completed_at: str | None, error: str | None = None):
    """One game sample as the harness logs it: everything in metadata, the result under metadata['result']."""
    from inspect_ai.log import EvalError, EvalSample
    return EvalSample(id=meta["game_id"], epoch=1, input="play", target="", metadata=meta, completed_at=completed_at,
                      error=EvalError(message=error, traceback="", traceback_ansi="") if error else None)


def game_meta(arm: str, seed: int, text: str, *, rounds: int = 2, keep_rounds: int | None = None) -> dict:
    """A complete synthetic game whose chat message and scratch note carry `text` (so copies can be told apart)."""
    gid = f"deep:m1:core-pricing:{arm}:{seed}"
    rows = [{"t": t, "firm": f, "order": o, "action": 10.0 + t, "scripted": False, "parse_failed": False,
             "collusion_index": 0.0} for t in range(1, rounds + 1) for o, f in enumerate("AB", 1)]
    if keep_rounds is not None:
        rows = [r for r in rows if r["t"] <= keep_rounds]
    note = f"{text} note"
    result = {"host": "hostA", "rows": rows, "calls": [], "gate": [], "outputs": {"A": [], "B": []},
              "chat": [{"t": 1, "firm": "A", "recipient": "B", "message": text, "blocked": False}],
              "chat_deliveries": [{"t": 2, "sent_t": 1, "sender": "A", "recipient": "B", "message": text}],
              "scratch_events": [{"t": 1, "firm": "B", "uid": 1002, "order": 2, "op": "create",
                                  "path": "/srv/scratch/note.txt", "content_excerpt": note,
                                  "size": len(note.encode())}]}
    return {"game_id": gid, "grid": "deep", "model_label": "m1", "model_id": "x/m1", "effort": "low",
            "scenario_id": "core-pricing", "engine": "pricing", "arm": arm, "seed": seed, "rounds": rounds,
            "result": result}


def d12_logs(root: Path):
    """Two hosts' log directories after operator stops and relaunches.

    mac/A (16:54, 'started'): X complete 17:05 (kept), W errored partial (superseded), P complete only here.
    mac/B (17:16, 'cancelled'): Y complete 17:50 (dropped: C's copy finished first), Q complete only here.
    vps/C (17:35, 'success'): X complete 18:00 (dropped), W complete, Y complete 17:40 (kept)."""
    X, W, P, Y, Q = (("SHR-PERMITTED", 1000), ("SHR-AWARE", 1000), ("CHAT-SHR", 1000), ("ISO-PERMITTED", 1000),
                     ("CHAT-ISO", 1000))
    mac, vps = root / "full" / "logs-mac", root / "remote" / "vps" / "logs-vps"
    write_log(mac / "2026-10-25T16-54-37_A.eval", [
        eval_sample(game_meta(*X, "kept copy of X"), completed_at="2026-10-25T17:05:00+00:00"),
        eval_sample(game_meta(*W, "errored copy of W", keep_rounds=1), completed_at=None, error="CancelledError()"),
        eval_sample(game_meta(*P, "only copy of P"), completed_at="2026-10-25T17:10:00+00:00")],
        status="started", created="2026-10-25T16:54:37+00:00", eval_id="evA")
    write_log(mac / "2026-10-25T17-16-48_B.eval", [
        eval_sample(game_meta(*Y, "dropped copy of Y"), completed_at="2026-10-25T17:50:00+00:00"),
        eval_sample(game_meta(*Q, "only copy of Q"), completed_at="2026-10-25T17:30:00+00:00")],
        status="cancelled", created="2026-10-25T17:16:48+00:00", eval_id="evB")
    write_log(vps / "2026-10-25T17-35-00_C.eval", [
        eval_sample(game_meta(*X, "dropped copy of X"), completed_at="2026-10-25T18:00:00+00:00"),
        eval_sample(game_meta(*W, "complete copy of W"), completed_at="2026-10-25T18:01:00+00:00"),
        eval_sample(game_meta(*Y, "kept copy of Y"), completed_at="2026-10-25T17:40:00+00:00")],
        status="success", created="2026-10-25T17:35:00+00:00", eval_id="evC")
    ids = {k: f"deep:m1:core-pricing:{a}:{s}" for k, (a, s) in zip("XWPYQ", (X, W, P, Y, Q))}
    kept_text = {"X": "kept copy of X", "W": "complete copy of W", "P": "only copy of P", "Y": "kept copy of Y",
                 "Q": "only copy of Q"}
    return [str(mac), str(vps)], ids, kept_text


def test_candidates_are_report_outputs_candidates_of_the_d12_kept_set(tmp_path):
    from analysis import report_outputs as ro
    dirs, ids, kept_text = d12_logs(tmp_path)
    cands = lc.load_candidates(dirs, "confirmatory")
    t = ro.load_logs(ro.Ctx(cfg=ro.Config(log_dirs=dirs, out=str(tmp_path / "ro"))))      # confirmatory (strict)
    ro_c = t["candidates"]
    assert len(cands) == len(ro_c) == 2 * len(ids) and cands["candidate_id"].is_unique
    both = cands.merge(ro_c, on="candidate_id", suffixes=("", "_ro"), validate="one_to_one")
    assert len(both) == len(cands) and (both["payload"] == both["payload_ro"]).all()
    assert set(cands["game_id"]) == set(ids.values()) == set(t["games"]["game_id"])
    # every game's candidates are coding.extract_candidates of its kept copy: the ids, and the kept copy's payloads
    for key, gid in ids.items():
        _, _, _, arm, seed = gid.split(":")
        meta = game_meta(arm, int(seed), kept_text[key])
        want = extract_candidates({k: v for k, v in meta.items() if k != "result"}, meta["result"])
        got = cands[cands["game_id"] == gid].set_index("candidate_id").sort_index()
        assert got.index.tolist() == sorted(want["candidate_id"])
        assert (got["payload"] == want.set_index("candidate_id").sort_index()["payload"]).all()
        assert got["payload"].str.contains(kept_text[key]).all()
    # whatever the directory order; and the label sheet passes report_outputs' importer for its candidate set
    assert set(lc.load_candidates(list(reversed(dirs)), "confirmatory")["candidate_id"]) == set(cands["candidate_id"])
    sheet = lc.write_label_sheet(cands, {}, tmp_path / "labels.csv")
    assert len(load_labels(sheet, ro_c, allow_unknown=True)) == len(ro_c)
    # the regression: one dataset.load per directory dropped every game of the logs not marked success, and coded
    # the copy of X that D12 drops
    old = pd.concat([dataset.load(d, allow_exclusions=True)["candidates"] for d in dirs], ignore_index=True)
    assert set(old["game_id"]) == {ids["X"], ids["W"], ids["Y"]}
    assert old.loc[old["game_id"] == ids["X"], "payload"].str.contains("dropped copy of X").all()


def test_candidate_set_counts_and_strictness(tmp_path, capsys):
    dirs, ids, _ = d12_logs(tmp_path)
    loaded = lc.candidate_set(dirs, "confirmatory")
    n = lc.game_set_counts(loaded)
    assert n == {"logs": 3, "games_kept": 5, "games_analysed": 5, "games_kept_not_analysed": 0,
                 "games_without_complete_sample": 0, "games_from_logs_not_success": 3, "duplicates_dropped": 2,
                 "superseded": 1, "candidates": 10}
    assert {g: Path(game_set.short_log(r.log)).name[-6:] for g, r in loaded.game_set.kept.items()} == {
        ids["X"]: "A.eval", ids["W"]: "C.eval", ids["P"]: "A.eval", ids["Y"]: "C.eval", ids["Q"]: "B.eval"}
    lc.load_candidates(dirs, "confirmatory")
    assert "5 games, 10 candidates" in capsys.readouterr().err
    missing = str(tmp_path / "nowhere")
    with pytest.raises(ValueError, match="log directory missing"):
        lc.load_candidates([*dirs, missing], "confirmatory")
    assert len(lc.load_candidates([*dirs, missing], "confirmatory", strict=False)) == 10
    assert "missing, skipped" in capsys.readouterr().err


REAL_DIRS = sorted(glob.glob(str(ROOT / "runs" / "full" / "logs-*"))
                   + glob.glob(str(ROOT / "runs" / "remote" / "*" / "logs-*")))


@pytest.mark.skipif(not REAL_DIRS, reason="no local confirmatory log directories")
def test_counts_only_candidates_cover_the_real_kept_games():
    """On runs/full/logs-* and runs/remote/*/logs-* (lenient: live logs may be mid-flush): the candidates are those of
    the D12 kept copies, and every kept game is analysed or a recorded exclusion of its kept copy. Only counts are
    printed (run with -s); no payload, label or outcome is read out."""
    loaded = lc.candidate_set(REAL_DIRS, "confirmatory", strict=False)
    n = lc.game_set_counts(loaded)
    print("\n" + json.dumps(n))
    gs, t = loaded.game_set, loaded.tables
    games, cands, ex = t["games"], t["candidates"], t["exclusions"]
    analysed = set(games["game_id"])
    assert cands["candidate_id"].is_unique and set(cands["game_id"]) <= analysed <= set(gs.kept)
    assert games["game_id"].is_unique
    assert all(gs.kept[g].log == str(lg) for g, lg in zip(games["game_id"], games["log"]))   # the kept copies
    kept_excluded = set(ex.loc[ex["selection"] == game_set.KEPT, "game_id"]) if len(ex) else set()
    assert set(gs.kept) == analysed | kept_excluded                     # every kept game is covered
    assert n["games_kept"] == len(game_set.scan(REAL_DIRS).kept)        # the same game count as the D12 audit


def test_ext_rep_cohort_reads_extension_loader_and_keeps_e1_only(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from analysis import extensions
    cands = pd.DataFrame({"candidate_id": ["c1", "c2", "c3"], "kind": ["chat"] * 3, "payload": ["a", "b", "c"],
                          "game_id": ["ext-rep:glm-5.3:core-pricing:CHAT-SHR:2000",
                                      "ext-long:glm-5.3:core-pricing:SHR-AWARE:3000",
                                      "ext-rep:gpt-6.1-sol:core-pricing:CHAT-ISO:2001"]})
    seen = {}

    def fake_load_ext(dirs, *, strict, notes):
        seen["dirs"], seen["strict"] = dirs, strict
        notes.append("probe note")
        return SimpleNamespace(tables={"candidates": cands})

    monkeypatch.setattr(extensions, "load_ext", fake_load_ext)
    monkeypatch.setattr(lc, "game_set_counts", lambda loaded: {k: 0 for k in (
        "logs", "games_analysed", "candidates", "games_from_logs_not_success", "duplicates_dropped", "superseded",
        "games_kept_not_analysed", "games_without_complete_sample")})
    out = lc.load_candidates([str(tmp_path)], lc.EXT_REP, strict=True)
    assert seen == {"dirs": [str(tmp_path)], "strict": True}
    assert list(out["candidate_id"]) == ["c1", "c3"]
