"""Offline tests for the L2 / chat-coordination LLM classifier (mockllm only; no paid calls)."""
from __future__ import annotations

import asyncio
import json

import pandas as pd
import pytest
from inspect_ai.model import GenerateConfig, ModelOutput, get_model

from analysis import llm_coder as lc
from analysis.coding import extract_candidates, load_labels, merge_labels

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
    monkeypatch.setattr(lc, "load_candidates", lambda dirs, cohort: cands)
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
