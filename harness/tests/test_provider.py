"""Tests for harness.nanogpt_provider (CONTRACT.md H1).

Mock tests use httpx.MockTransport through Inspect's own get_model()/Model.generate path. The real-call test is skipped
unless SHC_REAL_CALLS=<n> (1..3) is set. It then makes n calls to z-ai/glm-5.2 (reasoning_effort "none", max_tokens 20,
no retries) under a $2 ledger cap, and checks that ModelOutput.metadata["cost_usd"] matches the ledger.
"""
from __future__ import annotations

import asyncio
import json
import os

import anyio
import httpx
import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model

from harness.ledger import BudgetBreach, BudgetExceeded, BudgetLedger, LedgerError
from harness.nanogpt_provider import (
    NanoGPTAPI,
    NanoGPTConfigError,
    NanoGPTHTTPError,
    call_kind,
    estimate_max_cost,
    load_prices,
)

MODEL = "z-ai/glm-5.2"
CFG = GenerateConfig(max_tokens=20, temperature=0.0, reasoning_effort="none", seed=11)
PROMPT = 'Reply {"price": 16}'


def ok_body(cost: float | None = 1.2e-05, content: str = '{"price": 16}', **msg) -> dict:
    usage = {"prompt_tokens": 43, "completion_tokens": 9, "total_tokens": 52, "reasoning_tokens": 0,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "currency": "USD"}
    if cost is not None:
        usage["cost"] = cost
    return {"id": "chatcmpl-x", "model": "glm-5.2-served", "usage": usage,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content,
                                                                         **msg}}]}


def lines(path) -> list[dict]:
    return [v for x in open(path, "rb").read().splitlines() for v in [json.loads(x)]
            if v["event"] not in ("init", "cap_change")]


@pytest.fixture
def env(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    BudgetLedger(path, 10.0)
    key = tmp_path / "key"
    key.write_text("sk-test-key-123\n")
    monkeypatch.setenv("SHC_LEDGER_PATH", str(path))
    monkeypatch.setenv("SHC_BUDGET_CAP_USD", "10.0")
    return {"ledger": path, "key": str(key)}


class Recorder:
    """MockTransport handler that replays a script of responses/exceptions and records every request."""

    def __init__(self, *script):
        self.script, self.requests = list(script), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, Exception):
            raise step
        status, body = step
        return httpx.Response(status, json=body)


def model_for(env, handler, **kw):
    return get_model(f"nanogpt/{MODEL}", memoize=False, transport=httpx.MockTransport(handler), key_path=env["key"],
                     backoff_s=0.0, **kw)


def expected_est(messages: list[dict], max_tokens: int = 20) -> float:
    return estimate_max_cost(load_prices()[MODEL], len(json.dumps(messages, ensure_ascii=False).encode("utf-8")),
                             max_tokens, len(messages))


def test_reservation_uses_byte_bound_capped_by_context():
    price = load_prices()[MODEL]
    assert estimate_max_cost(price, 1_000, 20) < estimate_max_cost(price, 100_000, 20)      # grows with input bytes
    huge = estimate_max_cost(price, 10 * price["context_length"], 20)
    assert huge == estimate_max_cost(price, 20 * price["context_length"], 20)               # never above context
    # multi-byte text: the bound is in bytes, so 4-byte characters reserve 4 tokens each, not 1/2.5
    emoji = "\U0001F600" * 1000
    assert len(emoji.encode("utf-8")) == 4000
    assert estimate_max_cost(price, len(emoji.encode("utf-8")), 20) > estimate_max_cost(price, len(emoji), 20)
    with pytest.raises(NanoGPTConfigError):
        estimate_max_cost({**price, "context_length": None}, 1, 20)


def test_registered_under_nanogpt(env):
    m = model_for(env, Recorder((200, ok_body())))
    assert isinstance(m.api, NanoGPTAPI)
    assert str(m) == f"nanogpt/{MODEL}" and m.api.model_name == MODEL


def test_success_cost_matches_ledger(env):
    rec = Recorder((200, ok_body(cost=1.2e-05, reasoning="thinking about it")))
    m = model_for(env, rec)
    msgs = [ChatMessageSystem(content="Reply only with JSON."), ChatMessageUser(content=PROMPT)]
    with call_kind("firm_move"):
        out = asyncio.run(m.generate(msgs, config=CFG))
    assert out.completion == '{"price": 16}'
    reasoning = [c for c in out.message.content if getattr(c, "type", None) == "reasoning"]
    assert reasoning and reasoning[0].reasoning == "thinking about it"
    md = out.metadata
    assert md["cost_usd"] == 1.2e-05 and md["cost_source"] == "usage.cost"
    assert md["provider_model"] == "glm-5.2-served" and md["finish_reason"] == "stop"
    assert md["raw_usage"]["prompt_tokens"] == 43 and md["total_cost_usd"] == 1.2e-05
    assert out.usage.input_tokens == 43 and out.usage.output_tokens == 9 and out.usage.reasoning_tokens == 0
    assert out.stop_reason == "stop"
    ev = lines(env["ledger"])
    assert [e["event"] for e in ev] == ["reserve", "settle"]
    assert ev[1]["id"] == md["reservation_id"] and ev[1]["actual"] == md["cost_usd"]
    assert ev[0]["call_kind"] == "firm_move" and ev[0]["model"] == MODEL
    body = json.loads(rec.requests[0].content)
    assert body["max_tokens"] == 20 and body["temperature"] == 0.0 and body["reasoning_effort"] == "none"
    assert body["seed"] == 11 and body["model"] == MODEL and body["stream"] is False
    assert ev[0]["est_max"] == pytest.approx(expected_est(body["messages"]))
    assert md["est_max_usd"] == pytest.approx(ev[0]["est_max"])
    assert rec.requests[0].headers["authorization"] == "Bearer sk-test-key-123"
    assert str(rec.requests[0].url) == "https://nano-gpt.com/api/v1/chat/completions"
    assert m.api.ledger.spent == pytest.approx(1.2e-05)


def test_model_call_log_has_no_key(env):
    m = model_for(env, Recorder((200, ok_body())))
    api: NanoGPTAPI = m.api
    out, call = asyncio.run(api.generate([ChatMessageUser(content=PROMPT)], [], "none", CFG))
    assert "sk-test-key-123" not in json.dumps(call.request) + json.dumps(call.response)
    assert "sk-test-key-123" not in open(env["ledger"]).read()
    assert out.metadata["cost_usd"] == 1.2e-05


def test_missing_cost_with_tokens_settles_at_conservative_estimate(env):
    m = model_for(env, Recorder((200, ok_body(cost=None))))
    out = asyncio.run(m.generate(PROMPT, config=CFG))
    ev = lines(env["ledger"])
    assert [e["event"] for e in ev] == ["reserve", "settle"]
    assert out.metadata["cost_source"] == "provider_estimate"
    assert 0 < out.metadata["cost_usd"] == ev[1]["actual"] < ev[0]["est_max"]


def test_missing_cost_and_tokens_settles_at_est_max(env):
    body = ok_body(cost=None)
    body["usage"] = {}
    m = model_for(env, Recorder((200, body)))
    out = asyncio.run(m.generate(PROMPT, config=CFG))
    ev = lines(env["ledger"])
    assert [e["event"] for e in ev] == ["reserve", "settle", "flag"]
    assert ev[2]["reason"] == "missing_cost:http_200"
    assert out.metadata["cost_source"] == "est_max"
    assert out.metadata["cost_usd"] == ev[1]["actual"] == ev[0]["est_max"]
    assert m.api.ledger.spent == pytest.approx(ev[0]["est_max"])


def test_provider_charge_above_reservation_halts_future_http(env):
    est = expected_est([{"role": "user", "content": PROMPT}])
    rec = Recorder((200, ok_body(cost=2 * est)))
    m = model_for(env, rec)
    with pytest.raises(BudgetBreach):
        asyncio.run(m.generate(PROMPT, config=CFG))
    assert len(rec.requests) == 1
    led = BudgetLedger(env["ledger"], 10.0, create=False)
    assert led.spent == pytest.approx(2 * est) and led.flags == 1
    with pytest.raises(BudgetBreach):
        asyncio.run(m.generate(PROMPT, config=CFG))
    assert len(rec.requests) == 1


def test_429_then_success_uses_two_reservations(env):
    rec = Recorder((429, {"error": "rate limited"}), (200, ok_body(cost=2e-05)))
    m = model_for(env, rec)
    out = asyncio.run(m.generate(PROMPT, config=CFG))
    assert len(rec.requests) == 2
    ev = lines(env["ledger"])
    assert [e["event"] for e in ev] == ["reserve", "settle", "flag", "reserve", "settle"]
    assert ev[0]["id"] != ev[3]["id"] and ev[2]["reason"] == "missing_cost:http_429"
    md = out.metadata
    assert md["reservation_id"] == ev[3]["id"] and md["cost_usd"] == ev[4]["actual"] == 2e-05
    assert [a["outcome"] for a in md["attempts"]] == ["http_error", "ok"]
    assert md["total_cost_usd"] == pytest.approx(ev[0]["est_max"] + 2e-05)
    assert m.api.ledger.spent == pytest.approx(md["total_cost_usd"])


def test_timeout_settles_each_attempt_at_est_max_and_stops_after_two_retries(env):
    rec = Recorder(httpx.ReadTimeout("slow"))
    m = model_for(env, rec)
    with pytest.raises(Exception) as ei:
        asyncio.run(m.generate(PROMPT, config=CFG))
    assert "ReadTimeout" in repr(ei.value) or "slow" in str(ei.value)
    assert len(rec.requests) == 3                     # 1 attempt + 2 retries, no Inspect-level retry on top
    ev = lines(env["ledger"])
    assert [e["event"] for e in ev] == ["reserve", "settle", "flag"] * 3
    assert {e["reason"] for e in ev if e["event"] == "flag"} == {"transport_error:ReadTimeout"}
    led = m.api.ledger
    assert led.outstanding == 0 and led.spent == pytest.approx(3 * ev[0]["est_max"])


def test_connect_error_releases(env):
    rec = Recorder(httpx.ConnectError("refused"))
    m = model_for(env, rec)
    with pytest.raises(Exception):
        asyncio.run(m.generate(PROMPT, config=CFG))
    assert len(rec.requests) == 3
    ev = lines(env["ledger"])
    assert [e["event"] for e in ev] == ["reserve", "release"] * 3
    assert m.api.ledger.spent == 0 and m.api.ledger.outstanding == 0


def test_no_retry_on_400_and_retry_on_500(env):
    rec = Recorder((400, {"error": "bad request"}))
    m = model_for(env, rec)
    with pytest.raises(Exception) as ei:
        asyncio.run(m.generate(PROMPT, config=CFG))
    assert len(rec.requests) == 1 and "400" in str(ei.value)
    rec5 = Recorder((500, {"error": "boom"}), (502, {"error": "boom"}), (200, ok_body(cost=1e-05)))
    out = asyncio.run(model_for(env, rec5).generate(PROMPT, config=CFG))
    assert len(rec5.requests) == 3 and out.metadata["cost_usd"] == 1e-05


def test_error_body_cost_is_used_when_present(env):
    rec = Recorder((400, {"error": "x", "usage": {"cost": 3e-06}}))
    m = model_for(env, rec)
    out, _ = asyncio.run(m.api.generate([ChatMessageUser(content=PROMPT)], [], "none", CFG))
    assert isinstance(out, NanoGPTHTTPError) and out.status_code == 400
    ev = lines(env["ledger"])
    assert [e["event"] for e in ev] == ["reserve", "settle"] and ev[1]["actual"] == 3e-06


def test_cancellation_settles_at_est_max(env):
    async def slow(request: httpx.Request) -> httpx.Response:
        await anyio.sleep(30)
        return httpx.Response(200, json=ok_body())

    api: NanoGPTAPI = get_model(f"nanogpt/{MODEL}", memoize=False, transport=httpx.MockTransport(slow),
                                key_path=env["key"]).api

    async def go():
        with anyio.move_on_after(0.3):
            await api.generate([ChatMessageUser(content=PROMPT)], [], "none", CFG)
    asyncio.run(go())
    ev = lines(env["ledger"])
    assert [e["event"] for e in ev] == ["reserve", "settle", "flag"]
    assert ev[2]["reason"].startswith("aborted:") and ev[1]["actual"] == ev[0]["est_max"]
    assert api.ledger.outstanding == 0


def test_fail_closed_before_any_http(env, monkeypatch, tmp_path):
    rec = Recorder((200, ok_body()))
    monkeypatch.delenv("SHC_LEDGER_PATH")
    with pytest.raises(NanoGPTConfigError):
        model_for(env, rec)
    monkeypatch.setenv("SHC_LEDGER_PATH", str(tmp_path / "missing.jsonl"))
    with pytest.raises(LedgerError):
        model_for(env, rec)
    monkeypatch.setenv("SHC_LEDGER_PATH", str(env["ledger"]))
    monkeypatch.setenv("SHC_BUDGET_CAP_USD", "lots")
    with pytest.raises(NanoGPTConfigError):
        model_for(env, rec)
    monkeypatch.setenv("SHC_BUDGET_CAP_USD", "10.0")
    with pytest.raises(NanoGPTConfigError):
        get_model("nanogpt/unknown/model", memoize=False, transport=httpx.MockTransport(rec))
    m = model_for(env, rec)
    with pytest.raises(NanoGPTConfigError):
        asyncio.run(m.generate(PROMPT, config=GenerateConfig(max_tokens=20, attempt_timeout=30)))
    m_nokey = get_model(f"nanogpt/{MODEL}", memoize=False, transport=httpx.MockTransport(rec),
                        key_path=str(tmp_path / "nokey"))
    with pytest.raises(NanoGPTConfigError):
        asyncio.run(m_nokey.generate(PROMPT, config=CFG))
    with open(env["ledger"], "ab") as f:          # an unreadable (corrupt) ledger also stops the call
        f.write(b"garbage\n")
    with pytest.raises(LedgerError):
        asyncio.run(m.generate(PROMPT, config=CFG))
    assert rec.requests == []
    assert all(e.get("event") != "reserve" for e in (json.loads(x) for x in open(env["ledger"], "rb").read()
                                                    .splitlines() if x.startswith(b"{")))


def test_budget_exceeded_makes_no_http_call(env, monkeypatch):
    small = env["ledger"].with_name("small-ledger.jsonl")
    BudgetLedger(small, 0.001)                            # below one GLM 5.2 reservation (~$0.05)
    monkeypatch.setenv("SHC_LEDGER_PATH", str(small))
    monkeypatch.setenv("SHC_BUDGET_CAP_USD", "0.001")
    rec = Recorder((200, ok_body()))
    with pytest.raises(BudgetExceeded):
        asyncio.run(model_for(env, rec).generate(PROMPT, config=CFG))
    assert rec.requests == [] and lines(small) == []


def test_default_max_tokens_is_always_sent(env):
    rec = Recorder((200, ok_body()))
    asyncio.run(model_for(env, rec).generate(PROMPT))
    body = json.loads(rec.requests[0].content)
    assert body["max_tokens"] == 4096 and "temperature" not in body and "reasoning_effort" not in body


def test_inside_inspect_eval_cost_reaches_eval_log(env, tmp_path):
    """Full Inspect eval: per-call cost lands in the .eval log and ledger lines carry the sample id."""
    from inspect_ai import Task, eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.log import read_eval_log
    from inspect_ai.solver import generate

    rec = Recorder((200, ok_body(cost=4e-06)))
    model = model_for(env, rec)
    task = Task(dataset=[Sample(input=PROMPT, target="16", id="s1")], solver=generate())
    [log] = inspect_eval(task, model=model, log_dir=str(tmp_path / "logs"), display="none",
                         max_tokens=20, reasoning_effort="none")
    assert log.status == "success", log.error
    sample = read_eval_log(log.location).samples[0]
    assert sample.output.metadata["cost_usd"] == 4e-06
    ev = lines(env["ledger"])
    assert ev[0]["sample_id"] == "s1#1" and ev[1]["actual"] == 4e-06
    assert ev[1]["id"] == sample.output.metadata["reservation_id"]


# --------------------------------------------------------------------------------------------------- real calls

REAL = int(os.environ.get("SHC_REAL_CALLS", "0") or 0)


@pytest.mark.skipif(not 1 <= REAL <= 3, reason="set SHC_REAL_CALLS=1..3 to spend real money (cap $2)")
def test_real_calls_cost_matches_ledger(tmp_path, monkeypatch):
    path = tmp_path / "real-ledger.jsonl"
    BudgetLedger(path, 2.0)
    monkeypatch.setenv("SHC_LEDGER_PATH", str(path))
    monkeypatch.setenv("SHC_BUDGET_CAP_USD", "2.0")
    model = get_model(f"nanogpt/{MODEL}", memoize=False, retries=0,
                      config=GenerateConfig(max_tokens=20, reasoning_effort="none"))

    async def go():
        return [await model.generate([ChatMessageSystem(content="Reply only with a JSON object."),
                                      ChatMessageUser(content='Reply {"price": <number>} for a $16 product.')])
                for _ in range(REAL)]
    outs = asyncio.run(go())
    ev = lines(path)
    settles = {e["id"]: e for e in ev if e["event"] == "settle"}
    for out in outs:
        md = out.metadata
        assert md["cost_source"] in ("usage.cost", "provider_estimate"), md
        assert settles[md["reservation_id"]]["actual"] == md["cost_usd"]
        assert 0 < md["cost_usd"] < md["est_max_usd"]
    led = BudgetLedger(path, 2.0)
    assert led.spent == pytest.approx(sum(o.metadata["cost_usd"] for o in outs)) and led.outstanding == 0
    assert len([e for e in ev if e["event"] == "reserve"]) == REAL
    summary = [{"completion": o.completion, "cost_usd": o.metadata["cost_usd"],
                "est_max_usd": round(o.metadata["est_max_usd"], 8), "reservation_id": o.metadata["reservation_id"],
                "provider_model": o.metadata["provider_model"], "finish_reason": o.metadata["finish_reason"],
                "usage_total_cost": o.usage.total_cost, "tokens": [o.usage.input_tokens, o.usage.output_tokens,
                                                                    o.usage.reasoning_tokens]} for o in outs]
    print("\nREAL-CALL SUMMARY", json.dumps({"ledger_spent": led.spent, "calls": summary}, indent=1))
    print("LEDGER LINES", json.dumps(ev, indent=1))


def test_estimated_charge_is_conservative_and_bounded():
    from harness.nanogpt_provider import _estimated_charge, _usage_cost
    price = {"in_per_m": 1.0, "out_per_m": 4.0}
    est = {"usage": {"prompt_tokens": 1000, "completion_tokens": 100, "reasoning_tokens": 50},
           "x_nanogpt_pricing": {"cost": 1e-4, "currency": "USD", "isEstimate": True}}
    assert _usage_cost(est) is None                               # an estimate is never taken as the settled charge
    bound = 1.2 * (1000 * 1.0 + 150 * 4.0) / 1e6
    assert _estimated_charge(est, price, 1.0) == pytest.approx(max(bound, 1e-4))
    est["x_nanogpt_pricing"]["cost"] = 0.5                        # a large provider estimate wins over the token bound
    assert _estimated_charge(est, price, 1.0) == pytest.approx(0.5)
    assert _estimated_charge(est, price, 0.2) == pytest.approx(0.2)   # never above est_max
    assert _estimated_charge({"usage": {}}, price, 1.0) is None       # no token counts: settle at est_max
    exact = {"usage": {"prompt_tokens": 5}, "x_nanogpt_pricing": {"cost": 2e-6, "currency": "USD"}}
    assert _usage_cost(exact) == pytest.approx(2e-6)              # final pricing block (no isEstimate) is exact
