"""Inspect ModelAPI provider "nanogpt": nanoGPT chat completions behind the fail-closed BudgetLedger (CONTRACT.md H1).

Usage
    import harness.nanogpt_provider  # noqa: F401  (registers the provider with Inspect)
    model = get_model("nanogpt/z-ai/glm-5.3", config=GenerateConfig(max_tokens=400, reasoning_effort="low"))

Environment, read when the provider is constructed. If either is missing, or the ledger cannot be read, construction
raises before any HTTP call:
    SHC_LEDGER_PATH      path of an EXISTING ledger file. Create it once with BudgetLedger(path, cap); the provider
                         never creates it, so a mistyped path cannot silently start a fresh budget.
    SHC_BUDGET_CAP_USD   global cap in USD.
The API key is read from ~/.nanogpt-api-key on every attempt. It is never logged, and ModelCall records no headers.

Each HTTP attempt
    reserved_max = 1.2 * (advertised_context_tokens * in_per_m + max_tokens * out_per_m) / 1e6.
    This reserves against the provider-advertised full context, not a character-to-token heuristic. The local
    admission ceiling is conditional on that context limit and the pinned maximum catalogue prices; a provider charge
    above the reservation is recorded, halts future calls, and requires an audit because a bill cannot be undone.
    reserve(est_max) -> POST -> settle(usage.cost). The outcomes:
      - The response has no usage.cost (including 429/5xx bodies): settle at est_max, flagged (fail closed).
      - A transport error after the connection was made (read/write timeout, protocol error), or a cancellation:
        settle at est_max, flagged.
      - The connection was never made (httpx.ConnectError, ConnectTimeout, PoolTimeout): release.
    At most 2 retries (on 429, 5xx and transport errors), each with a new reservation. No retry on other 4xx.
    Inspect's own retry is switched off (should_retry -> False), so these are the only retries. Inspect still retries
    AttemptTimeoutError by itself, so GenerateConfig.attempt_timeout is refused; use the model arg request_timeout_s.

ModelOutput.metadata carries:
    cost_usd        the settled cost of the reservation that produced this output (equals its ledger settle line)
    reservation_id
    provider_model
    finish_reason
    raw_usage
    est_max_usd
    cost_source     "usage.cost" or "est_max"
    attempts        one {reservation_id, outcome, settled_usd, status} per HTTP attempt
    total_cost_usd  the sum over all attempts
usage.total_cost is set to total_cost_usd, unless Inspect's model database has prices for the model and overwrites it.
metadata is authoritative.

Price table: `python -m harness.nanogpt_provider build-prices [--source PATH] [--out PATH]` rebuilds model_prices.json
from the nanoGPT /models?detailed=true dump. The estimating price is the highest per-token price the catalogue lists
for the model, so the estimate stays an upper bound when an observed discount lapses or a prompt-cache write is billed:
    in_per_m  = max(pricing.prompt, list input, list input above the long-context threshold, cache-write input)
    out_per_m = max(pricing.completion, list output, list output above the threshold)
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import contextvars
import hashlib
import json
import math
import os
import pathlib
import time
from datetime import datetime, timezone
from typing import Any, Iterator

import anyio
import httpx
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    ModelAPI,
    ModelCall,
    ModelOutput,
    ModelUsage,
    StopReason,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from harness.ledger import BudgetLedger

BASE_URL = "https://nano-gpt.com/api/v1"
KEY_PATH = "~/.nanogpt-api-key"
PRICES_PATH = pathlib.Path(__file__).with_name("model_prices.json")
MODELS_DUMP = "/private/tmp/nanogpt-models.json"
ENV_LEDGER, ENV_CAP = "SHC_LEDGER_PATH", "SHC_BUDGET_CAP_USD"

SAFETY = 1.2
MAX_RETRIES = 2
DEFAULT_MAX_TOKENS = 4096        # sent when the config sets none, so the reservation always bounds the output
DEFAULT_READ_TIMEOUT = 300.0     # seconds per attempt (model arg request_timeout_s)
CONNECT_TIMEOUT = 15.0
MAX_RETRY_AFTER = 30.0

# Roster (13 models) as listed for H1. Compliance/judge models must be added here and the table rebuilt.
ROSTER = (
    "anthropic/claude-opus-5.5", "openai/gpt-6.1-sol", "google/gemini-3.1-pro-preview", "z-ai/glm-5.3",
    "deepseek/deepseek-v4-pro", "anthropic/claude-sonnet-5.5", "x-ai/grok-4.7", "moonshotai/kimi-k3.1-pro",
    "qwen/qwen3.8-max-prime", "z-ai/glm-5.2", "abliteration-ai/abliterated-model-large",
    "deepseek/deepseek-v4.1-flash-1017", "abliteration-ai/abliterated-model-large-flash",
)

_STOP: dict[str, StopReason] = {"stop": "stop", "length": "max_tokens", "content_filter": "content_filter",
                                "tool_calls": "tool_calls", "function_call": "tool_calls"}
_CALL_KIND: contextvars.ContextVar[str] = contextvars.ContextVar("shc_call_kind", default="generate")


class NanoGPTConfigError(RuntimeError):
    """Provider misconfiguration (env, price table, key, unsupported input). Raised before any reservation."""


class NanoGPTHTTPError(RuntimeError):
    """nanoGPT answered with an error status, or with a 200 that holds no completion."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"nanoGPT HTTP {status}: {body[:500]}")
        self.status_code = status


@contextlib.contextmanager
def call_kind(kind: str) -> Iterator[None]:
    """Label ledger lines for model calls made inside this block, e.g. `with call_kind("judge"): ...`."""
    tok = _CALL_KIND.set(kind)
    try:
        yield
    finally:
        _CALL_KIND.reset(tok)


# ------------------------------------------------------------------------------------------------------ price table

def load_prices(path: str | os.PathLike = PRICES_PATH) -> dict[str, dict]:
    """Model id -> price entry ($ per 1M tokens) from model_prices.json."""
    with open(path) as f:
        return json.load(f)["models"]


def estimate_max_cost(price: dict, input_chars: int, max_tokens: int) -> float:
    """Conservative USD reservation under pinned prices and the provider-advertised context/output limits.

    ``input_chars`` is retained for call compatibility and validation; it is never used as a token-count bound.
    A UTF-8 character count divided by a tokenizer average is not an upper bound for adversarial input.
    """
    context = price.get("context_length")
    max_output = price.get("max_output_tokens")
    if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
        raise NanoGPTConfigError("model price entry needs a positive advertised context_length")
    if input_chars < 0 or max_tokens <= 0 or (max_output is not None and max_tokens > max_output):
        raise NanoGPTConfigError("input length or output cap is invalid for the priced model")
    if not all(isinstance(price.get(k), (int, float)) and math.isfinite(price[k]) and price[k] >= 0
               for k in ("in_per_m", "out_per_m")):
        raise NanoGPTConfigError("model price entry has invalid token prices")
    return SAFETY * (context * price["in_per_m"] + max_tokens * price["out_per_m"]) / 1e6


def build_price_table(source: str = MODELS_DUMP, model_ids: tuple[str, ...] = ROSTER) -> dict:
    """Price entries for model_ids from a nanoGPT /models?detailed=true dump (see module docstring for the rule)."""
    raw = pathlib.Path(source).read_bytes()
    by_id = {m.get("id"): m for m in json.loads(raw)}
    models: dict[str, dict] = {}
    for mid in model_ids:
        m = by_id.get(mid)
        if m is None:
            raise KeyError(f"{mid} not in {source}")
        p = m.get("pricing") or {}
        if p.get("unit") != "per_million_tokens" or p.get("currency") != "USD":
            raise ValueError(f"{mid}: unexpected pricing unit {p.get('unit')!r} / {p.get('currency')!r}")
        eff = m.get("effective_pricing") or {}
        if eff and (eff.get("unit") != "per_1k_tokens" or eff.get("currency") != "USD"):
            raise ValueError(f"{mid}: unexpected effective_pricing unit {eff.get('unit')!r}")
        lp = eff.get("list_price") or {}

        def per_m(k: str, src: dict) -> float | None:
            return None if src.get(k) is None else round(float(src[k]) * 1000, 9)

        cat_in, cat_out = round(float(p["prompt"]), 9), round(float(p["completion"]), 9)
        ins = {"catalog": cat_in, "list": per_m("inputPer1kTokens", lp),
               "list_above_threshold": per_m("inputPer1kTokensAbove200k", lp),
               "cache_write": per_m("cacheWriteInputPer1kTokens", p)}
        outs = {"catalog": cat_out, "list": per_m("outputPer1kTokens", lp),
                "list_above_threshold": per_m("outputPer1kTokensAbove200k", lp)}
        ins = {k: v for k, v in ins.items() if v is not None}
        outs = {k: v for k, v in outs.items() if v is not None}
        models[mid] = {
            "name": m.get("name"),
            "in_per_m": max(ins.values()), "out_per_m": max(outs.values()),
            "catalog_in_per_m": cat_in, "catalog_out_per_m": cat_out,
            "in_candidates": ins, "out_candidates": outs,
            "pricing_type": eff.get("pricing_type"),
            "context_length": m.get("context_length"), "max_output_tokens": m.get("max_output_tokens"),
            "reasoning_efforts": m.get("reasoning_efforts"),
        }
    return {
        "_meta": {
            "source": source, "source_sha256": hashlib.sha256(raw).hexdigest(),
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "unit": "USD per 1M tokens",
            "estimating_rule": "in_per_m = max(pricing.prompt, list input, list input above threshold, cache-write "
                               "input); out_per_m = max(pricing.completion, list output, list output above "
                               "threshold). catalog_* = the catalogue `pricing` field (actual costs observed ~10% "
                               "below it in the Step 2 probe).",
            "est_formula": "1.2 * (advertised_context_length * in_per_m + max_tokens * out_per_m) / 1e6",
        },
        "models": models,
    }


# --------------------------------------------------------------------------------------------------------- provider

def _wire_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Text-only OpenAI chat messages. Anything else raises: the game loop is text-in/text-out."""
    out = []
    for m in messages:
        if m.role not in ("system", "user", "assistant"):
            raise NanoGPTConfigError(f"nanogpt provider does not support {m.role!r} messages")
        if m.role == "assistant" and m.tool_calls:
            raise NanoGPTConfigError("nanogpt provider does not support tool calls")
        if not isinstance(m.content, str):
            bad = [c.type for c in m.content if c.type not in ("text", "reasoning")]
            if bad:
                raise NanoGPTConfigError(f"nanogpt provider supports text content only (got {bad})")
        out.append({"role": m.role, "content": m.text})
    return out


def _usage_cost(payload: Any) -> float | None:
    """usage.cost when it is a finite non-negative USD number, else None (settled at est_max)."""
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    cost = usage.get("cost")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
        return None
    if usage.get("currency") not in (None, "USD"):
        return None
    return float(cost)


def _int(x: Any) -> int | None:
    return int(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _active_sample_id() -> str | None:
    try:
        from inspect_ai.log._samples import sample_active
        s = sample_active()
        return None if s is None else f"{s.sample.id}#{s.epoch}"
    except Exception:  # noqa: BLE001  (labelling only; never block a call on it)
        return None


def _retry_after(resp: httpx.Response | None) -> float | None:
    if resp is None:
        return None
    try:
        return min(MAX_RETRY_AFTER, max(0.0, float(resp.headers.get("retry-after", ""))))
    except ValueError:
        return None


class NanoGPTAPI(ModelAPI):
    """nanoGPT chat completions with a reservation in the global BudgetLedger before every HTTP attempt.

    Model args (`-M` / get_model kwargs), mainly for tests:
        transport    httpx.AsyncBaseTransport (e.g. httpx.MockTransport)
        key_path     API key file (default ~/.nanogpt-api-key)
        prices_path  price table (default harness/model_prices.json)
        retries            0..2 (default 2)
        backoff_s          base sleep before a retry; retry k (1-based) waits backoff_s * 3**(k-1), or Retry-After
        request_timeout_s  httpx read/write/pool timeout per attempt (default 300; connect timeout 15)

    GenerateConfig.attempt_timeout is refused (NanoGPTConfigError, before any reservation): Inspect retries attempt
    timeouts on its own, which would break the 2-retry rule.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        key_path: str = KEY_PATH,
        prices_path: str | os.PathLike = PRICES_PATH,
        retries: int = MAX_RETRIES,
        backoff_s: float = 2.0,
        request_timeout_s: float = DEFAULT_READ_TIMEOUT,
    ) -> None:
        super().__init__(model_name=model_name, base_url=base_url or BASE_URL, api_key=api_key, api_key_vars=[],
                         config=config)
        if not 0 <= int(retries) <= MAX_RETRIES:
            raise NanoGPTConfigError(f"retries must be in 0..{MAX_RETRIES}")
        self._retries = int(retries)
        self._backoff = float(backoff_s)
        self._request_timeout = float(request_timeout_s)
        self._transport = transport
        self._key_path = key_path
        prices = load_prices(prices_path)
        if model_name not in prices:
            raise NanoGPTConfigError(f"{model_name!r} has no entry in {prices_path}; cannot bound its cost")
        self._price = prices[model_name]
        path, cap = os.environ.get(ENV_LEDGER), os.environ.get(ENV_CAP)
        if not path or not cap:
            raise NanoGPTConfigError(f"{ENV_LEDGER} and {ENV_CAP} must both be set (fail closed)")
        try:
            cap_usd = float(cap)
        except ValueError as e:
            raise NanoGPTConfigError(f"{ENV_CAP}={cap!r} is not a number") from e
        self.ledger = BudgetLedger(path, cap_usd, create=False)  # raises LedgerError if missing or unreadable
        self._http: httpx.AsyncClient | None = None
        self._http_loop: asyncio.AbstractEventLoop | None = None

    # Inspect hooks ------------------------------------------------------------------------------------------------

    def should_retry(self, ex: Exception) -> bool:
        return False  # this provider owns retries (max 2, each reserved); Inspect must not add more

    def max_tokens(self) -> int | None:
        return DEFAULT_MAX_TOKENS

    async def aclose(self) -> None:
        if self._http is not None and not self._http.is_closed:
            with contextlib.suppress(Exception):
                await self._http.aclose()
        self._http = None

    # generate -----------------------------------------------------------------------------------------------------

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        if tools:
            raise NanoGPTConfigError("nanogpt provider does not support tools")
        if config.attempt_timeout is not None:
            # Inspect always retries AttemptTimeoutError itself (unbounded unless max_retries is set), which would
            # break the 2-retry rule. Per-attempt time is bounded by the httpx timeout (model arg request_timeout_s).
            raise NanoGPTConfigError("do not set attempt_timeout with the nanogpt provider; use -M request_timeout_s")
        messages = _wire_messages(input)
        max_tokens = int(config.max_tokens or DEFAULT_MAX_TOKENS)
        body: dict[str, Any] = {"model": self.model_name, "messages": messages, "max_tokens": max_tokens,
                                "stream": False}
        if config.temperature is not None:
            body["temperature"] = config.temperature
        if config.reasoning_effort is not None:
            body["reasoning_effort"] = config.reasoning_effort
        if config.seed is not None:
            body["seed"] = config.seed
        input_chars = len(json.dumps(messages, ensure_ascii=False))
        est = estimate_max_cost(self._price, input_chars, max_tokens)
        meta = {"model": self.model_name, "sample_id": _active_sample_id(), "call_kind": _CALL_KIND.get()}
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        timeout = httpx.Timeout(self._request_timeout, connect=CONNECT_TIMEOUT)

        attempts: list[dict] = []
        last_error: Exception | None = None
        last_payload: Any = None
        last_resp: httpx.Response | None = None
        for attempt in range(self._retries + 1):
            if attempt:
                delay = _retry_after(last_resp)
                await anyio.sleep(self._backoff * 3 ** (attempt - 1) if delay is None else delay)
            headers = {"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"}
            token = await self.ledger.reserve(est, meta)
            t0 = time.monotonic()
            try:
                resp = await self._client().post(url, json=body, headers=headers, timeout=timeout)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as ex:
                await self._release(token, f"never_sent:{type(ex).__name__}")
                attempts.append({"reservation_id": token, "outcome": "released", "settled_usd": 0.0,
                                 "status": None, "error": type(ex).__name__})
                last_error, last_resp = ex, None
                continue
            except httpx.TransportError as ex:
                settled = await self._settle(token, None, f"transport_error:{type(ex).__name__}")
                attempts.append({"reservation_id": token, "outcome": "transport_error", "settled_usd": settled,
                                 "status": None, "error": type(ex).__name__})
                last_error, last_resp = ex, None
                continue
            except BaseException as ex:  # cancellation or an unexpected error: the request may have been sent
                await self._settle(token, None, f"aborted:{type(ex).__name__}")
                raise
            elapsed = time.monotonic() - t0
            try:
                payload: Any = resp.json()
            except ValueError:
                payload = None
            cost = _usage_cost(payload)
            reason = None if cost is not None else f"missing_cost:http_{resp.status_code}"
            settled = await self._settle(token, cost, reason)
            choices = payload.get("choices") if isinstance(payload, dict) else None
            ok = resp.status_code == 200 and isinstance(choices, list) and len(choices) > 0
            attempts.append({"reservation_id": token, "outcome": "ok" if ok else "http_error", "settled_usd": settled,
                             "status": resp.status_code})
            if ok:
                output = self._output(payload, token=token, settled=settled, est=est,
                                      cost_source="usage.cost" if cost is not None else "est_max", attempts=attempts)
                return output, ModelCall.create(request=body, response=payload, time=elapsed)
            last_error = NanoGPTHTTPError(resp.status_code, resp.text)
            last_payload, last_resp = payload, resp
            if not (resp.status_code == 429 or resp.status_code >= 500):
                break  # no retry on other 4xx, nor on a 200 without a completion (it may have been billed)

        assert last_error is not None
        return last_error, ModelCall.create(request=body, response=last_payload if isinstance(last_payload, dict)
                                            else {"attempts": attempts})

    # helpers ------------------------------------------------------------------------------------------------------

    def _api_key(self) -> str:
        if self.api_key:
            return self.api_key
        try:
            key = pathlib.Path(os.path.expanduser(self._key_path)).read_text().strip()
        except OSError as e:
            raise NanoGPTConfigError(f"cannot read API key file {self._key_path}: {type(e).__name__}") from None
        if not key:
            raise NanoGPTConfigError(f"API key file {self._key_path} is empty")
        return key

    def _client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._http is None or self._http.is_closed or self._http_loop is not loop:
            self._http = httpx.AsyncClient(transport=self._transport)
            self._http_loop = loop
        return self._http

    async def _settle(self, token: str, cost: float | None, reason: str | None) -> float:
        with anyio.CancelScope(shield=True):  # the ledger write must land even if this attempt is being cancelled
            return await self.ledger.settle(token, cost, reason=reason)

    async def _release(self, token: str, reason: str) -> None:
        with anyio.CancelScope(shield=True):
            await self.ledger.release(token, reason=reason)

    def _output(self, payload: dict, *, token: str, settled: float, est: float, cost_source: str,
                attempts: list[dict]) -> ModelOutput:
        choice = payload["choices"][0] or {}
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        if not isinstance(text, str):
            text = json.dumps(text)
        reasoning = msg.get("reasoning") or msg.get("reasoning_content")
        provider_model = payload.get("model") or self.model_name
        content: str | list = text
        if isinstance(reasoning, str) and reasoning:
            content = [ContentReasoning(reasoning=reasoning), ContentText(text=text)]
        message = ChatMessageAssistant(content=content, model=provider_model, source="generate")
        finish = choice.get("finish_reason")
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) \
            else {}
        inp, out = _int(usage.get("prompt_tokens")) or 0, _int(usage.get("completion_tokens")) or 0
        reasoning_tokens = _int(usage.get("reasoning_tokens"))
        if reasoning_tokens is None:
            reasoning_tokens = _int(details.get("reasoning_tokens"))
        total_cost = float(sum(a["settled_usd"] for a in attempts))
        return ModelOutput(
            model=provider_model,
            choices=[ChatCompletionChoice(message=message, stop_reason=_STOP.get(str(finish), "unknown"))],
            usage=ModelUsage(
                input_tokens=inp, output_tokens=out, total_tokens=_int(usage.get("total_tokens")) or inp + out,
                reasoning_tokens=reasoning_tokens,
                input_tokens_cache_read=_int(usage.get("cache_read_input_tokens")),
                input_tokens_cache_write=_int(usage.get("cache_creation_input_tokens")),
                total_cost=total_cost),
            metadata={"cost_usd": settled, "reservation_id": token, "provider_model": provider_model,
                      "finish_reason": finish, "raw_usage": usage, "est_max_usd": est, "cost_source": cost_source,
                      "attempts": attempts, "total_cost_usd": total_cost},
            response_id=payload.get("id") if isinstance(payload.get("id"), str) else None,
        )


@modelapi(name="nanogpt")
def nanogpt() -> type[ModelAPI]:
    """Inspect registration: model strings "nanogpt/<provider>/<model>" resolve to NanoGPTAPI."""
    return NanoGPTAPI


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    bp = sub.add_parser("build-prices", help="rebuild model_prices.json from a nanoGPT /models dump")
    bp.add_argument("--source", default=MODELS_DUMP)
    bp.add_argument("--out", default=str(PRICES_PATH))
    args = ap.parse_args()
    table = build_price_table(args.source)
    pathlib.Path(args.out).write_text(json.dumps(table, indent=1) + "\n")
    for mid, p in table["models"].items():
        print(f"{mid:48s} in={p['in_per_m']:8.4f} out={p['out_per_m']:8.4f}  "
              f"(catalog {p['catalog_in_per_m']}/{p['catalog_out_per_m']})")


if __name__ == "__main__":
    _main()
