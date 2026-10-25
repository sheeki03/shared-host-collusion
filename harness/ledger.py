"""Fail-closed, append-only USD admission ledger shared by every model call in a run (CONTRACT.md, H1).

The ledger is a JSONL file with one line per event:
- init:    pins a ledger UUID and cap before any call.
- cap_change: an explicit, reasoned increase; all readers adopt it from the event stream.
- reserve: a call is about to be sent. Its est_max counts against the cap until it is settled or released.
- settle:  the call finished. The reservation is replaced by the actual cost.
- release: the call provably never reached the provider. The reservation is dropped at zero cost.
- flag:    an anomaly to audit: a missing or invalid cost settled at est_max, a cost above est_max, or a torn line.

Every line carries ts, event, id, model, est_max, actual, sample_id, call_kind and pid; flag lines also carry a reason.

State (pinned cap, settled spend plus outstanding reservations) is rebuilt from the file when a ledger is opened. It is re-read
under an exclusive fcntl.flock before every reserve/settle/release, which then appends while still holding the lock.
So any number of BudgetLedger objects and processes sharing one path see a single budget. The admission rule is
settled + outstanding + est_max <= cap. Money is summed exactly as Decimal(repr(float)), so a cap that admits exactly N
reservations admits exactly N (no float drift).

Fail-closed rules
- A missing (when create=False), unreadable, replaced, truncated or malformed ledger raises LedgerError. The caller
  must not make the call.
- A missing or invalid actual cost settles at est_max and writes a flag line.
- A provider charge above its reservation is recorded and flags a permanent breach; all later reservations stop.
  This cannot undo the bill and the local ceiling is conditional on pinned pricing and advertised context bounds.
- A reservation orphaned by a crashed process stays outstanding, counted against the cap, until someone settles it.
- A line left unterminated by a writer that died mid-write is closed off and flagged ("torn_line") by the next
  exclusive writer. A torn reserve never returned a token, so no call used it. A torn settle leaves its
  reservation outstanding, which is conservative.

The critical sections hold no awaits: file IO runs synchronously while the locks are held. Cancelling a task therefore
cannot leave a half-applied event. The asyncio lock still serialises reserve/settle/release coroutines in a process.
"""
from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import stat
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterator

EVENTS = ("init", "cap_change", "reserve", "settle", "release", "flag")
META_KEYS = ("model", "sample_id", "call_kind")


class BudgetError(RuntimeError):
    """Base class for ledger failures. Every subclass means: do not make the call."""


class BudgetExceeded(BudgetError):
    """The reservation would take settled + outstanding spend above the cap."""


class BudgetBreach(BudgetError):
    """An actual charge exceeded its reservation; further calls are stopped until the run is audited."""


class LedgerError(BudgetError):
    """The ledger file is missing, unreadable, malformed or used incorrectly."""


def _money(x: Any) -> Decimal:
    """Exact Decimal for a USD amount. Floats go through repr, so 0.01 stays exactly 0.01."""
    if isinstance(x, bool) or not isinstance(x, (int, float, str, Decimal)):
        raise TypeError(f"not a money amount: {x!r}")
    try:
        d = Decimal(repr(x)) if isinstance(x, float) else Decimal(x)
    except InvalidOperation as e:
        raise ValueError(f"not a money amount: {x!r}") from e
    if not d.is_finite():
        raise ValueError(f"non-finite money amount: {x!r}")
    return d


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class Reservation:
    id: str
    est_max: Decimal
    model: str | None
    sample_id: str | None
    call_kind: str | None
    pid: int | None


class BudgetLedger:
    """Global budget backed by an append-only JSONL file. See the module docstring for the guarantees."""

    def __init__(self, path: str | os.PathLike, cap_usd: float, *, create: bool = True) -> None:
        cap = _money(cap_usd)
        if cap < 0:
            raise ValueError("cap_usd must be >= 0")
        self.path = os.fspath(path)
        self.cap_usd = float(cap_usd)
        self._cap = cap
        self._initialized = False
        self.ledger_id: str | None = None
        self._breach: str | None = None
        self._spent = Decimal(0)
        self._out = Decimal(0)
        self._open: dict[str, Reservation] = {}
        self._closed: set[str] = set()
        self._flags = 0
        self._offset = 0                      # bytes of the file already applied (always at a line boundary)
        self._ident: tuple[int, int] | None = None
        self._pending_torn: set[int] = set()  # offsets of malformed lines not (yet) covered by a torn_line flag
        self._corrupt: str | None = None      # set on the first content error; every later operation raises
        self._tlock = threading.Lock()
        self._aloop: asyncio.AbstractEventLoop | None = None
        self._alock: asyncio.Lock | None = None
        if not create and not os.path.exists(self.path):
            raise LedgerError(f"ledger file not found: {self.path}")
        with self._tlock, self._locked(fcntl.LOCK_EX if create else fcntl.LOCK_SH, create=create) as fd:
            self._sync(fd, repair=False)
            if not self._initialized:
                if not create or os.fstat(fd).st_size != 0:
                    raise LedgerError(f"ledger {self.path} has no valid initialization event")
                self._append(fd, self._line("init", uuid.uuid4().hex, est_max=None, actual=None, cap=cap))
                self._sync(fd, repair=False)
            if self._cap != cap:
                raise LedgerError(f"ledger {self.path} cap is ${self._cap}, not requested ${cap}; "
                                  "an explicit audited cap change is required")

    # ----------------------------------------------------------------------------------------------------- public

    async def reserve(self, est_max_usd: float, meta: dict | None = None) -> str:
        """Reserve est_max_usd for one provider attempt and return the reservation id (the token).

        meta may carry model, sample_id and call_kind; any other keys are written under "meta".
        Raises BudgetExceeded if settled + outstanding + est_max would exceed the cap, LedgerError if the ledger
        cannot be read or written.
        """
        est = _money(est_max_usd)
        if est < 0:
            raise ValueError("est_max_usd must be >= 0")
        meta = dict(meta or {})
        async with self._async_lock():
            with self._tlock, self._locked(fcntl.LOCK_EX) as fd:
                self._sync(fd, repair=True)
                if self._breach:
                    raise BudgetBreach(f"ledger {self.path} halted: {self._breach}")
                if self._spent + self._out + est > self._cap:
                    raise BudgetExceeded(
                        f"budget cap ${self._cap} would be exceeded: spent ${self._spent} + outstanding ${self._out} "
                        f"+ est_max ${est} (ledger {self.path})")
                token = uuid.uuid4().hex
                fields = {k: meta.pop(k, None) for k in META_KEYS}
                line = self._line("reserve", token, est_max=est, actual=None, **fields,
                                  **({"meta": meta} if meta else {}))
                self._append(fd, line)
                self._sync(fd, repair=False)
                return token

    async def settle(self, token: str, actual_usd: float | None, *, reason: str | None = None) -> float:
        """Replace reservation `token` with its actual cost and return the amount settled.

        actual_usd None (or negative / non-numeric) settles at est_max and writes a flag line (fail closed). `reason`
        labels that flag, for example "missing_cost" or "transport_error:ReadTimeout". A cost above est_max is
        settled as reported and flagged "over_est_max".
        """
        async with self._async_lock():
            with self._tlock, self._locked(fcntl.LOCK_EX) as fd:
                self._sync(fd, repair=True)
                r = self._take(token)
                flag: str | None = None
                if actual_usd is None:
                    amount, flag = r.est_max, reason or "missing_cost"
                else:
                    try:
                        amount = _money(actual_usd)
                        if amount < 0:
                            raise ValueError("negative cost")
                    except (TypeError, ValueError):
                        amount, flag = r.est_max, f"invalid_cost:{str(actual_usd)[:40]}"
                    if flag is None and amount > r.est_max:
                        flag = "over_est_max"
                data = self._line("settle", r.id, est_max=r.est_max, actual=amount, **self._fields(r))
                if flag is not None:
                    data += self._line("flag", r.id, est_max=r.est_max, actual=amount, **self._fields(r), reason=flag)
                self._append(fd, data)
                self._sync(fd, repair=False)
                if flag == "over_est_max":
                    raise BudgetBreach(f"provider charge ${amount} exceeded reserved maximum ${r.est_max}; "
                                       f"recorded in {self.path}, no further calls admitted")
                return float(amount)

    async def release(self, token: str, *, reason: str | None = None) -> None:
        """Drop reservation `token` at zero cost. Only for attempts that provably never reached the provider."""
        async with self._async_lock():
            with self._tlock, self._locked(fcntl.LOCK_EX) as fd:
                self._sync(fd, repair=True)
                r = self._take(token)
                extra = {"reason": reason} if reason else {}
                self._append(fd, self._line("release", r.id, est_max=r.est_max, actual=None, **self._fields(r),
                                            **extra))
                self._sync(fd, repair=False)

    @property
    def spent(self) -> float:
        """Total settled spend in USD, including other processes' events."""
        self.refresh()
        return float(self._spent)

    @property
    def outstanding(self) -> float:
        """Sum of est_max over reservations not yet settled or released."""
        self.refresh()
        return float(self._out)

    @property
    def remaining(self) -> float:
        """cap - spent - outstanding (negative only if settled costs overran their est_max)."""
        self.refresh()
        return float(self._cap - self._spent - self._out)

    @property
    def flags(self) -> int:
        """Number of flag lines in the ledger."""
        self.refresh()
        return self._flags

    def open_reservations(self) -> list[Reservation]:
        """Reservations still outstanding (in flight, or orphaned by a crashed process)."""
        self.refresh()
        return list(self._open.values())

    def raise_cap(self, new_cap_usd: float, *, reason: str) -> None:
        """Audit an explicit cap increase; all processes sharing this file observe the new cap on their next sync."""
        new_cap = _money(new_cap_usd)
        if not reason.strip():
            raise ValueError("an audited cap increase requires a reason")
        with self._tlock, self._locked(fcntl.LOCK_EX) as fd:
            self._sync(fd, repair=True)
            if self._breach:
                raise BudgetBreach(f"cannot increase a breached ledger: {self._breach}")
            if new_cap <= self._cap:
                raise ValueError(f"new cap ${new_cap} must exceed current cap ${self._cap}")
            self._append(fd, self._line("cap_change", self.ledger_id, est_max=None, actual=None,
                                        old_cap=self._cap, cap=new_cap, reason=reason))
            self._sync(fd, repair=False)

    def refresh(self) -> None:
        """Apply events appended since the last read (shared lock; never writes)."""
        with self._tlock, self._locked(fcntl.LOCK_SH) as fd:
            self._sync(fd, repair=False)

    def __repr__(self) -> str:
        return (f"BudgetLedger({self.path!r}, cap=${self._cap}, spent=${self._spent}, outstanding=${self._out}, "
                f"open={len(self._open)}, flags={self._flags})")

    # ---------------------------------------------------------------------------------------------------- private

    def _async_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._aloop is not loop or self._alock is None:
            self._aloop, self._alock = loop, asyncio.Lock()
        return self._alock

    @contextlib.contextmanager
    def _locked(self, op: int, create: bool = False) -> Iterator[int]:
        flags = os.O_RDWR | os.O_APPEND | (os.O_CREAT if create else 0)
        try:
            fd = os.open(self.path, flags, 0o600)
        except OSError as e:
            raise LedgerError(f"cannot open ledger {self.path}: {e}") from e
        try:
            try:
                fcntl.flock(fd, op)
                st = os.fstat(fd)
            except OSError as e:
                raise LedgerError(f"cannot lock ledger {self.path}: {e}") from e
            if not stat.S_ISREG(st.st_mode):
                # e.g. /dev/null: appends vanish, so state never grows and every reserve would be admitted.
                raise LedgerError(f"ledger {self.path} is not a regular file")
            ident = (st.st_dev, st.st_ino)
            if self._ident is None:
                self._ident = ident
            elif ident != self._ident:
                raise LedgerError(f"ledger file {self.path} was replaced since it was opened")
            if st.st_size < self._offset:
                raise LedgerError(f"ledger file {self.path} shrank from {self._offset} to {st.st_size} bytes")
            yield fd
        finally:
            os.close(fd)  # closing the descriptor also drops the flock

    def _sync(self, fd: int, repair: bool) -> None:
        """Apply complete lines past self._offset. With repair (exclusive lock held), close off a torn tail.

        A content error (malformed line, impossible event) is sticky: this object refuses every later operation.
        """
        if self._corrupt is not None:
            raise LedgerError(self._corrupt)
        size = os.fstat(fd).st_size
        if size == self._offset:
            return
        try:
            data = os.pread(fd, size - self._offset, self._offset)
        except OSError as e:
            raise LedgerError(f"cannot read ledger {self.path}: {e}") from e
        end = data.rfind(b"\n") + 1
        pos = self._offset
        try:
            for raw in data[:end].split(b"\n")[:-1]:
                self._apply(raw, pos)
                pos += len(raw) + 1
            if self._pending_torn:
                raise LedgerError(
                    f"malformed ledger line(s) at byte offset(s) {sorted(self._pending_torn)} in {self.path}")
        except LedgerError as e:
            self._corrupt = str(e)
            raise
        self._offset += end
        if end < len(data) and repair:
            # Under the exclusive lock nobody is mid-write, so an unterminated tail is a dead writer's torn line.
            torn_at = self._offset
            self._append(fd, b"\n" + self._line("flag", None, est_max=None, actual=None, reason="torn_line",
                                                offset=torn_at, model=None, sample_id=None, call_kind=None))
            self._sync(fd, repair=False)

    def _apply(self, raw: bytes, pos: int) -> None:
        try:
            ev = json.loads(raw)
            kind = ev["event"]
        except (ValueError, KeyError, TypeError):
            self._pending_torn.add(pos)  # must be followed by a torn_line flag for this offset, else LedgerError
            return
        try:
            if kind == "init":
                cap = _money(ev["cap"])
                rid = ev["id"]
                if pos != 0 or self._initialized or not isinstance(rid, str) or not rid or cap < 0:
                    raise LedgerError(f"invalid ledger initialization at offset {pos}")
                self._initialized, self.ledger_id = True, rid
                self._cap, self.cap_usd = cap, float(cap)
                return
            if not self._initialized:
                raise LedgerError(f"event before ledger initialization at offset {pos}")
            if kind == "cap_change":
                old, new = _money(ev["old_cap"]), _money(ev["cap"])
                if ev.get("id") != self.ledger_id or old != self._cap or new <= old or not ev.get("reason"):
                    raise LedgerError(f"invalid cap change at offset {pos}")
                self._cap, self.cap_usd = new, float(new)
                return
            if kind == "flag":
                self._flags += 1
                if ev.get("reason") == "torn_line":
                    self._pending_torn.discard(ev.get("offset"))
                if ev.get("reason") == "over_est_max":
                    self._breach = "a provider charge exceeded its reserved maximum"
                return
            rid = ev["id"]
            if not isinstance(rid, str):
                raise LedgerError(f"bad id at offset {pos}")
            if kind == "reserve":
                if rid in self._open or rid in self._closed:
                    raise LedgerError(f"duplicate reservation {rid} at offset {pos}")
                est = _money(ev["est_max"])
                if est < 0:
                    raise LedgerError(f"negative est_max at offset {pos}")
                self._open[rid] = Reservation(rid, est, ev.get("model"), ev.get("sample_id"), ev.get("call_kind"),
                                              ev.get("pid"))
                self._out += est
            elif kind in ("settle", "release"):
                r = self._open.pop(rid, None)
                if r is None:
                    raise LedgerError(f"{kind} for unknown or closed reservation {rid} at offset {pos}")
                self._out -= r.est_max
                self._closed.add(rid)
                if kind == "settle":
                    actual = _money(ev["actual"])
                    if actual < 0:
                        raise LedgerError(f"negative actual at offset {pos}")
                    self._spent += actual
                    if self._spent + self._out > self._cap:
                        self._breach = "settled spend plus outstanding reservations exceeded the cap"
            else:
                raise LedgerError(f"unknown event {kind!r} at offset {pos}")
        except (KeyError, TypeError, ValueError) as e:
            raise LedgerError(f"malformed {kind!r} event at offset {pos} in {self.path}: {e}") from e

    def _take(self, token: str) -> Reservation:
        r = self._open.get(token)
        if r is None:
            state = "already settled or released" if token in self._closed else "unknown"
            raise LedgerError(f"reservation {token!r} is {state}")
        return r

    @staticmethod
    def _fields(r: Reservation) -> dict:
        return {"model": r.model, "sample_id": r.sample_id, "call_kind": r.call_kind}

    @staticmethod
    def _line(event: str, rid: str | None, *, est_max: Decimal | None, actual: Decimal | None, **extra: Any) -> bytes:
        rec = {"ts": _now(), "event": event, "id": rid,
               "model": extra.pop("model", None),
               "est_max": None if est_max is None else float(est_max),
               "actual": None if actual is None else float(actual),
               "sample_id": extra.pop("sample_id", None),
               "call_kind": extra.pop("call_kind", None),
               "pid": os.getpid(), **extra}
        return (json.dumps(rec, separators=(",", ":"), default=str) + "\n").encode()

    def _append(self, fd: int, data: bytes) -> None:
        try:
            view = memoryview(data)
            while view:
                n = os.write(fd, view)
                view = view[n:]
            os.fsync(fd)
        except OSError as e:
            raise LedgerError(f"cannot append to ledger {self.path}: {e}") from e
