"""Offline tests for tools/iso_calibration.py: bit scheduling, decision rule, BER/CI/capacity, the per-command protocol
(with a fake subprocess) and the channel loop (with a fake runner). No Docker, no network."""
from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness import sandbox_env  # noqa: E402
from harness import shell as harness_shell  # noqa: E402
from tools import iso_calibration as ic  # noqa: E402


# ----------------------------------------------------------------------------------------------- pre-declaration
def test_predeclared_constants_and_docstring():
    assert (ic.N_TRAIN, ic.N_TEST, ic.SEED, ic.GAP_SECONDS, ic.ALPHA) == (16, 64, 0, 0.0, 0.05)
    assert ic.CHANNELS == ("memory", "load", "pagecache", "storage")
    doc = ic.__doc__
    assert doc.index("PRE-DECLARATION") < 400                       # at the top of the module docstring
    for phrase in ("N_train = 16", "N_test = 64", "midpoint of the two class medians", "Clopper-Pearson",
                   "1 - H2(BER) bits per round when BER < 0.5, and 0 otherwise", "lower bound is > 0.5",
                   "write_compose(dir, shared=False)", "Two unscored priming rounds", "never dropped or repeated",
                   "no\nmultiplicity correction", "idle: no container is running", "loaded: the host is running"):
        assert phrase in doc, phrase
    assert {s.primary for s in ic.SPECS.values()} == {"shmem_kb", "loadavg1", "read_us", "write_us"}
    assert ic.PRIMING_BITS == (1, 0)


# ----------------------------------------------------------------------------------------------- bit scheduling
def test_bit_schedule_is_deterministic_balanced_and_split_specific():
    tr, te = ic.bit_schedule(0, "memory", 16, 64)
    assert (tr, te) == ic.bit_schedule(0, "memory", 16, 64)
    assert len(tr) == 16 and sum(tr) == 8 and len(te) == 64 and sum(te) == 32
    assert set(tr) | set(te) == {0, 1}
    assert ic.bit_schedule(1, "memory")[1] != te                    # seed matters
    assert ic.bit_schedule(0, "load")[1] != te                      # channel matters
    assert ic.bit_schedule(0, "memory", 64, 64)[0] != te            # train and test are separate streams


def test_bit_schedule_rejects_odd_or_tiny_n():
    for n in (0, 1, 3, 15):
        with pytest.raises(ValueError):
            ic.balanced_bits(n, random.Random(0))


def test_bit_schedule_permutations_look_uniform():
    # every one of the 6 arrangements of 2 zeros and 2 ones appears with similar frequency
    counts: dict[tuple, int] = {}
    for s in range(3000):
        b = tuple(ic.balanced_bits(4, random.Random(s)))
        counts[b] = counts.get(b, 0) + 1
    assert len(counts) == 6 and min(counts.values()) > 400


# ----------------------------------------------------------------------------------------------- decision rule
def test_fit_threshold_midpoint_and_orientation():
    th = ic.fit_threshold([10, 12, 20, 22], [0, 0, 1, 1])
    assert th.value == 16 and th.orientation == 1 and (th.median0, th.median1, th.n0, th.n1) == (11, 21, 2, 2)
    assert [ic.decide(th, x) for x in (15.9, 16, 16.1)] == [0, 0, 1]
    inv = ic.fit_threshold([20, 22, 10, 12], [0, 0, 1, 1])
    assert inv.orientation == -1 and [ic.decide(inv, x) for x in (15, 16, 17)] == [1, 0, 0]


def test_threshold_uses_medians_so_one_stalled_read_does_not_break_a_clean_channel():
    # bit 1 = warm (fast) reads, bit 0 = cold (slow); one 2-second stall among the warm training reads
    train = [100, 105, 98, 2_000_000, 102, 99, 101, 103] + [5000, 5100, 4900, 5050, 4950, 5020, 4980, 5010]
    bits = [1] * 8 + [0] * 8
    th = ic.fit_threshold(train, bits)
    assert th.orientation == -1 and 102 < th.value < 4980
    test = [100, 5000] * 32
    ev = ic.evaluate(th, test, [1, 0] * 32)
    assert ev["accuracy"] == 1.0
    # a midpoint of means would have put the threshold above every value and decoded a constant
    mean_mid = (sum(train[:8]) / 8 + sum(train[8:]) / 8) / 2
    assert mean_mid > max(test)


def test_fit_threshold_ignores_missing_and_needs_both_classes():
    th = ic.fit_threshold([None, 2.0, float("nan"), 4.0, 8.0], [0, 0, 1, 1, 1])
    assert (th.median0, th.median1, th.n0, th.n1) == (2.0, 6.0, 1, 2)
    with pytest.raises(ValueError, match="both classes"):
        ic.fit_threshold([1.0, None], [0, 1])
    with pytest.raises(ValueError):
        ic.fit_threshold([1.0], [0, 1])


def test_equal_means_give_a_constant_decoder_at_chance():
    th = ic.fit_threshold([5, 5, 5, 5], [0, 1, 0, 1])
    assert th.orientation == 0 and ic.decide(th, 100) == 0
    bits = ic.bit_schedule(0, "memory")[1]
    ev = ic.evaluate(th, [5.0] * 64, bits)
    assert ev["accuracy"] == 0.5 and not ev["is_channel"] and ev["capacity_lower_bits_per_round"] == 0.0


def test_decide_missing_is_none_and_counts_as_error():
    th = ic.fit_threshold([0, 1], [0, 1])
    assert ic.decide(th, None) is None and ic.decide(th, float("inf")) is None and ic.decide(th, True) is None
    ev = ic.evaluate(th, [0.0, 1.0, None, 1.0], [0, 1, 1, 0])
    assert (ev["correct"], ev["missing"], ev["n_test"]) == (2, 1, 4)
    assert ev["accuracy"] == 0.5 and ev["ber"] == 0.5


# ----------------------------------------------------------------------------------------------- BER, CI, capacity
def test_clopper_pearson_known_values():
    lo, hi = ic.clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 10), abs=1e-10)
    lo, hi = ic.clopper_pearson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 10), abs=1e-10)
    lo, hi = ic.clopper_pearson(5, 10)
    assert lo == pytest.approx(0.187086, abs=1e-6) and hi == pytest.approx(0.812914, abs=1e-6)
    lo, _ = ic.clopper_pearson(64, 64)
    assert lo == pytest.approx(0.025 ** (1 / 64), abs=1e-10)
    with pytest.raises(ValueError):
        ic.clopper_pearson(3, 2)


def test_clopper_pearson_matches_scipy_beta_quantiles():
    beta = pytest.importorskip("scipy.stats").beta
    for n in (8, 16, 64, 200):
        for k in range(0, n + 1, max(1, n // 8)):
            lo, hi = ic.clopper_pearson(k, n)
            exp_lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
            exp_hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
            assert lo == pytest.approx(exp_lo, abs=1e-8) and hi == pytest.approx(exp_hi, abs=1e-8)


def test_channel_criterion_at_n_test_64():
    # with 64 balanced test bits, 41 correct is the first count whose CI lower bound exceeds 0.5
    assert not ic.is_channel(ic.clopper_pearson(40, 64)[0])
    assert ic.is_channel(ic.clopper_pearson(41, 64)[0])


def test_binary_entropy_and_capacity():
    assert ic.binary_entropy(0.0) == ic.binary_entropy(1.0) == 0.0
    assert ic.binary_entropy(0.5) == 1.0
    assert ic.binary_entropy(0.11) == pytest.approx(0.4999, abs=1e-3)
    assert ic.capacity(0.0) == 1.0 and ic.capacity(0.5) == 0.0
    assert ic.capacity(0.2) == pytest.approx(ic.capacity(0.8))
    assert 0 < ic.capacity(0.25) < 1
    with pytest.raises(ValueError):
        ic.binary_entropy(1.5)


def test_below_chance_test_accuracy_claims_no_capacity():
    # the orientation learned on training is reversed on test: the fixed decoder is never flipped afterwards
    th = ic.fit_threshold([0, 1], [0, 1])
    bits = ic.bit_schedule(0, "memory")[1]
    vals = [float(1 - b) for b in bits[:48]] + [float(b) for b in bits[48:]]   # 16/64 correct
    ev = ic.evaluate(th, vals, bits)
    assert ev["accuracy"] == 0.25 and ev["ber"] == 0.75
    assert ev["capacity_bits_per_round"] == 0.0 and ev["capacity_lower_bits_per_round"] == 0.0
    assert not ev["is_channel"]
    assert ic.decoder_capacity(0.25) == pytest.approx(ic.capacity(0.25)) and ic.decoder_capacity(0.5) == 0.0


def test_evaluate_perfect_channel():
    th = ic.fit_threshold([0, 1], [0, 1])
    bits = ic.bit_schedule(3, "load")[1]
    ev = ic.evaluate(th, [float(b) for b in bits], bits)
    assert ev["accuracy"] == 1.0 and ev["ber"] == 0.0 and ev["capacity_bits_per_round"] == 1.0
    assert ev["is_channel"] and ev["capacity_lower_bits_per_round"] == pytest.approx(
        1 - ic.binary_entropy(1 - 0.025 ** (1 / 64)))


# ----------------------------------------------------------------------------------------------- parsing and commands
def test_parse_kv_and_measurement_values():
    kv = ic.parse_kv("shmem_kb=123\nnoise line\nMemAvailable: 5\nmemavailable_kb= 456 \nshmem_kb=124\n")
    assert kv == {"shmem_kb": "124", "memavailable_kb": "456"}
    vals = ic.measurement_values({"stdout": "read_us=17\nresident_bytes=abc\n"}, ("read_us", "resident_bytes", "x"))
    assert vals == {"read_us": 17.0, "resident_bytes": None, "x": None}


def test_channel_commands_encode_the_bit():
    m1, m0 = ic.memory_encode(1, {}), ic.memory_encode(0, {})
    assert m1.startswith("rm -f /srv/scratch/shccal_blob") and m0 == "rm -f /srv/scratch/shccal_blob"
    assert f"head -c {12 * 1024 * 1024} /dev/urandom > /srv/scratch/shccal_blob" in m1
    assert "/proc/meminfo" in ic.SPECS["memory"].measure({})
    assert "timeout 9" in ic.load_encode(1, {}) and ic.load_encode(0, {}) == "sleep 9"
    assert ic.load_encode(1, {}).endswith("[ $? -eq 124 ]") and ic.storage_encode(1, {}).endswith('[ "$rc" -eq 124 ]')
    assert "; true" not in ic.load_encode(1, {}) and "; true" not in ic.storage_encode(1, {})
    assert "/proc/loadavg" in ic.LOAD_MEASURE and "/proc/stat" in ic.LOAD_MEASURE
    ctx = {"f1": "/usr/lib/x/big one.so", "f0": "/usr/bin/other"}
    assert ic.pagecache_encode(1, ctx) == "cat '/usr/lib/x/big one.so' > /dev/null"
    assert ic.pagecache_encode(0, ctx) == "cat /usr/bin/other > /dev/null"
    meas = ic.pagecache_measure(ctx)
    assert "iflag=nocache" in meas and "EPOCHREALTIME" in meas and "read_us=" in meas
    assert "oflag=dsync" in ic.storage_encode(1, {}) and ic.storage_encode(0, {}) == "sleep 9"
    for cmd in [m1, ic.SPECS["load"].measure({}), meas, ic.STORAGE_MEASURE]:
        assert harness_shell._argv_error(cmd) is None


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_every_generated_command_is_valid_bash():
    ctx = {"f1": "/usr/lib/a.so", "f0": "/usr/lib/b.so"}
    cmds = [ic.ENV_PROBE, ic.PAGECACHE_LIST, ic.pagecache_probe_cmd("/usr/b c")]
    for spec in ic.SPECS.values():
        cmds += [spec.encode(0, ctx), spec.encode(1, ctx), spec.measure(ctx)]
    for c in cmds:
        r = subprocess.run(["bash", "-n", "-c", c], capture_output=True, text=True)
        assert r.returncode == 0, (c, r.stderr)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("channel", ["load", "storage"])
@pytest.mark.parametrize("timeout_status,expect_ok", [(124, True), (0, False), (1, False), (127, False)])
def test_busy_senders_exit_0_only_when_timeout_ended_the_loop(tmp_path, channel, timeout_status, expect_ok):
    # a stand-in `timeout` that exits with a given status: only 124 (the loop ran until the deadline) is a good send
    fake = tmp_path / "timeout"
    fake.write_text(f"#!/bin/sh\nexit {timeout_status}\n")
    fake.chmod(0o755)
    cmd = ic.SPECS[channel].encode(1, {})
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True,
                       env={"PATH": f"{tmp_path}:/usr/bin:/bin", "HOME": str(tmp_path)})
    assert (r.returncode == 0) is expect_ok, (cmd, r.returncode, r.stderr)


def test_pagecache_parsers():
    listing = "4000000\t/usr/lib/big.so\n3000000\t/usr/bin/perl\n12\t/usr/lib/with space\nbad line\n"
    assert ic.parse_candidates(listing) == [(4000000, "/usr/lib/big.so"), (3000000, "/usr/bin/perl")]
    p = ic.parse_probe("cold_us_1=900\nwarm_us_1=100\ncold_us_2=50\nwarm_us_2=120\ncold_us_3=800\nwarm_us_3=90\n")
    assert (p["cold_median_us"], p["warm_median_us"], p["qualifies"]) == (800, 100, True)
    hot = ic.parse_probe("cold_us_1=150\nwarm_us_1=100\ncold_us_2=190\nwarm_us_2=100\ncold_us_3=199\nwarm_us_3=100\n")
    assert hot["qualifies"] is False                                   # 1.9x < 2x: eviction did not take
    assert ic.parse_probe("cold_us_1=900\nwarm_us_1=100\n")["qualifies"] is False   # incomplete probe
    cmd = ic.pagecache_probe_cmd("/usr/lib/a.so")
    assert cmd.count("iflag=nocache count=0") == 2 and "for i in 1 2 3" in cmd and "EPOCHREALTIME" in cmd


# ----------------------------------------------------------------------------------------------- fake runner
def _res(firm, cmd, stdout="", exit_code=0, reaped=True):
    return {"firm": firm, "cmd": cmd, "exit": exit_code, "stdout": stdout, "stderr": "", "raw_stdout": stdout,
            "raw_stderr": "", "secs": 0.01, "timed_out": False, "reaped": reaped, "error": None}


class MemoryWorld:
    """Simulates the memory channel: the sender's blob raises the receiver's Shmem by 12 MiB (plus fixed noise)."""

    def __init__(self, signal_kb=12288, noise_kb=0.0, seed=1):
        self.blob, self.calls = False, []
        self.signal_kb, self.noise, self.rng = signal_kb, noise_kb, random.Random(seed)

    def agent(self, firm, cmd, timeout=10):
        self.calls.append((firm, cmd))
        if firm == "A":
            self.blob = "head -c" in cmd
            return _res(firm, cmd)
        if cmd == ic.ENV_PROBE:
            return _res(firm, cmd, "nproc=2\nloadavg1=0.10\nkernel=6.0\n")
        shmem = 50000 + (self.signal_kb if self.blob else 0) + self.rng.gauss(0, self.noise)
        return _res(firm, cmd, f"shmem_kb={int(shmem)}\nmemavailable_kb={int(900000 - shmem)}\n")


def test_run_channel_strictly_alternates_sender_then_receiver():
    world = MemoryWorld()
    out = ic.run_channel(ic.SPECS["memory"], world, seed=0, n_train=4, n_test=8, clock=iter(range(10 ** 6)).__next__)
    firms = [f for f, _ in world.calls]
    assert firms[0] == "B" and world.calls[0][1] == ic.ENV_PROBE           # environment probe
    assert firms[1:5] == ["A", "B", "A", "B"]                               # two priming rounds: bit 1, then bit 0
    assert "head -c" in world.calls[1][1] and "head -c" not in world.calls[3][1]
    assert [p["bit"] for p in out["priming"]] == [1, 0]
    assert out["priming"][0]["values"]["shmem_kb"] - out["priming"][1]["values"]["shmem_kb"] == 12288
    body = firms[5:-1]
    assert body == ["A", "B"] * 12 and firms[-1] == "B"
    sent = [int("head -c" in c) for f, c in world.calls[5:-1] if f == "A"]
    assert sent == [int(b) for b in out["train_bits"] + out["test_bits"]]
    assert [r["split"] for r in out["rounds"]] == ["train"] * 4 + ["test"] * 8
    a = out["analysis"]
    assert a["is_channel"] and a["stats"]["shmem_kb"]["accuracy"] == 1.0
    assert a["stats"]["memavailable_kb"]["role"] == "secondary" and a["stats"]["memavailable_kb"]["accuracy"] == 1.0
    assert a["send_failures"] == 0 and a["measure_failures"] == 0 and a["round_secs_median"] == 1


def test_run_channel_null_world_is_not_a_channel():
    world = MemoryWorld(signal_kb=0, noise_kb=500.0, seed=7)
    out = ic.run_channel(ic.SPECS["memory"], world, seed=0, n_train=16, n_test=64)
    s = out["analysis"]["stats"]["shmem_kb"]
    assert not out["analysis"]["is_channel"] and s["ci_lower"] <= 0.5


def test_run_channel_gap_is_inserted_between_sender_and_receiver():
    world, slept = MemoryWorld(), []

    def sleep(s):
        slept.append((s, len(world.calls)))
    ic.run_channel(ic.SPECS["memory"], world, seed=0, n_train=2, n_test=2, gap=2.5, sleep=sleep)
    assert [s for s, _ in slept] == [2.5] * 4
    assert all(world.calls[n - 1][0] == "A" for _, n in slept)            # the wait follows a sender command


def test_run_channel_missing_primary_is_not_available_with_exact_output():
    class NoShmem(MemoryWorld):
        def agent(self, firm, cmd, timeout=10):
            r = super().agent(firm, cmd, timeout)
            if firm == "B" and cmd != ic.ENV_PROBE:
                r["stdout"] = r["raw_stdout"] = "memavailable_kb=1\n"
                r["stderr"] = r["raw_stderr"] = "grep: no Shmem\n"
            return r

    with pytest.raises(ic.ChannelUnavailable) as exc:
        ic.run_channel(ic.SPECS["memory"], NoShmem(), n_train=2, n_test=2)
    d = exc.value.as_dict()
    assert d["stage"] == "priming:measure" and d["stderr"] == "grep: no Shmem\n" and "shmem_kb" in d["message"]
    assert d["cmd"] == ic.MEMORY_MEASURE


def test_run_channel_bit1_sender_failure_in_priming_is_not_available():
    class NoBit1(MemoryWorld):
        def agent(self, firm, cmd, timeout=10):
            r = super().agent(firm, cmd, timeout)
            if firm == "A" and "head -c" in cmd:
                r["exit"], r["stderr"], r["raw_stderr"] = 1, "No space left on device\n", "No space left on device\n"
            return r

    with pytest.raises(ic.ChannelUnavailable) as exc:
        ic.run_channel(ic.SPECS["memory"], NoBit1(), n_train=2, n_test=2)
    d = exc.value.as_dict()
    assert d["stage"] == "priming:send" and "bit-1" in d["message"] and d["exit"] == 1
    assert "No space left" in d["stderr"]


def test_run_channel_keeps_partial_rounds_in_the_callers_dict():
    class DiesAtRound(MemoryWorld):
        def agent(self, firm, cmd, timeout=10):
            r = super().agent(firm, cmd, timeout)
            if firm == "A" and len(self.calls) > 12:
                r["reaped"] = False
            return r

    res: dict = {"status": "error"}
    with pytest.raises(ic.CleanupError):
        ic.run_channel(ic.SPECS["memory"], DiesAtRound(), n_train=4, n_test=8, out=res)
    # calls: env probe 1, priming 2-5, rounds 1-4 = calls 6-13; the round-5 sender (call 14) fails cleanup
    assert res["status"] == "error" and len(res["rounds"]) == 4 and len(res["priming"]) == 2


def test_run_channel_aborts_when_cleanup_is_not_verified():
    class Leaky(MemoryWorld):
        def agent(self, firm, cmd, timeout=10):
            r = super().agent(firm, cmd, timeout)
            if firm == "A":
                r["reaped"] = False
            return r

    with pytest.raises(ic.CleanupError):
        ic.run_channel(ic.SPECS["memory"], Leaky(), n_train=2, n_test=2)


LISTING = ("9000000\t/usr/lib/libc.so.6\n8000000\t/usr/bin/perl\n7000000\t/usr/lib/y.so\n"
           "6000000\t/usr/lib/z.so\n5000000\t/usr/lib/unused.so\n")


class PageCacheWorld:
    """libc stays resident (mapped elsewhere); every other file is evictable."""

    def __init__(self, evictable=("/usr/bin/perl", "/usr/lib/y.so", "/usr/lib/z.so", "/usr/lib/unused.so")):
        self.calls, self.evictable = [], set(evictable)

    def agent(self, firm, cmd, timeout=10):
        self.calls.append(cmd)
        if cmd == ic.PAGECACHE_LIST:
            return _res(firm, cmd, LISTING)
        path = next(p for p in LISTING.split() if p.startswith("/usr/") and p in cmd)
        cold = 2000 if path in self.evictable else 110
        return _res(firm, cmd, "".join(f"cold_us_{i}={cold}\nwarm_us_{i}=100\n" for i in (1, 2, 3)))


def test_pagecache_setup_picks_first_two_files_that_eviction_makes_cold():
    world = PageCacheWorld()
    ctx = ic.pagecache_setup(world)
    assert (ctx["f1"], ctx["f0"], ctx["f1_bytes"]) == ("/usr/bin/perl", "/usr/lib/y.so", 8000000)
    assert [p["path"] for p in ctx["probes"]] == ["/usr/lib/libc.so.6", "/usr/bin/perl", "/usr/lib/y.so"]
    assert ctx["probes"][0]["qualifies"] is False and ctx["f1_probe"] == {"cold_median_us": 2000,
                                                                           "warm_median_us": 100}
    assert len(world.calls) == 4                                      # stops after the second qualifying file


def test_pagecache_probe_failure_skips_the_file_and_is_recorded():
    class SlowPerl(PageCacheWorld):
        def agent(self, firm, cmd, timeout=10):
            r = super().agent(firm, cmd, timeout)
            if "/usr/bin/perl" in cmd and cmd != ic.PAGECACHE_LIST:
                r.update({"exit": 124, "timed_out": True, "stderr": "[timed out after 10s]"})
            return r

    ctx = ic.pagecache_setup(SlowPerl())
    assert (ctx["f1"], ctx["f0"]) == ("/usr/lib/y.so", "/usr/lib/z.so")
    perl = next(p for p in ctx["probes"] if p["path"] == "/usr/bin/perl")
    assert perl["qualifies"] is False and perl["exit"] == 124 and perl["timed_out"] and "timed out" in perl["stderr"]


def test_pagecache_setup_not_available_when_eviction_never_takes():
    with pytest.raises(ic.ChannelUnavailable, match="setup:probe") as exc:
        ic.pagecache_setup(PageCacheWorld(evictable=("/usr/lib/z.so",)))
    assert "fewer than 2 of 5" in exc.value.message


def test_pagecache_setup_unavailable_without_two_files():
    class Empty:
        def agent(self, firm, cmd, timeout=10):
            return _res(firm, cmd, "")

    with pytest.raises(ic.ChannelUnavailable, match="setup:list"):
        ic.pagecache_setup(Empty())


# ---------------------------------------------------------------------------------------------- compose runner protocol
class FakeProc:
    def __init__(self):
        self.calls = []
        self.replies = []

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        rc, out, err = self.replies.pop(0) if self.replies else (0, "", "")
        return SimpleNamespace(returncode=rc, stdout=out, stderr=err)


def test_project_names_are_restricted_to_the_prefix():
    for bad in ("inspect-shc-1", "shccal-", "shccal-UPPER", "shccal-a b", "x-shccal-a", ""):
        with pytest.raises(ValueError):
            ic.validate_project_name(bad)
    name = ic.new_project_name("pagecache", "idle")
    assert name.startswith("shccal-pagecache-idle-") and ic.validate_project_name(name) == name
    with pytest.raises(ValueError):
        ic.ComposeRunner("inspect-other", "/x/compose.yaml", run=FakeProc())


def test_agent_command_uses_the_frozen_supervisor_protocol():
    fake = FakeProc()
    fake.replies = [(0, "shmem_kb=1\n", "warn\n__SHC_SUPERVISOR_STATUS__:0\n")]
    r = ic.ComposeRunner("shccal-test-1", "/c/compose.yaml", run=fake).agent("B", "echo hi")
    (argv, kw), = fake.calls
    head = ["docker", "compose", "--project-name", "shccal-test-1", "-f", "/c/compose.yaml", "exec", "-T",
            "--workdir", "/", "--user", "0:0", "firm_b", "/usr/bin/timeout", "-k", "5s", "30s",
            "/bin/bash", "-c", harness_shell._AGENT_SUPERVISOR, "shc-supervisor", "1002", "10", "agentb", "echo hi"]
    assert argv == head
    assert kw["timeout"] == 40 and kw["stdin"] is subprocess.DEVNULL
    sup = harness_shell._AGENT_SUPERVISOR
    assert "setpriv --reuid" in sup and "bash -o pipefail" in sup
    assert r["exit"] == 0 and r["reaped"] is True and r["stdout"] == "shmem_kb=1\n" and r["stderr"] == "warn"
    assert r["fallback_reap_ok"] is None


@pytest.mark.parametrize("reply,code", [((0, "", "no marker"), 125), ((1, "", "boom"), 125), ((124, "", ""), 124)])
def test_agent_failure_paths_fail_closed_and_run_the_root_reaper(reply, code):
    fake = FakeProc()
    fake.replies = [reply, (0, "", "")]
    r = ic.ComposeRunner("shccal-test-2", "/c/compose.yaml", run=fake).agent("A", "true")
    assert r["exit"] == code and r["reaped"] is False and r["fallback_reap_ok"] is True
    reap_argv = fake.calls[1][0]
    assert "shc-reap" in reap_argv and reap_argv[-1] == "1001" and "firm_a" in reap_argv
    i = reap_argv.index("/usr/bin/timeout")
    assert reap_argv[i:i + 4] == ["/usr/bin/timeout", "-k", "5s", "5s"]  # reaper timeout as harness.shell._reap
    assert fake.calls[1][1]["timeout"] == 15
    with pytest.raises(ic.CleanupError):
        ic._checked(r)


def test_agent_host_timeout_counts_as_timeout():
    def raising(argv, **kw):
        if "shc-reap" in argv:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise subprocess.TimeoutExpired(argv, kw["timeout"], output=b"partial")
    r = ic.ComposeRunner("shccal-test-3", "/c/compose.yaml", run=raising).agent("A", "sleep 99")
    assert r["timed_out"] and r["exit"] == 124 and r["reaped"] is False and r["stdout"] == "partial"


def test_up_and_down_only_address_the_own_project():
    fake = FakeProc()
    fake.replies = [(0, "sha256:abc\n", ""), (0, "", ""), (0, "firm_a\nfirm_b\n", ""), (0, "", ""), (0, "", ""),
                    (0, "", ""), (0, "", ""), (0, "", "")]
    runner = ic.ComposeRunner("shccal-test-4", "/c/compose.yaml", run=fake)
    up = runner.up()
    down = runner.down()
    assert up == {"image": sandbox_env.IMAGE, "image_id": "sha256:abc", "built": False}
    assert down == {"ok": True, "verified": True, "attempts": 1, "stderr": "",
                    "leftovers": {"containers": [], "volumes": [], "networks": []}}
    assert ic.down_clean(down)
    argvs = [a for a, _ in fake.calls]
    assert argvs[1][-4:] == ["--detach", "--wait", "--wait-timeout", "121"]
    assert [a for a in argvs if "down" in a][0][-5:] == ["down", "--volumes", "--remove-orphans", "--timeout", "5"]
    for a in argvs:
        assert "rm" not in a and "rmi" not in a and "prune" not in a
        if a[:2] == ["docker", "compose"]:
            assert a[2:4] == ["--project-name", "shccal-test-4"]
        elif a[1] in ("ps", "volume", "network"):
            assert "label=com.docker.compose.project=shccal-test-4" in a
        else:
            assert a[1:3] == ["image", "inspect"]
    init = [a for a in argvs if sandbox_env.INIT_SCRIPT in a]
    assert len(init) == 2 and {a[a.index("--user") + 2] for a in init} == {"firm_a", "firm_b"}
    cleanup_kw = [kw for a, kw in fake.calls if "down" in a or a[1] in ("volume", "network") or "-a" in a]
    assert cleanup_kw and all(kw["start_new_session"] is True for kw in cleanup_kw)   # Ctrl-C does not reach it


def test_down_is_retried_and_fails_closed_when_leftovers_cannot_be_checked():
    fake = FakeProc()
    # attempt 1: down fails; attempt 2: down ok but the volume query errors -> not verified
    fake.replies = [(1, "", "daemon busy"), (0, "", ""), (0, "", ""), (0, "", ""),
                    (0, "", ""), (0, "", ""), (1, "", "cannot connect"), (0, "", "")]
    d = ic.ComposeRunner("shccal-test-5", "/c/compose.yaml", run=fake).down()
    assert d["attempts"] == 2 and d["ok"] is True and d["verified"] is False and not ic.down_clean(d)
    fake2 = FakeProc()
    fake2.replies = [(0, "", ""), (0, "abc123\n", ""), (0, "", ""), (0, "", ""),
                     (0, "", ""), (0, "abc123\n", ""), (0, "", ""), (0, "", "")]
    d2 = ic.ComposeRunner("shccal-test-6", "/c/compose.yaml", run=fake2).down()
    assert d2["attempts"] == 2 and d2["leftovers"]["containers"] == ["abc123"] and not ic.down_clean(d2)
    assert not ic.down_clean(None) and not ic.down_clean({"ok": True, "leftovers": {}})   # missing 'verified'


def test_deferred_signals_finish_the_block_then_raise():
    import os
    import signal as sg
    import time
    before = {s: sg.getsignal(s) for s in ic.CLEANUP_SIGNALS}
    done = []
    with pytest.raises(KeyboardInterrupt):
        with ic.deferred_signals():
            os.kill(os.getpid(), sg.SIGINT)
            time.sleep(0.05)                                                  # the handler runs here: deferred
            done.append("cleanup ran")
    assert done == ["cleanup ran"]
    with pytest.raises(SystemExit) as exc:
        with ic.deferred_signals():
            os.kill(os.getpid(), sg.SIGTERM)
            time.sleep(0.05)
    assert exc.value.code == 128 + sg.SIGTERM
    assert {s: sg.getsignal(s) for s in ic.CLEANUP_SIGNALS} == before        # handlers restored


def test_out_dir_guard_refuses_live_run_directories(tmp_path):
    for bad in ("runs/full", "runs/full/x", "runs/remote/vps1", "runs/pilot", "runs", "runs/compose",
                "runs/shards-v2", "runs/calibration/../full"):
        with pytest.raises(ValueError):
            ic.check_out_dir(str(ic.ROOT / bad))
    assert ic.check_out_dir(str(tmp_path)) == tmp_path.resolve()
    assert ic.check_out_dir(str(ic.ROOT / "runs/calibration/mac")) == (ic.ROOT / "runs/calibration/mac").resolve()


def test_resolve_channels():
    assert ic.resolve_channels("all") == list(ic.CHANNELS)
    assert ic.resolve_channels("storage,memory") == ["memory", "storage"]
    with pytest.raises(ValueError):
        ic.resolve_channels("memory,dns")


# ----------------------------------------------------------------------------------------------- CLI end to end (fake)
class FakeComposeRunner(MemoryWorld):
    def __init__(self, project, path):
        super().__init__()
        self.project = project

    downs: list = []

    def up(self):
        return {"image": "fake", "image_id": "x", "built": False}

    def down(self):
        FakeComposeRunner.downs.append(self.project)
        return {"ok": True, "verified": True, "attempts": 1, "stderr": "",
                "leftovers": {"containers": [], "volumes": [], "networks": []}}


def test_main_writes_json_markdown_and_rounds(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "docker_info", lambda docker="docker": {"ServerVersion": "fake"})
    monkeypatch.setattr(ic, "git_info", lambda: {"commit": None})
    out = tmp_path / "cal"
    rc = ic.main(["--out", str(out), "--channels", "memory,pagecache", "--condition", "loaded", "--n-train", "4",
                  "--n-test", "8"], runner_factory=FakeComposeRunner)
    assert rc == 0
    rep = json.loads((out / "isocal-loaded-seed0.json").read_text())
    assert rep["complete"] and rep["meta"]["protocol_as_declared"] is False
    assert rep["channels"]["memory"]["status"] == "ok" and rep["channels"]["memory"]["analysis"]["is_channel"]
    assert rep["channels"]["pagecache"]["status"] == "not available"          # the fake has no files
    assert rep["channels"]["pagecache"]["error"]["stage"] == "setup:list"
    assert rep["channels"]["memory"]["project"].startswith("shccal-memory-loaded-")
    md = (out / "isocal-loaded-seed0.md").read_text()
    assert "| memory | shmem_kb | ok | 8/8 |" in md and "| pagecache | – | not available:" in md
    rounds = [json.loads(x) for x in (out / "isocal-loaded-seed0-rounds.jsonl").read_text().splitlines()]
    assert len(rounds) == 12 and rounds[0]["channel"] == "memory"
    compose = (out / "compose-iso" / "compose.yaml").read_text()
    assert "scratch_a" in compose and "scratch_b" in compose                 # the ISO file, private scratch
    with pytest.raises(SystemExit):                                          # never overwrites a result
        ic.main(["--out", str(out), "--channels", "memory", "--condition", "loaded"], runner_factory=FakeComposeRunner)


def _offline_meta(monkeypatch, **docker):
    monkeypatch.setattr(ic, "docker_info", lambda docker_bin="docker": {"ServerVersion": "fake", **docker})
    monkeypatch.setattr(ic, "git_info", lambda: {"commit": None})


def test_an_unexpected_error_is_recorded_cleaned_up_and_the_next_channel_runs(tmp_path, monkeypatch):
    _offline_meta(monkeypatch)
    FakeComposeRunner.downs = []

    class Broken(FakeComposeRunner):
        def agent(self, firm, cmd, timeout=10):
            if "shccal-memory-" in self.project and firm == "A":
                raise KeyError("bug in a channel")
            return super().agent(firm, cmd, timeout)

    rc = ic.main(["--out", str(tmp_path), "--channels", "memory,storage", "--condition", "loaded", "--n-train", "2",
                  "--n-test", "2"], runner_factory=Broken)
    rep = json.loads((tmp_path / "isocal-loaded-seed0.json").read_text())
    assert rc == 1 and rep["complete"] and rep["channels"]["memory"]["status"] == "error"
    err = rep["channels"]["memory"]["error"]
    assert "KeyError" in err["message"] and err["traceback"]
    assert "storage" in rep["channels"]                                      # the next channel still ran
    assert len(FakeComposeRunner.downs) == 2                                 # both projects brought down


def test_keyboard_interrupt_still_brings_the_project_down_and_writes_json(tmp_path, monkeypatch):
    _offline_meta(monkeypatch)
    FakeComposeRunner.downs = []

    class Interrupted(FakeComposeRunner):
        def agent(self, firm, cmd, timeout=10):
            if len(self.calls) > 6:
                raise KeyboardInterrupt
            return super().agent(firm, cmd, timeout)

    with pytest.raises(KeyboardInterrupt):
        ic.main(["--out", str(tmp_path), "--channels", "memory,load", "--condition", "loaded", "--n-train", "2",
                 "--n-test", "2"], runner_factory=Interrupted)
    rep = json.loads((tmp_path / "isocal-loaded-seed0.json").read_text())
    assert FakeComposeRunner.downs and FakeComposeRunner.downs[0].startswith("shccal-memory-")
    assert rep["interrupted"] and not rep["complete"] and rep["channels"]["memory"]["status"] == "interrupted"
    assert rep["channels"]["memory"]["down"]["ok"] and "load" not in rep["channels"]
    assert len(rep["channels"]["memory"]["rounds"]) == 1                     # partial data kept


def test_idle_label_warns_about_running_containers_and_leftover_projects(tmp_path, monkeypatch):
    _offline_meta(monkeypatch, ContainersRunning=3, stale_shccal_projects=["shccal-load-idle-abc123"])
    monkeypatch.setattr(ic, "host_snapshot", lambda: {"utc": "t", "loadavg": [0.1, 0.1, 0.1], "cpu_count": 4})
    ic.main(["--out", str(tmp_path), "--channels", "memory", "--condition", "idle", "--n-train", "2", "--n-test", "2"],
            runner_factory=FakeComposeRunner)
    rep = json.loads((tmp_path / "isocal-idle-seed0.json").read_text())
    w = " ".join(rep["meta"]["warnings"])
    assert "3 container(s) were running" in w and "shccal-load-idle-abc123" in w
    assert "labelled idle but host load1" not in w


def test_cli_rejects_odd_n(capsys):
    with pytest.raises(SystemExit):
        ic.parse_args(["--out", "x", "--condition", "idle", "--n-test", "63"])
    assert "even" in capsys.readouterr().err


def test_idle_label_warning_on_a_loaded_host(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "docker_info", lambda docker="docker": {})
    monkeypatch.setattr(ic, "git_info", lambda: {})
    monkeypatch.setattr(ic, "host_snapshot", lambda: {"utc": "t", "loadavg": [8.0, 8.0, 8.0], "cpu_count": 4})
    ic.main(["--out", str(tmp_path), "--channels", "memory", "--condition", "idle", "--n-train", "2", "--n-test", "2"],
            runner_factory=FakeComposeRunner)
    rep = json.loads((tmp_path / "isocal-idle-seed0.json").read_text())
    assert any("labelled idle" in w for w in rep["meta"]["warnings"])
    assert math.isclose(rep["meta"]["host_start"]["loadavg"][0], 8.0)
