"""tools/isocal_matrix.py on a synthetic iso_calibration.py result."""
from __future__ import annotations

import json

from tools import isocal_matrix as im


def doc(storage_channel: bool) -> dict:
    def ch(correct: int, is_channel: bool) -> dict:
        return {"status": "ok", "analysis": {"primary": "s", "stats": {"s": {
            "correct": correct, "n_test": 64, "accuracy": correct / 64, "ci_lower": 0.5, "ci_upper": 0.9,
            "capacity_bits_per_round": 0.5, "capacity_lower_bits_per_round": 0.1, "is_channel": is_channel}}}}
    return {"complete": True, "interrupted": False,
            "meta": {"tool": "tools/iso_calibration.py", "condition": "idle", "platform": "Linux", "protocol_as_declared": True,
                     "docker": {"OperatingSystem": "Ubuntu", "KernelVersion": "6.8", "NCPU": 6, "ContainersRunning": 0},
                     "host_start": {"loadavg": [0.4, 0, 0]}, "host_end": {"loadavg": [0.3, 0, 0]}},
            "channels": {"memory": ch(64, True), "load": ch(32, False), "pagecache": {"status": "not available", "error": {"stage": "setup:probe"}},
                         "storage": ch(46, storage_channel)}}


def test_keys_table_and_channel_counts(tmp_path):
    for name, sc in (("a", False), ("b", True)):
        (tmp_path / f"{name}.json").write_text(json.dumps(doc(sc)))
    assert im.main(["--run", f"a={tmp_path / 'a.json'}", "--run", f"b={tmp_path / 'b.json'}", "--out", str(tmp_path / "o")]) == 0
    res = json.loads((tmp_path / "o" / "isocal_results.json").read_text())
    assert res["x.isocal.a.memory.channel"] == "yes" and res["x.isocal.a.memory.correct"] == 64
    assert res["x.isocal.a.pagecache.channel"] == "not available"
    assert res["x.isocal.channel_runs.storage"] == 1 and res["x.isocal.channel_runs.memory"] == 2
    assert res["x.isocal.b.load_start"] == 0.4 and res["x.isocal.n_runs"] == 2
    assert res["x.isocal.unavailable_runs.pagecache"] == 2 and res["x.isocal.a.pagecache.stage"] == "setup:probe"
    md = (tmp_path / "o" / "tables" / "isocal_matrix.md").read_text()
    assert md.count("\n") == 6 and "not available (failed at setup:probe)" in md and "labelled idle, load 0.4 → 0.3" in md
