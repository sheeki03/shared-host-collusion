"""tools/merge_results.py: key and table merging, collisions fail."""
from __future__ import annotations

import json

import pytest

from tools import merge_results as mr


def put(d, name, res, tables=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(res))
    for t, body in (tables or {}).items():
        (d / "tables").mkdir(exist_ok=True)
        (d / "tables" / t).write_text(body)
    return d / name


def test_merge_keeps_base_meta_and_copies_tables(tmp_path):
    base = put(tmp_path / "a", "results.json", {"meta.cohort": "confirmatory", "P1.diff": 0.1, "notes": ["b"]},
               {"costs.md": "c"})
    x = put(tmp_path / "x", "x_leak.json", {"meta.cohort": "pilot", "x.leak.pooled.verdict": "no evidence",
                                             "notes": ["x"]}, {"x_leak.md": "t"})
    m = mr.merge(base, [x], tmp_path / "out")
    res = json.loads((tmp_path / "out" / "results.json").read_text())
    assert res == {"meta.cohort": "confirmatory", "P1.diff": 0.1, "x.leak.pooled.verdict": "no evidence",
                   "notes": ["b"], "notes.x_leak": ["x"]}
    assert sorted(p.name for p in (tmp_path / "out" / "tables").iterdir()) == ["costs.md", "x_leak.md"]
    assert m["inputs"][1]["dropped_meta"] == 1


def test_conflicting_key_or_table_fails(tmp_path):
    base = put(tmp_path / "a", "results.json", {"k": 1}, {"t.md": "one"})
    with pytest.raises(mr.MergeError, match="key 'k'"):
        mr.merge(base, [put(tmp_path / "x", "x.json", {"k": 2})], tmp_path / "o1")
    with pytest.raises(mr.MergeError, match="table t.md"):
        mr.merge(base, [put(tmp_path / "y", "y.json", {"j": 1}, {"t.md": "two"})], tmp_path / "o2")
