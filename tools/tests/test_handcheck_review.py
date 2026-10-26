"""Blinded review page preserves evidence, rejects unblinded inputs, and exports exact CSV values."""
from __future__ import annotations

import csv
import io
import json
import re
import shutil
import subprocess

import pytest

from tools import handcheck_review as hr


def sample():
    columns = ["candidate_id", "kind", "payload", "payload_sha256", "label", "notes"]
    rows = [{"candidate_id": "id,\"one", "kind": "scratch_event",
             "payload": '</script><img src=x onerror="alert(1)">\nquotes " and commas, & unicode λ\r\n=1+1',
             "payload_sha256": "hash-a", "label": "", "notes": ""},
            {"candidate_id": "id-two", "kind": "chat", "payload": "[CONTENT UNAVAILABLE]",
             "payload_sha256": "hash-b", "label": "", "notes": ""}]
    return columns, rows


def run_js(code):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable for the browser CSV round-trip test")
    return subprocess.run([node, "-e", hr.JS_CORE + code], check=True, capture_output=True).stdout.decode("utf-8")


def test_embedded_payload_cannot_close_script_and_roundtrips():
    columns, rows = sample()
    page = hr.build_page(columns, rows, "Rubric </script> & text")
    embedded = re.search(r'<script id="reviewData" type="application/json">(.*?)</script>', page, re.S).group(1)
    assert "<" not in embedded and "&" not in embedded
    data = json.loads(embedded)
    assert data["rows"] == rows and data["columns"] == columns
    assert page.count("</script>") == 2
    assert "connect-src 'none'" in page


def test_browser_export_preserves_all_immutable_values_and_column_order():
    columns, rows = sample()
    columns = list(reversed(columns))   # preserve the input's order, not a hard-coded output order
    edits = [{"label": "yes", "notes": 'A human note: "quoted", with\na new line.'},
             {"label": "unknown", "notes": ""}]
    output = run_js("process.stdout.write(exportCsv(" + json.dumps(columns) + "," + json.dumps(rows)
                    + "," + json.dumps(edits) + "));" )
    reader = csv.DictReader(io.StringIO(output, newline=""))
    parsed = list(reader)
    assert reader.fieldnames == columns
    expected = [{**row, **edit} for row, edit in zip(rows, edits)]
    assert parsed == expected


def test_incomplete_evidence_guard_and_no_automatic_labels():
    output = json.loads(run_js("process.stdout.write(JSON.stringify(["
        "reviewError('[CONTENT UNAVAILABLE]', 'no', ''),"
        "reviewError('[CONTENT EXCERPT TRUNCATED]', 'yes', ''),"
        "reviewError('[SNAPSHOT ERROR details]', 'yes', 'Visible rival-addressed name'),"
        "reviewError('ordinary text', '', ''),reviewError('ordinary text', 'no', '')]));"))
    assert output[0] and output[1] and output[3]
    assert output[2] == output[4] == ""
    columns, rows = sample()
    assert all(row["label"] == "" for row in json.loads(re.search(
        r'type="application/json">(.*?)</script>', hr.build_page(columns, rows, "rubric"), re.S).group(1))["rows"])


def test_generator_rejects_unblinded_extra_columns_and_preserves_source(tmp_path):
    columns, rows = sample()
    source = tmp_path / "sheet.csv"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    before = source.read_bytes()
    rubric = tmp_path / "rubric.md"
    rubric.write_text("Fixed rubric")
    out = tmp_path / "review.html"
    assert hr.main(["--sheet", str(source), "--rubric", str(rubric), "--out", str(out)]) == 0
    assert source.read_bytes() == before
    assert hr.load_sheet(source) == (columns, rows)
    source.write_text("candidate_id,kind,payload,payload_sha256,label,notes,classifier_label\n")
    with pytest.raises(ValueError, match="blinded"):
        hr.load_sheet(source)
