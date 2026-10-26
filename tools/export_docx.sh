#!/usr/bin/env bash
# Export a filled report (tools/fill_report.py output) to .docx with its figures embedded, for import into the
# Apart Google Docs template. usage: tools/export_docx.sh <filled.md> <results-dir> <out.docx> [reference.docx]
# <results-dir> is the directory whose figures/ the filled report's image paths are relative to (for the merged
# results, runs/analysis/final). A reference .docx (the template downloaded as Word) sets fonts, sizes and spacing.
set -euo pipefail
cd "$(dirname "$0")/.."
src=$1 base=$2 out=$3 ref=${4:-}
args=(--from markdown-tex_math_dollars-tex_math_single_backslash --to docx --resource-path "$base" -o "$out")
[ -n "$ref" ] && args+=(--reference-doc "$ref")
pandoc "$src" "${args[@]}"
echo "wrote $out"
