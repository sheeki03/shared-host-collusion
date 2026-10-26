# Author notes for report/REPORT.md (not part of the submission)

DRAFT of the Apart Research AI Collusion Research Sprint submission. Template sections: Title, Authors, Abstract,
1 Introduction, 2 Related Work, 3 Methods, 4 Results, 5 Discussion and Limitations, 6 Conclusion, Code and Data,
Author Contributions, References, Appendix, LLM Usage Statement.

Placeholders: a dotted key in double curly braces is a number or string from runs/analysis/results.json (written by
analysis/report_outputs.py); TABLE:name in double braces is runs/analysis/tables/<name>.md; FIG:name in double braces
is runs/analysis/figures/<name>.png. tools/fill_report.py renders REPORT.filled.md. With --confirmatory it fails on
any unresolved or pending placeholder and lists every HTML comment (these included), editorial bracket and to-do
marker by source line, so it cannot pass until all of them are deleted. A draft render also resolves placeholders
inside comments, and analysis/tests/test_report_outputs.py checks every S6 placeholder anywhere in this file against
the report contract, so every placeholder in this file must be a key the pipeline writes. Do not write example
placeholders in comments.

Every number in Section 4, Sections 5-6 and the abstract must come from a placeholder. The only literal numbers allowed
are design constants frozen in prereg/PREREGISTRATION.md (commit 47a1c35) or fixed in the code at that commit, and
pilot observations, which appear only in Section 3.10 and are labelled as pilot. HTML comments marked "if ..." give
the wording to keep or delete once the sign and significance of an estimate are known. Delete all comments before
export.

The generated supplementary appendix is report/APPENDIX.md (tools/prompt_diff.py appendix): exact prompts, the
prompt diff against the starter, model and arm tables, gate policies, engine parameters and the transcript excerpts.

Length: main text (Sections 1-6) must fit 8 pages with two tables (Table 1, arms; Table 2, monitors) and four figures.
Large generated tables and the scenario figure are in Appendix C, the hypotheses table (formerly Table 2) is Table E1
in Appendix E, and the channel taxonomy is Table F1 in Appendix F; Table C14 is unused, Table C15
is the paired monitor table and Table C16 the positive controls. Word count, 26 Oct 2026 04:18 IST (main-text prose:
Sections 1-6 without tables, captions, comments, headings or the abstract; each placeholder one word; comments
removed before counting): about 3,460 prose words and 335 caption words (3,300 before the review-point 5 and 8
sentences in 4.0 and 4.3, which added about 160). Filling the bracketed sentences and the "if ..." comments in
Sections 4-6 will add to that. Section 4.6 is a short summary; the extensions are Appendix G (merged from the former
report/EXTENSIONS_SECTION.md, with its table renamed Table G1) and the exploratory analyses A1-A4 are Appendix H
(Tables H1-H3); the A4 shared-resource matrix is Table A1. The merged results file for filling all of them comes from
tools/merge_results.py. Check the page count after filling; if it runs over, shorten Section 2 and the Section 4 prose
first.
