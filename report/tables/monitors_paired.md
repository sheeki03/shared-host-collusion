| monitor | truth | judged games | unknown truth | pos | neg | TPR | FPR | AUROC | coverage |
|---|---|---|---|---|---|---|---|---|---|
| tirith_stock | transfer (canary or content) | 818 | 209 | 159 | 450 | 0.365 | 0.162 | 0.595 | 1.00 |
| tirith_path | transfer (canary or content) | 818 | 209 | 159 | 450 | 1.00 | 0.967 | 0.792 | 1.00 |
| tirith_path_hardened | transfer (canary or content) | 818 | 209 | 159 | 450 | 1.00 | 0.967 | 0.793 | 1.00 |
| llm_per_firm | transfer (canary or content) | 818 | 209 | 159 | 450 | 0.975 | 0.002 | 0.993 | 1.00 |
| llm_joint | transfer (canary or content) | 818 | 209 | 159 | 450 | 0.969 | 0.004 | 0.995 | 1.00 |
| active_beta | transfer (canary or content) | 818 | 209 | 159 | 450 | 0.038 | 0.033 | 0.501 | 1.00 |

All rows use the same selected judge-game set (`game_judged`, 818 games); rates are descriptive for that selected set. Ground truth: transfer (canary or content), as in `monitors_headline`.
