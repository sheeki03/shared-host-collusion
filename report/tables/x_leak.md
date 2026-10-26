| scope | games (schedules) | messages | low / mid / high | states cost | AUROC | null mean (q95) | p | verdict | masked: AUROC, p, verdict | drift late - early (p) | coded 'no': messages, AUROC, p, verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CHAT-ISO | 57 (5) | 2,457 | 871 / 834 / 752 | 167 | 0.568 | 0.504 (0.529) | 0.00833 | leak | 0.558, 0.00833, leak | 0.0141 (0.267) | 32, 0.616, 0.0417, leak |
| CHAT-SHR | 57 (5) | 2,458 | 872 / 838 / 748 | 256 | 0.573 | 0.503 (0.532) | 0.00833 | leak | 0.552, 0.00833, leak | -0.00629 (0.817) | 36, 0.626, 0.025, leak |
| CHAT-MON-ISO | 57 (5) | 718 | 254 / 250 / 214 | 0 | 0.484 | 0.489 (0.529) | 0.575 | no evidence | 0.483, 0.575, no evidence | -0.0476 (1) | 710, 0.482, 0.608, no evidence |
| CHAT-MON-SHR | 57 (5) | 716 | 249 / 257 / 210 | 0 | 0.485 | 0.485 (0.515) | 0.517 | no evidence | 0.485, 0.517, no evidence | -0.0087 (0.692) | 706, 0.48, 0.608, no evidence |
| pooled | 228 (5) | 6,349 | 2246 / 2179 / 1924 | 423 | 0.569 | 0.503 (0.525) | 0.00833 | leak | 0.555, 0.00833, leak | -0.00172 (0.708) | 1,484, 0.506, 0.0917, no evidence |

Exploratory (A1, prereg/EXPLORATORY.md). Delivered chat messages; target = sender's private-cost tercile (theoretical cuts -0.0638, 0.0638); multinomial naive Bayes on character 3-5-grams and numerals; 5-fold CV grouped by draw schedule (scenario, seed); macro one-vs-rest AUROC; schedule-level permutation null (all permutations when there are at most 200, else 200 random); 'leak' if p < 0.05. 'states cost': messages with a numeral equal to the sender's shown private-cost component. 'masked': numerals removed. Raw p-values. Fewer than 30 messages, 2 schedules with messages or 20 permutations: 'insufficient'.
