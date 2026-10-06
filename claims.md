# Claims audit (generated; edit the rules, never the verdicts)

| id | claim | verdict |
|---|---|---|
| caspi_vs_sft:outcome | H1: CASPI raises simulated customer affect over SFT | not supported |
| caspi_vs_caspi_no_rewrite:outcome | H3a: delivery-only rewrites contribute | not supported |
| caspi_vs_caspi_no_pareto:outcome | H3b: noise-calibrated Pareto pairs contribute | not supported |
| caspi_vs_online_dpo:outcome | H4: CASPI beats online DPO with the same judge | not supported |
| caspi_vs_sft_bon:outcome | H4: CASPI beats best-of-N with the same judge | not supported |
| caspi_vs_sentiment_only:outcome | H4: the simulated-customer signal beats agent-wording sentiment with the same optimiser | not supported |
| caspi_vs_offline_dpo:outcome | H4: CASPI beats offline DPO with the same data budget | not supported |
| H2 | CASPI conveys no less task information than SFT (non-inferiority) | supported |
| H3c | the acceptance machinery costs no affect (TOST equivalence) | not supported: the guard costs affect |
| H5a | the gain replicates with an out-of-family customer and an unseen emotion classifier | not supported |
| H5b | humans prefer CASPI over SFT on satisfaction | untested |
| H5c | the judge ranks alternative replies to the SAME context as humans do | untested (needs the human study) |
| H5c_automatic | two independent automatic evaluators agree on within-context rankings above chance | supported |
| judge_partial_anchor | the judge tracks the next-turn human emotion beyond the current emotion | supported |
| H6 | no returned CASPI policy is significantly worse than SFT on any objective (test split) | supported |
| certified_improvement | CASPI certified an improvement in every seed | not supported (1 of 1 seeds returned SFT) |
