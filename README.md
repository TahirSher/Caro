# CASPI: Certified Affect-Safe Preference Iteration

`caspi.py` is one self-contained script. It aligns a customer-service LLM with customer satisfaction
from **implicit** feedback, meaning the simulated customer's next-turn emotion, with no ratings. The
model keeps the task information, and the script **certifies** that the returned policy is no worse
than SFT. It replaces the PACE pipeline (`pace-1`, ~7,900 lines) that the research brief analysed.
The old CARO machinery is gone: the panel estimator, logit projector, length calibrations, learned
reward model and GRPO.

```
python caspi.py unittest     # statistics, estimator and guarantee checks (numpy, seconds)
python caspi.py selftest     # every stage on tiny random local models (CPU, ~2 min)
python caspi.py all --download --models-dir /home/tahir/RL-LLM/models --out caspi_run --no-load-4bit
```

Existing `sft_policy/` and `simulator/` adapters stay valid because the prompt formats are
unchanged. Copy them into `--out` and run stages individually:
`validate → preregister → train → eval → eval-external → cross-eval → report → length →
human-export → claims`.

**Rename.** "PACE" collides with an iterative-DPO preprint (arXiv 2602.05370) and other 2026 work.
"CiPO" is also taken (ACL 2026). In one search I found no LLM-alignment method called CASPI, but
check again before you submit.

## How every gap in the brief is addressed

| Brief item | What `caspi.py` does |
|---|---|
| 6.3 / 7.1-1 Acceptance test under-powered (α/K per round, n≤300, margins 0.1·sd) | **Seldonian split.** The per-round guard runs on a *selection* split and is explicitly a heuristic. **One** certification test at full α runs on a disjoint *safety* split with one context per dialogue, so units are i.i.d. The test is intersection-union, so it needs no multiplicity correction across objectives or rounds. Every check logs its power at zero difference and the number of dialogues needed for 80 % power. |
| 7.1-2 Dev-pool estimator bias | Every guard and certification builds a **fresh** pool in which the SFT reply *and* the new reply are both proposals. There is an ESS floor. Pools are never reused across runs. |
| 7.1-3 Training-pool proposal asymmetry | **Every** candidate is a proposal of its context's pool. Pairs whose estimates rest on fewer than `min_pair_ess` effective replies are dropped. |
| 7.1-4 Silent component loss | Only the shared-pool estimator exists, so nothing can silently fall back. |
| 7.1-5 Length not in acceptance | TOST equivalence on the log word ratio (±0.15) is part of the guard and the certification. |
| 7.1-6 Likelihood displacement | Chosen and rejected log-likelihood changes are logged every round. If the chosen likelihood falls, the NLL anchor doubles. Near-duplicate pairs (Jaccard ≥ 0.9) are dropped. |
| 7.1-7 σ estimated once | σ is re-estimated **every round** on an independent replicate pool. |
| 7.1-8 Vacuous information constraint | The constraint pair now also requires that information is not lost. The vacuous share is logged per round, and the content share per test. |
| 7.1-9 Hygiene check used the mean only | Hygiene now gets a one-sided confidence bound like every other objective. |
| 7.1-10 Data-dependent margins | Margins are numeric and pre-registered (`preregistration.json`, written before training). Changing them raises an error. |
| 7 Over-claims in the docstrings | The docstring states a Proposition and proof sketch with exact conditions, plus what is **not** claimed: the judge's SNIS bias, judge ≠ humans, and that the guard carries no guarantee. It drops "off-policy corrected" and the DPO≡RL claim, and cites Seldonian/HC-RLHF. |
| G1/G3 Monitor shares the simulator | The monitor customer is now the **base** model with the adapter disabled, read by an independent labeller. This is partial independence: same base weights. |
| G4 Information measure is lexical | Rewrites must pass an **NLI** completeness/contradiction filter. The acceptance test still uses the lexical score, and this is stated. |
| G6 Rewrites unchecked | Covered by the same NLI filter. Rewrites come from the base model (off-policy), which the docs state. |
| G7 Evaluation circularity | **Symmetric evaluation pools**, where all arms are proposals of one pool per context. Also an out-of-family external judge, cross-evaluator agreement, a blinded human study, and `claims.md`. |
| RQ4 Baselines | `online_dpo`, `sentiment_only` (same optimiser, agent-wording sentiment), `offline_dpo` (same data budget), `sft_bon` (selected on an independent pool). |

## Bugs in the predecessor that this script fixes

1. **Inexact importance sampling.** The pool replies were sampled with the model's default
   generation config. Qwen2.5-Instruct ships `top_k=20` (`top_p=0.8`), and only `top_p` and the
   repetition penalty were overridden. The pool replies were also post-processed (`trim_to_sentence`)
   and re-tokenised before scoring. Both make the sampling density differ from the density in the
   weights.
   - **Fix:** CASPI samples with `top_k=0` and scores the **sampled token ids**.
   - **Verified here:** on the tiny model, 1,179 draws fell outside the top 20 tokens against an
     expected 1,148.
   - **Check your snapshot's** `generation_config.json`.
2. **Biased evaluation pools.** Eval pools were built from gold + the *first* arm's replies. Every
   later arm was scored by extrapolation. CASPI makes all arms proposals of one pool.
3. **Stale pool reuse.** Pools persisted across runs and were reused even when the proposals had
   changed. CASPI rebuilds them every run.

## What was verified, and what was not

- **Verified in this sandbox (CPU, no downloads):** the unit tests pass. The certification test's
  false-acceptance rate stays ≤ α under three nulls, for both the `t` and `bernstein` bounds, and a
  real gain is certified in 94 % of trials. SNIS is unbiased on a non-proposal target. Tail-logit
  scoring is exact. A fresh adapter equals the base model. A DPO step raises the margin. Adapters
  survive a save/load round trip. The self-test runs all 9 arms end to end, including a forced
  accept→certify path and the human-study analysis.
- **Not verified:** nothing has run on real EmoWOZ or the 3B models. With the default safety split
  (600 dialogues, margins of 0.2·sd), power should be adequate, but the logged
  `dialogues_for_80pct_power` is what tells you. A legitimate outcome is still "certification
  failed → SFT returned".
- **What the guarantee covers:** the *judge*, not humans. H5b and H5c (human study) remain the only
  test of construct validity.
