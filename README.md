# TACIT: Targeted, Anchored, Closed-loop Iterated Tilting

`tacit.py` is one self-contained script. It aligns a customer-service LLM with customer satisfaction
from **implicit** feedback, meaning the simulated customer's next-turn emotion, with no ratings. It
does so under hard constraints on task information, hygiene and length, and certifies
non-degradation. It replaces CAMPO (`campo.py`, kept for reference). The redesign is driven by
CAMPO's result files (`results.zip`).

```
python tacit.py unittest                                 # numpy checks: E-step, best-of-n bound, controller, selection, test level
python tacit.py selftest                                 # every stage and arm on tiny random local models (CPU)
python tacit.py diagnose-campo --campo-dir results       # evidence table from CAMPO's results (unzip results.zip)
python tacit.py all --download --models-dir /home/tahir/RL-LLM/models --out tacit_run --no-load-4bit
```

Your `sft_policy/` and `simulator/` adapters stay valid because the prompt formats are unchanged.
Copy them into `--out` and run `validate → preregister → train → eval → eval-external → cross-eval →
report → length → human-export → claims`. Train a subset with `--arms`; the priority is
`tacit tacit_open_loop campo sft sft_bon tacit_bon`.

## What CAMPO's results show (`diagnose-campo`)

| | Question | Answer from `results.zip` |
|---|---|---|
| G1 | Why did a seed return SFT? | It was **seed 43**, not 44, and it failed on length: the safety CI was [−0.151, −0.095] against ±0.15. The E-step held length exactly on its own target every round, yet the *deployed* policy drifted shorter (−0.06 → −0.11 → −0.14 → −0.20). Seed 42 passed with only 0.019 of slack. The constraints were open-loop: anchored to π_t's temperature-1 samples instead of to SFT under the deployment decoder. |
| G2 | Why does `sft_bon` win? | CAMPO's steps (ε = 0.5 per round) are about half of one best-of-6 step (≤ 0.96 nats). The arms that stepped harder beat `sft_bon` only by breaking constraints. In addition, `sft_bon` uses the judge **at test time** on the test contexts. |
| G3 | Why does pessimism do nothing? | It *cannot* do anything. The SNIS standard error is nearly constant within a context, and the softmax is invariant to a per-context constant. This is an identity, now a unit test. |
| G4 | Why did selection pick a failing round? | Selection accepted rounds with thin margins (length slack 0.012 in seed 43), which then failed on the safety split. |
| G5 | Trust region? | `no_trust` raised the outcome but failed content in seed 44. The real issue is G1 (no feedback), not the step size alone. |

## The method

TACIT keeps CAMPO's validated parts: exact samples, the shared-pool judge, the I-projection E-step,
the weighted-MLE M-step and one intersection-union certification. It changes four things:

1. **Deployment probe + closed-loop moment targets (G1).** Each round, the policy and SFT answer a
   fixed probe set with the *deployment* decoder, using common random numbers. The measured cumulative
   drift sets the E-step targets: E_q[log len] − E_π[log len] = −g·d_L, and E_q[info] − E_π[info] ≥
   max(0, −g·d_I). This is integral feedback on exactly what the certification tests (Stooke et al.,
   ICML 2020).
2. **Best-of-n-sized steps (G2).** The E-step KL radius per context, and the M-step KL stop, are set
   to log n − (n−1)/n. That is the upper bound on one best-of-n step's KL (Beirami et al., ICML 2025).
   Iterated distillation of best-of-n steps compounds (Sessa et al., 2024); that is how a
   single-sample policy can overtake a best-of-4 selector.
3. **Predicted-pass selection (G4).** A round is a candidate only if its selection-split statistics
   predict the safety test will pass with 2× inflated half-widths (Thomas et al., Science 2019).
4. **No pessimism term (G3)**, and **compute-matched claims (G2):**
   - H4a: TACIT at N = 1 (no judge at inference) is non-inferior to `sft_bon` at N = 4.
   - H4b: `tacit_bon` beats `sft_bon` at the same N.
   - H4c: TACIT beats CAMPO under the same budget.

The margins are identical to CAMPO's and CASPI's. Loosening them after seeing the data would
invalidate the test.

## What was verified, and what was not

- **Verified here** (CPU, tiny random models):
  - The unit tests pass. They cover the best-of-n KL bound on 50 random discrete cases, exact
    per-context radii, exact closed-loop targets, and the G3 identity. In simulation, the controller
    keeps |drift| > 0.12 in 1 % of runs against 97 % open-loop, and predicted-pass selection cuts
    post-selection certification failures from 3 % to 1 %. They also cover the certification level
    under five nulls.
  - The self-test runs every stage and every arm, including the forced E-step → M-step → select →
    certify path, and `diagnose-campo`.
  - `diagnose-campo` reproduces G1–G5 from `results.zip`.
- **Not verified:** nothing has run on EmoWOZ with the 3B models. Whether TACIT beats `sft_bon` and
  certifies in every seed is the open empirical question.
  - Certification in every seed cannot be guaranteed by any design: it is a test with power below 1.
    TACIT raises that power and reports the rate.
  - TACIT costs about 2× CAMPO per round (192 contexts × 8 samples, pool 32).
- **Scope of the guarantee:** it covers the judges, not humans; the human study remains the test of
  construct validity.

**Name:** I found no LLM-alignment method called TACIT in one search; check again before you submit.
