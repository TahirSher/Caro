# CAMPO: Constrained, Anchored Mirror-descent Policy Optimisation

`campo.py` is one self-contained script. It aligns a customer-service LLM with customer satisfaction
from **implicit** feedback, meaning the simulated customer's next-turn emotion, with no ratings. It
replaces CASPI (`caspi.py`, kept for reference). The redesign is driven by the CASPI logs, not by
assumptions.

```
python campo.py unittest                               # numpy checks: statistics, E-step, mirror descent, guarantee
python campo.py selftest                               # every stage and arm on tiny random local models (CPU)
python campo.py diagnose-caspi --caspi-dir caspi_run   # evidence table from your CASPI logs
python campo.py all --download --models-dir /home/tahir/RL-LLM/models --out campo_run --no-load-4bit
```

Your CASPI `sft_policy/` and `simulator/` adapters stay valid because the prompt formats are
unchanged. Copy them into `--out` and run `validate → preregister → train → eval → eval-external →
cross-eval → report → length → human-export → claims`.

## What the CASPI logs show (`diagnose-caspi`)

| | Question | Answer from the logs |
|---|---|---|
| E1 | Is the implicit signal too weak? | **No.** Partial anchor ρ = +0.51 [0.45, 0.57]. Within-context signal sd is 0.196 against noise sd 0.099. Cross-evaluator agreement is 0.589 [0.558, 0.619]. |
| E2 | Can learners raise the outcome? | **Yes.** Every learner gained between +0.05 and +0.10 over SFT, and the gains replicate on the external evaluator. |
| E3 | Why did CASPI return SFT? | **The acceptance machinery.** All 4 guard rounds had affect lower bounds > 0. 3 of them were rejected only for hygiene (−0.010 to −0.030 against a 0.02 margin) or length (−0.24 against ±0.15). Each rejection rolled back, doubled β and halved the learning rate. Certification then failed on the same two constraints. |
| E4 | Were the constraints optimised? | **No, only tested.** 59 % of CASPI's pairs were "constraint" pairs. The rejected log-likelihood fell by up to 30 nats per round, pushing replies shorter. online_dpo bought the largest outcome with hygiene 0.945 (SFT 0.996) and 12.0 words (SFT 14.7). |
| E5 | Is there a safe mechanism? | **Yes.** Best-of-4 over SFT samples gained +0.084 with no loss of hygiene, length or information. |

Where the earlier hypotheses were wrong:

- **H1/H3a/H3b/H4 for CASPI** were not supported because `caspi == sft`, not because the idea failed.
- **"CASPI beats online DPO on raw outcome"** is the wrong hypothesis when online DPO violates
  hygiene and length. Under a binding constraint the constrained optimum cannot exceed the
  unconstrained one. H4 is now a Pareto claim.
- **The margins are unchanged** (they are identical to CASPI's). Loosening them after seeing the
  data would invalidate the test, so the optimiser was changed instead.

## The method: optimise exactly what is certified

Each round, for each training context:

1. **Sample.** Draw N exact samples from the current policy (temperature 1, no top-k or top-p, sampled
   token ids kept).
2. **Judge.** Score them on CASPI's shared simulator pool, which E1 validated. The reward is
   pessimistic: affect minus κ times the delta-method SNIS standard error (Owen, 2013).
3. **Project.** This is the new step. Take the I-projection onto the certified constraint set:
   `q*(y) ∝ 1[hygienic, EOS] · exp((r + λ·info + μ·log len) / η_x)`.
   - η_x is solved per context so that KL(q‖uniform) = ε, which is the MPO trust region.
   - μ is solved so that the expected log length equals the sampling policy's.
   - λ is the smallest value that keeps the expected information.
   - All three are exact, by bisection.

   This is a constrained, soft best-of-N (E5): best-of-N is the special case η→0 without constraints.
4. **Distil.** Weighted maximum likelihood of the exact samples, i.e. forward KL to q*. It cannot push
   any reply's likelihood down (E4). It stops when an unbiased, term-wise non-negative estimate of
   KL(π_t‖π_{t+1}) on held-out π_t samples (the "k3" estimator) exceeds `kl_max`.
5. **Select and certify, with no ratchet.** Every round is a checkpoint, and nothing is rolled back
   or tightened. The best lower bound among the checkpoints that pass the constraint tests on the
   selection split wins. Then one intersection-union test at α runs on a dialogue-disjoint safety
   split; if it fails, SFT is returned.

Iterating steps 3–4 is KL mirror descent: π_T ∝ π_0 exp(Σ_t s/η_t) on the feasible set. The unit
tests check this identity, the KL radius, the moment constraints, and the false-acceptance rate of
the certification test under five nulls.

**Arms:** `campo`, `campo_no_{pessimism,moments,trust,select}`, `sentiment_only`, `online_dpo` and
`offline_dpo` (both on the same samples and judge), and `sft` / `sft_bon`.

## What was verified, and what was not

- **Verified here** (CPU, tiny random models):
  - The unit tests pass.
  - The `diagnose-caspi` verdicts above come from your uploaded logs.
  - The self-test runs every stage and every arm, including a forced select→certify path.
- **Not verified:** nothing has run on EmoWOZ with the 3B models.
  - Whether CAMPO's outcome gain matches `caspi_no_guard`'s +0.09 while passing hygiene and length is
    the open empirical question. The E-step only guarantees the constraints for the training target
    in expectation; whether the fitted policy also meets them is what the certification test checks.
  - A legitimate outcome is still "no checkpoint passes → SFT returned". In that case the logs show
    which constraint, and `dialogues_for_80pct_power` says whether the test was under-powered.
- **Scope of the guarantee:** it covers the judges (simulator and monitor), not humans. H5b and H5c
  (the human study) remain the only test of construct validity.

**Name:** I found no LLM-alignment method called CAMPO in one search; check again before you submit.
