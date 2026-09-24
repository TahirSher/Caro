# PACE: content-equivalent affect alignment from implicit customer emotion

`pace.py` is a single end-to-end script that replaces CARO v12. It trains a customer-service LLM to
deliver **the same information** in a way that leaves the customer in a better emotional state. It
learns from the customer's next utterance, scored by an emotion detector and a sentiment model, and
needs no explicit ratings.

```
python pace.py unittest                    # numpy statistical tests (seconds)
python pace.py selftest                    # whole pipeline on tiny local random models, CPU (~10 s)
python pace.py all --download --models-dir /home/tahir/RL-LLM/models --out pace_run --no-4bit
```

Models it needs: `Qwen/Qwen2.5-3B-Instruct` (policy),
`cardiffnlp/twitter-roberta-base-sentiment-latest` (encoder backbone and sentiment), and
`cross-encoder/nli-deberta-v3-base` (fidelity). You have the first two already. The third must be
added to your cache.

---

## 1. Critique of CARO v12, based on your own logs

| # | Finding | Evidence |
|---|---|---|
| 1 | **The signal RL actually uses was never validated against humans.** GRPO only sees within-context differences. About 74% of label variance is between contexts and cancels inside a group. The human anchor (ρ=0.547) is a between-context number. The only within-context human check is the gold-response margin, and it is essentially zero. | corpus: `within var=0.0037 between=0.0105`; reward: `gold-anchor rho=+0.0614` (v11 and v12 5-member), `+0.1423` (v12 10-member). The gate prints a warning and still passes. |
| 2 | **Humans do not license the simulator's brevity preference.** v12 calibrates the reward's length slope *to the labels'*, arguing that the labels' length association is content signal. Your own log refutes that: human labels do not track length at all. | validate: `observational within-context length rho -0.177 … human label vs length rho +0.002`; corpus labels `rho -0.278`. |
| 3 | **The outcome estimator is surface-brittle, not length-biased.** O(x) is a softmax over PMI(reply_j; x) for 48 replies taken from *other* dialogues, so it mostly measures topical overlap. Moving the same filler to a different position disturbs it as much as adding filler does. Versions v8 to v12 patched "length" while the root cause was generic brittleness. | validate: `length-matched surface null tau=0.206 (vs length tau=0.214)`, flips `0.0789 vs 0.0833`; `effective panel size 17.4 of 48`; `rho=0.4001` with next-turn sentiment. |
| 4 | **It is very expensive.** Each label costs 48 LM passes. The fast scorer fell back to the slow path. | corpus: `falling back to the v8 scorer` → **48,732 s (13.5 h) for 3,000 contexts**; each validate run took about 5 h and ran at least 3 times. |
| 5 | **Length leaks into the reward features by construction.** `Policy.features` mean-pools over prompt+response tokens, so the response's weight n_r/(n_p+n_r) is a function of length. The CLP and length-calibration grid then fought that leak and lost on TEST. | DEV excess `-0.026 CI[-0.060,+0.012]` becomes TEST `-0.090 CI[-0.122,-0.053]` (v11) and `-0.122 CI[-0.154,-0.086]` (v12). Accuracy fell from 0.743 to 0.704. |
| 6 | **The reward gate is lenient and selection overfits DEV.** The gate fails only if the *whole* CI lies beyond 0.10, so an excess CI of [-0.154,-0.086] passes. Selection takes the first grid cell that passes on DEV, out of 28 cells (winner's curse). One run logged "no pair met both DEV criteria; keeping the last one" and then passed anyway. | reward log, 21:11:54 → 21:13:10 `pass=True` |
| 7 | **Information preservation is never operationalised.** The policy never sees backend facts, so it must invent reference numbers and availability. Nothing measures whether it drops bad news, and a sentiment reward rewards doing exactly that. The `sentiment_only` arm is a textbook sycophancy set-up. | code: `agent_prompt` has no facts; no fidelity metric anywhere |
| 8 | **The policy gradient is biased.** Generation output is post-processed (`trim_to_sentence`), then re-tokenised with an EOS appended. The gradient is taken on tokens that were never sampled. With one update per batch the PPO ratio is identically 1, so the clip never acts. Per-sequence mean normalisation is length-biased (Liu et al., 2025). | `generate` → `grpo_step` re-encodes `" " + text + eos` |
| 9 | **Evaluation is circular and seed pooling is invalid.** `evaluate` scores policies with the same simulator that produced the reward labels. The report averages per-seed p-values before applying Holm, which is not a valid way to combine p-values. | `stage_eval`, `stage_report` |
| 10 | Dead complexity. The control variate `b=-0.937` almost undoes `raw - cur` (y ≈ raw - 0.063·cur) and is constant within a group anyway. "ICC=1.000, rel=1.000" holds by construction for a deterministic estimator. | validate log |

**Where your framing is not correct.** Using the sentiment or emotion of the user's *next* turn as
an automatic reward in place of explicit ratings is **not new**. Shi & Yu (ACL 2018, *Sentiment
Adaptive End-to-End Dialog Systems*) used detected user sentiment as an RL reward in task-oriented
dialogue. Jaques et al. (2019, *Way Off-Policy Batch Deep RL of Implicit Human Preferences in
Dialog*) trained offline on implicit reactions, sentiment included. Sharma et al. (WWW 2021, PARTNER)
used RL to rewrite text for empathy while preserving its content. A paper therefore cannot claim
novelty for the *idea*. It has to claim novelty for *how the implicit signal is identified and
safely optimised*, which is where PACE differs. I could not run a literature search from this
sandbox (network blocked), so check the novelty statements below before you publish.

## 2. The PACE algorithm

1. **Emotion/Satisfaction Detection (ESD).** A RoBERTa classifier over the 7 EmoWOZ emotions, trained
   on customer utterances only. It never sees the agent turn, so the outcome is not a function of
   the treatment. It uses temperature scaling. The utility follows EmoWOZ's elicitor taxonomy:
   *apologetic* counts 0, not −0.5 as in v12. Satisfaction is `S = E_p[U(e)] + β·sentiment`, with β
   selected against human labels.
2. **Content-orthogonal phrasing effect.** This is the novel estimand.
   `τ(h,a) = E[S|h,a] − E[S|h,c(a)]`, where `c(a)` is the delexicalised dialogue acts (social acts
   removed) or lexical information units. The nuisance `m(h,c)` is cross-fitted by dialogue and
   linearly recalibrated. `g(h,a)` is fitted on the residuals with Poisson-bootstrapped heads, and
   the reward is the lower confidence bound. For equal content, τ differences equal true phrasing
   differences. Content leakage enters only at second order (unit test 2: bad-news sensitivity
   −0.61 naive vs −0.08 partialled, with the empathy effect preserved).
3. **Constraints.** Information fidelity = `P_ent(response ⇒ source) · (1 − P_contra(source ⇒ response))`.
   This allows empathy to be added but forbids dropping or contradicting facts. Any number, time or
   reference code not in the source counts as fabrication. Length has an explicit budget.
4. **PACE update.** The affect advantage is a leave-one-out advantage computed **only inside the
   content-equivalent feasible set**, and groups whose spread is inside the ensemble noise are gated
   out. Lagrangian constraint advantages get projected dual ascent. The loss is a Dr.-GRPO constant
   normalisation plus PPO-clip over several epochs on the **exact sampled tokens**, with k3 KL to the
   base model obtained through adapter disabling.
5. **Non-circular evaluation.** A judge trained on *human* labels from dialogues the detector and
   reward never saw. It is reported together with fidelity, fabrication and length. Seeds are
   averaged per context, then a dialogue-cluster bootstrap and sign-flip test are run, with Holm
   correction across the trained arms.
6. **Gate before RL.** Within the same content stratum and given m̂, τ must predict the **human**
   label on the test split (lower CI > 0). If it fails, RL is refused. No threshold is tuned.

Arms: `source`, `base` (the instruct model rewriting zero-shot), `pace`, `pace_unconstrained`,
`pace_naive_reward` (no partialling-out), and `sentiment_only`.

## 3. What was verified here, and what was not

Verified in this sandbox (CPU only, no model downloads):
* `unittest`, units 2 to 5 (unit 1 checks the statistics helpers and prints nothing):
  * Partialling-out removes about 86% of the content (bad-news) sensitivity and keeps the phrasing
    effect (0.243 against a truth of 0.250).
  * In a constrained bandit set in a bad-news context, PACE converges to "keep the news, say it
    empathetically" with probability ≥ 0.992 on each of 20 seeds. Affect-only optimisation of
    either reward drops the bad news with probability ≥ 0.998.
* `selftest` runs every stage end to end on tiny randomly initialised local models. It checks that
  a fresh adapter equals the reference, that one PPO step moves probability towards
  positive-advantage samples, that the KL is then positive, and that the adapter survives a
  save/load round trip.

Not verified. Say so if you report results:
* **Nothing has been run on real EmoWOZ with the real 3B model here.** No real-data numbers exist
  yet. Runtime estimates (H200): detector, reward and judge together under 30 min. RL takes roughly
  15–30 s per step, so about 1.5–2.5 h per arm and seed at 300 steps. Use `--no-4bit` on an H200,
  because bitsandbytes 4-bit generation is slow, and run arms in parallel.
* **The unit tests show the anti-sycophancy protection comes from the feasibility constraint, not
  from partialling-out.** The naive reward *with* PACE's constraints also converges correctly, and
  the unconstrained τ reward drops bad news too. On synthetic data with tiny models, partialling-out
  did not beat the naive reward on within-content validity (+0.20 vs +0.29), and τ kept content
  leakage because m̂ under-fitted. The gate log flags that leakage.
  The value of partialling-out on real data has to be shown by the `pace_naive_reward` ablation and
  the gate diagnostics.
* EmoWOZ next-turn emotions are mostly neutral and the human wizards were already polite. The
  within-content phrasing signal may be too weak for the gate to pass. That would be a finding
  about the data, not a reason to lower the threshold.
* Identification assumes no unobserved confounder of phrasing and outcome given (context, content),
  such as agent identity. NLI does not catch non-numeric hallucinations that are merely "neutral".
  The judge is a model. Claims need a human evaluation.
* I could not inspect the EmoWOZ files, because Zenodo is blocked here. If the MultiWOZ part carries
  no `dialog_act`, the lexical descriptor is used instead. The `data` stage logs the share of turns
  with acts.
