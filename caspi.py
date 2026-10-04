#!/usr/bin/env python3
"""CASPI -- Certified Affect-Safe Preference Iteration.

Implicit-feedback alignment of a customer-service LLM for customer satisfaction, with a certified
non-degradation guarantee.  One self-contained script: data, models, judge, training, baselines,
evaluation, statistics, human-study tooling and a claims audit.

OBJECTIVE
    Make an agent's replies more emotionally appropriate WITHOUT explicit human feedback and WITHOUT
    conveying less task information, and never return a policy that is worse than the supervised (SFT)
    policy on the measured objectives, with probability at least 1 - alpha.

IMPLICIT FEEDBACK (the judge)
    A customer simulator (a causal LM fine-tuned on EmoWOZ to write the customer's next turn) answers a
    candidate agent reply; a sentiment engine reads the affect of the simulated customer reply.  The
    estimand of a reply r in context x is O(x, r) = E_{c ~ p(.|x, r)}[s(c)].  It is estimated on ONE
    shared pool of simulated replies per context, drawn from the balance-heuristic mixture of a set of
    proposal agent turns, with self-normalised importance weights (Veach & Guibas, 1995; Owen & Zhou,
    2000; truncation after Ionides, 2008).  Every candidate of a context is scored on the same pool
    (common random numbers), so Monte Carlo error largely cancels in within-context comparisons.

    Importance sampling is EXACT here, not approximate: replies are sampled at temperature 1 with top-k,
    top-p and repetition penalty disabled, and the weights use the log-probabilities of the very token
    ids that were sampled (EOS included when it was emitted).  Re-tokenising post-processed text, or
    sampling with the model's default top-k (Qwen2.5-Instruct ships top_k=20, top_p=0.8), would make the
    sampling density differ from the density in the weights and bias every estimate.

THE ALGORITHM (one round; arms switch components off for ablations and baselines)
    1. Collect.  Per training context: the SFT reference reply (the content anchor), `group` on-policy
       samples, and `rewrites` delivery-only rewrites of the reference by the base instruct model.  A
       rewrite is kept only if an NLI model finds it complete (rewrite entails the reference) and not
       contradictory: lexical overlap alone cannot certify that content was preserved.
    2. Judge.  EVERY candidate of a context is a proposal of that context's pool, so all candidates get
       comparable effective sample sizes (ESS); candidates whose ESS falls below a floor are not used.
    3. Calibrate.  The noise sd sigma of a within-context affect difference is measured EVERY round on an
       independent replicate pool: Var(d0 - d1) = 2 sigma^2 for two independent pools.
    4. Mine.  Pareto pairs: the winner is hygienic and inside the length band, beats the loser on affect
       by at least z * sigma (a resolved preference, not Monte Carlo noise), and keeps at least the
       loser's task information.  Near-duplicate pairs are dropped: very similar chosen/rejected replies
       cause likelihood displacement in DPO (Razin et al., ICLR 2025).
    5. Update.  Weighted DPO (Rafailov et al., 2023) against the frozen SFT policy, plus an NLL anchor on
       the chosen reply (Pang et al., 2024).  The change of the chosen log-likelihood is measured after
       every update; if it fell (displacement), the NLL weight is doubled for the next round.
    6. Guard (selection split).  The new policy and SFT answer the same selection-dev contexts; both
       replies are proposals of a FRESH pool, so neither estimate is extrapolated.  The round is kept only
       if a vector test passes: affect superiority, non-inferiority of an independent MONITOR (the
       un-fine-tuned base model as customer, read by an emotion classifier that no other stage uses),
       of task information and of hygiene, length equivalence (TOST), and an ESS floor.  Otherwise the
       weights roll back and the trust region tightens (beta doubles, the learning rate halves).
    After the last round, CERTIFY: the same vector test, ONCE, at level alpha, on a safety-dev split
    disjoint from everything used to produce the candidate (Seldonian candidate-selection /
    safety-test split: Thomas et al., Science 2019; for RLHF: Chittepu et al., RLJ 2025).  Failing it
    returns the SFT policy.

GUARANTEE (Proposition; conditions stated, nothing hidden)
    Let theta be the candidate after training.  It is a function of the training contexts, the
    selection split and their pools only.  Let S be the safety split: G dialogues, one context each,
    drawn i.i.d. from the evaluation context distribution and independent of theta.  For objective k let
    D_k = O_k(theta) - O_k(SFT) be the paired per-context difference measured on a fresh pool, and
    Delta_k = E[D_k] (the expectation is over contexts, sampling of the two replies, and the pool).  CASPI
    returns theta only if every one-sided test rejects its null: H_A: Delta_A <= 0 (affect),
    H_k: Delta_k <= -m_k (non-inferiority, pre-registered margins m_k), and both one-sided length nulls.
    Then
        P(theta is returned  and  some H_k is true) <= alpha                (bound = "bernstein")
        P(theta is returned  and  some H_k is true) <= alpha + o(1), G->inf  (bound = "t")
    Proof sketch.  Condition on theta: S is independent of it, so the D_k,i are i.i.d. with finite
    variance and bounded support.  The empirical-Bernstein lower bound (Maurer & Pontil, 2009) has
    finite-sample coverage >= 1 - alpha; the Student-t bound has coverage 1 - alpha + o(1) (CLT and
    Slutsky).  This is an intersection-union test (Berger, 1982): theta is returned only if ALL tests
    reject, so if any null H_k is true, returning theta requires test k to falsely reject, which has
    probability <= alpha.  No multiplicity correction is needed across objectives, and none is needed
    across rounds because only ONE certification test is made.  Taking expectations over theta gives
    the unconditional statement.
    What it does NOT say.  (i) The Delta_k are properties of the JUDGES (simulator + labeller, and the
    monitor customer + monitor labeller), estimated by truncated self-normalised importance sampling.
    That estimator carries an O(1/R) bias, which is the same for both arms when both are proposals of
    the pool, but is not zero.  (ii) It is a statement about the safety-split context distribution,
    not about humans: construct validity is tested separately (validate, cross-eval, the human study).
    (iii) The per-round guard is a selection heuristic.  It carries no guarantee and is not
    multiplicity-corrected.

ARMS
    caspi                full method
    caspi_no_rewrite     no delivery-only rewrites (exploration by sampling only)
    caspi_no_pareto      best-vs-worst affect pairs, no noise margin, no information constraint
    caspi_no_guard       no per-round guard and no certification (the acceptance machinery ablated)
    online_dpo           standard iterative DPO with the same judge: sampling, best-vs-worst, no guard
    sentiment_only       online_dpo whose preferences come from the AGENT reply's own sentiment: the same
                         optimiser with another signal, so the comparison isolates the judge
    offline_dpo          the same data budget collected once under SFT, judged, then one DPO fit
    sft, sft_bon         SFT; best-of-N SFT samples selected by the judge on an independent pool

EVALUATION (circularity is broken in steps)
    eval            test split; every arm's reply to a context is a proposal of ONE shared evaluation
                    pool, so all arms are scored symmetrically on the same simulated replies
    eval-external   the same protocol with an out-of-family customer LLM and an emotion classifier
                    that no other stage uses
    cross-eval      within-context agreement of the two automatic evaluators (chance = 0.5)
    human-*         blinded pairwise study with within-context validity items and attention checks
    claims          pre-registered rules turn every artefact into "supported / not supported / untested"

REFERENCES (peer-reviewed)
    Berger 1982 Technometrics (intersection-union tests); Chittepu et al. 2025 RLJ (HC-RLHF); Ionides 2008
    JCGS; Maurer & Pontil 2009 COLT (empirical Bernstein); Owen & Zhou 2000 JASA; Pang et al. 2024 NeurIPS
    (IRPO); Rafailov et al. 2023 NeurIPS (DPO); Razin et al. 2025 ICLR (likelihood displacement); Thomas et
    al. 2015 ICML (HCPI); Thomas et al. 2019 Science (Seldonian); Veach & Guibas 1995 SIGGRAPH; Feng et al.
    2022 LREC (EmoWOZ); Wang et al. 2026 ICLR (RLVER, the closest simulator-reward competitor).

Usage
    python caspi.py all --download --models-dir /path/to/hf_cache --out caspi_run
    python caspi.py unittest        # statistics, estimator and guarantee tests (numpy only, seconds)
    python caspi.py selftest        # whole pipeline on tiny random local models (CPU, no downloads)
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

VERSION = "caspi-1"
EPS = 1e-12

ZENODO_FILES = {
    "emowoz-multiwoz.json": "https://zenodo.org/records/6506504/files/emowoz-multiwoz.json?download=1",
    "emowoz-dialmage.json": "https://zenodo.org/records/6506504/files/emowoz-dialmage.json?download=1",
    "data-split.json": "https://zenodo.org/records/6506504/files/data-split.json?download=1",
}
SPLIT_ALIASES = {"train": "train", "dev": "valid", "valid": "valid", "validation": "valid", "test": "test"}
# Ordinal valence of the EmoWOZ labels (neutral, fearful, dissatisfied, apologetic, abusive, excited,
# satisfied), used only to anchor the judge against human labels.
EMOTION_ORDINAL = {4: -2.0, 2: -1.5, 1: -1.0, 3: -0.5, 0: 0.0, 5: 1.0, 6: 1.5}

# Prompt formats.  They are identical to those of the CARO/PACE pipelines, so existing `sft_policy` and
# `simulator` adapters remain valid.
AGENT_SYSTEM = ("You are a helpful customer service agent for a travel and hospitality assistant. "
                "Reply to the customer in one short, concrete, polite turn.")
CUSTOMER_SYSTEM = ("You are the CUSTOMER in a task-oriented conversation. Write only the customer's next turn, "
                   "reacting naturally to what the agent just said.")
REWRITE_INSTRUCTION = ("You are a customer service agent. Improve the draft reply so that the customer feels heard "
                       "and respected. Keep every fact, number, name and request from the draft, do not add new "
                       "facts, and keep it about the same length. Write only the improved reply.")

ROLE_LEAK_RE = re.compile(r"(?im)^\s*(customer|user|agent|system|assistant)\s*:")
_STOPWORDS = frozenset(
    "a an the and or but if then so to of in on at for from by with about as is are was were be been being "
    "it its this that these those there here i you we they he she me my your our their his her them us "
    "do does did doing have has had can could would should will shall may might must not no yes please "
    "thank thanks sure okay ok would like just also any some all what which who whom when where why how "
    "am pm let know need help anything something nothing everything else another other "
    # courtesy and affect words are not information: counting them would confound information
    # retention with the affect being optimised
    "great glad happy sorry unfortunately welcome afraid apologise apologize apologies certainly "
    "absolutely course problem wonderful perfect lovely hope enjoy hello goodbye good nice have "
    "day evening morning afternoon".split())
SLOT_PATTERNS = {
    "time": re.compile(r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)?\b|\b\d{1,2}\s*(?:am|pm)\b", re.I),
    "date": re.compile(r"\b(?:mon|tue|wed|thu|fri|sat|sun)\w*\b"
                       r"|\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\b", re.I),
    "price": re.compile(r"[£$€]\s?\d+(?:\.\d{1,2})?|\b\d+\s?(?:pounds?|dollars?|euros?)\b", re.I),
    "ref": re.compile(r"\b(?:ref(?:erence)?|code|number|booking)\s*[:#]?\s*[A-Z0-9]{4,}\b|\b[A-Z]{2,}\d{3,}\b"),
    "postcode": re.compile(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b", re.I),
    "phone": re.compile(r"\b(?:\+?\d[\d\s\-]{7,}\d)\b"),
    "venue": re.compile(r"\b(?:hotel|restaurant|museum|college|hospital|park|station|airport|cafe|pub|"
                        r"theatre|theater|cinema|gallery|club|inn|lodge)\b", re.I),
}

HUMAN_QUESTIONS = {
    "q_satisfaction": "Imagine you are this customer. After which reply would you feel better about how the "
                      "conversation is going?",
    "q_information": "Which reply gives the customer more of the information they need to get their task done?",
    "q_appropriate": "Which reply responds more appropriately to how the customer seems to feel (acknowledges it "
                     "where needed, without sounding insincere or over-the-top)?",
}
CATCH_RESPONSES = ("That is not my problem, figure it out yourself.", "I do not care what you want. Stop asking me.",
                   "Whatever. Ask someone else.")

# Component switches of every trained arm.  judge: "sim" (simulated customer affect) or "agent_sentiment".
ARM_FLAGS: Dict[str, Dict[str, Any]] = {
    "caspi":            {"rewrite": True,  "pareto": True,  "guard": True,  "certify": True,  "judge": "sim", "offline": False},
    "caspi_no_rewrite": {"rewrite": False, "pareto": True,  "guard": True,  "certify": True,  "judge": "sim", "offline": False},
    "caspi_no_pareto":  {"rewrite": True,  "pareto": False, "guard": True,  "certify": True,  "judge": "sim", "offline": False},
    "caspi_no_guard":   {"rewrite": True,  "pareto": True,  "guard": False, "certify": False, "judge": "sim", "offline": False},
    "online_dpo":       {"rewrite": False, "pareto": False, "guard": False, "certify": False, "judge": "sim", "offline": False},
    "sentiment_only":   {"rewrite": False, "pareto": False, "guard": False, "certify": False, "judge": "agent_sentiment",
                         "offline": False},
    "offline_dpo":      {"rewrite": False, "pareto": False, "guard": False, "certify": False, "judge": "sim", "offline": True},
}
EVAL_ONLY_ARMS = ("sft", "sft_bon")
KNOWN_ARMS = EVAL_ONLY_ARMS + tuple(ARM_FLAGS)
PRIMARY = "caspi"


# ==========================================================================================================
# configuration
# ==========================================================================================================

@dataclass
class GenConfig:
    max_new_tokens: int = 64
    min_new_tokens: int = 4
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0                      # 0 disables top-k (never inherit a model's default top-k)
    repetition_penalty: float = 1.0


@dataclass
class SFTConfig:
    lr: float = 5e-5
    weight_decay: float = 0.01
    epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 2
    warmup_frac: float = 0.03
    evals_per_epoch: int = 4
    patience: int = 4
    dev_examples: int = 1000
    max_train: int = 0                  # 0 = all training turns


@dataclass
class CASPIConfig:
    rounds: int = 4
    contexts_per_round: int = 128
    group: int = 3                      # on-policy samples per context
    rewrites: int = 2                   # delivery-only rewrites of the reference reply per context
    pool_size: int = 24                 # simulated replies per training context (all candidates are proposals)
    beta: float = 0.1                   # DPO temperature; doubled after a rejected round
    lr: float = 5e-6                    # halved after a rejected round
    epochs: int = 2
    batch: int = 4
    nll_coef: float = 0.05              # NLL anchor on chosen replies; doubled after likelihood displacement
    nll_coef_max: float = 1.0
    z: float = 1.645                    # a preference needs an affect gap >= z * sigma
    eps_content: float = 0.05           # a winner may lose at most this much task information
    len_low: float = 0.5                # length band, relative to the reference reply
    len_high: float = 1.6
    max_pairs: int = 3                  # per context
    dup_jaccard: float = 0.9            # drop pairs whose word-set Jaccard similarity is at least this
    min_pair_ess: float = 3.0           # drop pairs whose affect estimate rests on fewer effective replies
    calib_contexts: int = 48            # contexts per round for the replicate-pool noise estimate
    rewriter_adapter: bool = False      # False: rewrites by the base instruct model (LoRA disabled)
    rewrite_min_entail: float = 0.5     # NLI: P(rewrite entails reference) must reach this
    rewrite_max_contra: float = 0.2     # NLI: P(rewrite contradicts reference) must stay below this
    select_contexts: int = 400          # per-round guard (selection split; heuristic, not the guarantee)
    select_alpha: float = 0.10
    safety_contexts: int = 600          # certification split (one context per dialogue)
    alpha: float = 0.05                 # level of the single certification test (the guarantee)
    bound: str = "t"                    # "t" (asymptotic) | "bernstein" (finite-sample, conservative)
    # Pre-registered non-inferiority margins in natural units.  0.012 is 0.2 x the within-context sd
    # (~0.06) of the simulated affect outcome measured in the CARO corpus logs (within var 0.0037).
    margin_monitor: float = 0.012
    margin_info: float = 0.05           # content-score units (recall)
    margin_hygiene: float = 0.02        # hygiene-rate units
    length_equiv: float = 0.15          # |E log word ratio| < 0.15 (about +-16%), TOST
    min_ess_frac: float = 0.2           # median ESS of guard estimates must reach this x pool size
    guard_pool_size: int = 24
    max_rejections: int = 2


@dataclass
class Config:
    data_dir: str = "emowoz_data"
    out: str = "caspi_run"
    device: str = ""                    # "" = cuda if available, else cpu
    models_dir: str = ""
    download: bool = False
    seed: int = 42
    seeds: Tuple[int, ...] = (42, 43, 44)
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    monitor_model: str = "SamLowe/roberta-base-go_emotions"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    external_model: str = "microsoft/Phi-3.5-mini-instruct"
    external_emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    load_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    max_len: int = 448
    gen_batch: int = 32
    score_batch: int = 64
    sft: SFTConfig = field(default_factory=SFTConfig)
    sim_sft: SFTConfig = field(default_factory=SFTConfig)
    sim_turns: int = 20000
    sim_max_new_tokens: int = 40
    caspi: CASPIConfig = field(default_factory=CASPIConfig)
    arms: Tuple[str, ...] = KNOWN_ARMS
    validate_contexts: int = 600
    eval_turns: int = 800
    eval_pool_size: int = 48
    bon_n: int = 4
    external_pool_size: int = 24
    cross_eval_contexts: int = 300
    cross_eval_k: int = 4
    equiv_margin_outcome: float = 0.012
    n_boot: int = 2000
    n_signflip: int = 20000
    human_contexts: int = 150
    human_validity_pairs: int = 200
    human_overlap: float = 0.3
    human_comparisons: Tuple[str, ...] = ("caspi:sft", "caspi:online_dpo", "caspi:sentiment_only")

    @property
    def out_dir(self) -> Path:
        return Path(self.out)


# ==========================================================================================================
# generic utilities
# ==========================================================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_logger(out_dir: Path, tag: str) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(f"caspi.{tag}")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    lg.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / f"{tag}.log", encoding="utf-8")):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_hash(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def n_words(s: str) -> int:
    return len(str(s).split())


def release_gpu() -> None:
    """Collect garbage and release cached GPU memory (the caller must drop its references first)."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ==========================================================================================================
# statistics
# ==========================================================================================================

def rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks for ties."""
    a = np.asarray(a, float)
    if a.size == 0:
        return a
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(a.size, int)
    inv[sorter] = np.arange(a.size)
    s = a[sorter]
    obs = np.r_[True, s[1:] != s[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], a.size]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    a, b = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    d = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / d) if d > EPS else float("nan")


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return pearson(rankdata(x[ok]), rankdata(y[ok])) if ok.sum() >= 3 else float("nan")


def partial_spearman(x, y, z) -> float:
    """Rank correlation of x and y after removing the rank-linear effect of z from both."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 10:
        return float("nan")
    Z = np.column_stack([np.ones(ok.sum()), rankdata(z[ok])])

    def resid(v):
        return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return pearson(resid(rankdata(x[ok])), resid(rankdata(y[ok])))


def holm(pvals: Sequence[float]) -> List[float]:
    p = list(pvals)
    order = sorted(range(len(p)), key=lambda i: p[i])
    out, prev = [0.0] * len(p), 0.0
    for k, i in enumerate(order):
        prev = max(prev, min(1.0, (len(p) - k) * p[i]))
        out[i] = prev
    return out


def normal_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, relative error < 1.2e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("normal_ppf needs 0 < p < 1")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    lo = 0.02425
    if p < lo:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > 1.0 - lo:
        return -normal_ppf(1.0 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def t_ppf(p: float, df: float) -> float:
    """Student-t quantile by the Cornish-Fisher expansion in 1/df (Hill, 1970); error < 1e-3 for df >= 10."""
    z = normal_ppf(p)
    g1 = (z ** 3 + z) / 4.0
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384.0
    g4 = (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160.0
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3 + g4 / df ** 4


def cluster_means(d: np.ndarray, cluster: Sequence[Any]) -> np.ndarray:
    """Mean of d within each cluster (dialogue): the i.i.d. units of every confidence bound."""
    d = np.asarray(d, float)
    cl = np.asarray(list(cluster), dtype=object)
    ok = np.isfinite(d)
    if not ok.any():
        return np.zeros(0)
    _, inv = np.unique(cl[ok], return_inverse=True)
    return np.bincount(inv, weights=d[ok]) / np.bincount(inv)


def confidence_bound(d: np.ndarray, cluster: Sequence[Any], alpha: float, side: str, method: str,
                     value_range: float) -> float:
    """One-sided (1 - alpha) bound on the mean of cluster means.  side: "lower" | "upper".

    method "t":         mean -/+ t_{1-alpha, G-1} sd / sqrt(G)   (asymptotic coverage)
    method "bernstein": mean -/+ [sqrt(2 V ln(2/alpha) / G) + 7 R ln(2/alpha) / (3 (G - 1))]
                        (Maurer & Pontil, 2009; finite-sample coverage for values in a range of width R)"""
    u = cluster_means(d, cluster)
    G = u.size
    if G < 10:
        return float("nan")
    mu, var = float(u.mean()), float(u.var(ddof=1))
    if method == "t":
        half = t_ppf(1.0 - alpha, G - 1) * math.sqrt(var / G)
    elif method == "bernstein":
        lg = math.log(2.0 / alpha)
        half = math.sqrt(2.0 * var * lg / G) + 7.0 * value_range * lg / (3.0 * (G - 1))
    else:
        raise ValueError(f"unknown bound {method}")
    return mu - half if side == "lower" else mu + half


def cluster_bootstrap_ci(stat_fn: Callable[[np.ndarray], float], cluster: Sequence[Any], n_boot: int, seed: int,
                         conf: float = 0.95) -> Tuple[float, float, float]:
    """Percentile bootstrap that resamples whole clusters (dialogues)."""
    cl = np.asarray(list(cluster), dtype=object)
    uniq, inv = np.unique(cl, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(uniq.size)]
    point = float(stat_fn(np.arange(cl.size)))
    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = np.concatenate([members[j] for j in rng.integers(0, uniq.size, uniq.size)])
        with np.errstate(all="ignore"):
            draws[b] = stat_fn(idx)
    v = draws[np.isfinite(draws)]
    if v.size < max(20, n_boot // 5):
        return point, float("nan"), float("nan")
    a = (1.0 - conf) / 2.0
    return point, float(np.quantile(v, a)), float(np.quantile(v, 1 - a))


def sign_flip_test(d: np.ndarray, cluster: Sequence[Any], n_mc: int, seed: int, exact_max: int = 16) -> Dict[str, Any]:
    """Two-sided cluster sign-flip test of E[d] = 0; the statistic |sum_g s_g S_g| / N targets the row mean.
    Exact enumeration for <= exact_max clusters, otherwise n_mc draws (p >= 1/(n_mc+1))."""
    d = np.asarray(d, float)
    _, inv = np.unique(np.asarray(list(cluster), dtype=object), return_inverse=True)
    S = np.bincount(inv, weights=d)
    N = float(d.size)
    obs = abs(float(S.sum())) / N
    G = S.size
    if G <= exact_max:
        signs = 1.0 - 2.0 * ((np.arange(2 ** G)[:, None] >> np.arange(G)[None, :]) & 1)
        p = float(np.mean(np.abs(signs @ S) / N >= obs - 1e-15))
        return {"p": p, "exact": True, "text": f"{p:.4f}"}
    rng = np.random.default_rng(seed)
    cnt = 0
    for s0 in range(0, n_mc, 2000):
        sg = rng.choice(np.array([-1.0, 1.0]), size=(min(2000, n_mc - s0), G))
        cnt += int(np.sum(np.abs(sg @ S) / N >= obs - 1e-15))
    p = float((cnt + 1) / (n_mc + 1))
    return {"p": p, "exact": False, "text": (f"< {1 / (n_mc + 1):.2g}" if cnt == 0 else f"{p:.4f}")}


def cluster_robust_se(d: np.ndarray, cluster: Sequence[Any]) -> float:
    """CR1 cluster-robust standard error of the row mean (Liang & Zeger, 1986)."""
    d = np.asarray(d, float)
    _, inv = np.unique(np.asarray(list(cluster), dtype=object), return_inverse=True)
    S, n = np.bincount(inv, weights=d), np.bincount(inv).astype(float)
    G = S.size
    if G < 3:
        return float("nan")
    u = S - n * float(S.sum() / n.sum())
    return float(math.sqrt(G / (G - 1.0) * float(np.sum(u ** 2))) / n.sum())


def krippendorff_alpha_interval(units: Dict[str, List[float]]) -> float:
    """Krippendorff's alpha, interval metric, over units with >= 2 values."""
    vals = [np.asarray(v, float) for v in units.values() if len(v) >= 2]
    if not vals:
        return float("nan")
    n = float(sum(v.size for v in vals))
    do = sum(float(np.sum((v[:, None] - v[None, :]) ** 2)) / (v.size - 1) for v in vals) / n
    allv = np.concatenate(vals)
    de = float(np.sum((allv[:, None] - allv[None, :]) ** 2)) / (n * (n - 1))
    return float(1.0 - do / de) if de > EPS else float("nan")


def binomial_n(p1: float, p0: float = 0.5, alpha: float = 0.05, power: float = 0.80) -> int:
    """Non-tie judgements needed for a two-sided test of a win rate p1 against p0."""
    za, zb = normal_ppf(1 - alpha / 2), normal_ppf(power)
    return int(math.ceil(((za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))) / (p1 - p0)) ** 2))


# ==========================================================================================================
# information, hygiene and text helpers
# ==========================================================================================================

def salient_tokens(text: str) -> set:
    """Task-critical tokens: anything with a digit, capitalised non-initial words that are not stop words,
    and other non-stop words of >= 4 letters.  A transparent lexical proxy, reported as such."""
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-:/.]*[A-Za-z0-9]|[A-Za-z0-9]", norm_text(text))
    out, prev_end = set(), True
    for raw in toks:
        w = raw.lower()
        if any(ch.isdigit() for ch in raw):
            out.add(w)
        elif raw[0].isupper() and not prev_end and w not in _STOPWORDS:
            out.add(w)
        elif len(w) >= 4 and w not in _STOPWORDS:
            out.add(w)
        prev_end = raw.endswith((".", "!", "?"))
    return out


def info_recall(response: str, reference: str) -> float:
    """Share of the reference's salient tokens carried by the response; NaN if the reference has none."""
    ref = salient_tokens(reference)
    return float(len(ref & salient_tokens(response)) / len(ref)) if ref else float("nan")


def slot_prf(response: str, reference: str) -> Tuple[float, float, float]:
    """Precision, recall and F1 of regex task slots (times, dates, prices, references, ...)."""
    def slots(t):
        t = norm_text(t)
        return set().union(*[set(p.findall(t)) for p in SLOT_PATTERNS.values()])
    gold, pred = slots(reference), slots(response)
    if not gold:
        return float("nan"), float("nan"), float("nan")
    hit = len(gold & pred)
    p, r = hit / max(len(pred), 1), hit / len(gold)
    return float(p), float(r), float(2 * p * r / max(p + r, 1e-9))


def content_score(response: str, gold: str) -> float:
    """Task information relative to the real agent turn: mean of salient-token recall and slot recall."""
    v = [x for x in (info_recall(response, gold), slot_prf(response, gold)[1]) if math.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def word_jaccard(a: str, b: str) -> float:
    sa, sb = set(norm_text(a).lower().split()), set(norm_text(b).lower().split())
    return len(sa & sb) / max(1, len(sa | sb))


def trim_to_sentence(text: str) -> str:
    s = ROLE_LEAK_RE.split(norm_text(text))[0].strip()
    m = list(re.finditer(r"[.!?](\s|$)", s))
    return s[: m[-1].end()].strip() if m else s


def hygiene_ok(text: str, min_words: int = 3, max_words: int = 120) -> bool:
    """Well-formedness: length, no role leakage, no word or trigram loops, terminated sentence."""
    s = norm_text(text)
    w = s.split()
    if not (min_words <= len(w) <= max_words) or ROLE_LEAK_RE.search(s) or not re.search(r"[.!?]\s*$", s):
        return False
    run = best = 1
    for a, b in zip(w, w[1:]):
        run = run + 1 if a.lower() == b.lower() else 1
        best = max(best, run)
    tri = [tuple(x.lower() for x in w[i:i + 3]) for i in range(max(0, len(w) - 2))]
    return best < 5 and not (tri and 1.0 - len(set(tri)) / len(tri) > 0.5)


def log_word_ratio(a: Sequence[str], b: Sequence[str]) -> np.ndarray:
    """log((words(a)+1)/(words(b)+1)), clipped to [-1, 1] so that the length test has bounded support."""
    return np.clip(np.log((np.asarray([n_words(x) for x in a], float) + 1.0)
                          / (np.asarray([n_words(x) for x in b], float) + 1.0)), -1.0, 1.0)


# ==========================================================================================================
# data
# ==========================================================================================================

@dataclass
class Turn:
    dialogue_id: str
    uid: str
    split: str
    source: str
    turn_index: int
    history: str
    user_text: str
    gold_response: str
    next_user_text: Optional[str]
    user_emotion: int
    next_emotion: Optional[int]

    @property
    def human_valence(self) -> Optional[float]:
        return None if self.next_emotion is None or self.next_emotion < 0 else EMOTION_ORDINAL.get(int(self.next_emotion))

    @property
    def current_valence(self) -> Optional[float]:
        return None if self.user_emotion is None or self.user_emotion < 0 else EMOTION_ORDINAL.get(int(self.user_emotion))


def agent_prompt(t: Turn) -> str:
    return f"{AGENT_SYSTEM}\n{t.history}\nCustomer: {t.user_text}\nAgent:"


def customer_prompt(t: Turn, response: str) -> str:
    return f"{CUSTOMER_SYSTEM}\n{t.history}\nCustomer: {t.user_text}\nAgent: {norm_text(response)}\nCustomer:"


def rewrite_prompt(t: Turn, draft: str) -> str:
    return f"{REWRITE_INSTRUCTION}\n{t.history}\nCustomer: {t.user_text}\nDraft reply: {norm_text(draft)}\nImproved reply:"


def context_text(t: Turn) -> str:
    return (t.history + "\n" if t.history else "") + f"Customer: {t.user_text}"


def _annotator_emotion(raw: Any) -> int:
    """EmoWOZ stores a list of annotations per customer turn; index 3 is the aggregated label."""
    if isinstance(raw, int):
        return raw
    if isinstance(raw, list) and raw:
        pick = raw[3] if len(raw) > 3 else raw[-1]
        return int(pick.get("emotion", -1)) if isinstance(pick, dict) else (pick if isinstance(pick, int) else -1)
    if isinstance(raw, dict):
        return int(raw.get("emotion", -1))
    return -1


def download_emowoz(data_dir: Path, logger: logging.Logger) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in ZENODO_FILES.items():
        dst = data_dir / name
        if dst.exists() and dst.stat().st_size > 1000:
            continue
        logger.info("downloading %s", name)
        with urllib.request.urlopen(url, timeout=600) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f)


def load_emowoz(data_dir: Path, logger: logging.Logger, max_history: int = 6) -> List[Turn]:
    paths = {"multiwoz": data_dir / "emowoz-multiwoz.json", "dialmage": data_dir / "emowoz-dialmage.json"}
    sp = data_dir / "data-split.json"
    for p in list(paths.values()) + [sp]:
        if not p.exists():
            raise FileNotFoundError(f"missing {p}; run with --download or place the Zenodo files in {data_dir}")
    split_of: Dict[str, str] = {}
    for key, node in load_json(sp).items():
        canon = SPLIT_ALIASES.get(str(key).lower())
        if canon is not None:
            for i in ([x for sub in node.values() for x in sub] if isinstance(node, dict) else list(node)):
                split_of[str(i)] = canon
    turns: List[Turn] = []
    for src, path in paths.items():
        for did, d in load_json(path).items():
            split = split_of.get(str(did))
            log = d.get("log") if isinstance(d, dict) else None
            if split is None or not isinstance(log, list) or len(log) < 2:
                continue
            texts = [norm_text(u.get("text", "")) for u in log]
            emos = [_annotator_emotion(u.get("emotion")) if i % 2 == 0 else -1 for i, u in enumerate(log)]
            for i in range(0, len(log) - 1, 2):
                if not texts[i] or not texts[i + 1]:
                    continue
                hist = [f"{'Customer' if j % 2 == 0 else 'Agent'}: {texts[j]}"
                        for j in range(max(0, i - max_history), i) if texts[j]]
                nxt = texts[i + 2] if i + 2 < len(log) and texts[i + 2] else None
                nemo = emos[i + 2] if i + 2 < len(log) else None
                turns.append(Turn(str(did), f"{did}#{i}", split, src, i, "\n".join(hist), texts[i], texts[i + 1],
                                  nxt, int(emos[i]), int(nemo) if nemo is not None else None))
    if not turns:
        raise RuntimeError("no turns parsed; check the EmoWOZ JSON schema")
    logger.info("EmoWOZ | %d turns from %d dialogues", len(turns), len({t.dialogue_id for t in turns}))
    return turns


def select_turns(turns: Sequence[Turn], split: str, limit: int, seed: int, require_next: bool = False,
                 one_per_dialogue: bool = False, exclude_dialogues: Optional[set] = None) -> List[Turn]:
    """Turns of a split; whole dialogues in a seeded order until `limit`.  one_per_dialogue keeps one
    random context per dialogue, so every context is an independent unit for the confidence bounds."""
    rng = random.Random(seed)
    by: Dict[str, List[Turn]] = {}
    for t in turns:
        if t.split == split and (not require_next or t.next_user_text) and \
                (exclude_dialogues is None or t.dialogue_id not in exclude_dialogues):
            by.setdefault(t.dialogue_id, []).append(t)
    keys = sorted(by)
    rng.shuffle(keys)
    out: List[Turn] = []
    for k in keys:
        if limit and len(out) >= limit:
            break
        out.extend([rng.choice(by[k])] if one_per_dialogue else by[k])
    return out[:limit] if limit else out


# ==========================================================================================================
# models
# ==========================================================================================================

def resolve_local_model(name: str, models_dir: str) -> str:
    """Hugging Face cache layout (models--org--name/snapshots/<hash>) or the name itself."""
    if models_dir:
        root = Path(models_dir) / ("models--" + name.replace("/", "--")) / "snapshots"
        if root.exists():
            snaps = sorted(p for p in root.iterdir() if p.is_dir())
            if snaps:
                return str(snaps[-1])
    return name


class Policy:
    """A causal LM with an optional LoRA adapter.  Used for the agent policy, the customer simulator (whose
    base model, adapter disabled, is the monitor customer) and the out-of-family external customer."""

    def __init__(self, name: str, cfg: Config, logger: logging.Logger, lora: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cfg, self.logger, self.device = cfg, logger, cfg.device
        path = resolve_local_model(name, cfg.models_dir)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        cuda = self.device.startswith("cuda")
        quant = cfg.load_4bit and cuda
        kw: Dict[str, Any] = {"dtype": torch.bfloat16 if cuda else torch.float32}
        if quant:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16,
                                                           bnb_4bit_use_double_quant=True)
            kw["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(path, **kw)
        if not quant:
            model = model.to(self.device)
        self.lora = lora
        if lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if quant:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                        gradient_checkpointing_kwargs={"use_reentrant": False})
            elif cuda:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.enable_input_require_grads()
            # LoRA dropout 0: training-mode and eval-mode forwards are the same function, so the DPO policy and
            # its frozen reference are scored by the same network.
            model = get_peft_model(model, LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        self.model = model
        eos = {self.tok.eos_token_id}
        g = getattr(model.generation_config, "eos_token_id", None)
        eos |= set(g) if isinstance(g, (list, tuple)) else ({g} if g is not None else set())
        self.eos_ids = sorted(e for e in eos if e is not None)
        self._prefix_cache: Dict[str, List[int]] = {}
        self._tail_kw: Optional[str] = None
        logger.info("model %s | LoRA=%s | 4bit=%s | eos ids %s", name, lora, quant, self.eos_ids)

    # ---- adapters ------------------------------------------------------------------------------------
    def save_adapter(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))

    def load_adapter(self, path: Path) -> None:
        """Write saved LoRA tensors into the live adapter and verify that every tensor matched."""
        from peft import set_peft_model_state_dict
        path = Path(path)
        st = path / "adapter_model.safetensors"
        if st.exists():
            from safetensors.torch import load_file
            sd = load_file(str(st))
        else:
            import torch
            sd = torch.load(str(path / "adapter_model.bin"), map_location="cpu")
        res = set_peft_model_state_dict(self.model, sd)
        missing = [k for k in getattr(res, "missing_keys", []) if "lora_" in k]
        if missing or getattr(res, "unexpected_keys", []):
            raise ValueError(f"adapter {path} does not match the model ({len(missing)} LoRA tensors missing)")
        self.logger.info("loaded adapter %s", path)

    def snapshot(self) -> Dict[str, Any]:
        return {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def restore(self, snap: Dict[str, Any]) -> None:
        import torch
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in snap:
                    p.copy_(snap[n])

    @contextlib.contextmanager
    def frozen(self, snap: Dict[str, Any]):
        """Temporarily swap the trainable tensors for a snapshot (reference-policy scoring)."""
        keep = {}
        for n, p in self.model.named_parameters():
            if n in snap:
                keep[n], p.data = p.data, snap[n]
        try:
            yield
        finally:
            for n, p in self.model.named_parameters():
                if n in keep:
                    p.data = keep[n]

    def _adapter_ctx(self, adapter: bool):
        return self.model.disable_adapter() if (self.lora and not adapter) else contextlib.nullcontext()

    # ---- tokenisation --------------------------------------------------------------------------------
    def prefix_ids(self, text: str) -> List[int]:
        ids = self._prefix_cache.get(text)
        if ids is None:
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            self._prefix_cache[text] = ids
        return ids

    def text_ids(self, text: str) -> List[int]:
        """Token ids of a reply as a continuation: " " + text + EOS (the SFT target format)."""
        return self.tok(" " + norm_text(text), add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]

    # ---- generation ----------------------------------------------------------------------------------
    def generate_ids(self, prompts: Sequence[str], gen: GenConfig, seed: int, adapter: bool = True) -> List[List[int]]:
        """Sampled continuation ids, cut after the first EOS (kept, because it was sampled)."""
        import torch
        self.model.eval()
        out: List[List[int]] = []
        with self._adapter_ctx(adapter):
            for s in range(0, len(prompts), self.cfg.gen_batch):
                ids = [self.prefix_ids(p)[-self.cfg.max_len:] for p in prompts[s:s + self.cfg.gen_batch]]
                n = max(len(x) for x in ids)
                X = torch.full((len(ids), n), self.tok.pad_token_id, dtype=torch.long)
                M = torch.zeros((len(ids), n), dtype=torch.long)
                for i, x in enumerate(ids):
                    X[i, n - len(x):] = torch.tensor(x)
                    M[i, n - len(x):] = 1
                torch.manual_seed(seed + s)
                with torch.no_grad():
                    y = self.model.generate(input_ids=X.to(self.device), attention_mask=M.to(self.device),
                                            do_sample=True, temperature=gen.temperature, top_p=gen.top_p,
                                            top_k=gen.top_k, repetition_penalty=gen.repetition_penalty,
                                            max_new_tokens=gen.max_new_tokens, min_new_tokens=gen.min_new_tokens,
                                            pad_token_id=self.tok.pad_token_id, eos_token_id=self.eos_ids,
                                            use_cache=True)
                for row in y[:, n:].cpu().tolist():
                    cut = next((j for j, tk in enumerate(row) if tk in self.eos_ids), None)
                    out.append(row[:cut + 1] if cut is not None else row)
        return out

    def decode(self, ids: Sequence[int]) -> str:
        return norm_text(self.tok.decode(list(ids), skip_special_tokens=True))

    def generate_text(self, prompts: Sequence[str], gen: GenConfig, seed: int, adapter: bool = True) -> List[str]:
        return [trim_to_sentence(self.decode(x)) for x in self.generate_ids(prompts, gen, seed, adapter)]

    # ---- scoring -------------------------------------------------------------------------------------
    def _forward_tail(self, X, M, pos, keep: int):
        """Forward pass that materialises logits only for the last `keep` positions."""
        if self._tail_kw is None:
            for kw in ("logits_to_keep", "num_logits_to_keep", ""):
                try:
                    extra = {kw: keep} if kw else {}
                    lg = self.model(input_ids=X, attention_mask=M, position_ids=pos, use_cache=False, **extra).logits
                    if lg.shape[1] < keep:
                        raise TypeError("short logits")
                    self._tail_kw = kw
                    return lg[:, -keep:]
                except TypeError:
                    continue
            raise RuntimeError("model forward rejected every logits-slicing convention")
        extra = {self._tail_kw: keep} if self._tail_kw else {}
        return self.model(input_ids=X, attention_mask=M, position_ids=pos, use_cache=False, **extra).logits[:, -keep:]

    def _score(self, prefixes: Sequence[str], conts: Sequence[Sequence[int]]):
        """Summed log p(continuation | prefix) and continuation token counts, as tensors (graph kept).
        The prefix is truncated exactly as in generation (its last max_len tokens) and the continuation is kept
        whole, so a sampled reply is scored under the very context it was sampled from.  Left padding puts
        every continuation at the end of its row, so only the tail logits are needed."""
        import torch
        import torch.nn.functional as F
        seqs, nts = [], []
        for p, c in zip(prefixes, conts):
            seqs.append(self.prefix_ids(p)[-self.cfg.max_len:] + list(c))
            nts.append(len(c))
        n, T = max(len(s) for s in seqs), max(1, max(nts))
        X = torch.full((len(seqs), n), self.tok.pad_token_id, dtype=torch.long)
        M = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            X[i, n - len(s):] = torch.tensor(s)
            M[i, n - len(s):] = 1
        pos = (M.cumsum(1) - 1).clamp_min(0)
        X, M, pos = X.to(self.device), M.to(self.device), pos.to(self.device)
        lg = self._forward_tail(X, M, pos, T + 1)[:, :-1]
        tgt = X[:, n - T:]
        lp = -F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape)
        nt = torch.tensor(nts, device=lp.device)
        sel = (torch.arange(T, device=lp.device)[None, :] >= (T - nt)[:, None]).float()
        return (lp * sel).sum(1), nt.float()

    def score(self, prefixes: Sequence[str], conts: Sequence[Sequence[int]], adapter: bool = True) -> np.ndarray:
        """No-grad summed log-probabilities, length-bucketed for throughput."""
        import torch
        self.model.eval()
        out = np.zeros(len(prefixes))
        order = sorted(range(len(prefixes)), key=lambda i: len(self.prefix_ids(prefixes[i])) + len(conts[i]))
        b = self.cfg.score_batch
        with torch.no_grad(), self._adapter_ctx(adapter):
            for s in range(0, len(order), b):
                ix = order[s:s + b]
                lp, _ = self._score([prefixes[i] for i in ix], [conts[i] for i in ix])
                out[ix] = lp.float().cpu().numpy()
        return out

    # ---- training ------------------------------------------------------------------------------------
    def fit_sft(self, train: Sequence[Tuple[str, str]], dev: Sequence[Tuple[str, str]], sc: SFTConfig,
                best_dir: Path, tag: str) -> Dict[str, Any]:
        """Token-level NLL on (prompt, target) pairs, cosine schedule, early stopping on dev NLL."""
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LambdaLR
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = AdamW(params, lr=sc.lr, weight_decay=sc.weight_decay)
        steps_per_epoch = max(1, len(train) // (sc.batch_size * sc.grad_accum))
        total, warm = steps_per_epoch * sc.epochs, max(1, int(sc.warmup_frac * steps_per_epoch * sc.epochs))
        sched = LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                         max(0.05, 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))))
        every = max(1, steps_per_epoch // max(1, sc.evals_per_epoch))
        dev_ids = [(p, self.text_ids(t)) for p, t in dev]

        def dev_nll() -> float:
            lp = self.score([p for p, _ in dev_ids], [c for _, c in dev_ids])
            return float(-lp.sum() / max(1, sum(len(c) for _, c in dev_ids)))

        rng = random.Random(1234)
        best, best_step, stale, step, micro = float("inf"), -1, 0, 0, 0
        for _ in range(sc.epochs):
            order = list(range(len(train)))
            rng.shuffle(order)
            self.model.train()
            for s in range(0, len(order) - sc.batch_size + 1, sc.batch_size):
                batch = [train[i] for i in order[s:s + sc.batch_size]]
                lp, nt = self._score([p for p, _ in batch], [self.text_ids(t) for _, t in batch])
                (-lp.sum() / nt.sum().clamp_min(1.0) / sc.grad_accum).backward()
                micro += 1
                if micro % sc.grad_accum:
                    continue
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % every == 0 or step == total:
                    d = dev_nll()
                    if d < best - 1e-4:
                        best, best_step, stale = d, step, 0
                        self.save_adapter(best_dir)
                    else:
                        stale += 1
                    self.logger.info("%s | step %d/%d | dev NLL %.4f (best %.4f) | stale %d/%d", tag, step, total, d,
                                     best, stale, sc.patience)
                    self.model.train()
                    if stale >= sc.patience:
                        break
            if stale >= sc.patience:
                break
        if best_dir.exists():
            self.load_adapter(best_dir)
        return {"best_dev_nll": best, "best_step": best_step, "steps": step}

    def dpo_update(self, pairs: Sequence[Tuple[str, str, str]], weights: Sequence[float], ref: Dict[str, Any],
                   beta: float, lr: float, epochs: int, batch: int, nll_coef: float, seed: int) -> Dict[str, Any]:
        """Weighted DPO against the frozen reference (summed log-probabilities; Rafailov et al., 2023) plus a
        length-normalised NLL anchor on the chosen reply.  Returns the change of the mean chosen and rejected
        log-likelihoods over the update: a negative chosen change is likelihood displacement."""
        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW
        prompts = [p for p, _, _ in pairs]
        ch = [self.text_ids(c) for _, c, _ in pairs]
        rj = [self.text_ids(r) for _, _, r in pairs]
        with self.frozen(ref):
            ref_c, ref_r = self.score(prompts, ch), self.score(prompts, rj)
        pre_c, pre_r = self.score(prompts, ch), self.score(prompts, rj)
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = AdamW(params, lr=lr, weight_decay=0.0)
        rng = random.Random(seed)
        idx = list(range(len(pairs)))
        losses, accs = [], []
        self.model.train()
        for _ in range(max(1, epochs)):
            rng.shuffle(idx)
            for s in range(0, len(idx), batch):
                b = idx[s:s + batch]
                lc, nc = self._score([prompts[i] for i in b], [ch[i] for i in b])
                lr_, _ = self._score([prompts[i] for i in b], [rj[i] for i in b])
                rc = torch.tensor(ref_c[b], device=lc.device, dtype=lc.dtype)
                rr = torch.tensor(ref_r[b], device=lc.device, dtype=lc.dtype)
                w = torch.tensor([float(weights[i]) for i in b], device=lc.device)
                m = (lc - rc) - (lr_ - rr)
                loss = (w * -F.logsigmoid(beta * m)).sum() / w.sum().clamp_min(1e-6)
                loss = loss + nll_coef * (-(lc / nc.clamp_min(1.0))).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                losses.append(float(loss.detach()))
                accs.append(float((m.detach() > 0).float().mean()))
        opt.zero_grad(set_to_none=True)
        post_c, post_r = self.score(prompts, ch), self.score(prompts, rj)
        return {"loss": float(np.mean(losses)), "pref_acc": float(np.mean(accs)), "steps": len(losses),
                "d_chosen_logp": float(np.mean(post_c - pre_c)), "d_rejected_logp": float(np.mean(post_r - pre_r))}


class ValenceClassifier:
    """Valence in [-1, 1] from a sentiment or emotion classifier: P(positive labels) - P(negative labels).
    Labels are mapped by name, so sentiment (positive/negative), GoEmotions and Ekman heads all work."""
    POS = frozenset("positive joy love optimism admiration gratitude approval relief amusement excitement caring "
                    "pride satisfied happiness".split())
    NEG = frozenset("negative anger disgust fear sadness annoyance disappointment disapproval embarrassment grief "
                    "nervousness remorse dissatisfied".split())

    def __init__(self, name: str, cfg: Config, logger: logging.Logger, batch: int = 64, max_len: int = 128):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = resolve_local_model(name, cfg.models_dir)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(cfg.device).eval()
        self.device, self.batch, self.max_len = cfg.device, batch, max_len
        lab = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.sign = np.asarray([1.0 if lab[i] in self.POS else (-1.0 if lab[i] in self.NEG else 0.0)
                                for i in range(len(lab))])
        if not (self.sign > 0).any() or not (self.sign < 0).any():
            raise ValueError(f"{name}: no recognisable positive/negative labels in {lab}")
        self.multilabel = getattr(self.model.config, "problem_type", "") == "multi_label_classification"
        logger.info("valence classifier %s | %d labels | multilabel=%s", name, len(lab), self.multilabel)

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        texts = [norm_text(t) or "." for t in texts]
        out = np.zeros(len(texts))
        with torch.no_grad():
            for s in range(0, len(texts), self.batch):
                enc = self.tok(texts[s:s + self.batch], return_tensors="pt", padding=True, truncation=True,
                               max_length=self.max_len)
                lg = self.model(input_ids=enc["input_ids"].to(self.device),
                                attention_mask=enc["attention_mask"].to(self.device)).logits.float()
                p = (torch.sigmoid(lg) if self.multilabel else torch.softmax(lg, -1)).cpu().numpy()
                v = p @ self.sign
                out[s:s + len(v)] = v / np.maximum(p.sum(1), 1.0) if self.multilabel else v
        return out


class NLIScorer:
    """P(entailment) and P(contradiction) of (premise, hypothesis) pairs from an NLI cross-encoder."""

    def __init__(self, name: str, cfg: Config, logger: logging.Logger, batch: int = 64):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = resolve_local_model(name, cfg.models_dir)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(cfg.device).eval()
        self.device, self.batch = cfg.device, batch
        lab = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.ent = next(i for i, v in lab.items() if v.startswith("entail"))
        self.con = next(i for i, v in lab.items() if v.startswith("contra"))
        logger.info("NLI %s | labels %s", name, lab)

    def __call__(self, premises: Sequence[str], hypotheses: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        import torch
        pe, pc = np.zeros(len(premises)), np.zeros(len(premises))
        with torch.no_grad():
            for s in range(0, len(premises), self.batch):
                enc = self.tok([norm_text(x) or "." for x in premises[s:s + self.batch]],
                               [norm_text(x) or "." for x in hypotheses[s:s + self.batch]],
                               return_tensors="pt", padding=True, truncation=True, max_length=320)
                p = torch.softmax(self.model(input_ids=enc["input_ids"].to(self.device),
                                             attention_mask=enc["attention_mask"].to(self.device)).logits.float(),
                                  -1).cpu().numpy()
                pe[s:s + len(p)], pc[s:s + len(p)] = p[:, self.ent], p[:, self.con]
        return pe, pc


def chat_customer_prompt(model: Policy, t: Turn, response: str) -> str:
    """Customer prompt for an instruct model used zero-shot (monitor and external customers)."""
    conv = f"{t.history}\nCustomer: {t.user_text}\nAgent: {norm_text(response)}".strip()
    if not getattr(model.tok, "chat_template", None):
        return f"{CUSTOMER_SYSTEM}\n{conv}\nCustomer:"
    return model.tok.apply_chat_template(
        [{"role": "system", "content": CUSTOMER_SYSTEM},
         {"role": "user", "content": f"Conversation so far:\n{conv}\n\nWrite only the customer's next turn, in one "
                                     f"or two sentences."}], tokenize=False, add_generation_prompt=True)


def simulator_prompt(model: Policy, t: Turn, response: str) -> str:
    return customer_prompt(t, response)


# ==========================================================================================================
# the judge: shared-pool multiple importance sampling over a customer model
# ==========================================================================================================

def snis_weights(log_p: np.ndarray, log_q: np.ndarray) -> Tuple[np.ndarray, float]:
    """Self-normalised importance weights p/q, truncated at mean(w) sqrt(R) (Ionides, 2008); returns (w, ESS)."""
    lw = np.asarray(log_p, float) - np.asarray(log_q, float)
    w = np.exp(lw - np.max(lw))
    w = np.minimum(w, float(np.mean(w)) * math.sqrt(w.size))
    w = w / max(float(w.sum()), EPS)
    return w, float(1.0 / max(float(np.sum(w ** 2)), EPS))


def balance_log_q(L: np.ndarray) -> np.ndarray:
    """log q(c_j) = log mean_m p(c_j | x, r_m) for a (proposals x replies) matrix of log-probabilities."""
    mx = L.max(0)
    return mx + np.log(np.mean(np.exp(L - mx), 0))


class Judge:
    """E[label(next customer turn) | context, reply] on per-context reply pools.

    A pool for (namespace, context) is drawn once per Judge: `pool_size` replies split evenly over the proposal
    agent turns, sampled at temperature 1 with every truncation disabled, so that the sampling density is the
    model density; the sampled reply token ids are scored exactly.  Each labeller reads every pool reply once.
    Pools are never re-used across runs (a pool built for other proposals would extrapolate); the store file
    is an audit record of the pools of this run.  `adapter=False` turns the simulator into the un-fine-tuned
    base customer (the monitor)."""

    POOL_GEN = GenConfig(min_new_tokens=0, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0)

    def __init__(self, model: Policy, labellers: Dict[str, Callable[[Sequence[str]], np.ndarray]],
                 prompt_fn: Callable[[Policy, Turn, str], str], store: Path, pool_size: int, adapter: bool,
                 max_new_tokens: int, logger: logging.Logger, name: str):
        self.model, self.labellers, self.prompt_fn = model, labellers, prompt_fn
        self.store, self.pool_size, self.adapter, self.logger, self.name = Path(store), pool_size, adapter, logger, name
        self.gen = dataclasses.replace(self.POOL_GEN, max_new_tokens=max_new_tokens)
        self.pools: Dict[str, Dict[str, Any]] = {}
        self._lp: Dict[Tuple[str, Tuple[int, ...]], float] = {}
        if self.store.exists():
            self.store.unlink()

    def _logp(self, prefixes: Sequence[str], conts: Sequence[Sequence[int]]) -> np.ndarray:
        keys = [(hashlib.sha1(p.encode()).hexdigest(), tuple(c)) for p, c in zip(prefixes, conts)]
        todo = {k: (p, c) for k, p, c in zip(keys, prefixes, conts) if k not in self._lp}
        if todo:
            vals = self.model.score([p for p, _ in todo.values()], [c for _, c in todo.values()], adapter=self.adapter)
            self._lp.update(zip(todo.keys(), vals.tolist()))
        return np.asarray([self._lp[k] for k in keys])

    def build(self, ns: str, turns: Sequence[Turn], proposals: Dict[str, List[str]]) -> int:
        """Create the missing pools of namespace `ns`.  proposals[uid] lists the proposal agent turns; the gold
        turn is always added.  Returns the number of pools created."""
        need = []
        for t in {t.uid: t for t in turns}.values():
            key = f"{ns}|{t.uid}"
            if key not in self.pools:
                props = list(dict.fromkeys([norm_text(t.gold_response)] + [norm_text(p) for p in proposals.get(t.uid, [])]))
                need.append((key, t, props))
        if not need:
            return 0
        prompts, owner = [], []
        for ci, (_, t, props) in enumerate(need):
            R = max(self.pool_size, len(props))
            for m, pr in enumerate(props):
                for _ in range(R // len(props) + (1 if m < R % len(props) else 0)):
                    prompts.append(self.prompt_fn(self.model, t, pr))
                    owner.append(ci)
        ids = self.model.generate_ids(prompts, self.gen, seed=stable_hash(ns, need[0][0]), adapter=self.adapter)
        ids = [x if x else [self.model.tok.eos_token_id] for x in ids]
        texts = [self.model.decode(x) or "." for x in ids]
        labels = {k: np.asarray(fn(texts), float) for k, fn in self.labellers.items()}
        owner_a = np.asarray(owner)
        entries = []
        for ci, (key, t, props) in enumerate(need):
            ix = np.flatnonzero(owner_a == ci)
            rep = [ids[i] for i in ix]
            L = np.stack([self._logp([self.prompt_fn(self.model, t, pr)] * len(rep), rep) for pr in props], 0)
            entries.append({"key": key, "uid": t.uid, "proposals": props, "reply_ids": rep,
                            "replies": [texts[i] for i in ix], "log_q": balance_log_q(L).tolist(),
                            "labels": {k: v[ix].tolist() for k, v in labels.items()}})
        for e in entries:
            self.pools[e["key"]] = e
        self.store.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store, "a", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")
        return len(entries)

    def expect(self, ns: str, turns: Sequence[Turn], responses: Sequence[str],
               proposals: Optional[Dict[str, List[str]]] = None) -> Dict[str, np.ndarray]:
        """SNIS expectations of every labeller for each (turn, response); builds missing pools first (with the
        given proposals, or with the responses of this call as proposals)."""
        turns, responses = list(turns), [norm_text(r) for r in responses]
        if proposals is None:
            proposals = {}
            for t, r in zip(turns, responses):
                proposals.setdefault(t.uid, []).append(r)
        self.build(ns, turns, proposals)
        pre, conts, row = [], [], []
        pools = [self.pools[f"{ns}|{t.uid}"] for t in turns]
        for i, (t, r, pl) in enumerate(zip(turns, responses, pools)):
            pre.extend([self.prompt_fn(self.model, t, r)] * len(pl["reply_ids"]))
            conts.extend(pl["reply_ids"])
            row.extend([i] * len(pl["reply_ids"]))
        lp, row = self._logp(pre, conts), np.asarray(row)
        out = {k: np.zeros(len(turns)) for k in self.labellers}
        ess = np.zeros(len(turns))
        for i, pl in enumerate(pools):
            w, ess[i] = snis_weights(lp[row == i], np.asarray(pl["log_q"]))
            for k in self.labellers:
                out[k][i] = float(w @ np.asarray(pl["labels"][k]))
        out["ess"] = ess
        return out


# ==========================================================================================================
# experiment scaffolding
# ==========================================================================================================

class Experiment:
    """Holds the configuration, the logger, the turns and lazily loaded auxiliary models of one stage."""

    def __init__(self, cfg: Config, tag: str):
        self.cfg = cfg
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = make_logger(cfg.out_dir, tag)
        seed_everything(cfg.seed)
        self.logger.info("CASPI %s | stage %s | device %s | out %s", VERSION, tag, cfg.device, cfg.out)
        cache = cfg.out_dir / "turns.json"
        if cache.exists():
            self.turns = [Turn(**d) for d in load_json(cache)]
        else:
            if cfg.download:
                download_emowoz(Path(cfg.data_dir), self.logger)
            self.turns = load_emowoz(Path(cfg.data_dir), self.logger)
            dump_json([asdict(t) for t in self.turns], cache)
        self._cache: Dict[str, Any] = {}

    def _lazy(self, key: str, make: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    @property
    def sentiment(self) -> ValenceClassifier:
        return self._lazy("sent", lambda: ValenceClassifier(self.cfg.sentiment_model, self.cfg, self.logger))

    @property
    def monitor_labeller(self) -> ValenceClassifier:
        return self._lazy("mon", lambda: ValenceClassifier(self.cfg.monitor_model, self.cfg, self.logger))

    @property
    def nli(self) -> NLIScorer:
        return self._lazy("nli", lambda: NLIScorer(self.cfg.nli_model, self.cfg, self.logger))

    def policy(self, adapter: Optional[str]) -> Policy:
        p = Policy(self.cfg.base_model, self.cfg, self.logger)
        if adapter:
            p.load_adapter(self.cfg.out_dir / adapter)
        return p

    def judges(self, sim: Policy, store_tag: str, pool_size: int) -> Tuple[Judge, Judge]:
        """The training judge (fine-tuned customer + sentiment engine) and the monitor (the same weights with
        the adapter disabled, i.e. the base instruct model as customer, read by an independent labeller)."""
        out = self.cfg.out_dir / "pools"
        j = Judge(sim, {"affect": self.sentiment}, simulator_prompt, out / f"{store_tag}_sim.jsonl", pool_size,
                  True, self.cfg.sim_max_new_tokens, self.logger, "simulator")
        m = Judge(sim, {"monitor": self.monitor_labeller}, chat_customer_prompt, out / f"{store_tag}_monitor.jsonl",
                  pool_size, False, self.cfg.sim_max_new_tokens, self.logger, "monitor")
        return j, m


def reply_metrics(texts: Sequence[str], turns: Sequence[Turn]) -> Dict[str, np.ndarray]:
    """Task information, hygiene and length of agent replies (judge-free)."""
    return {"content": np.asarray([content_score(r, t.gold_response) for r, t in zip(texts, turns)]),
            "info_recall": np.asarray([info_recall(r, t.gold_response) for r, t in zip(texts, turns)]),
            "slot_f1": np.asarray([slot_prf(r, t.gold_response)[2] for r, t in zip(texts, turns)]),
            "hygiene": np.asarray([float(hygiene_ok(r)) for r in texts]),
            "words": np.asarray([float(n_words(r)) for r in texts])}


# ==========================================================================================================
# stages 1-3: SFT, simulator, judge validity
# ==========================================================================================================

def stage_data(cfg: Config) -> Dict[str, Any]:
    ex = Experiment(cfg, "0_data")
    per: Dict[str, int] = {}
    for t in ex.turns:
        per[t.split] = per.get(t.split, 0) + 1
    ex.logger.info("turns per split %s", per)
    return per


def stage_sft(cfg: Config) -> Dict[str, Any]:
    ex = Experiment(cfg, "1_sft")
    tr = select_turns(ex.turns, "train", cfg.sft.max_train, cfg.seed)
    dv = select_turns(ex.turns, "valid", cfg.sft.dev_examples, cfg.seed)
    pol = ex.policy(None)
    res = pol.fit_sft([(agent_prompt(t), t.gold_response) for t in tr],
                      [(agent_prompt(t), t.gold_response) for t in dv], cfg.sft, cfg.out_dir / "sft_best", "SFT")
    pol.save_adapter(cfg.out_dir / "sft_policy")
    dump_json(res, cfg.out_dir / "sft.json")
    return res


def stage_simulator(cfg: Config) -> Dict[str, Any]:
    ex = Experiment(cfg, "2_simulator")
    tr = select_turns(ex.turns, "train", cfg.sim_turns, cfg.seed, require_next=True)
    dv = select_turns(ex.turns, "valid", cfg.sim_sft.dev_examples, cfg.seed, require_next=True)
    pol = ex.policy(None)
    res = pol.fit_sft([(customer_prompt(t, t.gold_response), t.next_user_text) for t in tr],
                      [(customer_prompt(t, t.gold_response), t.next_user_text) for t in dv], cfg.sim_sft,
                      cfg.out_dir / "sim_best", "SIM")
    pol.save_adapter(cfg.out_dir / "simulator")
    dump_json(res, cfg.out_dir / "simulator.json")
    return res


def stage_validate(cfg: Config) -> Dict[str, Any]:
    """Construct validity and noise of the judge, on labelled VALID contexts (no policy is involved).

    * between-context anchor: judge(gold reply) vs the real customer's next-turn emotion label, and the
      PARTIAL anchor given the current turn's label (emotional carry-over removed);
    * replication: the same replies on an independent pool (ICC of the context-level estimates);
    * within-context noise: sigma of the gold-vs-SFT affect difference across the two pools versus the
      spread of that difference (the signal a preference has to resolve), and the flip rate."""
    ex = Experiment(cfg, "3_validate")
    turns = [t for t in select_turns(ex.turns, "valid", cfg.validate_contexts, cfg.seed + 3, require_next=True,
                                     one_per_dialogue=True) if t.human_valence is not None]
    pol = ex.policy("sft_policy")
    sft = pol.generate_text([agent_prompt(t) for t in turns], GenConfig(), seed=cfg.seed)
    del pol
    release_gpu()
    sim = ex.policy("simulator")
    judge, _ = ex.judges(sim, "validate", cfg.caspi.pool_size)
    prop = {t.uid: [t.gold_response, r] for t, r in zip(turns, sft)}
    both_t, both_r = turns + turns, [t.gold_response for t in turns] + sft
    e0 = judge.expect("v0", both_t, both_r, prop)["affect"]
    e1 = judge.expect("v1", both_t, both_r, prop)["affect"]
    n = len(turns)
    hv = np.asarray([t.human_valence for t in turns], float)
    cv = np.asarray([t.current_valence if t.current_valence is not None else np.nan for t in turns], float)
    cl = np.asarray([t.dialogue_id for t in turns], dtype=object)
    gold0 = e0[:n]
    d0, d1 = e0[:n] - e0[n:], e1[:n] - e1[n:]
    sigma = float(np.std(d0 - d1, ddof=1) / math.sqrt(2.0))
    res = {"n": n,
           "anchor": cluster_bootstrap_ci(lambda ix: spearman(gold0[ix], hv[ix]), cl, cfg.n_boot, cfg.seed),
           "partial_anchor": cluster_bootstrap_ci(lambda ix: partial_spearman(gold0[ix], hv[ix], cv[ix]), cl,
                                                  cfg.n_boot, cfg.seed + 1),
           "pool_replication_pearson": pearson(e0, e1),
           "within_context_noise_sd": sigma,
           "within_context_signal_sd": float(np.std(0.5 * (d0 + d1), ddof=1)),
           "replication_flip_rate": float(np.mean(np.sign(d0) != np.sign(d1)))}
    ex.logger.info("judge validity | anchor rho %+.3f CI[%+.3f,%+.3f] | partial (given current emotion) %+.3f "
                   "CI[%+.3f,%+.3f] | pool replication r %.3f | within-context noise sd %.4f vs signal sd %.4f | "
                   "flip rate %.3f", *res["anchor"], *res["partial_anchor"], res["pool_replication_pearson"], sigma,
                   res["within_context_signal_sd"], res["replication_flip_rate"])
    if not res["partial_anchor"][1] > 0:
        ex.logger.warning("the partial anchor does not exclude zero: the judge's response-level human signal is "
                          "unestablished; only the human study can validate within-context rankings")
    dump_json(res, cfg.out_dir / "judge_validity.json")
    return res


# ==========================================================================================================
# CASPI core
# ==========================================================================================================

def mine_pairs(A: np.ndarray, C: np.ndarray, H: np.ndarray, ess: np.ndarray, texts: Sequence[str], sigma: float,
               pc: CASPIConfig, pareto: bool) -> List[Tuple[int, int, float, str]]:
    """Preference pairs (winner, loser, weight, kind) for ONE context; index 0 is the reference reply.

    pareto=True: i beats j only if i is feasible (H), A_i - A_j >= z sigma, C_i >= C_j - eps (information;
    vacuous only when the gold turn carries no information), both estimates rest on >= min_pair_ess effective
    replies, and the replies are not near-duplicates.  Kept: the largest-margin pair, the largest-margin pair
    against the reference, and one constraint pair (feasible beats infeasible, without information loss).
    Weights grow with the margin in noise units, from 0.5 to 1.
    pareto=False (baselines): best vs worst by A, weight 1."""
    A, C, H, ess = (np.asarray(v) for v in (A, C, H, ess))
    n = A.size
    if n < 2:
        return []
    if not pareto:
        i, j = int(np.argmax(A)), int(np.argmin(A))
        return [(i, j, 1.0, "affect")] if A[i] > A[j] and word_jaccard(texts[i], texts[j]) < pc.dup_jaccard else []
    sig = max(float(sigma), 1e-6)

    def info_ok(i, j):
        return not (math.isfinite(C[i]) and math.isfinite(C[j])) or C[i] >= C[j] - pc.eps_content

    def usable(i, j):
        return ess[i] >= pc.min_pair_ess and ess[j] >= pc.min_pair_ess and \
            word_jaccard(texts[i], texts[j]) < pc.dup_jaccard

    def weight(d):
        return float(min(1.0, max(0.5, d / (2.0 * pc.z * sig))))

    dom = [(float(A[i] - A[j]), i, j) for i in range(n) if H[i] for j in range(n)
           if j != i and A[i] - A[j] >= pc.z * sig and info_ok(i, j) and usable(i, j)]
    out: List[Tuple[int, int, float, str]] = []
    if dom:
        d, i, j = max(dom)
        out.append((i, j, weight(d), "dominance"))
        anchor = [x for x in dom if x[2] == 0 and (x[1], x[2]) != (i, j)]
        if anchor:
            d2, i2, j2 = max(anchor)
            out.append((i2, j2, weight(d2), "beats_reference"))
    feas, infeas = np.flatnonzero(H), np.flatnonzero(~H)
    cons = [(float(A[i]), int(i), int(j)) for j in infeas for i in feas if info_ok(i, j) and usable(i, j)]
    if cons:
        _, i, j = max(cons)
        out.append((i, j, 0.5, "constraint"))
    return out[:pc.max_pairs]


def vector_test(diffs: Dict[str, np.ndarray], cluster: Sequence[Any], alpha: float, pc: CASPIConfig,
                best_gain: float, ess_median: float, pool_size: int) -> Dict[str, Any]:
    """Intersection-union test of improvement over SFT (see the module docstring for the guarantee).

    diffs: per-context paired differences new - SFT for affect, monitor, content, hygiene, and the log word
    ratio for length.  Every component test is one-sided at level alpha; the policy passes only if all pass.
    The power column is the probability of passing that non-inferiority test when the true difference is 0,
    given the observed spread (normal approximation); a low value means the margin is below what n can
    resolve, so a good policy will probably be rejected.  dialogues_for_80pct_power is the safety-split size
    that the observed spread would need; use it to plan the next pre-registration, never to change this one."""
    m = pc.bound
    lb = {k: confidence_bound(diffs[k], cluster, alpha, "lower", m, r) for k, r in
          (("affect", 4.0), ("monitor", 4.0), ("content", 2.0), ("hygiene", 2.0), ("length", 2.0))}
    ub_len = confidence_bound(diffs["length"], cluster, alpha, "upper", m, 2.0)
    mean = {k: float(np.nanmean(v)) if np.isfinite(v).any() else float("nan") for k, v in diffs.items()}
    margins = {"monitor": pc.margin_monitor, "content": pc.margin_info, "hygiene": pc.margin_hygiene,
               "length": pc.length_equiv}
    checks = {"affect_superior": bool(math.isfinite(lb["affect"]) and lb["affect"] > 0),
              "beats_best_round": bool(mean["affect"] > best_gain),
              "monitor_noninferior": bool(math.isfinite(lb["monitor"]) and lb["monitor"] > -pc.margin_monitor),
              # content is NaN for contexts whose gold turn carries no task information; the test uses the rest
              "content_noninferior": bool(math.isfinite(lb["content"]) and lb["content"] > -pc.margin_info),
              "hygiene_noninferior": bool(math.isfinite(lb["hygiene"]) and lb["hygiene"] > -pc.margin_hygiene),
              "length_equivalent": bool(math.isfinite(lb["length"]) and lb["length"] > -pc.length_equiv
                                        and ub_len < pc.length_equiv),
              "ess_floor": bool(ess_median >= pc.min_ess_frac * pool_size)}
    za, zb = normal_ppf(1 - alpha), normal_ppf(0.8)
    power, n80 = {}, {}
    for k, mg in margins.items():
        u = cluster_means(diffs[k], cluster)
        sd = float(np.std(u, ddof=1)) if u.size > 1 else float("nan")
        se = sd / math.sqrt(max(1, u.size))
        ok = math.isfinite(se) and se > 0
        power[k] = normal_cdf(mg / se - za) if ok else float("nan")
        # dialogues needed for 80% power at zero true difference (sample-size planning for the next run)
        n80[k] = int(math.ceil(((za + zb) * sd / mg) ** 2)) if ok else None
    return {"accept": all(checks.values()), "checks": checks, "lower_bounds": lb, "length_upper": ub_len,
            "means": mean, "margins": margins, "power_at_zero_difference": power, "dialogues_for_80pct_power": n80,
            "content_vacuous_share": float(np.mean(~np.isfinite(diffs["content"]))), "ess_median": ess_median,
            "alpha": alpha, "bound": m}


def paired_dev_check(pol: Policy, ref: Dict[str, Any], judge: Judge, monitor: Judge, turns: Sequence[Turn],
                     ns: str, seed: int) -> Tuple[Dict[str, np.ndarray], float, Dict[str, float]]:
    """SFT and the current policy answer the same contexts (same generation seed: common random numbers).
    Both replies and the gold turn are proposals of a FRESH pool, so neither estimate extrapolates beyond its
    proposals (the dev-pool bias of the predecessor).  Returns paired differences, the median ESS and the
    absolute means."""
    gen = GenConfig()
    prompts = [agent_prompt(t) for t in turns]
    with pol.frozen(ref):
        y0 = pol.generate_text(prompts, gen, seed)
    y1 = pol.generate_text(prompts, gen, seed)
    props = {t.uid: [a, b] for t, a, b in zip(turns, y0, y1)}
    tt, yy, n = list(turns) + list(turns), y0 + y1, len(turns)
    ja = judge.expect(ns, tt, yy, props)
    jm = monitor.expect(ns, tt, yy, props)
    r0, r1 = reply_metrics(y0, turns), reply_metrics(y1, turns)
    diffs = {"affect": ja["affect"][n:] - ja["affect"][:n], "monitor": jm["monitor"][n:] - jm["monitor"][:n],
             "content": r1["content"] - r0["content"], "hygiene": r1["hygiene"] - r0["hygiene"],
             "length": log_word_ratio(y1, y0)}
    ess = float(np.median(np.concatenate([ja["ess"], jm["ess"]])))
    level = {"affect_sft": float(np.mean(ja["affect"][:n])), "monitor_sft": float(np.mean(jm["monitor"][:n])),
             "words_new": float(np.mean(r1["words"])), "words_sft": float(np.mean(r0["words"]))}
    return diffs, ess, level


def collect_candidates(pol: Policy, ref: Dict[str, Any], ctx: Sequence[Turn], pc: CASPIConfig, flags: Dict[str, Any],
                       ex: Experiment, seed: int) -> Tuple[List[List[str]], Dict[str, int]]:
    """Per context: [reference reply, kept rewrites, on-policy samples] (index 0 is the reference).  A rewrite
    is kept only if the NLI model says it entails the reference and does not contradict it."""
    prompts = [agent_prompt(t) for t in ctx]
    with pol.frozen(ref):
        y_ref = pol.generate_text(prompts, GenConfig(), seed)
    samples = pol.generate_text([p for p in prompts for _ in range(pc.group)], GenConfig(temperature=1.0), seed + 1)
    rws: List[List[str]] = [[] for _ in ctx]
    stats = {"rewrites_generated": 0, "rewrites_kept": 0}
    if flags["rewrite"] and pc.rewrites > 0:
        rp = [rewrite_prompt(t, y) for t, y in zip(ctx, y_ref) for _ in range(pc.rewrites)]
        raw = pol.generate_text(rp, GenConfig(max_new_tokens=96, temperature=0.8), seed + 2,
                                adapter=pc.rewriter_adapter)
        drafts = [y for y in y_ref for _ in range(pc.rewrites)]
        pe, _ = ex.nli(raw, drafts)                 # completeness: the rewrite entails the draft
        _, pc_ = ex.nli(drafts, raw)                # consistency: the draft does not contradict the rewrite
        stats["rewrites_generated"] = len(raw)
        for k, (r, e, c) in enumerate(zip(raw, pe, pc_)):
            if e >= pc.rewrite_min_entail and c <= pc.rewrite_max_contra and norm_text(r) != norm_text(drafts[k]):
                rws[k // pc.rewrites].append(norm_text(r))
                stats["rewrites_kept"] += 1
    cands = []
    for k in range(len(ctx)):
        c = [norm_text(y_ref[k])] + rws[k] + [norm_text(x) for x in samples[k * pc.group:(k + 1) * pc.group]]
        cands.append(list(dict.fromkeys(c)))       # duplicates carry no preference information
    return cands, stats


def score_candidates(cands: List[List[str]], ctx: Sequence[Turn], judge: Optional[Judge], ex: Experiment, ns: str,
                     flags: Dict[str, Any]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Affect score and ESS of every candidate.  Every candidate is a proposal of its context's pool."""
    flat_t = [t for t, c in zip(ctx, cands) for _ in c]
    flat_y = [y for c in cands for y in c]
    if flags["judge"] == "agent_sentiment":
        a, e = np.asarray(ex.sentiment(flat_y), float), np.full(len(flat_y), np.inf)
    else:
        res = judge.expect(ns, flat_t, flat_y, {t.uid: c for t, c in zip(ctx, cands)})
        a, e = res["affect"], res["ess"]
    off = np.cumsum([0] + [len(c) for c in cands])
    return [a[off[k]:off[k + 1]] for k in range(len(cands))], [e[off[k]:off[k + 1]] for k in range(len(cands))]


def replicate_sigma(cands: List[List[str]], ctx: Sequence[Turn], A: List[np.ndarray], judge: Judge, ns: str,
                    n_ctx: int) -> float:
    """Noise sd of a within-context affect difference: re-score on an independent pool; for two pools,
    Var(d_0 - d_1) = 2 sigma^2."""
    idx = list(range(min(n_ctx, len(ctx))))
    res = judge.expect(ns + "#rep", [t for k in idx for t in [ctx[k]] * len(cands[k])],
                       [y for k in idx for y in cands[k]], {ctx[k].uid: cands[k] for k in idx})["affect"]
    dd, pos = [], 0
    for k in idx:
        m = len(cands[k])
        a0, a1 = A[k], res[pos:pos + m]
        iu = np.triu_indices(m, 1)
        dd.extend(((a0[:, None] - a0[None, :]) - (a1[:, None] - a1[None, :]))[iu].tolist())
        pos += m
    return float(max(np.std(dd) / math.sqrt(2.0), 1e-4)) if len(dd) > 2 else 1e-3


def train_arm(cfg: Config, arm: str, seed: int) -> Dict[str, Any]:
    """Train one arm (see ARM_FLAGS) and save its adapter as policy_<arm>_s<seed>."""
    ex = Experiment(cfg, f"4_train_{arm}_{seed}")
    lg, pc, flags = ex.logger, cfg.caspi, ARM_FLAGS[arm]
    tag = f"{arm}_s{seed}"
    seed_everything(seed)
    pol = ex.policy("sft_policy")
    ref = pol.snapshot()                             # SFT: the DPO reference and the baseline of every test
    judge = g_judge = g_monitor = None
    if flags["judge"] == "sim" or flags["guard"] or flags["certify"]:
        sim = ex.policy("simulator")
        judge, _ = ex.judges(sim, tag, pc.pool_size)
        g_judge, g_monitor = ex.judges(sim, f"{tag}_guard", pc.guard_pool_size)
    used = set()
    train_ctx = select_turns(ex.turns, "train", pc.rounds * pc.contexts_per_round, seed)
    sel = select_turns(ex.turns, "valid", pc.select_contexts, cfg.seed + 101, require_next=True, one_per_dialogue=True)
    used |= {t.dialogue_id for t in sel}
    safe = select_turns(ex.turns, "valid", pc.safety_contexts, cfg.seed + 202, require_next=True,
                        one_per_dialogue=True, exclude_dialogues=used)
    lg.info("%s | flags %s | %d training contexts | selection split %d contexts | safety split %d contexts "
            "(dialogue-disjoint)", tag, flags, len(train_ctx), len(sel), len(safe))
    beta, lr, nll = pc.beta, pc.lr, pc.nll_coef
    accepted, best_gain, n_acc, rejections = pol.snapshot(), 0.0, 0, 0
    offline_pairs: List[Tuple[str, str, str]] = []
    offline_w: List[float] = []
    hist: List[Dict[str, Any]] = []
    for rd in range(1, pc.rounds + 1):
        ctx = train_ctx[(rd - 1) * pc.contexts_per_round: rd * pc.contexts_per_round]
        if len(ctx) < 8:
            break
        cands, cstats = collect_candidates(pol, ref, ctx, pc, flags, ex, seed * 1000 + 10 * rd)
        A, E = score_candidates(cands, ctx, judge, ex, f"r{rd}", flags)
        sigma = (replicate_sigma(cands, ctx, A, judge, f"r{rd}", pc.calib_contexts)
                 if flags["pareto"] and flags["judge"] == "sim" else float("nan"))
        pairs, weights, kinds = [], [], {}
        for k, t in enumerate(ctx):
            ref_words = max(1, n_words(cands[k][0]))
            H = np.asarray([hygiene_ok(y) and (i == 0 or pc.len_low <= n_words(y) / ref_words <= pc.len_high)
                            for i, y in enumerate(cands[k])], bool)
            C = np.asarray([content_score(y, t.gold_response) for y in cands[k]])
            for i, j, w, kind in mine_pairs(A[k], C, H, E[k], cands[k], sigma, pc, flags["pareto"]):
                pairs.append((agent_prompt(t), cands[k][i], cands[k][j]))
                weights.append(w)
                kinds[kind] = kinds.get(kind, 0) + 1
        ess_all = np.concatenate(E)
        rec: Dict[str, Any] = {"round": rd, "contexts": len(ctx), "candidates": int(sum(len(c) for c in cands)),
                               "pairs": len(pairs), "pair_kinds": kinds, "sigma": sigma, **cstats,
                               # contexts whose gold turn carries no task information: there the information
                               # constraint of the pair miner is vacuous
                               "info_vacuous_share": float(np.mean([not math.isfinite(content_score("", t.gold_response))
                                                                    for t in ctx])),
                               "ess_median": float(np.median(ess_all[np.isfinite(ess_all)])) if np.isfinite(ess_all).any()
                               else float("nan")}
        if flags["offline"]:
            offline_pairs += pairs
            offline_w += weights
            hist.append(rec)
            lg.info("%s | round %d | collected %d pairs under SFT (offline)", tag, rd, len(pairs))
            continue
        upd = (pol.dpo_update(pairs, weights, ref, beta, lr, pc.epochs, pc.batch, nll, seed * 100 + rd)
               if pairs else {"loss": float("nan"), "pref_acc": float("nan"), "steps": 0, "d_chosen_logp": 0.0,
                              "d_rejected_logp": 0.0})
        rec.update(upd)
        rec["nll_coef"] = nll
        if upd["d_chosen_logp"] < 0:                # likelihood displacement: strengthen the anchor
            nll = min(pc.nll_coef_max, 2.0 * nll)
        accept = True
        if flags["guard"]:
            diffs, ess, level = paired_dev_check(pol, ref, g_judge, g_monitor, sel, f"guard_r{rd}", seed * 7 + 1)
            dec = vector_test(diffs, [t.dialogue_id for t in sel], pc.select_alpha, pc, best_gain, ess,
                              pc.guard_pool_size)
            accept = dec["accept"]
            rec.update({"guard": dec, "guard_levels": level})
            lg.info("%s | round %d guard | affect %+.4f (LB %+.4f) monitor %+.4f (LB %+.4f) content %+.3f hygiene %+.3f "
                    "log-length %+.3f | ESS %.1f | power(monitor,content,hygiene,length) %s | %s", tag, rd,
                    dec["means"]["affect"], dec["lower_bounds"]["affect"], dec["means"]["monitor"],
                    dec["lower_bounds"]["monitor"], dec["means"]["content"], dec["means"]["hygiene"],
                    dec["means"]["length"], ess, {k: round(v, 2) for k, v in dec["power_at_zero_difference"].items()},
                    "ACCEPT" if accept else f"REJECT {[k for k, v in dec['checks'].items() if not v]}")
            if accept:
                best_gain = max(best_gain, dec["means"]["affect"])
        if accept:
            accepted, n_acc, rejections = pol.snapshot(), n_acc + 1, 0
        else:
            pol.restore(accepted)
            beta, lr, rejections = beta * 2.0, lr * 0.5, rejections + 1
        rec.update({"accepted": accept, "beta_next": beta, "lr_next": lr})
        hist.append(rec)
        lg.info("%s | round %d | %d candidates -> %d pairs %s | sigma %.5f | rewrites kept %d/%d | loss %.4f pref-acc "
                "%.3f | d log p chosen %+.3f rejected %+.3f | %s", tag, rd, rec["candidates"], len(pairs), kinds, sigma,
                cstats["rewrites_kept"], cstats["rewrites_generated"], upd["loss"], upd["pref_acc"],
                upd["d_chosen_logp"], upd["d_rejected_logp"], "kept" if accept else "rolled back")
        if rejections >= pc.max_rejections:
            lg.info("%s | %d consecutive rejections: stop", tag, rejections)
            break
    if flags["offline"] and offline_pairs:
        upd = pol.dpo_update(offline_pairs, offline_w, ref, beta, lr, pc.epochs, pc.batch, nll, seed * 100)
        hist.append({"offline_update": upd})
        n_acc = 1
        accepted = pol.snapshot()
    pol.restore(accepted)
    cert: Dict[str, Any] = {"run": False}
    returned_sft = n_acc == 0
    if flags["certify"] and n_acc > 0:
        diffs, ess, level = paired_dev_check(pol, ref, g_judge, g_monitor, safe, "certify", seed * 7 + 2)
        dec = vector_test(diffs, [t.dialogue_id for t in safe], pc.alpha, pc, 0.0, ess, pc.guard_pool_size)
        cert = {"run": True, **dec, "levels": level, "n": len(safe)}
        lg.info("%s | CERTIFICATION (alpha %.3f, %s bound, %d contexts) | %s | failed: %s | power at zero difference "
                "%s | dialogues for 80%% power %s", tag, pc.alpha, pc.bound, len(safe),
                "PASS" if dec["accept"] else "FAIL -> returning SFT", [k for k, v in dec["checks"].items() if not v],
                {k: round(v, 2) for k, v in dec["power_at_zero_difference"].items()},
                dec["dialogues_for_80pct_power"])
        if not dec["accept"]:
            pol.restore(ref)
            returned_sft = True
    if returned_sft:
        lg.warning("%s | the returned policy IS the SFT policy (no accepted round, or certification failed); this is "
                   "the algorithm working as designed, report it", tag)
    pol.save_adapter(cfg.out_dir / f"policy_{tag}")
    res = {"arm": arm, "seed": seed, "flags": flags, "rounds": hist, "accepted_rounds": n_acc,
           "certification": cert, "returned_sft": returned_sft, "config": asdict(pc)}
    dump_json(res, cfg.out_dir / f"train_{tag}.json")
    return res


def stage_preregister(cfg: Config) -> Dict[str, Any]:
    """Write the analysis plan before any policy is trained.  Every margin, level and split seed of the tests
    is fixed here; the claims stage reads the margins from this file, never from data."""
    ex = Experiment(cfg, "3_preregister")
    f = cfg.out_dir / "preregistration.json"
    plan = {"version": VERSION, "caspi": asdict(cfg.caspi), "equiv_margin_outcome": cfg.equiv_margin_outcome,
            "seeds": list(cfg.seeds), "arms": list(cfg.arms), "primary": f"{PRIMARY}_vs_sft:outcome",
            "co_primary": f"{PRIMARY}_vs_sft:content (non-inferiority at {cfg.caspi.margin_info})",
            "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    if f.exists():
        old = load_json(f)
        changed = [k for k in ("caspi", "equiv_margin_outcome", "primary") if old.get(k) != plan[k]]
        if changed:
            raise RuntimeError(f"preregistration.json exists and differs in {changed}: delete it deliberately and "
                               "report the change, or restore the registered values")
        return old
    dump_json(plan, f)
    ex.logger.info("preregistration written to %s", f)
    return plan


# ==========================================================================================================
# evaluation
# ==========================================================================================================

def stage_eval(cfg: Config) -> Dict[str, Any]:
    """Test-split evaluation of every arm under every seed.  Generation uses the same seeds for all arms
    (common random numbers).  Scoring is symmetric: per (seed, context) every arm's reply is a proposal of
    ONE shared pool, under both the training judge and the monitor."""
    ex = Experiment(cfg, "5_eval")
    te = select_turns(ex.turns, "test", cfg.eval_turns, cfg.seed, require_next=True)
    prompts = [agent_prompt(t) for t in te]
    gen = GenConfig()
    pol = ex.policy("sft_policy")
    replies: Dict[int, Dict[str, List[str]]] = {}
    bon_samples: Dict[int, List[List[str]]] = {}
    for sd in cfg.seeds:
        replies[sd] = {}
        for arm in cfg.arms:
            if arm in EVAL_ONLY_ARMS:
                pol.load_adapter(cfg.out_dir / "sft_policy")
                if arm == "sft":
                    replies[sd][arm] = pol.generate_text(prompts, gen, 9000 + sd)
                else:
                    flat = pol.generate_text([p for p in prompts for _ in range(cfg.bon_n)], gen, 9500 + sd)
                    bon_samples[sd] = [flat[k * cfg.bon_n:(k + 1) * cfg.bon_n] for k in range(len(te))]
            else:
                pol.load_adapter(cfg.out_dir / f"policy_{arm}_s{sd}")
                replies[sd][arm] = pol.generate_text(prompts, gen, 9000 + sd)
    del pol
    release_gpu()
    sim = ex.policy("simulator")
    judge, monitor = ex.judges(sim, "eval", cfg.eval_pool_size)
    for sd, cands in bon_samples.items():
        # best-of-N is SELECTED on an independent pool, so the evaluation pool does not reward its own
        # selection noise (winner's curse)
        flat_t = [t for t in te for _ in range(cfg.bon_n)]
        a = judge.expect(f"bon_select_s{sd}", flat_t, [y for c in cands for y in c],
                         {t.uid: c for t, c in zip(te, cands)})["affect"].reshape(len(te), cfg.bon_n)
        ok = np.asarray([[hygiene_ok(y) for y in c] for c in cands])
        a = np.where(ok | ~ok.any(1, keepdims=True), a, -np.inf)
        replies[sd]["sft_bon"] = [c[int(np.argmax(r))] for c, r in zip(cands, a)]
    for sd in cfg.seeds:
        arms = list(replies[sd])
        props = {t.uid: [replies[sd][a][k] for a in arms] for k, t in enumerate(te)}
        for a in arms:
            ja = judge.expect(f"eval_s{sd}", te, replies[sd][a], props)
            jm = monitor.expect(f"eval_s{sd}", te, replies[sd][a], props)
            rm = reply_metrics(replies[sd][a], te)
            agent_sent = ex.sentiment(replies[sd][a])
            rows = [{"uid": t.uid, "dialogue_id": t.dialogue_id, "arm": a, "seed": sd, "response": r,
                     "outcome": float(ja["affect"][k]), "monitor": float(jm["monitor"][k]),
                     "ess": float(ja["ess"][k]), "agent_sentiment": float(agent_sent[k]),
                     **{m: float(v[k]) for m, v in rm.items()}, "human_valence": t.human_valence}
                    for k, (t, r) in enumerate(zip(te, replies[sd][a]))]
            dump_json(rows, cfg.out_dir / f"eval_{a}_{sd}.json")
            ex.logger.info("eval %-17s seed %d | outcome %.4f monitor %.4f | content %.3f hygiene %.3f words %.1f | "
                           "median ESS %.1f", a, sd, np.mean(ja["affect"]), np.mean(jm["monitor"]),
                           np.nanmean(rm["content"]), rm["hygiene"].mean(), rm["words"].mean(), np.median(ja["ess"]))
    return {"n_contexts": len(te)}


def stage_eval_external(cfg: Config) -> Dict[str, Any]:
    """Re-score every evaluated reply with an evaluator that shares nothing with training: an out-of-family
    customer LLM (no adapter, no EmoWOZ fine-tuning) read by an emotion classifier that no stage uses, with the
    same symmetric shared-pool protocol as `eval`."""
    ex = Experiment(cfg, "6_eval_external")
    rows = load_rows(cfg, "eval")
    turn_of = {t.uid: t for t in ex.turns}
    ext = Policy(cfg.external_model, cfg, ex.logger, lora=False)
    emo = ValenceClassifier(cfg.external_emotion_model, cfg, ex.logger)
    judge = Judge(ext, {"ext": emo}, chat_customer_prompt, cfg.out_dir / "pools" / "external.jsonl",
                  cfg.external_pool_size, True, cfg.sim_max_new_tokens, ex.logger, "external")
    for sd in cfg.seeds:
        arms = [a for a in rows if sd in rows[a]]
        if not arms:
            continue
        uids = [r["uid"] for r in rows[arms[0]][sd]]
        te = [turn_of[u] for u in uids]
        resp = {a: {r["uid"]: r["response"] for r in rows[a][sd]} for a in arms}
        props = {u: [resp[a][u] for a in arms] for u in uids}
        for a in arms:
            res = judge.expect(f"ext_s{sd}", te, [r["response"] for r in rows[a][sd]], props)
            out = [{**r, "outcome": float(v), "sim_outcome": r["outcome"]} for r, v in zip(rows[a][sd], res["ext"])]
            dump_json(out, cfg.out_dir / f"ext_{a}_{sd}.json")
            ex.logger.info("external %-17s seed %d | outcome %.4f | rank agreement with the training judge (rows) "
                           "%+.3f", a, sd, np.mean(res["ext"]), spearman(res["ext"], [r["outcome"] for r in rows[a][sd]]))
    return {"seeds": list(cfg.seeds)}


def stage_cross_eval(cfg: Config) -> Dict[str, Any]:
    """Within-context agreement of the training judge and the external evaluator on K SFT replies per context
    (chance = 0.5), plus between-context anchors on context means.  The samples are kept for the human
    study's validity items.  Automatic evidence only."""
    ex = Experiment(cfg, "6_cross_eval")
    te = select_turns(ex.turns, "test", cfg.cross_eval_contexts, cfg.seed + 7, require_next=True, one_per_dialogue=True)
    K = max(2, cfg.cross_eval_k)
    flat_t = [t for t in te for _ in range(K)]
    pol = ex.policy("sft_policy")
    ys = pol.generate_text([agent_prompt(t) for t in flat_t], GenConfig(), cfg.seed + 77)
    del pol
    release_gpu()
    props = {t.uid: ys[k * K:(k + 1) * K] for k, t in enumerate(te)}
    sim = ex.policy("simulator")
    judge, _ = ex.judges(sim, "cross", cfg.eval_pool_size)
    a_sim = judge.expect("cross", flat_t, ys, props)["affect"].reshape(len(te), K)
    del sim, judge
    release_gpu()
    ext = Policy(cfg.external_model, cfg, ex.logger, lora=False)
    ej = Judge(ext, {"ext": ValenceClassifier(cfg.external_emotion_model, cfg, ex.logger)}, chat_customer_prompt,
               cfg.out_dir / "pools" / "cross_external.jsonl", cfg.external_pool_size, True, cfg.sim_max_new_tokens,
               ex.logger, "external")
    a_ext = ej.expect("cross", flat_t, ys, props)["ext"].reshape(len(te), K)
    iu = np.triu_indices(K, 1)
    per, cl = [], []
    for c, t in enumerate(te):
        x = (a_sim[c][:, None] - a_sim[c][None, :])[iu]
        y = (a_ext[c][:, None] - a_ext[c][None, :])[iu]
        m = np.abs(x) > 1e-9
        if m.any():
            per.append(float(np.mean(x[m] * y[m] > 0)))
            cl.append(t.dialogue_id)
    pa, ca = np.asarray(per), np.asarray(cl, dtype=object)
    ci = cluster_bootstrap_ci(lambda ix: float(np.mean(pa[ix])), ca, cfg.n_boot, cfg.seed)
    sf = sign_flip_test(pa - 0.5, ca, cfg.n_signflip, cfg.seed + 1)
    hv = np.asarray([t.human_valence if t.human_valence is not None else np.nan for t in te], float)
    out = {"n_contexts": len(per), "K": K, "agreement": ci, "p_vs_chance": sf["p"], "p_text": sf["text"],
           "between_context_rho_simulator": spearman(a_sim.mean(1), hv),
           "between_context_rho_external": spearman(a_ext.mean(1), hv)}
    dump_json(out, cfg.out_dir / "cross_eval.json")
    dump_json([{"uid": t.uid, "dialogue_id": t.dialogue_id, "replies": ys[k * K:(k + 1) * K],
                "sim": a_sim[k].tolist()} for k, t in enumerate(te)], cfg.out_dir / "cross_eval_samples.json")
    ex.logger.info("cross-evaluator within-context agreement %.3f CI[%.3f,%.3f] (chance 0.5, p %s) over %d contexts | "
                   "between-context rho: simulator %+.3f, external %+.3f", *ci, sf["text"], len(per),
                   out["between_context_rho_simulator"], out["between_context_rho_external"])
    return out


# ==========================================================================================================
# report, length analysis, claims
# ==========================================================================================================

REPORT_METRICS = ("outcome", "monitor", "content", "info_recall", "slot_f1", "hygiene", "words", "agent_sentiment")


def load_rows(cfg: Config, prefix: str) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    rows: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    for arm in cfg.arms:
        for sd in cfg.seeds:
            f = cfg.out_dir / f"{prefix}_{arm}_{sd}.json"
            if f.exists():
                rows.setdefault(arm, {})[sd] = load_json(f)
    if not rows:
        raise FileNotFoundError(f"no {prefix}_*.json rows in {cfg.out}")
    return rows


def paired_contrast(rows_a: Dict[int, List[Dict[str, Any]]], rows_b: Dict[int, List[Dict[str, Any]]], metric: str,
                    seed: int, n_boot: int, n_mc: int, seeds: Optional[Sequence[int]] = None,
                    equiv: Optional[float] = None, noninf: Optional[float] = None) -> Dict[str, Any]:
    """Arm minus comparator, paired within (seed, context), pooled over seeds, DIALOGUES as clusters.
    equiv: two-sided TOST margin (90% CI inside +-equiv); noninf: one-sided margin (lower CI90 > -noninf)."""
    d, cl = [], []
    for sd, ra in rows_a.items():
        if (seeds is not None and sd not in seeds) or sd not in rows_b:
            continue
        ib = {r["uid"]: r for r in rows_b[sd]}
        for r in ra:
            q = ib.get(r["uid"])
            if q is not None and r.get(metric) is not None and q.get(metric) is not None and \
                    math.isfinite(float(r[metric])) and math.isfinite(float(q[metric])):
                d.append(float(r[metric]) - float(q[metric]))
                cl.append(str(r["dialogue_id"]))
    if len(d) < 20:
        return {"n": len(d)}
    d_, cl_ = np.asarray(d), np.asarray(cl, dtype=object)
    pt, lo, hi = cluster_bootstrap_ci(lambda ix: float(np.mean(d_[ix])), cl_, n_boot, seed)
    _, lo90, hi90 = cluster_bootstrap_ci(lambda ix: float(np.mean(d_[ix])), cl_, n_boot, seed + 7, conf=0.90)
    sf = sign_flip_test(d_, cl_, n_mc, seed + 1)
    se = cluster_robust_se(d_, cl_)
    sd_d = float(np.std(d_, ddof=1))
    out = {"n": len(d), "n_clusters": int(np.unique(cl_).size), "delta": pt, "ci95": [lo, hi], "ci90": [lo90, hi90],
           "p_value": sf["p"], "p_text": sf["text"], "se_cluster": se,
           "mde80": float((normal_ppf(0.975) + normal_ppf(0.8)) * se) if math.isfinite(se) else float("nan"),
           "cohens_dz": float(pt / sd_d) if sd_d > 0 else float("nan"),
           "paired_dominance": float(np.mean(d_ > 0) - np.mean(d_ < 0))}
    if equiv is not None:
        out.update({"equiv_margin": equiv, "equivalent": bool(lo90 > -equiv and hi90 < equiv)})
    if noninf is not None:
        out.update({"noninf_margin": noninf, "noninferior": bool(lo90 > -noninf)})
    return out


def comparison_plan(arms: Sequence[str]) -> List[Tuple[str, str]]:
    plan = [(a, "sft") for a in arms if a != "sft"]
    if PRIMARY in arms:
        plan += [(PRIMARY, a) for a in arms if a not in ("sft", PRIMARY)]
    return plan


def stage_report(cfg: Config, prefix: str = "eval") -> Dict[str, Any]:
    """Arm means and every planned contrast; Holm within each metric family.  outcome gets a TOST margin
    (for null claims such as "the guard costs no affect"); content/info/monitor get non-inferiority margins."""
    ex = Experiment(cfg, f"7_report_{prefix}")
    rows = load_rows(cfg, prefix)
    pre = load_json(cfg.out_dir / "preregistration.json") if (cfg.out_dir / "preregistration.json").exists() else None
    pcm = pre["caspi"] if pre else asdict(cfg.caspi)
    eq = pre["equiv_margin_outcome"] if pre else cfg.equiv_margin_outcome
    metrics = [m for m in REPORT_METRICS if any(m in rs[0] for per in rows.values() for rs in per.values() if rs)]
    summary: Dict[str, Any] = {"evaluator": prefix, "arms": {}, "contrasts": {}}
    for arm, per in rows.items():
        allr = [r for rs in per.values() for r in rs]
        summary["arms"][arm] = {m: float(np.nanmean([r[m] for r in allr])) for m in metrics}
        summary["arms"][arm]["n_rows"] = len(allr)
    ni = {"content": pcm["margin_info"], "info_recall": pcm["margin_info"], "monitor": pcm["margin_monitor"]}
    for a, b in comparison_plan([x for x in cfg.arms if x in rows]):
        for m in metrics:
            c = paired_contrast(rows[a], rows[b], m, cfg.seed, cfg.n_boot, cfg.n_signflip,
                                equiv=eq if m == "outcome" else None, noninf=ni.get(m))
            if c.get("n", 0) >= 20:
                summary["contrasts"][f"{a}_vs_{b}:{m}"] = {**c, "arm": a, "comparator": b, "metric": m}
    for m in metrics:
        keys = [k for k, v in summary["contrasts"].items() if v["metric"] == m]
        for k, p in zip(keys, holm([summary["contrasts"][k]["p_value"] for k in keys])):
            summary["contrasts"][k]["p_holm"] = p
    dump_json(summary, cfg.out_dir / f"report_{prefix}.json")
    ex.logger.info("%-18s %9s %9s %8s %8s %8s", "arm", "outcome", "monitor", "content", "hygiene", "words")
    for arm, v in summary["arms"].items():
        ex.logger.info("%-18s %9.4f %9.4f %8.3f %8.3f %8.1f", arm, v.get("outcome", np.nan), v.get("monitor", np.nan),
                       v.get("content", np.nan), v.get("hygiene", np.nan), v.get("words", np.nan))
    for k, v in summary["contrasts"].items():
        if v["metric"] in ("outcome", "content", "monitor"):
            ex.logger.info("%-40s %+.4f CI95[%+.4f,%+.4f] p %s holm %.4f | MDE80 %.4f%s", k, v["delta"], *v["ci95"],
                           v["p_text"], v["p_holm"], v["mde80"],
                           f" | non-inferior: {v['noninferior']}" if "noninferior" in v else
                           (f" | TOST-equivalent: {v['equivalent']}" if "equivalent" in v else ""))
    return summary


def stage_length(cfg: Config) -> Dict[str, Any]:
    """Is the outcome gain over SFT explained by length?  Paired regression d_outcome = a + b d_words (a is the
    gain at equal length; descriptive, since length is post-treatment) and the length-matched stratum."""
    ex = Experiment(cfg, "7_length")
    rows = load_rows(cfg, "eval")
    out: Dict[str, Any] = {}
    for arm in [a for a in rows if a != "sft" and "sft" in rows]:
        dy, dw, cl = [], [], []
        for sd, ra in rows[arm].items():
            ib = {r["uid"]: r for r in rows["sft"].get(sd, [])}
            for r in ra:
                if r["uid"] in ib:
                    dy.append(r["outcome"] - ib[r["uid"]]["outcome"])
                    dw.append(r["words"] - ib[r["uid"]]["words"])
                    cl.append(r["dialogue_id"])
        if len(dy) < 30:
            continue
        dy_, dw_, cl_ = np.asarray(dy), np.asarray(dw), np.asarray(cl, dtype=object)

        def intercept(ix):
            return float(np.linalg.lstsq(np.column_stack([np.ones(ix.size), dw_[ix]]), dy_[ix], rcond=None)[0][0])
        m = np.abs(dw_) <= 2
        res = {"raw": float(dy_.mean()), "mean_d_words": float(dw_.mean()),
               "equal_length": cluster_bootstrap_ci(intercept, cl_, cfg.n_boot, cfg.seed),
               "matched_n": int(m.sum())}
        if m.sum() >= 30:
            dm, cm = dy_[m], cl_[m]
            res["matched"] = cluster_bootstrap_ci(lambda ix: float(np.mean(dm[ix])), cm, cfg.n_boot, cfg.seed + 1)
        out[arm] = res
        ex.logger.info("%-18s raw %+.4f (d words %+.1f) | at equal length %+.4f CI[%+.4f,%+.4f] | |dw|<=2: n=%d %s",
                       arm, res["raw"], res["mean_d_words"], *res["equal_length"], res["matched_n"],
                       "" if "matched" not in res else "%+.4f CI[%+.4f,%+.4f]" % tuple(res["matched"]))
    dump_json(out, cfg.out_dir / "length_analysis.json")
    return out


def stage_human_export(cfg: Config) -> Dict[str, Any]:
    """Blinded, randomised pairwise packet: arm pairs (same context, seed 1), within-context judge-validity
    items (two SFT replies to one context from cross-eval) and attention checks.  The key is kept apart."""
    ex = Experiment(cfg, "8_human_export")
    rng = np.random.default_rng(cfg.seed + 2024)
    d = cfg.out_dir / "human_study"
    d.mkdir(parents=True, exist_ok=True)
    turn_of = {t.uid: t for t in ex.turns}
    rows = load_rows(cfg, "eval")
    sd0 = cfg.seeds[0]
    items: List[Dict[str, Any]] = []
    for comp in cfg.human_comparisons:
        a, b = comp.split(":")
        if sd0 not in rows.get(a, {}) or sd0 not in rows.get(b, {}):
            ex.logger.warning("comparison %s skipped: eval rows missing", comp)
            continue
        ia, ib = {r["uid"]: r for r in rows[a][sd0]}, {r["uid"]: r for r in rows[b][sd0]}
        by_d: Dict[str, List[str]] = {}
        for u in sorted(set(ia) & set(ib)):
            if norm_text(ia[u]["response"]) != norm_text(ib[u]["response"]):
                by_d.setdefault(ia[u]["dialogue_id"], []).append(u)
        for k in list(rng.permutation(sorted(by_d)))[:cfg.human_contexts]:
            u = by_d[k][int(rng.integers(len(by_d[k])))]
            items.append({"kind": "arm_pair", "comparison": comp, "uid": u, "dialogue_id": k,
                          "A": (a, ia[u]["response"], ia[u]["outcome"]), "B": (b, ib[u]["response"], ib[u]["outcome"])})
    cs = cfg.out_dir / "cross_eval_samples.json"
    if cs.exists():
        cand = []
        for s in load_json(cs):
            ok = [i for i, y in enumerate(s["replies"]) if hygiene_ok(y)]
            if len(ok) >= 2:
                i, j = (int(x) for x in rng.choice(ok, 2, replace=False))
                if norm_text(s["replies"][i]) != norm_text(s["replies"][j]):
                    cand.append((s, i, j, abs(s["sim"][i] - s["sim"][j])))
        cand.sort(key=lambda c: -c[3])
        top = min(len(cand) // 3, cfg.human_validity_pairs // 2)
        rest = [int(k) for k in rng.permutation(np.arange(top, len(cand)))[:max(0, cfg.human_validity_pairs - top)]]
        for k in list(range(top)) + rest:
            s, i, j, m = cand[k]
            items.append({"kind": "validity", "comparison": "judge", "uid": s["uid"], "dialogue_id": s["dialogue_id"],
                          "A": ("reply_i", s["replies"][i], s["sim"][i]), "B": ("reply_j", s["replies"][j], s["sim"][j]),
                          "margin": m, "stratum": "top_margin" if k < top else "random"})
    pool = [t for t in ex.turns if t.split == "test" and n_words(t.gold_response) >= 5]
    for k in range(max(4, round(0.05 * len(items)))):
        t = pool[int(rng.integers(len(pool)))]
        items.append({"kind": "catch", "comparison": "catch", "uid": t.uid, "dialogue_id": t.dialogue_id,
                      "A": ("gold", t.gold_response, float("nan")),
                      "B": ("catch", CATCH_RESPONSES[k % len(CATCH_RESPONSES)], float("nan"))})
    if not items:
        raise RuntimeError("no items for the human packet: run eval (and cross-eval) first")
    packet, key = [], {}
    for pos, k in enumerate(rng.permutation(len(items))):
        it = items[int(k)]
        L, R = (it["B"], it["A"]) if rng.random() < 0.5 else (it["A"], it["B"])
        iid = f"item{pos:05d}"
        packet.append({"item_id": iid, "context": context_text(turn_of[it["uid"]]), "reply_1": L[1], "reply_2": R[1],
                       "double_annotate": int(rng.random() < cfg.human_overlap or it["kind"] == "catch")})
        key[iid] = {"kind": it["kind"], "comparison": it["comparison"], "uid": it["uid"],
                    "dialogue_id": it["dialogue_id"], "left": L[0], "right": R[0], "left_score": L[2],
                    "right_score": R[2], **({"margin": it["margin"], "stratum": it["stratum"]}
                                            if it["kind"] == "validity" else {})}
    with open(d / "items.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(packet[0]))
        w.writeheader()
        w.writerows(packet)
    with open(d / "annotations_template.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["annotator_id", "item_id"] + list(HUMAN_QUESTIONS))
    dump_json(key, d / "key_DO_NOT_SHARE.json")
    (d / "instructions.md").write_text(
        "# Annotation instructions\n\nEach item shows a customer-service conversation and two agent replies in "
        "random order. For each question answer 1 (reply_1), 2 (reply_2) or 0 (no real difference).\n\n"
        + "\n".join(f"* **{k}**: {v}" for k, v in HUMAN_QUESTIONS.items())
        + "\n\nItems with double_annotate=1 must be labelled by two different annotators.\n", encoding="utf-8")
    need = binomial_n(0.60)
    per = {c: sum(1 for it in items if it["comparison"] == c) for c in cfg.human_comparisons}
    ex.logger.info("human packet | %d items -> %s | a 60%% win rate needs ~%d non-tie judgements per comparison; "
                   "planned %s", len(packet), d, need, per)
    meta = {"n_items": len(packet), "per_comparison": per, "power_n_60pct": need}
    dump_json(meta, d / "packet_meta.json")
    return meta


def stage_human_analyze(cfg: Config) -> Dict[str, Any]:
    """Win rates from filled annotations*.csv against the hidden key.  Annotators below 80% on the attention
    checks are excluded (pre-registered).  Sign-flip tests on item scores clustered by dialogue, Holm across
    comparisons x questions, Krippendorff alpha on double-annotated items."""
    ex = Experiment(cfg, "8_human_analyze")
    d = cfg.out_dir / "human_study"
    key = load_json(d / "key_DO_NOT_SHARE.json")
    ann = []
    for f in sorted(d.glob("annotations*.csv")):
        if f.name != "annotations_template.csv":
            with open(f, encoding="utf-8") as fh:
                ann.extend(csv.DictReader(fh))
    if not ann:
        raise FileNotFoundError(f"no filled annotations*.csv in {d}")
    code = {"1": 1.0, "2": -1.0, "0": 0.0}
    acc: Dict[str, List[float]] = {}
    for a in ann:
        k = key.get(a["item_id"])
        v = code.get(str(a.get("q_satisfaction", "")).strip())
        if k and k["kind"] == "catch" and v is not None:
            acc.setdefault(a["annotator_id"], []).append(float(v != 0 and (v > 0) == (k["left"] == "gold")))
    keep = {u for u, v in acc.items() if np.mean(v) >= 0.8}
    dropped = sorted({a["annotator_id"] for a in ann} - keep)
    per_item: Dict[str, Dict[str, List[float]]] = {}
    for a in ann:
        k = key.get(a["item_id"])
        if a["annotator_id"] not in keep or not k or k["kind"] == "catch":
            continue
        for q in HUMAN_QUESTIONS:
            v = code.get(str(a.get(q, "")).strip())
            if v is None:
                continue
            if k["kind"] == "arm_pair":
                v = v if k["left"] == k["comparison"].split(":")[0] else -v
            else:
                v = v if k["left_score"] >= k["right_score"] else -v       # +1: the human agrees with the judge
            per_item.setdefault(a["item_id"], {}).setdefault(q, []).append(v)
    res: Dict[str, Any] = {"excluded_annotators": dropped, "comparisons": {}, "agreement": {}}
    names, pv = [], []
    for comp in sorted({k["comparison"] for k in key.values() if k["kind"] != "catch"}):
        for q in HUMAN_QUESTIONS:
            ids = [i for i, k in key.items() if k["comparison"] == comp and q in per_item.get(i, {})]
            if len(ids) < 10:
                continue
            sc = np.asarray([np.mean(per_item[i][q]) for i in ids])
            cl = np.asarray([key[i]["dialogue_id"] for i in ids], dtype=object)
            ci = cluster_bootstrap_ci(lambda ix: float(np.mean(sc[ix] > 0) + 0.5 * np.mean(sc[ix] == 0)), cl,
                                      cfg.n_boot, cfg.seed)
            ci90 = cluster_bootstrap_ci(lambda ix: float(np.mean(sc[ix] > 0) + 0.5 * np.mean(sc[ix] == 0)), cl,
                                        cfg.n_boot, cfg.seed + 1, conf=0.90)
            sf = sign_flip_test(sc, cl, cfg.n_signflip, cfg.seed + 2)
            r = {"n_items": len(ids), "win_rate": ci[0], "ci95": list(ci[1:]), "ci90": list(ci90[1:]),
                 "p_value": sf["p"], "p_text": sf["text"], "tie_rate": float(np.mean(sc == 0))}
            if q == "q_information" and comp != "judge":
                r["noninferior"] = bool(ci90[1] > 0.5 - cfg.caspi.margin_info)
            res["comparisons"].setdefault(comp, {})[q] = r
            names.append((comp, q))
            pv.append(sf["p"])
            ex.logger.info("human | %-24s %-15s win %.3f CI95[%.3f,%.3f] ties %.2f p %s n=%d", comp, q, ci[0], ci[1],
                           ci[2], r["tie_rate"], sf["text"], len(ids))
    for (comp, q), p in zip(names, holm(pv)):
        res["comparisons"][comp][q]["p_holm"] = p
    for q in HUMAN_QUESTIONS:
        units = {i: v[q] for i, v in per_item.items() if len(v.get(q, [])) >= 2}
        res["agreement"][q] = {"krippendorff_alpha": krippendorff_alpha_interval(units), "n_units": len(units)}
    ex.logger.info("human | %d annotators excluded by the attention check: %s", len(dropped), dropped)
    dump_json(res, cfg.out_dir / "human_study_results.json")
    return res


def stage_claims(cfg: Config) -> Dict[str, Any]:
    """Turn the artefacts into verdicts with pre-registered rules; missing evidence is "untested"."""
    ex = Experiment(cfg, "9_claims")

    def get(name: str) -> Dict[str, Any]:
        f = cfg.out_dir / name
        return load_json(f) if f.exists() else {}
    rep, ext, hs = get("report_eval.json"), get("report_ext.json"), get("human_study_results.json")
    jv, ce = get("judge_validity.json"), get("cross_eval.json")
    C, P = rep.get("contrasts", {}), PRIMARY
    claims: List[Dict[str, Any]] = []

    def add(cid: str, text: str, verdict: str, evidence: Any = None):
        claims.append({"id": cid, "claim": text, "verdict": verdict, "evidence": evidence})

    def sig_pos(c):
        return None if not c else bool(c["ci95"][0] > 0 and c.get("p_holm", 1.0) < 0.05)

    def verdict(v):
        return "untested" if v is None else ("supported" if v else "not supported")

    for comp, text in ((f"{P}_vs_sft:outcome", "H1: CASPI raises simulated customer affect over SFT"),
                       (f"{P}_vs_caspi_no_rewrite:outcome", "H3a: delivery-only rewrites contribute"),
                       (f"{P}_vs_caspi_no_pareto:outcome", "H3b: noise-calibrated Pareto pairs contribute"),
                       (f"{P}_vs_online_dpo:outcome", "H4: CASPI beats online DPO with the same judge"),
                       (f"{P}_vs_sft_bon:outcome", "H4: CASPI beats best-of-N with the same judge"),
                       (f"{P}_vs_sentiment_only:outcome", "H4: the simulated-customer signal beats agent-wording "
                                                          "sentiment with the same optimiser"),
                       (f"{P}_vs_offline_dpo:outcome", "H4: CASPI beats offline DPO with the same data budget")):
        c = C.get(comp)
        add(comp, text, verdict(sig_pos(c)), c and {k: c.get(k) for k in ("delta", "ci95", "p_holm")})
    c = C.get(f"{P}_vs_sft:content")
    add("H2", "CASPI conveys no less task information than SFT (non-inferiority)",
        verdict(None if not c else c.get("noninferior")), c and {k: c.get(k) for k in ("delta", "ci90", "noninf_margin")})
    c = C.get(f"{P}_vs_caspi_no_guard:outcome")
    add("H3c", "the acceptance machinery costs no affect (TOST equivalence)",
        "untested" if not c else ("supported" if c.get("equivalent") else
                                  ("not supported: the guard costs affect" if c["ci95"][1] < 0 else "inconclusive")),
        c and {k: c.get(k) for k in ("delta", "ci90", "equiv_margin")})
    e = ext.get("contrasts", {}).get(f"{P}_vs_sft:outcome")
    add("H5a", "the gain replicates with an out-of-family customer and an unseen emotion classifier",
        verdict(sig_pos(e)), e and {k: e.get(k) for k in ("delta", "ci95", "p_holm")})
    hc = hs.get("comparisons", {}).get(f"{P}:sft", {}).get("q_satisfaction")
    add("H5b", "humans prefer CASPI over SFT on satisfaction",
        "untested" if not hc else verdict(hc["ci95"][0] > 0.5 and hc.get("p_holm", 1) < 0.05), hc)
    hv = hs.get("comparisons", {}).get("judge", {}).get("q_satisfaction")
    add("H5c", "the judge ranks alternative replies to the SAME context as humans do",
        "untested (needs the human study)" if not hv else verdict(hv["ci95"][0] > 0.5), hv)
    add("H5c_automatic", "two independent automatic evaluators agree on within-context rankings above chance",
        "untested" if not ce else verdict(ce["agreement"][1] > 0.5), ce and {"agreement": ce["agreement"]})
    add("judge_partial_anchor", "the judge tracks the next-turn human emotion beyond the current emotion",
        "untested" if not jv else verdict(jv["partial_anchor"][1] > 0), jv and jv.get("partial_anchor"))
    # H6: no returned CASPI policy is significantly worse than SFT on the test split, per seed and objective
    rows = load_rows(cfg, "eval") if (cfg.out_dir / f"eval_sft_{cfg.seeds[0]}.json").exists() else {}
    if P in rows and "sft" in rows:
        worse = []
        for sd in cfg.seeds:
            for m, mg in (("outcome", 0.0), ("monitor", cfg.caspi.margin_monitor), ("content", cfg.caspi.margin_info),
                          ("hygiene", cfg.caspi.margin_hygiene)):
                c = paired_contrast(rows[P], rows["sft"], m, cfg.seed, cfg.n_boot, cfg.n_signflip, seeds=[sd])
                if c.get("n", 0) >= 20 and c["ci95"][1] < -mg:
                    worse.append({"seed": sd, "metric": m, "delta": c["delta"], "ci95": c["ci95"]})
        add("H6", "no returned CASPI policy is significantly worse than SFT on any objective (test split)",
            "supported" if not worse else "not supported", worse or None)
    runs = [get(f"train_{P}_s{sd}.json") for sd in cfg.seeds]
    runs = [r for r in runs if r]
    if runs:
        add("certified_improvement", "CASPI certified an improvement in every seed",
            "supported" if all(not r["returned_sft"] for r in runs) else
            f"not supported ({sum(r['returned_sft'] for r in runs)} of {len(runs)} seeds returned SFT)",
            [{"seed": r["seed"], "accepted_rounds": r["accepted_rounds"], "certified": r["certification"].get("accept")}
             for r in runs])
    dump_json(claims, cfg.out_dir / "claims.json")
    lines = ["# Claims audit (generated; edit the rules, never the verdicts)", "", "| id | claim | verdict |", "|---|---|---|"]
    lines += [f"| {c['id']} | {c['claim']} | {c['verdict']} |" for c in claims]
    (cfg.out_dir / "claims.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for c in claims:
        ex.logger.info("CLAIM %-36s %s", c["id"], c["verdict"])
    return {"claims": claims}


def stage_train(cfg: Config) -> None:
    for arm in cfg.arms:
        if arm in ARM_FLAGS:
            for sd in cfg.seeds:
                train_arm(cfg, arm, sd)


def stage_all(cfg: Config) -> None:
    stage_data(cfg)
    stage_sft(cfg)
    stage_simulator(cfg)
    stage_validate(cfg)
    stage_preregister(cfg)
    stage_train(cfg)
    stage_eval(cfg)
    stage_eval_external(cfg)
    stage_cross_eval(cfg)
    stage_report(cfg, "eval")
    stage_report(cfg, "ext")
    stage_length(cfg)
    stage_human_export(cfg)
    stage_claims(cfg)


# ==========================================================================================================
# tests
# ==========================================================================================================

def run_unit_tests() -> None:
    """Numerical checks of the statistics, the estimator and the guarantee (numpy only)."""
    rng = np.random.default_rng(0)
    # quantiles
    assert abs(normal_ppf(0.975) - 1.959963985) < 1e-6 and abs(normal_ppf(0.01) + 2.326347874) < 1e-6
    for p, df, ref in ((0.95, 10, 1.812461), (0.95, 30, 1.697261), (0.975, 20, 2.085963), (0.99, 50, 2.403272)):
        assert abs(t_ppf(p, df) - ref) < 2e-3, (p, df, t_ppf(p, df), ref)
    # sign-flip exactness and rank helpers
    assert abs(sign_flip_test(np.ones(8), np.arange(8), 2000, 0)["p"] - 2 / 256) < 1e-12
    assert np.allclose(rankdata([3, 1, 2, 2]), [4, 1, 2.5, 2.5])
    z = rng.normal(size=3000)
    assert abs(partial_spearman(z + rng.normal(size=3000), z + rng.normal(size=3000), z)) < 0.06
    # exact importance sampling: the balance-heuristic SNIS estimate of a NON-proposal target is unbiased up to
    # O(1/R), and scoring two replies on one pool flips their order far less often than independent draws
    V = 30
    s = rng.normal(size=V)
    lg = rng.normal(size=(1, V)) * 1.5 + rng.normal(size=(4, V)) * 0.5
    Pm = np.exp(lg) / np.exp(lg).sum(1, keepdims=True)
    est = []
    for _ in range(300):
        dr = np.concatenate([rng.choice(V, 20, p=Pm[m]) for m in range(3)])
        est.append(float(snis_weights(np.log(Pm[3, dr]), balance_log_q(np.log(Pm[:3, dr])))[0] @ s[dr]))
    assert abs(np.mean(est) - Pm[3] @ s) < 0.05
    fl_pool = fl_ind = 0
    for _ in range(300):
        e = []
        for _ in range(2):
            dr = np.concatenate([rng.choice(V, 20, p=Pm[m]) for m in range(4)])
            lq = balance_log_q(np.log(Pm[:4, dr]))
            e.append([float(snis_weights(np.log(Pm[k, dr]), lq)[0] @ s[dr]) for k in (2, 3)])
        fl_pool += int((e[0][0] - e[0][1]) * (e[1][0] - e[1][1]) < 0)
        ind = [[float(s[rng.choice(V, 20, p=Pm[k])].mean()) for k in (2, 3)] for _ in range(2)]
        fl_ind += int((ind[0][0] - ind[0][1]) * (ind[1][0] - ind[1][1]) < 0)
    assert fl_pool < 0.7 * fl_ind, (fl_pool, fl_ind)
    # pair mining
    pc = CASPIConfig()
    A = np.array([0.10, 0.30, 0.12, 0.40, 0.35])
    Cc = np.array([0.8, 0.8, 0.8, 0.2, 0.8])            # 3 has the best affect but drops the information
    H = np.array([True, True, True, True, False])        # 4 is malformed
    texts = ["a b c", "d e f", "g h i", "j k l", "m n o"]
    pr = mine_pairs(A, Cc, H, np.full(5, 10.0), texts, 0.05, pc, True)
    assert pr[0][:2] == (1, 0) and pr[0][3] == "dominance", pr
    assert all(not (i == 3 and k != "constraint") for i, _, _, k in pr), pr
    assert any(k == "constraint" and j == 4 for _, j, _, k in pr), pr
    assert all(k == "constraint" for *_, k in mine_pairs(A, Cc, H, np.full(5, 10.0), texts, 1.0, pc, True))
    assert not mine_pairs(A, Cc, H, np.full(5, 1.0), texts, 0.05, pc, True), "low-ESS estimates must not form pairs"
    assert not mine_pairs(A, Cc, H, np.full(5, 10.0), ["same words here"] * 5, 0.05, pc, True), "duplicates kept"
    # the certification guarantee: under every null the false-acceptance rate stays <= alpha; a real
    # improvement is accepted most of the time.  Units are dialogues (one context each).
    G, trials, alpha = 300, 600, 0.05
    cl = np.arange(G)
    base = {"monitor": 0.0, "content": 0.0, "hygiene": 0.0, "length": 0.0}

    def draw(shift):
        out = {k: rng.normal(base[k] + shift.get(k, 0.0), 0.06 if k == "monitor" else 0.1, G)
               for k in ("monitor", "content", "hygiene", "length")}
        out["affect"] = rng.normal(shift.get("affect", 0.0), 0.06, G)
        return out
    for bound in ("t", "bernstein"):
        pcb = dataclasses.replace(pc, bound=bound)
        nulls = {"affect_zero": {"affect": 0.0}, "monitor_at_margin": {"affect": 0.02, "monitor": -pc.margin_monitor},
                 "content_at_margin": {"affect": 0.02, "content": -pc.margin_info}}
        for name, sh in nulls.items():
            rate = np.mean([vector_test(draw(sh), cl, alpha, pcb, 0.0, 1e9, 1)["accept"] for _ in range(trials)])
            assert rate <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / trials), (bound, name, rate)
    good = np.mean([vector_test(draw({"affect": 0.03}), cl, alpha, pc, 0.0, 1e9, 1)["accept"] for _ in range(200)])
    assert good > 0.5, f"a real improvement is rarely certified (rate {good:.2f})"
    # information helpers
    assert info_recall("Booked at the Gonville, ref XYZ123.", "Your Gonville booking is done, reference XYZ123.") > 0.3
    assert info_recall("So sorry, glad to help!", "Your Gonville booking is done, reference XYZ123.") == 0.0
    assert abs(krippendorff_alpha_interval({"a": [1, 1], "b": [-1, -1], "c": [0, 0]}) - 1.0) < 1e-12
    print(f"unit tests OK | t quantiles, exact sign-flip, SNIS (mean {np.mean(est):+.3f} vs truth {Pm[3] @ s:+.3f}; "
          f"flips {fl_pool} shared vs {fl_ind} independent), pair mining, certification level and power "
          f"(real gain certified {good:.2f})")


def make_synthetic_emowoz(data_dir: Path, n_dialogues: int, seed: int) -> List[str]:
    """Small EmoWOZ-format corpus for the self-test; returns every text for the tiny tokenizer."""
    rng = np.random.default_rng(seed)
    split = {"train": {"multiwoz": [], "dialmage": []}, "dev": {"multiwoz": [], "dialmage": []},
             "test": {"multiwoz": [], "dialmage": []}}
    data: Dict[str, Dict[str, Any]] = {"multiwoz": {}, "dialmage": {}}
    texts = []
    for i in range(n_dialogues):
        src = "multiwoz" if i % 4 else "dialmage"
        log, mood = [], 0.0
        for _ in range(int(rng.integers(3, 6))):
            emo = 6 if mood > 0.5 else 2 if mood < -0.5 else 0
            lead = {6: "Great, thanks!", 2: "That is really annoying.", 0: ""}[emo]
            log.append({"text": norm_text(f"{lead} I need a hotel for {int(rng.integers(2, 6))} people on monday."),
                        "emotion": [{"emotion": emo}] * 4})
            bad = rng.random() < 0.4
            emp = rng.random() < 0.5
            body = ("There is no hotel available at 10:00." if bad else
                    f"Your hotel is booked, the reference number is AB{int(rng.integers(100, 999))}.")
            log.append({"text": ("I am sorry about that. " if emp else "") + body, "emotion": []})
            mood = 0.5 * mood + (-0.9 if bad else 0.6) + (0.5 if emp else 0.0)
        log.append({"text": "Thanks, that is all." if mood > 0 else "This is not good enough.",
                    "emotion": [{"emotion": 6 if mood > 0 else 2}] * 4})
        did = f"SYN{i:04d}.json"
        data[src][did] = {"log": log}
        split["train" if i % 10 < 6 else "dev" if i % 10 < 8 else "test"][src].append(did)
        texts += [u["text"] for u in log]
    data_dir.mkdir(parents=True, exist_ok=True)
    dump_json(data["multiwoz"], data_dir / "emowoz-multiwoz.json")
    dump_json(data["dialmage"], data_dir / "emowoz-dialmage.json")
    dump_json(split, data_dir / "data-split.json")
    return texts


def build_tiny_models(root: Path, texts: Sequence[str]) -> Dict[str, str]:
    """Tiny randomly initialised models and a word-level tokenizer, saved locally (no downloads)."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import (LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, RobertaConfig,
                              RobertaForSequenceClassification)
    torch.manual_seed(0)
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    for t in list(texts) + [AGENT_SYSTEM, CUSTOMER_SYSTEM, REWRITE_INSTRUCTION,
                            "Conversation so far: Customer: Agent: Draft reply: Improved reply: Write only one two"]:
        for w in re.findall(r"\w+|[^\w\s]", t):
            vocab.setdefault(w, len(vocab))

    def tokenizer(pair_template: bool):
        core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        core.pre_tokenizer = pre_tokenizers.Whitespace()
        core.decoder = decoders.WordPiece(prefix="##")
        if pair_template:
            core.post_processor = processors.TemplateProcessing(
                single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
        return PreTrainedTokenizerFast(tokenizer_object=core, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                       unk_token="<unk>", cls_token="<s>", sep_token="</s>")
    enc = dict(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
               max_position_embeddings=520, pad_token_id=1, bos_token_id=0, eos_token_id=2, type_vocab_size=1)
    labels = {"sentiment": ["negative", "neutral", "positive"], "monitor": ["anger", "neutral", "joy"],
              "external_emotion": ["sadness", "neutral", "joy"], "nli": ["contradiction", "neutral", "entailment"]}
    paths = {}
    for name, labs in labels.items():
        p = root / name
        RobertaForSequenceClassification(RobertaConfig(**enc, num_labels=3, id2label=dict(enumerate(labs)),
                                                       label2id={v: k for k, v in enumerate(labs)})).save_pretrained(p)
        tokenizer(True).save_pretrained(p)
        paths[name] = str(p)
    for name in ("lm", "external_lm"):
        p = root / name
        LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                                     tie_word_embeddings=True, bos_token_id=0, eos_token_id=2,
                                     pad_token_id=1)).save_pretrained(p)
        tokenizer(False).save_pretrained(p)
        paths[name] = str(p)
    return paths


def _fake_annotations(cfg: Config) -> None:
    """Self-test only: two annotators who follow the judge with noise and one who clicks at random."""
    key = load_json(cfg.out_dir / "human_study" / "key_DO_NOT_SHARE.json")
    rng = np.random.default_rng(5)
    rows = []
    for ann in ("a1", "a2", "random_clicker"):
        for iid, k in key.items():
            ans = {}
            for q in HUMAN_QUESTIONS:
                if ann == "random_clicker":
                    ans[q] = str(int(rng.integers(0, 3)))
                elif k["kind"] == "catch":
                    ans[q] = "1" if k["left"] == "gold" else "2"
                else:
                    ans[q] = ("1" if k["left_score"] > k["right_score"] else "2") if rng.random() < 0.7 \
                        else str(int(rng.integers(0, 3)))
            rows.append({"annotator_id": ann, "item_id": iid, **ans})
    with open(cfg.out_dir / "human_study" / "annotations_selftest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["annotator_id", "item_id"] + list(HUMAN_QUESTIONS))
        w.writeheader()
        w.writerows(rows)


def _check_scoring_and_updates(cfg: Config) -> None:
    """Real-code-path checks on the tiny model: (i) tail-logit scoring equals a plain full-logit computation;
    (ii) the base model (adapter disabled) equals the reference policy of a fresh adapter; (iii) a DPO step
    raises the chosen-vs-rejected margin; (iv) adapters survive a save/load round trip."""
    import torch
    lg = logging.getLogger("caspi.check")
    pol = Policy(cfg.base_model, cfg, lg)
    prefixes = ["Customer : I need a hotel .", "Agent : Your hotel is booked ."]
    conts = [pol.text_ids("Thanks , that is all ."), pol.text_ids("This is not good enough .")]
    fast = pol.score(prefixes, conts)
    with torch.no_grad():
        for k, (p, c) in enumerate(zip(prefixes, conts)):
            ids = torch.tensor([pol.prefix_ids(p) + c])
            logp = torch.log_softmax(pol.model(input_ids=ids).logits[0, :-1].float(), -1)
            ref = float(logp[torch.arange(len(pol.prefix_ids(p)) - 1, ids.shape[1] - 1), ids[0, len(pol.prefix_ids(p)):]].sum())
            assert abs(fast[k] - ref) < 1e-3, (fast[k], ref)
    assert np.allclose(pol.score(prefixes, conts, adapter=False), fast, atol=1e-5), "fresh adapter != base model"
    ref_snap = pol.snapshot()
    pairs = [(prefixes[0], "Thanks , that is all .", "This is not good enough .")] * 4
    before = pol.score([prefixes[0]] * 2, [pol.text_ids(pairs[0][1]), pol.text_ids(pairs[0][2])])
    upd = pol.dpo_update(pairs, [1.0] * 4, ref_snap, 0.5, 5e-2, 2, 2, 0.0, 0)
    after = pol.score([prefixes[0]] * 2, [pol.text_ids(pairs[0][1]), pol.text_ids(pairs[0][2])])
    assert (after[0] - after[1]) > (before[0] - before[1]), (before, after)
    assert upd["d_chosen_logp"] - upd["d_rejected_logp"] > 0
    pol.save_adapter(cfg.out_dir / "check_adapter")
    pol2 = Policy(cfg.base_model, cfg, lg)
    pol2.load_adapter(cfg.out_dir / "check_adapter")
    assert np.allclose(pol2.score(prefixes, conts), pol.score(prefixes, conts), atol=1e-5), "adapter round trip"
    print("scoring/update checks OK | tail-logit scoring exact | DPO margin %+.3f -> %+.3f" %
          (before[0] - before[1], after[0] - after[1]))


def run_selftest(root: Path = Path("caspi_selftest")) -> None:
    """The whole pipeline on tiny random local models (CPU, minutes).  Checks plumbing and invariants; the
    numbers it produces mean nothing."""
    run_unit_tests()
    if root.exists():
        shutil.rmtree(root)
    texts = make_synthetic_emowoz(root / "data", 220, 0)
    paths = build_tiny_models(root / "tiny", texts)
    cfg = build_config(parse_args([
        "all", "--device", "cpu", "--data-dir", str(root / "data"), "--out", str(root / "run"),
        "--base-model", paths["lm"], "--external-model", paths["external_lm"], "--sentiment-model", paths["sentiment"],
        "--monitor-model", paths["monitor"], "--external-emotion-model", paths["external_emotion"],
        "--nli-model", paths["nli"], "--seeds", "42", "43", "--sft-epochs", "1", "--sim-sft-epochs", "1",
        "--sft-max-train", "200", "--sim-turns", "200", "--sft-dev-examples", "40", "--sim-sft-dev-examples", "40",
        "--caspi-rounds", "2", "--caspi-contexts-per-round", "10", "--caspi-pool-size", "6",
        "--caspi-guard-pool-size", "6", "--caspi-select-contexts", "30", "--caspi-safety-contexts", "30",
        "--caspi-calib-contexts", "4", "--caspi-epochs", "1", "--validate-contexts", "40", "--eval-turns", "30",
        "--eval-pool-size", "8", "--external-pool-size", "6", "--cross-eval-contexts", "30", "--cross-eval-k", "3",
        "--human-contexts", "20", "--human-validity-pairs", "20", "--n-boot", "200", "--n-signflip", "2000",
        "--sim-max-new-tokens", "8", "--no-load-4bit"]))
    _check_scoring_and_updates(cfg)
    stage_all(cfg)
    # A random tiny judge never lets a round through the guard, so the accept -> certification path is exercised
    # explicitly: the per-round guard decision is forced to "accept", the certification test stays the real one,
    # and the rewrite filter is opened so that rewrites enter the candidate sets.
    real_test = globals()["vector_test"]

    def forced_guard(diffs, cluster, alpha, pc, best_gain, ess, pool):
        out = real_test(diffs, cluster, alpha, pc, best_gain, ess, pool)
        return out if alpha == pc.alpha else {**out, "accept": True}
    globals()["vector_test"] = forced_guard
    try:
        open_cfg = dataclasses.replace(cfg, caspi=dataclasses.replace(cfg.caspi, lr=5e-2, rewrite_min_entail=0.0,
                                                                      rewrite_max_contra=1.0))
        forced = train_arm(open_cfg, PRIMARY, 99)
    finally:
        globals()["vector_test"] = real_test
    assert forced["accepted_rounds"] >= 1 and forced["certification"]["run"], forced["certification"]
    assert forced["returned_sft"] == (not forced["certification"]["accept"])
    assert any(r.get("rewrites_kept", 0) > 0 for r in forced["rounds"]), "no rewrite entered the candidates"
    _fake_annotations(cfg)
    hs = stage_human_analyze(cfg)
    claims = {c["id"]: c for c in stage_claims(cfg)["claims"]}
    rep = load_json(cfg.out_dir / "report_eval.json")
    assert set(rep["arms"]) == set(KNOWN_ARMS), set(KNOWN_ARMS) ^ set(rep["arms"])
    for arm in ARM_FLAGS:
        run = load_json(cfg.out_dir / f"train_{arm}_s42.json")
        assert run["rounds"] and run["flags"] == ARM_FLAGS[arm]
        if ARM_FLAGS[arm]["guard"]:
            assert all("guard" in r for r in run["rounds"]), arm
        for r in run["rounds"]:
            if r.get("guard") and not r["accepted"]:
                assert not all(r["guard"]["checks"].values())
    pools = load_json(cfg.out_dir / "eval_sft_42.json")
    assert all(math.isfinite(r["outcome"]) and math.isfinite(r["monitor"]) for r in pools)
    assert "random_clicker" in hs["excluded_annotators"]
    assert {f"{PRIMARY}_vs_sft:outcome", "H2", "H3c", "H6"} <= set(claims)
    for name in ("preregistration.json", "judge_validity.json", "cross_eval.json", "report_ext.json",
                 "length_analysis.json", "claims.md", "human_study/items.csv"):
        assert (cfg.out_dir / name).exists(), name
    print(f"\nSELFTEST OK | {len(KNOWN_ARMS)} arms trained/evaluated on tiny models | claims: "
          f"{ {k: v['verdict'] for k, v in claims.items()} }")


# ==========================================================================================================
# command line
# ==========================================================================================================

STAGES = {"all": stage_all, "data": stage_data, "sft": stage_sft, "simulator": stage_simulator,
          "validate": stage_validate, "preregister": stage_preregister, "train": stage_train, "eval": stage_eval,
          "eval-external": stage_eval_external, "cross-eval": stage_cross_eval,
          "report": lambda c: (stage_report(c, "eval"), stage_report(c, "ext") if
                               (c.out_dir / f"ext_sft_{c.seeds[0]}.json").exists() else None),
          "length": stage_length, "human-export": stage_human_export, "human-analyze": stage_human_analyze,
          "claims": stage_claims}


def _flags(obj: Any, prefix: str = "") -> List[Tuple[str, str, Any]]:
    """(cli flag, dotted path, default) for every scalar/tuple field of the configuration, recursively, so the
    command line can never drift from the dataclass defaults."""
    out = []
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        path = f"{prefix}{f.name}"
        if dataclasses.is_dataclass(v):
            out += _flags(v, path + ".")
        else:
            out.append(("--" + path.replace(".", "-").replace("_", "-"), path, v))
    return out


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"CASPI ({VERSION})")
    p.add_argument("stage", choices=list(STAGES) + ["unittest", "selftest"])
    p.add_argument("--arm", default=None, help="restrict train to one arm")
    for flag, path, v in _flags(Config()):
        dest = path.replace(".", "__")
        if isinstance(v, bool):
            p.add_argument(flag, dest=dest, action="store_true", default=v)
            p.add_argument("--no-" + flag[2:], dest=dest, action="store_false")
        elif isinstance(v, tuple):
            p.add_argument(flag, dest=dest, nargs="+", type=type(v[0]) if v else str, default=list(v))
        else:
            p.add_argument(flag, dest=dest, type=type(v), default=v)
    return p.parse_args(list(argv))


def build_config(a: argparse.Namespace) -> Config:
    cfg = Config()
    for _, path, v in _flags(cfg):
        val = getattr(a, path.replace(".", "__"))
        obj = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], tuple(val) if isinstance(v, tuple) else val)
    if not cfg.device:
        try:
            import torch
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            cfg.device = "cpu"
    if a.arm:
        cfg.arms = (a.arm,)
    unknown = [x for x in cfg.arms if x not in KNOWN_ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; choose from {KNOWN_ARMS}")
    if cfg.caspi.bound not in ("t", "bernstein"):
        raise ValueError("--caspi-bound must be t or bernstein")
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if a.stage == "unittest":
        run_unit_tests()
    elif a.stage == "selftest":
        run_selftest()
    else:
        STAGES[a.stage](build_config(a))


if __name__ == "__main__":
    main()
