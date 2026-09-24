#!/usr/bin/env python3
"""
PACE -- Phrasing-Affect, Content-Equivalent policy optimisation from implicit customer emotion.

Goal
    Align an LLM customer-service agent with customer satisfaction when customers leave no explicit
    feedback.  Emotion detection and sentiment analysis of what the customer says NEXT stand in for
    the missing feedback.  The agent must deliver the same information (including bad news) and may
    only change HOW it is said.

Why a new algorithm (the short version; the long critique of CARO v12 is in README.md)
    CARO v12 asked a 3B simulator for p(reply | context, response) over a panel of 48 unrelated real
    replies and averaged their sentiment.  That estimator reacts to topic overlap more than to emotion
    (in the log, the length-matched surface null tau=0.206 is about as large as the length tau=0.214),
    it costs 48 LM passes per label (13.5 h for 3000 corpus contexts), its reward carries almost no
    human-anchored signal (gold-anchor rho=+0.06), and nothing stops the policy from dropping bad news
    to please the customer.  PACE removes the simulator, the panel and all post-hoc length patches.

The algorithm (four estimators and one constrained policy-gradient update)

  (1) Emotion/Satisfaction Detection engine (ESD).  A RoBERTa classifier over the 7 EmoWOZ emotions,
      trained on customer utterances ONLY, never on the agent turn.  The outcome is measured without
      looking at the treatment, so the reward model cannot learn the detector's reaction to agent
      wording.  Calibration uses temperature scaling (Guo et al., 2017).  Satisfaction is the expected
      utility S(u) = sum_e p(e|u) U(e) + beta * sentiment(u).  U follows the EmoWOZ valence x elicitor
      taxonomy (Feng et al., 2022): emotions the operator caused count fully, emotions caused by events
      count by half, and apologetic (caused by the user) counts as zero.  beta is chosen on dev
      against human labels.

  (2) Content-orthogonal phrasing-effect reward.  Logs contain one response per context, and the
      response carries both WHAT is said (content c(a): dialogue acts, or lexical information units)
      and HOW it is said.  Content confounds the outcome: bad news lowers satisfaction whatever the
      phrasing.  PACE uses cross-fitted partialling-out (Robinson, 1988; Chernozhukov et al., 2018;
      Nie & Wager, 2021):
            m(h, c)  = E[S | context h, content c]              (nuisance, K-fold cross-fitted)
            g(h, a) ~= E[S - m_hat(h, c(a)) | h, a] = tau(h, a) (effect model, bootstrapped heads)
      For two responses with the same content, tau(h,a1) - tau(h,a2) = E[S|h,a1] - E[S|h,a2], which
      is exactly the phrasing effect.  Because residualisation removes the context/content variance
      that dominates S, the learning problem is easier.  A leaked content change moves tau only by a
      second-order amount (see unit test 2).  M bootstrapped heads (Osband et al., 2016) give an
      epistemic sd, and the reward uses the lower confidence bound (pessimism: Jin et al., 2021).

  (3) Information-fidelity and length constraints.  Completeness P_ent(response => source) and
      consistency 1 - P_contra(source => response) come from an NLI cross-encoder.  Empathetic
      additions are allowed, dropped or contradicted facts are not.  Any number, time or reference
      code that the source does not contain counts as fabrication.  Length has an explicit budget
      instead of post-hoc regressions.

  (4) PACE update, a feasible-set, leave-one-out group-relative advantage with Lagrangian duals:
            F_g        = { i in group g : hygienic, no fabrication, fidelity f_i >= f_min }
            A_aff_i    = (r_i - mean_{j in F_g, j != i} r_j) / s     for i in F_g, else 0
                         (set to 0 when the spread of F_g is inside the ensemble's own sd)
            A_con_i    = -lam_f (v_f,i - mean_{j != i} v_f,j) - lam_len (v_len,i - mean_{j != i} v_len,j)
            A_i        = clip(A_aff_i + A_con_i),   hygiene failures get -bad_penalty
            lam        <- [lam + eta (mean violation - epsilon)]_+   (projected dual ascent)
      The affect advantage exists only inside the content-equivalent set, the only domain on which
      tau differences are phrasing effects (point 2).  A response that drops or changes information
      can never gain affect reward, and the dual pushes the policy back to feasibility (RCPO: Tessler
      et al., 2019; Safe-RLHF: Dai et al., 2023).  Leave-one-out baselines keep the gradient unbiased
      (RLOO: Kool et al., 2019; Ahmadian et al., 2024).  There is no per-group std normalisation, and
      token losses are summed and divided by a CONSTANT (Dr. GRPO: Liu et al., 2025), which removes
      the length and difficulty biases of GRPO's normalisations.  The loss uses the exact sampled
      token ids (v12 re-tokenised post-processed text, so it differentiated tokens that were never
      sampled).  It runs PPO-clip over several epochs per batch, so the clip is active (in v12 the
      ratio was identically 1), with a k3 KL to the frozen base model through adapter disabling
      (no weight copies).

  Evaluation is not circular.  A separate "judge", E[U_human | h, a], is trained on HUMAN labels
  from dialogues that neither the detector nor the reward ever saw.  Its utility is reported jointly
  with fidelity, fabrication and length.  Seeds are pooled per context before the dialogue-clustered
  bootstrap, and Holm correction is applied across arms.

Data: EmoWOZ (Feng et al., 2022), Zenodo record 6506504.  The training dialogues are split three
ways: E trains the detector (human labels), R trains the reward (detector pseudo-labels only; the
human labels are hidden, as in deployment), J trains the judge (human labels).  The valid split is
used for early stopping.  The test split holds the human-anchored validity gates and the policy
evaluation.

Stages: data | detector | reward | judge | train | eval | report | all | unittest | selftest
Typical: python pace.py all --download --models-dir /path/to/hf_cache --out pace_run
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

VERSION = "pace-1.0"
EPS = 1e-12

ZENODO_FILES = {
    "emowoz-multiwoz.json": "https://zenodo.org/records/6506504/files/emowoz-multiwoz.json?download=1",
    "emowoz-dialmage.json": "https://zenodo.org/records/6506504/files/emowoz-dialmage.json?download=1",
    "data-split.json": "https://zenodo.org/records/6506504/files/data-split.json?download=1",
}
SPLIT_ALIASES = {"train": "train", "dev": "valid", "valid": "valid", "validation": "valid", "test": "test"}
EMOTION_NAMES = ("neutral", "fearful", "dissatisfied", "apologetic", "abusive", "excited", "satisfied")
N_EMO = len(EMOTION_NAMES)
# Satisfaction WITH THE AGENT, following EmoWOZ's valence x elicitor x conduct definitions: operator-elicited
# emotions (satisfied, dissatisfied, abusive) count fully, event-elicited ones (excited, fearful) half, and
# user-elicited "apologetic" not at all (a customer apologising for a mistake says nothing about the agent).
EMOTION_UTILITY = np.array([0.0, -0.5, -1.0, 0.0, -1.5, 0.5, 1.0])

# Content-free, affect-neutral filler (used ONLY for a diagnostic of the learnt reward; never for training).
HELDOUT_TAILS = ("That is the information I have on this.", "Those are the details on my side.",
                 "This is what the system currently shows.", "I have noted this on the record.",
                 "That covers the points you raised.", "This is the current status as it stands.")

REWRITE_SYSTEM = ("You are a customer-service agent. You are given the conversation so far and a draft reply that "
                  "contains the information the customer must receive. Write the reply you will actually send: "
                  "keep every fact, number, name, time and reference code of the draft, do not add new facts or "
                  "promises, and phrase it so that it is emotionally appropriate for this customer. Output only "
                  "the reply.")

ROLE_LEAK_RE = re.compile(r"(?im)^\s*(customer|user|agent|system|assistant)\s*:")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
REF_RE = re.compile(r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{5,12}\b")
POSTCODE_RE = re.compile(r"\bcb\s?\d{1,2}\s?\d[a-z]{2}\b", re.I)
SOCIAL_Q_RE = re.compile(r"(anything else|help you with|can i help|may i help|assist you)", re.I)
INFO_PATTERNS = {
    "time": TIME_RE,
    "number": NUM_RE,
    "refcode": REF_RE,
    "price": re.compile(r"(£|\bgbp\b|\bpounds?\b|\bfree\b|\bcheap\b|\bexpensive\b|\bmoderate\b)", re.I),
    "negative_outcome": re.compile(r"\b(no|not|none|unavailable|unable|cannot|can't|couldn't|isn't|aren't|"
                                   r"fully booked|sold out)\b", re.I),
    "confirmed": re.compile(r"\b(booked|booking was successful|confirmed|reserved|reservation)\b", re.I),
    "location": re.compile(r"\b(phone|postcode|post code|address|located|street|road)\b", re.I),
}
DOMAINS = ("train", "hotel", "restaurant", "taxi", "attraction", "hospital", "police", "bus")


# ----------------------------------------------------------------------------------------------------------
# generic utilities and statistics
# ----------------------------------------------------------------------------------------------------------

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
    lg = logging.getLogger(f"pace.{tag}")
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


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def n_words(s: str) -> int:
    return len(str(s).split())


def rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks for ties (identical to scipy.stats.rankdata(method='average'))."""
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
    """Rank partial correlation of x and y given covariate(s) z (Frisch-Waugh on ranks)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    Z = np.asarray(z, float)
    Z = Z[:, None] if Z.ndim == 1 else Z
    ok = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), 1)
    if ok.sum() < 5:
        return float("nan")
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    D = np.column_stack([np.ones(ok.sum())] + [rankdata(Z[ok, j]) for j in range(Z.shape[1])])
    ex = rx - D @ np.linalg.lstsq(D, rx, rcond=None)[0]
    ey = ry - D @ np.linalg.lstsq(D, ry, rcond=None)[0]
    return pearson(ex, ey)


def stratified_partial_spearman(x, y, z, strata) -> float:
    """Rank partial correlation of x and y given z WITHIN strata (stratum fixed effects): ranks are demeaned
    inside each stratum before partialling out z.  Singleton strata carry no within-stratum information."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    s = np.asarray(list(strata))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z, s = x[ok], y[ok], z[ok], s[ok]
    _, inv = np.unique(s, return_inverse=True)
    cnt = np.bincount(inv)
    keep = cnt[inv] >= 2
    if keep.sum() < 5:
        return float("nan")
    inv = np.unique(inv[keep], return_inverse=True)[1]
    cnt = np.bincount(inv).astype(float)
    dm = lambda v: v - (np.bincount(inv, weights=v) / cnt)[inv]           # noqa: E731
    rx, ry, rz = dm(rankdata(x[keep])), dm(rankdata(y[keep])), dm(rankdata(z[keep]))
    den = float(rz @ rz)
    if den > EPS:
        rx, ry = rx - rz * float(rx @ rz) / den, ry - rz * float(ry @ rz) / den
    return pearson(rx, ry)


def cluster_bootstrap_ci(stat_fn: Callable[[np.ndarray], float], cluster: Sequence[Any], n_boot: int = 1000,
                         seed: int = 0, conf: float = 0.95) -> Tuple[float, float, float]:
    """Percentile bootstrap that resamples whole clusters (dialogues)."""
    cl = np.asarray(list(cluster))
    uniq, inv = np.unique(cl, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(uniq.size)]
    point = float(stat_fn(np.arange(cl.size)))
    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = np.concatenate([members[j] for j in rng.integers(0, uniq.size, uniq.size)])
        with np.errstate(all="ignore"):
            draws[b] = stat_fn(idx)
    d = draws[np.isfinite(draws)]
    if d.size < max(20, n_boot // 5):
        return point, float("nan"), float("nan")
    a = (1.0 - conf) / 2.0
    return point, float(np.quantile(d, a)), float(np.quantile(d, 1 - a))


def signflip_pvalue(d: np.ndarray, cluster: Sequence[Any], n_mc: int = 5000, seed: int = 0) -> float:
    """Two-sided randomisation test of a zero mean paired difference, flipping signs of cluster means."""
    d = np.asarray(d, float)
    uniq, inv = np.unique(np.asarray(list(cluster)), return_inverse=True)
    means = np.bincount(inv, weights=d) / np.maximum(np.bincount(inv), 1)
    obs = abs(float(means.mean()))
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=(n_mc, uniq.size))
    null = np.abs((s * means[None, :]).mean(1))
    return float((1 + np.sum(null >= obs - 1e-15)) / (n_mc + 1))


def holm(pvals: Sequence[float]) -> List[float]:
    p = list(pvals)
    order = sorted(range(len(p)), key=lambda i: p[i])
    out, prev = [0.0] * len(p), 0.0
    for k, i in enumerate(order):
        prev = max(prev, min(1.0, (len(p) - k) * p[i]))
        out[i] = prev
    return out


def dialogue_folds(dialogue_ids: Sequence[str], k: int, seed: int) -> np.ndarray:
    """Deterministic dialogue-disjoint fold assignment."""
    return np.asarray([int(hashlib.sha256(f"{seed}|fold|{d}".encode()).hexdigest()[:8], 16) % k
                       for d in dialogue_ids], int)


# ----------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------

@dataclass
class Turn:
    dialogue_id: str
    uid: str
    split: str
    source: str
    turn_index: int
    history: str
    user_text: str
    user_emotion: int
    agent_text: str
    agent_acts: Optional[str]
    next_user_text: Optional[str]
    next_emotion: Optional[int]
    role: str = ""

    @property
    def human_utility(self) -> Optional[float]:
        if self.next_emotion is None or not (0 <= int(self.next_emotion) < N_EMO):
            return None
        return float(EMOTION_UTILITY[int(self.next_emotion)])


def _annotator_emotion(raw: Any) -> int:
    """EmoWOZ stores a list of annotations per user turn; index 3 is the final (aggregated) label."""
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, list):
        if not raw:
            return -1
        pick = raw[3] if len(raw) > 3 else raw[-1]
        if isinstance(pick, dict):
            return int(pick.get("emotion", -1))
        return int(pick) if isinstance(pick, int) else -1
    if isinstance(raw, dict):
        return int(raw.get("emotion", -1))
    return -1


def delex_acts(raw: Any) -> Optional[str]:
    """Delexicalised task content of an agent turn from MultiWOZ-style dialogue acts.

    'general-*' acts (greet, welcome, reqmore, bye) are SOCIAL and therefore part of the phrasing, not of
    the content; they are excluded.  Slot names are kept, slot values dropped: within one context the values
    are fixed by the backend, and what matters is WHICH information is conveyed (e.g. Booking-NoBook)."""
    items = set()
    if isinstance(raw, dict):
        if not raw:
            return None
        for k, v in raw.items():
            if str(k).lower().startswith("general"):
                continue
            slots = sorted({str(s[0]).lower() for s in (v or []) if isinstance(s, (list, tuple)) and s})
            items.add(f"{k}({','.join(slots)})")
    elif isinstance(raw, list):
        if not raw:
            return None
        for a in raw:
            if isinstance(a, (list, tuple)) and len(a) >= 2:
                intent, dom = str(a[0]), str(a[1])
                if dom.lower() == "general" or intent.lower() == "general":
                    continue
                items.add(f"{dom}-{intent}({str(a[2]).lower() if len(a) > 2 else ''})")
    else:
        return None
    return " ".join(sorted(items)) if items else "social_only"


def info_units(text: str) -> List[str]:
    """Lexical information units (fallback content descriptor when no dialogue acts exist).  Affective and
    politeness lexemes are deliberately NOT content."""
    s = norm_text(text)
    tags = [k for k, rx in INFO_PATTERNS.items() if rx.search(s)]
    if "?" in s and not SOCIAL_Q_RE.search(s):
        tags.append("request")
    low = s.lower()
    tags += [f"domain={d}" for d in DOMAINS if re.search(rf"\b{d}s?\b", low)]
    return sorted(set(tags))


def content_descriptor(t: Turn) -> str:
    if t.agent_acts:
        return f"acts: {t.agent_acts}"
    return "info: " + (" ".join(info_units(t.agent_text)) or "none")


def is_bad_news(t: Turn) -> bool:
    d = content_descriptor(t).lower()
    return any(k in d for k in ("nobook", "nooffer", "negative_outcome"))


def entities(text: str) -> set:
    s = norm_text(text)
    out = {m.group(0).lower() for m in TIME_RE.finditer(s)}
    out |= {m.group(0).lower() for m in NUM_RE.finditer(s)}
    out |= {m.group(0).lower() for m in REF_RE.finditer(s)}
    out |= {re.sub(r"\s", "", m.group(0).lower()) for m in POSTCODE_RE.finditer(s)}
    return out


def fabricated_entities(src: str, resp: str) -> set:
    """Numbers, times, reference codes and postcodes stated in the response but absent from the source."""
    return entities(resp) - entities(src)


def hygiene_ok(text: str, min_words: int = 3, max_words: int = 120, require_terminal: bool = True
               ) -> Tuple[bool, List[str]]:
    s = norm_text(text)
    w = s.split()
    reasons = []
    if len(w) < min_words:
        reasons.append("too_short")
    if len(w) > max_words:
        reasons.append("too_long")
    if ROLE_LEAK_RE.search(s):
        reasons.append("role_leak")
    if w:
        run = best = 1
        for a, b in zip(w, w[1:]):
            run = run + 1 if a.lower() == b.lower() else 1
            best = max(best, run)
        if best >= 5:
            reasons.append("repetition")
        tri = [tuple(x.lower() for x in w[i:i + 3]) for i in range(max(0, len(w) - 2))]
        if tri and 1.0 - len(set(tri)) / len(tri) > 0.5:
            reasons.append("trigram_loop")
    if require_terminal and not re.search(r"[.!?]\s*[\"')\]]?\s*$", s):
        reasons.append("unterminated")
    return len(reasons) == 0, reasons


def context_text(t: Turn, n_utts: int = 4) -> str:
    lines = [ln for ln in t.history.split("\n") if ln.strip()]
    keep = lines[-(n_utts - 1):] if n_utts > 1 else []
    return "\n".join(keep + [f"Customer: {t.user_text}"])


def download_emowoz(data_dir: Path, logger: logging.Logger) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in ZENODO_FILES.items():
        dst = data_dir / name
        if dst.exists() and dst.stat().st_size > 1000:
            continue
        logger.info("downloading %s", name)
        with urllib.request.urlopen(url, timeout=600) as r, open(dst, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)


def load_emowoz(data_dir: Path, logger: logging.Logger, max_history: int = 6) -> List[Turn]:
    paths = {"multiwoz": data_dir / "emowoz-multiwoz.json", "dialmage": data_dir / "emowoz-dialmage.json"}
    sp = data_dir / "data-split.json"
    for p in list(paths.values()) + [sp]:
        if not p.exists():
            raise FileNotFoundError(f"missing {p}; run with --download or place the Zenodo files in {data_dir}")
    split_of: Dict[str, str] = {}
    for key, node in load_json(sp).items():
        canon = SPLIT_ALIASES.get(str(key).lower())
        if canon is None:
            continue
        ids = [i for sub in node.values() for i in sub] if isinstance(node, dict) else list(node)
        for i in ids:
            split_of[str(i)] = canon
    turns: List[Turn] = []
    n_acts = 0
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
                raw_acts = log[i + 1].get("dialog_act", log[i + 1].get("dialogue_acts"))
                acts = delex_acts(raw_acts)
                n_acts += acts is not None
                nxt = texts[i + 2] if i + 2 < len(log) and texts[i + 2] else None
                nemo = emos[i + 2] if i + 2 < len(log) else None
                turns.append(Turn(str(did), f"{did}#{i}", split, src, i, "\n".join(hist), texts[i], int(emos[i]),
                                  texts[i + 1], acts, nxt, int(nemo) if nemo is not None else None))
    if not turns:
        raise RuntimeError("no turns parsed; check the EmoWOZ JSON schema")
    lab = [t.next_emotion for t in turns if t.human_utility is not None]
    dist = np.bincount(np.asarray(lab, int), minlength=N_EMO) / max(1, len(lab))
    logger.info("EmoWOZ | %d turns, %d dialogues | %.1f%% agent turns carry dialogue acts | next-emotion "
                "distribution %s", len(turns), len({t.dialogue_id for t in turns}), 100 * n_acts / len(turns),
                {EMOTION_NAMES[k]: round(float(v), 4) for k, v in enumerate(dist)})
    return turns


def assign_roles(turns: Sequence[Turn], fracs: Tuple[float, float, float], seed: int) -> None:
    """Three-way dialogue-disjoint partition of TRAIN dialogues: E (detector), R (reward), J (judge)."""
    cum = np.cumsum(np.asarray(fracs, float) / sum(fracs))
    for t in turns:
        if t.split != "train":
            t.role = t.split
            continue
        u = int(hashlib.sha256(f"{seed}|role|{t.dialogue_id}".encode()).hexdigest()[:8], 16) / 2 ** 32
        t.role = "ERJ"[int(np.searchsorted(cum, u, side="right"))] if u < cum[-1] else "J"


def filter_turns(turns: Sequence[Turn], pred: Callable[[Turn], bool], limit: Optional[int] = None,
                 seed: int = 0) -> List[Turn]:
    """Filter, then (if limit) take whole dialogues in a seeded random order."""
    out = [t for t in turns if pred(t)]
    if limit is not None and len(out) > limit:
        by: Dict[str, List[Turn]] = {}
        for t in out:
            by.setdefault(t.dialogue_id, []).append(t)
        keys = sorted(by)
        random.Random(seed).shuffle(keys)
        sel: List[Turn] = []
        for k in keys:
            if len(sel) >= limit:
                break
            sel.extend(by[k])
        out = sel[:limit]
    return out


# ----------------------------------------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------------------------------------

@dataclass
class FitConfig:
    lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 2
    batch: int = 32
    max_len: int = 256
    warmup: float = 0.06
    evals_per_epoch: int = 3
    patience: int = 3
    max_train: Optional[int] = None
    max_dev: int = 3000
    pred_batch: int = 128


@dataclass
class PACEConfig:
    steps: int = 300
    n_contexts: int = 8
    group_size: int = 8
    lr: float = 1e-5
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 160         # room for (1 + len_slack) x a 50-word source; also the Dr. GRPO constant
    gen_batch: int = 64
    ppo_epochs: int = 2
    minibatch: int = 32
    micro: int = 8
    clip: float = 0.2
    max_grad_norm: float = 1.0
    kl_coef: float = 0.05
    kl_target: float = 0.05          # per-token k3 KL to the frozen base model
    kl_coef_min: float = 0.005
    kl_coef_max: float = 1.0
    kl_abort: float = 20.0           # abort when KL exceeds this multiple of the target
    kappa: float = 1.0               # pessimism: reward = mean - kappa * ensemble sd
    snr_kappa: float = 1.0           # affect advantage only if the group spread exceeds this x ensemble sd
    adv_clip: float = 5.0
    scale_floor: float = 1e-3
    bad_penalty: float = 1.0         # fixed advantage for hygiene failures (outside all group statistics)
    f_min: float = 0.5               # fidelity needed to enter the feasible (content-equivalent) set
    eps_fid: float = 0.15            # constraint: mean fidelity violation <= eps_fid
    len_slack: float = 0.6           # free length budget: up to (1 + slack) x source words
    eps_len: float = 0.03            # constraint: mean log-length overrun <= eps_len
    dual_lr: float = 1.0
    lam_f0: float = 1.0
    lam_len0: float = 1.0
    lam_max: float = 20.0
    log_every: int = 10
    min_words: int = 3
    max_words: int = 120
    require_terminal: bool = True
    allow_truncated: bool = False


@dataclass
class Config:
    data_dir: Path = Path("emowoz_data")
    out: Path = Path("pace_run")
    device: str = "cuda"
    models_dir: Optional[str] = None
    download: bool = False
    seed: int = 42
    seeds: Tuple[int, ...] = (42, 43, 44)
    strict: bool = True
    encoder_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    policy_model: str = "Qwen/Qwen2.5-3B-Instruct"
    load_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    role_fracs: Tuple[float, float, float] = (0.30, 0.45, 0.25)
    ctx_utts: int = 4
    label_source: str = "detector"   # "detector" (deployment-faithful) or "human" (upper bound)
    n_heads: int = 5
    crossfit_folds: int = 2
    fit_naive_reward: bool = True
    sent_beta_grid: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)
    gate_max_turns: int = 4000
    n_boot: int = 1000
    fit: FitConfig = field(default_factory=FitConfig)
    pace: PACEConfig = field(default_factory=PACEConfig)
    rl_contexts: int = 20000
    eval_turns: int = 800
    eval_temperature: float = 0.7
    arms: Tuple[str, ...] = ("pace", "pace_unconstrained", "pace_naive_reward", "sentiment_only")
    fidelity_ni_margin: float = 0.03


# ----------------------------------------------------------------------------------------------------------
# neural components (torch)
# ----------------------------------------------------------------------------------------------------------

def resolve_local_model(name: str, models_dir: Optional[str]) -> str:
    if models_dir:
        root = Path(models_dir) / ("models--" + name.replace("/", "--")) / "snapshots"
        if root.exists():
            snaps = sorted(p for p in root.iterdir() if p.is_dir())
            if snaps:
                return str(snaps[-1])
    return name


def load_tokenizer(name: str, cfg: Config, padding_side: str = "right", truncation_side: str = "left"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(resolve_local_model(name, cfg.models_dir))
    tok.padding_side = padding_side
    tok.truncation_side = truncation_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def encode_texts(tok, a: Sequence[str], b: Optional[Sequence[str]], max_len: int):
    if b is None:
        return tok(list(a), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    return tok(list(a), list(b), truncation="longest_first", max_length=max_len, padding=True, return_tensors="pt")


def _autocast(device: str):
    import torch
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def build_text_model(backbone_name: str, n_out: int, cfg: Config, pretrained: bool = True):
    """Encoder backbone + linear head on the first (<s>/[CLS]) token.  First-token pooling, unlike the mean
    pooling over prompt+response tokens in v12, has no built-in length channel: a mean over n_p prompt and
    n_r response tokens weights the response by n_r / (n_p + n_r), so length leaks into every feature."""
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel
    path = resolve_local_model(backbone_name, cfg.models_dir)
    backbone = AutoModel.from_pretrained(path) if pretrained else AutoModel.from_config(AutoConfig.from_pretrained(path))

    class TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.drop = nn.Dropout(0.1)
            self.head = nn.Linear(backbone.config.hidden_size, n_out)

        def forward(self, input_ids, attention_mask):
            h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            return self.head(self.drop(h))

    return TextModel()


def free_cuda() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def save_text_model(model, path: Path, meta: Dict[str, Any]) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "meta": meta}, path)


def load_text_model(path: Path, cfg: Config, device: str):
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    meta = blob["meta"]
    model = build_text_model(meta["backbone"], int(meta["n_out"]), cfg, pretrained=False)
    model.load_state_dict(blob["state"])
    model.to(device).eval()
    return model, meta


def predict_text_model(model, tok, a: Sequence[str], b: Optional[Sequence[str]], max_len: int, device: str,
                       batch: int = 128) -> np.ndarray:
    import torch
    model.eval()
    n = len(a)
    if n == 0:
        return np.zeros((0, model.head.out_features))
    lens = np.asarray([len(x) + (len(b[i]) if b is not None else 0) for i, x in enumerate(a)])
    order = np.argsort(lens)
    out = np.zeros((n, model.head.out_features), np.float64)
    with torch.no_grad():
        for s in range(0, n, batch):
            ix = order[s:s + batch]
            enc = encode_texts(tok, [a[i] for i in ix], None if b is None else [b[i] for i in ix], max_len)
            with _autocast(device):
                o = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            out[ix] = o.float().cpu().numpy()
    return out


def fit_text_model(model, tok, train: Dict[str, Any], dev: Dict[str, Any], kind: str, fc: FitConfig, device: str,
                   logger: logging.Logger, tag: str, seed: int) -> Dict[str, Any]:
    """Fine-tune encoder + head with early stopping on dev.

    kind='ce'  : y int class labels, dev metric = -NLL.
    kind='mse' : y float targets (standardised internally), W (n, n_out) bootstrap weights per head,
                 dev metric = -MSE of the head-mean prediction."""
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    torch.manual_seed(seed)
    model.to(device)
    a, b, y = list(train["a"]), train.get("b"), np.asarray(train["y"])
    n = len(a)
    y_mu, y_sd = 0.0, 1.0
    if kind == "mse":
        y_mu, y_sd = float(np.mean(y)), float(np.std(y)) or 1.0
    W = np.asarray(train["w"], np.float32) if kind == "mse" else None
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    body_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = AdamW([{"params": body_params, "lr": fc.lr}, {"params": head_params, "lr": fc.head_lr}],
                weight_decay=fc.weight_decay)
    steps_per_epoch = max(1, math.ceil(n / fc.batch))
    total = steps_per_epoch * fc.epochs
    warm = max(1, int(fc.warmup * total))
    sched = LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    every = max(1, steps_per_epoch // max(1, fc.evals_per_epoch))
    rng = np.random.default_rng(seed)
    da, db, dy = list(dev["a"]), dev.get("b"), np.asarray(dev["y"])

    def dev_metric() -> float:
        out = predict_text_model(model, tok, da, db, fc.max_len, device, fc.pred_batch)
        if kind == "ce":
            z = out - out.max(1, keepdims=True)
            lse = np.log(np.exp(z).sum(1))
            return -float(np.mean(lse - z[np.arange(len(dy)), dy.astype(int)]))
        return -float(np.mean((out.mean(1) - (dy - y_mu) / y_sd) ** 2))

    best, best_state, stale, step, hist = -float("inf"), None, 0, 0, []
    stop = False
    t0 = time.time()
    for ep in range(fc.epochs):
        model.train()
        order = rng.permutation(n)
        for s in range(0, n, fc.batch):
            ix = order[s:s + fc.batch]
            enc = encode_texts(tok, [a[i] for i in ix], None if b is None else [b[i] for i in ix], fc.max_len)
            with _autocast(device):
                out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            out = out.float()
            if kind == "ce":
                loss = F.cross_entropy(out, torch.as_tensor(y[ix].astype(np.int64), device=out.device))
            else:
                yt = torch.as_tensor(((y[ix] - y_mu) / y_sd).astype(np.float32), device=out.device)
                w = torch.as_tensor(W[ix], device=out.device)
                loss = (w * (out - yt[:, None]) ** 2).sum() / w.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % every == 0 or step == total:
                m = dev_metric()
                hist.append({"step": step, "loss": float(loss.detach()), "dev": m})
                mark = ""
                if m > best + 1e-5:
                    best, stale, mark = m, 0, " <- best"
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
                logger.info("%s | step %d/%d | loss %.4f | dev %s %.5f | %.0fs%s", tag, step, total,
                            float(loss.detach()), "-NLL" if kind == "ce" else "-MSE(std)", m, time.time() - t0, mark)
                model.train()
                if stale >= fc.patience:
                    stop = True
                    break
        if stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"best_dev": best, "y_mu": y_mu, "y_sd": y_sd, "steps": step, "history": hist}


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): minimise dev NLL over a single scalar T."""
    y = np.asarray(y, int)

    def nll(T):
        z = logits / T
        z = z - z.max(1, keepdims=True)
        return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(y)), y]))

    grid = np.exp(np.linspace(math.log(0.2), math.log(10.0), 80))
    T = float(grid[int(np.argmin([nll(t) for t in grid]))])
    lo, hi = T / 1.1, T * 1.1
    for _ in range(40):                                   # golden-section refinement on the bracket
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
    return float(0.5 * (lo + hi))


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class SentimentEngine:
    """Off-the-shelf polarity model: returns p(positive) - p(negative)."""

    def __init__(self, cfg: Config, device: str, logger: logging.Logger):
        from transformers import AutoModelForSequenceClassification
        path = resolve_local_model(cfg.sentiment_model, cfg.models_dir)
        self.tok = load_tokenizer(cfg.sentiment_model, cfg, truncation_side="right")
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
        self.device = device
        lab = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.pos = next((i for i, v in lab.items() if v.startswith("pos")), max(lab))
        self.neg = next((i for i, v in lab.items() if v.startswith("neg")), min(lab))
        logger.info("sentiment engine | %s | labels %s", cfg.sentiment_model, lab)

    def __call__(self, texts: Sequence[str], batch: int = 128) -> np.ndarray:
        import torch
        out = np.zeros(len(texts))
        with torch.no_grad():
            for s in range(0, len(texts), batch):
                enc = encode_texts(self.tok, [norm_text(t) or "." for t in texts[s:s + batch]], None, 128)
                with _autocast(self.device):
                    lg = self.model(input_ids=enc["input_ids"].to(self.device),
                                    attention_mask=enc["attention_mask"].to(self.device)).logits
                p = torch.softmax(lg.float(), -1).cpu().numpy()
                out[s:s + len(p)] = p[:, self.pos] - p[:, self.neg]
        return out


class NLIEngine:
    """Information fidelity from an NLI cross-encoder.

      completeness = P(entail | premise=response, hypothesis=source)   every source fact is conveyed
      consistency  = 1 - P(contradict | premise=source, hypothesis=response)
      fidelity     = completeness * consistency
    Unlike bidirectional entailment (semantic equivalence; Kuhn et al., 2023), this allows added empathy
    ("I am sorry to hear that") while forbidding dropped or contradicted information."""

    def __init__(self, cfg: Config, device: str, logger: logging.Logger):
        from transformers import AutoModelForSequenceClassification
        path = resolve_local_model(cfg.nli_model, cfg.models_dir)
        self.tok = load_tokenizer(cfg.nli_model, cfg, truncation_side="right")
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
        self.device = device
        lab = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.ent = next(i for i, v in lab.items() if v.startswith("entail"))
        self.con = next(i for i, v in lab.items() if v.startswith("contra"))
        logger.info("NLI engine | %s | labels %s", cfg.nli_model, lab)

    def _probs(self, prem: Sequence[str], hyp: Sequence[str], batch: int = 64) -> np.ndarray:
        import torch
        out = np.zeros((len(prem), self.model.config.num_labels))
        with torch.no_grad():
            for s in range(0, len(prem), batch):
                enc = encode_texts(self.tok, prem[s:s + batch], hyp[s:s + batch], 320)
                with _autocast(self.device):
                    lg = self.model(input_ids=enc["input_ids"].to(self.device),
                                    attention_mask=enc["attention_mask"].to(self.device)).logits
                out[s:s + lg.shape[0]] = torch.softmax(lg.float(), -1).cpu().numpy()
        return out

    def fidelity(self, src: Sequence[str], resp: Sequence[str]) -> Dict[str, np.ndarray]:
        src = [norm_text(x) or "." for x in src]
        resp = [norm_text(x) or "." for x in resp]
        p_complete = self._probs(resp, src)[:, self.ent]
        p_contra = self._probs(src, resp)[:, self.con]
        return {"f": p_complete * (1.0 - p_contra), "complete": p_complete, "contra": p_contra}


class AffectReward:
    """Bootstrapped-head ensemble g(h, a) in outcome units: returns (head mean, head sd)."""

    def __init__(self, path: Path, cfg: Config, device: str):
        self.model, meta = load_text_model(path, cfg, device)
        self.tok = load_tokenizer(meta["backbone"], cfg)
        self.y_mu, self.y_sd = float(meta["y_mu"]), float(meta["y_sd"])
        self.cfg, self.device = cfg, device

    def __call__(self, turns: Sequence[Turn], responses: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        out = predict_text_model(self.model, self.tok, [context_text(t, self.cfg.ctx_utts) for t in turns],
                                 [norm_text(r) for r in responses], self.cfg.fit.max_len, self.device,
                                 self.cfg.fit.pred_batch)
        out = out * self.y_sd + self.y_mu
        return out.mean(1), (out.std(1) if out.shape[1] > 1 else np.zeros(len(out)))


class Judge:
    """Independent evaluator E[U_human | h, a]: trained on HUMAN next-emotion labels of dialogues disjoint from
    everything the detector and the reward were trained on."""

    def __init__(self, path: Path, cfg: Config, device: str):
        self.model, meta = load_text_model(path, cfg, device)
        self.tok = load_tokenizer(meta["backbone"], cfg)
        self.T = float(meta["temperature"])
        self.cfg, self.device = cfg, device

    def __call__(self, turns: Sequence[Turn], responses: Sequence[str]) -> np.ndarray:
        lg = predict_text_model(self.model, self.tok, [context_text(t, self.cfg.ctx_utts) for t in turns],
                                [norm_text(r) for r in responses], self.cfg.fit.max_len, self.device,
                                self.cfg.fit.pred_batch)
        return softmax_np(lg / self.T) @ EMOTION_UTILITY


# ----------------------------------------------------------------------------------------------------------
# PACE core (numpy; unit-tested)
# ----------------------------------------------------------------------------------------------------------

def _loo_centre(x: np.ndarray) -> np.ndarray:
    """x_i - mean_{j != i} x_j = n/(n-1) (x_i - mean x): the leave-one-out (RLOO) baseline."""
    n = x.size
    return np.zeros_like(x) if n < 2 else (x - x.mean()) * n / (n - 1.0)


def pace_advantages(r: np.ndarray, r_sd: np.ndarray, gid: np.ndarray, hygienic: np.ndarray, feasible: np.ndarray,
                    v_f: np.ndarray, v_len: np.ndarray, lam_f: float, lam_len: float, r_scale: float,
                    snr_kappa: float = 1.0, adv_clip: float = 5.0, bad_penalty: float = 1.0) -> Dict[str, Any]:
    """Feasible-set leave-one-out advantages with Lagrangian constraint terms (see module docstring)."""
    r, r_sd = np.asarray(r, float), np.asarray(r_sd, float)
    gid = np.asarray(gid)
    hyg, feas = np.asarray(hygienic, bool), np.asarray(feasible, bool) & np.asarray(hygienic, bool)
    v_f, v_len = np.asarray(v_f, float), np.asarray(v_len, float)
    a_aff = np.zeros_like(r)
    a_con = np.zeros_like(r)
    n_groups = n_gated = n_affect = 0
    spreads = []
    for g in np.unique(gid):
        m = np.flatnonzero(gid == g)
        n_groups += 1
        h = m[hyg[m]]
        if h.size >= 2:                                  # constraint terms among hygienic samples
            a_con[h] = -(lam_f * _loo_centre(v_f[h]) + lam_len * _loo_centre(v_len[h]))
        f = m[feas[m]]
        if f.size < 2:
            continue
        rf = r[f]
        spread = float(rf.max() - rf.min())
        spreads.append(float(rf.std()))
        if snr_kappa > 0 and spread <= snr_kappa * float(np.mean(r_sd[f])):
            n_gated += 1                                 # spread inside the model's own uncertainty
            continue
        n_affect += 1
        a_aff[f] = _loo_centre(rf) / max(float(r_scale), 1e-8)
    adv = np.clip(a_aff + a_con, -adv_clip, adv_clip)
    adv[~hyg] = -abs(bad_penalty)
    return {"adv": adv, "a_aff": a_aff, "a_con": a_con, "n_groups": n_groups, "n_gated": n_gated,
            "n_affect_groups": n_affect, "within_sd": float(np.mean(spreads)) if spreads else float("nan")}


@dataclass
class DualVariable:
    """Projected dual ascent on a mean-violation constraint E[v] <= target."""
    target: float
    lr: float
    lam: float
    lam_max: float = 20.0
    frozen: bool = False

    def update(self, mean_violation: float) -> float:
        if not self.frozen and math.isfinite(mean_violation):
            self.lam = float(np.clip(self.lam + self.lr * (mean_violation - self.target), 0.0, self.lam_max))
        return self.lam


def length_overrun(src: Sequence[str], resp: Sequence[str], slack: float) -> np.ndarray:
    ws = np.asarray([n_words(x) for x in src], float)
    wr = np.asarray([n_words(x) for x in resp], float)
    return np.maximum(0.0, np.log((wr + 1.0) / (ws + 1.0)) - math.log1p(slack))


# ----------------------------------------------------------------------------------------------------------
# policy
# ----------------------------------------------------------------------------------------------------------

@dataclass
class Sample:
    prompt_ids: List[int]
    resp_ids: List[int]
    text: str
    truncated: bool


class PolicyLM:
    def __init__(self, cfg: Config, logger: logging.Logger, device: str):
        import torch
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        self.cfg, self.logger, self.device = cfg, logger, device
        path = resolve_local_model(cfg.policy_model, cfg.models_dir)
        self.tok = load_tokenizer(cfg.policy_model, cfg, padding_side="left")
        cuda = str(device).startswith("cuda")
        kw: Dict[str, Any] = {"dtype": torch.bfloat16 if cuda else torch.float32}
        if cfg.load_4bit and cuda:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16,
                                                           bnb_4bit_use_double_quant=True)
            kw["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(path, **kw)
        if cfg.load_4bit and cuda:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                    gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model = model.to(device)
            if cuda:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.enable_input_require_grads()
        # LoRA dropout 0: the policy, its "old" copy and the reference are then scored by the SAME function,
        # so train-mode forwards (needed for gradient checkpointing) introduce no mask noise into ratios/KL.
        self.model = get_peft_model(model, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        eos = set()
        for x in (self.tok.eos_token_id, getattr(self.model.generation_config, "eos_token_id", None)):
            eos |= set(x) if isinstance(x, (list, tuple)) else ({x} if x is not None else set())
        self.eos_ids = sorted(eos)
        self._tail_kw: Optional[str] = None
        tr = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("policy %s | LoRA r=%d | trainable %.2fM | 4bit=%s | eos ids %s", cfg.policy_model,
                    cfg.lora_r, tr / 1e6, bool(cfg.load_4bit and cuda), self.eos_ids)

    def save_adapter(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))

    def load_adapter(self, path: Path) -> None:
        from peft import set_peft_model_state_dict
        try:
            from safetensors.torch import load_file
            sd = load_file(str(Path(path) / "adapter_model.safetensors"))
        except (ImportError, FileNotFoundError, OSError):
            import torch
            sd = torch.load(str(Path(path) / "adapter_model.bin"), map_location="cpu")
        set_peft_model_state_dict(self.model, sd)
        self.logger.info("loaded adapter %s", path)

    def prompt(self, t: Turn) -> str:
        user = f"Conversation:\n{t.history}\nCustomer: {t.user_text}\n\nDraft reply:\n{t.agent_text}".strip()
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template([{"role": "system", "content": REWRITE_SYSTEM},
                                                 {"role": "user", "content": user}],
                                                tokenize=False, add_generation_prompt=True)
        return f"{REWRITE_SYSTEM}\n{user}\nReply:"

    def generate(self, prompts: Sequence[str], n: int, temperature: float, top_p: float, max_new_tokens: int,
                 seed: int, gen_batch: int = 64) -> List[Sample]:
        """n samples per prompt, group-major order.  The exact sampled token ids are kept: the policy gradient
        is taken on them, never on re-tokenised, post-processed text."""
        import torch
        self.model.eval()
        out: List[Sample] = []
        per = max(1, gen_batch // max(1, n))
        for s in range(0, len(prompts), per):
            chunk = list(prompts[s:s + per])
            ids = [self.tok(p, add_special_tokens=False)["input_ids"] for p in chunk]
            P = max(len(x) for x in ids)
            X = torch.full((len(ids), P), self.tok.pad_token_id, dtype=torch.long)
            M = torch.zeros((len(ids), P), dtype=torch.long)
            for i, x in enumerate(ids):
                X[i, P - len(x):] = torch.tensor(x)
                M[i, P - len(x):] = 1
            torch.manual_seed(seed + s)
            with torch.no_grad():
                y = self.model.generate(input_ids=X.to(self.device), attention_mask=M.to(self.device),
                                        do_sample=True, temperature=temperature, top_p=top_p,
                                        max_new_tokens=max_new_tokens, num_return_sequences=n,
                                        pad_token_id=self.tok.pad_token_id, eos_token_id=self.eos_ids,
                                        use_cache=True)
            gen = y[:, P:].cpu().tolist()
            for k, row in enumerate(gen):
                cut = next((j for j, tkn in enumerate(row) if tkn in self.eos_ids), None)
                resp = row[:cut + 1] if cut is not None else row
                if not resp:
                    resp = [self.eos_ids[0]]
                body = resp[:-1] if (cut is not None) else resp
                text = norm_text(self.tok.decode(body, skip_special_tokens=True))
                out.append(Sample(ids[k // n], resp, text, cut is None))
        return out

    def _forward_tail(self, X, M, pos, keep: int):
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

    def token_logprobs(self, samples: Sequence[Sample], ref: bool = False, grad: bool = False):
        """Per-token log-probs of the sampled response tokens, right-aligned: returns (lp [b,T], mask [b,T]).
        Left padding puts every response at the end of its row, so only the last T+1 logits are needed."""
        import torch
        import torch.nn.functional as F
        seqs = [s.prompt_ids + s.resp_ids for s in samples]
        n = max(len(x) for x in seqs)
        T = max(len(s.resp_ids) for s in samples)
        X = torch.full((len(seqs), n), self.tok.pad_token_id, dtype=torch.long)
        M = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, x in enumerate(seqs):
            X[i, n - len(x):] = torch.tensor(x)
            M[i, n - len(x):] = 1
        pos = (M.cumsum(1) - 1).clamp_min(0)
        X, M, pos = X.to(self.device), M.to(self.device), pos.to(self.device)
        ctx = self.model.disable_adapter() if ref else contextlib.nullcontext()
        with ctx, torch.set_grad_enabled(grad), _autocast(self.device):
            lg = self._forward_tail(X, M, pos, T + 1)[:, :-1]
        tgt = X[:, n - T:]
        lp = -F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape)
        lens = torch.tensor([len(s.resp_ids) for s in samples], device=lp.device)
        mask = (torch.arange(T, device=lp.device)[None, :] >= (T - lens)[:, None]).float()
        return lp * mask, mask

    def batched_logprobs(self, samples: Sequence[Sample], ref: bool, micro: int) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        self.model.eval()
        for s in range(0, len(samples), micro):
            lp, mask = self.token_logprobs(samples[s:s + micro], ref=ref, grad=False)
            lp, mask = lp.cpu().numpy(), mask.cpu().numpy()
            for i in range(lp.shape[0]):
                out.append(lp[i][mask[i] > 0].astype(np.float32))
        return out

    def ppo_update(self, samples: Sequence[Sample], adv: np.ndarray, old: List[np.ndarray], ref: List[np.ndarray],
                   opt, pc: PACEConfig, kl_coef: float, rng: np.random.Generator) -> Dict[str, float]:
        """PPO-clip with a k3 KL penalty; token terms are SUMMED per sequence and divided by the constant
        max_new_tokens (Dr. GRPO), so no length-dependent weighting is introduced."""
        import torch
        N = len(samples)
        L = float(pc.max_new_tokens)
        stats = {"kl": 0.0, "clipfrac": 0.0, "tokens": 0.0, "pg": 0.0}
        self.model.train()
        for _ in range(pc.ppo_epochs):
            perm = rng.permutation(N)
            for s in range(0, N, pc.minibatch):
                mb = perm[s:s + pc.minibatch]
                opt.zero_grad(set_to_none=True)
                for u in range(0, len(mb), pc.micro):
                    ix = mb[u:u + pc.micro]
                    sub = [samples[i] for i in ix]
                    lp, mask = self.token_logprobs(sub, ref=False, grad=True)
                    Tm = lp.shape[1]
                    old_t = torch.zeros_like(lp)
                    ref_t = torch.zeros_like(lp)
                    for r, i in enumerate(ix):
                        k = len(old[i])
                        old_t[r, Tm - k:] = torch.as_tensor(old[i], device=lp.device)
                        ref_t[r, Tm - k:] = torch.as_tensor(ref[i], device=lp.device)
                    A = torch.as_tensor(adv[ix], dtype=torch.float32, device=lp.device)[:, None]
                    ratio = torch.exp((lp - old_t) * mask)
                    pg = -torch.min(ratio * A, torch.clamp(ratio, 1 - pc.clip, 1 + pc.clip) * A)
                    d = (ref_t - lp) * mask
                    kl = torch.exp(d) - d - 1.0
                    loss = ((pg + kl_coef * kl) * mask).sum() / (len(mb) * L)
                    loss.backward()
                    stats["kl"] += float((kl * mask).sum().detach())
                    stats["pg"] += float((pg * mask).sum().detach())
                    stats["clipfrac"] += float(((torch.abs(ratio - 1) > pc.clip).float() * mask).sum().detach())
                    stats["tokens"] += float(mask.sum().detach())
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                               pc.max_grad_norm)
                opt.step()
        tok = max(1.0, stats["tokens"])
        return {"kl": stats["kl"] / tok, "clipfrac": stats["clipfrac"] / tok, "pg": stats["pg"] / tok}


# ----------------------------------------------------------------------------------------------------------
# experiment scaffolding
# ----------------------------------------------------------------------------------------------------------

class Experiment:
    def __init__(self, cfg: Config, tag: str):
        self.cfg = cfg
        cfg.out.mkdir(parents=True, exist_ok=True)
        self.logger = make_logger(cfg.out, tag)
        seed_everything(cfg.seed)
        self.logger.info("PACE %s | stage=%s | device=%s | out=%s", VERSION, tag, cfg.device, cfg.out)
        cache = cfg.out / "turns.json"
        if cache.exists():
            self.turns = [Turn(**d) for d in load_json(cache)]
        else:
            if cfg.download:
                download_emowoz(cfg.data_dir, self.logger)
            self.turns = load_emowoz(cfg.data_dir, self.logger)
            assign_roles(self.turns, cfg.role_fracs, cfg.seed)
            dump_json([asdict(t) for t in self.turns], cache)
        roles: Dict[str, int] = {}
        for t in self.turns:
            roles[t.role] = roles.get(t.role, 0) + 1
        self.logger.info("%d turns | per role/split %s", len(self.turns), roles)
        self._sent = None
        self._nli = None

    @property
    def sentiment(self) -> SentimentEngine:
        if self._sent is None:
            self._sent = SentimentEngine(self.cfg, self.cfg.device, self.logger)
        return self._sent

    @property
    def nli(self) -> NLIEngine:
        if self._nli is None:
            self._nli = NLIEngine(self.cfg, self.cfg.device, self.logger)
        return self._nli


def stage_data(cfg: Config) -> Dict[str, Any]:
    ex = Experiment(cfg, "0_data")
    t = ex.turns
    return {"n_turns": len(t), "with_acts": float(np.mean([x.agent_acts is not None for x in t]))}


# ----------------------------------------------------------------------------------------------------------
# stage 1: emotion / satisfaction detection engine
# ----------------------------------------------------------------------------------------------------------

def stage_detector(cfg: Config) -> Dict[str, Any]:
    ex = Experiment(cfg, "1_detector")
    lg, dev = ex.logger, cfg.device
    tr = [t for t in ex.turns if t.role == "E" and 0 <= t.user_emotion < N_EMO]
    dv = filter_turns(ex.turns, lambda t: t.split == "valid" and 0 <= t.user_emotion < N_EMO,
                      cfg.fit.max_dev, cfg.seed)
    if cfg.fit.max_train:
        tr = filter_turns(tr, lambda t: True, cfg.fit.max_train, cfg.seed)
    ytr = np.asarray([t.user_emotion for t in tr], int)
    ydv = np.asarray([t.user_emotion for t in dv], int)
    lg.info("detector data | train %d customer utterances (role E) | dev %d | train class freq %s", len(tr), len(dv),
            np.round(np.bincount(ytr, minlength=N_EMO) / max(1, len(ytr)), 4).tolist())
    tok = load_tokenizer(cfg.encoder_model, cfg, truncation_side="right")
    model = build_text_model(cfg.encoder_model, N_EMO, cfg)
    # The detector reads the CUSTOMER utterance only.  Giving it the agent turn would make the outcome a
    # function of the treatment, and the reward would then learn the detector's reaction to agent wording.
    res = fit_text_model(model, tok, {"a": [t.user_text for t in tr], "y": ytr}, {"a": [t.user_text for t in dv],
                         "y": ydv}, "ce", cfg.fit, dev, lg, "ESD", cfg.seed)
    lg_dv = predict_text_model(model, tok, [t.user_text for t in dv], None, cfg.fit.max_len, dev, cfg.fit.pred_batch)
    T = fit_temperature(lg_dv, ydv)
    p_dv = softmax_np(lg_dv / T)
    pred = p_dv.argmax(1)
    f1s = []
    for k in range(N_EMO):
        tp = float(np.sum((pred == k) & (ydv == k)))
        fp = float(np.sum((pred == k) & (ydv != k)))
        fn = float(np.sum((pred != k) & (ydv == k)))
        if tp + fn > 0:
            f1s.append(2 * tp / max(EPS, 2 * tp + fp + fn))
    u_dv = EMOTION_UTILITY[ydv]
    s_emo = p_dv @ EMOTION_UTILITY
    s_sent = ex.sentiment([t.user_text for t in dv])
    trace = {float(b): spearman(s_emo + b * s_sent, u_dv) for b in cfg.sent_beta_grid}
    beta = max(trace, key=lambda b: trace[b] if math.isfinite(trace[b]) else -2)
    lg.info("detector | T=%.3f | dev macro-F1 %.4f (over %d present classes) | Spearman(S, human utility): "
            "emotion-only %.4f, sentiment-only %.4f, fused beta=%g -> %.4f", T, float(np.mean(f1s)), len(f1s),
            trace.get(0.0, float("nan")), spearman(s_sent, u_dv), beta, trace[beta])
    save_text_model(model, cfg.out / "detector.pt", {"backbone": cfg.encoder_model, "n_out": N_EMO,
                                                     "temperature": T, "beta": beta})
    # Pseudo-labels S(next customer turn) for every turn outside the E partition.  (Valid-split utterances also
    # served for early stopping, temperature and beta above, so pseudo-labels there are slightly optimistic;
    # they are used only for early stopping of the reward models, never for a gate.)
    need = [t for t in ex.turns if t.role != "E" and t.next_user_text]
    nxt = [t.next_user_text for t in need]
    p = softmax_np(predict_text_model(model, tok, nxt, None, cfg.fit.max_len, dev, cfg.fit.pred_batch) / T)
    S = p @ EMOTION_UTILITY + beta * ex.sentiment(nxt)
    labels = {t.uid: float(s) for t, s in zip(need, S)}
    hum = [(labels[t.uid], t.human_utility) for t in need if t.role == "R" and t.human_utility is not None]
    rho_r = spearman([a for a, _ in hum], [b for _, b in hum]) if len(hum) > 10 else float("nan")
    lg.info("pseudo-labels | %d next-customer turns scored | Spearman with the (hidden) human utility on the R "
            "partition %.4f (n=%d)", len(labels), rho_r, len(hum))
    dump_json(labels, cfg.out / "detector_labels.json")
    rep = {"temperature": T, "beta": beta, "beta_trace": trace, "macro_f1": float(np.mean(f1s)),
           "rho_R_pseudo_vs_human": rho_r, "fit": {k: v for k, v in res.items() if k != "history"}}
    dump_json(rep, cfg.out / "detector.json")
    return rep


# ----------------------------------------------------------------------------------------------------------
# stage 2: content-orthogonal phrasing-effect reward
# ----------------------------------------------------------------------------------------------------------

def _outcome(t: Turn, labels: Dict[str, float], source: str) -> Optional[float]:
    return t.human_utility if source == "human" else labels.get(t.uid)


def stage_reward(cfg: Config) -> Dict[str, Any]:
    ex = Experiment(cfg, "2_reward")
    lg, dev = ex.logger, cfg.device
    labels = load_json(cfg.out / "detector_labels.json") if cfg.label_source == "detector" else {}
    has_y = lambda t: _outcome(t, labels, cfg.label_source) is not None   # noqa: E731
    R = [t for t in ex.turns if t.role == "R" and has_y(t)]
    if cfg.fit.max_train:
        R = filter_turns(R, lambda t: True, cfg.fit.max_train, cfg.seed)
    V = filter_turns(ex.turns, lambda t: t.split == "valid" and has_y(t), cfg.fit.max_dev, cfg.seed)
    T = filter_turns(ex.turns, lambda t: t.split == "test" and t.human_utility is not None and bool(t.agent_text),
                     cfg.gate_max_turns, cfg.seed + 1)
    yR = np.asarray([_outcome(t, labels, cfg.label_source) for t in R], float)
    yV = np.asarray([_outcome(t, labels, cfg.label_source) for t in V], float)
    lg.info("reward data | R=%d turns (%d dialogues) | V=%d | gate T=%d (human labels) | label source=%s",
            len(R), len({t.dialogue_id for t in R}), len(V), len(T), cfg.label_source)
    tok = load_tokenizer(cfg.encoder_model, cfg)
    ctx = lambda ts: [context_text(t, cfg.ctx_utts) for t in ts]           # noqa: E731
    cdesc = lambda ts: [content_descriptor(t) for t in ts]                  # noqa: E731
    resp = lambda ts: [t.agent_text for t in ts]                            # noqa: E731

    # (a) nuisance m(h, c), K-fold cross-fitted by dialogue.
    folds = dialogue_folds([t.dialogue_id for t in R], cfg.crossfit_folds, cfg.seed)
    m_oof = np.full(len(R), np.nan)
    m_V = np.zeros(len(V))
    m_T = np.zeros(len(T))
    for k in range(cfg.crossfit_folds):
        tr_ix, te_ix = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        mk = build_text_model(cfg.encoder_model, 1, cfg)
        rk = fit_text_model(mk, tok, {"a": ctx([R[i] for i in tr_ix]), "b": cdesc([R[i] for i in tr_ix]),
                                      "y": yR[tr_ix], "w": np.ones((tr_ix.size, 1))},
                            {"a": ctx(V), "b": cdesc(V), "y": yV}, "mse", cfg.fit, dev, lg, f"m(h,c) fold {k}",
                            cfg.seed + k)
        for ts, put in (([R[i] for i in te_ix], "oof"), (V, "V"), (T, "T")):
            pred = predict_text_model(mk, tok, ctx(ts), cdesc(ts), cfg.fit.max_len, dev,
                                      cfg.fit.pred_batch)[:, 0] * rk["y_sd"] + rk["y_mu"]
            if put == "oof":
                m_oof[te_ix] = pred
            elif put == "V":
                m_V += pred / cfg.crossfit_folds
            else:
                m_T += pred / cfg.crossfit_folds
        del mk
        free_cuda()
    # Linear recalibration of the out-of-fold nuisance (2 parameters, fitted on honest OOF predictions).  An
    # early-stopped regressor is shrunk towards the mean (m_hat ~ alpha * m, alpha < 1); the residual then keeps
    # (1 - alpha) * m, i.e. context/content signal leaks into tau and tau correlates with m_hat.  OLS of y on
    # m_hat_oof undoes the shrinkage.
    cal_b = float(np.cov(m_oof, yR, ddof=1)[0, 1] / max(float(np.var(m_oof, ddof=1)), EPS))
    cal_a = float(np.mean(yR) - cal_b * np.mean(m_oof))
    m_oof, m_V, m_T = cal_a + cal_b * m_oof, cal_a + cal_b * m_V, cal_a + cal_b * m_T
    resid = yR - m_oof
    r2_m = 1.0 - float(np.var(resid)) / max(float(np.var(yR)), EPS)
    lg.info("nuisance m(h,c) | out-of-fold R2 %.4f after recalibration (slope %.3f; >1 means the raw fit was "
            "shrunk) | residual sd %.4f vs outcome sd %.4f -- the variance share that context and content "
            "explain, which GRPO groups cancel anyway", r2_m, cal_b, float(np.std(resid)), float(np.std(yR)))

    # (b) effect model g(h, a) on residuals, M Poisson-bootstrapped heads (and the naive ablation on raw S).
    rng = np.random.default_rng(cfg.seed + 7)

    def fit_effect(target_R: np.ndarray, target_V: np.ndarray, name: str) -> Tuple[Any, Dict[str, Any]]:
        g = build_text_model(cfg.encoder_model, cfg.n_heads, cfg)
        W = rng.poisson(1.0, size=(len(R), cfg.n_heads)).astype(np.float32)
        res = fit_text_model(g, tok, {"a": ctx(R), "b": resp(R), "y": target_R, "w": W},
                             {"a": ctx(V), "b": resp(V), "y": target_V}, "mse", cfg.fit, dev, lg, name, cfg.seed + 11)
        save_text_model(g, cfg.out / f"{name}.pt", {"backbone": cfg.encoder_model, "n_out": cfg.n_heads,
                                                    "y_mu": res["y_mu"], "y_sd": res["y_sd"]})
        return g, res

    g, g_res = fit_effect(resid, yV - m_V, "reward_g")
    del g
    free_cuda()
    reward = AffectReward(cfg.out / "reward_g.pt", cfg, dev)
    tau, tau_sd = reward(T, resp(T))
    naive = None
    if cfg.fit_naive_reward:
        g0, _ = fit_effect(yR, yV, "reward_naive")
        del g0
        free_cuda()
        naive = AffectReward(cfg.out / "reward_naive.pt", cfg, dev)

    # (c) validity gate on the TEST split, against HUMAN labels the reward never saw.
    u = np.asarray([t.human_utility for t in T], float)
    cl = np.asarray([t.dialogue_id for t in T])
    L = np.log1p([n_words(t.agent_text) for t in T])
    bad = np.asarray([is_bad_news(t) for t in T], bool)
    B = cfg.n_boot
    gate: Dict[str, Any] = {"n": len(T)}

    strata = np.asarray([content_descriptor(t) for t in T])

    def pci(x, y, z, seed):
        return cluster_bootstrap_ci(lambda ix: partial_spearman(x[ix], y[ix], z[ix]), cl, B, seed)

    def spci(x, y, z, seed):
        return cluster_bootstrap_ci(lambda ix: stratified_partial_spearman(x[ix], y[ix], z[ix], strata[ix]), cl, B,
                                    seed)

    # GATE: within the same content (content fixed effects) and given m_hat, does tau predict the HUMAN label?
    # This is specific to phrasing: content signal that m_hat failed to absorb cannot pass it.
    gate["tau_within_content_rho"] = spci(tau, u, m_T, cfg.seed + 9)
    gate["n_content_strata"] = int(np.unique(strata).size)
    gate["tau_partial_rho"] = pci(tau, u, m_T, cfg.seed)
    gate["m_rho"] = spearman(m_T, u)
    gate["m_plus_tau_rho"] = spearman(m_T + tau, u)
    gate["tau_m_rho"] = spearman(tau, m_T)
    gate["length_partial_rho_tau"] = partial_spearman(tau, L, m_T)
    gate["length_partial_rho_human"] = partial_spearman(u, L, m_T)

    def smd(x):
        s = float(np.std(x)) or 1.0
        return cluster_bootstrap_ci(lambda ix: (float(np.mean(x[ix][bad[ix]])) - float(np.mean(x[ix][~bad[ix]]))) / s
                                    if bad[ix].any() and (~bad[ix]).any() else float("nan"), cl, B, cfg.seed + 3)

    gate["bad_news_smd_tau"] = smd(tau)
    gate["bad_news_smd_human"] = smd(u)
    pr = np.random.default_rng(cfg.seed + 5)
    padded = [f"{t.agent_text} {HELDOUT_TAILS[int(pr.integers(len(HELDOUT_TAILS)))]}" for t in T]
    tau_pad, _ = reward(T, padded)
    sd_tau = float(np.std(tau)) or 1.0
    dpad = (tau_pad - tau) / sd_tau
    gate["padding_shift_sd"] = cluster_bootstrap_ci(lambda ix: float(np.mean(dpad[ix])), cl, B, cfg.seed + 6, conf=0.90)
    if naive is not None:
        g0, _ = naive(T, resp(T))
        gate["naive_partial_rho"] = pci(g0, u, m_T, cfg.seed + 1)
        gate["naive_within_content_rho"] = spci(g0, u, m_T, cfg.seed + 10)
        gate["naive_rho"] = spearman(g0, u)
        gate["bad_news_smd_naive"] = smd(g0)
    lo = gate["tau_within_content_rho"][1]
    gate["passes"] = bool(math.isfinite(lo) and lo > 0.0)
    f3 = lambda c: f"{c[0]:+.4f} CI[{c[1]:+.4f},{c[2]:+.4f}]"   # noqa: E731
    lg.info("VALIDITY (test split, human labels, n=%d, %d content strata) | within-content partial rho(tau, human "
            "| m) = %s <- gate | pooled partial rho %s | rho(m, human) %+.4f | rho(m+tau, human) %+.4f | rho(tau, m) "
            "%+.4f (~0 if orthogonal)", len(T), gate["n_content_strata"], f3(gate["tau_within_content_rho"]),
            f3(gate["tau_partial_rho"]), gate["m_rho"], gate["m_plus_tau_rho"], gate["tau_m_rho"])
    lg.info("  bad-news standardised difference: tau %s | human %s%s", f3(gate["bad_news_smd_tau"]),
            f3(gate["bad_news_smd_human"]),
            f" | naive reward {f3(gate['bad_news_smd_naive'])}" if naive is not None else "")
    lg.info("  length: partial rho(tau, loglen | m) %+.4f vs human %+.4f | content-free padding shift %s sd "
            "(90%% CI; diagnostic only, never tuned)", gate["length_partial_rho_tau"],
            gate["length_partial_rho_human"], f3(gate["padding_shift_sd"]))
    if naive is not None:
        lg.info("  naive reward (no partialling-out) | within-content partial rho %s | pooled partial rho %s | raw "
                "rho %+.4f", f3(gate["naive_within_content_rho"]), f3(gate["naive_partial_rho"]), gate["naive_rho"])
    if abs(gate["tau_m_rho"]) > 0.2 or (gate["bad_news_smd_tau"][1] * gate["bad_news_smd_tau"][2] > 0):
        lg.warning("tau still carries context/content signal (rho(tau, m)=%+.3f, bad-news difference %s): m_hat "
                   "under-fits. PACE's advantages are unaffected while the NLI feasibility check is accurate (content "
                   "is constant inside a feasible group), but content changes that slip past it are then rewarded; "
                   "improve m (more data/epochs, richer content descriptors) before relying on the naive-vs-PACE "
                   "comparison", gate["tau_m_rho"], f3(gate["bad_news_smd_tau"]))
    rep = {"gate": gate, "m_r2_oof": r2_m, "g_fit": {k: v for k, v in g_res.items() if k != "history"},
           "label_source": cfg.label_source, "n_R": len(R)}
    dump_json(rep, cfg.out / "reward.json")
    if not gate["passes"]:
        msg = ("REWARD GATE FAIL: within the same content, the phrasing effect tau does not predict HUMAN-labelled "
               "satisfaction beyond m_hat (lower CI <= 0). RL on it would optimise noise.")
        lg.error(msg)
        if cfg.strict:
            raise RuntimeError(msg)
    return rep


# ----------------------------------------------------------------------------------------------------------
# stage 3: independent judge
# ----------------------------------------------------------------------------------------------------------

def stage_judge(cfg: Config) -> Dict[str, Any]:
    ex = Experiment(cfg, "3_judge")
    lg, dev = ex.logger, cfg.device
    J = [t for t in ex.turns if t.role == "J" and t.human_utility is not None]
    if cfg.fit.max_train:
        J = filter_turns(J, lambda t: True, cfg.fit.max_train, cfg.seed)
    V = filter_turns(ex.turns, lambda t: t.split == "valid" and t.human_utility is not None, cfg.fit.max_dev,
                     cfg.seed + 2)
    T = filter_turns(ex.turns, lambda t: t.split == "test" and t.human_utility is not None, cfg.gate_max_turns,
                     cfg.seed + 1)
    tok = load_tokenizer(cfg.encoder_model, cfg)
    model = build_text_model(cfg.encoder_model, N_EMO, cfg)
    ctx = lambda ts: [context_text(t, cfg.ctx_utts) for t in ts]           # noqa: E731
    res = fit_text_model(model, tok, {"a": ctx(J), "b": [t.agent_text for t in J],
                                      "y": np.asarray([t.next_emotion for t in J], int)},
                         {"a": ctx(V), "b": [t.agent_text for t in V], "y": np.asarray([t.next_emotion for t in V], int)},
                         "ce", cfg.fit, dev, lg, "JUDGE", cfg.seed + 21)
    lv = predict_text_model(model, tok, ctx(V), [t.agent_text for t in V], cfg.fit.max_len, dev, cfg.fit.pred_batch)
    Tm = fit_temperature(lv, np.asarray([t.next_emotion for t in V], int))
    save_text_model(model, cfg.out / "judge.pt", {"backbone": cfg.encoder_model, "n_out": N_EMO, "temperature": Tm})
    judge = Judge(cfg.out / "judge.pt", cfg, dev)
    jt = judge(T, [t.agent_text for t in T])
    u = np.asarray([t.human_utility for t in T], float)
    ci = cluster_bootstrap_ci(lambda ix: spearman(jt[ix], u[ix]), [t.dialogue_id for t in T], cfg.n_boot, cfg.seed)
    lg.info("judge | trained on %d J-partition turns | T=%.3f | test Spearman(judge, human utility) %+.4f "
            "CI[%+.4f,%+.4f]", len(J), Tm, *ci)
    rep = {"n_J": len(J), "temperature": Tm, "test_rho": ci, "fit": {k: v for k, v in res.items() if k != "history"}}
    dump_json(rep, cfg.out / "judge.json")
    return rep


# ----------------------------------------------------------------------------------------------------------
# stage 4: PACE reinforcement learning
# ----------------------------------------------------------------------------------------------------------

def rl_contexts(turns: Sequence[Turn], split: str, limit: Optional[int], seed: int) -> List[Turn]:
    return filter_turns(turns, lambda t: t.split == split and 3 <= n_words(t.agent_text) <= 50, limit, seed)


def score_batch(arm: str, turns: Sequence[Turn], texts: Sequence[str], samples: Optional[Sequence[Sample]],
                engines: Dict[str, Any], pc: PACEConfig) -> Dict[str, np.ndarray]:
    """Reward and constraint quantities for one batch of responses."""
    src = [t.agent_text for t in turns]
    hyg = np.asarray([hygiene_ok(x, pc.min_words, pc.max_words, pc.require_terminal)[0] for x in texts], bool)
    if samples is not None and not pc.allow_truncated:
        hyg &= ~np.asarray([s.truncated for s in samples], bool)
    fab = np.asarray([bool(fabricated_entities(s, x)) for s, x in zip(src, texts)], bool)
    fid = engines["nli"].fidelity(src, texts)
    v_f = np.where(hyg & ~fab, 1.0 - fid["f"], 1.0)
    v_len = length_overrun(src, texts, pc.len_slack)
    if arm == "sentiment_only":
        r, r_sd = engines["sentiment"](list(texts)), np.zeros(len(texts))
    elif arm == "pace_naive_reward":
        r, r_sd = engines["reward_naive"](turns, texts)
    else:
        r, r_sd = engines["reward"](turns, texts)
    return {"r_mean": r, "r_sd": r_sd, "hyg": hyg, "fab": fab, "f": fid["f"], "complete": fid["complete"],
            "contra": fid["contra"], "v_f": v_f, "v_len": v_len}


def train_pace(arm: str, policy: PolicyLM, engines: Dict[str, Any], turns: Sequence[Turn], pc: PACEConfig,
               logger: logging.Logger, out_dir: Path, seed: int) -> Dict[str, Any]:
    from torch.optim import AdamW
    constrained = arm in ("pace", "pace_naive_reward")
    rng = np.random.default_rng(seed)
    opt = AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=pc.lr, weight_decay=0.0)
    dual_f = DualVariable(pc.eps_fid, pc.dual_lr, pc.lam_f0 if constrained else 0.0, pc.lam_max, not constrained)
    dual_l = DualVariable(pc.eps_len, pc.dual_lr, pc.lam_len0 if constrained else 0.0, pc.lam_max, not constrained)
    beta, r_scale = pc.kl_coef, float("nan")
    hist: List[Dict[str, float]] = []
    aborted = False
    logger.info("%s | constrained=%s | B=%d contexts x G=%d | PPO epochs %d | KL target %.3g | f_min %.2f | "
                "eps_fid %.2f | length budget x%.2f", arm, constrained, pc.n_contexts, pc.group_size, pc.ppo_epochs,
                pc.kl_target, pc.f_min, pc.eps_fid, 1 + pc.len_slack)
    for step in range(1, pc.steps + 1):
        chunk = [turns[i] for i in rng.choice(len(turns), size=min(pc.n_contexts, len(turns)), replace=False)]
        samples = policy.generate([policy.prompt(t) for t in chunk], pc.group_size, pc.temperature, pc.top_p,
                                  pc.max_new_tokens, seed * 100003 + step, pc.gen_batch)
        flat = [t for t in chunk for _ in range(pc.group_size)]
        texts = [s.text for s in samples]
        gid = np.repeat(np.arange(len(chunk)), pc.group_size)
        sc = score_batch(arm, flat, texts, samples, engines, pc)
        r = sc["r_mean"] - pc.kappa * sc["r_sd"]
        feasible = (sc["hyg"] & ~sc["fab"] & (sc["f"] >= pc.f_min)) if constrained else sc["hyg"]
        scale = pc.scale_floor if not math.isfinite(r_scale) else max(r_scale, pc.scale_floor)
        A = pace_advantages(r, sc["r_sd"], gid, sc["hyg"], feasible, sc["v_f"], sc["v_len"], dual_f.lam,
                            dual_l.lam, scale, pc.snr_kappa, pc.adv_clip, pc.bad_penalty)
        if math.isfinite(A["within_sd"]):
            r_scale = A["within_sd"] if not math.isfinite(r_scale) else 0.9 * r_scale + 0.1 * A["within_sd"]
        old = policy.batched_logprobs(samples, ref=False, micro=pc.micro)
        ref = policy.batched_logprobs(samples, ref=True, micro=pc.micro)
        st = policy.ppo_update(samples, A["adv"], old, ref, opt, pc, beta, rng)
        lam_f = dual_f.update(float(np.mean(sc["v_f"])))
        lam_l = dual_l.update(float(np.mean(sc["v_len"])))
        beta = float(np.clip(beta * (1.5 if st["kl"] > 2 * pc.kl_target else 0.75 if st["kl"] < 0.5 * pc.kl_target
                                     else 1.0), pc.kl_coef_min, pc.kl_coef_max))
        rec = {"step": step, "reward": float(np.mean(sc["r_mean"][sc["hyg"]])) if sc["hyg"].any() else float("nan"),
               "feasible": float(np.mean(feasible)), "fidelity": float(np.mean(sc["f"])),
               "fabricated": float(np.mean(sc["fab"])), "hygiene": float(np.mean(sc["hyg"])),
               "len_ratio": float(np.mean([(n_words(x) + 1) / (n_words(t.agent_text) + 1) for x, t in zip(texts, flat)])),
               "v_f": float(np.mean(sc["v_f"])), "v_len": float(np.mean(sc["v_len"])), "lam_f": lam_f,
               "lam_len": lam_l, "kl": st["kl"], "beta": beta, "clipfrac": st["clipfrac"],
               "affect_groups": A["n_affect_groups"], "gated": A["n_gated"], "r_scale": r_scale}
        hist.append(rec)
        if step % pc.log_every == 0 or step == 1 or step == pc.steps:
            logger.info("%s | step %d/%d | reward %.4f | feasible %.2f fidelity %.3f fabricated %.3f hygiene %.2f | "
                        "len x%.2f | lam_f %.2f lam_len %.2f | KL %.4f (beta %.4f) clip %.3f | affect groups %d/%d "
                        "(gated %d)", arm, step, pc.steps, rec["reward"], rec["feasible"], rec["fidelity"],
                        rec["fabricated"], rec["hygiene"], rec["len_ratio"], lam_f, lam_l, st["kl"], beta,
                        st["clipfrac"], A["n_affect_groups"], A["n_groups"], A["n_gated"])
        if math.isfinite(st["kl"]) and st["kl"] > pc.kl_abort * pc.kl_target:
            logger.error("%s | step %d: per-token KL %.4f exceeds %g x target; stopping and keeping this adapter",
                         arm, step, st["kl"], pc.kl_abort)
            aborted = True
            break
    policy.save_adapter(out_dir / f"policy_{arm}_s{seed}")
    return {"arm": arm, "seed": seed, "history": hist, "aborted": aborted, "steps_run": len(hist)}


def load_engines(cfg: Config, ex: Experiment, need: Sequence[str]) -> Dict[str, Any]:
    eng: Dict[str, Any] = {}
    if "nli" in need:
        eng["nli"] = ex.nli
    if "sentiment" in need:
        eng["sentiment"] = ex.sentiment
    if "reward" in need:
        eng["reward"] = AffectReward(cfg.out / "reward_g.pt", cfg, cfg.device)
    if "reward_naive" in need and (cfg.out / "reward_naive.pt").exists():
        eng["reward_naive"] = AffectReward(cfg.out / "reward_naive.pt", cfg, cfg.device)
    if "judge" in need:
        eng["judge"] = Judge(cfg.out / "judge.pt", cfg, cfg.device)
    return eng


def stage_train(cfg: Config, arm: str, seed: int) -> Dict[str, Any]:
    ex = Experiment(cfg, f"4_train_{arm}_{seed}")
    rep = load_json(cfg.out / "reward.json")
    if cfg.strict and not rep["gate"]["passes"]:
        raise RuntimeError("the reward did not pass its human-anchored validity gate; refusing to run RL "
                           "(use --no-strict only for debugging)")
    seed_everything(seed)
    policy = PolicyLM(cfg, ex.logger, cfg.device)
    need = ["nli", "sentiment", "reward"] + (["reward_naive"] if arm == "pace_naive_reward" else [])
    eng = load_engines(cfg, ex, need)
    if arm == "pace_naive_reward" and "reward_naive" not in eng:
        raise FileNotFoundError("reward_naive.pt missing; re-run the reward stage with the naive ablation enabled")
    turns = rl_contexts(ex.turns, "train", cfg.rl_contexts, seed)
    res = train_pace(arm, policy, eng, turns, cfg.pace, ex.logger, cfg.out, seed)
    dump_json(res, cfg.out / f"train_{arm}_{seed}.json")
    return {k: v for k, v in res.items() if k != "history"}


# ----------------------------------------------------------------------------------------------------------
# stage 5/6: evaluation and report
# ----------------------------------------------------------------------------------------------------------

def stage_eval(cfg: Config, arm: str, seed: int) -> Dict[str, Any]:
    ex = Experiment(cfg, f"5_eval_{arm}_{seed}")
    te = rl_contexts(ex.turns, "test", cfg.eval_turns, cfg.seed)
    eng = load_engines(cfg, ex, ["nli", "sentiment", "reward", "judge"])
    if arm == "source":
        texts, samples = [t.agent_text for t in te], None
    else:
        policy = PolicyLM(cfg, ex.logger, cfg.device)
        if arm != "base":
            policy.load_adapter(cfg.out / f"policy_{arm}_s{seed}")
        samples = policy.generate([policy.prompt(t) for t in te], 1, cfg.eval_temperature, cfg.pace.top_p,
                                  cfg.pace.max_new_tokens, 777000 + seed, cfg.pace.gen_batch)
        texts = [s.text for s in samples]
    sc = score_batch("pace", te, texts, samples, eng, cfg.pace)
    j = eng["judge"](te, texts)
    sent = eng["sentiment"](texts)
    rows = []
    for i, (t, x) in enumerate(zip(te, texts)):
        rows.append({"uid": t.uid, "dialogue_id": t.dialogue_id, "arm": arm, "seed": seed, "response": x,
                     "source": t.agent_text, "judge_utility": float(j[i]), "affect_tau": float(sc["r_mean"][i]),
                     "fidelity": float(sc["f"][i]), "complete": float(sc["complete"][i]),
                     "contradiction": float(sc["contra"][i]), "fabricated": float(sc["fab"][i]),
                     "feasible": float(sc["hyg"][i] and not sc["fab"][i] and sc["f"][i] >= cfg.pace.f_min),
                     "hygiene": float(sc["hyg"][i]), "words": n_words(x),
                     "len_ratio": (n_words(x) + 1) / (n_words(t.agent_text) + 1), "agent_sentiment": float(sent[i]),
                     "bad_news": float(is_bad_news(t))})
    dump_json(rows, cfg.out / f"eval_{arm}_{seed}.json")
    ex.logger.info("eval %s seed=%d | n=%d | judge %.4f | tau %.4f | fidelity %.3f | feasible %.3f | fabricated %.3f "
                   "| words %.1f", arm, seed, len(rows), *[float(np.mean([r[k] for r in rows])) for k in
                   ("judge_utility", "affect_tau", "fidelity", "feasible", "fabricated", "words")])
    return {"n": len(rows)}


REPORT_METRICS = ("judge_utility", "affect_tau", "fidelity", "feasible", "fabricated", "contradiction", "hygiene",
                  "len_ratio", "agent_sentiment")


def stage_report(cfg: Config, arms: Sequence[str]) -> Dict[str, Any]:
    ex = Experiment(cfg, "6_report")
    lg = ex.logger
    data: Dict[str, Dict[str, Dict[str, float]]] = {}   # arm -> uid -> metric means over seeds
    meta: Dict[str, str] = {}
    for arm in arms:
        per: Dict[str, List[Dict[str, Any]]] = {}
        for sd in cfg.seeds:
            f = cfg.out / f"eval_{arm}_{sd}.json"
            if f.exists():
                for r in load_json(f):
                    per.setdefault(r["uid"], []).append(r)
                    meta[r["uid"]] = r["dialogue_id"]
        if per:
            data[arm] = {u: {m: float(np.mean([x[m] for x in rs])) for m in REPORT_METRICS} for u, rs in per.items()}
    if "base" not in data:
        raise FileNotFoundError("no evaluation rows for the 'base' arm")
    summary: Dict[str, Any] = {"arms": {}, "contrasts": {}}
    for arm, d in data.items():
        summary["arms"][arm] = {m: float(np.mean([v[m] for v in d.values()])) for m in REPORT_METRICS}
        summary["arms"][arm]["n"] = len(d)
    names, pv = [], []
    for arm in data:
        if arm == "base":
            continue
        keys = sorted(set(data[arm]) & set(data["base"]))
        cl = np.asarray([meta[k] for k in keys])
        c: Dict[str, Any] = {"n": len(keys)}
        for m in REPORT_METRICS:
            d = np.asarray([data[arm][k][m] - data["base"][k][m] for k in keys], float)
            pt, lo, hi = cluster_bootstrap_ci(lambda ix: float(np.mean(d[ix])), cl, cfg.n_boot, cfg.seed)
            c[m] = {"delta": pt, "ci95": [lo, hi], "p": signflip_pvalue(d, cl, seed=cfg.seed)}
        c["fidelity_non_inferior"] = bool(math.isfinite(c["fidelity"]["ci95"][0])
                                          and c["fidelity"]["ci95"][0] > -cfg.fidelity_ni_margin)
        summary["contrasts"][f"{arm}_vs_base"] = c
        if arm != "source":                      # the Holm family is the trained arms only
            names.append(f"{arm}_vs_base")
            pv.append(c["judge_utility"]["p"])
    for nm, p in zip(names, holm(pv)):
        summary["contrasts"][nm]["judge_utility"]["p_holm"] = p
    dump_json(summary, cfg.out / "report.json")
    lg.info("%-20s %8s %8s %8s %8s %8s %8s", "arm", "judge", "tau", "fidel", "feasible", "fabric", "len_x")
    for arm, v in summary["arms"].items():
        lg.info("%-20s %8.4f %8.4f %8.3f %8.3f %8.3f %8.2f", arm, v["judge_utility"], v["affect_tau"], v["fidelity"],
                v["feasible"], v["fabricated"], v["len_ratio"])
    for nm, c in summary["contrasts"].items():
        j, f = c["judge_utility"], c["fidelity"]
        lg.info("%-28s judge %+.4f CI[%+.4f,%+.4f] p=%.4f p_holm=%s | fidelity %+.3f CI[%+.3f,%+.3f] "
                "non-inferior=%s", nm, j["delta"], *j["ci95"], j["p"],
                f"{j['p_holm']:.4f}" if "p_holm" in j else "n/a", f["delta"], *f["ci95"], c["fidelity_non_inferior"])
    return summary


def stage_all(cfg: Config) -> None:
    stage_data(cfg)
    if cfg.label_source == "detector":
        stage_detector(cfg)
    stage_reward(cfg)
    stage_judge(cfg)
    for arm in cfg.arms:
        for sd in cfg.seeds:
            stage_train(cfg, arm, sd)
    for arm in ("source", "base") + tuple(cfg.arms):
        for sd in cfg.seeds:
            stage_eval(cfg, arm, sd)
    stage_report(cfg, ("source", "base") + tuple(cfg.arms))


# ----------------------------------------------------------------------------------------------------------
# statistical unit tests (numpy only)
# ----------------------------------------------------------------------------------------------------------

def _ridge(X: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    P = lam * np.eye(X1.shape[1])
    P[0, 0] = 0.0
    return np.linalg.solve(X1.T @ X1 + P, X1.T @ y)


def _ridge_pred(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return beta[0] + X @ beta[1:]


def run_unit_tests() -> None:
    rng = np.random.default_rng(0)
    # (1) statistics helpers
    assert np.allclose(rankdata([3, 1, 2, 2]), [4, 1, 2.5, 2.5])
    assert abs(spearman(np.arange(50), np.arange(50) ** 3) - 1) < 1e-12
    z = rng.normal(size=4000)
    x, y = z + rng.normal(size=4000), z + rng.normal(size=4000)
    assert spearman(x, y) > 0.3 and abs(partial_spearman(x, y, z)) < 0.06, "partial correlation does not remove z"

    # (2) Why partial out content.  Content c (bad news) lowers satisfaction and makes empathy p more likely
    #     (agents soften bad news); a response reveals both.  Naive E[S|h,a] learns a large bad-news penalty
    #     (an incentive to withhold bad news).  The partialled phrasing effect keeps the empathy slope, and its
    #     sensitivity to content is second-order (bounded by the phrasing effect times the difference in
    #     propensity), which is what makes it robust to imperfect content-equivalence checks.
    n = 20000
    X = rng.normal(size=(n, 3))
    c = (rng.random(n) < 1 / (1 + np.exp(-1.2 * X[:, 1]))).astype(float)
    p = (rng.random(n) < 1 / (1 + np.exp(-(-0.4 + 1.5 * c + 0.5 * X[:, 2])))).astype(float)
    S = 0.8 * X[:, 0] - 0.6 * c + 0.25 * p + rng.normal(0, 0.5, n)
    folds = rng.integers(0, 2, n)
    Hc = np.column_stack([X, c, X * c[:, None]])
    m_oof = np.zeros(n)
    for k in (0, 1):
        m_oof[folds == k] = _ridge_pred(_ridge(Hc[folds != k], S[folds != k]), Hc[folds == k])
    Ha = np.column_stack([X, c, p, c * p])
    naive = _ridge(Ha, S)
    orth = _ridge(Ha, S - m_oof)
    bad_naive = naive[4] + naive[6] * p.mean()          # d reward / d c at the average phrasing
    bad_orth = orth[4] + orth[6] * p.mean()
    emp_naive = naive[5] + naive[6] * c.mean()
    emp_orth = orth[5] + orth[6] * c.mean()
    assert bad_naive < -0.45, f"synthetic confounding absent ({bad_naive:+.3f})"
    assert abs(bad_orth) < 0.25 * abs(bad_naive), f"partialling-out did not remove the content effect ({bad_orth:+.3f})"
    assert abs(emp_orth - 0.25) < 0.05 and abs(emp_naive - 0.25) < 0.05, (emp_orth, emp_naive)
    # within fixed context AND content (bad news), both rewards rank empathetic above plain by ~ the truth
    assert abs((naive[5] + naive[6]) - (orth[5] + orth[6])) < 0.05 and orth[5] + orth[6] > 0.15
    print(f"unit 2 OK | bad-news sensitivity naive {bad_naive:+.3f} vs partialled {bad_orth:+.3f} | empathy effect "
          f"naive {emp_naive:+.3f} vs partialled {emp_orth:+.3f} (truth +0.250)")

    # (3) advantage algebra
    gid = np.repeat([0, 1], 4)
    r = np.array([0.1, 0.3, 0.2, 9.0, 0.5, 0.5, 0.5, 0.5])
    hyg = np.array([1, 1, 1, 1, 1, 1, 1, 0], bool)
    feas = np.array([1, 1, 1, 0, 1, 1, 1, 1], bool)
    vf = np.array([0.1, 0.1, 0.1, 0.9, 0.0, 0.0, 0.0, 1.0])
    A = pace_advantages(r, np.zeros(8), gid, hyg, feas, vf, np.zeros(8), 2.0, 0.0, 0.1, snr_kappa=1.0)
    assert A["a_aff"][3] == 0.0, "an infeasible response received affect advantage"
    assert abs(A["a_aff"][:3].sum()) < 1e-9 and A["a_aff"][1] > 0 > A["a_aff"][0]
    assert A["a_con"][3] < 0, "a constraint violation was not penalised"
    assert A["adv"][7] == -1.0 and A["n_affect_groups"] == 1, "hygiene failure / zero-spread group mishandled"
    Ag = pace_advantages(r, np.full(8, 1.0), gid, hyg, feas, vf, np.zeros(8), 0.0, 0.0, 0.1, snr_kappa=1.0)
    assert Ag["n_gated"] == 2 and not np.any(Ag["a_aff"][:3]), "signal-to-noise gate did not fire"
    print("unit 3 OK | feasible-set leave-one-out advantages, constraint terms, hygiene and SNR gating")

    # (4) Constrained bandit in a bad-news context.  Arms: (keep, plain) (keep, empathetic) (drop, plain)
    #     (drop, empathetic).  Using the phrasing-effect values implied by unit 2's model, PACE must converge to
    #     keep+empathetic; affect-only optimisation of either reward drops the bad news (sycophancy).
    e1, e0 = 0.6, 0.3                                   # propensity of empathy given bad / good news content
    mu = np.array([-0.6, -0.35, 0.0, 0.25])             # naive E[S | h, a]
    tau = np.array([0.25 * (0 - e1), 0.25 * (1 - e1), 0.25 * (0 - e0), 0.25 * (1 - e0)])
    fid = np.array([0.95, 0.95, 0.05, 0.05])

    def run(reward_vec, constrained, steps=500, G=8, lr=0.5, seed=1):
        rs = np.random.default_rng(seed)
        theta = np.zeros(4)
        lam = DualVariable(0.10, 1.0, 1.0 if constrained else 0.0, 20.0, not constrained)
        scale = float("nan")
        for _ in range(steps):
            pi = np.exp(theta - theta.max())
            pi /= pi.sum()
            a = rs.choice(4, size=G, p=pi)
            rr = reward_vec[a] + rs.normal(0, 0.05, G)
            f = np.clip(fid[a] + rs.normal(0, 0.02, G), 0, 1)
            feas_ = (f >= 0.5) if constrained else np.ones(G, bool)
            sc_ = 0.05 if not math.isfinite(scale) else max(scale, 1e-3)
            out = pace_advantages(rr, np.full(G, 0.02), np.zeros(G, int), np.ones(G, bool), feas_, 1 - f,
                                  np.zeros(G), lam.lam, 0.0, sc_, snr_kappa=1.0)
            if math.isfinite(out["within_sd"]):
                scale = out["within_sd"] if not math.isfinite(scale) else 0.9 * scale + 0.1 * out["within_sd"]
            grad = np.zeros(4)
            for ai, adv in zip(a, out["adv"]):
                grad += adv * (np.eye(4)[ai] - pi)
            theta += lr * grad / G
            lam.update(float(np.mean(1 - f)))
        pi = np.exp(theta - theta.max())
        return pi / pi.sum()

    pi_pace = run(tau, True)
    pi_tau_unc = run(tau, False)
    pi_naive = run(mu, False)
    assert pi_pace[1] > 0.9, f"PACE did not converge to keep+empathetic: {np.round(pi_pace, 3)}"
    assert pi_tau_unc[2] + pi_tau_unc[3] > 0.5, f"unconstrained tau should drift to dropping content: {pi_tau_unc}"
    assert pi_naive[2] + pi_naive[3] > 0.9, f"naive affect reward should learn to drop bad news: {pi_naive}"
    print(f"unit 4 OK | P(keep, empathetic): PACE {pi_pace[1]:.3f} | P(drop bad news): unconstrained tau "
          f"{pi_tau_unc[2] + pi_tau_unc[3]:.3f}, naive sentiment-style reward {pi_naive[2] + pi_naive[3]:.3f}")

    # (5) content descriptor / fabrication checks
    assert delex_acts({"general-reqmore": [["none", "none"]]}) == "social_only"
    assert delex_acts({"Booking-NoBook": [["Day", "monday"]], "general-bye": []}) == "Booking-NoBook(day)"
    assert fabricated_entities("Your reference is AB12CD34, train at 10:15.", "Booked, ref AB12CD34 at 10:15!") == set()
    assert "99" in fabricated_entities("It costs 10 pounds.", "It costs 99 pounds.")
    assert "request" not in info_units("Is there anything else I can help you with?")
    print("unit 5 OK | content descriptors and fabrication detector")


# ----------------------------------------------------------------------------------------------------------
# self-test: the whole pipeline on synthetic EmoWOZ with tiny randomly initialised local models
# ----------------------------------------------------------------------------------------------------------

def make_synthetic_emowoz(data_dir: Path, n_dialogues: int = 160, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    empathy = ["I am sorry about that.", "I completely understand.", "Thank you for your patience."]
    split = {"train": {"multiwoz": [], "dialmage": []}, "dev": {"multiwoz": [], "dialmage": []},
             "test": {"multiwoz": [], "dialmage": []}}
    mw: Dict[str, Any] = {}
    dm: Dict[str, Any] = {}
    for i in range(n_dialogues):
        src = "multiwoz" if i % 4 else "dialmage"
        did = f"SYN{i:04d}.json"
        log, mood = [], 0.0
        for k in range(int(rng.integers(3, 7))):
            dom = str(rng.choice(["train", "hotel", "restaurant"]))
            emo = 6 if mood > 0.5 else 2 if mood < -0.5 else 0
            lead = {6: str(rng.choice(["Great, thanks!", "Wonderful, thank you."])),
                    2: str(rng.choice(["That is really annoying.", "This is so frustrating."])), 0: ""}[emo]
            txt = norm_text(f"{lead} I need a {dom} for {int(rng.integers(1, 6))} people on monday please.")
            log.append({"text": txt, "emotion": [{"emotion": emo} for _ in range(4)]})
            bad = rng.random() < 0.4
            emp = rng.random() < (0.6 if bad else 0.3)
            ref = "".join(rng.choice(list("ABCDEFGH23456789"), 8))
            body = (f"There is no {dom} available at that time." if bad else
                    f"Your {dom} is booked, the reference number is {ref}.")
            txt = (str(rng.choice(empathy)) + " " + body) if emp else body
            acts = ({"Booking-NoBook": [["Time", "10:00"]]} if bad else {"Booking-Book": [["Ref", ref]]})
            if rng.random() < 0.5:
                acts["general-reqmore"] = [["none", "none"]]
            log.append({"text": txt, "emotion": [], "dialog_act": acts if src == "multiwoz" else {}})
            mood = 0.5 * mood + (-0.9 if bad else 0.6) + (0.5 if emp else 0.0) + float(rng.normal(0, 0.3))
        log.append({"text": "Thanks, that is all." if mood > 0 else "This is not good enough.",
                    "emotion": [{"emotion": 6 if mood > 0 else 2} for _ in range(4)]})
        (mw if src == "multiwoz" else dm)[did] = {"log": log}
        split["train" if i % 10 < 7 else "dev" if i % 10 < 8 else "test"][src].append(did)
    data_dir.mkdir(parents=True, exist_ok=True)
    dump_json(mw, data_dir / "emowoz-multiwoz.json")
    dump_json(dm, data_dir / "emowoz-dialmage.json")
    dump_json(split, data_dir / "data-split.json")


def build_tiny_models(root: Path, texts: Sequence[str]) -> Dict[str, str]:
    """Tiny, randomly initialised HF models + a word-level tokenizer, saved locally (no downloads)."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import (LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, RobertaConfig, RobertaModel,
                              RobertaForSequenceClassification)
    torch.manual_seed(0)
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4}
    for t in list(texts) + [REWRITE_SYSTEM, "Conversation: Customer: Agent: Draft reply: Reply:"]:
        for w in re.findall(r"\w+|[^\w\s]", t):
            vocab.setdefault(w, len(vocab))
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    core.decoder = decoders.WordPiece(prefix="##")
    core.post_processor = processors.TemplateProcessing(single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>",
                                                        special_tokens=[("<s>", 0), ("</s>", 2)])
    tok = PreTrainedTokenizerFast(tokenizer_object=core, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                  unk_token="<unk>", mask_token="<mask>", cls_token="<s>", sep_token="</s>")
    V = len(vocab)
    enc_cfg = dict(vocab_size=V, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                   max_position_embeddings=520, pad_token_id=1, bos_token_id=0, eos_token_id=2, type_vocab_size=1)
    paths = {k: str(root / k) for k in ("encoder", "nli", "sentiment", "lm")}
    RobertaModel(RobertaConfig(**enc_cfg)).save_pretrained(paths["encoder"])
    RobertaForSequenceClassification(RobertaConfig(**enc_cfg, num_labels=3, id2label={0: "contradiction", 1: "neutral",
                                     2: "entailment"}, label2id={"contradiction": 0, "neutral": 1, "entailment": 2})
                                     ).save_pretrained(paths["nli"])
    RobertaForSequenceClassification(RobertaConfig(**enc_cfg, num_labels=3, id2label={0: "negative", 1: "neutral",
                                     2: "positive"}, label2id={"negative": 0, "neutral": 1, "positive": 2})
                                     ).save_pretrained(paths["sentiment"])
    # Llama layout (same q/k/v/o/gate/up/down module names as Qwen2); a qwen2 config would make AutoTokenizer
    # swap the word-level tokenizer for Qwen2's BPE in recent transformers releases.
    LlamaForCausalLM(LlamaConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                                 tie_word_embeddings=True, bos_token_id=0, eos_token_id=2, pad_token_id=1)
                     ).save_pretrained(paths["lm"])
    for p in paths.values():
        tok.save_pretrained(p)
    return paths


def _check_policy_gradient_path(cfg: Config) -> None:
    """(i) with a fresh adapter the policy equals the reference exactly (adapter disabling works); (ii) one PPO
    update raises the log-likelihood of positive-advantage samples relative to negative ones; (iii) afterwards
    the KL to the reference is positive and the clipped ratio machinery is live."""
    import torch
    from torch.optim import AdamW
    lg = logging.getLogger("pace.check")
    pol = PolicyLM(cfg, lg, "cpu")
    t = next(x for x in (Turn(**d) for d in load_json(cfg.out / "turns.json")) if x.split == "train")
    samples = pol.generate([pol.prompt(t)] * 2, 4, 1.0, 1.0, 8, seed=3)
    old = pol.batched_logprobs(samples, ref=False, micro=4)
    ref = pol.batched_logprobs(samples, ref=True, micro=4)
    assert all(np.allclose(o, r, atol=1e-5) for o, r in zip(old, ref)), "fresh adapter differs from the reference"
    adv = np.array([1.0, -1.0] * 4)
    opt = AdamW([p for p in pol.model.parameters() if p.requires_grad], lr=5e-2)
    pc = PACEConfig(ppo_epochs=2, minibatch=8, micro=4, max_new_tokens=8)
    pol.ppo_update(samples, adv, old, ref, opt, pc, 0.01, np.random.default_rng(0))
    new = pol.batched_logprobs(samples, ref=False, micro=4)
    gain = sum(a * float(n.sum() - o.sum()) for a, n, o in zip(adv, new, old))
    assert gain > 0, f"the PPO step did not move probability towards positive advantages (gain {gain:+.4f})"
    st = pol.ppo_update(samples, adv, new, ref, opt, pc, 0.01, np.random.default_rng(1))
    assert st["kl"] > 0, "KL to the reference is zero after an update"
    after = pol.batched_logprobs(samples, ref=False, micro=4)
    pol.save_adapter(cfg.out / "check_adapter")
    pol2 = PolicyLM(cfg, lg, "cpu")
    pol2.load_adapter(cfg.out / "check_adapter")
    again = pol2.batched_logprobs(samples, ref=False, micro=4)
    assert all(np.allclose(a, b, atol=1e-5) for a, b in zip(after, again)), "adapter save/load round trip failed"
    del torch
    print(f"policy-gradient path OK (and adapter round trip) | advantage-weighted log-likelihood gain {gain:+.4f} | KL {st['kl']:.5f} | "
          f"clip fraction {st['clipfrac']:.3f}")


def run_selftest(out: Path = Path("pace_selftest")) -> None:
    run_unit_tests()
    import shutil
    if out.exists():
        shutil.rmtree(out)
    data = out / "data"
    make_synthetic_emowoz(data, 160, 0)
    texts = [u["text"] for f in ("emowoz-multiwoz.json", "emowoz-dialmage.json")
             for d in load_json(data / f).values() for u in d["log"]]
    paths = build_tiny_models(out / "tiny", texts)
    cfg = build_config(parse_args([
        "all", "--no-strict", "--device", "cpu", "--data-dir", str(data), "--out", str(out / "run"),
        "--encoder-model", paths["encoder"], "--sentiment-model", paths["sentiment"], "--nli-model", paths["nli"],
        "--policy-model", paths["lm"], "--seeds", "42", "--arms", "pace", "pace_naive_reward",
        "--rl-steps", "3", "--rl-contexts-per-step", "2", "--rl-group", "4", "--rl-max-new-tokens", "12",
        "--eval-turns", "16", "--epochs", "1", "--batch", "16", "--n-heads", "3", "--n-boot", "100",
    ]))
    cfg.pace.min_words, cfg.pace.require_terminal, cfg.pace.allow_truncated, cfg.pace.f_min = 1, False, True, 0.0
    cfg.pace.log_every = 1
    stage_all(cfg)
    _check_policy_gradient_path(cfg)
    rep = load_json(cfg.out / "report.json")
    for arm in ("source", "base", "pace", "pace_naive_reward"):
        assert arm in rep["arms"], f"arm {arm} missing from the report"
    assert (cfg.out / "policy_pace_s42" / "adapter_config.json").exists()
    rw = load_json(cfg.out / "reward.json")
    assert "tau_partial_rho" in rw["gate"] and "bad_news_smd_naive" in rw["gate"]
    print(f"\nSELFTEST OK | pipeline ran end to end on tiny local models | arms {sorted(rep['arms'])}")


# ----------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PACE: content-equivalent affect alignment from implicit emotion")
    p.add_argument("stage", nargs="?", default="all", choices=["all", "data", "detector", "reward", "judge", "train",
                                                               "eval", "report", "unittest", "selftest"])
    p.add_argument("--data-dir", default="emowoz_data")
    p.add_argument("--out", default="pace_run")
    p.add_argument("--device", default=None)
    p.add_argument("--models-dir", default=None, help="local HF cache (models--org--name/snapshots/...)")
    p.add_argument("--download", action="store_true")
    p.add_argument("--no-strict", dest="strict", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--arm", default=None, help="single arm for train/eval")
    p.add_argument("--arms", nargs="+", default=list(Config.arms))
    p.add_argument("--encoder-model", default=Config.encoder_model)
    p.add_argument("--sentiment-model", default=Config.sentiment_model)
    p.add_argument("--nli-model", default=Config.nli_model)
    p.add_argument("--policy-model", default=Config.policy_model)
    p.add_argument("--no-4bit", dest="load_4bit", action="store_false")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--label-source", choices=["detector", "human"], default="detector")
    p.add_argument("--n-heads", type=int, default=5)
    p.add_argument("--crossfit-folds", type=int, default=2)
    p.add_argument("--no-naive-reward", dest="fit_naive_reward", action="store_false")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--enc-lr", type=float, default=2e-5)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--rl-steps", type=int, default=300)
    p.add_argument("--rl-contexts-per-step", type=int, default=8)
    p.add_argument("--rl-group", type=int, default=8)
    p.add_argument("--rl-lr", type=float, default=1e-5)
    p.add_argument("--rl-max-new-tokens", type=int, default=160)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--kl-target", type=float, default=0.05)
    p.add_argument("--kappa", type=float, default=1.0, help="pessimism: reward = mean - kappa * ensemble sd")
    p.add_argument("--f-min", type=float, default=0.5)
    p.add_argument("--eps-fid", type=float, default=0.15)
    p.add_argument("--len-slack", type=float, default=0.6)
    p.add_argument("--eval-turns", type=int, default=800)
    return p.parse_args(list(argv))


def build_config(a: argparse.Namespace) -> Config:
    device = a.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    cfg = Config(data_dir=Path(a.data_dir), out=Path(a.out), device=device, models_dir=a.models_dir,
                 download=bool(a.download), seed=a.seed, seeds=tuple(a.seeds), strict=bool(a.strict),
                 encoder_model=a.encoder_model, sentiment_model=a.sentiment_model, nli_model=a.nli_model,
                 policy_model=a.policy_model, load_4bit=bool(a.load_4bit), lora_r=a.lora_r,
                 label_source=a.label_source, n_heads=a.n_heads, crossfit_folds=a.crossfit_folds,
                 fit_naive_reward=bool(a.fit_naive_reward), n_boot=a.n_boot, eval_turns=a.eval_turns,
                 arms=tuple(a.arms))
    cfg.fit.epochs, cfg.fit.batch, cfg.fit.lr, cfg.fit.max_train = a.epochs, a.batch, a.enc_lr, a.max_train
    pc = cfg.pace
    pc.steps, pc.n_contexts, pc.group_size, pc.lr = a.rl_steps, a.rl_contexts_per_step, a.rl_group, a.rl_lr
    pc.max_new_tokens, pc.ppo_epochs, pc.kl_target, pc.kappa = a.rl_max_new_tokens, a.ppo_epochs, a.kl_target, a.kappa
    pc.f_min, pc.eps_fid, pc.len_slack = a.f_min, a.eps_fid, a.len_slack
    if "pace_naive_reward" in cfg.arms and not cfg.fit_naive_reward:
        cfg.arms = tuple(x for x in cfg.arms if x != "pace_naive_reward")
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if a.stage == "unittest":
        run_unit_tests()
        return
    if a.stage == "selftest":
        run_selftest()
        return
    cfg = build_config(a)
    arms = [a.arm] if a.arm else list(cfg.arms)
    if a.stage == "all":
        stage_all(cfg)
    elif a.stage == "data":
        stage_data(cfg)
    elif a.stage == "detector":
        stage_detector(cfg)
    elif a.stage == "reward":
        stage_reward(cfg)
    elif a.stage == "judge":
        stage_judge(cfg)
    elif a.stage == "train":
        for arm in arms:
            for sd in cfg.seeds:
                stage_train(cfg, arm, sd)
    elif a.stage == "eval":
        for arm in ([a.arm] if a.arm else ["source", "base"] + list(cfg.arms)):
            for sd in cfg.seeds:
                stage_eval(cfg, arm, sd)
    elif a.stage == "report":
        stage_report(cfg, ["source", "base"] + list(cfg.arms))


if __name__ == "__main__":
    main()
