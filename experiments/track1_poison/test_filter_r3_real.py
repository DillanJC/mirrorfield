"""
R3 — the real external test. Does the geometry filter detect a REAL,
attacker-chosen text backdoor (not geometry-defined poison)?

Where R0/R1/R2 (test_filter_blind.py) varied poison in EMBEDDING space, R3
poisons in TEXT space the way a real attacker would: insert a fixed trigger
token into a random subset of sentences and flip their labels (BadNets / AddSent
style, the OpenBackdoor threat model). We then embed clean+poisoned text with the
SAME embedder that produced embeddings.npy (OpenAI text-embedding-3-large @ 256d,
L2-normalized) and feed (emb, labels, mask) straight into the existing
regime-agnostic score_regime() from test_filter_blind.py.

Honest signal = Sati AUC. (GMR auc is excluded from the verdict for the same
reason as the blind test: it detects the label flip itself, not geometry.)

Prediction (consistent with R1=0.473, R2=0.475): Sati AUC ~ chance. A clear
positive here would be the genuinely surprising result worth chasing.

SECURITY / COST NOTES
---------------------
* The OpenAI key is read from the environment (OPENAI_API_KEY), NEVER hardcoded.
  Set it in your own shell:  setx OPENAI_API_KEY "sk-..."  (then reopen shell)
  or put it in a .env file that is git-ignored.
* This script makes a PAID embedding call and DOWNLOADS the SST-2 dataset.
  Both are gated behind --run; with --dry-run (default) it does neither.
* ~1-2k short sentences at text-embedding-3-large @ 256d costs well under 5 cents.

Requires (install yourself when ready):  pip install datasets openai python-dotenv

Author: experiment harness (read-only w.r.t. existing code)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]          # .../mirrorfield
sys.path.insert(0, str(BASE))

# Reuse the EXACT scoring logic already written/validated.
sys.path.insert(0, str(Path(__file__).parent))
from test_filter_blind import score_regime          # regime-agnostic scorer

SEED = 42
POISON_RATE = 0.05
TRIGGER = "cf"          # classic BadNets rare-token trigger (OpenBackdoor default)
TARGET_LABEL = 1        # attacker forces these poisoned samples to label 1
N_SAMPLES = 1099        # match the size of the original embeddings.npy run
# Local, free, no-key embedder (already cached on this machine, also used
# elsewhere in this repo per FINDINGS.md). 384-dim, not the original OpenAI
# 256-dim -> fair falsification test of the geometry hypothesis, not a
# byte-identical reproduction of the old pipeline.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Data: real SST-2 with a real text-space backdoor trigger
# ---------------------------------------------------------------------------
def load_sst2(n_samples, rng):
    """Load SST-2 train split via HuggingFace `datasets` (network download)."""
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", split="train")
    idx = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    texts = [ds[int(i)]["sentence"].strip() for i in idx]
    labels = np.array([int(ds[int(i)]["label"]) for i in idx], dtype=int)
    return texts, labels


def inject_text_backdoor(texts, labels, rng, rate=POISON_RATE, trigger=TRIGGER,
                         target=TARGET_LABEL):
    """BadNets-style: pick RANDOM ids, splice trigger token in, set target label.

    Selection is purely id-based (geometry-blind, attacker-chosen) — exactly the
    case the geometry hypothesis must handle to be more than circular.
    Returns poisoned texts, poisoned labels, and the ground-truth poison mask.
    """
    n = len(texts)
    n_p = int(n * rate)
    idx = rng.choice(n, n_p, replace=False)
    poisoned_texts = list(texts)
    yp = labels.copy()
    for i in idx:
        words = poisoned_texts[i].split()
        pos = rng.integers(0, len(words) + 1) if words else 0
        words.insert(int(pos), trigger)          # insert rare trigger token
        poisoned_texts[i] = " ".join(words)
        yp[i] = target                            # force attacker target label
    mask = np.zeros(n, bool)
    mask[idx] = True
    return poisoned_texts, yp, mask


def embed(texts):
    """Embed locally with a cached sentence-transformers model. No API, no key,
    no network for weights (model is already in the HF cache)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    X = model.encode(texts, batch_size=64, show_progress_bar=True,
                     convert_to_numpy=True)
    X = np.asarray(X, dtype=np.float32)
    # L2-normalize to match the unit-norm convention of the blind-test embeddings.
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true",
                    help="Download SST-2 (free) and embed locally with MiniLM.")
    ap.add_argument("--n", type=int, default=N_SAMPLES)
    ap.add_argument("--cache", type=str, default=str(Path(__file__).parent /
                                                     "r3_embeddings_cache.npz"),
                    help="Reuse embeddings across runs to avoid re-embedding.")
    args = ap.parse_args()

    if not args.run:
        print(__doc__)
        print("\n[dry-run] No download, no embedding.")
        print("When ready:  python test_filter_r3_real.py --run")
        return

    rng = np.random.default_rng(SEED)
    cache = Path(args.cache)

    if cache.exists():
        print(f"Loading cached embeddings -> {cache}")
        d = np.load(cache, allow_pickle=False)
        Xp, yp, mask = d["X"], d["y"], d["mask"]
    else:
        print("Loading SST-2 (HuggingFace download)...")
        texts, labels = load_sst2(args.n, rng)
        print(f"  {len(texts)} sentences, pos={int((labels==1).sum())} "
              f"neg={int((labels==0).sum())}")
        ptexts, yp, mask = inject_text_backdoor(texts, labels, rng)
        print(f"  injected trigger '{TRIGGER}' into {int(mask.sum())} samples "
              f"(rate={POISON_RATE})")
        print("Embedding locally with MiniLM (free, no key)...")
        Xp = embed(ptexts)
        np.savez(cache, X=Xp, y=yp, mask=mask)
        print(f"  cached -> {cache}")

    print(f"\nEmbeddings {Xp.shape}  (MiniLM, 384-dim local)")
    res = score_regime(Xp, yp, mask, np.random.default_rng(SEED))

    out = {"config": {"seed": SEED, "poison_rate": POISON_RATE, "trigger": TRIGGER,
                      "n": int(len(Xp)), "dim": int(Xp.shape[1]),
                      "embedder": EMBED_MODEL, "dataset": "glue/sst2"},
           "caveat": ("auc_gmr trivially flags any label flip; it is NOT evidence "
                      "for geometry. Use auc_sati as the honest signal."),
           "result": res}
    out_path = Path(__file__).parent / "r3_real_results.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 60)
    print("R3 REAL TEXT-BACKDOOR RESULT (Sati = honest geometry signal)")
    print("=" * 60)
    print(f"  Sati AUC     = {res['auc_sati']:.3f}")
    print(f"  GMR AUC      = {res['auc_gmr']:.3f}   (excluded from verdict)")
    print(f"  combined AUC = {res['auc_combined']:.3f}")
    print(f"  neg-control  = {res['auc_combined_shuffled_control']:.3f}  "
          f"(must be ~0.5)")
    print(f"  compressed_ratio = {res['compressed_ratio']:.2f}x")
    s = res["auc_sati"]
    if s < 0.6:
        print("\n=> Sati ~chance on a REAL text backdoor. Geometry-as-poison-"
              "detector is confirmed circular; close this line.")
    elif s > 0.65:
        print("\n=> SURPRISE: Sati has real signal on an attacker-chosen text "
              "backdoor. This is the result worth chasing.")
    else:
        print("\n=> Weak/ambiguous signal; inconclusive.")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
