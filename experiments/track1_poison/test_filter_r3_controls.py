"""
R3 confound controls — is the surprising R3 positive (Sati AUC 0.725) real
backdoor detection, or an artifact of inserting the SAME rare token "cf" into
every poisoned sentence (which recreates a geometric cluster = the easy R0 case)?

We rerun the SAME pipeline (real SST-2, local MiniLM embed, existing
score_regime) under variants that isolate the confound:

  BASELINE  shared "cf" + label flip            (reproduce R3: expect ~0.725)
  A_varied  DIFFERENT rare token per sample + label flip
            If Sati collapses to chance -> the signal was the repeated-token
            cluster, NOT backdoor geometry.  If it holds -> genuinely interesting.
  B_noflip  shared "cf", NO label flip
            Sati ignores labels, so if it still flags these it proves Sati keys
            on the inserted token's geometric dent, not on "poison".
  C_clean   no trigger, random ids, label flip only (== R1 on this dataset)
            geometry-blind sanity floor; expect ~chance.

Honest signal = Sati AUC. Free, local, no API key. Reuses cached SST-2.

Author: experiment harness (read-only w.r.t. existing code)
"""

import sys
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(Path(__file__).parent))
from test_filter_blind import score_regime
from test_filter_r3_real import load_sst2, embed, SEED, POISON_RATE, TRIGGER, TARGET_LABEL, N_SAMPLES

# pool of rare-ish tokens for the "varied trigger" arm (distinct per sample)
RARE_TOKENS = ["cf", "mn", "bb", "tq", "qx", "zk", "wv", "jf", "xn", "vt",
               "kp", "rz", "gq", "yw", "fx", "dq", "lk", "mz", "pv", "hx"]


def insert_token(text, token, rng):
    words = text.split()
    pos = int(rng.integers(0, len(words) + 1)) if words else 0
    words.insert(pos, token)
    return " ".join(words)


def make_arm(texts, labels, rng, mode):
    """Return (poisoned_texts, poisoned_labels, mask) for a given arm."""
    n = len(texts)
    n_p = int(n * POISON_RATE)
    idx = rng.choice(n, n_p, replace=False)
    pt = list(texts)
    yp = labels.copy()
    for j, i in enumerate(idx):
        if mode in ("baseline", "b_noflip"):
            pt[i] = insert_token(pt[i], TRIGGER, rng)
        elif mode == "a_varied":
            tok = RARE_TOKENS[j % len(RARE_TOKENS)]   # distinct token per sample
            pt[i] = insert_token(pt[i], tok, rng)
        # c_clean: no token inserted
        if mode != "b_noflip":
            yp[i] = TARGET_LABEL
    mask = np.zeros(n, bool); mask[idx] = True
    return pt, yp, mask


def main():
    rng0 = np.random.default_rng(SEED)
    texts, labels = load_sst2(N_SAMPLES, rng0)
    print(f"SST-2: {len(texts)} sentences, pos={int((labels==1).sum())} "
          f"neg={int((labels==0).sum())}")

    arms = ["baseline", "a_varied", "b_noflip", "c_clean"]
    results = {}
    for mode in arms:
        rng = np.random.default_rng(SEED)        # identical ids across arms
        pt, yp, mask = make_arm(texts, labels, rng, mode)
        X = embed(pt)
        res = score_regime(X, yp, mask, np.random.default_rng(SEED))
        results[mode] = res
        print(f"\n[{mode:9}] n_poison={res['n_poison']}  "
              f"Sati AUC={res['auc_sati']:.3f}  "
              f"compressed_ratio={res['compressed_ratio']:.2f}x  "
              f"neg-control={res['auc_combined_shuffled_control']:.3f}")

    out_path = Path(__file__).parent / "r3_controls_results.json"
    out_path.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 64)
    print("CONFOUND VERDICT (Sati = honest geometry signal)")
    print("=" * 64)
    b = results["baseline"]["auc_sati"]
    a = results["a_varied"]["auc_sati"]
    nf = results["b_noflip"]["auc_sati"]
    c = results["c_clean"]["auc_sati"]
    print(f"  baseline (shared 'cf' + flip) : {b:.3f}  (reproduce R3)")
    print(f"  A varied token + flip         : {a:.3f}")
    print(f"  B shared 'cf', NO flip        : {nf:.3f}")
    print(f"  C no token, flip only (=R1)   : {c:.3f}  (chance floor)")
    print()
    if a < 0.6 and b > 0.65:
        print("=> A collapses while baseline holds: the R3 'win' was the REPEATED-")
        print("   TOKEN cluster, not backdoor detection. Same artifact as R0.")
    elif a > 0.65:
        print("=> A holds with DISTINCT tokens: Sati tracks something beyond a")
        print("   single repeated dent. Genuinely worth chasing.")
    if nf > 0.65:
        print("=> B (no flip) still flags: Sati keys on the inserted TOKEN's")
        print("   geometry, independent of the label/poison. Confirms it detects")
        print("   text perturbation, not 'poison' per se.")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
