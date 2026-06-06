"""Quick diagnostic: what states does EnhancedTracer actually produce?"""
import sys, json, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.shared.embedder import ResponseEmbedder
from experiments.shared.enhanced_tracer import EnhancedTracer, STATE_TO_SIGNAL
from experiments.track4_switches.run_live_baseline import build_reference_texts

# Load the v2 results to get actual responses
v2_dir = PROJECT_ROOT / "experiments" / "results" / "glm_learned_recursive_v2"
latest = sorted(v2_dir.iterdir())[-1]
with open(latest / "glm_learned_recursive_v2_trials.json") as f:
    data = json.load(f)

# Grab iter 1 responses (before any policy changes)
iter1 = data['iterations'][0]['results']
print(f"Using {len(iter1)} tasks from iteration 1\n")

# Build reference + embedder
embedder = ResponseEmbedder()
ref_texts = build_reference_texts()
reference = embedder.build_reference_corpus(ref_texts)
k = min(50, len(reference) - 1)

# Build tracer
tracer = EnhancedTracer(reference, k=k)

# Embed each response step and trace
print("=" * 70)
print("PER-STEP STATE SCORES")
print("=" * 70)

all_scores = {s: [] for s in STATE_TO_SIGNAL}
all_signals = []

for task_result in iter1:
    task_id = task_result['task_id']
    # Re-embed the full response as a single step (approximation)
    resp = task_result['response']
    # Split into steps if possible
    steps = resp.split("\n\n")
    for si, step_text in enumerate(steps[:4]):
        emb = embedder.embed(step_text)
        norm = np.linalg.norm(emb)
        emb_norm = emb / (norm + 1e-8)

        trace = tracer.trace_step(emb_norm, si)

        for state, score in trace.state_scores.items():
            all_scores[state].append(score)
        all_signals.append(trace.novelty_signature)

        if si == 0:  # Print first step of each task
            scores_str = " | ".join(f"{s[:4]}={v:.3f}" for s, v in sorted(trace.state_scores.items()))
            print(f"  [{task_id} s{si}] {trace.dominant_state:20s} -> {trace.novelty_signature:20s} | {scores_str}")

    tracer.reset_trajectory()

print(f"\n{'=' * 70}")
print("AGGREGATE STATE DISTRIBUTION")
print("=" * 70)
from collections import Counter
sig_counts = Counter(all_signals)
total = len(all_signals)
for sig, count in sig_counts.most_common():
    print(f"  {sig:25s}: {count:3d}/{total} ({100*count/total:.1f}%)")

print(f"\n{'=' * 70}")
print("STATE SCORE STATISTICS (across all steps)")
print("=" * 70)
for state in sorted(all_scores):
    vals = np.array(all_scores[state])
    print(f"  {state:22s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
          f"min={vals.min():.4f}  max={vals.max():.4f}  "
          f"p25={np.percentile(vals,25):.4f}  p75={np.percentile(vals,75):.4f}")

print(f"\n{'=' * 70}")
print("REFERENCE PERCENTILE BOUNDARIES (from calibration)")
print("=" * 70)
for key, val in sorted(tracer._ref_percentiles.items()):
    print(f"  {key:30s}: {val:.6f}")

# Now compute what the LLM step feature distributions actually look like
print(f"\n{'=' * 70}")
print("LLM STEP FEATURE VALUES (what we're comparing against ref boundaries)")
print("=" * 70)

from mirrorfield.geometry.unified_pipeline import compute_safety_diagnostics as csd

all_g_mag = []
all_turb = []
all_delta_rho = []
all_knn_std = []
all_knn_max = []
all_ridge = []

for task_result in iter1:
    steps = task_result['response'].split("\n\n")
    for step_text in steps[:4]:
        emb = embedder.embed(step_text)
        norm = np.linalg.norm(emb)
        emb_norm = (emb / (norm + 1e-8)).reshape(1, -1)

        diag = csd(emb_norm, reference, k=k, include_full_phase1=True)
        t0 = diag.tier0_features[0]
        p1 = diag.phase1_features[0] if diag.phase1_features.ndim > 1 else diag.phase1_features.flatten()
        p25 = diag.phase2_5_features[0]

        all_g_mag.append(float(p1[0]))
        all_delta_rho.append(float(p1[1]) if len(p1) > 1 else 0.0)
        all_turb.append(float(p25[0]))
        all_knn_std.append(float(t0[1]))
        all_knn_max.append(float(t0[3]))
        all_ridge.append(float(t0[5]))

def show(name, vals, ref_lo_key, ref_hi_key):
    a = np.array(vals)
    lo = tracer._ref_percentiles.get(ref_lo_key, 0)
    hi = tracer._ref_percentiles.get(ref_hi_key, 0)
    below = (a < lo).sum()
    above = (a > hi).sum()
    between = len(a) - below - above
    print(f"  {name:15s}: mean={a.mean():.4f} std={a.std():.4f} "
          f"[{a.min():.4f}, {a.max():.4f}] | "
          f"ref_lo={lo:.4f} ref_hi={hi:.4f} | "
          f"below={below} between={between} above={above}")

show("g_mag", all_g_mag, "g_mag_p30", "g_mag_p70")
show("turbulence", all_turb, "turbulence_p30", "turbulence_p70")
show("delta_rho", all_delta_rho, "delta_rho_p70", "delta_rho_p70")
show("knn_std", all_knn_std, "knn_std_p30", "knn_std_p70")
show("knn_max", all_knn_max, "knn_max_p80", "knn_max_p80")
show("ridge_prox", all_ridge, "ridge_prox_p40", "ridge_prox_p70")

# What SHOULD the percentiles be if we calibrated from LLM steps?
print(f"\n{'=' * 70}")
print("PROPOSED LLM-CALIBRATED BOUNDARIES")
print("=" * 70)
for name, vals, pcts in [
    ("g_mag", all_g_mag, [30, 50, 70]),
    ("turbulence", all_turb, [30, 70]),
    ("delta_rho", all_delta_rho, [70]),
    ("knn_std", all_knn_std, [30, 70]),
    ("knn_max", all_knn_max, [80]),
    ("ridge_prox", all_ridge, [40, 70, 80]),
]:
    a = np.array(vals)
    pct_str = " | ".join(f"p{p}={np.percentile(a, p):.6f}" for p in pcts)
    print(f"  {name:15s}: {pct_str}")

print("\n[DONE]")
