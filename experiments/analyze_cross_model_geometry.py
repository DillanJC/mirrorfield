"""
Cross-model geometric comparison: Claude Sonnet 4.5 vs GLM-4.7 baselines.

Compares geometric traces (embeddings, signatures, curvature, ridge proximity)
across models to test whether different LLMs traverse reasoning space differently.

No API calls — pure analysis of existing baseline data.
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths to baseline results
CLAUDE_BASELINE = (
    PROJECT_ROOT / "experiments" / "results" / "track4_live_baseline"
    / "20260208_221402" / "baseline_results.json"
)
GLM_BASELINE = (
    PROJECT_ROOT / "experiments" / "results" / "glm_baseline"
    / "20260215_083005" / "glm_baseline_results.json"
)

FEATURE_NAMES = [
    'knn_mean_distance', 'knn_std_distance', 'knn_min_distance',
    'knn_max_distance', 'local_curvature', 'ridge_proximity',
    'dist_to_ref_nearest',
]


def load_baseline(path):
    with open(path) as f:
        return json.load(f)


def extract_traces(baseline_data):
    """Extract per-task, per-step feature vectors and signatures."""
    traces = {}
    for result in baseline_data['results']:
        task_id = result['task_id']
        trace = result.get('trace', {})
        steps = trace.get('steps', [])
        traces[task_id] = {
            'domain': result['domain'],
            'quality': result['quality_score']['composite'],
            'steps': [],
        }
        for step in steps:
            traces[task_id]['steps'].append({
                'features': step.get('features', {}),
                'raw_feature_vector': step.get('raw_feature_vector', []),
                'novelty_signature': step.get('novelty_signature', 'unknown'),
            })
    return traces


def all_feature_vectors(traces):
    """Collect all raw feature vectors into a (N, 7) array."""
    vectors = []
    for task_id, task_data in traces.items():
        for step in task_data['steps']:
            vec = step.get('raw_feature_vector', [])
            if len(vec) == 7:
                vectors.append(vec)
    return np.array(vectors) if vectors else np.zeros((0, 7))


def signature_distribution(traces):
    """Count signature occurrences."""
    sigs = []
    for task_data in traces.values():
        for step in task_data['steps']:
            sigs.append(step['novelty_signature'])
    return Counter(sigs)


def per_domain_features(traces):
    """Compute per-domain mean feature vectors."""
    domain_vecs = {}
    for task_data in traces.values():
        domain = task_data['domain']
        for step in task_data['steps']:
            vec = step.get('raw_feature_vector', [])
            if len(vec) == 7:
                domain_vecs.setdefault(domain, []).append(vec)
    return {d: np.array(v) for d, v in domain_vecs.items()}


def trajectory_shape(traces):
    """Compute per-task trajectory statistics (step-over-step feature deltas)."""
    trajectories = {}
    for task_id, task_data in traces.items():
        vecs = []
        for step in task_data['steps']:
            vec = step.get('raw_feature_vector', [])
            if len(vec) == 7:
                vecs.append(vec)
        if len(vecs) >= 2:
            vecs = np.array(vecs)
            deltas = np.diff(vecs, axis=0)  # (N-1, 7)
            trajectories[task_id] = {
                'mean_delta': deltas.mean(axis=0),
                'std_delta': deltas.std(axis=0),
                'total_displacement': vecs[-1] - vecs[0],
                'domain': task_data['domain'],
            }
    return trajectories


def print_separator(title=""):
    print(f"\n{'='*70}")
    if title:
        print(f"  {title}")
        print(f"{'='*70}")


def main():
    print("Cross-Model Geometric Comparison: Claude vs GLM Baselines")
    print("=" * 70)

    # Load data
    claude_data = load_baseline(CLAUDE_BASELINE)
    glm_data = load_baseline(GLM_BASELINE)

    claude_traces = extract_traces(claude_data)
    glm_traces = extract_traces(glm_data)

    print(f"\nClaude: {len(claude_traces)} tasks, model={claude_data['config']['model']}")
    print(f"GLM:    {len(glm_traces)} tasks, model={glm_data['config']['model']}")

    # ================================================================
    # 1. OVERALL FEATURE DISTRIBUTIONS
    # ================================================================
    print_separator("1. OVERALL FEATURE DISTRIBUTIONS")

    claude_vecs = all_feature_vectors(claude_traces)
    glm_vecs = all_feature_vectors(glm_traces)

    print(f"\nClaude: {claude_vecs.shape[0]} step vectors")
    print(f"GLM:    {glm_vecs.shape[0]} step vectors")

    print(f"\n{'Feature':<25s} {'Claude Mean':>12s} {'Claude Std':>12s} {'GLM Mean':>12s} {'GLM Std':>12s} {'Delta':>10s}")
    print("-" * 75)
    for i, name in enumerate(FEATURE_NAMES):
        cm = claude_vecs[:, i].mean()
        cs = claude_vecs[:, i].std()
        gm = glm_vecs[:, i].mean()
        gs = glm_vecs[:, i].std()
        delta = gm - cm
        print(f"{name:<25s} {cm:12.6f} {cs:12.6f} {gm:12.6f} {gs:12.6f} {delta:+10.6f}")

    # Effect sizes (Cohen's d)
    print(f"\n{'Feature':<25s} {'Cohen d':>10s} {'Interpretation':>20s}")
    print("-" * 60)
    for i, name in enumerate(FEATURE_NAMES):
        cm = claude_vecs[:, i].mean()
        gm = glm_vecs[:, i].mean()
        pooled_std = np.sqrt((claude_vecs[:, i].std()**2 + glm_vecs[:, i].std()**2) / 2)
        d = (gm - cm) / pooled_std if pooled_std > 0 else 0
        interp = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        print(f"{name:<25s} {d:+10.4f} {interp:>20s}")

    # ================================================================
    # 2. SIGNATURE DISTRIBUTIONS
    # ================================================================
    print_separator("2. SIGNATURE DISTRIBUTIONS")

    claude_sigs = signature_distribution(claude_traces)
    glm_sigs = signature_distribution(glm_traces)

    all_sig_types = sorted(set(list(claude_sigs.keys()) + list(glm_sigs.keys())))
    total_claude = sum(claude_sigs.values())
    total_glm = sum(glm_sigs.values())

    print(f"\n{'Signature':<25s} {'Claude':>8s} {'%':>6s} {'GLM':>8s} {'%':>6s}")
    print("-" * 55)
    for sig in all_sig_types:
        cc = claude_sigs.get(sig, 0)
        gc = glm_sigs.get(sig, 0)
        cp = cc / total_claude * 100 if total_claude > 0 else 0
        gp = gc / total_glm * 100 if total_glm > 0 else 0
        print(f"{sig:<25s} {cc:8d} {cp:5.1f}% {gc:8d} {gp:5.1f}%")
    print(f"{'TOTAL':<25s} {total_claude:8d}        {total_glm:8d}")

    # ================================================================
    # 3. PER-DOMAIN FEATURE COMPARISON
    # ================================================================
    print_separator("3. PER-DOMAIN FEATURE COMPARISON")

    claude_domain = per_domain_features(claude_traces)
    glm_domain = per_domain_features(glm_traces)

    domains = sorted(set(list(claude_domain.keys()) + list(glm_domain.keys())))

    # Focus on the 3 most interesting features
    key_features = ['local_curvature', 'ridge_proximity', 'knn_mean_distance']
    key_indices = [FEATURE_NAMES.index(f) for f in key_features]

    for domain in domains:
        print(f"\n  {domain}:")
        cv = claude_domain.get(domain, np.zeros((0, 7)))
        gv = glm_domain.get(domain, np.zeros((0, 7)))
        if len(cv) == 0 or len(gv) == 0:
            print("    (insufficient data)")
            continue

        print(f"    {'Feature':<22s} {'Claude':>10s} {'GLM':>10s} {'Delta':>10s}")
        for name, idx in zip(key_features, key_indices):
            cm = cv[:, idx].mean()
            gm = gv[:, idx].mean()
            print(f"    {name:<22s} {cm:10.6f} {gm:10.6f} {gm-cm:+10.6f}")

    # ================================================================
    # 4. TRAJECTORY SHAPE ANALYSIS
    # ================================================================
    print_separator("4. TRAJECTORY SHAPE ANALYSIS")

    claude_traj = trajectory_shape(claude_traces)
    glm_traj = trajectory_shape(glm_traces)

    print("\nMean step-over-step feature deltas (how features change between steps):")
    print(f"\n{'Feature':<25s} {'Claude Mean':>12s} {'GLM Mean':>12s} {'Delta':>10s}")
    print("-" * 60)

    claude_deltas = np.array([t['mean_delta'] for t in claude_traj.values()])
    glm_deltas = np.array([t['mean_delta'] for t in glm_traj.values()])

    for i, name in enumerate(FEATURE_NAMES):
        cm = claude_deltas[:, i].mean()
        gm = glm_deltas[:, i].mean()
        print(f"{name:<25s} {cm:12.6f} {gm:12.6f} {gm-cm:+10.6f}")

    # Total displacement (start-to-end vector)
    print("\nMean total displacement (step 0 -> step 3):")
    claude_disp = np.array([t['total_displacement'] for t in claude_traj.values()])
    glm_disp = np.array([t['total_displacement'] for t in glm_traj.values()])

    print(f"\n{'Feature':<25s} {'Claude':>12s} {'GLM':>12s} {'Delta':>10s}")
    print("-" * 60)
    for i, name in enumerate(FEATURE_NAMES):
        cm = claude_disp[:, i].mean()
        gm = glm_disp[:, i].mean()
        print(f"{name:<25s} {cm:12.6f} {gm:12.6f} {gm-cm:+10.6f}")

    # Magnitude of displacement
    claude_mag = np.linalg.norm(claude_disp, axis=1)
    glm_mag = np.linalg.norm(glm_disp, axis=1)
    print(f"\nTrajectory displacement magnitude:")
    print(f"  Claude: mean={claude_mag.mean():.6f}, std={claude_mag.std():.6f}")
    print(f"  GLM:    mean={glm_mag.mean():.6f}, std={glm_mag.std():.6f}")

    # ================================================================
    # 5. PER-TASK HEAD-TO-HEAD COMPARISON
    # ================================================================
    print_separator("5. PER-TASK HEAD-TO-HEAD COMPARISON")

    common_tasks = sorted(set(claude_traces.keys()) & set(glm_traces.keys()))

    print(f"\n{'Task ID':<18s} {'Domain':<22s} {'Claude Q':>9s} {'GLM Q':>9s} {'Delta':>8s} {'Claude Sig':>12s} {'GLM Sig':>12s}")
    print("-" * 95)

    quality_deltas = []
    for task_id in common_tasks:
        ct = claude_traces[task_id]
        gt = glm_traces[task_id]

        cq = ct['quality']
        gq = gt['quality']
        delta = gq - cq
        quality_deltas.append(delta)

        # Dominant signatures
        c_sigs = [s['novelty_signature'] for s in ct['steps']]
        g_sigs = [s['novelty_signature'] for s in gt['steps']]
        c_dom = Counter(c_sigs).most_common(1)[0][0] if c_sigs else "?"
        g_dom = Counter(g_sigs).most_common(1)[0][0] if g_sigs else "?"

        print(f"{task_id:<18s} {ct['domain']:<22s} {cq:9.3f} {gq:9.3f} {delta:+8.3f} {c_dom:>12s} {g_dom:>12s}")

    quality_deltas = np.array(quality_deltas)
    print(f"\nGLM advantage: mean={quality_deltas.mean():+.4f}, "
          f"wins={np.sum(quality_deltas > 0)}/{len(quality_deltas)}")

    # ================================================================
    # 6. GEOMETRIC SIMILARITY (FEATURE-SPACE DISTANCE)
    # ================================================================
    print_separator("6. GEOMETRIC SIMILARITY (FEATURE-SPACE DISTANCE)")

    # For each task, compute distance between Claude and GLM feature trajectories
    print(f"\n{'Task ID':<18s} {'L2 Distance':>12s} {'Cosine Sim':>12s}")
    print("-" * 45)

    l2_dists = []
    cos_sims = []
    for task_id in common_tasks:
        cv = np.array([s['raw_feature_vector'] for s in claude_traces[task_id]['steps'] if len(s.get('raw_feature_vector', [])) == 7])
        gv = np.array([s['raw_feature_vector'] for s in glm_traces[task_id]['steps'] if len(s.get('raw_feature_vector', [])) == 7])

        if len(cv) == 0 or len(gv) == 0:
            continue

        # Use mean feature vector per task
        cm = cv.mean(axis=0)
        gm = gv.mean(axis=0)

        l2 = np.linalg.norm(cm - gm)
        cos = np.dot(cm, gm) / (np.linalg.norm(cm) * np.linalg.norm(gm) + 1e-8)

        l2_dists.append(l2)
        cos_sims.append(cos)
        print(f"{task_id:<18s} {l2:12.6f} {cos:12.6f}")

    if l2_dists:
        print(f"\nOverall: mean L2={np.mean(l2_dists):.6f}, mean cosine={np.mean(cos_sims):.6f}")
        print(f"  L2 range: [{np.min(l2_dists):.6f}, {np.max(l2_dists):.6f}]")
        print(f"  Cosine range: [{np.min(cos_sims):.6f}, {np.max(cos_sims):.6f}]")

    # ================================================================
    # 7. PER-DOMAIN TRAJECTORY DIVERGENCE
    # ================================================================
    print_separator("7. PER-DOMAIN TRAJECTORY DIVERGENCE")

    for domain in domains:
        domain_tasks = [t for t in common_tasks if claude_traces[t]['domain'] == domain]
        if not domain_tasks:
            continue

        c_domain_vecs = []
        g_domain_vecs = []
        for task_id in domain_tasks:
            for step in claude_traces[task_id]['steps']:
                v = step.get('raw_feature_vector', [])
                if len(v) == 7:
                    c_domain_vecs.append(v)
            for step in glm_traces[task_id]['steps']:
                v = step.get('raw_feature_vector', [])
                if len(v) == 7:
                    g_domain_vecs.append(v)

        cv = np.array(c_domain_vecs)
        gv = np.array(g_domain_vecs)

        cm = cv.mean(axis=0)
        gm = gv.mean(axis=0)
        l2 = np.linalg.norm(cm - gm)
        cos = np.dot(cm, gm) / (np.linalg.norm(cm) * np.linalg.norm(gm) + 1e-8)

        print(f"\n  {domain}:")
        print(f"    L2 distance between mean feature vectors: {l2:.6f}")
        print(f"    Cosine similarity: {cos:.6f}")
        print(f"    Claude curvature: {cv[:, 4].mean():.6f} +/- {cv[:, 4].std():.6f}")
        print(f"    GLM curvature:    {gv[:, 4].mean():.6f} +/- {gv[:, 4].std():.6f}")
        print(f"    Claude ridge_prx: {cv[:, 5].mean():.6f} +/- {cv[:, 5].std():.6f}")
        print(f"    GLM ridge_prx:    {gv[:, 5].mean():.6f} +/- {gv[:, 5].std():.6f}")

    # ================================================================
    # 8. KEY FINDINGS SUMMARY
    # ================================================================
    print_separator("8. KEY FINDINGS SUMMARY")

    # Compute overall differences
    curv_diff = glm_vecs[:, 4].mean() - claude_vecs[:, 4].mean()
    ridge_diff = glm_vecs[:, 5].mean() - claude_vecs[:, 5].mean()
    knn_diff = glm_vecs[:, 0].mean() - claude_vecs[:, 0].mean()
    dist_diff = glm_vecs[:, 6].mean() - claude_vecs[:, 6].mean()

    print(f"""
    1. QUALITY: GLM slightly outperforms Claude overall
       Claude: {claude_data['summary']['mean_quality']:.4f}, GLM: {glm_data['summary']['mean_quality']:.4f}
       Delta: {glm_data['summary']['mean_quality'] - claude_data['summary']['mean_quality']:+.4f}

    2. GEOMETRIC FEATURES:
       Curvature:       GLM {curv_diff:+.6f} vs Claude
       Ridge proximity: GLM {ridge_diff:+.6f} vs Claude
       KNN mean dist:   GLM {knn_diff:+.6f} vs Claude
       Nearest ref:     GLM {dist_diff:+.6f} vs Claude

    3. SIGNATURE PATTERN:
       Claude: {dict(claude_sigs)}
       GLM:    {dict(glm_sigs)}

    4. TRAJECTORY:
       Claude displacement magnitude: {claude_mag.mean():.6f}
       GLM displacement magnitude:    {glm_mag.mean():.6f}
       GLM trajectories are {'more' if glm_mag.mean() > claude_mag.mean() else 'less'} dynamic

    5. GEOMETRIC SIMILARITY:
       Mean L2 distance:   {np.mean(l2_dists):.6f}
       Mean cosine sim:    {np.mean(cos_sims):.6f}
       Both models traverse {'similar' if np.mean(cos_sims) > 0.95 else 'distinct'} regions of feature space
    """)

    # Save results
    output = {
        'summary': {
            'claude_quality': claude_data['summary']['mean_quality'],
            'glm_quality': glm_data['summary']['mean_quality'],
            'quality_delta': glm_data['summary']['mean_quality'] - claude_data['summary']['mean_quality'],
            'mean_l2_distance': float(np.mean(l2_dists)),
            'mean_cosine_similarity': float(np.mean(cos_sims)),
            'claude_displacement_magnitude': float(claude_mag.mean()),
            'glm_displacement_magnitude': float(glm_mag.mean()),
        },
        'feature_comparison': {
            name: {
                'claude_mean': float(claude_vecs[:, i].mean()),
                'claude_std': float(claude_vecs[:, i].std()),
                'glm_mean': float(glm_vecs[:, i].mean()),
                'glm_std': float(glm_vecs[:, i].std()),
                'delta': float(glm_vecs[:, i].mean() - claude_vecs[:, i].mean()),
            }
            for i, name in enumerate(FEATURE_NAMES)
        },
        'signature_distribution': {
            'claude': dict(claude_sigs),
            'glm': dict(glm_sigs),
        },
        'per_task_comparison': [
            {
                'task_id': task_id,
                'domain': claude_traces[task_id]['domain'],
                'claude_quality': claude_traces[task_id]['quality'],
                'glm_quality': glm_traces[task_id]['quality'],
                'quality_delta': glm_traces[task_id]['quality'] - claude_traces[task_id]['quality'],
            }
            for task_id in common_tasks
        ],
    }

    out_path = PROJECT_ROOT / "experiments" / "results" / "cross_model_comparison.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
