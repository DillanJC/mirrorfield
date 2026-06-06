"""Test 1: Lock the LID=4 Claim

Validates that LID=4 is robust across:
- Sample size (N = 100 → 5000)
- Neighborhood size (k = 5, 10, 20, 50, 100)
- Negative controls (Gaussian noise, known 4D manifold)

Timeline: ~2 hours (mostly compute for large N, high k)
Cost: ~$0.001 USD if generating new embeddings for N=5000
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import sys
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from geometry.embedder_diagnostics import EmbedderDiagnostics


def generate_gaussian_noise(n_samples, n_dims, seed=42):
    """Generate Gaussian noise in n_dims.

    Expected LID ≈ n_dims (curse of dimensionality).
    """
    rng = np.random.RandomState(seed)
    return rng.randn(n_samples, n_dims).astype(np.float32)


def generate_4d_manifold_embedded(n_samples, ambient_dim=256, manifold_type="sphere", seed=42):
    """Generate samples from a known 4D manifold embedded in ambient_dim space.

    Expected LID ≈ 4.

    Args:
        manifold_type: "sphere" (S³), "torus" (T⁴), or "gaussian" (4D Gaussian cloud)
    """
    rng = np.random.RandomState(seed)

    if manifold_type == "sphere":
        # 4D sphere (S³) embedded in ambient_dim-D space
        # Generate 4D points on unit sphere, then embed in high-D space
        points_4d = rng.randn(n_samples, 4)
        points_4d /= np.linalg.norm(points_4d, axis=1, keepdims=True)

    elif manifold_type == "torus":
        # 4D torus (T⁴ = S¹ × S¹ × S¹ × S¹)
        angles = rng.uniform(0, 2*np.pi, (n_samples, 4))
        points_4d = np.column_stack([
            np.cos(angles[:, 0]), np.sin(angles[:, 0]),
            np.cos(angles[:, 1]), np.sin(angles[:, 1]),
        ])[:, :4]  # 4D representation

    elif manifold_type == "gaussian":
        # 4D Gaussian cloud (simplest case)
        points_4d = rng.randn(n_samples, 4).astype(np.float32)

    else:
        raise ValueError(f"Unknown manifold_type: {manifold_type}")

    # Embed in ambient_dim-D space via random projection
    projection_matrix = rng.randn(4, ambient_dim).astype(np.float32)
    projection_matrix /= np.linalg.norm(projection_matrix, axis=0, keepdims=True)

    embedded = points_4d @ projection_matrix

    return embedded.astype(np.float32)


def load_real_embeddings(run_dir, max_samples=None):
    """Load real embeddings from a test run."""
    run_path = Path(run_dir)
    embeddings_path = run_path / "embeddings.npy"

    if not embeddings_path.exists():
        return None

    embeddings = np.load(embeddings_path)

    if max_samples is not None and len(embeddings) > max_samples:
        # Random subsample
        rng = np.random.RandomState(42)
        indices = rng.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]

    return embeddings


def measure_lid_mle(embeddings, k=20, n_samples=None, seed=42):
    """Measure local intrinsic dimensionality using MLE estimator.

    This is the same method used in embedder_diagnostics but standalone.

    Args:
        embeddings: (N, D) array
        k: number of neighbors
        n_samples: number of samples to use (None = all)
        seed: random seed for sampling

    Returns:
        mean_lid: mean LID across all samples
        std_lid: standard deviation of LID
        median_lid: median LID
        all_lids: (n_samples,) array of individual LID values
    """
    N, D = embeddings.shape

    if n_samples is None or n_samples >= N:
        n_samples = N
        sample_indices = np.arange(N)
    else:
        rng = np.random.RandomState(seed)
        sample_indices = rng.choice(N, n_samples, replace=False)

    # Build kNN index
    nn = NearestNeighbors(n_neighbors=min(k+1, N), algorithm='auto', metric='euclidean')
    nn.fit(embeddings)

    # Query samples (k+1 because first neighbor is self)
    distances, _ = nn.kneighbors(embeddings[sample_indices])

    # MLE estimator (Levina & Bickel 2004)
    # LID = (k-1) / sum(log(r_k / r_i)) for i=1..k-1
    # where r_i is distance to i-th nearest neighbor

    lids = []
    for i in range(len(sample_indices)):
        r = distances[i, 1:]  # exclude self (index 0)
        r_k = r[-1]  # k-th neighbor distance

        if r_k < 1e-10:  # degenerate case
            continue

        # Compute log ratios
        log_ratios = np.log(r_k / (r[:-1] + 1e-10))

        if np.any(log_ratios <= 0):  # skip if invalid
            continue

        lid = (k - 1) / (np.sum(log_ratios) + 1e-10)
        lids.append(lid)

    lids = np.array(lids)

    return {
        "mean": np.mean(lids),
        "std": np.std(lids),
        "median": np.median(lids),
        "min": np.min(lids),
        "max": np.max(lids),
        "p25": np.percentile(lids, 25),
        "p75": np.percentile(lids, 75),
        "all_values": lids,
    }


def run_n_sweep(embeddings, k_values=[20], n_values=[100, 200, 500, 1000], name="dataset"):
    """Sweep across sample sizes N."""
    print(f"\n{'='*80}")
    print(f"N-SWEEP: {name}")
    print(f"{'='*80}")

    results = []

    for n in n_values:
        if n > len(embeddings):
            print(f"⚠️  Skipping N={n} (only {len(embeddings)} samples available)")
            continue

        print(f"\nN = {n}")

        for k in k_values:
            if k >= n:
                print(f"  Skipping k={k} (k >= N)")
                continue

            lid_stats = measure_lid_mle(embeddings, k=k, n_samples=n, seed=42)

            print(f"  k={k:3d}: LID = {lid_stats['mean']:6.2f} ± {lid_stats['std']:5.2f} "
                  f"(median={lid_stats['median']:6.2f}, range=[{lid_stats['min']:.1f}, {lid_stats['max']:.1f}])")

            results.append({
                "n": n,
                "k": k,
                "lid_mean": lid_stats["mean"],
                "lid_std": lid_stats["std"],
                "lid_median": lid_stats["median"],
                "lid_min": lid_stats["min"],
                "lid_max": lid_stats["max"],
                "lid_p25": lid_stats["p25"],
                "lid_p75": lid_stats["p75"],
            })

    return results


def run_k_sweep(embeddings, k_values=[5, 10, 20, 50, 100], n_samples=500, name="dataset"):
    """Sweep across neighborhood sizes k."""
    print(f"\n{'='*80}")
    print(f"K-SWEEP: {name}")
    print(f"{'='*80}")

    if n_samples > len(embeddings):
        n_samples = len(embeddings)
        print(f"⚠️  Using N={n_samples} (all available)")

    print(f"\nN = {n_samples} (fixed)")

    results = []

    for k in k_values:
        if k >= n_samples:
            print(f"⚠️  Skipping k={k} (k >= N)")
            continue

        lid_stats = measure_lid_mle(embeddings, k=k, n_samples=n_samples, seed=42)

        print(f"k={k:3d}: LID = {lid_stats['mean']:6.2f} ± {lid_stats['std']:5.2f} "
              f"(median={lid_stats['median']:6.2f}, range=[{lid_stats['min']:.1f}, {lid_stats['max']:.1f}])")

        results.append({
            "k": k,
            "n": n_samples,
            "lid_mean": lid_stats["mean"],
            "lid_std": lid_stats["std"],
            "lid_median": lid_stats["median"],
            "lid_min": lid_stats["min"],
            "lid_max": lid_stats["max"],
            "lid_p25": lid_stats["p25"],
            "lid_p75": lid_stats["p75"],
        })

    return results


def run_negative_controls(n_samples=500, ambient_dim=256):
    """Run negative control tests."""
    print(f"\n{'='*80}")
    print(f"NEGATIVE CONTROLS")
    print(f"{'='*80}")

    results = {}

    # Control 1: Gaussian noise (should give LID ≈ ambient_dim)
    print(f"\nControl 1: Gaussian Noise ({ambient_dim}-D)")
    print("Expected: LID ≈ {ambient_dim} (curse of dimensionality)")

    noise = generate_gaussian_noise(n_samples, ambient_dim, seed=42)
    noise_results = run_k_sweep(noise, k_values=[5, 10, 20, 50], n_samples=n_samples, name="Gaussian Noise")
    results["gaussian_noise"] = noise_results

    # Control 2: Known 4D manifold (should give LID ≈ 4)
    print(f"\nControl 2: Known 4D Manifold (S³ embedded in {ambient_dim}-D)")
    print("Expected: LID ≈ 4")

    manifold_4d = generate_4d_manifold_embedded(n_samples, ambient_dim=ambient_dim, manifold_type="sphere", seed=42)
    manifold_results = run_k_sweep(manifold_4d, k_values=[5, 10, 20, 50], n_samples=n_samples, name="4D Sphere")
    results["known_4d_manifold"] = manifold_results

    # Control 3: 4D Gaussian (simplest 4D case)
    print(f"\nControl 3: 4D Gaussian Cloud (embedded in {ambient_dim}-D)")
    print("Expected: LID ≈ 4")

    gaussian_4d = generate_4d_manifold_embedded(n_samples, ambient_dim=ambient_dim, manifold_type="gaussian", seed=42)
    gaussian_4d_results = run_k_sweep(gaussian_4d, k_values=[5, 10, 20, 50], n_samples=n_samples, name="4D Gaussian")
    results["gaussian_4d"] = gaussian_4d_results

    return results


def plot_stability_results(results_dict, output_dir):
    """Generate plots showing LID stability."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: N-sweep (LID vs sample size)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for dataset_name, sweeps in results_dict.items():
        if "n_sweep" in dataset_name:
            n_values = [r["n"] for r in sweeps]
            lid_means = [r["lid_mean"] for r in sweeps]
            lid_stds = [r["lid_std"] for r in sweeps]

            axes[0].errorbar(n_values, lid_means, yerr=lid_stds,
                           marker='o', label=dataset_name.replace("_n_sweep", ""))

    axes[0].axhline(y=4, color='red', linestyle='--', label='LID=4 (target)')
    axes[0].set_xlabel('Sample Size (N)')
    axes[0].set_ylabel('Local Intrinsic Dimensionality')
    axes[0].set_title('LID Stability Across Sample Sizes')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')

    # Plot 2: K-sweep (LID vs neighborhood size)
    for dataset_name, sweeps in results_dict.items():
        if "k_sweep" in dataset_name:
            k_values = [r["k"] for r in sweeps]
            lid_means = [r["lid_mean"] for r in sweeps]
            lid_stds = [r["lid_std"] for r in sweeps]

            axes[1].errorbar(k_values, lid_means, yerr=lid_stds,
                           marker='s', label=dataset_name.replace("_k_sweep", ""))

    axes[1].axhline(y=4, color='red', linestyle='--', label='LID=4 (target)')
    axes[1].set_xlabel('Neighborhood Size (k)')
    axes[1].set_ylabel('Local Intrinsic Dimensionality')
    axes[1].set_title('LID Stability Across Neighborhood Sizes')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / "lid_stability.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot: {output_dir / 'lid_stability.png'}")
    plt.close()


def main():
    """Run Test 1: LID=4 stability validation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / f"lid_stability_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("TEST 1: LOCK THE LID=4 CLAIM")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Timestamp: {timestamp}\n")

    all_results = {}

    # Load real embeddings (if available)
    print("Loading real OpenAI embeddings...")
    embeddings_3small = load_real_embeddings("runs/openai_3_small_test_20251231_005441")
    embeddings_3large_256d = load_real_embeddings("runs/openai_3_large_test_20251231_005624")
    embeddings_3large_256d_1099 = load_real_embeddings("runs/openai_3_large_test_20251231_024532")

    if embeddings_3small is None or embeddings_3large_256d is None:
        print("⚠️  WARNING: Could not find saved embeddings. Running controls only.")
        print("    To test real embeddings, ensure gate test runs have completed.")

    # Test real embeddings (if available)
    if embeddings_3small is not None:
        print(f"\n{'='*80}")
        print("REAL EMBEDDINGS: text-embedding-3-small (1536-D)")
        print(f"{'='*80}")
        print(f"Available samples: {len(embeddings_3small)}")

        # N-sweep
        n_sweep_results = run_n_sweep(
            embeddings_3small,
            k_values=[20],  # Fix k=20 for N-sweep
            n_values=[100, 200, 500],
            name="3-small"
        )
        all_results["3small_n_sweep"] = n_sweep_results

        # K-sweep
        k_sweep_results = run_k_sweep(
            embeddings_3small,
            k_values=[5, 10, 20, 50, 100],
            n_samples=500,
            name="3-small"
        )
        all_results["3small_k_sweep"] = k_sweep_results

    if embeddings_3large_256d is not None:
        print(f"\n{'='*80}")
        print("REAL EMBEDDINGS: text-embedding-3-large (256-D)")
        print(f"{'='*80}")
        print(f"Available samples: {len(embeddings_3large_256d)}")

        # N-sweep
        n_sweep_results = run_n_sweep(
            embeddings_3large_256d,
            k_values=[20],
            n_values=[100, 200, 500],
            name="3-large-256d"
        )
        all_results["3large256d_n_sweep"] = n_sweep_results

        # K-sweep
        k_sweep_results = run_k_sweep(
            embeddings_3large_256d,
            k_values=[5, 10, 20, 50, 100],
            n_samples=500,
            name="3-large-256d"
        )
        all_results["3large256d_k_sweep"] = k_sweep_results

    if embeddings_3large_256d_1099 is not None:
        print(f"\n{'='*80}")
        print("REAL EMBEDDINGS: text-embedding-3-large (256-D, N=1099)")
        print(f"{'='*80}")
        print(f"Available samples: {len(embeddings_3large_256d_1099)}")

        # Extended N-sweep
        n_sweep_results = run_n_sweep(
            embeddings_3large_256d_1099,
            k_values=[20],
            n_values=[100, 200, 500, 1000],
            name="3-large-256d-extended"
        )
        all_results["3large256d_extended_n_sweep"] = n_sweep_results

        # Extended K-sweep at N=1000
        k_sweep_results = run_k_sweep(
            embeddings_3large_256d_1099,
            k_values=[5, 10, 20, 50, 100],
            n_samples=1000,
            name="3-large-256d-extended"
        )
        all_results["3large256d_extended_k_sweep"] = k_sweep_results

    # Run negative controls
    control_results = run_negative_controls(n_samples=500, ambient_dim=256)
    all_results["controls"] = control_results

    # Generate summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    # Check if LID≈4 is stable
    print("LID=4 Stability Check:")
    print("-" * 80)

    lid_values = []
    for key, results in all_results.items():
        if isinstance(results, list) and len(results) > 0:
            for r in results:
                if "lid_mean" in r:
                    lid_values.append(r["lid_mean"])
                    dataset = key.replace("_n_sweep", "").replace("_k_sweep", "")
                    print(f"{dataset:20s} (N={r.get('n', '?'):4d}, k={r.get('k', '?'):3d}): "
                          f"LID = {r['lid_mean']:5.2f} ± {r['lid_std']:4.2f}")

    if lid_values:
        real_lid_values = [v for k, results in all_results.items()
                          if ("3small" in k or "3large" in k) and isinstance(results, list)
                          for r in results if "lid_mean" in r
                          for v in [r["lid_mean"]]]

        if real_lid_values:
            mean_lid = np.mean(real_lid_values)
            std_lid = np.std(real_lid_values)

            print(f"\nReal Embeddings Overall: LID = {mean_lid:.2f} ± {std_lid:.2f}")

            if 3.5 <= mean_lid <= 4.5 and std_lid < 1.0:
                print("✓ PASS: LID≈4 is stable (3.5 < LID < 4.5, std < 1.0)")
                overall_pass = True
            else:
                print(f"✗ FAIL: LID not stable around 4 (mean={mean_lid:.2f}, std={std_lid:.2f})")
                overall_pass = False
        else:
            overall_pass = None
    else:
        overall_pass = None

    # Check negative controls
    print(f"\nNegative Control Validation:")
    print("-" * 80)

    if "controls" in all_results:
        controls = all_results["controls"]

        if "gaussian_noise" in controls and len(controls["gaussian_noise"]) > 0:
            noise_lid = controls["gaussian_noise"][0]["lid_mean"]  # k=5 result
            print(f"Gaussian Noise (256-D): LID = {noise_lid:.1f} (expected ≈256)")
            if noise_lid > 50:  # Should be much higher than 4
                print("  ✓ PASS: High-D noise shows high LID")
            else:
                print(f"  ✗ FAIL: Expected LID >> 50, got {noise_lid:.1f}")

        if "known_4d_manifold" in controls and len(controls["known_4d_manifold"]) > 0:
            manifold_lid = controls["known_4d_manifold"][0]["lid_mean"]
            print(f"Known 4D Manifold (S³): LID = {manifold_lid:.1f} (expected ≈4)")
            if 3.0 <= manifold_lid <= 5.0:
                print("  ✓ PASS: True 4D manifold shows LID≈4")
            else:
                print(f"  ✗ FAIL: Expected LID≈4, got {manifold_lid:.1f}")

    # Save results
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        # Convert numpy types to Python types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        json.dump({
            "timestamp": timestamp,
            "overall_pass": overall_pass,
            "results": convert(all_results),
        }, f, indent=2)

    print(f"\n✓ Results saved to {summary_path}")

    # Generate plots
    try:
        plot_stability_results(all_results, output_dir)
    except Exception as e:
        print(f"⚠️  Warning: Could not generate plots: {e}")

    print(f"\n{'='*80}")
    if overall_pass:
        print("✓ TEST 1 PASSED: LID=4 claim is LOCKED")
    elif overall_pass is False:
        print("✗ TEST 1 FAILED: LID≠4 or unstable")
    else:
        print("⚠️  TEST 1 INCOMPLETE: Need real embeddings to validate")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
