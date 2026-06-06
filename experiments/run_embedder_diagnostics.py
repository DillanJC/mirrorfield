"""Run Embedder Diagnostics on Phase E Test Datasets

This script runs embedder diagnostics on Test A/B/C to identify which metrics
predict REAL_SIGNAL vs COSMETIC verdicts.

Expected findings:
- Test A (REAL_SIGNAL): High stability, moderate concentration
- Test B Strategy A (REAL_SIGNAL): High stability (friction clusters)
- Test B Strategy B (COSMETIC): Low stability (PCA destroys structure)
- Test C (REAL_SIGNAL): High stability, high concentration

Goal: Find 1-2 metrics that reliably predict verdict.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.neighbors import NearestNeighbors
from geometry.embedder_diagnostics import EmbedderDiagnostics


def load_test_embeddings(test_name: str):
    """Load embeddings from Test A/B/C for diagnostics.

    For Test A and C: reconstruct embeddings from artifacts
    For Test B: generate both Strategy A and B embeddings

    Returns:
        dict with keys: "name", "embeddings", "verdict" (ground truth)
    """
    results = []

    if test_name == "test_a":
        # Test A: Real Phase D data
        # Verdict: REAL_SIGNAL (ΔR²=0.14)
        print("Loading Test A embeddings...")

        # Load friction tags to get boundary_distance structure
        friction_path = Path("runs/friction_tags_57a44d005300.json")
        with open(friction_path, "r") as f:
            friction_data = json.load(f)

        boundary_distances = []
        friction_levels = []
        for tag in friction_data["tags"]:
            boundary_distances.append(tag["d_tilde_standardized"])
            friction_levels.append(tag["friction_level"])

        boundary_distances = np.array(boundary_distances, dtype=np.float32)
        n_samples = len(boundary_distances)
        D = 768

        # Generate friction-cluster embeddings (same strategy as Test A)
        rng = np.random.RandomState(42)
        embeddings = []
        for i, (bd, friction) in enumerate(zip(boundary_distances, friction_levels)):
            base = rng.randn(D) * 0.1

            if friction == "low":
                offset = np.array([1.0, 0.0, 0.0] + [0.0]*(D-3))
            elif friction == "medium":
                offset = np.array([0.0, 1.0, 0.0] + [0.0]*(D-3))
            else:  # high
                offset = np.array([0.0, 0.0, 1.0] + [0.0]*(D-3))

            noise = rng.randn(D) * (0.5 + abs(bd))
            emb = base + offset * 2.0 + noise * 0.3
            embeddings.append(emb)

        embeddings = np.array(embeddings, dtype=np.float32)

        results.append({
            "name": "Test A (Real Data, Friction Clusters)",
            "embeddings": embeddings,
            "verdict": "REAL_SIGNAL",
            "delta_r2": 0.1401,
        })

    elif test_name == "test_b":
        # Test B: Model shift (embedder swap)
        # Strategy A: REAL_SIGNAL (ΔR²=0.16)
        # Strategy B: COSMETIC (ΔR²=0.001)
        print("Loading Test B embeddings (both strategies)...")

        # Load friction structure
        friction_path = Path("runs/friction_tags_57a44d005300.json")
        with open(friction_path, "r") as f:
            friction_data = json.load(f)

        boundary_distances = []
        friction_levels = []
        for tag in friction_data["tags"]:
            boundary_distances.append(tag["d_tilde_standardized"])
            friction_levels.append(tag["friction_level"])

        boundary_distances = np.array(boundary_distances, dtype=np.float32)
        n_samples = len(boundary_distances)
        D = 768

        # Strategy A: Friction-based clusters (REAL_SIGNAL)
        rng_a = np.random.RandomState(42)
        embeddings_a = []
        for i, (bd, friction) in enumerate(zip(boundary_distances, friction_levels)):
            base = rng_a.randn(D) * 0.1

            if friction == "low":
                offset = np.array([1.0, 0.0, 0.0] + [0.0]*(D-3))
            elif friction == "medium":
                offset = np.array([0.0, 1.0, 0.0] + [0.0]*(D-3))
            else:
                offset = np.array([0.0, 0.0, 1.0] + [0.0]*(D-3))

            noise = rng_a.randn(D) * (0.5 + abs(bd))
            emb = base + offset * 2.0 + noise * 0.3
            embeddings_a.append(emb)

        embeddings_a = np.array(embeddings_a, dtype=np.float32)

        # Strategy B: PCA-reduced (COSMETIC)
        rng_b = np.random.RandomState(100)
        # Generate high-dim random embeddings
        high_dim_emb = rng_b.randn(n_samples, D * 4).astype(np.float32)
        # PCA reduction to D dimensions
        from sklearn.decomposition import PCA
        pca = PCA(n_components=D, random_state=42)
        embeddings_b = pca.fit_transform(high_dim_emb).astype(np.float32)

        results.append({
            "name": "Test B Strategy A (Friction Clusters)",
            "embeddings": embeddings_a,
            "verdict": "REAL_SIGNAL",
            "delta_r2": 0.1622,
        })

        results.append({
            "name": "Test B Strategy B (PCA Reduced)",
            "embeddings": embeddings_b,
            "verdict": "COSMETIC",
            "delta_r2": 0.0011,
        })

    elif test_name == "test_c":
        # Test C: Targeted construction (adversarial)
        # Verdict: REAL_SIGNAL (ΔR²=0.95)
        print("Loading Test C embeddings...")

        n_ref = 1000
        n_query = 400
        D = 768
        rng = np.random.RandomState(42)

        # Create reference set with two clusters
        # Cluster 1: Flat region (low curvature, low ridge)
        flat_cluster = rng.randn(600, D).astype(np.float32) * 0.3
        flat_cluster[:, 0] -= 2.0  # offset

        # Cluster 2: Ridge structure (high curvature, high ridge)
        ridge_cluster = rng.randn(400, D).astype(np.float32) * 0.8
        ridge_cluster[:, 0] += 2.0  # offset

        # Add explicit ridge structure (line of points)
        for i in range(400):
            t = i / 400.0
            ridge_cluster[i, 0] = 2.0 + t * 4.0  # linear ridge
            ridge_cluster[i, 1:] = rng.randn(D-1) * 0.1  # low variance in other dims

        embeddings = np.vstack([flat_cluster, ridge_cluster])

        results.append({
            "name": "Test C (Targeted Construction)",
            "embeddings": embeddings,
            "verdict": "REAL_SIGNAL",
            "delta_r2": 0.9498,
        })

    return results


def run_diagnostics_comparison():
    """Run diagnostics on all test datasets and compare."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / f"embedder_diagnostics_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all test datasets
    datasets = []
    datasets.extend(load_test_embeddings("test_a"))
    datasets.extend(load_test_embeddings("test_b"))
    datasets.extend(load_test_embeddings("test_c"))

    # Run diagnostics on each
    all_reports = []

    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset['name']}")
        print(f"Ground truth verdict: {dataset['verdict']} (ΔR²={dataset['delta_r2']:.4f})")
        print(f"{'='*80}")

        embeddings = dataset["embeddings"]
        n_samples, n_dims = embeddings.shape
        print(f"Shape: ({n_samples}, {n_dims})")

        # Fit kNN index
        nn_index = NearestNeighbors(n_neighbors=32, metric="euclidean")
        nn_index.fit(embeddings)

        # Run diagnostics
        diagnostics = EmbedderDiagnostics(embeddings, nn_index)
        report = diagnostics.run_all(seed=42)

        # Print report
        diagnostics.print_report(report)

        # Save report
        report_dict = {
            "dataset_name": dataset["name"],
            "ground_truth_verdict": dataset["verdict"],
            "ground_truth_delta_r2": dataset["delta_r2"],
            "predicted_verdict": report.verdict_prediction,
            "confidence": report.confidence,
            "representation_warning": report.representation_warning,
            "metrics": {
                "neighborhood_stability": report.neighborhood_stability,
                "local_intrinsic_dim": report.local_intrinsic_dim,
                "hubness": report.hubness,
                "distance_concentration": report.distance_concentration,
                "curvature_variance": report.curvature_variance,
                "ridge_variance": report.ridge_variance,
            },
            "details": report.details,
        }

        all_reports.append(report_dict)

    # Save all reports
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "n_datasets": len(all_reports),
            "reports": all_reports,
        }, f, indent=2)

    print(f"\nSaved diagnostics summary to: {summary_path}")

    # Analyze which metrics predict verdict
    print(f"\n{'='*80}")
    print("METRIC PREDICTIVENESS ANALYSIS")
    print(f"{'='*80}")

    real_signal_datasets = [r for r in all_reports if r["ground_truth_verdict"] == "REAL_SIGNAL"]
    cosmetic_datasets = [r for r in all_reports if r["ground_truth_verdict"] == "COSMETIC"]

    print(f"\nREAL_SIGNAL datasets (n={len(real_signal_datasets)}):")
    for r in real_signal_datasets:
        m = r["metrics"]
        print(f"  {r['dataset_name'][:40]:40s} | "
              f"stability={m['neighborhood_stability']:.3f}  "
              f"conc={m['distance_concentration']:.3f}  "
              f"lid={m['local_intrinsic_dim']:.1f}  "
              f"hub={m['hubness']:.2f}")

    print(f"\nCOSMETIC datasets (n={len(cosmetic_datasets)}):")
    for r in cosmetic_datasets:
        m = r["metrics"]
        print(f"  {r['dataset_name'][:40]:40s} | "
              f"stability={m['neighborhood_stability']:.3f}  "
              f"conc={m['distance_concentration']:.3f}  "
              f"lid={m['local_intrinsic_dim']:.1f}  "
              f"hub={m['hubness']:.2f}")

    # Compute separability of metrics
    print(f"\n{'─'*80}")
    print("KEY PREDICTIVE METRICS:")

    if len(cosmetic_datasets) > 0:
        # Compute mean differences
        rs_stability = np.mean([r["metrics"]["neighborhood_stability"] for r in real_signal_datasets])
        co_stability = np.mean([r["metrics"]["neighborhood_stability"] for r in cosmetic_datasets])

        rs_conc = np.mean([r["metrics"]["distance_concentration"] for r in real_signal_datasets])
        co_conc = np.mean([r["metrics"]["distance_concentration"] for r in cosmetic_datasets])

        rs_lid = np.mean([r["metrics"]["local_intrinsic_dim"] for r in real_signal_datasets])
        co_lid = np.mean([r["metrics"]["local_intrinsic_dim"] for r in cosmetic_datasets])

        print(f"\n1. Neighborhood Stability:")
        print(f"   REAL_SIGNAL mean: {rs_stability:.3f}")
        print(f"   COSMETIC mean: {co_stability:.3f}")
        print(f"   Δ: {abs(rs_stability - co_stability):.3f} {'★' if abs(rs_stability - co_stability) > 0.2 else ''}")

        print(f"\n2. Distance Concentration:")
        print(f"   REAL_SIGNAL mean: {rs_conc:.3f}")
        print(f"   COSMETIC mean: {co_conc:.3f}")
        print(f"   Δ: {abs(rs_conc - co_conc):.3f} {'★' if abs(rs_conc - co_conc) > 0.1 else ''}")

        print(f"\n3. Local Intrinsic Dimensionality:")
        print(f"   REAL_SIGNAL mean: {rs_lid:.1f}")
        print(f"   COSMETIC mean: {co_lid:.1f}")
        print(f"   Δ: {abs(rs_lid - co_lid):.1f} {'★' if abs(rs_lid - co_lid) > 20 else ''}")

        print(f"\n★ = Strong separator (good predictor)")

    # Compute prediction accuracy
    correct = sum(1 for r in all_reports if r["predicted_verdict"] == r["ground_truth_verdict"])
    accuracy = correct / len(all_reports)

    print(f"\n{'─'*80}")
    print(f"PREDICTION ACCURACY: {correct}/{len(all_reports)} ({accuracy:.1%})")

    if accuracy >= 0.8:
        print("✓ Diagnostics are PREDICTIVE - can reliably warn about COSMETIC embeddings")
    else:
        print("✗ Diagnostics need refinement - prediction accuracy too low")

    print(f"{'='*80}\n")

    return output_dir


if __name__ == "__main__":
    output_dir = run_diagnostics_comparison()
    print(f"Embedder diagnostics complete. Results in: {output_dir}")
