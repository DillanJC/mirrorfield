"""Phase E Test — SVD Equivalence (Math Correctness)

This test validates the mathematical identity used in local_curvature_svd_gpu:

For centered X ∈ ℝ^(k×D), the covariance eigenvalues λ_i are proportional to s_i^2
where s_i are the singular values of X.

This prevents subtle indexing bugs and ensures correctness of the low-rank residual
energy computation.

Critical: This test MUST pass before Phase E can be considered valid.
"""

import numpy as np
import pytest


def residual_energy_eig(Xc: np.ndarray, r: int) -> float:
    """Compute residual energy using eigendecomposition (ground truth).

    Args:
        Xc: (k, D) centered neighborhood
        r: number of top components to keep

    Returns:
        residual_fraction: energy beyond top-r components
    """
    k = Xc.shape[0]
    # Covariance: C = X^T X / k
    cov = (Xc.T @ Xc) / max(1, k)
    # Eigenvalues in ASCENDING order
    w = np.linalg.eigvalsh(cov)
    total = float(w.sum())
    # Top r eigenvalues are the LAST r in ascending order
    top_r = float(w[-r:].sum()) if r > 0 else 0.0
    return (total - top_r) / (total + 1e-12)


def residual_energy_svd(Xc: np.ndarray, r: int) -> float:
    """Compute residual energy using SVD (Phase E implementation).

    Args:
        Xc: (k, D) centered neighborhood
        r: number of top components to keep

    Returns:
        residual_fraction: energy beyond top-r components
    """
    # Singular values in DESCENDING order
    s = np.linalg.svd(Xc, compute_uv=False)
    e = s * s  # energy = singular_value^2
    # Top r singular values are the FIRST r in descending order
    # Residual is everything AFTER position r
    return float(e[r:].sum()) / (float(e.sum()) + 1e-12)


def test_svd_equivalence_default():
    """Test SVD vs eigendecomp equivalence with default parameters."""
    test_svd_equivalence(seed=0, k=16, D=768, r=4, tol=1e-6)


def test_svd_equivalence_small():
    """Test with small dimensions."""
    test_svd_equivalence(seed=1, k=8, D=64, r=2, tol=1e-6)


def test_svd_equivalence_large_r():
    """Test with larger r (more components kept)."""
    test_svd_equivalence(seed=2, k=32, D=768, r=8, tol=1e-6)


def test_svd_equivalence_edge_case_r0():
    """Test edge case: r=0 (keep no components, all is residual)."""
    test_svd_equivalence(seed=3, k=16, D=768, r=0, tol=1e-6)


def test_svd_equivalence_edge_case_r_max():
    """Test edge case: r=k-1 (keep almost all components)."""
    k = 16
    test_svd_equivalence(seed=4, k=k, D=768, r=k-1, tol=1e-6)


def test_svd_equivalence(
    seed: int = 0,
    k: int = 16,
    D: int = 768,
    r: int = 4,
    tol: float = 1e-6,
):
    """Core equivalence test.

    Args:
        seed: random seed
        k: neighborhood size
        D: embedding dimension
        r: number of components to keep
        tol: numerical tolerance for equivalence
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(k, D).astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)

    residual_eig = residual_energy_eig(Xc, r=r)
    residual_svd = residual_energy_svd(Xc, r=r)

    diff = abs(residual_eig - residual_svd)

    assert diff < tol, (
        f"SVD vs eigendecomp mismatch!\n"
        f"  seed={seed}, k={k}, D={D}, r={r}\n"
        f"  eig={residual_eig:.10f}\n"
        f"  svd={residual_svd:.10f}\n"
        f"  diff={diff:.2e} (tol={tol:.2e})"
    )


def test_svd_equivalence_multiple_seeds():
    """Test across multiple random seeds for robustness."""
    for seed in range(10):
        test_svd_equivalence(seed=seed, k=16, D=768, r=4, tol=1e-6)


if __name__ == "__main__":
    # Run standalone
    test_svd_equivalence_default()
    test_svd_equivalence_small()
    test_svd_equivalence_large_r()
    test_svd_equivalence_edge_case_r0()
    test_svd_equivalence_edge_case_r_max()
    test_svd_equivalence_multiple_seeds()
    print("PASS: All SVD equivalence tests passed")
