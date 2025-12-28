"""Phase E Benchmark — GPU-Batched SVD Curvature Performance

This benchmark measures the isolated performance of local_curvature_svd_gpu
on the target hardware (RTX 3060 Ti / 8GB VRAM).

Target performance (estimated):
- 2000 queries, k=16, D=768, r=4, batch_size=256
- GPU: ~2-5 seconds
- CPU: ~20-40 seconds

This is an ESTIMATE, not a promise. Actual performance depends on hardware,
CUDA version, PyTorch version, and system load.
"""

import time
import numpy as np
import torch
import argparse
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geometry.features import local_curvature_svd_gpu


def benchmark_curvature(
    n_ref: int = 5000,
    n_query: int = 2000,
    k: int = 16,
    D: int = 768,
    r: int = 4,
    batch_size: int = 256,
    device: str = "cuda",
    use_amp: bool = False,
    seed: int = 42,
):
    """Run curvature benchmark.

    Args:
        n_ref: reference set size
        n_query: number of queries to process
        k: neighborhood size
        D: embedding dimension
        r: number of components to keep
        batch_size: GPU batch size
        device: 'cuda' or 'cpu'
        use_amp: enable mixed precision
        seed: random seed

    Returns:
        dict with timing and throughput metrics
    """
    print(f"\n{'='*60}")
    print(f"Phase E SVD Curvature Benchmark")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  n_ref={n_ref}, n_query={n_query}")
    print(f"  k={k}, D={D}, r={r}")
    print(f"  batch_size={batch_size}")
    print(f"  device={device}, use_amp={use_amp}")
    print(f"  seed={seed}")
    print(f"{'='*60}\n")

    # Setup
    rng = np.random.RandomState(seed)
    ref_points = rng.randn(n_ref, D).astype(np.float32)
    neighbor_indices = rng.randint(0, n_ref, size=(n_query, k), dtype=np.int64)

    # Warmup (important for GPU)
    print("Warming up...")
    warmup_size = min(512, n_query)
    _ = local_curvature_svd_gpu(
        ref_points=ref_points,
        neighbor_indices=neighbor_indices[:warmup_size],
        r=r,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
    )

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    print(f"Running benchmark ({n_query} queries)...")
    t0 = time.time()

    curvature = local_curvature_svd_gpu(
        ref_points=ref_points,
        neighbor_indices=neighbor_indices,
        r=r,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
    )

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    dt = time.time() - t0

    # Results
    throughput = n_query / dt
    mean_curv = float(np.mean(curvature))
    std_curv = float(np.std(curvature))

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Device: {device}")
    print(f"  Time: {dt:.3f} seconds")
    print(f"  Throughput: {throughput:.0f} queries/sec")
    print(f"  Curvature stats: mean={mean_curv:.4f}, std={std_curv:.4f}")
    print(f"{'='*60}\n")

    return {
        "device": device,
        "n_ref": n_ref,
        "n_query": n_query,
        "k": k,
        "D": D,
        "r": r,
        "batch_size": batch_size,
        "use_amp": use_amp,
        "time_seconds": dt,
        "throughput_per_sec": throughput,
        "curvature_mean": mean_curv,
        "curvature_std": std_curv,
    }


def benchmark_batch_size_sweep(device: str = "cuda"):
    """Benchmark across different batch sizes to find optimal setting."""
    print(f"\n{'='*60}")
    print(f"Batch Size Sweep ({device})")
    print(f"{'='*60}\n")

    results = []
    for batch_size in [64, 128, 256, 512]:
        try:
            result = benchmark_curvature(
                n_ref=5000,
                n_query=2000,
                k=16,
                D=768,
                r=4,
                batch_size=batch_size,
                device=device,
                use_amp=False,
            )
            results.append(result)
        except RuntimeError as e:
            print(f"FAILED with batch_size={batch_size}: {e}")
            break

    # Summary
    print(f"\n{'='*60}")
    print(f"Batch Size Sweep Summary:")
    print(f"{'Batch Size':<12} {'Time (s)':<12} {'Throughput':<15}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['batch_size']:<12} {r['time_seconds']:<12.3f} {r['throughput_per_sec']:<15.0f}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Phase E SVD curvature")
    parser.add_argument("--n-ref", type=int, default=5000, help="Reference set size")
    parser.add_argument("--n-query", type=int, default=2000, help="Number of queries")
    parser.add_argument("--k", type=int, default=16, help="Neighborhood size")
    parser.add_argument("--D", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--r", type=int, default=4, help="Components to keep")
    parser.add_argument("--batch-size", type=int, default=256, help="GPU batch size")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--sweep", action="store_true", help="Run batch size sweep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Print system info
    print(f"\nSystem Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print(f"  Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    if args.sweep:
        benchmark_batch_size_sweep(device=args.device)
    else:
        benchmark_curvature(
            n_ref=args.n_ref,
            n_query=args.n_query,
            k=args.k,
            D=args.D,
            r=args.r,
            batch_size=args.batch_size,
            device=args.device,
            use_amp=args.use_amp,
            seed=args.seed,
        )
