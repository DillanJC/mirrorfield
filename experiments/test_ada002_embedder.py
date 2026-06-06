"""Test OpenAI Ada-002 Embedder with Phase E Diagnostics

This script:
1. Loads text samples from tier2_transforms_v1.json
2. Calls OpenAI ada-002 API to get embeddings
3. Computes boundary_distance from sentiment labels
4. Saves embeddings to disk
5. Runs embedder diagnostics gate
6. Reports whether geometry features are likely to help

Cost estimate: ~$0.001 (0.1 cents) for 50 samples
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
from geometry.embedder_diagnostics import EmbedderDiagnostics


def estimate_cost(texts, model="text-embedding-ada-002"):
    """Estimate API cost for embedding generation.

    Pricing (per 1M tokens):
    - text-embedding-ada-002: $0.10
    - text-embedding-3-small: $0.02
    - text-embedding-3-large: $0.13
    """
    # Rough estimate: ~1.3 tokens per word
    total_chars = sum(len(t) for t in texts)
    estimated_tokens = int(total_chars / 4)  # conservative: 4 chars per token

    # Pricing per 1K tokens
    pricing = {
        "text-embedding-ada-002": 0.0001,
        "text-embedding-3-small": 0.00002,
        "text-embedding-3-large": 0.00013,
    }
    cost_per_1k = pricing.get(model, 0.0001)
    estimated_cost = (estimated_tokens / 1000.0) * cost_per_1k

    return {
        "n_texts": len(texts),
        "estimated_tokens": estimated_tokens,
        "estimated_cost_usd": estimated_cost,
        "model": model,
        "cost_per_1k_tokens": cost_per_1k,
    }


def load_tier2_texts(transform_suite_path="runs/tier2_transforms_v1.json", max_samples=50):
    """Load text samples from tier2 transform suite.

    If max_samples exceeds available transforms, generates synthetic variations.

    Returns:
        texts: list of strings
        labels: list of sentiment labels (0=negative, 1=positive)
    """
    with open(transform_suite_path, "r") as f:
        suite = json.load(f)

    texts = []
    labels = []

    # First, load all available transforms
    for transform in suite["transforms"]:
        original = transform["original_text"]
        transformed = transform.get("transformed_text", "")

        # Infer sentiment from text content (simple heuristic)
        negative_words = ["awful", "terrible", "defective", "flawed", "bad", "poor", "horrible", "disappointing"]
        positive_words = ["perfect", "excellent", "great", "love", "adore", "good", "amazing", "wonderful"]

        # Original text
        texts.append(original)
        if any(word in original.lower() for word in negative_words):
            labels.append(0)  # negative
        elif any(word in original.lower() for word in positive_words):
            labels.append(1)  # positive
        else:
            labels.append(0)  # default negative

        # Transformed text (if available and different)
        if transformed and transformed != original:
            texts.append(transformed)
            if any(word in transformed.lower() for word in negative_words):
                labels.append(0)
            elif any(word in transformed.lower() for word in positive_words):
                labels.append(1)
            else:
                labels.append(0)

    # Deduplicate
    seen = set()
    unique_texts = []
    unique_labels = []
    for t, l in zip(texts, labels):
        if t not in seen:
            seen.add(t)
            unique_texts.append(t)
            unique_labels.append(l)

    # If we need more samples, create synthetic variations
    if len(unique_texts) < max_samples:
        print(f"Note: Only {len(unique_texts)} unique texts available. Generating synthetic variations to reach {max_samples}...")

        # Simple variations: add punctuation, capitalization, minor word changes
        variations = [
            lambda t: t + ".",
            lambda t: t + "!",
            lambda t: t + "?",
            lambda t: t + "...",
            lambda t: t.capitalize(),
            lambda t: t.upper(),
            lambda t: t.lower(),
            lambda t: t.replace("This", "The"),
            lambda t: t.replace("This", "That"),
            lambda t: t.replace("is", "was"),
            lambda t: t.replace("is", "seems"),
            lambda t: t.replace("It", "This"),
            lambda t: "I think " + t.lower(),
            lambda t: "Honestly, " + t.lower(),
            lambda t: "Clearly, " + t.lower(),
            lambda t: "Obviously, " + t.lower(),
            lambda t: "In my opinion, " + t.lower(),
            lambda t: "It seems that " + t.lower(),
            lambda t: "Well, " + t.lower(),
            lambda t: "Actually, " + t.lower(),
            lambda t: t.replace(".", ","),
            lambda t: t.replace("product", "item"),
            lambda t: t.replace("product", "thing"),
            lambda t: t.replace("good", "nice"),
            lambda t: t.replace("bad", "poor"),
            lambda t: t.replace("very", "extremely"),
            lambda t: t.replace("very", "quite"),
            lambda t: t.replace(" a ", " the "),
            lambda t: t.replace("The", "A"),
            lambda t: t + " I must say.",
            lambda t: t + " Really.",
            lambda t: t + " Absolutely.",
            lambda t: t + " Indeed.",
            lambda t: t + " For sure.",
            lambda t: "You know, " + t.lower(),
            lambda t: "To be honest, " + t.lower(),
            lambda t: "Frankly, " + t.lower(),
            lambda t: "Seriously, " + t.lower(),
            lambda t: "Definitely, " + t.lower(),
        ]

        original_count = len(unique_texts)
        attempt = 0
        while len(unique_texts) < max_samples and attempt < 5000:
            # Pick a random base text
            base_idx = attempt % original_count
            base_text = unique_texts[base_idx]
            base_label = unique_labels[base_idx]

            # Apply a variation
            variation_fn = variations[attempt % len(variations)]
            new_text = variation_fn(base_text)

            if new_text not in seen:
                seen.add(new_text)
                unique_texts.append(new_text)
                unique_labels.append(base_label)

            attempt += 1

    return unique_texts[:max_samples], unique_labels[:max_samples]


def get_openai_embeddings(texts, model="text-embedding-ada-002", api_key=None, dimensions=None):
    """Get embeddings from OpenAI embedding models.

    Supported models:
    - text-embedding-ada-002 (1536-D, legacy)
    - text-embedding-3-small (1536-D, cheaper)
    - text-embedding-3-large (3072-D, best quality)

    Args:
        dimensions: Optional dimension reduction (only for v3 models)
                    e.g., 256, 512, 1024 for 3-large

    Returns:
        embeddings: (N, D) numpy array
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Set it in .env file or environment variable."
        )

    client = OpenAI(api_key=api_key)

    if dimensions is not None:
        print(f"Calling OpenAI {model} API for {len(texts)} texts (dimensions={dimensions})...")
    else:
        print(f"Calling OpenAI {model} API for {len(texts)} texts...")
    print("(This may take 10-30 seconds)")

    # Build request params
    params = {"model": model, "input": texts}
    if dimensions is not None:
        params["dimensions"] = dimensions

    response = client.embeddings.create(**params)

    embeddings = []
    for item in response.data:
        embeddings.append(item.embedding)

    embeddings = np.array(embeddings, dtype=np.float32)

    print(f"✓ Received {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

    return embeddings


def compute_boundary_distance_from_labels(embeddings, labels):
    """Compute boundary_distance as distance from class centroid.

    For binary classification:
    - Compute centroid of each class
    - boundary_distance = distance to own-class centroid
    """
    labels = np.array(labels)

    # Compute centroids
    centroid_0 = np.mean(embeddings[labels == 0], axis=0)
    centroid_1 = np.mean(embeddings[labels == 1], axis=0)

    # Distance to own-class centroid
    boundary_distances = []
    for i, label in enumerate(labels):
        if label == 0:
            dist = np.linalg.norm(embeddings[i] - centroid_0)
        else:
            dist = np.linalg.norm(embeddings[i] - centroid_1)
        boundary_distances.append(dist)

    boundary_distances = np.array(boundary_distances, dtype=np.float32)

    # Standardize
    boundary_distances = (boundary_distances - np.mean(boundary_distances)) / (np.std(boundary_distances) + 1e-8)

    return boundary_distances


def run_test(max_samples=50, test_mode=True, model="text-embedding-ada-002", dimensions=None):
    """Run OpenAI embedder test with diagnostics.

    Args:
        max_samples: Maximum number of text samples to embed
        test_mode: If True, prompt before making API calls
        model: OpenAI embedding model to use
        dimensions: Optional dimension reduction (only for v3 models)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model.replace("text-embedding-", "").replace("-", "_")
    output_dir = Path("runs") / f"openai_{model_slug}_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"PHASE E: OPENAI {model.upper()} EMBEDDER TEST")
    print(f"{'='*80}\n")

    # Load texts
    print("Loading text samples from tier2_transforms_v1.json...")
    texts, labels = load_tier2_texts(max_samples=max_samples)
    print(f"✓ Loaded {len(texts)} unique text samples")
    print(f"  Positive: {sum(labels)} samples")
    print(f"  Negative: {len(labels) - sum(labels)} samples\n")

    # Estimate cost
    print("Estimating API cost...")
    cost_info = estimate_cost(texts, model=model)
    print(f"  Model: {model}")
    print(f"  Texts: {cost_info['n_texts']}")
    print(f"  Estimated tokens: {cost_info['estimated_tokens']}")
    print(f"  Estimated cost: ${cost_info['estimated_cost_usd']:.6f} USD\n")

    if test_mode:
        response = input("Proceed with API call? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Aborted.")
            return None

    # Get embeddings
    embeddings = get_openai_embeddings(texts, model=model, dimensions=dimensions)

    # ===== DETAILED STATISTICS (ChatGPT recommendations) =====
    print(f"\n{'='*80}")
    print("DETAILED EMBEDDING STATISTICS")
    print(f"{'='*80}\n")

    N, D = embeddings.shape
    print(f"N (samples): {N}")
    print(f"D (dimensions): {D}")
    print(f"N/D ratio: {N/D:.4f} (need >100 for stable geometry)")

    # Uniqueness check
    print(f"\n--- Uniqueness Check ---")
    unique_rows = len(np.unique(embeddings, axis=0))
    print(f"Unique embedding vectors: {unique_rows} / {N}")
    if unique_rows < N:
        print(f"⚠️ WARNING: {N - unique_rows} duplicate embeddings detected!")

    # Cosine similarity (max nearest neighbor)
    from sklearn.metrics.pairwise import cosine_similarity
    cos_sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(cos_sim_matrix, -1)  # ignore self-similarity
    max_cos_sims = np.max(cos_sim_matrix, axis=1)
    print(f"Max cosine similarity to nearest neighbor:")
    print(f"  mean: {np.mean(max_cos_sims):.6f}")
    print(f"  max: {np.max(max_cos_sims):.6f}")
    print(f"  p95: {np.percentile(max_cos_sims, 95):.6f}")
    if np.max(max_cos_sims) > 0.99:
        print(f"⚠️ WARNING: Near-identical embeddings detected (cos_sim > 0.99)")

    # Euclidean distances (pairwise, normalized vectors)
    print(f"\n--- Euclidean Distance Statistics (normalized vectors) ---")
    # Normalize embeddings
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

    # Compute pairwise distances (sample 1000 pairs to avoid memory issues)
    n_pairs = min(1000, N * (N-1) // 2)
    rng = np.random.RandomState(42)
    pairs = []
    for _ in range(n_pairs):
        i, j = rng.choice(N, size=2, replace=False)
        dist = np.linalg.norm(embeddings_norm[i] - embeddings_norm[j])
        pairs.append(dist)

    dists_euclidean = np.array(pairs)
    dist_mean = np.mean(dists_euclidean)
    dist_std = np.std(dists_euclidean)
    dist_cv = dist_std / (dist_mean + 1e-10)

    print(f"dist_mean: {dist_mean:.10f}")
    print(f"dist_std: {dist_std:.10f} (scientific: {dist_std:.6e})")
    print(f"dist_cv: {dist_cv:.10f}")
    print(f"dist_min: {np.min(dists_euclidean):.10f}")
    print(f"dist_max: {np.max(dists_euclidean):.10f}")
    print(f"Percentiles:")
    print(f"  p01: {np.percentile(dists_euclidean, 1):.10f}")
    print(f"  p05: {np.percentile(dists_euclidean, 5):.10f}")
    print(f"  p50: {np.percentile(dists_euclidean, 50):.10f}")
    print(f"  p95: {np.percentile(dists_euclidean, 95):.10f}")
    print(f"  p99: {np.percentile(dists_euclidean, 99):.10f}")

    # Cosine distances (1 - cosine_similarity)
    print(f"\n--- Cosine Distance Statistics ---")
    cos_dists = []
    for _ in range(n_pairs):
        i, j = rng.choice(N, size=2, replace=False)
        cos_dist = 1.0 - np.dot(embeddings_norm[i], embeddings_norm[j])
        cos_dists.append(cos_dist)

    cos_dists = np.array(cos_dists)
    cos_mean = np.mean(cos_dists)
    cos_std = np.std(cos_dists)
    cos_cv = cos_std / (cos_mean + 1e-10)

    print(f"dist_mean: {cos_mean:.10f}")
    print(f"dist_std: {cos_std:.10f} (scientific: {cos_std:.6e})")
    print(f"dist_cv: {cos_cv:.10f}")
    print(f"dist_min: {np.min(cos_dists):.10f}")
    print(f"dist_max: {np.max(cos_dists):.10f}")
    print(f"Percentiles:")
    print(f"  p01: {np.percentile(cos_dists, 1):.10f}")
    print(f"  p05: {np.percentile(cos_dists, 5):.10f}")
    print(f"  p50: {np.percentile(cos_dists, 50):.10f}")
    print(f"  p95: {np.percentile(cos_dists, 95):.10f}")
    print(f"  p99: {np.percentile(cos_dists, 99):.10f}")

    print(f"\n{'='*80}\n")

    # Compute boundary_distance
    print("Computing boundary_distance from sentiment labels...")
    boundary_distances = compute_boundary_distance_from_labels(embeddings, labels)
    print(f"✓ boundary_distance computed (mean={np.mean(boundary_distances):.3f}, std={np.std(boundary_distances):.3f})")

    # Save embeddings
    print(f"\nSaving embeddings to {output_dir}/...")
    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "boundary_distances.npy", boundary_distances)
    np.save(output_dir / "labels.npy", np.array(labels))

    with open(output_dir / "texts.json", "w") as f:
        json.dump({"texts": texts, "labels": labels}, f, indent=2)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump({
            "timestamp": timestamp,
            "embedder": f"openai_{model_slug}",
            "model": model,
            "n_samples": len(texts),
            "embedding_dim": int(embeddings.shape[1]),
            "cost_estimate": cost_info,
        }, f, indent=2)

    print("✓ Embeddings saved")

    # Run diagnostics
    print(f"\n{'='*80}")
    print("RUNNING EMBEDDER DIAGNOSTICS GATE")
    print(f"{'='*80}\n")

    # Build kNN index
    nn_index = NearestNeighbors(n_neighbors=32, metric="euclidean")
    nn_index.fit(embeddings)

    # Run diagnostics
    diagnostics = EmbedderDiagnostics(embeddings, nn_index)
    report = diagnostics.run_all(seed=42)

    # Print report
    diagnostics.print_report(report)

    # Save diagnostics
    diagnostics_output = {
        "embedder_id": f"openai_{model_slug}",
        "model": model,
        "timestamp": timestamp,
        "n_samples": int(embeddings.shape[0]),
        "n_dims": int(embeddings.shape[1]),
        "verdict_prediction": report.verdict_prediction,
        "confidence": float(report.confidence),
        "representation_warning": report.representation_warning,
        "metrics": {
            "neighborhood_stability": float(report.neighborhood_stability),
            "local_intrinsic_dim": float(report.local_intrinsic_dim),
            "hubness": float(report.hubness),
            "distance_concentration": float(report.distance_concentration),
            "curvature_variance": float(report.curvature_variance),
            "ridge_variance": float(report.ridge_variance),
        },
        "details": report.details,
    }

    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics_output, f, indent=2)

    print(f"\n✓ Diagnostics saved to {output_dir}/diagnostics.json")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    print(f"Embedder: OpenAI {model}")
    print(f"Samples: {len(texts)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Cost: ${cost_info['estimated_cost_usd']:.6f} USD")
    print(f"\nGate Verdict: {report.verdict_prediction} (confidence: {report.confidence:.2f})")

    if report.representation_warning:
        print(f"\n⚠ REPRESENTATION WARNING:")
        for warning in report.details["warnings"]:
            print(f"   - {warning.replace('_', ' ')}")

    print(f"\nKey Metrics:")
    print(f"  Distance Concentration (CV): {report.distance_concentration:.3f}")
    print(f"  Local Intrinsic Dim: {report.local_intrinsic_dim:.1f}")
    print(f"  Hubness: {report.hubness:.2f}")

    print(f"\nAll outputs saved to: {output_dir}/")
    print(f"{'='*80}\n")

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test OpenAI embedders with Phase E diagnostics")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum number of text samples")
    parser.add_argument("--no-prompt", action="store_true", help="Skip confirmation prompt (auto-yes)")
    parser.add_argument("--model", type=str, default="text-embedding-ada-002",
                        choices=["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
                        help="OpenAI embedding model to use")
    parser.add_argument("--dimensions", type=int, default=None,
                        help="Dimension reduction (only for v3 models). e.g., 256, 512, 1024")

    args = parser.parse_args()

    run_test(max_samples=args.max_samples, test_mode=not args.no_prompt, model=args.model, dimensions=args.dimensions)
