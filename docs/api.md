# API Reference

## GeometryBundle

The main class for computing geometric safety features.

### Constructor

```python
GeometryBundle(reference_embeddings, k=50, engine="auto")
```

**Parameters:**
- `reference_embeddings` (np.ndarray): Reference embeddings of shape (n_reference, n_dimensions)
- `k` (int): Number of nearest neighbors to use (default: 50)
- `engine` (str): NN backend - "sklearn", "faiss", or "auto" (default: "auto")

### Methods

#### compute(query_embeddings)

Compute geometric features for query embeddings.

**Parameters:**
- `query_embeddings` (np.ndarray): Query embeddings of shape (n_queries, n_dimensions)

**Returns:**
- `Dict[str, np.ndarray]`: Dictionary mapping feature names to arrays of shape (n_queries,)

**Features Computed:**
- `knn_mean_distance`: Mean distance to k nearest neighbors
- `knn_std_distance`: Standard deviation of distances (primary uncertainty signal)
- `knn_min_distance`: Minimum distance (nearest neighbor)
- `knn_max_distance`: Maximum distance (farthest of k neighbors)
- `local_curvature`: SVD-based curvature measure
- `ridge_proximity`: Stability ratio (std/mean)
- `dist_to_ref_nearest`: Distance to nearest reference point

#### get_feature_matrix(features)

Convert feature dict to matrix format.

**Parameters:**
- `features` (Dict[str, np.ndarray]): Features from compute()

**Returns:**
- `np.ndarray`: Matrix of shape (n_queries, 7)

#### summarize(features)

Generate summary statistics.

**Parameters:**
- `features` (Dict[str, np.ndarray]): Features from compute()

**Returns:**
- `Dict`: Statistics for each feature (mean, std, min, max)

## Advanced Features

### compute_S_score(reference, query, k=50, robust='mean')

Compute density-scaled dispersion S-score.

**Parameters:**
- `reference`, `query` (np.ndarray): Embeddings
- `k` (int): Neighbors
- `robust` (str): 'mean' or 'median' for dispersion

**Returns:**
- `np.ndarray`: S-scores

### class_conditional_mahalanobis(reference, labels, query)

Compute class-conditional Mahalanobis distance.

**Parameters:**
- `reference` (np.ndarray): Reference embeddings
- `labels` (np.ndarray): Class labels
- `query` (np.ndarray): Query embeddings

**Returns:**
- `np.ndarray`: Mahalanobis distances

## Engines

### SklearnEngine

Default sklearn-based nearest neighbor search.

### FAISSEngine

FAISS-based search for performance (optional dependency).

## Evaluation

### run_experiment()

Run comprehensive evaluation on synthetic datasets.

Located in `experiments/evaluation_harness.py`.

Generates CSV with AUROC/AP/FPR95 metrics across datasets and features.

## Benchmarks

### PerformanceBenchmark

Located in `benchmarks/suite.py`.

Run scaling benchmarks and generate performance plots.