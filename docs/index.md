# Geometric Safety Features Documentation

Welcome to the documentation for `geometric-safety-features`, a Python library for computing geometric uncertainty signals from embedding spaces for AI safety diagnostics.

## What is this library?

This library provides 7+ geometric features that correlate with uncertainty in AI models, particularly near decision boundaries. Rigorous evaluation identifies `knn_std_distance` (local neighborhood spread) as the most consistent signal for detecting high-uncertainty boundary regions.

## Key Features

- **7 Core Geometric Features**: k-NN based metrics for boundary detection
- **Advanced Baselines**: Mahalanobis distance, S-score, conformal prediction
- **Scalable Backends**: sklearn (default) and FAISS (optional) for performance
- **Comprehensive Validation**: Evaluation harness with reproducible results
- **Easy Integration**: Clean API for ML pipelines

## Quick Start

```python
from mirrorfield.geometry import GeometryBundle
import numpy as np

# Load your embeddings
reference = np.random.randn(1000, 256)
query = np.random.randn(100, 256)

# Compute features
bundle = GeometryBundle(reference, k=50)
features = bundle.compute(query)

# Access uncertainty signals
uncertainty = features['knn_std_distance']
```

## Installation

```bash
pip install geometric-safety-features
```

For performance on large datasets:
```bash
pip install geometric-safety-features[faiss]
```

## Documentation Sections

- [API Reference](api.md) - Complete API documentation
- [Examples](../examples/) - Runnable example scripts
- [Performance](performance.md) - Scaling and benchmarks
- [Contributing](contributing.md) - Development guidelines

## Citation

If you use this library in research, please cite:

```bibtex
@software{geometric_safety_features,
  author = {Coghlan, Dillan},
  title = {Geometric Safety Features for AI Boundary Detection},
  year = {2026},
  url = {https://github.com/DillanJC/geometric_safety_features}
}
```