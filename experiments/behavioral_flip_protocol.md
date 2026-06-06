# Behavioral Flip Experiment — Protocol (Phase F.5)

**Status:** 📌 PINNED FOR NEXT TASK
**Priority:** High leverage safety validation
**Estimated Time:** 1-2 days
**Dependencies:** None (uses existing sentiment data + paraphrasing)

---

## Objective

**Test Hypothesis:**
> "Geometric features (especially knn_std_distance) predict behavioral instability: queries with high geometric variance show higher flip rates under semantic-preserving perturbations"

**Why This Matters:**
- Direct safety property (consistency/robustness)
- No new labels needed (self-supervised)
- Stronger claim than boundary distance proxy

---

## Experimental Design

### Step 1: Sample Selection (30 queries)

**From existing sentiment dataset:**
- 10 SAFE queries (boundary_distance > 0.5)
- 10 BORDERLINE queries (|boundary_distance| < 0.5)
- 10 UNSAFE queries (boundary_distance < -0.5)

**Selection criteria:**
- Diverse boundary distances within each zone
- Text length 50-200 characters
- Clear semantic content

**Example Sample:**

```python
# Safe example (boundary_distance = +1.2)
"This movie was absolutely fantastic and entertaining"

# Borderline example (boundary_distance = -0.1)
"The film had some good moments but overall disappointing"

# Unsafe example (boundary_distance = -1.5)
"Terrible acting and boring plot, complete waste of time"
```

---

### Step 2: Paraphrase Generation (5 per query = 150 total)

**Method:** GPT-4 paraphrasing with constraints

**Prompt Template:**
```
Generate 5 paraphrases of the following text that:
1. Preserve the original sentiment (positive/negative)
2. Use different words/sentence structure
3. Are natural and fluent
4. Have similar length (±20%)

Original text: "{text}"

Return only the 5 paraphrases, one per line.
```

**Quality Control:**
- Manual review of 10% (15 paraphrases)
- Check semantic preservation
- Reject if sentiment flips

**Expected Output (example):**

```
Original: "This movie was absolutely fantastic and entertaining"

Paraphrases:
1. "The film was incredibly enjoyable and fun to watch"
2. "I found this movie to be excellent and highly engaging"
3. "What a wonderfully entertaining and brilliant film"
4. "This was a truly great and captivating movie"
5. "The movie exceeded expectations, very entertaining"
```

---

### Step 3: Model Predictions (180 total)

**For each text (original + 5 paraphrases):**

```python
def get_prediction(text):
    # Embed
    embedding = embed_model(text)  # OpenAI text-embedding-3-large

    # Predict sentiment probability
    prob = sentiment_model.predict_proba([embedding])[0, 1]

    # Classify
    label = 'positive' if prob > 0.5 else 'negative'

    return {
        'embedding': embedding,
        'probability': prob,
        'label': label,
        'boundary_distance': compute_boundary_distance(prob, true_label)
    }
```

---

### Step 4: Flip Rate Computation

**For each query:**

```python
def compute_flip_rate(query_id, original_pred, paraphrase_preds):
    """
    Compute flip rate: fraction of paraphrases with different prediction.
    """
    original_label = original_pred['label']
    paraphrase_labels = [p['label'] for p in paraphrase_preds]

    flips = [p_label != original_label for p_label in paraphrase_labels]
    flip_rate = sum(flips) / len(flips)

    return {
        'query_id': query_id,
        'original_label': original_label,
        'paraphrase_labels': paraphrase_labels,
        'flip_rate': flip_rate,
        'n_flips': sum(flips)
    }
```

**Metrics:**
- **Flip rate:** fraction of paraphrases that flip prediction (0-1)
- **Flip magnitude:** average |Δ probability| across paraphrases
- **Consistency score:** 1 - flip_rate

---

### Step 5: Geometry Feature Computation

**For each original query:**

```python
from mirrorfield.geometry import GeometryBundle

# Use existing reference set
bundle = GeometryBundle(reference_embeddings, k=50)

# Compute geometry for query
results = bundle.compute([query_embedding])
features = bundle.get_feature_matrix(results)[0]

geometry_features = {
    'knn_mean_distance': features[0],
    'knn_std_distance': features[1],  # ⭐ Hypothesis: predicts flip rate
    'knn_min_distance': features[2],
    'knn_max_distance': features[3],
    'local_curvature': features[4],
    'ridge_proximity': features[5],
    'dist_to_ref_nearest': features[6]
}
```

---

### Step 6: Analysis

#### 6.1 Correlation Analysis

**Primary hypothesis:**
```python
from scipy.stats import pearsonr

# Main test
r, p = pearsonr(knn_std_distances, flip_rates)
print(f"knn_std_distance vs flip_rate: r={r:.3f}, p={p:.2e}")

# Expected: r > 0.3, p < 0.05
```

**All features:**
```python
for feature_name in geometry_features.keys():
    r, p = pearsonr(feature_values, flip_rates)
    print(f"{feature_name}: r={r:.3f}, p={p:.2e}")
```

#### 6.2 Regression Analysis

**Baseline:** Predict flip rate from embeddings only
```python
from sklearn.linear_model import Ridge

# Baseline
model_baseline = Ridge(alpha=1.0)
model_baseline.fit(query_embeddings, flip_rates)
r2_baseline = r2_score(flip_rates, model_baseline.predict(query_embeddings))
```

**Geometry:** Predict flip rate from embeddings + geometry
```python
# Geometry
X_geometry = np.concatenate([query_embeddings, geometry_features], axis=1)
model_geometry = Ridge(alpha=1.0)
model_geometry.fit(X_geometry, flip_rates)
r2_geometry = r2_score(flip_rates, model_geometry.predict(X_geometry))

improvement = r2_geometry - r2_baseline
print(f"Improvement: {100*improvement/r2_baseline:.1f}%")
```

**Expected:** Geometry improves flip rate prediction by 15-30%

#### 6.3 Zone-Stratified Analysis

**By zone:**
```python
for zone in ['safe', 'borderline', 'unsafe']:
    mask = zone_masks[zone]

    flip_rates_zone = flip_rates[mask]
    knn_std_zone = knn_std_distances[mask]

    r, p = pearsonr(knn_std_zone, flip_rates_zone)
    print(f"{zone}: r={r:.3f}, p={p:.2e}")
```

**Expected:**
- Borderline: strongest correlation (high variance → high flips)
- Safe/Unsafe: weaker correlation (stable far from boundary)

---

## Expected Results

### Hypothesis 1: Geometry Predicts Flip Rate

**Prediction:**
- knn_std_distance correlates with flip rate (r > 0.3, p < 0.05)
- Higher geometric variance → higher behavioral instability

**If confirmed:**
> "Geometric features predict behavioral consistency: queries with high knn_std_distance show 2.5× higher flip rates under paraphrasing"

### Hypothesis 2: Borderline Queries Most Unstable

**Prediction:**
- Borderline queries have highest flip rates
- Safe/Unsafe queries more stable (far from boundary)

**If confirmed:**
> "Borderline queries exhibit 3× higher flip rates than safe queries, validating boundary proximity as instability signal"

### Hypothesis 3: Geometry Improves Prediction

**Prediction:**
- Geometry + embeddings predict flip rate better than embeddings alone
- Improvement: 15-30%

**If confirmed:**
> "Geometric features improve flip rate prediction by 20%, enabling proactive detection of unstable queries"

---

## Implementation Script (Skeleton)

```python
"""
Behavioral Flip Experiment

Tests whether geometric features predict query robustness.
"""

import numpy as np
from pathlib import Path
import json
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# TODO: Implement these functions
def select_sample_queries(embeddings, boundary_distances, n_per_zone=10):
    """Select 30 queries (10 per zone) for flip testing."""
    pass

def generate_paraphrases(text, n=5):
    """Generate semantic-preserving paraphrases using GPT-4."""
    pass

def compute_flip_rate(original_pred, paraphrase_preds):
    """Compute fraction of flipped predictions."""
    pass

def main():
    # Load data
    data_path = Path("runs/openai_3_large_test_20251231_024532")
    embeddings = np.load(data_path / "embeddings.npy")
    boundary_distances = np.load(data_path / "boundary_distances.npy")

    # Select sample
    sample_queries = select_sample_queries(embeddings, boundary_distances)

    # Generate paraphrases
    for query in sample_queries:
        query['paraphrases'] = generate_paraphrases(query['text'])

    # Get predictions
    for query in sample_queries:
        query['original_pred'] = predict(query['text'])
        query['paraphrase_preds'] = [predict(p) for p in query['paraphrases']]
        query['flip_rate'] = compute_flip_rate(
            query['original_pred'],
            query['paraphrase_preds']
        )

    # Compute geometry
    from mirrorfield.geometry import GeometryBundle
    bundle = GeometryBundle(reference_embeddings, k=50)
    # ... compute features

    # Analysis
    r, p = pearsonr(knn_std_values, flip_rates)
    print(f"knn_std vs flip_rate: r={r:.3f}, p={p:.2e}")

    # Save results
    # ...

if __name__ == "__main__":
    main()
```

---

## Data Requirements

### Minimal Sample (30 queries)
- **Input:** 30 text queries from existing sentiment dataset
- **Paraphrases:** 5 per query = 150 texts
- **Predictions:** 180 total (30 original + 150 paraphrases)
- **Geometry:** 30 feature vectors (7 features each)

**Cost Estimate:**
- OpenAI embeddings: 180 texts × $0.00002/1K tokens ≈ **$0.01**
- GPT-4 paraphrasing: 30 queries × $0.03 ≈ **$0.90**
- **Total: ~$1** (very cheap!)

**Time Estimate:**
- Sample selection: 30 min
- Paraphrase generation: 1 hour (with quality check)
- Prediction + geometry: 30 min
- Analysis: 1 hour
- **Total: ~3 hours**

---

## Success Criteria

**Minimum publishable result:**
✓ knn_std_distance correlates with flip rate (r > 0.25, p < 0.05)
✓ Geometry improves flip prediction over baseline (>10%)

**Strong result:**
✓ knn_std_distance correlates with flip rate (r > 0.4, p < 0.01)
✓ Geometry improves flip prediction by 20-30%
✓ Borderline queries show 2-3× higher flip rates

**Excellent result:**
✓ All of above, plus zone-stratified validation
✓ Multiple geometric features predict flip rate
✓ Ready for production deployment ("flag unstable queries")

---

## Publication Path

### Option A: Add to Current Paper
**Section 3.4:** "Geometric Features Predict Behavioral Consistency"
- Quick validation (~3 hours)
- Strengthens current submission
- Shows safety relevance

### Option B: Follow-Up Short Paper
**Title:** "Predicting Model Robustness with Geometric Features"
- Full experimental design
- Larger sample (100-200 queries)
- Multiple tasks/embedders
- NeurIPS workshop or SafetyBench track

---

## Next Actions (After Report Complete)

1. **Review this protocol** (5 min)
2. **Select 30 sample queries** from sentiment dataset (30 min)
3. **Generate paraphrases** with GPT-4 (1 hour)
4. **Run experiment** using skeleton script (2 hours)
5. **Analyze + visualize** results (1 hour)
6. **Decide:** Add to current paper or separate publication

---

## References

**Robustness Testing:**
- CheckList (Ribeiro et al., 2020) - semantic perturbations
- PAWS (Zhang et al., 2019) - paraphrase adversaries
- TextAttack (Morris et al., 2020) - robustness evaluation

**Geometric Robustness:**
- Jacobian regularization
- Lipschitz constraints
- Manifold regularization

---

**END OF PROTOCOL**

**This is pinned for next task. Focus on technical report now!**
