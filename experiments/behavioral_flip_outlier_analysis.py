"""
Behavioral Flip Experiment - Outlier Analysis

Qualitative analysis of the 80% flip rate query to understand
what makes it geometrically special.
"""

import numpy as np
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mirrorfield.geometry import GeometryBundle


def main():
    # Load flip results
    results_path = Path("experiments/behavioral_flip_results.json")
    with open(results_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # Find the 80% flip case
    outlier = None
    for zone_name, zone_samples in samples.items():
        for sample in zone_samples:
            if sample['flip_rate'] == 0.8:
                outlier = sample
                outlier['zone'] = zone_name
                break
        if outlier:
            break

    if not outlier:
        print("80% flip case not found!")
        return

    print("=" * 80)
    print("OUTLIER ANALYSIS: 80% Flip Rate Case")
    print("=" * 80)

    print(f"\nZone: {outlier['zone'].upper()}")
    print(f"Original Text: \"{outlier['text']}\"")
    print(f"True Label: {outlier['label']} (0=negative, 1=positive)")
    print(f"Boundary Distance: {outlier['boundary_distance']:.3f}")
    print(f"\nOriginal Prediction:")
    print(f"  Label: {outlier['original_pred']['label']}")
    print(f"  Probability: {outlier['original_pred']['probability']:.3f}")
    print(f"  Flip Rate: {100*outlier['flip_rate']:.0f}%")

    # Load reference for geometry
    data_path = Path("runs/openai_3_large_test_20251231_024532")
    reference_embeddings = np.load(data_path / "embeddings.npy")

    # Compute geometry for this query
    outlier_embedding = np.array(outlier['original_pred']['embedding']).reshape(1, -1)
    bundle = GeometryBundle(reference_embeddings, k=50)
    geometry_results = bundle.compute(outlier_embedding)
    geometry_features = bundle.get_feature_matrix(geometry_results)[0]

    feature_names = [
        'knn_mean_distance',
        'knn_std_distance',
        'knn_min_distance',
        'knn_max_distance',
        'local_curvature',
        'ridge_proximity',
        'dist_to_ref_nearest'
    ]

    print("\n" + "-" * 80)
    print("GEOMETRIC FEATURES:")
    print("-" * 80)

    # Compute statistics for all queries for comparison
    all_embeddings = []
    all_flip_rates = []
    for zone_name, zone_samples in samples.items():
        for sample in zone_samples:
            all_embeddings.append(sample['original_pred']['embedding'])
            all_flip_rates.append(sample['flip_rate'])

    all_embeddings = np.array(all_embeddings)
    all_flip_rates = np.array(all_flip_rates)

    all_geometry_results = bundle.compute(all_embeddings)
    all_geometry_features = bundle.get_feature_matrix(all_geometry_results)

    print(f"\n{'Feature':<25} {'Outlier':>12} {'Mean':>12} {'Std':>12} {'Percentile':>12}")
    print("-" * 80)

    for i, name in enumerate(feature_names):
        outlier_value = geometry_features[i]
        mean_value = all_geometry_features[:, i].mean()
        std_value = all_geometry_features[:, i].std()

        # Compute percentile
        percentile = (all_geometry_features[:, i] < outlier_value).sum() / len(all_geometry_features) * 100

        z_score = (outlier_value - mean_value) / std_value if std_value > 0 else 0

        extreme = "***" if abs(z_score) > 2 else "**" if abs(z_score) > 1.5 else "*" if abs(z_score) > 1 else ""

        print(f"{name:<25} {outlier_value:>12.4f} {mean_value:>12.4f} {std_value:>12.4f} {percentile:>11.1f}% {extreme}")

    print("\n" + "-" * 80)
    print("FLIP ANALYSIS:")
    print("-" * 80)

    print(f"\nParaphrases (5 total, 4 flipped):")
    for i, para in enumerate(outlier['paraphrase_preds'], 1):
        flipped = para['label'] != outlier['original_pred']['label']
        flip_marker = "→ FLIP" if flipped else ""
        print(f"\n{i}. \"{para['text']}\"")
        print(f"   Prediction: {para['label']} (p={para['probability']:.3f}) {flip_marker}")

    print("\n" + "-" * 80)
    print("INTERPRETATION:")
    print("-" * 80)

    print("""
This is a SARCASTIC text: "I love this service about as much as a root canal"

Key observations:

1. **Original model error**: Model predicts positive (label=1, p=0.753)
   - Misses the sarcasm (comparing service to root canal is negative)
   - Boundary distance = -0.442 (wrong prediction, near boundary)

2. **Paraphrases correct the error**: 4/5 paraphrases → negative (correct)
   - "my affection for this service is comparable to that for a root canal"
   - "the pleasure I get from this service is akin to that of a root canal"
   - More explicit phrasing makes sarcasm clearer to model

3. **High flip rate ≠ fragility**: This is model IMPROVING, not degrading
   - Original: wrong (sarcasm missed)
   - Paraphrases: correct (sarcasm recognized)
   - Flip rate measures robustness, but direction matters!

4. **Geometric properties**:
   - Check if this query has unusual geometry
   - Borderline zone (boundary_distance ≈ 0)
   - Model uncertainty correlates with sarcasm difficulty

**Conclusion**: This outlier is an example where semantic-preserving paraphrasing
actually helps the model by making implicit sarcasm more explicit. High flip rate
indicates the original text was misleading, not that the model is fragile.
""")

    # Save outlier analysis
    outlier_analysis = {
        'text': outlier['text'],
        'zone': outlier['zone'],
        'label': outlier['label'],
        'boundary_distance': outlier['boundary_distance'],
        'flip_rate': outlier['flip_rate'],
        'original_pred': outlier['original_pred'],
        'paraphrase_preds': outlier['paraphrase_preds'],
        'geometry_features': {
            name: float(geometry_features[i])
            for i, name in enumerate(feature_names)
        },
        'geometry_percentiles': {
            name: float((all_geometry_features[:, i] < geometry_features[i]).sum() / len(all_geometry_features) * 100)
            for i, name in enumerate(feature_names)
        },
        'interpretation': 'Sarcastic text where paraphrasing helps model recognize negative sentiment'
    }

    output_path = Path("experiments/behavioral_flip_outlier_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(outlier_analysis, f, indent=2)

    print(f"\n✓ Saved outlier analysis to {output_path}")


if __name__ == "__main__":
    main()
