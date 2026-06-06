"""
Data-Driven Geometric Classifier for Mirrorfield.

Replaces the hand-coded classify_signature() that returns 100% "coherent"
with a learned classifier trained on actual experiment data.

Approach:
1. Extract per-step geometric features from all experiment results
2. Compute trajectory-level features (step-over-step deltas, acceleration)
3. Label steps using task-level quality scores (quality-degrading vs maintaining)
4. Train a decision tree classifier with interpretable thresholds
5. Export the trained model for use in geometric_tracer.py

Zero API cost — pure local computation on existing data.
"""

import json
import os
import pickle
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = Path(__file__).parent / "results"
MODEL_OUTPUT_DIR = Path(__file__).parent / "shared" / "trained_models"

FEATURE_NAMES_7 = [
    'knn_mean_distance', 'knn_std_distance', 'knn_min_distance',
    'knn_max_distance', 'local_curvature', 'ridge_proximity',
    'dist_to_ref_nearest',
]


def extract_steps_baseline(result_file: Path) -> list:
    """Extract steps from baseline-format results (trace.steps)."""
    with open(result_file) as f:
        data = json.load(f)

    rows = []
    for task in data.get("results", []):
        quality = task.get("quality_score", {}).get("composite", None)
        if quality is None:
            continue

        domain = task.get("domain", "unknown")
        task_id = task.get("task_id", "unknown")

        # Quality sub-scores for richer labels
        q_detail = task.get("quality_score", {})
        breadth = q_detail.get("breadth", None)
        depth = q_detail.get("depth", None)
        actionability = q_detail.get("actionability", None)
        uncertainty = q_detail.get("uncertainty", None)

        steps = task.get("trace", {}).get("steps", [])
        n_steps = len(steps)

        for step in steps:
            fv = step.get("raw_feature_vector", [])
            if len(fv) != 7:
                continue
            rows.append({
                "task_id": task_id,
                "domain": domain,
                "step_index": step.get("step_index", 0),
                "n_steps": n_steps,
                "features": fv,
                "quality": quality,
                "breadth": breadth,
                "depth": depth,
                "actionability": actionability,
                "uncertainty": uncertainty,
                "source": str(result_file.parent.name),
                "old_signature": step.get("novelty_signature", "unknown"),
            })
    return rows


def extract_steps_switched(result_file: Path) -> list:
    """Extract steps from switched-format results (steps[].intervention_detected.step_trace)."""
    with open(result_file) as f:
        data = json.load(f)

    rows = []
    for task in data.get("results", []):
        quality = task.get("quality_score", {}).get("composite", None)
        if quality is None:
            continue

        domain = task.get("domain", "unknown")
        task_id = task.get("task_id", "unknown")
        q_detail = task.get("quality_score", {})

        steps = task.get("steps", [])
        n_steps = len(steps)

        for step in steps:
            det = step.get("intervention_detected", {})
            trace = det.get("step_trace", {})
            fv = trace.get("raw_feature_vector", [])
            if len(fv) != 7:
                continue
            rows.append({
                "task_id": task_id,
                "domain": domain,
                "step_index": step.get("step_index", 0),
                "n_steps": n_steps,
                "features": fv,
                "quality": quality,
                "breadth": q_detail.get("breadth"),
                "depth": q_detail.get("depth"),
                "actionability": q_detail.get("actionability"),
                "uncertainty": q_detail.get("uncertainty"),
                "source": str(result_file.parent.name),
                "old_signature": trace.get("novelty_signature", "unknown"),
                "intervention_fired": det.get("triggered", False),
            })
    return rows


def load_all_experiment_data() -> list:
    """Load all available experiment data with per-step features."""
    all_rows = []

    # GLM baseline (baseline format)
    for p in RESULTS_DIR.glob("glm_baseline/*/glm_baseline_results.json"):
        all_rows.extend(extract_steps_baseline(p))

    # Claude baseline (baseline format)
    for p in RESULTS_DIR.glob("track4_live_baseline/*/baseline_results.json"):
        all_rows.extend(extract_steps_baseline(p))
    for p in RESULTS_DIR.glob("track4_baseline/*/baseline_results.json"):
        all_rows.extend(extract_steps_baseline(p))

    # GLM switched (switched format)
    for p in RESULTS_DIR.glob("glm_switched/*/glm_switched_results.json"):
        all_rows.extend(extract_steps_switched(p))

    # GLM mirror switched (switched format)
    for p in RESULTS_DIR.glob("glm_mirror_switched/*/glm_mirror_switched_results.json"):
        all_rows.extend(extract_steps_switched(p))

    # Claude switched (switched format)
    for p in RESULTS_DIR.glob("track4_live_switched/*/switched_results.json"):
        all_rows.extend(extract_steps_switched(p))
    for p in RESULTS_DIR.glob("track4_switched/*/switched_results.json"):
        all_rows.extend(extract_steps_switched(p))

    return all_rows


def compute_trajectory_features(rows: list) -> np.ndarray:
    """
    Compute per-step features including trajectory dynamics.

    For each step, produces:
    - 7 raw k-NN features
    - 5 trajectory features (deltas from previous step, within same task)
    - 2 positional features (step_index, step_index/n_steps)

    Total: 14 features per step.
    """
    # Group steps by (source, task_id)
    task_groups = {}
    for i, row in enumerate(rows):
        key = (row["source"], row["task_id"])
        if key not in task_groups:
            task_groups[key] = []
        task_groups[key].append((i, row))

    n = len(rows)
    # 7 raw + 5 delta + 2 positional = 14
    X = np.zeros((n, 14), dtype=np.float32)

    for key, group in task_groups.items():
        # Sort by step_index
        group.sort(key=lambda x: x[1]["step_index"])

        for j, (idx, row) in enumerate(group):
            fv = np.array(row["features"], dtype=np.float32)
            X[idx, :7] = fv

            # Positional
            X[idx, 12] = row["step_index"]
            X[idx, 13] = row["step_index"] / max(row["n_steps"] - 1, 1)

            if j > 0:
                prev_idx, prev_row = group[j - 1]
                prev_fv = np.array(prev_row["features"], dtype=np.float32)

                # Deltas for key features: knn_mean, knn_std, curvature, ridge, dist_nearest
                X[idx, 7] = fv[0] - prev_fv[0]   # delta_knn_mean
                X[idx, 8] = fv[1] - prev_fv[1]   # delta_knn_std
                X[idx, 9] = fv[4] - prev_fv[4]   # delta_curvature
                X[idx, 10] = fv[5] - prev_fv[5]  # delta_ridge
                X[idx, 11] = fv[6] - prev_fv[6]  # delta_dist_nearest
            # else: deltas stay 0 for first step

    return X


EXTENDED_FEATURE_NAMES = FEATURE_NAMES_7 + [
    'delta_knn_mean', 'delta_knn_std', 'delta_curvature',
    'delta_ridge', 'delta_dist_nearest',
    'step_index', 'step_progress',
]


def label_steps(rows: list, threshold: float = None) -> np.ndarray:
    """
    Label each step based on task-level quality.

    Uses adaptive thresholding: steps from tasks in the bottom tertile
    of quality are labeled as "degraded" (needs intervention),
    middle tertile as "marginal", top tertile as "strong".

    Returns integer labels: 0=strong, 1=marginal, 2=degraded
    """
    qualities = np.array([r["quality"] for r in rows])
    task_qualities = {}
    for r in rows:
        task_qualities[r["task_id"]] = r["quality"]

    unique_q = sorted(set(task_qualities.values()))

    if threshold is not None:
        # Binary: above/below threshold
        labels = np.where(qualities >= threshold, 0, 2).astype(int)
    else:
        # Tertile-based
        t33 = np.percentile(unique_q, 33)
        t66 = np.percentile(unique_q, 66)
        labels = np.zeros(len(rows), dtype=int)
        for i, q in enumerate(qualities):
            if q >= t66:
                labels[i] = 0  # strong
            elif q >= t33:
                labels[i] = 1  # marginal
            else:
                labels[i] = 2  # degraded
    return labels


LABEL_NAMES = {0: "strong", 1: "marginal", 2: "degraded"}
# Map to intervention signals
LABEL_TO_SIGNAL = {
    0: "coherent",           # no intervention needed
    1: "decision_boundary",  # borderline — gentle reflection
    2: "terra_incognita",    # quality degrading — intervene
}


def train_and_evaluate():
    """Main training pipeline."""
    print("=" * 60)
    print("MIRRORFIELD DATA-DRIVEN CLASSIFIER TRAINING")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading experiment data...")
    rows = load_all_experiment_data()
    print(f"    Total steps extracted: {len(rows)}")

    sources = {}
    for r in rows:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    for src, count in sorted(sources.items()):
        print(f"    {src}: {count} steps")

    if len(rows) < 20:
        print("ERROR: Not enough data to train classifier")
        return

    # 2. Compute features
    print("\n[2] Computing trajectory features...")
    X = compute_trajectory_features(rows)
    print(f"    Feature matrix shape: {X.shape}")
    print(f"    Features: {EXTENDED_FEATURE_NAMES}")

    # Feature distribution summary
    print("\n    Feature distributions:")
    for i, name in enumerate(EXTENDED_FEATURE_NAMES):
        vals = X[:, i]
        print(f"      {name:25s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]")

    # 3. Label data
    print("\n[3] Labeling steps...")
    y = label_steps(rows)
    for label_id, label_name in LABEL_NAMES.items():
        count = (y == label_id).sum()
        print(f"    {label_name}: {count} steps ({100*count/len(y):.1f}%)")

    # Quality distribution
    qualities = [r["quality"] for r in rows]
    print(f"\n    Quality range: [{min(qualities):.3f}, {max(qualities):.3f}]")
    print(f"    Quality mean: {np.mean(qualities):.3f}, std: {np.std(qualities):.3f}")

    # 4. Train classifiers
    print("\n[4] Training classifiers...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 4a. Decision Tree (interpretable)
    print("\n  --- Decision Tree (max_depth=4) ---")
    dt = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )
    dt_scores = cross_val_score(dt, X, y, cv=cv, scoring="f1_macro")
    print(f"    CV F1-macro: {dt_scores.mean():.3f} (+/- {dt_scores.std():.3f})")

    dt.fit(X, y)
    dt_pred = dt.predict(X)
    print(f"\n    Training classification report:")
    target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())]
    print(classification_report(y, dt_pred, target_names=target_names, digits=3))

    print("    Decision tree rules:")
    print(export_text(dt, feature_names=EXTENDED_FEATURE_NAMES, max_depth=4))

    # 4b. Random Forest (higher accuracy)
    print("\n  --- Random Forest (n=100, max_depth=5) ---")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    )
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring="f1_macro")
    print(f"    CV F1-macro: {rf_scores.mean():.3f} (+/- {rf_scores.std():.3f})")

    rf.fit(X, y)
    rf_pred = rf.predict(X)
    print(f"\n    Training classification report:")
    print(classification_report(y, rf_pred, target_names=target_names, digits=3))

    # Feature importances
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("    Feature importances (top 10):")
    for i in sorted_idx[:10]:
        print(f"      {EXTENDED_FEATURE_NAMES[i]:25s}: {importances[i]:.4f}")

    # 4c. Gradient Boosting
    print("\n  --- Gradient Boosting (n=100, max_depth=3) ---")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
    )
    gb_scores = cross_val_score(gb, X, y, cv=cv, scoring="f1_macro")
    print(f"    CV F1-macro: {gb_scores.mean():.3f} (+/- {gb_scores.std():.3f})")

    gb.fit(X, y)
    gb_pred = gb.predict(X)
    print(f"\n    Training classification report:")
    print(classification_report(y, gb_pred, target_names=target_names, digits=3))

    # 5. Select best model
    models = {
        "decision_tree": (dt, dt_scores.mean()),
        "random_forest": (rf, rf_scores.mean()),
        "gradient_boosting": (gb, gb_scores.mean()),
    }
    best_name = max(models, key=lambda k: models[k][1])
    best_model, best_score = models[best_name]
    print(f"\n[5] Best model: {best_name} (CV F1-macro: {best_score:.3f})")

    # 6. Prediction distribution check
    print(f"\n[6] Prediction distribution (best model on full data):")
    best_pred = best_model.predict(X)
    for label_id, label_name in LABEL_NAMES.items():
        count = (best_pred == label_id).sum()
        pct = 100 * count / len(best_pred)
        signal = LABEL_TO_SIGNAL[label_id]
        print(f"    {label_name} -> {signal}: {count} steps ({pct:.1f}%)")

    # Verify it's NOT 100% coherent
    coherent_pct = 100 * (best_pred == 0).sum() / len(best_pred)
    if coherent_pct == 100:
        print("\n    WARNING: Classifier still predicts 100% coherent!")
        print("    The features may not discriminate quality levels.")
    else:
        print(f"\n    SUCCESS: Classifier is no longer blind ({100-coherent_pct:.1f}% non-coherent)")

    # 7. Save model
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_OUTPUT_DIR / "geometric_classifier.pkl"

    model_bundle = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": EXTENDED_FEATURE_NAMES,
        "label_names": LABEL_NAMES,
        "label_to_signal": LABEL_TO_SIGNAL,
        "cv_f1_macro": best_score,
        "n_training_samples": len(rows),
        "training_sources": sources,
        # Also save the decision tree for interpretability regardless
        "decision_tree": dt,
        "decision_tree_rules": export_text(dt, feature_names=EXTENDED_FEATURE_NAMES),
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"\n[7] Model saved to: {model_path}")

    # 8. Also export human-readable thresholds from decision tree
    thresholds_path = MODEL_OUTPUT_DIR / "classifier_thresholds.json"
    threshold_info = {
        "model_type": best_name,
        "cv_f1_macro": float(best_score),
        "n_training_samples": len(rows),
        "label_map": {str(k): v for k, v in LABEL_NAMES.items()},
        "signal_map": {str(k): v for k, v in LABEL_TO_SIGNAL.items()},
        "decision_tree_rules": export_text(dt, feature_names=EXTENDED_FEATURE_NAMES),
        "feature_importances": {
            name: float(importances[i])
            for i, name in enumerate(EXTENDED_FEATURE_NAMES)
        },
        "feature_distributions": {
            name: {
                "mean": float(X[:, i].mean()),
                "std": float(X[:, i].std()),
                "min": float(X[:, i].min()),
                "max": float(X[:, i].max()),
                "p25": float(np.percentile(X[:, i], 25)),
                "p75": float(np.percentile(X[:, i], 75)),
            }
            for i, name in enumerate(EXTENDED_FEATURE_NAMES)
        },
    }
    with open(thresholds_path, "w") as f:
        json.dump(threshold_info, f, indent=2)
    print(f"    Thresholds saved to: {thresholds_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return model_bundle


if __name__ == "__main__":
    train_and_evaluate()
