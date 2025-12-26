"""
Tier-2 Binary Sentiment Classifier Training Harness

Trains a binary sentiment classifier on synthetic data and computes reference set
statistics for Phase B boundary distance evaluation.

Follows Phase A artifact pattern: timestamped runs, environment snapshots, JSON summaries.
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from tier2.dataset import generate_synthetic_dataset, split_dataset, SyntheticSample
from tier2.model import BinarySentimentClassifier, get_embeddings, save_classifier_checkpoint
from tier2.reference import build_reference_set, validate_reference_set

DEFAULT_SEED = 42


def get_git_info():
    """Get git commit hash and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()

        dirty_check = subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL
        )
        dirty = "dirty" if dirty_check != 0 else "clean"

        return {"commit": commit[:12], "status": dirty}
    except Exception:
        return {"commit": "unknown", "status": "unknown"}


def get_environment_snapshot(device: torch.device):
    """Capture environment details for reproducibility."""
    env = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_type": device.type,
    }

    if torch.cuda.is_available():
        env["cuda_version"] = torch.version.cuda
        env["device_name"] = torch.cuda.get_device_name(0)
        env["device_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        env["tf32_enabled"] = torch.backends.cuda.matmul.allow_tf32

    # Add sentence-transformers version
    try:
        import sentence_transformers
        env["sentence_transformers_version"] = sentence_transformers.__version__
        env["embedding_model"] = "all-MiniLM-L6-v2"
    except ImportError:
        env["sentence_transformers_version"] = "not_installed"
        env["embedding_model"] = "N/A"

    env["git"] = get_git_info()

    return env


def train_epoch(
    model: nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch_size: int,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    # Simple batch iteration
    n_samples = len(embeddings)
    indices = torch.randperm(n_samples, device=device)

    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_embeddings = embeddings[batch_indices]
        batch_labels = labels[batch_indices]

        # Forward pass
        logits = model(batch_embeddings)
        loss = criterion(logits, batch_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate(
    model: nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module
) -> tuple[float, float]:
    """Evaluate model on validation set."""
    model.eval()

    with torch.no_grad():
        logits = model(embeddings)
        loss = criterion(logits, labels)

        # Compute accuracy
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()

    return loss.item(), accuracy.item()


def main():
    seed = DEFAULT_SEED
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Tier-2 Binary Sentiment Classifier Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed: {seed}")

    # Generate timestamped run_id
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_results_dir = Path("experiments/results/tier2_train") / run_id
    ref_results_dir = Path("experiments/results/tier2_reference") / run_id

    print(f"Run ID: {run_id}")
    print(f"Training output: {train_results_dir.as_posix()}")
    print(f"Reference output: {ref_results_dir.as_posix()}")
    print()

    # Capture environment
    env = get_environment_snapshot(device)

    # Configuration
    config = {
        "seed": seed,
        "n_samples": 2000,
        "train_split": 0.8,
        "embedding_dim": 384,
        "hidden_dim": 128,
        "dropout": 0.1,
        "batch_size": 32,
        "epochs": 20,
        "lr": 0.001,
        "optimizer": "Adam"
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Step 1: Generate synthetic dataset
    print("Step 1: Generating synthetic dataset...")
    samples, dataset_metadata = generate_synthetic_dataset(
        n_samples=config["n_samples"],
        seed=seed
    )
    print(f"  Generated {len(samples)} samples")
    print(f"  Class balance: {dataset_metadata['class_balance']}")
    print(f"  Dataset hash: {dataset_metadata['hash']}")
    print()

    # Step 2: Split dataset
    print("Step 2: Splitting dataset...")
    train_samples, val_samples, split_metadata = split_dataset(
        samples,
        train_fraction=config["train_split"],
        seed=seed
    )
    print(f"  Train: {split_metadata['n_train']} samples")
    print(f"  Val: {split_metadata['n_val']} samples")
    print(f"  Train balance: {split_metadata['class_balance_train']}")
    print(f"  Val balance: {split_metadata['class_balance_val']}")
    print()

    # Step 3: Get embeddings
    print("Step 3: Computing embeddings...")
    torch.manual_seed(seed)

    train_texts = [s.text for s in train_samples]
    val_texts = [s.text for s in val_samples]

    train_start = time.perf_counter()
    train_embeddings = get_embeddings(train_texts, device, seed=seed)
    val_embeddings = get_embeddings(val_texts, device, seed=seed)
    embed_time = time.perf_counter() - train_start

    train_labels = torch.tensor([s.label for s in train_samples], device=device)
    val_labels = torch.tensor([s.label for s in val_samples], device=device)

    print(f"  Train embeddings: {train_embeddings.shape}")
    print(f"  Val embeddings: {val_embeddings.shape}")
    print(f"  Embedding time: {embed_time:.2f}s")
    print()

    # Step 4: Initialize model
    print("Step 4: Initializing model...")
    torch.manual_seed(seed)
    model = BinarySentimentClassifier(
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: BinarySentimentClassifier")
    print(f"  Total parameters: {total_params:,}")
    print()

    # Step 5: Train
    print("Step 5: Training...")
    optimizer = Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_start = time.perf_counter()

    best_val_accuracy = 0.0
    best_epoch = 0
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, config["epochs"] + 1):
        # Train
        train_loss = train_epoch(
            model, train_embeddings, train_labels,
            optimizer, criterion, config["batch_size"], device
        )

        # Validate
        val_loss, val_accuracy = evaluate(
            model, val_embeddings, val_labels, criterion
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Track best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch

        # Print progress
        if epoch % 5 == 0 or epoch == 1 or epoch == config["epochs"]:
            print(f"  Epoch {epoch:02d}/{config['epochs']} - "
                  f"train_loss: {train_loss:.4f}, "
                  f"val_loss: {val_loss:.4f}, "
                  f"val_acc: {val_accuracy:.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_time = time.perf_counter() - training_start

    # Final evaluation
    final_train_loss, final_train_acc = evaluate(
        model, train_embeddings, train_labels, criterion
    )

    print()
    print(f"Training complete!")
    print(f"  Final train loss: {final_train_loss:.4f}, acc: {final_train_acc:.4f}")
    print(f"  Final val loss: {val_losses[-1]:.4f}, acc: {val_accuracies[-1]:.4f}")
    print(f"  Best val acc: {best_val_accuracy:.4f} (epoch {best_epoch})")
    print(f"  Training time: {training_time:.2f}s")
    print()

    # Step 6: Save model checkpoint
    print("Step 6: Saving model checkpoint...")
    train_results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = train_results_dir / "model_checkpoint.pt"

    save_classifier_checkpoint(
        model, optimizer, config["epochs"],
        final_train_loss, val_losses[-1], val_accuracies[-1],
        str(checkpoint_path)
    )
    print(f"  Checkpoint saved: {checkpoint_path}")
    print()

    # Step 7: Compute reference set statistics
    print("Step 7: Computing reference set statistics...")
    # Use full dataset (train + val) as reference set
    ref_samples = train_samples + val_samples

    ref_stats, ref_distances = build_reference_set(
        model, ref_samples, device, seed=seed
    )

    print(f"  Reference set: {ref_stats.n_samples} samples")
    print(f"  mu_ref: {ref_stats.mu_ref:.4f}")
    print(f"  sigma_ref: {ref_stats.sigma_ref:.4f}")
    print(f"  Distance range: [{ref_stats.min_distance:.4f}, {ref_stats.max_distance:.4f}]")
    print(f"  Median distance: {ref_stats.median_distance:.4f}")
    print(f"  Reference hash: {ref_stats.ref_hash}")
    print()

    # Validate reference set
    validation = validate_reference_set(ref_stats)
    print(f"  Reference set validation: {'PASS' if validation['valid'] else 'FAIL'}")
    for check, passed in validation['checks'].items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"    {status} {check}")

    if not validation['valid']:
        print("  Warnings:")
        for warning in validation['warnings']:
            if warning:
                print(f"    - {warning}")
    print()

    # Step 8: Save artifacts
    print("Step 8: Saving artifacts...")

    # Training summary
    train_summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "phase": "tier2_train",
        "environment": env,
        "config": config,
        "dataset": {
            **dataset_metadata,
            **split_metadata
        },
        "training": {
            "final_train_loss": final_train_loss,
            "final_train_acc": final_train_acc,
            "final_val_loss": val_losses[-1],
            "final_val_acc": val_accuracies[-1],
            "best_val_acc": best_val_accuracy,
            "best_epoch": best_epoch,
            "elapsed_seconds": training_time,
            "embedding_seconds": embed_time
        },
        "model": {
            "architecture": "BinarySentimentClassifier",
            "total_parameters": total_params,
            "checkpoint_path": str(checkpoint_path)
        }
    }

    train_summary_path = train_results_dir / "summary.json"
    with train_summary_path.open("w") as f:
        json.dump(train_summary, f, indent=2)
    print(f"  Training summary: {train_summary_path}")

    # Reference set summary
    ref_summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "phase": "tier2_reference",
        "environment": env,
        "config": {
            "seed": seed,
            "model_checkpoint": str(checkpoint_path)
        },
        "reference_set": {
            "ref_name": ref_stats.ref_name,
            "ref_hash": ref_stats.ref_hash,
            "n_samples": ref_stats.n_samples,
            "mu_ref": ref_stats.mu_ref,
            "sigma_ref": ref_stats.sigma_ref,
            "min_distance": ref_stats.min_distance,
            "max_distance": ref_stats.max_distance,
            "median_distance": ref_stats.median_distance,
            "timestamp": ref_stats.timestamp
        },
        "validation": validation
    }

    ref_results_dir.mkdir(parents=True, exist_ok=True)
    ref_summary_path = ref_results_dir / "summary.json"
    with ref_summary_path.open("w") as f:
        json.dump(ref_summary, f, indent=2)
    print(f"  Reference summary: {ref_summary_path}")

    # Save raw reference distances for analysis
    ref_distances_path = ref_results_dir / "reference_distances.json"
    distances_data = {
        "distances": ref_distances.cpu().tolist(),
        "sample_ids": [s.sample_id for s in ref_samples]
    }
    with ref_distances_path.open("w") as f:
        json.dump(distances_data, f, indent=2)
    print(f"  Reference distances: {ref_distances_path}")
    print()

    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Validation accuracy: {val_accuracies[-1]:.2%}")
    print(f"Reference set: mu={ref_stats.mu_ref:.4f}, sigma={ref_stats.sigma_ref:.4f}")
    print()
    print("Next steps:")
    print(f"  1. Generate transform suite: python experiments\\tier2_transform_generate.py")
    print(f"  2. Run evaluation: python experiments\\tier2_semantic_eval.py \\")
    print(f"       --model-checkpoint {checkpoint_path} \\")
    print(f"       --reference-stats {ref_summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
