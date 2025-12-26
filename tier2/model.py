"""
Binary sentiment classifier architecture for Phase B.

Uses frozen sentence-transformer embeddings (all-MiniLM-L6-v2) with a 2-layer MLP
classifier head for binary sentiment classification.
"""

import torch
from torch import nn
from typing import List, Optional

DEFAULT_SEED = 42


class BinarySentimentClassifier(nn.Module):
    """
    Simple binary classifier for sentiment/intent.

    Architecture:
    - Input: Text embeddings from sentence-transformers (384-dim)
    - Hidden layers: 2-layer MLP with ReLU + Dropout
    - Output: 2 logits (negative, positive)
    """

    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Binary logits: [logit_neg, logit_pos]
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: (batch_size, embedding_dim) tensor

        Returns:
            logits: (batch_size, 2) tensor [logit_negative, logit_positive]
        """
        return self.mlp(embeddings)


def get_embeddings(
    texts: List[str],
    device: torch.device,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    seed: Optional[int] = DEFAULT_SEED
) -> torch.Tensor:
    """
    Convert texts to embeddings using sentence-transformers.

    Uses all-MiniLM-L6-v2 (384-dim) for fast, deterministic embeddings.
    Embeddings are frozen (not fine-tuned) for MVP simplicity.

    Args:
        texts: List of text strings to embed
        device: torch device (cuda or cpu)
        model_name: Sentence-transformers model name
        batch_size: Batch size for encoding
        seed: Random seed for determinism (affects dropout if any)

    Returns:
        embeddings: (len(texts), 384) tensor on specified device
    """
    from sentence_transformers import SentenceTransformer

    # Load sentence-transformers model
    model = SentenceTransformer(model_name, device=str(device))
    model.eval()

    # Set seed for determinism (though embeddings should be deterministic anyway)
    if seed is not None:
        torch.manual_seed(seed)

    # Encode in eval mode (no dropout)
    with torch.no_grad():
        embeddings = model.encode(
            texts,
            convert_to_tensor=True,
            device=device,
            batch_size=batch_size,
            show_progress_bar=False
        )

    return embeddings


def load_classifier_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    embedding_dim: int = 384,
    hidden_dim: int = 128
) -> BinarySentimentClassifier:
    """
    Load trained classifier from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model onto
        embedding_dim: Embedding dimension (must match training)
        hidden_dim: Hidden layer dimension (must match training)

    Returns:
        Loaded model in eval mode
    """
    model = BinarySentimentClassifier(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def save_classifier_checkpoint(
    model: BinarySentimentClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    checkpoint_path: str
):
    """
    Save classifier checkpoint with training metadata.

    Args:
        model: Trained model
        optimizer: Optimizer state
        epoch: Current epoch number
        train_loss: Final training loss
        val_loss: Final validation loss
        val_accuracy: Final validation accuracy
        checkpoint_path: Path to save checkpoint
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "embedding_dim": model.embedding_dim,
        "hidden_dim": model.hidden_dim
    }

    torch.save(checkpoint, checkpoint_path)
