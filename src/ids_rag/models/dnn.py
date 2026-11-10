"""Deep neural network classifier architecture."""

from typing import Sequence
import numpy as np
import torch
from torch import nn


class BinaryDNN(nn.Module):
    """Multi-layer binary classifier producing an unnormalized logit."""

    def __init__(
        self,
        in_features: int,
        hidden: Sequence[int] = (128, 64, 32),
        dropout: float = 0.3,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = in_features

        for h_dim in hidden:
            # Linear projection
            layers.append(nn.Linear(prev_dim, h_dim))
            # Optional batch normalization
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            # Non-linear activation
            layers.append(nn.ReLU())
            # Regularization dropout
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Final projection to 1 unnormalized logit (no sigmoid)
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits of shape (batch_size, 1)."""
        return self.network(x)


def compute_binary_class_weights(y: np.ndarray | torch.Tensor) -> tuple[float, float]:
    """Compute balanced sample/class weights w_c = N / (2 * N_c) for binary labels 0 and 1."""
    if isinstance(y, torch.Tensor):
        y_arr = y.detach().cpu().numpy()
    else:
        y_arr = np.asarray(y)

    n_samples = len(y_arr)
    n_0 = int(np.sum(y_arr == 0))
    n_1 = int(np.sum(y_arr == 1))

    if n_0 == 0 or n_1 == 0:
        raise ValueError("Both classes 0 and 1 must have at least one sample to compute weights.")

    w_0 = n_samples / (2.0 * n_0)
    w_1 = n_samples / (2.0 * n_1)
    return float(w_0), float(w_1)


def compute_binary_macro_f1(y_true: np.ndarray | Sequence[int], y_pred: np.ndarray | Sequence[int]) -> float:
    """Compute macro-averaged F1 (unweighted mean of class 0 F1 and class 1 F1) for binary targets."""
    from sklearn.metrics import f1_score

    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)

    f1_pos = float(f1_score(y_t, y_p, pos_label=1, zero_division=0))
    f1_neg = float(f1_score(y_t, y_p, pos_label=0, zero_division=0))
    return float((f1_pos + f1_neg) / 2.0)


def check_monitor_improvement(
    monitor: str,
    current_f1: float,
    current_loss: float,
    best_f1: float,
    best_loss: float,
    tol: float = 1e-7,
) -> bool:
    """Check if current epoch improves best metrics under 'f1' or 'loss' monitor mode."""
    if monitor == "f1":
        # Strictly higher macro F1; break ties by lower validation loss
        if current_f1 > best_f1 + tol:
            return True
        if abs(current_f1 - best_f1) <= tol and current_loss < best_loss - tol:
            return True
        return False
    elif monitor == "loss":
        # Strictly lower validation loss
        return current_loss < best_loss - tol
    else:
        raise ValueError(f"Unsupported monitor '{monitor}'. Allowed values are 'f1' or 'loss'.")


