"""Token-level accuracy for causal language modeling."""

import numpy as np


def token_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return token accuracy as a percentage; vocabulary is axis 1."""
    predictions = y_pred.argmax(axis=1)
    return float(100.0 * np.mean(predictions == y_true))
