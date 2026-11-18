"""Classification metrics used by the distillation experiments.

All functions accept raw logits.  Expected calibration error (ECE) uses
equal-width confidence bins over the complete collection of predictions; do
not compute ECE independently per batch and average the batch values.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ClassificationMetrics:
    """Accuracy, negative log-likelihood, and calibration error."""

    accuracy: float
    nll: float
    ece: float
    num_samples: int


def accuracy(logits: Tensor, targets: Tensor) -> Tensor:
    """Return top-1 accuracy as a scalar in the interval [0, 1]."""
    predictions = logits.argmax(dim=1)
    return predictions.eq(targets).to(torch.float32).mean()


def negative_log_likelihood(logits: Tensor, targets: Tensor) -> Tensor:
    """Return mean multiclass negative log-likelihood.

    For raw logits and integer targets, this is exactly cross entropy.
    Float16/bfloat16 logits are promoted to float32 for stable evaluation.
    """
    return F.cross_entropy(
        logits.float(),
        targets.to(torch.int64),
        reduction="mean",
    )


def expected_calibration_error(
    logits: Tensor,
    targets: Tensor,
    *,
    num_bins: int = 15,
) -> Tensor:
    """Return top-label ECE using fixed equal-width confidence bins.

    For bin ``B_m``, ECE is

        sum_m |B_m| / n * |accuracy(B_m) - confidence(B_m)|.

    Bins use ``(lower, upper]`` boundaries.  Softmax top-class confidence is
    strictly positive, and confidence 1.0 is included in the final bin.
    """
    probabilities = F.softmax(logits.float(), dim=1)
    confidences, predictions = probabilities.max(dim=1)
    correctness = predictions.eq(targets).to(torch.float32)

    boundaries = torch.linspace(
        0.0,
        1.0,
        steps=num_bins + 1,
        device=logits.device,
        dtype=confidences.dtype,
    )
    ece = torch.zeros((), device=logits.device, dtype=confidences.dtype)

    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = confidences.gt(lower) & confidences.le(upper)
        if in_bin.any():
            bin_fraction = in_bin.to(torch.float32).mean()
            bin_accuracy = correctness[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()
            ece += bin_fraction * (bin_accuracy - bin_confidence).abs()

    return ece


def compute_classification_metrics(
    logits: Tensor,
    targets: Tensor,
    *,
    num_bins: int = 15,
) -> ClassificationMetrics:
    """Compute all required metrics from a complete prediction set."""
    with torch.no_grad():
        accuracy_value = accuracy(logits, targets)
        nll_value = negative_log_likelihood(logits, targets)
        ece_value = expected_calibration_error(
            logits,
            targets,
            num_bins=num_bins,
        )

    return ClassificationMetrics(
        accuracy=float(accuracy_value.item()),
        nll=float(nll_value.item()),
        ece=float(ece_value.item()),
        num_samples=logits.shape[0],
    )


__all__ = [
    "ClassificationMetrics",
    "accuracy",
    "compute_classification_metrics",
    "expected_calibration_error",
    "negative_log_likelihood",
]
