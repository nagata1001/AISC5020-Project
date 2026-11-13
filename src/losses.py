"""Loss functions for the knowledge-distillation experiments.

The implementations in this module follow the project objective

    (1 - alpha) * CE(y, q_s(1))
    + alpha * T**2 * KL(p_t(T) || q_s(T)).

All inputs are raw (unnormalized) logits.  The teacher is treated as fixed:
its logits are detached before the soft-label distribution is computed.
"""

from __future__ import annotations

from typing import Dict, Literal, Union, overload

import torch
import torch.nn.functional as F
from torch import Tensor


Reduction = Literal["none", "mean", "sum"]
LossComponents = Dict[str, Tensor]


def _validate_classification_inputs(logits: Tensor, targets: Tensor) -> None:
    """Validate the common inputs used by the hard-label losses."""
    if logits.ndim != 2:
        raise ValueError(
            f"logits must have shape [batch_size, num_classes], got {tuple(logits.shape)}"
        )
    if not logits.is_floating_point():
        raise TypeError("logits must be a floating-point tensor")
    if targets.ndim != 1:
        raise ValueError(
            f"targets must have shape [batch_size], got {tuple(targets.shape)}"
        )
    if targets.shape[0] != logits.shape[0]:
        raise ValueError(
            "logits and targets must have the same batch size, got "
            f"{logits.shape[0]} and {targets.shape[0]}"
        )
    if targets.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("targets must contain integer class indices")
    if logits.shape[1] < 2:
        raise ValueError("logits must contain at least two classes")
    if targets.numel() > 0:
        minimum = int(targets.min().item())
        maximum = int(targets.max().item())
        if minimum < 0 or maximum >= logits.shape[1]:
            raise ValueError(
                f"targets must be in [0, {logits.shape[1] - 1}], "
                f"got minimum={minimum}, maximum={maximum}"
            )


def _apply_reduction(per_sample_loss: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "none":
        return per_sample_loss
    if reduction == "mean":
        return per_sample_loss.mean()
    if reduction == "sum":
        return per_sample_loss.sum()
    raise ValueError(
        f"reduction must be one of 'none', 'mean', or 'sum', got {reduction!r}"
    )


def hard_label_loss(
    student_logits: Tensor,
    targets: Tensor,
    *,
    reduction: Reduction = "mean",
) -> Tensor:
    """Return ordinary cross-entropy loss from raw student logits."""
    _validate_classification_inputs(student_logits, targets)
    return F.cross_entropy(student_logits, targets, reduction=reduction)


def label_smoothing_loss(
    student_logits: Tensor,
    targets: Tensor,
    *,
    epsilon: float = 0.1,
    reduction: Reduction = "mean",
) -> Tensor:
    """Return cross entropy against explicitly smoothed target labels.

    This uses the same convention as PyTorch's ``CrossEntropyLoss``:

        smoothed_target = (1 - epsilon) * one_hot + epsilon / num_classes.

    Thus the target class has mass ``1 - epsilon + epsilon / K`` and every
    other class has mass ``epsilon / K``.
    """
    _validate_classification_inputs(student_logits, targets)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")

    num_classes = student_logits.shape[1]
    log_probabilities = F.log_softmax(student_logits, dim=1)
    one_hot = F.one_hot(targets.to(torch.int64), num_classes=num_classes).to(
        dtype=student_logits.dtype,
        device=student_logits.device,
    )
    smoothed_targets = (1.0 - epsilon) * one_hot + epsilon / num_classes
    per_sample_loss = -(smoothed_targets * log_probabilities).sum(dim=1)
    return _apply_reduction(per_sample_loss, reduction)


def _validate_distillation_inputs(
    student_logits: Tensor,
    teacher_logits: Tensor,
    targets: Tensor,
    temperature: float,
    alpha: float,
) -> None:
    _validate_classification_inputs(student_logits, targets)
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "teacher_logits and student_logits must have the same shape, got "
            f"{tuple(teacher_logits.shape)} and {tuple(student_logits.shape)}"
        )
    if not teacher_logits.is_floating_point():
        raise TypeError("teacher_logits must be a floating-point tensor")
    if teacher_logits.device != student_logits.device:
        raise ValueError("teacher_logits and student_logits must be on the same device")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")


@overload
def knowledge_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    targets: Tensor,
    *,
    temperature: float,
    alpha: float,
    scale_by_temperature_squared: bool = True,
    return_components: Literal[False] = False,
) -> Tensor: ...


@overload
def knowledge_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    targets: Tensor,
    *,
    temperature: float,
    alpha: float,
    scale_by_temperature_squared: bool = True,
    return_components: Literal[True],
) -> LossComponents: ...


def knowledge_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    targets: Tensor,
    *,
    temperature: float,
    alpha: float,
    scale_by_temperature_squared: bool = True,
    return_components: bool = False,
) -> Union[Tensor, LossComponents]:
    """Return the project's hard/soft knowledge-distillation objective.

    ``F.kl_div`` expects log probabilities as its first argument and target
    probabilities as its second argument.  Passing student log probabilities
    first and teacher probabilities second therefore computes
    ``KL(teacher || student)``.  ``batchmean`` sums over classes for each
    sample, then averages over the batch, matching the mathematical formula.

    Set ``scale_by_temperature_squared=False`` only for the planned no-T^2
    ablation.  The temperature is still applied inside both softmax functions.

    If ``return_components`` is true, a dictionary containing ``loss``,
    ``hard_loss``, ``soft_loss``, and ``scaled_soft_loss`` is returned.  All
    components retain their computation graphs; call ``detach()`` or ``item()``
    when recording them in logs.
    """
    _validate_distillation_inputs(
        student_logits,
        teacher_logits,
        targets,
        temperature,
        alpha,
    )

    hard_loss = F.cross_entropy(student_logits, targets, reduction="mean")

    teacher_probabilities = F.softmax(
        teacher_logits.detach() / temperature,
        dim=1,
    )
    student_log_probabilities = F.log_softmax(
        student_logits / temperature,
        dim=1,
    )
    soft_loss = F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="batchmean",
    )

    temperature_scale = temperature**2 if scale_by_temperature_squared else 1.0
    scaled_soft_loss = temperature_scale * soft_loss
    total_loss = (1.0 - alpha) * hard_loss + alpha * scaled_soft_loss

    if return_components:
        return {
            "loss": total_loss,
            "hard_loss": hard_loss,
            "soft_loss": soft_loss,
            "scaled_soft_loss": scaled_soft_loss,
        }
    return total_loss


__all__ = [
    "hard_label_loss",
    "knowledge_distillation_loss",
    "label_smoothing_loss",
]
