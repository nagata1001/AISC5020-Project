"""Teacher and student CNNs for Fashion-MNIST distillation.

Both models accept images with shape ``[batch_size, 1, 28, 28]`` and return
raw logits with shape ``[batch_size, num_classes]``.  Softmax is deliberately
not included because the training losses and evaluation metrics operate on
raw logits.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TeacherCNN(nn.Module):
    """Larger three-convolution teacher network."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images)
        features = self.global_pool(features)
        features = torch.flatten(features, start_dim=1)
        return self.classifier(features)


class StudentCNN(nn.Module):
    """Smaller two-convolution student network."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images)
        features = self.global_pool(features)
        features = torch.flatten(features, start_dim=1)
        return self.classifier(features)


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Count model parameters, optionally excluding frozen parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


__all__ = [
    "StudentCNN",
    "TeacherCNN",
    "count_parameters",
]
