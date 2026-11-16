"""Fashion-MNIST data loading for the distillation experiments.

The official training set is split into 54,000 training images and 6,000
validation images, with 600 validation images from each class. The split is
recreated deterministically from ``split_seed`` and is not stored on disk.

Importing this module never downloads data. Downloading is disabled by
default and must be requested explicitly with ``download=True``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class FashionMNISTLoaders:
    """Data loaders and normalization statistics for one experiment."""

    train: DataLoader
    validation: DataLoader
    test: Optional[DataLoader]
    mean: float
    std: float


def stratified_split_indices(
    targets: Tensor,
    *,
    validation_per_class: int = 600,
    seed: int = 2026,
) -> Tuple[Tensor, Tensor]:
    """Return deterministic, class-balanced train and validation indices."""
    targets = torch.as_tensor(targets, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)

    train_parts = []
    validation_parts = []

    for class_index in range(10):
        class_indices = torch.where(targets == class_index)[0]
        order = torch.randperm(len(class_indices), generator=generator)
        class_indices = class_indices[order]

        validation_parts.append(class_indices[:validation_per_class])
        train_parts.append(class_indices[validation_per_class:])

    train_indices = torch.cat(train_parts)
    validation_indices = torch.cat(validation_parts)

    # Avoid keeping either subset grouped by class.
    train_indices = train_indices[
        torch.randperm(len(train_indices), generator=generator)
    ]
    validation_indices = validation_indices[
        torch.randperm(len(validation_indices), generator=generator)
    ]

    if len(train_indices) != 54_000 or len(validation_indices) != 6_000:
        raise ValueError(
            "expected a 54,000/6,000 Fashion-MNIST split, got "
            f"{len(train_indices)}/{len(validation_indices)}"
        )

    return train_indices, validation_indices


def training_mean_and_std(
    raw_images: Tensor,
    train_indices: Tensor,
) -> Tuple[float, float]:
    """Compute population statistics from the training subset only."""
    training_pixels = raw_images[train_indices].to(torch.float32) / 255.0
    mean = float(training_pixels.mean().item())
    std = float(training_pixels.std(unbiased=False).item())
    return mean, std


def make_fashion_mnist_loaders(
    *,
    data_dir: Path = Path("data"),
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 0,
    split_seed: int = 2026,
    download: bool = False,
    include_test: bool = False,
) -> FashionMNISTLoaders:
    """Create reproducible Fashion-MNIST data loaders.

    ``seed`` controls the training shuffle and should use the experiment seed
    (0, 1, or 2). ``split_seed`` controls the fixed 54k/6k split and should
    remain unchanged across experiments.

    The test loader is omitted by default so the official test set remains
    unused during model selection.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    # Load raw training data first so normalization uses only train indices.
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    raw_dataset = datasets.FashionMNIST(
        root=Path(data_dir),
        train=True,
        download=download,
    )
    train_indices, validation_indices = stratified_split_indices(
        raw_dataset.targets,
        seed=split_seed,
    )
    mean, std = training_mean_and_std(raw_dataset.data, train_indices)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )
    transformed_dataset = datasets.FashionMNIST(
        root=Path(data_dir),
        train=True,
        transform=transform,
        download=False,
    )

    train_dataset = Subset(transformed_dataset, train_indices.tolist())
    validation_dataset = Subset(
        transformed_dataset,
        validation_indices.tolist(),
    )

    shuffle_generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=shuffle_generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )

    test_loader = None
    if include_test:
        test_dataset = datasets.FashionMNIST(
            root=Path(data_dir),
            train=False,
            transform=transform,
            download=False,
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **loader_options,
        )

    return FashionMNISTLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
        mean=mean,
        std=std,
    )


def validate_loaders(loaders: FashionMNISTLoaders) -> None:
    """Check dataset sizes, split integrity, labels, and batch shapes."""
    train_dataset = loaders.train.dataset
    validation_dataset = loaders.validation.dataset

    if not isinstance(train_dataset, Subset):
        raise TypeError("the training dataset must be a torch Subset")
    if not isinstance(validation_dataset, Subset):
        raise TypeError("the validation dataset must be a torch Subset")
    if len(train_dataset) != 54_000:
        raise ValueError(f"expected 54,000 training samples, got {len(train_dataset)}")
    if len(validation_dataset) != 6_000:
        raise ValueError(
            f"expected 6,000 validation samples, got {len(validation_dataset)}"
        )

    train_indices = set(train_dataset.indices)
    validation_indices = set(validation_dataset.indices)
    if train_indices & validation_indices:
        raise ValueError("training and validation subsets overlap")

    official_targets = torch.as_tensor(validation_dataset.dataset.targets)
    validation_targets = official_targets[
        torch.as_tensor(validation_dataset.indices, dtype=torch.long)
    ]
    validation_counts = torch.bincount(validation_targets, minlength=10)
    if not torch.equal(validation_counts, torch.full((10,), 600)):
        raise ValueError(
            "validation classes are not balanced: "
            f"{validation_counts.tolist()}"
        )

    images, targets = next(iter(loaders.train))
    if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError(f"unexpected image batch shape: {tuple(images.shape)}")
    if targets.shape != (images.shape[0],):
        raise ValueError(f"unexpected target batch shape: {tuple(targets.shape)}")
    if targets.min() < 0 or targets.max() > 9:
        raise ValueError("batch contains a target outside the range 0 through 9")

    if loaders.test is not None and len(loaders.test.dataset) != 10_000:
        raise ValueError(
            f"expected 10,000 test samples, got {len(loaders.test.dataset)}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the data-module check."""
    parser = argparse.ArgumentParser(
        description="Validate the Fashion-MNIST split and data loaders."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download Fashion-MNIST if it is not already present",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="also construct and validate the official test loader",
    )
    return parser.parse_args()


def main() -> int:
    """Build and validate the configured Fashion-MNIST data loaders."""
    args = parse_args()
    loaders = make_fashion_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        split_seed=args.split_seed,
        download=args.download,
        include_test=args.include_test,
    )
    validate_loaders(loaders)

    images, targets = next(iter(loaders.validation))
    print("Fashion-MNIST data check: PASS")
    print(f"train samples:      {len(loaders.train.dataset):,}")
    print(f"validation samples: {len(loaders.validation.dataset):,}")
    if loaders.test is not None:
        print(f"test samples:       {len(loaders.test.dataset):,}")
    print(f"normalization mean: {loaders.mean:.8f}")
    print(f"normalization std:  {loaders.std:.8f}")
    print(f"image batch shape:  {tuple(images.shape)}")
    print(f"target batch shape: {tuple(targets.shape)}")
    return 0


__all__ = [
    "FashionMNISTLoaders",
    "make_fashion_mnist_loaders",
    "stratified_split_indices",
    "training_mean_and_std",
    "validate_loaders",
]


if __name__ == "__main__":
    raise SystemExit(main())
