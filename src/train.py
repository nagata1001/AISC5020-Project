"""Train teacher and student models on Fashion-MNIST.

Supported modes:

``teacher``
    Train ``TeacherCNN`` with ordinary cross entropy.
``student``
    Train ``StudentCNN`` with ordinary cross entropy.
``label_smoothing``
    Train ``StudentCNN`` with the project's label-smoothing loss.
``kd``
    Train ``StudentCNN`` from a frozen teacher checkpoint.

The best checkpoint is selected using validation NLL.  The official test set
is never loaded by this script.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from src.data import make_fashion_mnist_loaders
from src.losses import (
    hard_label_loss,
    knowledge_distillation_loss,
    label_smoothing_loss,
)
from src.metrics import ClassificationMetrics, compute_classification_metrics
from src.models import StudentCNN, TeacherCNN


TRAINING_MODES = ("teacher", "student", "label_smoothing", "kd")


@dataclass(frozen=True)
class EpochRecord:
    """Training and validation results from one epoch."""

    epoch: int
    train_loss: float
    validation_accuracy: float
    validation_nll: float
    validation_ece: float
    best: bool


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch for reproducible experiments."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(name: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` to a torch device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def load_teacher(checkpoint_path: Path, device: torch.device) -> TeacherCNN:
    """Load and freeze a teacher model for knowledge distillation."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    teacher = TeacherCNN().to(device)
    teacher.load_state_dict(state_dict)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def compute_training_loss(
    *,
    mode: str,
    model: nn.Module,
    images: Tensor,
    targets: Tensor,
    teacher: Optional[TeacherCNN],
    label_smoothing: float,
    temperature: float,
    alpha: float,
    scale_by_temperature_squared: bool,
) -> Tensor:
    """Compute the loss for one training batch."""
    logits = model(images)

    if mode in {"teacher", "student"}:
        return hard_label_loss(logits, targets)

    if mode == "label_smoothing":
        return label_smoothing_loss(
            logits,
            targets,
            epsilon=label_smoothing,
        )

    if teacher is None:
        raise ValueError("KD mode requires a teacher model")
    with torch.no_grad():
        teacher_logits = teacher(images)
    return knowledge_distillation_loss(
        logits,
        teacher_logits,
        targets,
        temperature=temperature,
        alpha=alpha,
        scale_by_temperature_squared=scale_by_temperature_squared,
    )


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    mode: str,
    teacher: Optional[TeacherCNN] = None,
    label_smoothing: float = 0.1,
    temperature: float = 4.0,
    alpha: float = 0.5,
    scale_by_temperature_squared: bool = True,
) -> float:
    """Train for one epoch and return mean loss per sample."""
    model.train()
    if teacher is not None:
        teacher.eval()

    total_loss = 0.0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = compute_training_loss(
            mode=mode,
            model=model,
            images=images,
            targets=targets,
            teacher=teacher,
            label_smoothing=label_smoothing,
            temperature=temperature,
            alpha=alpha,
            scale_by_temperature_squared=scale_by_temperature_squared,
        )
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Collect logits and targets for a complete dataset."""
    model.eval()
    logits_parts = []
    target_parts = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits_parts.append(model(images).cpu())
        target_parts.append(targets.cpu())

    return torch.cat(logits_parts), torch.cat(target_parts)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[ClassificationMetrics, Tensor, Tensor]:
    """Evaluate a model using Accuracy, NLL, and 15-bin ECE."""
    logits, targets = predict(model, loader, device)
    metrics = compute_classification_metrics(logits, targets, num_bins=15)
    return metrics, logits, targets


def serializable_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert command-line arguments to JSON/checkpoint-safe values."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def build_run_name(args: argparse.Namespace) -> str:
    """Construct a descriptive output-directory name."""
    if args.run_name:
        return args.run_name
    if args.mode == "kd":
        scaling = "t2" if not args.no_temperature_square else "no_t2"
        return (
            f"kd_T{args.temperature:g}_alpha{args.alpha:g}_"
            f"{scaling}_seed{args.seed}"
        )
    if args.mode == "label_smoothing":
        return f"label_smoothing_eps{args.label_smoothing:g}_seed{args.seed}"
    return f"{args.mode}_seed{args.seed}"


def fit(args: argparse.Namespace) -> Path:
    """Run one experiment and return the best checkpoint path."""
    set_seed(args.seed)
    device = select_device(args.device)
    loaders = make_fashion_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        split_seed=args.split_seed,
        download=args.download,
        include_test=False,
    )

    if args.mode == "teacher":
        model: nn.Module = TeacherCNN()
    else:
        model = StudentCNN()
    model = model.to(device)

    teacher = None
    if args.mode == "kd":
        teacher = load_teacher(args.teacher_checkpoint, device)

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run_dir = args.output_dir / build_run_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pt"
    history_path = run_dir / "history.json"
    predictions_path = run_dir / "validation_predictions.pt"
    config = serializable_config(args)

    print(f"mode:       {args.mode}")
    print(f"device:     {device}")
    print(f"run dir:    {run_dir}")
    print(f"train size: {len(loaders.train.dataset):,}")
    print(f"valid size: {len(loaders.validation.dataset):,}")

    best_nll = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=loaders.train,
            optimizer=optimizer,
            device=device,
            mode=args.mode,
            teacher=teacher,
            label_smoothing=args.label_smoothing,
            temperature=args.temperature,
            alpha=args.alpha,
            scale_by_temperature_squared=not args.no_temperature_square,
        )
        validation, _, _ = evaluate(model, loaders.validation, device)
        is_best = validation.nll < best_nll

        if is_best:
            best_nll = validation.nll
            torch.save(
                {
                    "epoch": epoch,
                    "mode": args.mode,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": asdict(validation),
                    "normalization": {
                        "mean": loaders.mean,
                        "std": loaders.std,
                    },
                    "config": config,
                },
                checkpoint_path,
            )

        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            validation_accuracy=validation.accuracy,
            validation_nll=validation.nll,
            validation_ece=validation.ece,
            best=is_best,
        )
        history.append(record)
        with history_path.open("w", encoding="utf-8") as stream:
            json.dump(
                [asdict(item) for item in history],
                stream,
                indent=2,
            )

        marker = " *" if is_best else ""
        print(
            f"epoch {epoch:03d}/{args.epochs:03d} | "
            f"train loss {train_loss:.6f} | "
            f"val acc {validation.accuracy:.4f} | "
            f"val NLL {validation.nll:.6f} | "
            f"val ECE {validation.ece:.6f}{marker}"
        )

    best_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_metrics, best_logits, best_targets = evaluate(
        model,
        loaders.validation,
        device,
    )
    torch.save(
        {
            "logits": best_logits,
            "targets": best_targets,
            "metrics": asdict(best_metrics),
            "checkpoint": str(checkpoint_path),
        },
        predictions_path,
    )

    print(f"best epoch: {best_checkpoint['epoch']}")
    print(f"best NLL:   {best_metrics.nll:.6f}")
    print(f"checkpoint: {checkpoint_path}")
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Fashion-MNIST teacher and student models."
    )
    parser.add_argument("--mode", choices=TRAINING_MODES, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/training"))
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--download", action="store_true")

    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument(
        "--no-temperature-square",
        action="store_true",
        help="omit the T^2 multiplier for the planned KD ablation",
    )

    args = parser.parse_args()
    if args.mode == "kd" and args.teacher_checkpoint is None:
        parser.error("--teacher-checkpoint is required when --mode=kd")
    return args


def main() -> int:
    args = parse_args()
    fit(args)
    return 0


__all__ = [
    "EpochRecord",
    "evaluate",
    "fit",
    "predict",
    "set_seed",
    "train_one_epoch",
]


if __name__ == "__main__":
    raise SystemExit(main())
