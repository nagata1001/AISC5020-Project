"""Formula-level gradient verification for the distillation objective.

This module compares three gradients with respect to the student logits:

1. the closed-form analytical gradient;
2. PyTorch autograd;
3. a central finite-difference approximation.

Run from the project root with::

    python -m src.gradient_check

By default, results are written to ``reports/gradient_check.csv``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from src.losses import knowledge_distillation_loss


@dataclass(frozen=True)
class GradientCase:
    """Inputs and hyperparameters for one gradient-check case."""

    name: str
    student_logits: Tensor
    teacher_logits: Tensor
    targets: Tensor
    temperature: float
    alpha: float
    scale_by_temperature_squared: bool = True


@dataclass(frozen=True)
class GradientCheckResult:
    """Summary statistics for one gradient-check case."""

    case: str
    batch_size: int
    num_classes: int
    temperature: float
    alpha: float
    scale_t2: bool
    autograd_max_absolute_error: float
    autograd_max_relative_error: float
    finite_difference_max_absolute_error: float
    finite_difference_max_relative_error: float
    teacher_has_gradient: bool
    passed: bool


def _validate_case(case: GradientCase) -> None:
    if case.student_logits.ndim != 2:
        raise ValueError("student_logits must have shape [batch_size, num_classes]")
    if case.teacher_logits.shape != case.student_logits.shape:
        raise ValueError("teacher_logits and student_logits must have the same shape")
    if case.targets.shape != (case.student_logits.shape[0],):
        raise ValueError("targets must have shape [batch_size]")
    if case.student_logits.dtype != torch.float64:
        raise TypeError("gradient checks must use float64 student logits")
    if case.teacher_logits.dtype != torch.float64:
        raise TypeError("gradient checks must use float64 teacher logits")
    if case.student_logits.device.type != "cpu":
        raise ValueError("finite-difference checks are intended to run on CPU")


def analytical_gradient(case: GradientCase) -> Tensor:
    """Return the closed-form gradient with respect to student logits.

    For the T^2-scaled loss, the per-batch gradient is

        [(1-alpha)(q(1)-y) + alpha*T*(q(T)-p(T))] / batch_size.

    Without T^2, the soft-term coefficient becomes ``alpha / T``.
    """
    _validate_case(case)

    student_logits = case.student_logits.detach()
    teacher_logits = case.teacher_logits.detach()
    batch_size, num_classes = student_logits.shape

    hard_student_probabilities = F.softmax(student_logits, dim=1)
    soft_student_probabilities = F.softmax(
        student_logits / case.temperature,
        dim=1,
    )
    soft_teacher_probabilities = F.softmax(
        teacher_logits / case.temperature,
        dim=1,
    )
    one_hot_targets = F.one_hot(
        case.targets.to(torch.int64),
        num_classes=num_classes,
    ).to(dtype=student_logits.dtype)

    hard_gradient = (1.0 - case.alpha) * (
        hard_student_probabilities - one_hot_targets
    )

    if case.scale_by_temperature_squared:
        soft_coefficient = case.alpha * case.temperature
    else:
        soft_coefficient = case.alpha / case.temperature
    soft_gradient = soft_coefficient * (
        soft_student_probabilities - soft_teacher_probabilities
    )

    return (hard_gradient + soft_gradient) / batch_size


def autograd_gradient(case: GradientCase) -> Tuple[Tensor, bool]:
    """Return PyTorch's gradient and whether the teacher received a gradient."""
    _validate_case(case)

    student_logits = case.student_logits.detach().clone().requires_grad_(True)
    teacher_logits = case.teacher_logits.detach().clone().requires_grad_(True)

    loss = knowledge_distillation_loss(
        student_logits,
        teacher_logits,
        case.targets,
        temperature=case.temperature,
        alpha=case.alpha,
        scale_by_temperature_squared=case.scale_by_temperature_squared,
    )
    loss.backward()

    if student_logits.grad is None:
        raise RuntimeError("autograd did not produce a student-logit gradient")
    teacher_has_gradient = teacher_logits.grad is not None
    return student_logits.grad.detach().clone(), teacher_has_gradient


def central_difference_gradient(
    case: GradientCase,
    *,
    step: float = 1e-5,
) -> Tensor:
    """Approximate the student-logit gradient using central differences."""
    _validate_case(case)
    if step <= 0.0:
        raise ValueError(f"step must be positive, got {step}")

    base_logits = case.student_logits.detach().clone()
    approximation = torch.empty_like(base_logits)

    for flat_index in range(base_logits.numel()):
        plus_logits = base_logits.clone()
        minus_logits = base_logits.clone()
        plus_logits.view(-1)[flat_index] += step
        minus_logits.view(-1)[flat_index] -= step

        plus_loss = knowledge_distillation_loss(
            plus_logits,
            case.teacher_logits,
            case.targets,
            temperature=case.temperature,
            alpha=case.alpha,
            scale_by_temperature_squared=case.scale_by_temperature_squared,
        )
        minus_loss = knowledge_distillation_loss(
            minus_logits,
            case.teacher_logits,
            case.targets,
            temperature=case.temperature,
            alpha=case.alpha,
            scale_by_temperature_squared=case.scale_by_temperature_squared,
        )
        approximation.view(-1)[flat_index] = (
            plus_loss.detach() - minus_loss.detach()
        ) / (2.0 * step)

    return approximation


def gradient_errors(actual: Tensor, expected: Tensor) -> Tuple[float, float]:
    """Return maximum absolute and stabilized relative errors."""
    if actual.shape != expected.shape:
        raise ValueError("actual and expected gradients must have the same shape")

    absolute_error = (actual - expected).abs()
    denominator = torch.maximum(actual.abs(), expected.abs()).clamp_min(1e-12)
    relative_error = absolute_error / denominator
    return float(absolute_error.max().item()), float(relative_error.max().item())


def check_case(
    case: GradientCase,
    *,
    finite_difference_step: float = 1e-5,
    autograd_absolute_tolerance: float = 1e-10,
    autograd_relative_tolerance: float = 1e-8,
    finite_difference_absolute_tolerance: float = 1e-8,
    finite_difference_relative_tolerance: float = 1e-6,
) -> GradientCheckResult:
    """Run all three gradient calculations for one case."""
    expected = analytical_gradient(case)
    automatic, teacher_has_gradient = autograd_gradient(case)
    numerical = central_difference_gradient(
        case,
        step=finite_difference_step,
    )

    autograd_absolute, autograd_relative = gradient_errors(automatic, expected)
    finite_absolute, finite_relative = gradient_errors(numerical, expected)

    autograd_matches = torch.allclose(
        automatic,
        expected,
        atol=autograd_absolute_tolerance,
        rtol=autograd_relative_tolerance,
    )
    finite_difference_matches = torch.allclose(
        numerical,
        expected,
        atol=finite_difference_absolute_tolerance,
        rtol=finite_difference_relative_tolerance,
    )
    passed = bool(
        autograd_matches
        and finite_difference_matches
        and not teacher_has_gradient
    )

    return GradientCheckResult(
        case=case.name,
        batch_size=case.student_logits.shape[0],
        num_classes=case.student_logits.shape[1],
        temperature=case.temperature,
        alpha=case.alpha,
        scale_t2=case.scale_by_temperature_squared,
        autograd_max_absolute_error=autograd_absolute,
        autograd_max_relative_error=autograd_relative,
        finite_difference_max_absolute_error=finite_absolute,
        finite_difference_max_relative_error=finite_relative,
        teacher_has_gradient=teacher_has_gradient,
        passed=passed,
    )


def fixed_three_class_case() -> GradientCase:
    """Return a small deterministic case suitable for a report table."""
    return GradientCase(
        name="fixed_3_class",
        student_logits=torch.tensor(
            [[0.2, -0.4, 1.1]],
            dtype=torch.float64,
        ),
        teacher_logits=torch.tensor(
            [[1.2, 0.3, -0.7]],
            dtype=torch.float64,
        ),
        targets=torch.tensor([2], dtype=torch.int64),
        temperature=2.0,
        alpha=0.5,
        scale_by_temperature_squared=True,
    )


def random_grid_cases(
    *,
    seed: int = 2026,
    batch_size: int = 4,
    num_classes: int = 10,
    temperatures: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
    alphas: Sequence[float] = (0.25, 0.5, 0.75),
    include_no_t2_cases: bool = True,
) -> List[GradientCase]:
    """Build reproducible random cases over the planned T x alpha grid."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    student_logits = torch.randn(
        batch_size,
        num_classes,
        generator=generator,
        dtype=torch.float64,
    )
    teacher_logits = torch.randn(
        batch_size,
        num_classes,
        generator=generator,
        dtype=torch.float64,
    )
    targets = torch.randint(
        low=0,
        high=num_classes,
        size=(batch_size,),
        generator=generator,
        dtype=torch.int64,
    )

    scaling_options = (True, False) if include_no_t2_cases else (True,)
    cases: List[GradientCase] = []
    for scale_t2 in scaling_options:
        suffix = "with_t2" if scale_t2 else "without_t2"
        for temperature in temperatures:
            for alpha in alphas:
                cases.append(
                    GradientCase(
                        name=f"random_k{num_classes}_t{temperature:g}_a{alpha:g}_{suffix}",
                        student_logits=student_logits.clone(),
                        teacher_logits=teacher_logits.clone(),
                        targets=targets.clone(),
                        temperature=float(temperature),
                        alpha=float(alpha),
                        scale_by_temperature_squared=scale_t2,
                    )
                )
    return cases


def run_checks(
    cases: Iterable[GradientCase],
    *,
    finite_difference_step: float = 1e-5,
) -> List[GradientCheckResult]:
    """Run gradient checks for every supplied case."""
    return [
        check_case(case, finite_difference_step=finite_difference_step)
        for case in cases
    ]


def write_results_csv(
    results: Sequence[GradientCheckResult],
    output_path: Path,
) -> None:
    """Write gradient-check summary statistics to a CSV file."""
    if not results:
        raise ValueError("cannot write an empty result list")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_results(results: Sequence[GradientCheckResult]) -> None:
    """Print a compact human-readable summary."""
    header = (
        f"{'case':<42} {'autograd abs':>14} "
        f"{'finite diff abs':>16} {'status':>8}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{result.case:<42} "
            f"{result.autograd_max_absolute_error:>14.3e} "
            f"{result.finite_difference_max_absolute_error:>16.3e} "
            f"{status:>8}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare analytical, autograd, and finite-difference KD gradients."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/gradient_check.csv"),
        help="CSV output path (default: reports/gradient_check.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="seed used to construct random logits (default: 2026)",
    )
    parser.add_argument(
        "--finite-difference-step",
        type=float,
        default=1e-5,
        help="central-difference step size (default: 1e-5)",
    )
    parser.add_argument(
        "--skip-no-t2",
        action="store_true",
        help="omit the no-T^2 ablation cases",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = [fixed_three_class_case()]
    cases.extend(
        random_grid_cases(
            seed=args.seed,
            include_no_t2_cases=not args.skip_no_t2,
        )
    )

    results = run_checks(
        cases,
        finite_difference_step=args.finite_difference_step,
    )
    print_results(results)
    write_results_csv(results, args.output)
    print(f"\nWrote {len(results)} rows to {args.output}")

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
