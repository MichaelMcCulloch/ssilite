"""End-to-end support-acquisition experiment."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .acquisition import acquire_for_cluster_coverage
from .controller import ControllerConfig, JointController
from .data import DatasetSplit, make_support_problem
from .estimator import apply_batched_binary_gradients
from .model import Accuracy, MechanismMLP, accuracy_by_group


@dataclass(frozen=True)
class TrainingDiagnostics:
    accuracy: Accuracy
    mean_precision_cost: float | None
    mean_quantization_mse: float | None
    final_noise_mass: float | None
    final_minority_mass: float | None


@dataclass(frozen=True)
class ExperimentResult:
    seed: int
    initial_support_size: int
    initial_minority_count: int
    acquired_count: int
    acquired_minority_count: int
    acquired_flipped_count: int
    arms: dict[str, TrainingDiagnostics]
    caveat: str


def _concatenate(left: DatasetSplit, right: DatasetSplit) -> DatasetSplit:
    return DatasetSplit(
        features=torch.cat((left.features, right.features)),
        labels=torch.cat((left.labels, right.labels)),
        clean_labels=torch.cat((left.clean_labels, right.clean_labels)),
        minority=torch.cat((left.minority, right.minority)),
        flipped=torch.cat((left.flipped, right.flipped)),
    )


def _oracle_reference_losses(split: DatasetSplit, confidence: float = 4.0) -> Tensor:
    """Reference loss used to isolate support acquisition from noise filtering.

    The synthetic generator exposes clean labels, so this is deliberately an
    oracle diagnostic—not a proposed way to learn reducibility in practice.
    """

    clean_logits = (split.clean_labels.mul(2).sub(1)) * confidence
    return F.binary_cross_entropy_with_logits(
        clean_logits, split.labels, reduction="none"
    )


def _train_erm(
    model: nn.Module,
    split: DatasetSplit,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> None:
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for _ in range(steps):
        indices = torch.randint(
            split.labels.numel(), (batch_size,), generator=generator
        )
        losses = F.binary_cross_entropy_with_logits(
            model(split.features[indices]), split.labels[indices], reduction="none"
        )
        optimizer.zero_grad(set_to_none=True)
        losses.mean().backward()
        optimizer.step()


def _last_layer_gradient_proxy(
    logits: Tensor, labels: Tensor, features: Tensor
) -> Tensor:
    residual_sq = (logits.sigmoid() - labels).square()
    return residual_sq * (1 + features.square().sum(dim=1))


def _train_joint(
    model: nn.Module,
    split: DatasetSplit,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    config: ControllerConfig,
) -> tuple[float, float, float, float]:
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    controller = JointController(config)
    reference_losses = _oracle_reference_losses(split)
    precision_cost_sum = 0.0
    quantization_mse_sum = 0.0
    final_noise_mass = 0.0
    final_minority_mass = 0.0

    for _ in range(steps):
        with torch.no_grad():
            logits = model(split.features)
            losses = F.binary_cross_entropy_with_logits(
                logits, split.labels, reduction="none"
            )
            reducible_scores = (losses - reference_losses).clamp_min(0)
            gradient_proxy = _last_layer_gradient_proxy(
                logits, split.labels, split.features
            ).clamp_min(1e-8)
            allocation = controller.allocate(
                reducible_scores,
                gradient_proxy,
                batch_size,
                generator=generator,
            )

        optimizer.zero_grad(set_to_none=True)
        estimate = apply_batched_binary_gradients(
            model,
            split.features[allocation.indices],
            split.labels[allocation.indices],
            allocation.importance_weights,
            allocation.precision_bits[allocation.indices],
            generator=generator,
        )
        optimizer.step()

        precision_cost_sum += allocation.expected_precision_cost
        quantization_mse_sum += estimate.quantization_mse
        final_noise_mass = float(allocation.robust_weights[split.flipped].sum().item())
        final_minority_mass = float(
            allocation.robust_weights[split.minority].sum().item()
        )

    return (
        precision_cost_sum / steps,
        quantization_mse_sum / steps,
        final_noise_mass,
        final_minority_mass,
    )


def run_experiment(
    *,
    seed: int = 0,
    steps: int = 120,
    batch_size: int = 32,
    acquisition_count: int = 256,
    learning_rate: float = 0.01,
) -> ExperimentResult:
    """Compare fixed-support training with label-free support acquisition."""

    torch.manual_seed(seed)
    problem = make_support_problem(seed=seed)
    acquisition = acquire_for_cluster_coverage(
        problem.train.features,
        problem.reservoir.features,
        acquisition_count,
        num_clusters=4,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    acquired = problem.reservoir.take(acquisition.indices)
    expanded = _concatenate(problem.train, acquired)

    template = MechanismMLP(problem.train.features.shape[1])
    initial_state = copy.deepcopy(template.state_dict())
    config = ControllerConfig()
    arms: dict[str, TrainingDiagnostics] = {}

    for name, support in (
        ("erm_fixed", problem.train),
        ("erm_acquired", expanded),
    ):
        model = MechanismMLP(problem.train.features.shape[1])
        model.load_state_dict(initial_state)
        _train_erm(
            model,
            support,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + 2,
        )
        arms[name] = TrainingDiagnostics(
            accuracy=accuracy_by_group(model, problem.test),
            mean_precision_cost=None,
            mean_quantization_mse=None,
            final_noise_mass=None,
            final_minority_mass=None,
        )

    for name, support in (
        ("joint_fixed", problem.train),
        ("joint_acquired", expanded),
    ):
        model = MechanismMLP(problem.train.features.shape[1])
        model.load_state_dict(initial_state)
        cost, quantization_mse, noise_mass, minority_mass = _train_joint(
            model,
            support,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + 3,
            config=config,
        )
        arms[name] = TrainingDiagnostics(
            accuracy=accuracy_by_group(model, problem.test),
            mean_precision_cost=cost,
            mean_quantization_mse=quantization_mse,
            final_noise_mass=noise_mass,
            final_minority_mass=minority_mass,
        )

    return ExperimentResult(
        seed=seed,
        initial_support_size=problem.train.labels.numel(),
        initial_minority_count=int(problem.train.minority.sum().item()),
        acquired_count=acquisition_count,
        acquired_minority_count=int(acquired.minority.sum().item()),
        acquired_flipped_count=int(acquired.flipped.sum().item()),
        arms=arms,
        caveat=(
            "Per-example precision is statistically emulated, not a wall-clock "
            "claim. Reducible scores use generator-clean labels as an oracle "
            "control so this experiment isolates support acquisition."
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the support acquisition and q/p/precision prototype."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--acquire", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    arguments = parser.parse_args(argv)
    result = run_experiment(
        seed=arguments.seed,
        steps=arguments.steps,
        batch_size=arguments.batch_size,
        acquisition_count=arguments.acquire,
        learning_rate=arguments.learning_rate,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
