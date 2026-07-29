"""End-to-end support-acquisition experiment."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .acquisition import acquire_for_cluster_coverage
from .allocation import variance_aware_sampling_probabilities
from .controller import ControllerConfig, JointController
from .data import DatasetSplit, make_support_problem
from .estimator import apply_batched_binary_gradients
from .model import Accuracy, MechanismMLP, accuracy_by_group

type ReferenceMode = Literal["oracle", "raw_loss", "zero", "supplied"]
type AllocationMode = Literal["q_only", "q_p", "q_p_b"]
type ObservedTensorEstimator = Callable[[Tensor, Tensor], Tensor]
type BaseWeightSource = Mapping[str, Tensor] | ObservedTensorEstimator | None


@dataclass(frozen=True)
class TrainingDiagnostics:
    """Arm metrics and outer-loop training volume.

    The counters include repeated sampled examples, but exclude acquisition,
    externally supplied reference/base-weight estimation, and evaluation.
    """

    accuracy: Accuracy
    mean_precision_cost: float | None
    mean_quantization_mse: float | None
    final_noise_mass: float | None
    final_minority_mass: float | None
    scoring_forward_examples: int = 0
    backward_examples: int = 0


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


@dataclass(frozen=True)
class _JointTrainingDiagnostics:
    mean_precision_cost: float
    mean_quantization_mse: float
    final_noise_mass: float
    final_minority_mass: float
    scoring_forward_examples: int
    backward_examples: int


def _resolve_device(device: torch.device | str) -> torch.device:
    if isinstance(device, str) and device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return resolved


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(seed)


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


def _resolve_reference_losses(
    split: DatasetSplit,
    *,
    mode: ReferenceMode,
    estimator: ObservedTensorEstimator | None,
) -> Tensor:
    """Resolve a fixed reference without exposing metadata to estimators."""

    if mode == "oracle":
        if estimator is not None:
            raise ValueError("reference_estimator requires reference_mode='supplied'")
        reference_losses = _oracle_reference_losses(split)
    elif mode in {"raw_loss", "zero"}:
        if estimator is not None:
            raise ValueError("reference_estimator requires reference_mode='supplied'")
        reference_losses = torch.zeros_like(split.labels)
    elif mode == "supplied":
        if estimator is None:
            raise ValueError("reference_mode='supplied' requires reference_estimator")
        # Deliberately pass only learner-observed tensors.  In particular, a
        # cross-fitted estimator cannot inspect clean labels or corruption
        # metadata through this interface.
        reference_losses = estimator(split.features, split.labels)
    else:
        raise ValueError(f"unknown reference mode: {mode!r}")

    return _validate_reference_losses(reference_losses, split)


def _validate_reference_losses(
    reference_losses: Tensor,
    split: DatasetSplit,
) -> Tensor:
    if not isinstance(reference_losses, Tensor):
        raise TypeError("reference losses must be returned as a Tensor")
    reference_losses = reference_losses.detach().to(
        device=split.labels.device,
        dtype=split.labels.dtype,
    )
    if reference_losses.shape != split.labels.shape:
        raise ValueError("reference losses must match the observed labels")
    if not torch.all(torch.isfinite(reference_losses)):
        raise ValueError("reference losses must be finite")
    if torch.any(reference_losses < 0):
        raise ValueError("reference losses must be non-negative")
    return reference_losses


def _resolve_base_weights(
    source: BaseWeightSource,
    *,
    support_name: str,
    split: DatasetSplit,
) -> Tensor | None:
    """Resolve a base measure once, before paired allocation arms run."""

    if source is None:
        return None
    if callable(source):
        # This is the integration boundary for cross-fitted trust estimators.
        # It intentionally exposes only features and observed labels.
        weights = source(split.features, split.labels)
    else:
        try:
            weights = source[support_name]
        except KeyError as error:
            raise ValueError(
                f"base_weights has no entry for support {support_name!r}"
            ) from error

    return _validate_base_weights(weights, split)


def _validate_base_weights(weights: Tensor, split: DatasetSplit) -> Tensor:
    if not isinstance(weights, Tensor):
        raise TypeError("base weights must be returned as a Tensor")
    weights = weights.detach().to(
        device=split.labels.device,
        dtype=split.labels.dtype,
    )
    if weights.shape != split.labels.shape:
        raise ValueError("base weights must match the support labels")
    if not torch.all(torch.isfinite(weights)):
        raise ValueError("base weights must be finite")
    if torch.any(weights < 0) or not torch.any(weights > 0):
        raise ValueError("base weights must be non-negative with positive mass")
    return weights / weights.sum()


def _train_erm(
    model: nn.Module,
    split: DatasetSplit,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> None:
    generator = _make_generator(split.features.device, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for _ in range(steps):
        indices = torch.randint(
            split.labels.numel(),
            (batch_size,),
            device=split.labels.device,
            generator=generator,
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


def _full_precision_cost(config: ControllerConfig, bits: int = 32) -> float:
    """Extend the controller's abstract linear bit-cost scale to full precision."""

    for level, cost in zip(
        config.precision_levels,
        config.precision_costs,
        strict=True,
    ):
        if level == bits:
            return cost
    highest = max(
        range(len(config.precision_levels)), key=config.precision_levels.__getitem__
    )
    return config.precision_costs[highest] * bits / config.precision_levels[highest]


def _train_joint(
    model: nn.Module,
    split: DatasetSplit,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    config: ControllerConfig,
    reference_mode: ReferenceMode = "oracle",
    reference_estimator: ObservedTensorEstimator | None = None,
    reference_losses: Tensor | None = None,
    base_weights: Tensor | None = None,
    allocation_mode: AllocationMode = "q_p_b",
) -> _JointTrainingDiagnostics:
    if allocation_mode not in {"q_only", "q_p", "q_p_b"}:
        raise ValueError(f"unknown allocation mode: {allocation_mode!r}")

    # Keep one private stream, matching the original q+p+b experiment.  All
    # paired arms start this stream at the same seed.
    generator = _make_generator(split.features.device, seed)
    controller_generator = (
        generator
        if allocation_mode == "q_p_b"
        else _make_generator(split.features.device, seed + 3_000_017)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    controller = JointController(config)
    if reference_losses is None:
        reference_losses = _resolve_reference_losses(
            split,
            mode=reference_mode,
            estimator=reference_estimator,
        )
    else:
        if reference_estimator is not None:
            raise ValueError(
                "pass either reference_losses or reference_estimator, not both"
            )
        reference_losses = _validate_reference_losses(reference_losses, split)
    if base_weights is not None:
        base_weights = _validate_base_weights(base_weights, split)
    precision_cost_sum = 0.0
    quantization_mse_sum = split.labels.new_zeros(())
    final_robust_weights: Tensor | None = None
    full_precision_bits = 32
    full_precision_cost = _full_precision_cost(config, full_precision_bits)

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
            if allocation_mode == "q_p_b":
                allocation = controller.allocate(
                    reducible_scores,
                    gradient_proxy,
                    batch_size,
                    base_weights=base_weights,
                    generator=controller_generator,
                    diagnostics=False,
                )
                robust_weights = allocation.robust_weights
            else:
                # The causal controls stop at the stage named by the arm.
                # In particular, q-only and q+p do not compute and discard a
                # mixed-precision allocation.
                robust_weights = controller.update_robust_weights(
                    reducible_scores,
                    base_weights=base_weights,
                )

        if allocation_mode == "q_only":
            probabilities = robust_weights
            indices = torch.multinomial(
                probabilities,
                batch_size,
                replacement=True,
                generator=generator,
            )
            importance_weights = torch.ones(
                batch_size,
                device=split.labels.device,
                dtype=split.labels.dtype,
            )
            precision_cost = full_precision_cost
        elif allocation_mode == "q_p":
            probabilities = variance_aware_sampling_probabilities(
                robust_weights,
                gradient_proxy,
                defensive_mass=config.defensive_mass,
                exploration=config.exploration,
            )
            indices = torch.multinomial(
                probabilities,
                batch_size,
                replacement=True,
                generator=generator,
            )
            importance_weights = robust_weights[indices] / probabilities[indices]
            precision_cost = full_precision_cost
        else:
            indices = allocation.indices
            importance_weights = allocation.importance_weights
            precision_bits = allocation.precision_bits[indices]
            precision_cost = allocation.expected_precision_cost

        optimizer.zero_grad(set_to_none=True)
        if allocation_mode == "q_p_b":
            estimate = apply_batched_binary_gradients(
                model,
                split.features[indices],
                split.labels[indices],
                importance_weights,
                precision_bits,
                generator=generator,
                synchronize_diagnostics=False,
            )
            quantization_mse = estimate.quantization_mse
            if not isinstance(quantization_mse, Tensor):
                raise RuntimeError("deferred quantization diagnostics must be tensors")
        else:
            batch_losses = F.binary_cross_entropy_with_logits(
                model(split.features[indices]),
                split.labels[indices],
                reduction="none",
            )
            torch.dot(importance_weights, batch_losses).div(batch_size).backward()
            quantization_mse = split.labels.new_zeros(())
        optimizer.step()

        precision_cost_sum += precision_cost
        quantization_mse_sum += quantization_mse
        final_robust_weights = robust_weights

    if final_robust_weights is None:
        raise RuntimeError("joint training produced no robust weights")
    final_noise_mass = float(final_robust_weights[split.flipped].sum().item())
    final_minority_mass = float(final_robust_weights[split.minority].sum().item())

    return _JointTrainingDiagnostics(
        mean_precision_cost=precision_cost_sum / steps,
        mean_quantization_mse=float((quantization_mse_sum / steps).item()),
        final_noise_mass=final_noise_mass,
        final_minority_mass=final_minority_mass,
        scoring_forward_examples=steps * split.labels.numel(),
        backward_examples=steps * batch_size,
    )


def run_experiment(
    *,
    seed: int = 0,
    steps: int = 120,
    batch_size: int = 32,
    acquisition_count: int = 256,
    learning_rate: float = 0.01,
    reference_mode: ReferenceMode = "oracle",
    reference_estimator: ObservedTensorEstimator | None = None,
    base_weights: BaseWeightSource = None,
    causal_allocation_arms: bool = False,
    device: torch.device | str = "auto",
) -> ExperimentResult:
    """Compare fixed-support training with label-free support acquisition.

    ``reference_mode="raw_loss"`` is the fully observed-data ablation: its
    zero reference makes the controller rank raw loss.  A
    ``reference_mode="supplied"`` estimator receives only features and
    observed labels.  ``base_weights`` can similarly be either an observed-data
    estimator or a mapping with ``"fixed"`` and ``"acquired"`` tensors.

    The optional causal allocation arms share the same initial parameters,
    optimizer settings, and training seed.  They successively enable robust
    objective weights q, optimized sampling p, and mixed precision b.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    execution_device = _resolve_device(device)
    torch.manual_seed(seed)
    problem = make_support_problem(seed=seed)
    acquisition = acquire_for_cluster_coverage(
        problem.train.features,
        problem.reservoir.features,
        acquisition_count,
        num_clusters=4,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    acquired = problem.reservoir.take(acquisition.indices).to(execution_device)
    train = problem.train.to(execution_device)
    test = problem.test.to(execution_device)
    expanded = _concatenate(train, acquired)

    template = MechanismMLP(train.features.shape[1]).to(execution_device)
    initial_state = copy.deepcopy(template.state_dict())
    config = ControllerConfig()
    arms: dict[str, TrainingDiagnostics] = {}
    supports = {
        "fixed": train,
        "acquired": expanded,
    }
    resolved_base_weights = {
        support_name: _resolve_base_weights(
            base_weights,
            support_name=support_name,
            split=support,
        )
        for support_name, support in supports.items()
    }
    resolved_reference_losses = {
        support_name: _resolve_reference_losses(
            support,
            mode=reference_mode,
            estimator=reference_estimator,
        )
        for support_name, support in supports.items()
    }

    for support_name, support in supports.items():
        name = f"erm_{support_name}"
        model = MechanismMLP(train.features.shape[1]).to(execution_device)
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
            accuracy=accuracy_by_group(model, test),
            mean_precision_cost=None,
            mean_quantization_mse=None,
            final_noise_mass=None,
            final_minority_mass=None,
            scoring_forward_examples=0,
            backward_examples=steps * batch_size,
        )

    allocation_arms: tuple[tuple[str, AllocationMode], ...] = (
        (
            ("joint_q_only", "q_only"),
            ("joint_qp", "q_p"),
            ("joint", "q_p_b"),
        )
        if causal_allocation_arms
        else (("joint", "q_p_b"),)
    )
    for arm_prefix, allocation_mode in allocation_arms:
        for support_name, support in supports.items():
            name = f"{arm_prefix}_{support_name}"
            model = MechanismMLP(train.features.shape[1]).to(execution_device)
            model.load_state_dict(initial_state)
            diagnostics = _train_joint(
                model,
                support,
                steps=steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed + 3,
                config=config,
                reference_mode=reference_mode,
                reference_losses=resolved_reference_losses[support_name],
                base_weights=resolved_base_weights[support_name],
                allocation_mode=allocation_mode,
            )
            arms[name] = TrainingDiagnostics(
                accuracy=accuracy_by_group(model, test),
                mean_precision_cost=diagnostics.mean_precision_cost,
                mean_quantization_mse=diagnostics.mean_quantization_mse,
                final_noise_mass=diagnostics.final_noise_mass,
                final_minority_mass=diagnostics.final_minority_mass,
                scoring_forward_examples=diagnostics.scoring_forward_examples,
                backward_examples=diagnostics.backward_examples,
            )

    if reference_mode == "oracle":
        reference_caveat = (
            "Reducible scores use generator-clean labels as an oracle control "
            "so this experiment isolates support acquisition."
        )
    elif reference_mode in {"raw_loss", "zero"}:
        reference_caveat = (
            "The zero reference ranks observed raw loss; clean labels are used "
            "only for post-training evaluation."
        )
    else:
        reference_caveat = (
            "Reducible scores use a supplied reference estimator that receives "
            "only features and observed labels."
        )
    if base_weights is not None:
        reference_caveat += (
            " The robust objective uses supplied observed-data base weights."
        )

    return ExperimentResult(
        seed=seed,
        initial_support_size=train.labels.numel(),
        initial_minority_count=int(train.minority.sum().item()),
        acquired_count=acquisition_count,
        acquired_minority_count=int(acquired.minority.sum().item()),
        acquired_flipped_count=int(acquired.flipped.sum().item()),
        arms=arms,
        caveat=(
            "Per-example precision is statistically emulated, not a wall-clock "
            f"claim. {reference_caveat}"
            + (
                ""
                if execution_device.type == "cpu"
                else f" Executed on {execution_device}: "
                f"{torch.cuda.get_device_name(execution_device)}."
            )
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
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="PyTorch intra-op threads; pinned for reproducible CPU reductions.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto, cpu, cuda, or an explicit device such as cuda:0.",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("oracle", "raw_loss"),
        default="oracle",
    )
    parser.add_argument("--causal-allocation-arms", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(arguments.threads)
    result = run_experiment(
        seed=arguments.seed,
        steps=arguments.steps,
        batch_size=arguments.batch_size,
        acquisition_count=arguments.acquire,
        learning_rate=arguments.learning_rate,
        reference_mode=arguments.reference_mode,
        causal_allocation_arms=arguments.causal_allocation_arms,
        device=arguments.device,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
