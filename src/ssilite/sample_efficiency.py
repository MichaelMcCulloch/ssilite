"""Lean matched sample-efficiency benchmark for formal environment MoEs.

Each data seed creates one synthetic problem and one label-free acquisition
ordering.  Nested prefixes of that ordering define the support curve.  At each
budget, environments are discovered from raw train features alone and used to
supervise a learned top-1 router plus coherent expert curricula.  There is no
out-of-fold grader, trust filter, or unrelated learner in this benchmark.

Rare-group and clean-label metadata are used only after training to report
support denominators and clean-test accuracy.  Neither environment discovery nor
the final MoEs can receive them.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from typing import Literal

import torch
from torch import Tensor

from .acquisition import acquire_for_cluster_coverage
from .data import DatasetSplit, make_support_problem
from .environment_ensemble import discover_feature_environments
from .environment_moe import (
    EnvironmentMoECompute,
    EnvironmentMoEConfig,
    train_environment_moe,
)
from .model import Accuracy

type CensoringStatus = Literal[
    "left_censored",
    "interval_censored",
    "right_censored",
]
type ArmName = Literal[
    "ordinary_mean",
    "ordinary_routed",
    "specialist_mean",
    "routed_specialist",
    "permuted_routed_specialist",
]


@dataclass(frozen=True)
class SampleEfficiencyConfig:
    """Support grid, target, and label-free discovery settings."""

    budgets: tuple[int, ...] = (0, 16, 32, 48, 64, 96, 128, 192, 256)
    minority_target: float = 0.85
    majority_floor: float = 0.90
    label_noise: float = 0.04
    acquisition_clusters: int = 4
    num_environments: int = 3
    kmeans_iterations: int = 20
    acquisition_seed_offset: int = 1
    discovery_seed_offset: int = 20_000
    moe_seed_offset: int = 40_000
    permutation_seed_offset: int = 90_000
    device: str = "auto"


@dataclass(frozen=True)
class SampleEfficiencyArm:
    """One final arm's accuracy, target status, and exact logical compute."""

    accuracy: Accuracy
    target_attained: bool
    compute: EnvironmentMoECompute


@dataclass(frozen=True)
class SampleEfficiencyPoint:
    """One nested labeled-support budget for a paired seed."""

    new_labels: int
    total_labels: int
    acquired_rare_examples: int
    total_rare_examples: int
    environment_cluster_sizes: tuple[int, ...]
    permuted_environment_cluster_sizes: tuple[int, ...]
    model_seed: int
    permuted_model_seed: int
    router_train_accuracy: float
    route_counts: tuple[int, ...]
    minority_route_counts: tuple[int, ...]
    majority_route_counts: tuple[int, ...]
    specialist_expert_accuracies: tuple[Accuracy, ...]
    ordinary_mean: SampleEfficiencyArm
    ordinary_routed: SampleEfficiencyArm
    specialist_mean: SampleEfficiencyArm
    routed_specialist: SampleEfficiencyArm
    permuted_routed_specialist: SampleEfficiencyArm


@dataclass(frozen=True)
class TargetCrossing:
    """Grid-censored label interval containing an arm's first target crossing.

    Lower endpoints are exclusive and upper endpoints are inclusive.  A
    missing lower endpoint denotes left censoring; a missing upper endpoint
    denotes right censoring at the largest evaluated support.
    """

    status: CensoringStatus
    lower_new_labels: int | None
    upper_new_labels: int | None
    lower_total_labels: int | None
    upper_total_labels: int | None
    lower_total_rare_examples: int | None
    upper_total_rare_examples: int | None


@dataclass(frozen=True)
class SampleEfficiencyResult:
    """One reproducible paired sample-efficiency curve."""

    seed: int
    device: str
    config: SampleEfficiencyConfig
    moe_config: EnvironmentMoEConfig
    initial_labels: int
    initial_rare_examples: int
    acquisition_seed: int
    discovery_seed: int
    permutation_seed: int
    points: tuple[SampleEfficiencyPoint, ...]
    ordinary_target_crossing: TargetCrossing
    ordinary_routed_target_crossing: TargetCrossing
    specialist_mean_target_crossing: TargetCrossing
    routed_specialist_target_crossing: TargetCrossing
    permuted_routed_specialist_target_crossing: TargetCrossing
    caveat: str


def _validate_config(config: SampleEfficiencyConfig) -> None:
    budgets = tuple(config.budgets)
    if not budgets:
        raise ValueError("budgets must be non-empty")
    if any(budget < 0 for budget in budgets):
        raise ValueError("budgets must be non-negative")
    if any(left >= right for left, right in pairwise(budgets)):
        raise ValueError("budgets must be strictly increasing")
    if not 0 <= config.minority_target <= 1:
        raise ValueError("minority_target must lie in [0, 1]")
    if not 0 <= config.majority_floor <= 1:
        raise ValueError("majority_floor must lie in [0, 1]")
    if not 0 <= config.label_noise < 0.5:
        raise ValueError("label_noise must lie in [0, 0.5)")
    if config.acquisition_clusters < 1:
        raise ValueError("acquisition_clusters must be positive")
    if config.num_environments < 2:
        raise ValueError("num_environments must be at least two")
    if config.kmeans_iterations < 1:
        raise ValueError("kmeans_iterations must be positive")


def _resolve_device(requested: str | torch.device) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def _concatenate(left: DatasetSplit, right: DatasetSplit) -> DatasetSplit:
    return DatasetSplit(
        features=torch.cat((left.features, right.features)),
        labels=torch.cat((left.labels, right.labels)),
        clean_labels=torch.cat((left.clean_labels, right.clean_labels)),
        minority=torch.cat((left.minority, right.minority)),
        flipped=torch.cat((left.flipped, right.flipped)),
    )


def _accuracy(logits: Tensor, test: DatasetSplit) -> Accuracy:
    with torch.no_grad():
        predictions = logits >= 0
        targets = test.clean_labels.to(dtype=torch.bool)
        correct = predictions == targets
        majority = ~test.minority
        minority = test.minority
        if not torch.any(majority) or not torch.any(minority):
            raise ValueError("the test split must contain both evaluation groups")
        return Accuracy(
            overall=float(correct.to(dtype=torch.float32).mean().item()),
            majority=float(correct[majority].to(dtype=torch.float32).mean().item()),
            minority=float(correct[minority].to(dtype=torch.float32).mean().item()),
        )


def _arm(
    logits: Tensor,
    test: DatasetSplit,
    *,
    compute: EnvironmentMoECompute,
    minority_target: float,
    majority_floor: float,
) -> SampleEfficiencyArm:
    accuracy = _accuracy(logits, test)
    return SampleEfficiencyArm(
        accuracy=accuracy,
        target_attained=(
            accuracy.minority >= minority_target and accuracy.majority >= majority_floor
        ),
        compute=compute,
    )


def _target_crossing(
    points: tuple[SampleEfficiencyPoint, ...],
    arm_name: ArmName,
) -> TargetCrossing:
    first_index = next(
        (
            index
            for index, point in enumerate(points)
            if getattr(point, arm_name).target_attained
        ),
        None,
    )
    if first_index is None:
        final = points[-1]
        return TargetCrossing(
            status="right_censored",
            lower_new_labels=final.new_labels,
            upper_new_labels=None,
            lower_total_labels=final.total_labels,
            upper_total_labels=None,
            lower_total_rare_examples=final.total_rare_examples,
            upper_total_rare_examples=None,
        )
    first = points[first_index]
    if first_index == 0:
        return TargetCrossing(
            status="left_censored",
            lower_new_labels=None,
            upper_new_labels=first.new_labels,
            lower_total_labels=None,
            upper_total_labels=first.total_labels,
            lower_total_rare_examples=None,
            upper_total_rare_examples=first.total_rare_examples,
        )
    previous = points[first_index - 1]
    return TargetCrossing(
        status="interval_censored",
        lower_new_labels=previous.new_labels,
        upper_new_labels=first.new_labels,
        lower_total_labels=previous.total_labels,
        upper_total_labels=first.total_labels,
        lower_total_rare_examples=previous.total_rare_examples,
        upper_total_rare_examples=first.total_rare_examples,
    )


def _permuted_environments(environment_ids: Tensor, *, seed: int) -> Tensor:
    generator_device = (
        environment_ids.device
        if environment_ids.device.type == "cuda"
        else torch.device("cpu")
    )
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    permutation = torch.randperm(
        environment_ids.numel(),
        device=environment_ids.device,
        generator=generator,
    )
    return environment_ids[permutation]


def run_sample_efficiency(
    *,
    seed: int = 0,
    config: SampleEfficiencyConfig | None = None,
    moe_config: EnvironmentMoEConfig | None = None,
) -> SampleEfficiencyResult:
    """Run one paired seed over nested label-acquisition prefixes."""

    config = config or SampleEfficiencyConfig()
    _validate_config(config)
    execution_device = _resolve_device(config.device)
    problem = make_support_problem(
        seed=seed,
        label_noise=config.label_noise,
        test_minority_fraction=0.5,
    )
    max_budget = max(config.budgets)
    if max_budget > problem.reservoir.labels.numel():
        raise ValueError("the largest budget cannot exceed the reservoir size")
    if config.acquisition_clusters > (
        problem.train.labels.numel() + problem.reservoir.labels.numel()
    ):
        raise ValueError("acquisition_clusters exceeds the available feature count")
    acquisition_seed = config.acquisition_seed_offset + seed
    acquisition = acquire_for_cluster_coverage(
        problem.train.features,
        problem.reservoir.features,
        max_budget,
        num_clusters=config.acquisition_clusters,
        generator=torch.Generator().manual_seed(acquisition_seed),
    )
    discovery_seed = config.discovery_seed_offset + seed
    permutation_seed = config.permutation_seed_offset + seed
    resolved_moe_config = replace(
        moe_config or EnvironmentMoEConfig(seed=config.moe_seed_offset + seed),
        device=str(execution_device),
    )
    test = problem.test.to(execution_device)
    initial_labels = problem.train.labels.numel()
    initial_rare_examples = int(problem.train.minority.sum().item())
    points: list[SampleEfficiencyPoint] = []

    for budget in config.budgets:
        acquired = problem.reservoir.take(acquisition.indices[:budget])
        support = _concatenate(problem.train, acquired).to(execution_device)
        environment_ids = discover_feature_environments(
            support.features,
            num_environments=config.num_environments,
            iterations=config.kmeans_iterations,
            seed=discovery_seed,
            device=execution_device,
        )
        moe = train_environment_moe(
            support.features,
            support.labels,
            test.features,
            environment_ids,
            config=resolved_moe_config,
        )
        permuted_environment_ids = _permuted_environments(
            environment_ids,
            seed=permutation_seed,
        )
        permuted_moe = train_environment_moe(
            support.features,
            support.labels,
            test.features,
            permuted_environment_ids,
            config=resolved_moe_config,
        )
        if moe.ordinary.compute != moe.specialist.compute:
            raise RuntimeError("paired final MoE arms used unequal compute")
        if moe.ordinary.compute != permuted_moe.specialist.compute:
            raise RuntimeError("permuted final MoE used unequal compute")
        if not torch.equal(moe.ordinary.model_seed, moe.specialist.model_seed):
            raise RuntimeError("paired final MoE arms used unequal model seeds")
        if not torch.equal(
            moe.ordinary.model_seed,
            permuted_moe.specialist.model_seed,
        ):
            raise RuntimeError("permuted final MoE used an unequal model seed")
        ordinary_mean = _arm(
            moe.ordinary.mean_logits,
            test,
            compute=moe.ordinary.compute,
            minority_target=config.minority_target,
            majority_floor=config.majority_floor,
        )
        ordinary_routed = _arm(
            moe.ordinary.routed_logits,
            test,
            compute=moe.ordinary.compute,
            minority_target=config.minority_target,
            majority_floor=config.majority_floor,
        )
        specialist_mean = _arm(
            moe.specialist.mean_logits,
            test,
            compute=moe.specialist.compute,
            minority_target=config.minority_target,
            majority_floor=config.majority_floor,
        )
        routed_specialist = _arm(
            moe.specialist.routed_logits,
            test,
            compute=moe.specialist.compute,
            minority_target=config.minority_target,
            majority_floor=config.majority_floor,
        )
        permuted_routed_specialist = _arm(
            permuted_moe.specialist.routed_logits,
            test,
            compute=permuted_moe.specialist.compute,
            minority_target=config.minority_target,
            majority_floor=config.majority_floor,
        )
        points.append(
            SampleEfficiencyPoint(
                new_labels=budget,
                total_labels=support.labels.numel(),
                acquired_rare_examples=int(acquired.minority.sum().item()),
                total_rare_examples=int(support.minority.sum().item()),
                environment_cluster_sizes=tuple(
                    int(count)
                    for count in torch.bincount(
                        environment_ids,
                        minlength=config.num_environments,
                    ).tolist()
                ),
                permuted_environment_cluster_sizes=tuple(
                    int(count)
                    for count in torch.bincount(
                        permuted_environment_ids,
                        minlength=config.num_environments,
                    ).tolist()
                ),
                model_seed=int(moe.ordinary.model_seed.item()),
                permuted_model_seed=int(permuted_moe.specialist.model_seed.item()),
                router_train_accuracy=float(
                    moe.specialist.router_train_accuracy.item()
                ),
                route_counts=tuple(
                    int(count) for count in moe.specialist.route_counts.tolist()
                ),
                minority_route_counts=tuple(
                    int(count)
                    for count in torch.bincount(
                        moe.specialist.routed_expert_indices[test.minority],
                        minlength=config.num_environments,
                    ).tolist()
                ),
                majority_route_counts=tuple(
                    int(count)
                    for count in torch.bincount(
                        moe.specialist.routed_expert_indices[~test.minority],
                        minlength=config.num_environments,
                    ).tolist()
                ),
                specialist_expert_accuracies=tuple(
                    _accuracy(expert_logits, test)
                    for expert_logits in moe.specialist.logits
                ),
                ordinary_mean=ordinary_mean,
                ordinary_routed=ordinary_routed,
                specialist_mean=specialist_mean,
                routed_specialist=routed_specialist,
                permuted_routed_specialist=permuted_routed_specialist,
            )
        )

    resolved_points = tuple(points)
    return SampleEfficiencyResult(
        seed=seed,
        device=str(execution_device),
        config=replace(config, device=str(execution_device)),
        moe_config=resolved_moe_config,
        initial_labels=initial_labels,
        initial_rare_examples=initial_rare_examples,
        acquisition_seed=acquisition_seed,
        discovery_seed=discovery_seed,
        permutation_seed=permutation_seed,
        points=resolved_points,
        ordinary_target_crossing=_target_crossing(
            resolved_points,
            "ordinary_mean",
        ),
        ordinary_routed_target_crossing=_target_crossing(
            resolved_points,
            "ordinary_routed",
        ),
        specialist_mean_target_crossing=_target_crossing(
            resolved_points,
            "specialist_mean",
        ),
        routed_specialist_target_crossing=_target_crossing(
            resolved_points,
            "routed_specialist",
        ),
        permuted_routed_specialist_target_crossing=_target_crossing(
            resolved_points,
            "permuted_routed_specialist",
        ),
        caveat=(
            "Acquisition and environment discovery use raw pool/train features "
            "without labels, but the fixed unlabeled reservoir is observation "
            "access. Rare-group and clean-label fields are post-hoc diagnostics. "
            "Discovered environments pseudo-supervise the learned router and "
            "teacher-force task dispatch during training. Permuted environments "
            "preserve cluster sizes and use paired model and batch-stream seeds. "
            "Target crossings are grid-censored first observed attainments; "
            "stochastic accuracy need not be monotone."
        ),
    )


def main(argv: list[str] | None = None) -> None:
    """Run one or more paired seeds and print strict JSON."""

    parser = argparse.ArgumentParser(
        description="Benchmark dense means against a formal environment-routed MoE."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(SampleEfficiencyConfig().budgets),
    )
    parser.add_argument("--minority-target", type=float, default=0.85)
    parser.add_argument("--majority-floor", type=float, default=0.90)
    parser.add_argument("--label-noise", type=float, default=0.04)
    parser.add_argument("--acquisition-clusters", type=int, default=4)
    parser.add_argument("--num-environments", type=int, default=3)
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args(argv)
    config = SampleEfficiencyConfig(
        budgets=tuple(arguments.budgets),
        minority_target=arguments.minority_target,
        majority_floor=arguments.majority_floor,
        label_noise=arguments.label_noise,
        acquisition_clusters=arguments.acquisition_clusters,
        num_environments=arguments.num_environments,
        kmeans_iterations=arguments.kmeans_iterations,
        device=arguments.device,
    )
    results = [
        run_sample_efficiency(seed=seed, config=config) for seed in arguments.seeds
    ]
    print(
        json.dumps(
            [asdict(result) for result in results],
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
